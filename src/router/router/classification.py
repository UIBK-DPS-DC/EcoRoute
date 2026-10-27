import json
import logging
import os
import random
import re
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import DatasetDict, load_from_disk
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from scipy.special import softmax
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .config import TaskClassifierConfig

import time

logger = logging.getLogger(__name__)


class Classifier(ABC):
    def __init__(self, model, config):
        self.model = model

    @abstractmethod
    def train(self, datasets): ...

    @abstractmethod
    def predict(self, input): ...


class TaskClassifier(Classifier):
    def __init__(self, model, config: TaskClassifierConfig):
        super().__init__(model, config)
        self.temperature = config.temperature
        self.dataset_path = config.dataset_path
        self.prototypes = {}
        self.prompt_templates = {
            "squad": [
                "Answer the following question based on the provided text.\n\n"
                "Text:\n{{context}}\n\n"
                "Question:\n{{question}}",
            ],
            "cnn_dailymail": [
                "Please provide a concise summary of the following text.\n\n"
                "{{article}}"
            ],
            "xsum": [
                "Please provide a concise summary of the following text.\n\n"
                "{{document}}"
            ],
            "glue_mnli": [
                "Consider the following statement and hypothesis."
                "Does the hypothesis logically follow from the statement?\n\n"
                "Statement: {{premise}}"
                "Hypothesis: {{hypothesis}}",
            ],
            "glue_qqp": [
                "Do the following two questions ask about the same thing?\n\n"
                "Question 1: {{question1}}"
                "Question 2: {{question1}}",
            ],
            "glue_sst2": [
                "What is the overall sentiment expressed in the following sentence?\n\n"
                "{{sentence}}",
            ],
            "gsm8k": [
                "{{question}}",
            ],
            "bbh": [
                "{{input}}",
            ],
            "natural_questions": [
                "{{query}}\nAnswer with a short phrase only.",
            ],
            "mbpp": [
                "{{text}}",
            ],
        }

    def train(self, datasets):
        sets = [sets[0] for sets in datasets]
        tasks = [sets[1] for sets in datasets]

        s = self._sample_from_datasets(self.dataset_path, sets, 20)

        rows = []

        for task, (title, value) in zip(tasks, s.items()):
            for sample in value:
                prompt = self._build_prompt(title, sample, self.prompt_templates)
                rows.append({"task": task, "prompt": prompt})
        df = pd.DataFrame(rows)

        embeddings = self.model.encode(df["prompt"].tolist())
        df["embeddings"] = list(embeddings)

        for task in set(tasks):
            prototype = df[df["task"] == task]["embeddings"].values.mean()
            self.prototypes[task] = prototype

        self.tasks = sorted(self.prototypes.keys())

    def predict(self, input):
        z = self.model.encode(input)

        distances = []

        for task in self.tasks:
            prototype = self.prototypes[task]
            d = np.sum((z - prototype) ** 2)
            distances.append(d)

        distances = np.array(distances)

        probs = softmax(-distances / self.temperature)

        pred_index = np.argmax(probs)

        return self.tasks[pred_index], probs

    def _build_prompt(self, dataset_name, sample, prompt_templates):
        if dataset_name not in prompt_templates:
            raise ValueError(f"No templates defined for dataset '{dataset_name}'")

        template = random.choice(prompt_templates[dataset_name])

        placeholders = re.findall(r"\{\{(.*?)\}\}", template)

        for key in placeholders:
            if key not in sample:
                raise KeyError(f"Placeholder '{key}' not found in sample")

            value = str(sample[key])
            template = template.replace(f"{{{{{key}}}}}", value)

        return template

    def _sample_from_datasets(
        self,
        base_path: str,
        dataset_names: list[str],
        n: int,
        seed: int = 42,
    ):
        sampled_datasets = {}

        for name in dataset_names:
            dataset_path = os.path.join(base_path, name)
            dataset = load_from_disk(dataset_path)

            # If dataset has splits (DatasetDict)
            if isinstance(dataset, DatasetDict):

                split_dataset = dataset["train"]

                k = min(n, len(split_dataset))

                sampled_split = split_dataset.shuffle(seed=seed).select(range(k))

                sampled_datasets[name] = sampled_split

            else:
                # Single split dataset
                k = min(n, len(dataset))
                sampled = dataset.shuffle(seed=seed).select(range(k))
                sampled_datasets[name] = sampled

        return sampled_datasets


class DifficultyClassifier(Classifier):
    def __init__(self, model, config):

        class MeanPooling(nn.Module):
            def __init__(self):
                super(MeanPooling, self).__init__()

            def forward(self, last_hidden_state, attention_mask):
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1)
                    .expand(last_hidden_state.size())
                    .float()
                )
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

                sum_mask = input_mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)

                mean_embeddings = sum_embeddings / sum_mask
                return mean_embeddings

        class MulticlassHead(nn.Module):
            def __init__(self, input_size, num_classes):
                super(MulticlassHead, self).__init__()
                self.fc = nn.Linear(input_size, num_classes)

            def forward(self, x):
                x = self.fc(x)
                return x

        class CustomModel(nn.Module, PyTorchModelHubMixin):
            def __init__(self, target_sizes, task_type_map, weights_map, divisor_map):
                super(CustomModel, self).__init__()

                self.backbone = AutoModel.from_pretrained("microsoft/DeBERTa-v3-base")
                self.target_sizes = target_sizes.values()
                self.task_type_map = task_type_map
                self.weights_map = weights_map
                self.divisor_map = divisor_map

                self.heads = [
                    MulticlassHead(self.backbone.config.hidden_size, sz)
                    for sz in self.target_sizes
                ]

                for i, head in enumerate(self.heads):
                    self.add_module(f"head_{i}", head)

                self.pool = MeanPooling()

            def compute_results(self, preds, target, decimal=4):
                if target == "task_type":
                    task_type = {}

                    top2_indices = torch.topk(preds, k=2, dim=1).indices
                    softmax_probs = torch.softmax(preds, dim=1)
                    top2_probs = softmax_probs.gather(1, top2_indices)
                    top2 = top2_indices.detach().cpu().tolist()
                    top2_prob = top2_probs.detach().cpu().tolist()

                    top2_strings = [
                        [self.task_type_map[str(idx)] for idx in sample]
                        for sample in top2
                    ]
                    top2_prob_rounded = [
                        [round(value, 3) for value in sublist] for sublist in top2_prob
                    ]

                    counter = 0
                    for sublist in top2_prob_rounded:
                        if sublist[1] < 0.1:
                            top2_strings[counter][1] = "NA"
                        counter += 1

                    task_type_1 = [sublist[0] for sublist in top2_strings]
                    task_type_2 = [sublist[1] for sublist in top2_strings]
                    task_type_prob = [sublist[0] for sublist in top2_prob_rounded]

                    return (task_type_1, task_type_2, task_type_prob)

                else:
                    preds = torch.softmax(preds, dim=1)

                    weights = np.array(self.weights_map[target])
                    weighted_sum = np.sum(
                        np.array(preds.detach().cpu()) * weights, axis=1
                    )
                    scores = weighted_sum / self.divisor_map[target]

                    scores = [round(value, decimal) for value in scores]
                    if target == "number_of_few_shots":
                        scores = [x if x >= 0.05 else 0 for x in scores]
                    return scores

            def process_logits(self, logits):
                result = {}

                # Round 1: "task_type"
                task_type_logits = logits[0]
                task_type_results = self.compute_results(
                    task_type_logits, target="task_type"
                )
                result["task_type_1"] = task_type_results[0]
                result["task_type_2"] = task_type_results[1]
                result["task_type_prob"] = task_type_results[2]

                # Round 2: "creativity_scope"
                creativity_scope_logits = logits[1]
                target = "creativity_scope"
                result[target] = self.compute_results(
                    creativity_scope_logits, target=target
                )

                # Round 3: "reasoning"
                reasoning_logits = logits[2]
                target = "reasoning"
                result[target] = self.compute_results(reasoning_logits, target=target)

                # Round 4: "contextual_knowledge"
                contextual_knowledge_logits = logits[3]
                target = "contextual_knowledge"
                result[target] = self.compute_results(
                    contextual_knowledge_logits, target=target
                )

                # Round 5: "number_of_few_shots"
                number_of_few_shots_logits = logits[4]
                target = "number_of_few_shots"
                result[target] = self.compute_results(
                    number_of_few_shots_logits, target=target
                )

                # Round 6: "domain_knowledge"
                domain_knowledge_logits = logits[5]
                target = "domain_knowledge"
                result[target] = self.compute_results(
                    domain_knowledge_logits, target=target
                )

                # Round 7: "no_label_reason"
                no_label_reason_logits = logits[6]
                target = "no_label_reason"
                result[target] = self.compute_results(
                    no_label_reason_logits, target=target
                )

                # Round 8: "constraint_ct"
                constraint_ct_logits = logits[7]
                target = "constraint_ct"
                result[target] = self.compute_results(
                    constraint_ct_logits, target=target
                )

                # Round 9: "prompt_complexity_score"
                result["prompt_complexity_score"] = [
                    round(
                        0.35 * creativity
                        + 0.25 * reasoning
                        + 0.15 * constraint
                        + 0.15 * domain_knowledge
                        + 0.05 * contextual_knowledge
                        + 0.05 * few_shots,
                        5,
                    )
                    for creativity, reasoning, constraint, domain_knowledge, contextual_knowledge, few_shots in zip(
                        result["creativity_scope"],
                        result["reasoning"],
                        result["constraint_ct"],
                        result["domain_knowledge"],
                        result["contextual_knowledge"],
                        result["number_of_few_shots"],
                    )
                ]

                return result

            def forward(self, batch):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                outputs = self.backbone(
                    input_ids=input_ids, attention_mask=attention_mask
                )

                last_hidden_state = outputs.last_hidden_state
                mean_pooled_representation = self.pool(
                    last_hidden_state, attention_mask
                )

                logits = [
                    self.heads[k](mean_pooled_representation)
                    for k in range(len(self.target_sizes))
                ]

                return self.process_logits(logits)

        config_path = hf_hub_download(
            "nvidia/prompt-task-and-complexity-classifier",
            "config.json",
        )

        with open(config_path) as f:
            config = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "nvidia/prompt-task-and-complexity-classifier"
        )

        model = CustomModel(
            target_sizes=config["target_sizes"],
            task_type_map=config["task_type_map"],
            weights_map=config["weights_map"],
            divisor_map=config["divisor_map"],
        ).from_pretrained("nvidia/prompt-task-and-complexity-classifier")

        model.eval()

        self.classifier_model = model

    def train(self, datasets):
        return super().train(datasets)

    def predict(self, input):

        prompt = [input]
        st_token = time.perf_counter()
        encoded_texts = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
        )
        logger.info(
            f"Difficulty classification tokenization time: {time.perf_counter() - st_token}"
        )

        st_pred = time.perf_counter()
        result = self.classifier_model(encoded_texts)
        logger.info(
            f"Difficulty classification prediction time: {time.perf_counter() - st_pred}"
        )
        if "prompt_complexity_score" not in result:
            logger.warning(
                f"prompt_complexity_score is not in result of DifficultyClassifier"
            )
            return 0.0

        return result["prompt_complexity_score"][0]
