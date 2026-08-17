import logging
import os
import random
import re
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from datasets import DatasetDict, load_from_disk
from scipy.special import softmax

from .config import TaskClassifierConfig

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
