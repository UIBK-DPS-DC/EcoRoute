# Prepares datasets for evaluation

import os
import random
import re

import pandas as pd
from datasets import DatasetDict, load_from_disk
from tqdm import tqdm

hf_dir = "../hf_datasets/"


def _build_prompt(dataset_name, sample, prompt_templates):
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


prompt_templates = {
    "squad": [
        "Answer the following question based on the provided text.\n\n"
        "Text:\n{{context}}\n\n"
        "Question:\n{{question}}",
    ],
    "cnn_dailymail": [
        "Please provide a concise summary of the following text.\n\n" "{{article}}"
    ],
    "xsum": [
        "Please provide a concise summary of the following text.\n\n" "{{document}}"
    ],
    "glue_mnli": [
        "Consider the following statement and hypothesis. "
        "Determine if the statement entails, contradicts or is neutral to the hypothesis.\n"
        "Answer with ONLY one label. For entailment answer with 0\n For neutral answer with 1\n For contradiction answer with 2\n\n"
        "Statement: {{premise}}\n"
        "Hypothesis: {{hypothesis}}",
    ],
    "glue_qqp": [
        "Do the following two questions ask about the same thing? "
        "Answer with ONLY one label. For yes answer with 1. For no answer with 0.\n\n"
        "Question 1: {{question1}}\n"
        "Question 2: {{question2}}",
    ],
    "glue_sst2": [
        "What is the overall sentiment expressed in the following sentence? "
        "Answer with ONLY one label. For positive answer with 1. For negative answer with 0.\n\n"
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
    "bbh_word_sorting": [
        "{{input}}\nAnswer only with the list of sorted words. The list should be in a single line. The words shall be separated by a space.",
    ],
}

datasets = [
    ("squad", "qa", "answers"),
    ("cnn_dailymail", "summarization", "highlights"),
    ("xsum", "summarization", "summary"),
    ("glue_mnli", "classification", "label"),
    ("glue_qqp", "classification", "label"),
    ("glue_sst2", "classification", "label"),
    ("gsm8k", "reasoning", "answer"),
    ("natural_questions", "qa", "answer"),
    ("mbpp", "coding", ["code", "test_list"]),
]

sets = [sets[0] for sets in datasets]
tasks = [sets[1] for sets in datasets]
targets = [sets[2] for sets in datasets]

s = _sample_from_datasets(hf_dir, sets, 1e10)


for task, (title, value), target in zip(tasks, s.items(), targets):
    rows = []
    for sample in tqdm(value, total=len(value), desc=f"Prompt generation for {title}"):
        prompt = _build_prompt(title, sample, prompt_templates)
        if isinstance(target, str):
            rows.append({"task": task, "prompt": prompt, "target": sample[target]})
        else:
            rows.append(
                {
                    "task": task,
                    "prompt": prompt,
                    "code": sample[target[0]],
                    "test_list": sample[target[1]],
                }
            )
    df = pd.DataFrame(rows)
    filename = f"{hf_dir}targets/prepared-{title}.pkl"
    df.to_pickle(filename)
    print(f"Saved to {filename}")


oc = load_from_disk(f"{hf_dir}bbh_object_counting")
be = load_from_disk(f"{hf_dir}bbh_boolean_expressions")
ws = load_from_disk(f"{hf_dir}bbh_word_sorting")

mat = load_from_disk(f"{hf_dir}bbh_multistep_arithmetic_two")
tsoto = load_from_disk(f"{hf_dir}bbh_tracking_shuffled_objects_three_objects")
ts = load_from_disk(f"{hf_dir}bbh_temporal_sequences")

ff = load_from_disk(f"{hf_dir}bbh_formal_fallacies")
ldso = load_from_disk(f"{hf_dir}bbh_logical_deduction_seven_objects")
h = load_from_disk(f"{hf_dir}bbh_hyperbaton")

datasets = [
    "object_counting",
    "boolean_expressions",
    "word_sorting",
    "multistep_arithmetic_two",
    "tracking_shuffled_objects_three_objects",
    "temporal_sequences",
    "formal_fallacies",
    "logical_deduction_seven_objects",
    "hyperbaton",
]

rows = []
for ds in datasets:
    for v in load_from_disk(f"{hf_dir}bbh_{ds}")["test"]:
        prompt = _build_prompt("bbh", v, prompt_templates)
        if ds == "word_sorting":
            prompt = _build_prompt("bbh_word_sorting", v, prompt_templates)
        rows.append(
            {"task": "reasoning", "subset": ds, "prompt": prompt, "target": v["target"]}
        )

df = pd.DataFrame(rows)

# Create index inside each subset
df["order"] = df.groupby("subset").cumcount()

# Sort to interleave subsets
df = df.sort_values(["order", "subset"]).drop(columns="order")

filename = f"{hf_dir}targets/prepared-bbh.pkl"
df.to_pickle(filename)

print(f"Saved to {filename}")
