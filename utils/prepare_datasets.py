# Downloads training datasets to disk

from datasets import load_dataset


save_dir = "../hf_datasets/"

DATASETS = {
    "squad": ("squad", None),
    "natural_questions": ("natural_questions", None),
    "cnn_dailymail": ("cnn_dailymail", "3.0.0"),
    "xsum": ("xsum", None),
    "glue_mnli": ("glue", "mnli"),
    "glue_qqp": ("glue", "qqp"),
    "glue_sst2": ("glue", "sst2"),
    # "daily_dialog": ("daily_dialog", None),
    # "persona_chat": ("persona_chat", None),
    "gsm8k": ("openai/gsm8k", "main"),
    "bbh_object_counting": ("lukaemon/bbh", "object_counting"),
    "bbh_boolean_expressions": ("lukaemon/bbh", "boolean_expressions"),
    "bbh_word_sorting": ("lukaemon/bbh", "word_sorting"),
    "bbh_multistep_arithmetic_two": ("lukaemon/bbh", "multistep_arithmetic_two"),
    "bbh_tracking_shuffled_objects_three_objects": ("lukaemon/bbh", "tracking_shuffled_objects_three_objects"),
    "bbh_temporal_sequences": ("lukaemon/bbh", "temporal_sequences"),
    "bbh_formal_fallacies": ("lukaemon/bbh", "formal_fallacies"),
    "bbh_logical_deduction_seven_objects": ("lukaemon/bbh", "logical_deduction_seven_objects"),
    "bbh_hyperbaton": ("lukaemon/bbh", "hyperbaton"),
    "natural_questions": ("sentence-transformers/natural-questions", None),
    "mbpp": ("nlile/mbpp", "full")
}

for name, (dataset_name, subset) in DATASETS.items():
    print(f"Downloading {name}...")

    if subset:
        dataset = load_dataset(dataset_name, subset)
    else:
        dataset = load_dataset(dataset_name)

    dataset.save_to_disk(f"../hf_datasets/{name}")

print("Finished downloading")
