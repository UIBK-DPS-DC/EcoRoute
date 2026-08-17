import argparse
import asyncio
import json
import os
import re
import time
from glob import glob

import httpx
import pandas as pd
from tqdm import tqdm


def create_json_for_sample(sample):
    target = sample["target"]
    if sample["dataset"] in ["glue_qqp", "glue_sst2", "glue_mnli"]:
        target = str(sample["target"])
    elif sample["dataset"] in ["squad", "mbpp"]:
        target = json.dumps(sample["target"])

    j = {
        "query": sample["prompt"],
        "reference": {
            "target": target,
            "dataset": sample["dataset"],
            "subset": sample["subset"] if sample["dataset"] == "bbh" else None,
        },
    }
    return j


def create_request(sample):
    target_column = "target"

    if sample["dataset"] == "mbpp":
        target_column = "test_list"

    reference = {"target": sample[target_column], "dataset": dataset}

    if sample["dataset"] == "bbh":
        reference["subset"] = sample["subset"]

    return {
        "query": sample["prompt"],
        "reference": reference,
    }


async def send_request(client, request):
    r = await client.post(
        f"{URL}:{PORT}/route",
        json=request,
    )
    return r.json()


async def send_at_rate(df, rps):
    interval = 1.0 / rps

    async with httpx.AsyncClient() as client:
        start = time.perf_counter()
        tasks = []

        for i, (_, row) in tqdm(
            enumerate(df.iterrows()), desc="Sending requests", total=df.shape[0]
        ):
            target_time = start + i * interval

            now = time.perf_counter()
            if target_time > now:
                await asyncio.sleep(target_time - now)

            tasks.append(
                asyncio.create_task(send_request(client, create_json_for_sample(row)))
            )

        await asyncio.gather(*tasks)


parser = argparse.ArgumentParser(prog="Trace generator")

parser.add_argument(
    "endpoint", type=str, help="URL of the router endpoint to send the queries to"
)
parser.add_argument("-P", "--port", type=str, default="8000")
parser.add_argument(
    "-n", "--num", type=float, default=1.0, help="Number of queries per second"
)
parser.add_argument(
    "-p",
    "--path",
    type=str,
    default="/code/hf_datasets/",
    help="Path to the prepared datasets",
)
parser.add_argument(
    "-s",
    "--skip",
    type=str,
    nargs="+",
    default=[],
    help="Names of datasets that shall be excluded for the samples",
)
parser.add_argument(
    "-d",
    "--datasets",
    type=str,
    nargs="+",
    default=[],
    help="Names of datasets that shall be used exclusively for the samples",
)

parser.add_argument("-t", "--time", type=int, default=60, help="Runtime in minutes")
parser.add_argument(
    "-e",
    "--end",
    type=int,
    default=2000,
    help="Amount of samples in the testset (from back to front)",
)


args = parser.parse_args()

URL = args.endpoint
PORT = args.port

data_path = args.path

used_datasets = args.datasets
exceptions = args.skip

num_samples = int(args.num * 60 * args.time)

datasets = []


for file in glob(os.path.join(data_path, "prepared-*.pkl")):
    pattern = r".*/prepared-(?P<dataset>.+)\.pkl$"
    m = re.match(pattern, file)
    if m:
        dataset = m.group(1)
        if len(used_datasets) > 0 and dataset not in used_datasets:
            continue
        elif len(exceptions) > 0 and dataset in exceptions:
            continue
        datasets.append(dataset)

num_samples_per_dataset = num_samples // len(datasets)

total_samples = []

for file in glob(os.path.join(data_path, "prepared-*.pkl")):
    pattern = r".*/prepared-(?P<dataset>.+)\.pkl$"
    m = re.match(pattern, file)

    if m:
        dataset = m.group(1)
    else:
        print("no dataset found")

    if len(used_datasets) > 0 and dataset not in used_datasets:
        continue
    elif len(exceptions) > 0 and dataset in exceptions:
        continue

    print(f"Adding samples for dataset {dataset}...")

    df = pd.read_pickle(file)
    test_set = df.tail(args.end)
    samples = test_set.sample(n=min(num_samples_per_dataset, len(test_set))).copy()

    if dataset == "mbpp":
        samples = samples.rename(columns={"test_list": "target"}).drop(
            columns=["code"], errors="ignore"
        )

    # Add dataset column
    samples["dataset"] = dataset

    total_samples.append(samples)

combined = pd.concat(total_samples, ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

asyncio.run(send_at_rate(combined, args.num))
