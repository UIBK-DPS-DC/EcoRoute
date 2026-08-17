import argparse
import asyncio
import logging
import time

import httpx
import pandas as pd
from tqdm import tqdm
from energy_tracker import EnergyTracker
from uuid import uuid4


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)



async def call_vllm_remote(prompt, model):
    timeout = httpx.Timeout(60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": model,
                # "prompt": prompt,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
            },
        )
        return r.json()
async def call_vllm_local(prompt, model):
    return {
        "choices": [
            {
                "text": "Lorem ipsum dolor sit amet, " \
                        "consetetur sadipscing elitr, sed diam " \
                        "nonumy eirmod tempor invidunt ut labore " \
                        "et dolore magna aliquyam erat, sed diam " \
                        "voluptua. At vero eos et accusam et justo " \
                        "duo dolores et ea rebum. Stet clita kasd " \
                        "gubergren, no sea takimata sanctus est " \
                        "Lorem ipsum dolor sit amet."
            }
        ]
    }

datasets = ["cnn_dailymail", "xsum", "glue_mnli", "glue_qqp", "glue_sst2", "squad","mbpp", "natural_questions", "gsm8k", "bbh"]
# datasets = ["bbh"]

# models = [
#     "Qwen/Qwen2.5-1.5B-Instruct",
#     "Qwen/Qwen2.5-7B-Instruct",
#     "Qwen/Qwen2.5-Coder-3B-Instruct",
#     "Qwen/Qwen2.5-Coder-7B-Instruct",
#     "Qwen/Qwen2.5-0.5B-Instruct",
#     "Qwen/Qwen2.5-3B-Instruct",
#     "mistralai/Mistral-7B-Instruct-v0.3",
#     "google/gemma-3-1b-it",
#     "google/gemma-3-4b-it",
#     "google/gemma-3-12b-it",
#     "meta-llama/Llama-3.2-1B-Instruct",
#     "meta-llama/Llama-3.1-8B-Instruct",
#     "google/pegasus-xsum",
#     "google/pegasus-cnn_dailymail"
# ]

models = {
    "Qwen/Qwen2.5-1.5B-Instruct": 0,
    "Qwen/Qwen2.5-7B-Instruct": 1,
    "Qwen/Qwen2.5-Coder-3B-Instruct": 2,
    "Qwen/Qwen2.5-Coder-7B-Instruct": 3,
    "Qwen/Qwen2.5-0.5B-Instruct": 4,
    "Qwen/Qwen2.5-3B-Instruct": 5,
    "google/gemma-4-E2B-it": 6,
    "google/gemma-4-E4B-it": 7,
    "google/gemma-4-31B-it": 8,
    "meta-llama/Llama-3.2-1B-Instruct": 9,
    "meta-llama/Llama-3.2-3B-Instruct": 10,
    "meta-llama/Llama-3.1-8B-Instruct": 11,
}

async def worker():

    parser = argparse.ArgumentParser(prog="Trace generator")

    parser.add_argument("model", type=str)
    parser.add_argument("-n", type=int, default=500)
    parser.add_argument("-s", "--start", type=int, default=0)
    parser.add_argument("-p", "--path", type=str, default="../src/hf_datasets/")
    parser.add_argument("-l", "--local", action="store_true", default=False)

    
    args = parser.parse_args()

    model = args.model
    N = args.n
    S = args.start
    path = args.path

    logger.info(models)
    model_idx = models[model]

    call_vllm = call_vllm_remote
    if(args.local):
        call_vllm = call_vllm_local


    logger.info(f"Running locally: {args.local}")

    logger.info(f"From {S} to {S+N} with model {model} ({model_idx})")
    
    for ds in datasets:
        logger.info(f"========== {ds} ==========")
        df = pd.read_pickle(f"{path}targets/prepared-{ds}.pkl")
        df = df.iloc[S:S+N]
        df_annot = df.copy()
        responses = []
        times = []
        energies = []
        finish_reasons = []
        for r in tqdm(df.itertuples(), total=df.shape[0], desc=f"Processing prompts in {ds} for {model}"):
            await tracker.increase_active_queries()
            query_id = uuid4()
            tracker.init_query(query_id)
            start = time.perf_counter()
            try:
                resp = await call_vllm(r.prompt, model)
            finally:
                await tracker.decrease_active_queries()
            exec_time = time.perf_counter() - start
            energy = tracker.retrieve_energy_for_query(query_id)

            # responses.append(resp["choices"][0]["text"])
            responses.append(resp["choices"][0]["message"]["content"])
            times.append(exec_time)
            energies.append(energy)
            finish_reasons.append(resp["choices"][0]["finish_reason"])
        
        df_annot[f"response_{model}"] = responses
        df_annot[f"time_{model}"] = times
        df_annot[f"energy_{model}"] = energies
        df_annot[f"finish_reason_{model}"] = finish_reasons
        filename = f"{path}targets/processed-{ds}-{model_idx}-{S:04}_{S+N:04}.pkl"
        df_annot.to_pickle(filename)
    return
    

tracker = EnergyTracker()

async def main():
    monitor_task = asyncio.create_task(tracker.run())
    worker_task = asyncio.create_task(worker())

    try:
        await worker_task  # wait until worker is done
    finally:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

asyncio.run(main())