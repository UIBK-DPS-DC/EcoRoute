import logging


import asyncio
import json
import os
import random
import time
import uuid

import httpx
from dotenv import load_dotenv
from energy_tracker import EnergyTracker
from nats.aio.client import Client as NATS
from nats.errors import TimeoutError
from nats.js.errors import NoStreamResponseError
from config import load_config


def convert_llm_id(id: str):
    return id.replace(".", "-").replace("/", "-")


config = load_config()

logging_level = logging.INFO
if config.model.model_logging_level.lower() == "debug":
    logging_level = logging.DEBUG

logging.basicConfig(
    level=logging_level,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filename=config.model.model_log_file,
)
logger = logging.getLogger(__name__)

load_dotenv()

NATS_URL = os.getenv("NATS_URL", "nats")
SITE = os.getenv("SITE", "uc")
LOCAL = os.getenv("LOCAL", False)

LLM_ID = os.getenv("LLM_ID", None)

if LLM_ID is None:
    LLM_ID = f"{SITE}-{uuid.uuid4()}"

LLM_ID_CLEANED = convert_llm_id(LLM_ID)

heartbeat_interval = config.model.heartbeat_interval_seconds

logger.info(f"{LLM_ID=}, {NATS_URL=}")

semaphore = asyncio.Semaphore(config.model.max_concurrent_queries)
tracker = EnergyTracker(config.energy_tracker)


def create_request(prompt):
    if SITE == "uc":
        return "http://localhost:8000/v1/completions", {
            "model": LLM_ID,
            "prompt": prompt,
            "max_tokens": 200,
        }
    elif SITE == "tacc" or SITE == "edge":
        return "http://localhost:11434/api/chat", {
            "model": LLM_ID,
            "stream": False,
            "keep_alive": "24h",
            "messages": [{"role": "user", "content": prompt}],
            "options": {"num_predict": config.model.max_output_tokens},
        }


async def call_vllm_remote(prompt, active_queries=None):
    url, body = create_request(prompt)
    timeout = httpx.Timeout(config.model.llm_request_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            url,
            json=body,
        )
        return r.json()


async def call_vllm_local(prompt, active_queries=None):

    latency = random.gauss(performance["latency"]["mu"], performance["latency"]["std"])

    logger.info(f"Waiting for {latency} seconds")

    if active_queries is None:
        await asyncio.sleep(latency)
    else:
        # Simulating query congestion under high workload
        # await asyncio.sleep(latency + 1.2**active_queries)
        await asyncio.sleep(latency)

    return {"choices": [{"text": json.dumps(performance["tasks"])}]}

    # return {
    #     "choices": [
    #         {
    #             "text": "Lorem ipsum dolor sit amet, "
    #             "consetetur sadipscing elitr, sed diam "
    #             "nonumy eirmod tempor invidunt ut labore "
    #             "et dolore magna aliquyam erat, sed diam "
    #             "voluptua. At vero eos et accusam et justo "
    #             "duo dolores et ea rebum. Stet clita kasd "
    #             "gubergren, no sea takimata sanctus est "
    #             "Lorem ipsum dolor sit amet."
    #         }
    #     ]
    # }


call_vllm = call_vllm_local if LOCAL else call_vllm_remote


async def heartbeat(performance):
    registration = {"llm_id": LLM_ID_CLEANED, "performance": performance}

    while True:
        await js.publish(
            f"llm.update.{SITE}.{LLM_ID_CLEANED}",
            json.dumps(registration).encode(),
        )

        logger.debug(f"{LLM_ID_CLEANED} heartbeat sent")
        await asyncio.sleep(heartbeat_interval)


async def handle_request(msg):
    receive_time = time.perf_counter()
    receive_timestamp = time.time()
    query = json.loads(msg.data.decode())
    query_id = query["query_id"]
    prompt = query["query"]
    logger.info(f"Received request {query_id}")

    await tracker.increase_active_queries()
    tracker.init_query(query_id)

    start_time = time.perf_counter()
    start_timestamp = time.time()

    try:
        response = await call_vllm(prompt, tracker.active_queries)
    except:
        logger.error(
            f"Response for {query_id} was not received. Returning 'model crashed'"
        )
        response = {"message": {"content": "model crashed"}}
    finally:
        await tracker.decrease_active_queries()
    end_time = time.perf_counter()
    end_timestamp = time.time()
    execution_time = end_time - start_time
    energy = tracker.retrieve_energy_for_query(query_id)

    publish_time = time.perf_counter()
    publish_timestamp = time.time()
    await js.publish(
        f"routing.response.{SITE}.{query['query_id']}",
        json.dumps(
            {
                "query_id": query["query_id"],
                "output": (
                    response["choices"][0]["text"]
                    if "choices" in response
                    else response["message"]["content"]
                ),
                "execution_time": execution_time,
                "energy": energy,
                "model_receive_time": receive_time,
                "publish_time": publish_time,
                "inference_start_time": start_time,
                "inference_finish_time": end_time,
                "model_receive_timestamp": receive_timestamp,
                "publish_timestamp": publish_timestamp,
                "inference_start_timestamp": start_timestamp,
                "inference_finish_timestamp": end_timestamp,
            }
        ).encode(),
    )
    logger.info(f"Sent response for {query_id}")
    await msg.ack()


async def worker():
    global js
    global performance

    nc = NATS()
    await nc.connect(f"nats://{NATS_URL}:4222")
    js = nc.jetstream()

    logger.info(f"{LLM_ID_CLEANED} connected to NATS")

    tasks = [
        "qa",
        "summarization",
        "classification",
        "coding",
        "reasoning",
    ]

    latency_mu_min = 5.0
    latency_mu_max = 9.0
    latency_std_min = 0.1
    latency_std_max = 2.0
    if SITE == "tacc":
        latency_mu_min = 9.0
        latency_mu_max = 15.0
        latency_std_min = 1.5
        latency_std_max = 3.0
    elif SITE == "edge":
        latency_mu_min = 15.0
        latency_mu_max = 20.0
        latency_std_min = 1.5
        latency_std_max = 3.0

    # just for testing
    # used to draw samples for reward, latency and energy for each model
    # to test the MAB routing
    performance = {
        "tasks": {},
        "latency": {
            "mu": random.uniform(latency_mu_min, latency_mu_max),
            "std": random.uniform(latency_std_min, latency_std_max),
        },
        "energy": {
            "mu": random.uniform(10000.0, 50000.0),
            "std": random.uniform(100.0, 5000.0),
        },
    }

    logger.info(
        f"Performance setting: {latency_mu_min=}, {latency_mu_max=}, {latency_std_min=}, {latency_std_max=} | {performance['latency']['mu']=}, {performance['latency']['std']=}"
    )

    for task in tasks:
        performance["tasks"][task] = {
            "mu": random.uniform(0.0, 1.0),
            "std": random.uniform(0.05, 0.07),
        }

    registration = {"llm_id": LLM_ID_CLEANED, "performance": performance}

    while True:
        try:
            await js.publish(
                f"llm.update.{SITE}.{LLM_ID_CLEANED}",
                json.dumps(registration).encode(),
            )
            break
        except NoStreamResponseError:
            logger.warning(
                f"Stream 'llm.update.{SITE}.{LLM_ID_CLEANED}' not ready. Retrying..."
            )
            await asyncio.sleep(2)

    sub = await js.pull_subscribe(
        f"routing.request.{SITE}.{LLM_ID_CLEANED}", f"worker-{LLM_ID_CLEANED}"
    )
    logger.info(f"Subscribing to routing.request.{SITE}.{LLM_ID_CLEANED}")

    logger.info(f"{LLM_ID_CLEANED} registered")

    asyncio.create_task(heartbeat(performance))

    async def process(msg):
        async with semaphore:
            await handle_request(msg)

    while True:
        try:
            msgs = await sub.fetch(1, timeout=1)
        except:
            continue

        for msg in msgs:
            asyncio.create_task(process(msg))


async def main():
    monitor_task = asyncio.create_task(tracker.run())
    worker_task = asyncio.create_task(worker())

    await asyncio.gather(monitor_task, worker_task)


asyncio.run(main())
