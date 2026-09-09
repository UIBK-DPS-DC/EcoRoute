import logging
import os
import uuid

from fastapi import FastAPI
from nats.aio.client import Client as NATS
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from .classification import TaskClassifier
from .config import load_config
from .jetstream import JetStreamClient
from .models import Query, QueryRequest
from .router import Router

app = FastAPI()


nc = NATS()
js = None


@app.on_event("startup")
async def startup():
    await main()


@app.on_event("shutdown")
async def shutdown():
    await nc.close()


@app.post("/route")
async def route_query(request: QueryRequest):
    query_id = str(uuid.uuid4())

    query = Query(
        query_id=query_id,
        query=request.query,
        origin_site=None,
        reference=request.reference,
    )

    try:
        response = await router.handle_query(query)
        logger.info(f"Received response to send to user: {response}")
    except Exception as e:
        logger.exception(f"handle_query threw an exception")
        return "Error: Internal Server Error"

    if response is None:
        return "Error: There are no models available to serve your query"

    return response["output"]


@app.get("/routers")
def get_routers():
    return router.get_routers()


@app.get("/save")
def save_model():
    filename = router.save_mab_model()
    return f"Saved Multi-Armed-Bandit model to {filename}"


@app.get("/reset")
def reset_model():
    router.reset_mab_model()
    return f"Reset Multi-Armed-Bandit model"


async def main():
    global js
    global router
    global logger

    config = load_config()

    logging_level = logging.INFO
    if config.router.router_logging_level.lower() == "debug":
        logging_level = logging.DEBUG

    router_id = os.getenv("ROUTER_ID", "router")

    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        filename=f"{config.router.router_log_file}app-{router_id}.log",
    )
    logger = logging.getLogger(__name__)

    client = JetStreamClient()
    await client.connect()
    await client.setup_streams()

    js = client.js

    model = SentenceTransformer(config.task_classifier.model_name)

    task_classifier = TaskClassifier(model, config.task_classifier)

    datasets = [
        ("squad", "qa"),
        ("cnn_dailymail", "summarization"),
        ("xsum", "summarization"),
        ("glue_mnli", "classification"),
        ("glue_qqp", "classification"),
        ("glue_sst2", "classification"),
        ("mbpp", "coding"),
        ("gsm8k", "reasoning"),
        ("natural_questions", "qa"),
    ]
    task_classifier.train(datasets)

    router = Router(js=js, task_classifier=task_classifier, config=config)
    await router.init()

    logger.info("Router started, waiting for messages...")
