import asyncio
import json
import logging
import os
import time
from typing import Dict, List

import numpy as np
from nats.js.errors import BadRequestError, BucketNotFoundError

from .classification import Classifier
from .config import AppConfig
from .database import AvailableModel, AvailableRouter, DuckDB
from .models import PendingRequest, Query, QueryTimes
from .routing import Routing
from .trainer import Trainer

logger = logging.getLogger(__name__)


class Router:
    def __init__(
        self,
        js,
        task_classifier: Classifier,
        config: AppConfig,
    ):
        self.config = config.router
        self.router_id = os.getenv("ROUTER_ID", "router")
        self.site = os.getenv("SITE", "uc")

        self.llm_registry = {}
        self.router_registry = {}
        self.js = js
        self.task_classifier = task_classifier
        self.pending_requests: dict[str, PendingRequest] = {}
        self.pending_requests_per_llm: dict[str, int] = {}
        self.routing = Routing(config.routing)
        self.duckdb = DuckDB(config.database, self.router_id)
        self.trainer = Trainer(self.duckdb, self.routing, config.trainer)
        self.model_ttl = self.config.model_ttl
        self.heartbeat_interval = self.config.router_heartbeat_interval_seconds
        logger.info("Router initialized")

    async def init(self):
        logger.info("Creating bucket LLM_REGISTRY...")
        self.llm_kv = await self.create_kv_bucket("LLM_REGISTRY")
        logger.info("Creating bucket ROUTERS...")
        self.router_kv = await self.create_kv_bucket("ROUTERS")
        logger.info("Creating bucket QUERIES...")
        self.query_kv = await self.create_kv_bucket("QUERIES")
        logger.info("Registering to NATS...")
        await self.register()
        logger.info("Registered to NATS")
        logger.info("Starting NATS subscriptions...")
        await self.start_subscriptions()
        logger.info("Started NATS subscriptions")
        asyncio.create_task(self.watch_router_updates())
        asyncio.create_task(self.trainer.run())
        asyncio.create_task(
            self.register_task(interval_seconds=self.heartbeat_interval)
        )
        logger.info("Router all tasks started")

    async def create_kv_bucket(self, bucket):
        try:
            # Try to get existing bucket
            kv = await self.js.key_value(bucket)
            return kv

        except BucketNotFoundError:
            try:
                # Try to create it
                kv = await self.js.create_key_value(bucket=bucket)
                logger.info(f"Created {bucket} bucket")
                return kv

            except BadRequestError:
                # Someone else created it at the same time
                kv = await self.js.key_value(bucket)
                return kv

    async def register(self):
        performance_averages = self.duckdb.get_router_performance_metrics(self.site)

        payload = {
            "router_id": self.router_id,
            "host": self.router_id,
            "port": int(os.getenv("PORT", 8000)),
            "performance": {
                performance.task: {
                    "avg_reward": performance.mean_reward,
                    "avg_response_time": performance.mean_response_time,
                    "avg_energy": performance.mean_energy,
                }
                for performance in performance_averages
            },
        }

        await self.router_kv.put(
            f"routers.{self.router_id}", json.dumps(payload).encode()
        )

    async def register_task(self, interval_seconds=15):
        while True:
            try:
                await self.register()
            except Exception as e:
                logger.error(f"Router registration failed: {e}")

            await asyncio.sleep(interval_seconds)

    def save_mab_model(self) -> str:
        return self.routing.save_model()

    def reset_mab_model(self):
        self.routing.reset_model()

    def _get_available_routers(self, task) -> List[AvailableRouter]:
        available_routers = []
        for router_id, data in self.router_registry.items():
            if self.router_id in router_id:
                continue

            router = AvailableRouter(id=router_id)

            if "performance" in data and task in data["performance"]:
                performance = data["performance"][task]
                router.mean_reward = (
                    performance["mean_reward"] if "mean_reward" in performance else 0.0
                )
                router.mean_response_time = (
                    performance["mean_response_time"]
                    if "mean_response_time" in performance
                    else 0.0
                )
                router.mean_energy = (
                    performance["mean_energy"] if "mean_energy" in performance else 0.0
                )

            available_routers.append(router)

        return available_routers

    def _get_available_models(self):
        available_models = []
        for llm_id, data in self.llm_registry.items():
            if time.perf_counter() - data["last_update"] > self.model_ttl:
                logger.info(f"Skipping {llm_id}. Model is outdated.")
                continue

            available_models.append(llm_id)

        return available_models

    def generate_routing_context(self, query: str):
        if not self.config.use_task_classification:
            return {}

        task_pred, probs = self.task_classifier.predict(query)

        logger.info(
            f"Task classification: Predicted task = {task_pred} | Probabilities = {probs}"
        )

        return {"task": task_pred}

    def route(self, query: Query, pending_requests_per_llm: dict[str, int]):
        available_models = self._get_available_models()

        if not available_models:
            logger.warning("No LLMs available")
            return None

        context = self.generate_routing_context(query.query)

        possible_models = self.duckdb.get_model_performance_metrics(
            available_models, context["task"]
        )

        # Not allowing sending query to another router if the query was received from a router
        # Otherwise the original router learns to send queries to router B through router A
        # instead of sending to B directly
        available_routers = []
        if not query.sent_from_router():
            available_routers = self._get_available_routers(context["task"])

        for model in possible_models:
            model.pending_requests = 0
            if model.id in pending_requests_per_llm:
                model.pending_requests = pending_requests_per_llm[model.id]

        for router in available_routers:
            router.pending_requests = 0
            if router.id in pending_requests_per_llm:
                router.pending_requests = pending_requests_per_llm[router.id]

        selection_probs = self.routing.predict(
            available_routers, possible_models, context
        )
        logger.info(f"Model selection probabilities = {selection_probs}")

        selection_idx = np.random.choice(
            np.arange(len(selection_probs)), p=selection_probs
        )

        if selection_idx >= len(possible_models):
            selected_model = available_routers[selection_idx - len(possible_models)]
        else:
            selected_model = possible_models[selection_idx]
        logger.info(
            f"Selected model: {selected_model} ({selection_probs[selection_idx]})"
        )
        return (
            selected_model,
            selection_probs[selection_idx],
            context,
            possible_models,
            available_routers,
        )

    async def handle_llm_update(self, msg):
        data = json.loads(msg.data.decode())

        if "llm_id" not in data:
            logger.error(f"Invalid LLM update received. LLM_ID missing")
            await msg.ack()
            return

        data["last_update"] = time.perf_counter()

        self.llm_registry[data["llm_id"]] = data
        logger.debug(f"LLM update for {data['llm_id']} received")

        await msg.ack()

    async def start_subscriptions(self):
        await self.js.subscribe(
            f"llm.update.{self.site}.*",
            durable=f"router-llm-updates-{self.router_id}",
            cb=self.handle_llm_update,
        )

        await self.js.subscribe(
            f"routing.response.{self.site}.*",
            cb=self._handle_response,
            durable=f"router-response-consumer-{self.router_id}",
        )

        await self.js.subscribe(
            f"routing.request-router.{self.router_id}",
            cb=self._handle_request_from_router,
            durable=f"router-request-consumer-{self.router_id}",
        )

    async def _handle_response(self, msg):
        receive_time = time.perf_counter()
        receive_timestamp = time.time()
        logger.info(f"_handle_request: Received msg: {msg}")
        data = json.loads(msg.data.decode())
        query_id = data["query_id"]
        logger.info(f"Router received response for query {query_id}")
        data["router_receive_time"] = receive_time
        data["router_receive_timestamp"] = receive_timestamp

        pending_request = self.pending_requests.get(query_id)
        future = pending_request.future

        if future and not future.done():
            future.set_result(data)
        else:
            logger.info(f"There was no pending request for query {query_id}")

        await msg.ack()

    async def handle_query(
        self, query: Query, query_from_router_receive_timestamp: float = None
    ):
        start_time = time.perf_counter()
        start_time_timestamp = time.time()

        future = asyncio.get_event_loop().create_future()

        routing_start_time = time.perf_counter()

        pending_requests_per_llm_snapshot = self.pending_requests_per_llm

        routing_result = await asyncio.to_thread(
            self.route, query, pending_requests_per_llm_snapshot
        )
        if routing_result is None:
            return

        selection, prob, context, possible_models, available_routers = routing_result

        routing_time = time.perf_counter() - routing_start_time

        if isinstance(selection, AvailableRouter):
            logger.info(f"Selected a router: {selection.id}")
            selection_id = selection.id

            send_time = time.perf_counter()
            send_time_timestamp = time.time()
            query.origin_site = self.site
            await self.js.publish(
                f"routing.request-router.{selection_id}",
                query.model_dump_json().encode(),
            )
        elif isinstance(selection, AvailableModel):
            logger.info(f"Selected a model: {selection.id}")
            selection_id = selection.id
            response = {
                "query_id": query.query_id,
                "selected_llm": selection_id,
                "query": query.query,
            }

            send_time = time.perf_counter()
            send_time_timestamp = time.time()
            await self.js.publish(
                f"routing.request.{self.site}.{selection_id}",
                json.dumps(response).encode(),
            )

        self.pending_requests[query.query_id] = PendingRequest(
            future=future, llm_id=selection_id
        )
        if selection_id not in self.pending_requests_per_llm:
            self.pending_requests_per_llm[selection_id] = 0
        self.pending_requests_per_llm[selection_id] += 1

        logger.info(f"Routed {query.query_id} to {selection_id}")

        try:
            response = await asyncio.wait_for(future, timeout=self.config.query_timeout)
            future_complete_time = time.perf_counter()
            future_complete_timestamp = time.time()
        except asyncio.TimeoutError:
            del self.pending_requests[query.query_id]
            logger.error(f"Query {query.query_id} timeout")
            logger.info(f"Timeout Routing time: {routing_time}")
            logger.info(f"Timeout Execution time: not available")
            logger.info(f"Timeout Network time: not available")
            logger.info(f"Timeout Response time: {time.perf_counter() - start_time}")
            logger.info(f"Timeout Future complete time: {time.perf_counter()}")

            query_times = QueryTimes(
                query_received=start_time_timestamp,
                query_published=send_time_timestamp,
                query_from_router_received=query_from_router_receive_timestamp,
                model_query_received=None,
                model_inference_started=None,
                model_inference_finshed=None,
                model_response_published=None,
                response_received=None,
                future_completed=None,
            )

            self.duckdb.insert_query_times(query.query_id, selection_id, query_times)

            await self._store_query(
                id=query.query_id,
                llm_id=selection_id,
                task=context["task"],
                response=None,
                routing_time=routing_time,
                execution_time=None,
                network_time=None,
                response_time=time.perf_counter() - start_time,
                energy=None,
                selection_probability=prob,
                possible_models=possible_models,
                available_routers=available_routers,
            )

            if query.has_reference():
                await self._store_reference(query.query_id, query.reference)

            raise Exception("LLM response timeout")

        finally:
            self.pending_requests_per_llm[selection_id] -= 1

        del self.pending_requests[query.query_id]

        end_time = time.perf_counter()
        latency = end_time - start_time
        network_time = (
            response["router_receive_time"] - send_time - response["execution_time"]
        )

        router_receive_timestamp = query_from_router_receive_timestamp
        if (
            query_from_router_receive_timestamp is None
            and "query_from_router_received" in response
        ):
            router_receive_timestamp = response["query_from_router_received"]

        query_times = QueryTimes(
            query_received=start_time_timestamp,
            query_published=send_time_timestamp,
            query_from_router_received=router_receive_timestamp,
            model_query_received=response["model_receive_timestamp"],
            model_inference_started=response["inference_start_timestamp"],
            model_inference_finshed=response["inference_finish_timestamp"],
            model_response_published=response["publish_timestamp"],
            response_received=response["router_receive_timestamp"],
            future_completed=future_complete_timestamp,
        )

        self.duckdb.insert_query_times(query.query_id, selection_id, query_times)

        logger.info(f"Results for query {query.query_id}")
        logger.info(f"Energy: {response['energy']}")
        logger.info(f"Routing time: {routing_time}")
        logger.info(f"Execution time: {response['execution_time']}")
        logger.info(f"Network time: {network_time}")
        logger.info(f"Response time: {latency}")
        logger.info(
            f"Future delay: {future_complete_time - response['router_receive_time']}"
        )
        logger.info(f"Request receive time: {start_time}")
        logger.info(f"Request publish time: {send_time}")
        logger.info(f"Model receive time: {response['model_receive_time']}")
        logger.info(f"Model inference start time: {response['inference_start_time']}")
        logger.info(f"Model inference finish time: {response['inference_finish_time']}")
        logger.info(f"Model response publish time: {response['publish_time']}")
        logger.info(f"Response receive time: {response['router_receive_time']}")
        logger.info(f"Future complete time: {future_complete_time}")

        await self._store_query(
            query.query_id,
            selection_id,
            context["task"],
            response["output"],
            routing_time,
            response["execution_time"],
            network_time,
            latency,
            response["energy"],
            prob,
            possible_models,
            available_routers,
        )

        if query.has_reference():
            await self._store_reference(query.query_id, query.reference)

        if query_from_router_receive_timestamp is not None:
            response["query_from_router_received"] = query_from_router_receive_timestamp

        return response

    async def _in_process_heartbeat(self, msg):
        while True:
            await asyncio.sleep(5)
            await msg.in_progress()

    async def process_router_query(self, query, query_from_router_receive_timestamp):
        try:
            response = await self.handle_query(
                query, query_from_router_receive_timestamp
            )

            await self.js.publish(
                f"routing.response.{query.origin_site}.{query.query_id}",
                json.dumps(response).encode(),
            )
        except:
            logger.error(f"_handle_request_from_router failed")

    async def _handle_request_from_router(self, msg):
        query_from_router_receive_time = time.perf_counter()
        query_from_router_receive_timestamp = time.time()
        query = Query.model_validate_json(msg.data)

        asyncio.create_task(
            self.process_router_query(query, query_from_router_receive_timestamp)
        )

        try:
            await msg.ack()
        except:
            logger.error(f"ACK failed for {query.query_id}")

    async def _store_query(
        self,
        id,
        llm_id,
        task,
        response,
        routing_time,
        execution_time,
        network_time,
        response_time,
        energy,
        selection_probability,
        possible_models,
        available_routers,
    ):
        self.duckdb.insert(
            id,
            llm_id,
            self.site,
            task,
            response,
            response_time,
            network_time,
            routing_time,
            execution_time,
            energy,
            selection_probability,
            possible_models,
            available_routers,
        )

    async def _store_reference(self, id, reference):
        self.duckdb.insert_reference(
            id,
            reference.target,
            reference.dataset,
            reference.subset,
        )

    async def watch_router_updates(self):
        watcher = await self.router_kv.watch("routers.*")
        async for update in watcher:
            if update is None:
                continue
            data = json.loads(update.value.decode())
            router_id: str = update.key
            router_id = router_id.removeprefix("routers.")
            self.router_registry[router_id] = data
            logger.debug(f"Updated router registry for {router_id}")
            logger.info(f"Router info for {router_id}: {data}")

    def get_routers(self):
        return self.router_registry
