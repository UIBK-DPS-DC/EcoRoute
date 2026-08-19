import logging
import os
import random
from datetime import datetime
from typing import List, Optional

import duckdb
from pydantic import BaseModel

from .config import DatabaseConfig
from .models import QueryTimes

logger = logging.getLogger(__name__)


class RouterPerformance(BaseModel):
    task: str
    mean_reward: Optional[float] = None
    mean_response_time: Optional[float] = None
    mean_energy: Optional[float] = None


class AvailableAction(BaseModel):
    id: str
    mean_reward: Optional[float] = None
    mean_response_time: Optional[float] = None
    mean_energy: Optional[float] = None
    pending_requests: Optional[int] = None

    def has_metrics(self) -> bool:
        return (
            self.mean_reward is not None
            and self.mean_response_time is not None
            and self.mean_energy is not None
        )

    def __lt__(self, other):
        less_than = self.id.__lt__(other.id)
        return less_than


class AvailableRouter(AvailableAction):
    pass


class AvailableModel(AvailableAction):
    pass


class TrainingSample(BaseModel):
    query_id: str
    timestamp: datetime
    llm_id: str
    site: str
    task: str
    response: Optional[str] = None
    reward: Optional[float] = None
    output_quality: Optional[float] = None
    normalized_output_quality: Optional[float] = None
    response_time: float
    network_time: Optional[float] = None
    routing_time: Optional[float] = None
    execution_time: Optional[float] = None
    energy: Optional[float] = None
    routing_confidence: float
    processed: bool = False
    target: Optional[str] = None
    dataset: Optional[str] = None
    subset: Optional[str] = None
    available_models: List[AvailableModel]
    available_routers: Optional[List[AvailableRouter]]

    def failed(self) -> bool:
        return self.response is None


class DuckDB:
    def __init__(self, config: DatabaseConfig, router_id: str):
        self.db_file = self._create_db_file_name(config, router_id)
        self.con = duckdb.connect(self.db_file)
        self.metrics_since_last_training_batch = 0

        self.con.execute("""
    CREATE TABLE IF NOT EXISTS metrics (
        query_id VARCHAR,
        timestamp TIMESTAMP,
        llm_id VARCHAR,
        site VARCHAR,
        task VARCHAR,
        response VARCHAR DEFAULT NULL,
        reward DOUBLE DEFAULT NULL,
        output_quality DOUBLE DEFAULT NULL,
        normalized_output_quality DOUBLE DEFAULT NULL,
        response_time DOUBLE,
        network_time DOUBLE DEFAULT NULL,
        routing_time DOUBLE,
        execution_time DOUBLE DEFAULT NULL,
        energy DOUBLE DEFAULT NULL,
        routing_confidence DOUBLE,
        processed BOOLEAN DEFAULT FALSE      
    )
    """)
        self.con.execute("""                 
    CREATE TABLE IF NOT EXISTS llm (
        query_id VARCHAR,
        id VARCHAR,
        mean_reward DOUBLE,
        mean_response_time DOUBLE,
        mean_energy DOUBLE,
        pending INTEGER
    )
    """)

        self.con.execute("""                 
    CREATE TABLE IF NOT EXISTS reference (
        query_id VARCHAR,
        target VARCHAR,
        dataset VARCHAR,
        subset VARCHAR DEFAULT NULL
    )
    """)

        self.con.execute("""                 
    CREATE TABLE IF NOT EXISTS router (
        query_id VARCHAR,
        id VARCHAR,
        mean_reward DOUBLE,
        mean_response_time DOUBLE,
        mean_energy DOUBLE,
        pending INTEGER
    )
    """)
        self.con.execute("""
    CREATE TABLE IF NOT EXISTS request_time (
        query_id VARCHAR,
        timestamp TIMESTAMP,
        llm_id VARCHAR,
        request_received TIMESTAMP DEFAULT NULL,
        request_published TIMESTAMP DEFAULT NULL,
        request_from_router_received TIMESTAMP DEFAULT NULL,
        model_message_received TIMESTAMP DEFAULT NULL,
        model_inference_started TIMESTAMP DEFAULT NULL,
        model_inference_finished TIMESTAMP DEFAULT NULL,
        model_response_published TIMESTAMP DEFAULT NULL,
        router_response_received TIMESTAMP DEFAULT NULL,
        router_future_completed TIMESTAMP DEFAULT NULL
    )
    """)
        logger.info("Database initialized")

    def _create_db_file_name(self, config: DatabaseConfig, router_id: str):
        filename = config.file_name.replace("{{ROUTER_ID}}", router_id)
        return os.path.join(config.file_path, filename)

    def insert_query_times(self, query_id: str, llm_id: str, query_times: QueryTimes):
        con = duckdb.connect(self.db_file)

        params = [
            query_id,
            datetime.now(),
            llm_id,
            query_times.query_received,
            query_times.query_published,
            query_times.query_from_router_received,
            query_times.model_query_received,
            query_times.model_inference_started,
            query_times.model_inference_finshed,
            query_times.model_response_published,
            query_times.response_received,
            query_times.future_completed,
        ]

        query = """
            INSERT INTO request_time (
                query_id,
                timestamp,
                llm_id,
                request_received,
                request_published,
                request_from_router_received,
                model_message_received,
                model_inference_started,
                model_inference_finished,
                model_response_published,
                router_response_received,
                router_future_completed
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """

        con.execute(
            query,
            parameters=params,
        )

    def insert(
        self,
        query_id,
        llm_id,
        site,
        task,
        response,
        response_time,
        network_time,
        routing_time,
        execution_time,
        energy,
        routing_confidence,
        available_models: List[AvailableModel],
        available_routers: List[AvailableRouter],
        processed=False,
    ):
        # Needed for thread safety
        con = duckdb.connect(self.db_file)

        params = [
            query_id,
            datetime.now(),
            llm_id,
            site,
            task,
            response,
            response_time,
            network_time,
            routing_time,
            execution_time,
            energy,
            routing_confidence,
            processed,
        ]
        con.execute(
            f"INSERT INTO metrics (query_id, timestamp, llm_id, site, task, response, response_time, network_time, routing_time, execution_time, energy, routing_confidence, processed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            parameters=params,
        )
        self.metrics_since_last_training_batch += 1

        if len(available_models) > 0:
            llm_params = []
            for model in available_models:
                llm_params.append(
                    [
                        query_id,
                        model.id,
                        model.mean_reward,
                        model.mean_response_time,
                        model.mean_energy,
                        model.pending_requests,
                    ]
                )
            logger.info(f"Saving the following models: {llm_params}")
            con.executemany(
                f"INSERT INTO llm VALUES (?,?,?,?,?,?)", parameters=llm_params
            )

        if len(available_routers) > 0:
            router_params = []

            for router in available_routers:
                router_params.append(
                    [
                        query_id,
                        router.id,
                        router.mean_reward,
                        router.mean_response_time,
                        router.mean_energy,
                        router.pending_requests,
                    ]
                )
            logger.info(f"Saving the following routers: {router_params}")
            con.executemany(
                f"INSERT INTO router VALUES (?, ?, ?, ?, ?, ?)",
                parameters=router_params,
            )

    def insert_reference(self, query_id, target, dataset, subset):
        con = duckdb.connect(self.db_file)
        params = [query_id, target, dataset, subset]
        con.execute(f"INSERT INTO reference VALUES (?, ?, ?, ?)", parameters=params)

    def insert_test_metrics(self):
        for i in range(100):
            num_models = random.randint(3, 8)
            selected_model = random.randint(0, num_models)
            available_models = [(f"llm_id{i}", 0) for i in range(num_models)]

            self.insert(
                i,
                f"llm_id{selected_model}",
                "uc",
                "qa",
                0.0,
                0.5,
                0.001,
                0.01,
                0.4,
                100.0,
                0.23,
                available_models,
            )

    def get_training_batch(self):
        con = duckdb.connect(self.db_file)

        rows = con.execute("""
            WITH llm_agg AS (
                SELECT
                    query_id,
                    list(
                        struct_pack(
                            id := id,
                            mean_reward := mean_reward,
                            mean_response_time := mean_response_time,
                            mean_energy := mean_energy,
                            pending_requests := pending
                        )
                    ) AS available_models
                FROM llm
                GROUP BY query_id
            ),
            router_agg AS (
                SELECT
                    query_id,
                    list(
                        struct_pack(
                            id := id,
                            mean_reward := mean_reward,
                            mean_response_time := mean_response_time,
                            mean_energy := mean_energy,
                            pending_requests := pending
                        )
                    ) AS available_routers
                    
                FROM router
                GROUP BY query_id
            )

            SELECT
                m.*,
                l.available_models,
                r.available_routers,
                ref.target,
                ref.dataset,
                ref.subset
            FROM metrics m
            LEFT JOIN llm_agg l
                ON m.query_id = l.query_id
            LEFT JOIN router_agg r
                ON m.query_id = r.query_id
            LEFT JOIN reference ref
                ON m.query_id = ref.query_id
            WHERE processed = false
            ORDER BY m.timestamp;
        """).fetch_arrow_table().to_pylist()

        self.metrics_since_last_training_batch = 0

        return [TrainingSample.model_validate(row) for row in rows]

    def get_model_performance_metrics(self, llm_ids, task) -> list[AvailableModel]:
        con = duckdb.connect(self.db_file)

        placeholders = ",".join(["?"] * len(llm_ids))

        rows = (
            con.execute(
                f"""
            SELECT
                m.llm_id as id,
                AVG(m.reward) AS mean_reward,
                AVG(m.response_time) AS mean_response_time,
                AVG(m.energy) AS mean_energy
            FROM metrics m
            WHERE
                m.processed = true
                AND m.task = ?
                AND m.llm_id IN ({placeholders})
            GROUP BY m.llm_id
        """,
                [task, *llm_ids],
            )
            .fetch_arrow_table()
            .to_pylist()
        )

        # Build lookup from returned rows
        row_map = {row["id"]: row for row in rows}

        # Ensure every llm_id exists
        result = []
        for llm_id in llm_ids:
            row = row_map.get(
                llm_id,
                {
                    "id": llm_id,
                    "mean_reward": None,
                    "mean_response_time": None,
                    "mean_energy": None,
                },
            )

            result.append(AvailableModel.model_validate(row))

        return result

    def get_router_performance_metrics(self, site):
        con = duckdb.connect(self.db_file)

        params = [site]

        rows = (
            con.execute(
                """
        SELECT
            m.task,
            AVG(m.reward) AS mean_reward, 
            AVG(m.response_time) AS mean_response_time,
            AVG(m.energy) AS mean_energy
        FROM llm l
        JOIN metrics m USING (query_id)
        WHERE m.processed = true AND m.site = ?
        GROUP BY l.id, m.task
        """,
                parameters=params,
            )
            .fetch_arrow_table()
            .to_pylist()
        )

        return [RouterPerformance.model_validate(row) for row in rows]

    def set_processed(self, training_batch: List[TrainingSample]):
        params = [
            [
                batch.reward,
                batch.normalized_output_quality,
                batch.output_quality,
                batch.query_id,
            ]
            for batch in training_batch
        ]

        con = duckdb.connect(self.db_file)

        con.executemany(
            f"""
            UPDATE metrics SET reward = ?, normalized_output_quality = ?, output_quality = ?, processed = true WHERE query_id = ?
        """,
            parameters=params,
        )
