import asyncio
import logging
import os

from nats.aio.client import Client as NATS
from nats.js.api import StreamConfig

logger = logging.getLogger(__name__)

NATS_URL = os.getenv("NATS_URL", "nats")


class JetStreamClient:
    def __init__(self, servers=f"nats://{NATS_URL}:4222"):
        self.servers = servers
        self.nc = NATS()
        self.js = None
        self._connect_lock = asyncio.Lock()

    async def connect(self):
        async with self._connect_lock:
            if self.nc.is_connected:
                return

            logger.info("Connecting to NATS...")

            while True:
                try:
                    await self.nc.connect(self.servers)
                    self.js = self.nc.jetstream()

                    logger.info("Connected to NATS")
                    break
                except Exception as e:
                    try:
                        await self.nc.close()
                    except Exception:
                        pass
                    logger.warning(f"NATS not ready: {e}")
                    await asyncio.sleep(2)

            while True:
                try:
                    await self.nc.jetstream().account_info()
                    logger.info("Jetstream ready")
                    break
                except Exception as e:
                    logger.warning(f"Jetstream not ready: {e}")
                    await asyncio.sleep(2)

    async def setup_streams(self):
        streams = [
            StreamConfig(
                name="LLM_UPDATES",
                subjects=["llm.update.>"],
            ),
            StreamConfig(
                name="ROUTING_REQUESTS",
                subjects=["routing.request.>"],
            ),
            StreamConfig(
                name="ROUTING_REQUESTS_ROUTER",
                subjects=["routing.request-router.>"],
            ),
            StreamConfig(
                name="ROUTING_RESPONSE",
                subjects=["routing.response.>"],
            ),
        ]

        for cfg in streams:
            try:
                await self.js.add_stream(cfg)
                logger.info(f"Created stream {cfg.name}")
            except Exception as e:
                logger.info(f"Stream {cfg.name} already exists: {e}")
