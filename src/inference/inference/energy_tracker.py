import asyncio
import time
import logging
import platform
import psutil
from config import EnergyTrackerConfig

logger = logging.getLogger(__name__)


class EnergyBackend:
    def get_energy_joules(self):
        raise NotImplementedError

    def get_power_watts(self):
        raise NotImplementedError


class NvidiaBackend(EnergyBackend):
    def __init__(self):
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()

        self.handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i)
            for i in range(pynvml.nvmlDeviceGetCount())
        ]

        self.has_energy_counter = True

        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handles[0])
        except Exception:
            self.has_energy_counter = False

    def get_energy_joules(self):
        if not self.has_energy_counter:
            raise RuntimeError("Energy counter unavailable")

        total = 0.0
        for h in self.handles:
            total += self.nvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0
        return total

    def get_power_watts(self):
        total = 0.0
        for h in self.handles:
            total += self.nvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        return total


class AMDBackend(EnergyBackend):
    def __init__(self):
        import amdsmi

        self.amdsmi = amdsmi

        amdsmi.amdsmi_init()

        self.gpus = amdsmi.amdsmi_get_processor_handles()

        self.has_energy_counter = True

        try:
            amdsmi.amdsmi_get_energy_count(self.gpus[0])
        except Exception:
            self.has_energy_counter = False

    def get_energy_joules(self):
        if not self.has_energy_counter:
            raise RuntimeError("Energy counter unavailable")

        total = 0.0

        for gpu in self.gpus:
            e = self.amdsmi.amdsmi_get_energy_count(gpu)

            total += e["energy_accumulator"] * e["counter_resolution"] / 1e6

        return total

    def get_power_watts(self):
        total = 0.0

        for gpu in self.gpus:
            p = self.amdsmi.amdsmi_get_power_info(gpu)

            total += p["current_socket_power"] / 1e6

        return total


class RaspberryPiBackend(EnergyBackend):
    def __init__(self):
        self.idle_power = 3.0  # estimated value
        self.max_power = 7.0  # estimated value
        self.has_energy_counter = False

    def get_power_watts(self):
        cpu = psutil.cpu_percent(interval=None)

        return self.idle_power + (self.max_power - self.idle_power) * cpu / 100.0


def detect_backend():
    try:
        import pynvml

        pynvml.nvmlInit()
        logger.info("Detected NVIDIA GPU(s)")
        return NvidiaBackend()
    except:
        pass

    try:
        import amdsmi

        amdsmi.amdsmi_init()
        logger.info("Detected AMD GPU(s)")
        return AMDBackend()
    except:
        pass

    if platform.machine() in ("armv7l", "aarch64", "x86_64"):
        return RaspberryPiBackend()

    logger.warning("No supported energy backend found")
    return None


class EnergyTracker:
    def __init__(self, config: EnergyTrackerConfig):
        self.backend = detect_backend()

        self.sample_interval = config.sample_interval

        self.active_queries = 0
        self.total_energy = 0.0
        self.energy_per_query = {}

        self.running = True
        self.lock = asyncio.Lock()

    async def increase_active_queries(self):
        async with self.lock:
            self.active_queries += 1

    async def decrease_active_queries(self):
        async with self.lock:
            self.active_queries -= 1

    def init_query(self, query_id):
        self.energy_per_query[query_id] = 0.0

    def retrieve_energy_for_query(self, query_id):
        value = self.energy_per_query.pop(query_id)
        return value

    async def run(self):
        if self.backend is None:
            logger.warning("GPU energy tracking disabled")
            return

        if self.backend.has_energy_counter:
            previous_energy = self.backend.get_energy_joules()

            while self.running:
                await asyncio.sleep(self.sample_interval)

                current_energy = self.backend.get_energy_joules()

                delta_energy = max(0.0, current_energy - previous_energy)

                previous_energy = current_energy

                if self.active_queries > 0:
                    self.total_energy += delta_energy

                    share = delta_energy / self.active_queries

                    for qid in self.energy_per_query:
                        self.energy_per_query[qid] += share

        else:
            last_time = time.perf_counter()

            while self.running:
                await asyncio.sleep(self.sample_interval)

                now = time.perf_counter()
                dt = now - last_time
                last_time = now

                power = self.backend.get_power_watts()

                delta_energy = power * dt

                if self.active_queries > 0:
                    self.total_energy += delta_energy

                    share = delta_energy / self.active_queries

                    for qid in self.energy_per_query:
                        self.energy_per_query[qid] += share
