import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MetricNormalizer(ABC):
    required_keys = ()

    def normalize(self, bounds: dict, value: float):
        missing = [key for key in self.required_keys if key not in bounds]

        if missing:
            logger.warning(
                f"{self.__class__.__name__}: "
                f"missing required bounds entries: {missing}"
            )
            return value

        return self._normalize_impl(bounds, value)

    @abstractmethod
    def _normalize_impl(self, bounds: dict, value: float):
        pass


class MeanNormalizer(MetricNormalizer):
    required_keys = ("mean", "std")

    def _normalize_impl(self, bounds, value):
        return (value - bounds["mean"]) / bounds["std"]


class MinMaxNormalizer(MetricNormalizer):
    required_keys = ("min", "max")

    def _normalize_impl(self, bounds, value):
        min_value = bounds["min"]
        max_value = bounds["max"]
        return (value - min_value) / (max_value - min_value)


NORMALIZERS = {
    "mean": MeanNormalizer,
    "min-max": MinMaxNormalizer,
}


def get_normalizer(name: str) -> MetricNormalizer:
    try:
        return NORMALIZERS[name]()
    except KeyError:
        logger.warning(f"Unknown normalizer {name}. Selecting default 'mean'")
        return NORMALIZERS["mean"]()
