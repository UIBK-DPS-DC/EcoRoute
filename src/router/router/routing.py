import logging
import os
import random
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

import numpy as np
import vowpalwabbit

from .config import RoutingConfig
from .database import AvailableAction, AvailableModel, AvailableRouter

logger = logging.getLogger(__name__)


class RoutingAlgorithm(ABC):
    def __init__(self, config: RoutingConfig):
        self.config = config

    @abstractmethod
    def predict(
        self,
        router_library: List[AvailableRouter],
        llm_library: List[AvailableModel],
        context,
    ):
        pass

    @abstractmethod
    def learn(
        self, router_library, llm_library, context, selected_llm, selection_prob, cost
    ):
        pass

    def save_model(self) -> str:
        pass

    def reset_model(self):
        pass


class Heuristic(RoutingAlgorithm):
    def __init__(self, config):
        super().__init__(config)

        self.epsilon = 0.1

    def predict(
        self,
        router_library: List[AvailableRouter],
        llm_library: List[AvailableModel],
        context,
    ):
        num_actions = len(router_library) + len(llm_library)

        if random.random() < self.epsilon:
            return self._predict_random(num_actions=num_actions)

        llm_max_reward = None
        best_llm = None
        best_llm_idx = None
        for i, llm in enumerate(llm_library):
            if llm.mean_reward is None:
                continue
            elif llm_max_reward is None or llm.mean_reward > llm_max_reward:
                llm_max_reward = llm.mean_reward
                best_llm = llm
                best_llm_idx = i

        router_max_reward = None
        best_router = None
        best_router_idx = None
        for j, router in enumerate(router_library):
            if router.mean_reward is None:
                continue
            if router_max_reward is None or router.mean_reward > router_max_reward:
                router_max_reward = router.mean_reward
                best_router = router
                best_router_idx = j

        if best_llm is None and best_router is None:
            return self._predict_random(num_actions=num_actions)

        if best_router is None:
            return self._predict_idx(num_actions=num_actions, idx=best_llm_idx)

        if best_llm is None or router_max_reward > llm_max_reward:
            return self._predict_idx(
                num_actions=num_actions, idx=best_router_idx + len(llm_library)
            )

        return self._predict_idx(num_actions=num_actions, idx=best_llm_idx)

    def _predict_random(
        self,
        num_actions: int,
    ):
        selection_idx = random.randint(0, num_actions - 1)
        return self._predict_idx(num_actions=num_actions, idx=selection_idx)

    def _predict_idx(
        self,
        num_actions: int,
        idx: int,
    ):
        selection_probs = [0.0 for i in range(num_actions)]
        selection_probs[idx] = 1.0
        return selection_probs

    def learn(
        self, router_library, llm_library, context, selected_llm, selection_prob, cost
    ):
        pass


class Routing(RoutingAlgorithm):
    def __init__(self, config: RoutingConfig):
        self.save_dir = config.mab_model_save_dir

        mab_options = " ".join(["--cb_explore_adf", config.mab_options])

        if config.mab_model_path is None:
            self.vw = vowpalwabbit.Workspace(mab_options, quiet=True)
        else:
            self.vw = vowpalwabbit.Workspace(
                f"{mab_options} -i {config.mab_model_path}", quiet=True
            )
        logger.info("Routing initialized")

    def predict(
        self,
        router_library: List[AvailableRouter],
        llm_library: List[AvailableModel],
        context,
    ):
        vw_string = self._create_vw_format(router_library, llm_library, context)
        logger.info(f"VW format:\n{vw_string}")

        selection_probs = self.vw.predict(vw_string)

        # Avoid probabilities not summing to 1 due to rounding errors
        selection_probs[-1] = max(0, 1 - np.sum(selection_probs[0:-1]))
        return selection_probs

    def learn(
        self, router_library, llm_library, context, selected_llm, selection_prob, cost
    ):
        training_data = {selected_llm: {"cost": cost, "prob": selection_prob}}
        vw_string = self._create_vw_format(
            router_library, llm_library, context, training_data
        )
        logger.info(f"Learning string:\n{vw_string}")
        self.vw.learn(vw_string)

    def save_model(self) -> str:
        filename = os.path.join(self.save_dir, f"mab_model_{datetime.now()}.vw")
        self.vw.save(filename)
        return filename

    def reset_model(self):
        self.vw = vowpalwabbit.Workspace("--cb_explore_adf", quiet=True)

    def _create_vw_format(
        self,
        router_library: List[AvailableRouter],
        llm_library: List[AvailableModel],
        context,
        training_data=None,
    ):
        context_string = " ".join(
            [f"{cont}:{value}" for cont, value in context.items()]
        )
        shared_features = f"shared | {context_string}"
        action_string = self._create_actions_string(
            router_library=router_library,
            llm_library=llm_library,
            training_data=training_data,
        )

        return f"{shared_features}\n{action_string}"

    def _create_actions_string(
        self,
        router_library: List[AvailableRouter],
        llm_library: List[AvailableModel],
        training_data=None,
    ):
        model_string = self._create_model_string(
            llm_library=llm_library, training_data=training_data
        )
        router_string = self._create_router_string(
            llm_library=llm_library,
            router_library=router_library,
            training_data=training_data,
        )
        return f"{model_string}\n{router_string}"

    def _create_model_string(
        self, llm_library: List[AvailableModel], training_data=None
    ):
        return self._create_action_string(
            library=llm_library, id_name="model", training_data=training_data
        )

    def _create_router_string(
        self,
        router_library: List[AvailableRouter],
        llm_library: List[AvailableModel],
        training_data=None,
    ):
        return self._create_action_string(
            library=router_library,
            id_name="router",
            training_data=training_data,
            start_idx=len(llm_library),
        )

    def _create_action_string(
        self,
        library: List[AvailableAction],
        id_name: str,
        training_data=None,
        start_idx=0,
    ):
        if library is None or len(library) <= 0:
            return ""

        library.sort(key=lambda x: x.id)

        action_strings = []
        for i, action in enumerate(library):
            training_string = ""
            action_context = ""
            if training_data is not None and action.id in training_data:
                training_string = f"{i+start_idx}:{training_data[action.id]['cost']}:{training_data[action.id]['prob']} "

            if action.has_metrics():
                action_context = f"mean_reward={action.mean_reward} mean_response_time={action.mean_response_time} mean_energy={action.mean_energy}"

            action_context += f" pending_requests={action.pending_requests}"

            action_strings.append(
                f"{training_string}| {id_name}={action.id} {action_context}"
            )
        return "\n".join(action_strings)


ROUTING_ALGORITHMS = {
    "heuristic": Heuristic,
    "mab": Routing,
}


def get_algorithm(config: RoutingConfig) -> RoutingAlgorithm:
    try:
        return ROUTING_ALGORITHMS[config.algorithm](config)
    except KeyError:
        logger.warning(
            f"Unknown routing algorithm {config.algorithm}. Selecting default 'mab'"
        )
        return ROUTING_ALGORITHMS["mab"](config)
