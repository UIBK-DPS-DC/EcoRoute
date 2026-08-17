import logging
import os
from datetime import datetime
from typing import List

import numpy as np
import vowpalwabbit

from .config import RoutingConfig
from .database import AvailableAction, AvailableModel, AvailableRouter

logger = logging.getLogger(__name__)


class Routing:
    def __init__(self, config: RoutingConfig):
        self.save_dir = config.mab_model_save_dir
        if config.mab_model_path is None:
            self.vw = vowpalwabbit.Workspace("--cb_explore_adf", quiet=True)
        else:
            self.vw = vowpalwabbit.Workspace(
                f"--cb_explore_adf -i {config.mab_model_path}", quiet=True
            )
        logger.info("Routing initialized")

    def predict(self, router_library, llm_library: List[AvailableModel], context):
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
