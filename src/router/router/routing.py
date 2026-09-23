import logging
import os
import random
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List
import asyncio

import numpy as np
import pandas as pd
import vowpalwabbit
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import RoutingConfig
from .database import AvailableAction, AvailableModel, AvailableRouter

logger = logging.getLogger(__name__)


class RoutingAlgorithm(ABC):
    def __init__(self, config: RoutingConfig, **kwargs):
        self.config = config

    @abstractmethod
    def predict(
        self,
        router_library: List[AvailableRouter],
        llm_library: List[AvailableModel],
        context,
        prompt: str,
    ):
        pass

    @abstractmethod
    def learn(
        self, router_library, llm_library, context, selected_llm, selection_prob, cost
    ):
        pass

    def model_ready(self):
        return True

    def save_model(self) -> str:
        pass

    def reset_model(self):
        pass


class Heuristic(RoutingAlgorithm):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.epsilon = 0.1

    def predict(
        self,
        router_library: List[AvailableRouter],
        llm_library: List[AvailableModel],
        context,
        prompt: str,
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
    def __init__(self, config: RoutingConfig, **kwargs):
        self.save_dir = config.model_save_dir

        mab_options = " ".join(["--cb_explore_adf -q TM", config.mab.mab_options])

        if config.model_path is None:
            self.vw = vowpalwabbit.Workspace(mab_options, quiet=True)
        else:
            self.vw = vowpalwabbit.Workspace(
                f"{mab_options} -i {config.model_path}", quiet=True
            )
        logger.info("Routing initialized")

    def predict(
        self,
        router_library: List[AvailableRouter],
        llm_library: List[AvailableModel],
        context,
        prompt: str,
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
        shared_features = f"shared |Task {context_string}"
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

            # action_context += f" pending_requests={action.pending_requests}"
            action_context = f"pending_requests={action.pending_requests}"
            # action_context = ""

            action_strings.append(
                f"{training_string}|Model {id_name}={action.id} {action_context}"
            )
        return "\n".join(action_strings)


class RouterBenchAlgorithm(RoutingAlgorithm):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.pre_train_done = False

        self.train_data = pd.read_pickle(config.training_data_path)
        models = []
        for c in self.train_data.columns:
            if "|output_quality" in c:
                models.append(f"{c.split('|')[0]}|output_quality")
        self.models_to_route = list(set(models))

        self.embedding_model = config.embedding_model

        self.encoding_model = SentenceTransformer(self.embedding_model)
        embeddings = self.encoding_model.encode(self.train_data["prompt"].tolist())
        self.train_data["embeddings"] = list(embeddings)

        self.embeddings_to_fit = self.train_data["embeddings"]

        model_names = [f"{model.split('|')[0]}" for model in self.models_to_route]

        resp_max = (
            self.train_data[[f"{model}|response_time" for model in model_names]]
            .max()
            .max()
        )
        resp_min = (
            self.train_data[[f"{model}|response_time" for model in model_names]]
            .min()
            .min()
        )

        energy_max = (
            self.train_data[[f"{model}|energy" for model in model_names]].max().max()
        )
        energy_min = (
            self.train_data[[f"{model}|energy" for model in model_names]].min().min()
        )

        self.model_stats = {}
        for model in model_names:
            self.model_stats[f"{model}|output_quality"] = {
                "response_time": (
                    self.train_data[f"{model}|response_time"].mean() - resp_min
                )
                / (resp_max - resp_min),
                "energy": (self.train_data[f"{model}|energy"].mean() - energy_min)
                / (energy_max - energy_min),
            }

        asyncio.create_task(self.pre_train_async())

    def model_ready(self):
        return self.pre_train_done

    def get_prediction_probability(
        self,
        router_library: List[AvailableRouter],
        llm_library: List[AvailableModel],
        predicted_id: str,
    ):
        model_order = {}
        for idx, llm in enumerate(llm_library):
            model_order[llm.id] = idx

        for idx, router in enumerate(router_library):
            model_order[router.id] = len(llm_library) + idx

        probability = [0.0 for _ in range(len(llm_library) + len(router_library))]
        probability[model_order[predicted_id]] = 1.0

        return probability

    async def pre_train_async(self):
        await asyncio.to_thread(self.pre_train)

    @abstractmethod
    def pre_train(self):
        pass


class KNNRouterBench(RouterBenchAlgorithm):
    def __init__(
        self,
        config: RoutingConfig,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)

    def pre_train(self):
        try:
            from sklearn.neighbors import NearestNeighbors

            self.n_neighbors = self.config.knn.n_neighbors
            self.distance_metric = self.config.knn.distance_metric
            self.leaf_size = self.config.knn.leaf_size

            self.knn = NearestNeighbors(
                n_neighbors=self.n_neighbors,
                metric=self.distance_metric,
                leaf_size=self.leaf_size,
                n_jobs=-1,
            ).fit(np.vstack(self.embeddings_to_fit))
        finally:
            self.pre_train_done = True

    def predict(
        self,
        router_library: List[AvailableRouter],
        llm_library: List[AvailableModel],
        context,
        prompt: str,
    ):
        prompt_embeddings = self.encoding_model.encode([prompt])
        if len(prompt_embeddings.shape) == 1:
            prompt_embeddings = np.vstack(prompt_embeddings)

        performance_scores = self.calc_performance_scores(prompt_embeddings)

        weight = 1.0
        for model_name in self.models_to_route:
            performance_scores[model_name] = performance_scores[
                model_name
            ] * weight - 0.5 * (
                self.model_stats[model_name]["response_time"]
                + self.model_stats[model_name]["energy"]
            )
        models_to_route_to = performance_scores.idxmax(axis=1)
        selected_model = models_to_route_to.values[0]
        selected_model = selected_model.split("|")[0]

        return self.get_prediction_probability(
            router_library=router_library,
            llm_library=llm_library,
            predicted_id=selected_model,
        )

    def learn(
        self, router_library, llm_library, context, selected_llm, selection_prob, cost
    ):
        return super().learn(
            router_library, llm_library, context, selected_llm, selection_prob, cost
        )

    def save_model(self):
        return super().save_model()

    def reset_model(self):
        return super().reset_model()

    def calc_performance_scores(self, prompt_embeddings: np.ndarray) -> pd.DataFrame:
        """

        :param prompt_embeddings: first dim is batch size, second dim is embedding size.
        :param kwargs:
        :return:
        """
        _, indices = self.knn.kneighbors(prompt_embeddings)

        indices_df = pd.DataFrame(indices)
        top_k_rows = self.train_data.iloc[indices_df.values.flatten()]
        top_k_rows = top_k_rows.assign(
            group=np.repeat(indices_df.index, indices_df.shape[1])
        )
        performance_scores = top_k_rows.groupby("group")[self.models_to_route].mean()
        return performance_scores


class MLPRouterBench(RouterBenchAlgorithm):
    def __init__(
        self,
        config: RoutingConfig,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)

    def pre_train(self):
        try:
            from sklearn.neural_network import MLPRegressor

            self.mlp = MLPRegressor(
                hidden_layer_sizes=self.config.mlp.hidden_layer_sizes,
                max_iter=200,
                random_state=1234,
                activation=self.config.mlp.activation_function,
                learning_rate=self.config.mlp.learning_rate_method,
                learning_rate_init=self.config.mlp.learning_rate,
                verbose=False,
            )

            self.mlps = {}

            for model in tqdm(self.models_to_route):
                self.mlps[model] = MLPRegressor(
                    hidden_layer_sizes=self.config.mlp.hidden_layer_sizes,
                    max_iter=200,
                    random_state=1234,
                    activation=self.config.mlp.activation_function,
                    learning_rate=self.config.mlp.learning_rate_method,
                    learning_rate_init=self.config.mlp.learning_rate,
                    verbose=False,
                )
                non_nan_idxs = self.train_data[model].notna()
                input_embeddings = np.vstack(self.embeddings_to_fit)[non_nan_idxs]
                performance_values = self.train_data[model].values[non_nan_idxs]
                self.mlps[model].fit(
                    input_embeddings,
                    performance_values,
                )
        finally:
            self.pre_train_done = True

    def predict(self, router_library, llm_library, context, prompt):
        prompt_embeddings = self.encoding_model.encode([prompt])
        if len(prompt_embeddings.shape) == 1:
            prompt_embeddings = np.vstack(prompt_embeddings)

        performance_scores: dict = self.return_optimal_model(prompt_embeddings)

        weight = 1.0
        for model_name in self.models_to_route:
            performance_scores[model_name] = performance_scores[
                model_name
            ] * weight - 0.5 * (
                self.model_stats[model_name]["response_time"]
                + self.model_stats[model_name]["energy"]
            )

        best_model = pd.DataFrame(performance_scores).idxmax(axis=1)
        selected_model = best_model.values[0]
        selected_model = selected_model.split("|")[0]

        return self.get_prediction_probability(
            router_library=router_library,
            llm_library=llm_library,
            predicted_id=selected_model,
        )

    def learn(
        self, router_library, llm_library, context, selected_llm, selection_prob, cost
    ):
        return super().learn(
            router_library, llm_library, context, selected_llm, selection_prob, cost
        )

    def save_model(self):
        return super().save_model()

    def reset_model(self):
        return super().reset_model()

    def get_model_to_route_to(self, input_embedding):
        model_scores = {}
        for model in self.mlps:
            model_scores[model] = self.mlps[model].predict(input_embedding)
        return model_scores

    def return_optimal_model(self, input_embedding) -> dict:
        models_to_route_to = self.get_model_to_route_to(input_embedding)
        return models_to_route_to


ROUTING_ALGORITHMS = {
    "heuristic": Heuristic,
    "mab": Routing,
    "knn": KNNRouterBench,
    "mlp": MLPRouterBench,
}


def get_algorithm(config: RoutingConfig) -> RoutingAlgorithm:
    try:
        return ROUTING_ALGORITHMS[config.algorithm](config)
    except KeyError as e:
        logger.exception(
            f"Unknown routing algorithm {config.algorithm}. Selecting default 'mab'"
        )
        return ROUTING_ALGORITHMS["mab"](config)
