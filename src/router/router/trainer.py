import ast
import asyncio
import json
import logging
import math
import multiprocessing
import random
import re
import time
import traceback

import evaluate
import yaml

from .config import TrainerConfig
from .database import DuckDB, TrainingSample
from .metrics import get_normalizer
from .routing import Routing

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        duckdb: DuckDB,
        routing: Routing,
        config: TrainerConfig,
    ):
        self.duckdb = duckdb
        self.routing = routing
        self.config = config
        self.training_interval_threshold = config.training_interval_threshold_seconds
        self.batch_size = config.training_batch_size
        self.time_since_last_training = time.perf_counter()

        self.squad_metric = evaluate.load("squad")
        self.rouge = evaluate.load("rouge")
        self.accuracy = evaluate.load("accuracy")
        self.f1 = evaluate.load("f1")
        self.exact_match = evaluate.load("exact_match")

        self.normalizer = get_normalizer(config.normalizer)

        self.output_quality_weight = config.reward_output_quality_weight
        self.response_time_weight = config.reward_response_time_weight
        self.energy_consumption_weight = config.reward_energy_consumption_weight

        self.metrics_map = {
            "squad": "f1",
            "natural_questions": "rougeLsum",
            "mbpp": "pass_at_1",
            "gsm8k": "accuracy",
            "glue_sst2": "accuracy",
            "glue_qqp": "accuracy",
            "glue_mnli": "accuracy",
            "cnn_dailymail": "rougeLsum",
            "xsum": "rougeLsum",
            "bbh": {
                "boolean_expressions": "exact_match",
                "object_counting": "accuracy",
                "hyperbaton": "accuracy",
                "logical_deduction_seven_objects": "accuracy",
                "temporal_sequences": "accuracy",
                "tracking_shuffled_objects_three_objects": "accuracy",
                "formal_fallacies": "exact_match",
                "multistep_arithmetic_two": "accuracy",
                "word_sorting": "exact_match",
            },
        }
        try:
            with open(config.bounds_file_path, "r") as f:
                self.bounds = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error("Bounds file not found")
            self.bounds = {}

        logger.info("Trainer initialized")

    def normalize_metric(self, value, metric_name, dataset, subset=None):
        if not isinstance(value, float):
            logger.error(f"Cannot normalize. Value {value} has to be of type float.")
            return value

        try:
            if dataset != "bbh":
                ds = self.bounds[dataset]
                output_quality_metric = self.metrics_map[dataset]
            else:
                ds = self.bounds[dataset][subset]
                output_quality_metric = self.metrics_map[dataset][subset]

            if metric_name == "output_quality":
                bounds = ds[output_quality_metric]
            else:
                bounds = ds[metric_name]
        except:
            logger.error(
                f"bounds not found for dataset {dataset} with subset {subset} and metric {metric_name}"
            )
            return value

        return self.normalizer.normalize(bounds, value)

    def compute_reward(
        self, normalized_output_quality, response_time, energy_consumption, failed
    ):
        if failed:
            return -1.0
        time_component = 1.0 / (1 + math.exp((response_time - 10.0) / 3))

        return self.output_quality_weight * (normalized_output_quality)
        return (self.output_quality_weight * normalized_output_quality) * (
            self.response_time_weight * time_component
        )
        # - self.energy_consumption_weight * energy_consumption

    def _compute_dummy_output_quality(self, training_sample: TrainingSample):
        """
        Only for local deployment: Randomly draws output quality
        from mean and standard deviation extracted from LLMs response
        """

        response = training_sample.response

        performance = json.loads(response)
        mu = performance[training_sample.task]["mu"]
        std = performance[training_sample.task]["std"]

        output_quality = random.gauss(mu, std)
        if output_quality > 1.0:
            output_quality = 1.0
        if output_quality < 0.0:
            output_quality = 0.0

        return output_quality, output_quality

    def _compute_output_quality(self, training_sample: TrainingSample):
        if training_sample.failed():
            return 0.0, 0.0
        if self.config.dummy_output_quality:
            return self._compute_dummy_output_quality(training_sample)

        response = training_sample.response
        target = training_sample.target
        dataset = training_sample.dataset
        subset = training_sample.subset

        logger.info(f"TARGET: {target}")

        match dataset:
            case "squad":
                output_quality = self._compute_squad(response, json.loads(target))
                pass
            case "cnn_dailymail":
                output_quality = self._compute_cnn_dailymail(response, target)
                pass
            case "xsum":
                output_quality = self._compute_xsum(response, target)
                pass
            case "natural_questions":
                output_quality = self._compute_natural_questions(response, target)
                pass
            case "glue_mnli":
                output_quality = self._compute_glue_mnli(response, target)
                pass
            case "glue_qqp":
                output_quality = self._compute_glue_qqp(response, target)
                pass
            case "glue_sst2":
                output_quality = self._compute_glue_sst2(response, target)
                pass
            case "gsm8k":
                output_quality = self._compute_gsm8k(response, target)
                pass
            case "bbh":
                output_quality = self._compute_bbh(response, target, subset)
                pass
            case "mbpp":
                output_quality = self._compute_mbpp(response, target)
                pass
            case _:
                output_quality = 0.0

        normalized_output_quality = self.normalize_metric(
            output_quality, "output_quality", dataset, subset
        )
        return normalized_output_quality, output_quality

    def compute_cost(self, training_sample: TrainingSample):
        dataset = training_sample.dataset
        subset = training_sample.subset

        normalized_output_quality, output_quality = self._compute_output_quality(
            training_sample
        )

        normalized_response_time = self.normalize_metric(
            training_sample.response_time, "response_time", dataset, subset
        )

        normalized_energy = 1.0
        if not training_sample.failed():
            normalized_energy = self.normalize_metric(
                training_sample.energy, "energy", dataset, subset
            )

        reward = self.compute_reward(
            output_quality,
            training_sample.response_time,
            normalized_energy,
            training_sample.failed(),
        )

        return -reward, normalized_output_quality, output_quality

    def _do_train(self):
        enough_unprocessed_metrics = (
            self.duckdb.metrics_since_last_training_batch >= self.batch_size
        )
        enough_time_since_last_training = (
            time.perf_counter() - self.time_since_last_training
        ) > self.training_interval_threshold
        at_least_one_metric = self.duckdb.metrics_since_last_training_batch > 0
        return enough_unprocessed_metrics or (
            enough_time_since_last_training and at_least_one_metric
        )

    async def run(self):
        while True:
            if self._do_train():
                batch = self.duckdb.get_training_batch()
                logger.info(f"Training with batch of size {len(batch)}")

                for sample in batch:
                    context = {"task": sample.task}
                    logger.info(f"Available models: {sample.available_models}")
                    logger.info(f"Available routers: {sample.available_routers}")

                    selected_llm = sample.llm_id
                    cost, normalized_output_quality, output_quality = self.compute_cost(
                        sample
                    )
                    sample.reward = -cost
                    sample.normalized_output_quality = normalized_output_quality
                    sample.output_quality = output_quality
                    self.routing.learn(
                        sample.available_routers,
                        sample.available_models,
                        context,
                        selected_llm,
                        sample.routing_confidence,
                        cost,
                    )

                self.time_since_last_training = time.perf_counter()
                self.duckdb.set_processed(batch)

            await asyncio.sleep(1)

    def _compute_squad(self, response, label):
        predictions = [{"id": "0", "prediction_text": response}]
        references = [{"id": "0", "answers": label}]

        res = self.squad_metric.compute(predictions=predictions, references=references)
        return res["f1"] / 100.0

    def _compute_natural_questions(self, response, label):
        res = self.rouge.compute(predictions=[response], references=[label])
        return res[self.metrics_map["natural_questions"]]

    def _compute_cnn_dailymail(self, response, label):
        res = self.rouge.compute(predictions=[response], references=[label])
        return res[self.metrics_map["cnn_dailymail"]]

    def _compute_xsum(self, response, label):
        res = self.rouge.compute(predictions=[response], references=[label])
        return res[self.metrics_map["xsum"]]

    def _compute_glue_mnli(self, response, label):
        return self._compute_accuracy(response, label)

    def _compute_glue_qqp(self, response, label):
        return self._compute_accuracy(response, label)

    def _compute_glue_sst2(self, response, label):
        return self._compute_accuracy(response, label)

    def _compute_gsm8k(self, response, label):
        prediction = self.extract_last_number(response)
        reference = self.extract_gsm8k_answer(label)

        if prediction is None or reference is None:
            return 0.0

        return self._compute_accuracy(prediction, reference, float)

    def _compute_bbh(self, response, label, subset):
        metric = self.accuracy
        metric_name = "accuracy"

        references = [label]
        if subset == "object_counting" or subset == "multistep_arithmetic_two":
            last_number = self.extract_last_number(response)
            if last_number is None:
                preds = [last_number]
            else:
                preds = [float(self.extract_last_number(response))]
        elif (
            subset == "hyperbaton"
            or subset == "logical_deduction_seven_objects"
            or subset == "temporal_sequences"
            or subset == "tracking_shuffled_objects_three_objects"
        ):
            preds = [ord(self.extract_last_choice(response))]
            references = [ord(self.extract_last_choice(label))]
        elif subset == "formal_fallacies":
            preds = [self.extract_last_validity(response)]
            metric = self.exact_match
            metric_name = "exact_match"
        elif subset == "boolean_expressions":
            preds = [self.extract_last_boolean(response)]
            references = [label.lower()]
            metric = self.exact_match
            metric_name = "exact_match"
        elif subset == "word_sorting":
            preds = [self.extract_word_list(response)]
            references = [label.lower()]
            metric = self.exact_match
            metric_name = "exact_match"

        if None in preds:
            return 0.0

        res = metric.compute(predictions=preds, references=references)
        return res[metric_name]

    def _compute_mbpp(self, response, label):
        code, function_name = self.extract_python_code(response)

        label_converted = json.loads(label)

        res = self.run_code_with_tests(code, function_name, label_converted)
        return self.pass_at_k([e["passed"] for e in res])

    def _compute_accuracy(self, response, label, casting_class=int):
        try:
            response = casting_class(float(response))
        except:
            logger.warning(f"Response '{response}' is not numeric")
            return 0.0

        res = self.accuracy.compute(predictions=[response], references=[label])
        return res["accuracy"]

    def pass_at_k(self, results, k=1):
        return sum(results[:k]) / k

    def extract_gsm8k_answer(self, text):
        match = re.search(r"####\s*(-?\d+)", text)
        if match:
            return match.group(1)
        return None

    def extract_last_number(self, text):
        numbers = re.findall(r"-?\d+\.?\d*", text)
        return numbers[-1] if numbers else None

    def extract_last_choice(self, text):
        """
        Returns the last occurrence of (X) or X) where X is A-Z.
        Returns just the letter (e.g., 'A'), or 'Z' if not found.
        """
        pattern = r"(?:\(([A-Z])\)|([A-Z])\))"

        matches = re.findall(pattern, text)

        if not matches:
            return "Z"

        # Each match is a tuple like ('A', '') or ('', 'A')
        last_match = matches[-1]

        # Extract the non-empty group
        letter = last_match[0] if last_match[0] else last_match[1]

        return letter

    def extract_last_validity(self, text):
        """
        Returns the last occurrence of 'valid' or 'invalid' (case-insensitive).
        Returns 'valid', 'invalid', or None if not found.
        """
        pattern = r"\b(valid|invalid)\b"

        return self._extract_last_word(text, pattern)

    def extract_last_boolean(self, text):
        """
        Returns the last occurrence of 'true' or 'false' (case-insensitive).
        Returns 'true', 'false', or None if not found.
        """
        pattern = r"\b(true|false)\b"

        return self._extract_last_word(text, pattern)

    def _extract_last_word(self, text, pattern):
        """
        Returns the last occurrence of the pattern given (case-insensitive).
        Returns the match, or None if not found.
        """

        matches = re.findall(pattern, text, flags=re.IGNORECASE)

        if not matches:
            return None

        return matches[-1].lower()

    def extract_word_list(self, text):
        """
        Extracts a list of words from messy text.
        Handles:
        - enumerations (1., A), - bullets)
        - comma-separated lists
        - leading/trailing sentences
        - quotes and punctuation
        """

        text = text.strip()
        text = text.replace(",", "")
        return text.lower()

    def extract_python_code(self, text):
        """
        Extracts the first Python code block and the first function name.

        Returns:
            (code: str | None, function_name: str | None)
        """
        # --- Step 1: extract code block ---
        pattern = r"\`\`\`python\s*(.*?)\`\`\`"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

        if not match:
            return None, None

        code = match.group(1).strip()

        # --- Step 2: parse function name using AST ---
        try:
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    return code, node.name
        except Exception:
            pass  # fall through if parsing fails

        return code, None

    def _replace_function_name_in_tests(self, tests, new_name):
        """
        Replaces the function name in assert statements with a new name.

        Args:
            tests (list of str): list of assert statements
            new_name (str): new function name

        Returns:
            list of str: updated test statements
        """

        class FuncNameReplacer(ast.NodeTransformer):
            def visit_Call(self, node):
                # If it's a simple function call: f(...)
                if isinstance(node.func, ast.Name):
                    node.func.id = new_name
                return self.generic_visit(node)

        updated_tests = []

        for test in tests:
            try:
                tree = ast.parse(test)
                tree = FuncNameReplacer().visit(tree)
                ast.fix_missing_locations(tree)

                updated_test = ast.unparse(tree)
                updated_tests.append(updated_test)

            except Exception:
                # fallback: leave unchanged if parsing fails
                updated_tests.append(test)

        print(tests)
        print(updated_tests)
        return updated_tests

    def run_code_with_tests(self, code, function_name, tests: list, timeout=3):
        tests = self._replace_function_name_in_tests(tests, function_name)

        def target(queue):
            results = []

            try:
                # --- sandbox setup ---
                ALLOWED_MODULES = {"math", "itertools", "collections", "re"}

                def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
                    if name in ALLOWED_MODULES:
                        return __import__(name, globals, locals, fromlist, level)
                    raise ImportError(f"Module {name} not allowed")

                safe_builtins = {
                    "range": range,
                    "len": len,
                    "print": print,
                    "int": int,
                    "float": float,
                    "str": str,
                    "list": list,
                    "dict": dict,
                    "set": set,
                    "abs": abs,
                    "min": min,
                    "max": max,
                    "sum": sum,
                    "any": any,
                    "all": all,
                    "sorted": sorted,
                    "enumerate": enumerate,
                    "zip": zip,
                    "__import__": safe_import,
                }

                safe_globals = {"__builtins__": safe_builtins}

                # --- execute code once ---
                exec(code, safe_globals, safe_globals)

                # --- run tests individually ---
                for test in tests:
                    try:
                        exec(test, safe_globals, safe_globals)
                        results.append({"passed": True, "error": None})
                    except Exception:
                        results.append(
                            {"passed": False, "error": traceback.format_exc()}
                        )

                queue.put(results)

            except Exception:
                # code itself failed → all tests fail
                err = traceback.format_exc()
                queue.put([{"passed": False, "error": err} for _ in tests])

        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=target, args=(queue,))
        p.start()
        p.join(timeout)

        if p.is_alive():
            p.terminate()
            return [{"passed": False, "error": "Timeout"} for _ in tests]

        return (
            queue.get()
            if not queue.empty()
            else [{"passed": False, "error": "No result"} for _ in tests]
        )
