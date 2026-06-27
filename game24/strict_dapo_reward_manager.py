"""DAPO-style strict reward manager for Game24 improved GRPO."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

import torch

from game24.reward_strict import compute_score as strict_compute_score
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


def _decode_nested(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        import json

        try:
            return _decode_nested(json.loads(value))
        except json.JSONDecodeError:
            return value
    if isinstance(value, Mapping):
        return {key: _decode_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_nested(item) for item in value]
    return value


def _canonical_group_key(data_item: Any, ground_truth: Any) -> str:
    uid = data_item.non_tensor_batch.get("uid", None)
    if uid is not None:
        return f"uid:{uid}"

    extra_info = _decode_nested(data_item.non_tensor_batch.get("extra_info", {}))
    if isinstance(extra_info, Mapping) and "numbers" in extra_info:
        numbers = extra_info["numbers"]
        target = extra_info.get("target", 24)
    else:
        truth = _decode_nested(ground_truth)
        if isinstance(truth, Mapping) and "numbers" in truth:
            numbers = truth["numbers"]
            target = truth.get("target", 24)
        else:
            return f"row:{id(data_item)}"

    canonical = tuple(sorted(int(number) for number in numbers))
    return f"numbers:{canonical}:target:{int(target)}"


@register("game24_strict_dapo")
class Game24StrictDAPORewardManager(AbstractRewardManager):
    """Strict 0/1 Game24 reward with per-group DAPO diagnostics.

    verl v0.7.1's DAPO reward manager records dict fields returned by
    ``compute_score``. This manager keeps that behavior and additionally logs
    group-level strict-correctness statistics for rollout groups.
    """

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: Any = None,
        reward_fn_key: str = "data_source",
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or strict_compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources: dict[str, int] = {}
        item_results: list[dict[str, Any]] = []
        groups: dict[str, list[int]] = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if eos_token and response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            if isinstance(extra_info, dict):
                extra_info["rollout_reward_scores"] = rollout_reward_scores

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            if not isinstance(result, dict):
                result = {"score": float(result), "acc": float(result)}

            score = float(result["score"])
            acc = float(result.get("acc", score))
            reward_tensor[i, int(valid_response_length) - 1] = score

            group_key = _canonical_group_key(data_item, ground_truth)
            groups[group_key].append(i)
            item_results.append(
                {
                    "result": result,
                    "acc": acc,
                    "data_source": data_source,
                    "prompt": prompt_str,
                    "response": response_str,
                    "ground_truth": ground_truth,
                    "response_length": int(valid_response_length),
                    "group_key": group_key,
                }
            )

        group_sizes = [len(indices) for indices in groups.values()]
        common_group_size = Counter(group_sizes).most_common(1)[0][0] if group_sizes else 0
        group_stats: dict[str, dict[str, float | int]] = {}
        for group_key, indices in groups.items():
            k_correct = int(sum(item_results[index]["acc"] for index in indices))
            group_size = len(indices)
            mixed = int(0 < k_correct < group_size)
            all_wrong = int(k_correct == 0)
            all_correct = int(k_correct == group_size)
            group_stats[group_key] = {
                "group_size": group_size,
                "k_correct": k_correct,
                "all_wrong": all_wrong,
                "all_correct": all_correct,
                "mixed": mixed,
                "zero_reward_std": int(all_wrong or all_correct),
            }

        for index, item in enumerate(item_results):
            result = item["result"]
            stats = group_stats[item["group_key"]]

            for key, value in result.items():
                if isinstance(value, (int, float, bool)):
                    reward_extra_info[key].append(float(value))
            reward_extra_info["response_length"].append(item["response_length"])
            reward_extra_info["group_size"].append(stats["group_size"])
            reward_extra_info["k_correct"].append(stats["k_correct"])
            reward_extra_info["all_wrong_rate"].append(stats["all_wrong"])
            reward_extra_info["all_correct_rate"].append(stats["all_correct"])
            reward_extra_info["mixed_group_rate"].append(stats["mixed"])
            reward_extra_info["zero_reward_std_rate"].append(stats["zero_reward_std"])
            reward_extra_info["generated_prompt_count"].append(1)
            reward_extra_info["accepted_prompt_count"].append(stats["mixed"])
            reward_extra_info["acceptance_rate"].append(stats["mixed"])
            reward_extra_info["generation_rounds"].append(1)

            for k in range(common_group_size + 1):
                reward_extra_info[f"k_correct_hist_{k}"].append(int(stats["k_correct"] == k))

            data_source = item["data_source"]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", item["prompt"])
                print("[response]", item["response"])
                print("[ground_truth]", item["ground_truth"])
                print("[group_key]", item["group_key"])
                print("[k_correct]", stats["k_correct"])
                for key, value in result.items():
                    print(f"[{key}]", value)

        total = len(item_results)
        if total:
            strict_correct = int(sum(float(item["result"].get("acc", 0.0)) for item in item_results))
            format_rate = sum(float(item["result"].get("format_valid", 0.0)) for item in item_results) / total
            parse_rate = sum(float(item["result"].get("parse_valid", 0.0)) for item in item_results) / total
            number_usage_rate = (
                sum(float(item["result"].get("number_usage_valid", 0.0)) for item in item_results) / total
            )
            mean_response_length = sum(float(item["response_length"]) for item in item_results) / total
            mixed_group_rate = (
                sum(float(stats["mixed"]) for stats in group_stats.values()) / len(group_stats) if group_stats else 0.0
            )
            print(
                "[game24_strict_metrics] "
                f"strict_exact={strict_correct}/{total} "
                f"strict_exact_accuracy={strict_correct / total:.6f} "
                f"format_rate={format_rate:.6f} "
                f"parse_rate={parse_rate:.6f} "
                f"number_usage_rate={number_usage_rate:.6f} "
                f"mean_response_length={mean_response_length:.2f} "
                f"mixed_group_rate={mixed_group_rate:.6f}"
            )

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
