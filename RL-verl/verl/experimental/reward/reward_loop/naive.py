# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import numpy as np

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.utils.reward_score import default_compute_score


@register("naive")
class NaiveRewardLoopManager(RewardLoopManagerBase):
    """The reward manager."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        
        # Only set in extra_info if needed by reward function, to avoid duplication in non_tensor_batch
        num_turns_value = data_item.non_tensor_batch.get("__num_turns__", None)
        if num_turns_value is not None:
            # Extract scalar from numpy array if needed
            if isinstance(num_turns_value, np.ndarray):
                num_turns_value = num_turns_value.item() if num_turns_value.size == 1 else num_turns_value
            extra_info["num_turns"] = num_turns_value
        
        # Handle rollout_reward_scores if present
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        if rollout_reward_scores:
            extra_info["rollout_reward_scores"] = rollout_reward_scores
        
        pred_mask_value = data_item.non_tensor_batch["pred_mask"]

        if isinstance(pred_mask_value, np.ndarray) and pred_mask_value.dtype == object:
            if pred_mask_value.ndim == 1:
                # 1D object array: [mask1, mask2, ...] - this is the expected format
                pred_mask_value = list(pred_mask_value)
            elif pred_mask_value.ndim == 2:
                # 2D object array (H, W): This is a SINGLE mask incorrectly wrapped
                try:
                    first_elem = pred_mask_value.flat[0]
                    # Treat the whole thing as a single mask
                    pred_mask_value = [pred_mask_value]
                except Exception as e:
                    pred_mask_value = [pred_mask_value]
            elif pred_mask_value.ndim == 3:
                # 3D object array (N, H, W): Multiple masks
                pred_mask_value = [pred_mask_value[turn_idx] for turn_idx in range(pred_mask_value.shape[0])]
            else:
                pred_mask_value = [pred_mask_value]

        extra_info["pred_mask"] = pred_mask_value

        # Handle stopped field
        if "stopped" in data_item.non_tensor_batch:
            stopped_value = data_item.non_tensor_batch["stopped"]
            # Extract the first element if it's a single-element array
            if isinstance(stopped_value, np.ndarray) and stopped_value.size == 1:
                extra_info["stopped"] = stopped_value.item()
            else:
                extra_info["stopped"] = stopped_value
            
        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )
        
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                reward_router_address=self.reward_router_address,
                reward_model_tokenizer=self.reward_model_tokenizer,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    reward_router_address=self.reward_router_address,
                    reward_model_tokenizer=self.reward_model_tokenizer,
                ),
            )

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        reward = score

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
