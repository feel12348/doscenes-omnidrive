# Copyright (c) 2024-2025, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia License.
# To view a copy of this license, visit
# https://github.com/NVlabs/OmniDrive/blob/main/LICENSE
#
# SPDX-License-Identifier: Apache-2.0
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from .llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from mmcv.runner import auto_fp16
class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.hidden_size = config.hidden_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.number_projector = nn.Sequential(
            nn.Linear(1, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.number_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1),
        )
        self.pretraining_tp = config.pretraining_tp

        number_tokens = [
                718,
                448,
                29900,
                29889,
                29896,
                29906,
                29941,
                29946,
                29945,
                29953,
                29955,
                29947,
                29929,
            ]  # +-0.123456789
        weighted_mask = torch.ones(self.config.vocab_size)
        weighted_mask[number_tokens] = 3.0
        self.register_buffer("weighted_mask", weighted_mask)
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def _drivecode_numbers_enabled(self):
        return bool(getattr(
            self, 'drivecode_enabled',
            getattr(self.config, 'drivecode_enabled', False)))

    def _resize_embedding_layer(self, old_embedding, vocab_size):
        old_vocab_size, embedding_dim = old_embedding.weight.shape
        if vocab_size <= old_vocab_size:
            return old_embedding
        new_embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=old_embedding.padding_idx,
            device=old_embedding.weight.device,
            dtype=old_embedding.weight.dtype)
        new_embedding.weight.data[:old_vocab_size] = old_embedding.weight.data
        new_embedding.weight.data[old_vocab_size:] = old_embedding.weight.data.mean(
            dim=0, keepdim=True)
        return new_embedding

    def _resize_lm_head(self, old_head, vocab_size):
        old_vocab_size, hidden_size = old_head.weight.shape
        if vocab_size <= old_vocab_size:
            return old_head
        new_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=old_head.bias is not None,
            device=old_head.weight.device,
            dtype=old_head.weight.dtype)
        new_head.weight.data[:old_vocab_size] = old_head.weight.data
        new_head.weight.data[old_vocab_size:] = old_head.weight.data.mean(
            dim=0, keepdim=True)
        if old_head.bias is not None:
            new_head.bias.data[:old_vocab_size] = old_head.bias.data
            new_head.bias.data[old_vocab_size:] = old_head.bias.data.mean()
        return new_head

    def _ensure_input_id_vocab_size(self, input_ids):
        if input_ids is None:
            return
        valid_input_ids = input_ids[input_ids >= 0]
        if valid_input_ids.numel() == 0:
            return
        required_vocab_size = int(valid_input_ids.max().item()) + 1
        if required_vocab_size <= self.get_input_embeddings().num_embeddings:
            return
        self.resize_token_embeddings(required_vocab_size)
        if required_vocab_size > self.get_input_embeddings().num_embeddings:
            self.model.embed_tokens = self._resize_embedding_layer(
                self.model.embed_tokens, required_vocab_size)
        if required_vocab_size > self.lm_head.out_features:
            self.lm_head = self._resize_lm_head(self.lm_head, required_vocab_size)
        self.config.vocab_size = max(self.config.vocab_size, required_vocab_size)
        self.vocab_size = self.config.vocab_size

    def _check_input_id_range(self, input_ids):
        if input_ids is None:
            return
        valid_input_ids = input_ids[input_ids >= 0]
        if valid_input_ids.numel() == 0:
            return
        max_token_id = int(valid_input_ids.max().item())
        vocab_size = self.get_input_embeddings().num_embeddings
        if max_token_id >= vocab_size:
            raise ValueError(
                f"input_ids contain token id {max_token_id}, but the LLM "
                f"embedding table has size {vocab_size}. This usually means "
                "the tokenizer was extended without resizing the model token "
                "embeddings.")

    def _get_ce_weight(self, device):
        if self.weighted_mask.numel() == self.config.vocab_size:
            return self.weighted_mask.float().to(device)
        weight = torch.ones(self.config.vocab_size, device=device)
        num_copy = min(self.weighted_mask.numel(), self.config.vocab_size)
        weight[:num_copy] = self.weighted_mask[:num_copy].float().to(device)
        return weight

    def _number_regression_loss(self, hidden_states, labels, number_values, number_token_mask=None):
        if not self._drivecode_numbers_enabled():
            return None, None
        number_token_id = getattr(
            self, 'number_token_id',
            getattr(self.config, 'number_token_id', None))
        if number_token_id is None or number_values is None:
            return None, None

        preds = []
        targets = []
        num_matched = 0
        number_values = number_values.to(hidden_states.device).float()
        batch_size = number_token_mask.shape[0] if number_token_mask is not None else labels.shape[0]
        for batch_idx in range(batch_size):
            if number_token_mask is not None:
                label_positions = torch.where(number_token_mask[batch_idx].to(hidden_states.device))[0]
            elif labels is not None:
                label_positions = torch.where(labels[batch_idx] == number_token_id)[0]
            else:
                label_positions = hidden_states.new_empty((0,), dtype=torch.long)
            label_positions = label_positions[label_positions > 0]
            cur_targets = number_values[batch_idx]
            cur_targets = cur_targets[torch.isfinite(cur_targets)]
            num_to_use = min(label_positions.numel(), cur_targets.numel())
            if num_to_use == 0:
                continue
            num_matched += num_to_use
            cur_hidden = hidden_states[batch_idx, label_positions[:num_to_use] - 1]
            head_dtype = next(self.number_head.parameters()).dtype
            cur_preds = self.number_head(cur_hidden.to(head_dtype)).squeeze(-1).float()
            cur_preds = torch.nan_to_num(cur_preds, nan=0.0, posinf=1e4, neginf=-1e4)
            preds.append(cur_preds)
            targets.append(cur_targets[:num_to_use].float())
        if len(preds) == 0:
            return None, hidden_states.new_tensor(0.0)

        pred_values = torch.cat(preds).float()
        target_values = torch.cat(targets).float()
        finite_mask = torch.isfinite(pred_values) & torch.isfinite(target_values)
        pred_values = pred_values[finite_mask]
        target_values = target_values[finite_mask]
        if pred_values.numel() == 0:
            return None, hidden_states.new_tensor(0.0)
        count = hidden_states.new_tensor(float(num_matched))
        if pred_values.numel() >= 2 and pred_values.numel() % 2 == 0:
            pred_points = pred_values.view(-1, 2)
            target_points = target_values.view(-1, 2)
            number_loss = torch.linalg.vector_norm(pred_points - target_points, dim=-1).mean()
        else:
            number_loss = F.l1_loss(pred_values, target_values)
        number_loss = torch.nan_to_num(number_loss, nan=0.0, posinf=1e4, neginf=1e4)
        return number_loss.to(hidden_states.dtype), count

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        number_values: Optional[torch.FloatTensor] = None,
        drivecode_enabled: Optional[bool] = None,
        number_token_id: Optional[int] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if drivecode_enabled is not None:
            self.drivecode_enabled = bool(drivecode_enabled)
            self.config.drivecode_enabled = bool(drivecode_enabled)
        drivecode_active = self._drivecode_numbers_enabled()
        if drivecode_active and number_token_id is not None:
            self.number_token_id = number_token_id
            self.config.number_token_id = number_token_id
        if not drivecode_active:
            number_values = None
            self.last_number_token_mask = None

        if inputs_embeds is None:
            self._ensure_input_id_vocab_size(input_ids)
            self._check_input_id_range(input_ids)
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                number_values=number_values if drivecode_active else None
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        ce_loss = None
        number_loss = None
        number_count = None
        self.last_ce_loss = None
        self.last_number_loss = None
        self.last_number_count = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = torch.nan_to_num(
                shift_logits, nan=0.0, posinf=1e4, neginf=-1e4)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(weight=self._get_ce_weight(shift_logits.device))
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            valid_label_mask = shift_labels.ne(-100)
            if valid_label_mask.any():
                ce_loss = loss_fct(shift_logits, shift_labels)
            else:
                ce_loss = shift_logits.sum() * 0.0
            ce_loss = torch.nan_to_num(ce_loss, nan=0.0, posinf=1e4, neginf=-1e4)
            loss = ce_loss
            number_token_mask = (
                getattr(self, 'last_number_token_mask', None)
                if drivecode_active else None)
            if drivecode_active and os.environ.get('DRIVECODE_DEBUG', '0') == '1':
                debug_count = getattr(self, '_drivecode_debug_count', 0)
                if debug_count < 5:
                    number_token_id = getattr(
                        self, 'number_token_id',
                        getattr(self.config, 'number_token_id', None))
                    mask_count = (
                        int(number_token_mask.sum().item())
                        if number_token_mask is not None else -1)
                    label_count = (
                        int((labels == number_token_id).sum().item())
                        if labels is not None and number_token_id is not None else -1)
                    value_count = (
                        int((~torch.isnan(number_values)).sum().item())
                        if number_values is not None else -1)
                    print(
                        f"[DriveCodeDebug][lm] enabled={getattr(self, 'drivecode_enabled', getattr(self.config, 'drivecode_enabled', None))} "
                        f"number_token_id={number_token_id} "
                        f"mask_count={mask_count} label_count={label_count} "
                        f"value_count={value_count} labels_shape={tuple(labels.shape) if labels is not None else None}",
                        flush=True)
                    self._drivecode_debug_count = debug_count + 1
            if drivecode_active:
                number_loss, number_count = self._number_regression_loss(
                    hidden_states, labels, number_values, number_token_mask)
                if number_loss is not None:
                    number_loss = torch.nan_to_num(number_loss, nan=0.0, posinf=1e4, neginf=1e4)
                    loss = loss + number_loss.to(loss.dtype)
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
            self.last_ce_loss = ce_loss.detach() if ce_loss is not None else None
            self.last_number_loss = number_loss.detach() if number_loss is not None else None
            self.last_number_count = number_count.detach() if number_count is not None else None

        if not return_dict:
            output = (logits,) + outputs[1:]
            if loss is not None:
                ce_log = ce_loss.detach() if ce_loss is not None else loss.detach()
                ce_log = torch.nan_to_num(ce_log, nan=0.0, posinf=1e4, neginf=-1e4)
                number_log = number_loss.detach() if number_loss is not None else loss.detach().new_tensor(0.0)
                count_log = number_count.detach() if number_count is not None else loss.detach().new_tensor(0.0)
                return (loss,) + output + (ce_log, number_log, count_log)
            return output

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        output.ce_loss = (
            torch.nan_to_num(ce_loss.detach(), nan=0.0, posinf=1e4, neginf=-1e4)
            if ce_loss is not None else None)
        output.number_loss = number_loss.detach() if number_loss is not None else None
        output.number_count = number_count.detach() if number_count is not None else None
        return output
       

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    @torch.no_grad()
    def generate_with_numbers(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        max_new_tokens: int = 320,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        number_token_id = getattr(
            self, 'number_token_id',
            getattr(self.config, 'number_token_id', None))
        if number_token_id is None:
            output_ids = self.generate(
                inputs=inputs,
                images=images,
                image_sizes=image_sizes,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                **kwargs)
            return output_ids, [[] for _ in range(output_ids.shape[0])]

        if images is not None:
            (
                _,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            if attention_mask is None:
                attention_mask = torch.ones(
                    inputs.shape, dtype=torch.long, device=inputs.device)

        batch_size = inputs_embeds.shape[0]
        generated_ids = []
        generated_numbers = [[] for _ in range(batch_size)]
        past_key_values = None
        cur_embeds = inputs_embeds
        cur_attention_mask = attention_mask

        for _ in range(max_new_tokens):
            outputs = self.model(
                inputs_embeds=cur_embeds,
                attention_mask=cur_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state[:, -1, :]
            logits = self.lm_head(hidden).float()
            next_ids = torch.argmax(logits, dim=-1)
            generated_ids.append(next_ids)

            next_embeds = self.get_model().embed_tokens(next_ids)
            number_mask = next_ids == number_token_id
            if number_mask.any():
                number_values = self.number_head(hidden[number_mask]).squeeze(-1)
                projected_numbers = self.number_projector(
                    number_values.to(next_embeds.dtype).unsqueeze(-1))
                next_embeds[number_mask] = projected_numbers
                number_indices = torch.where(number_mask)[0].tolist()
                for idx, value in zip(number_indices, number_values.detach().float().cpu().tolist()):
                    generated_numbers[idx].append(value)

            past_key_values = outputs.past_key_values
            cur_embeds = next_embeds.unsqueeze(1)
            if cur_attention_mask is not None:
                cur_attention_mask = torch.cat(
                    [cur_attention_mask,
                     torch.ones((batch_size, 1), dtype=cur_attention_mask.dtype,
                                device=cur_attention_mask.device)],
                    dim=1)

            if eos_token_id is not None and torch.all(next_ids == eos_token_id):
                break

        if len(generated_ids) == 0:
            output_ids = inputs.new_zeros((batch_size, 0))
        else:
            output_ids = torch.stack(generated_ids, dim=1)
        return output_ids, generated_numbers

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
