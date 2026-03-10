from typing import Dict, List, Optional, Tuple

from . import default_agents
from models import ModelWrapper, _past_length
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout
from experiment_logger import ExperimentLogger
import torch
import argparse
from vllm import SamplingParams
import pdb
import sys

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


class LatentMASMethod:
    """Communication-augmented Latent Multi-Agent System (LatentMASMethod).

    Extends LatentMAS by recording hidden states at ALL transformer layers
    for both the last input token and each latent step of every non-judger agent.

    The collected data is stored in ``self.hidden_states_record`` after each call
    to ``run_batch`` / ``run_batch_vllm``.  It is a list (one entry per agent,
    excluding the judger) of dicts with the following schema::

        {
            "agent_name": str,
            "last_input_token": Tensor[B, num_layers+1, D],
            # hidden states of the last input token at every layer
            # (index 0 = embedding layer, index L = last transformer layer)

            "latent_steps": List[Tensor[B, num_layers+1, D]],
            # hidden states of each latent step token at every layer;
            # list length == self.latent_steps
        }
    """

    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
        logger: Optional[ExperimentLogger] = None,
    ) -> None:
        self.args = args
        self.model = model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.method_name = 'latent_mas'
        self.vllm_device = args.device
        self.HF_device = args.device2
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = bool(getattr(args, "sequential_info_only", False)) if args else False
        self.accumulate_latent = bool(getattr(args, "accumulate_latent", False)) if args else False
        self.logger = logger
        self.save_kv = bool(getattr(args, "save_kv", False)) if args else False

        if self.latent_only:
            self.sequential_info_only = True

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=args.max_new_tokens,
        )
        self.task = args.task

        # ---------------------------------------------------------------------------
        # Storage for hidden states collected during the latest run_batch call.
        # Shape details are in the class docstring.
        # ---------------------------------------------------------------------------
        self.hidden_states_record: List[Dict] = []

        # When True, the judger agent is skipped entirely in run_batch.
        # AETrainer sets this flag to speed up data collection (no need for
        # the judger's text output during hidden-state collection).
        self.skip_judger: bool = False

    # ------------------------------------------------------------------
    # Helper utilities (identical to LatentMASMethod)
    # ------------------------------------------------------------------

    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()

    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        if past_kv is None or tokens_to_keep <= 0:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            legacy = past_kv.to_legacy_cache()
            trimmed_legacy = tuple(
                tuple(self._slice_tensor(t, tokens_to_keep) for t in layer)
                for layer in legacy
            )
            return past_kv.__class__.from_legacy_cache(trimmed_legacy)
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            elif torch.is_tensor(layer):
                trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)

    def _strip_input_from_past(
        self,
        past_kv: Optional[Tuple],
        input_start: int,
        input_end: int,
    ) -> Optional[Tuple]:
        """Remove input token entries at positions [input_start, input_end) from KV cache."""
        if past_kv is None or input_start >= input_end:
            return past_kv

        def _strip_tensor(t: torch.Tensor) -> torch.Tensor:
            left = t[..., :input_start, :]
            right = t[..., input_end:, :]
            return torch.cat([left, right], dim=-2).contiguous()

        if Cache is not None and isinstance(past_kv, Cache):
            legacy = past_kv.to_legacy_cache()
            stripped = tuple(
                tuple(_strip_tensor(t) for t in layer)
                for layer in legacy
            )
            return past_kv.__class__.from_legacy_cache(stripped)

        stripped_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                stripped_layers.append(tuple(_strip_tensor(t) for t in layer))
            elif torch.is_tensor(layer):
                stripped_layers.append(_strip_tensor(layer))
            else:
                stripped_layers.append(layer)
        return tuple(stripped_layers)

    # ------------------------------------------------------------------
    # Core generation — HF-only path
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        # Reset hidden states record for this batch
        self.hidden_states_record = []

        # Allocate query indices for logging
        query_indices = []
        if self.logger:
            for _ in range(batch_size):
                query_indices.append(self.logger.next_query_index())

        agent_idx_counter = 0
        accumulated_agent_names: List[str] = []
        last_past_kv = None

        for agent in self.agents:

            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]

            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)

                if self.args.think:
                    wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else:
                    wrapped_prompts = prompts

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                # -------------------------------------------------------
                # Two-phase flow: Think purely → AE filter for transfer
                # If no AE loaded, fall back to standard all-layers method.
                # -------------------------------------------------------
                if self.model._ae_controllers:
                    # Phase 1: Think (standard latent steps, no AE)
                    # Phase 2: AE reconstruct + filter → transfer KV
                    ae_layers = getattr(self.args, "ae_target_layers_parsed", None)
                    past_kv, _, hs_record = self.model.generate_latent_then_transfer(
                        wrapped_ids,
                        attention_mask=wrapped_mask,
                        latent_steps=self.latent_steps,
                        past_key_values=past_kv,
                        ae_target_layers=ae_layers,
                    )
                    # Transfer KV = prev_agents_KV + N_ae_layers transfer tokens
                    # For sequential/latent modes: keep only this agent's transfer tokens
                    if self.sequential_info_only or self.latent_only:
                        n_transfer = len(
                            ae_layers or list(self.model._ae_controllers.keys())
                        )
                        past_kv = self._truncate_past(past_kv, n_transfer)
                    # accumulate_latent: no truncation — transfer tokens accumulate naturally
                else:
                    past_kv, hs_record = self.model.generate_latent_batch_all_layers_hidden_states(
                        wrapped_ids,
                        attention_mask=wrapped_mask,
                        latent_steps=self.latent_steps,
                        past_key_values=past_kv,
                    )
                    if self.accumulate_latent:
                        input_len = wrapped_ids.shape[1]
                        past_kv = self._strip_input_from_past(
                            past_kv, prev_past_len, prev_past_len + input_len
                        )
                    elif self.sequential_info_only or self.latent_only:
                        new_past_len = _past_length(past_kv)
                        tokens_added = new_past_len - prev_past_len
                        tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                        past_kv = self._truncate_past(past_kv, tokens_to_keep)

                # Store hidden states for this agent
                self.hidden_states_record.append({
                    "agent_name": agent.name,
                    "last_input_token": hs_record["last_input_token"],
                    "latent_steps": hs_record["latent_steps"],
                })

                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    trace_entry = {
                        "name": agent.name,
                        "role": agent.role,
                        "input": wrapped_prompts[idx],
                        "input_ids": trimmed_ids,
                        "input_tokens": wrapped_tokens_batch[idx],
                        "latent_steps": self.latent_steps,
                        "output": "",
                        "input_token_count": len(trimmed_ids),
                        "output_token_count": 0,
                    }
                    agent_traces[idx].append(trace_entry)

                    if self.logger:
                        self.logger.save_agent_trace(
                            query_idx=query_indices[idx],
                            agent_idx=agent_idx_counter,
                            agent_name=agent.name,
                            trace=trace_entry,
                        )

                accumulated_agent_names.append(agent.name)
                last_past_kv = past_kv

            else:
                # ── Skip judger entirely when collecting hidden states ──
                if self.skip_judger:
                    agent_idx_counter += 1
                    continue

                past_for_decoding = past_kv if self.latent_steps > 0 else None

                if self.args.think:
                    judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else:
                    judger_prompts = prompts

                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                judger_ids = judger_encoded["input_ids"].to(self.model.device)
                judger_mask = judger_encoded["attention_mask"].to(self.model.device)
                judger_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(judger_ids, judger_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    judger_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                generated_batch, judger_past_kv = self.model.generate_text_batch(
                    judger_ids,
                    judger_mask,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    past_key_values=past_for_decoding,
                )
                for idx in range(batch_size):
                    final_text = generated_batch[idx].strip()
                    final_texts[idx] = final_text
                    mask = judger_mask[idx].bool()
                    trimmed_ids = judger_ids[idx][mask].to("cpu").tolist()
                    input_token_count = len(trimmed_ids)
                    output_token_count = len(self.model.tokenizer.encode(final_text, add_special_tokens=False))
                    trace_entry = {
                        "name": agent.name,
                        "role": agent.role,
                        "input": judger_prompts[idx],
                        "input_ids": trimmed_ids,
                        "input_tokens": judger_tokens_batch[idx],
                        "output": final_text,
                        "input_token_count": input_token_count,
                        "output_token_count": output_token_count,
                    }
                    agent_traces[idx].append(trace_entry)

                    if self.logger:
                        self.logger.save_agent_trace(
                            query_idx=query_indices[idx],
                            agent_idx=agent_idx_counter,
                            agent_name=agent.name,
                            trace=trace_entry,
                        )

                accumulated_agent_names.append(agent.name)
                last_past_kv = judger_past_kv

            agent_idx_counter += 1

        # Save accumulated KV cache once per query (after the last agent)
        if self.logger and self.save_kv and last_past_kv is not None:
            for idx in range(batch_size):
                self.logger.save_kv_cache_accumulated(
                    query_idx=query_indices[idx],
                    agent_names=accumulated_agent_names,
                    past_key_values=last_past_kv,
                    batch_idx=idx,
                )

        # Free GPU memory between batches
        del past_kv, last_past_kv
        torch.cuda.empty_cache()

        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            if self.task in ['mbppplus', 'humanevalplus']:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")

                if pred is None:
                    ok = False
                    error_msg = "python error: No python code block found"
                else:
                    python_code_to_exe = pred + "\n" + gold
                    ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)

                print(f'=========================================')
                print(f'Question {idx}')
                print(f'error_msg: {error_msg}')

            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    pred_int = int(pred)
                    gold_int = int(gold)
                    ok = (pred_int == gold_int)
                    error_msg = None
                except ValueError:
                    ok = False
                    error_msg = f'Value error in parsing answer. Pred: {pred}, Gold: {gold}'

            else:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None

            query_input_tokens = sum(a.get("input_token_count", 0) for a in agent_traces[idx])
            query_output_tokens = sum(a.get("output_token_count", 0) for a in agent_traces[idx])

            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                    "query_total_input_tokens": query_input_tokens,
                    "query_total_output_tokens": query_output_tokens,
                    "query_total_tokens": query_input_tokens + query_output_tokens,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Core generation — vLLM path
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_batch_vllm(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        # Reset hidden states record for this batch
        self.hidden_states_record = []

        # Allocate query indices for logging
        query_indices = []
        if self.logger:
            for _ in range(batch_size):
                query_indices.append(self.logger.next_query_index())

        agent_idx_counter = 0
        embedding_record = []
        accumulated_agent_names: List[str] = []

        for agent in self.agents:

            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]

            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)

                if self.args.think:
                    wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else:
                    wrapped_prompts = prompts

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.HF_device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.HF_device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                # -------------------------------------------------------
                # Two-phase flow: Think purely → AE filter for transfer
                # If no AE loaded, fall back to standard all-layers method.
                # -------------------------------------------------------
                if self.model._ae_controllers:
                    # Phase 1: Think (standard latent steps, no AE)
                    # Phase 2: AE reconstruct + filter → transfer embeddings
                    ae_layers = getattr(self.args, "ae_target_layers_parsed", None)
                    past_kv, transfer_embeds, hs_record = self.model.generate_latent_then_transfer(
                        wrapped_ids,
                        attention_mask=wrapped_mask,
                        latent_steps=self.latent_steps,
                        past_key_values=past_kv,
                        ae_target_layers=ae_layers,
                    )
                    # transfer_embeds: [B, N_ae_layers, D] — already AE-filtered & aligned
                    previous_hidden_embedding = transfer_embeds.detach()

                    if self.sequential_info_only or self.latent_only:
                        n_transfer = transfer_embeds.shape[1]
                        past_kv = self._truncate_past(past_kv, n_transfer)
                    # accumulate_latent: no truncation needed
                else:
                    past_kv, hs_record = self.model.generate_latent_batch_all_layers_hidden_states(
                        wrapped_ids,
                        attention_mask=wrapped_mask,
                        latent_steps=self.latent_steps,
                        past_key_values=past_kv,
                    )
                    # Build embedding record from hs_record for vLLM judger
                    last_input_emb = hs_record["last_input_token"][:, 0:1, :]
                    latent_last_layers = torch.cat(
                        [s[:, -1:, :] for s in hs_record["latent_steps"]], dim=1
                    ) if hs_record["latent_steps"] else last_input_emb[:, 0:0, :]
                    previous_hidden_embedding = torch.cat(
                        [last_input_emb, latent_last_layers], dim=1
                    )

                    if self.accumulate_latent:
                        input_len = wrapped_ids.shape[1]
                        past_kv = self._strip_input_from_past(
                            past_kv, prev_past_len, prev_past_len + input_len
                        )
                        if self.latent_steps > 0:
                            previous_hidden_embedding = previous_hidden_embedding[:, -self.latent_steps:, :]
                        else:
                            previous_hidden_embedding = previous_hidden_embedding[:, 0:0, :]
                    elif self.sequential_info_only or self.latent_only:
                        new_past_len = _past_length(past_kv)
                        tokens_added = new_past_len - prev_past_len
                        tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                        past_kv = self._truncate_past(past_kv, tokens_to_keep)
                        if self.latent_only:
                            if self.latent_steps > 0:
                                previous_hidden_embedding = previous_hidden_embedding[:, -self.latent_steps:, :]
                            else:
                                previous_hidden_embedding = previous_hidden_embedding[:, 0:0, :]

                # Store hidden states for this agent
                self.hidden_states_record.append({
                    "agent_name": agent.name,
                    "last_input_token": hs_record["last_input_token"],
                    "latent_steps": hs_record["latent_steps"],
                })

                embedding_record.append(previous_hidden_embedding)

                if self.sequential_info_only or self.latent_only:
                    embedding_record = embedding_record[-1:]

                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    trace_entry = {
                        "name": agent.name,
                        "role": agent.role,
                        "input": wrapped_prompts[idx],
                        "input_ids": trimmed_ids,
                        "input_tokens": wrapped_tokens_batch[idx],
                        "latent_steps": self.latent_steps,
                        "output": "",
                        "input_token_count": len(trimmed_ids),
                        "output_token_count": 0,
                    }
                    agent_traces[idx].append(trace_entry)

                    if self.logger:
                        self.logger.save_agent_trace(
                            query_idx=query_indices[idx],
                            agent_idx=agent_idx_counter,
                            agent_name=agent.name,
                            trace=trace_entry,
                        )

                accumulated_agent_names.append(agent.name)

            else:
                # A stack of [B, L_i, H]
                past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)

                if self.args.think:
                    judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else:
                    judger_prompts = prompts

                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                judger_encoded = judger_encoded["input_ids"].to(self.model.HF_device)
                curr_prompt_emb = self.model.embedding_layer(judger_encoded).squeeze(0).to(self.vllm_device)

                # assert Qwen model
                assert "Qwen" in self.args.model_name or "qwen" in self.args.model_name, \
                    "latent_embedding_position is only supported for Qwen models currently."

                len_of_left = []
                for p in judger_prompts:
                    idx = p.find("<|im_start|>user\n")
                    left = p[: idx + len("<|im_start|>user\n")]
                    len_of_left.append(len(self.model.tokenizer(left)['input_ids']))

                B, L, H = curr_prompt_emb.shape
                _, Lp, H = past_embedding.shape

                whole_prompt_emb_list = []
                for i in range(B):
                    insert_idx = len_of_left[i]
                    left_emb = curr_prompt_emb[i, :insert_idx, :]
                    right_emb = curr_prompt_emb[i, insert_idx:, :]
                    combined = torch.cat([left_emb, past_embedding[i], right_emb], dim=0)
                    whole_prompt_emb_list.append(combined)

                max_len = max(x.shape[0] for x in whole_prompt_emb_list)
                whole_prompt_emb = torch.stack([
                    torch.cat([x, torch.zeros(max_len - x.shape[0], H, device=x.device)], dim=0)
                    for x in whole_prompt_emb_list
                ])

                prompt_embeds_list = [
                    {
                        "prompt_embeds": embeds
                    } for embeds in whole_prompt_emb
                ]

                outputs = self.model.vllm_engine.generate(
                    prompt_embeds_list,
                    self.sampling_params,
                )

                generated_texts = [out.outputs[0].text.strip() for out in outputs]

                for idx in range(batch_size):
                    text_out = generated_texts[idx].strip()
                    final_texts[idx] = text_out
                    input_token_count = len(self.model.tokenizer.encode(judger_prompts[idx], add_special_tokens=False))
                    output_token_count = len(self.model.tokenizer.encode(text_out, add_special_tokens=False))
                    trace_entry = {
                        "name": agent.name,
                        "role": agent.role,
                        "input": judger_prompts[idx],
                        "output": text_out,
                        "input_token_count": input_token_count,
                        "output_token_count": output_token_count,
                    }
                    agent_traces[idx].append(trace_entry)

                    if self.logger:
                        self.logger.save_agent_trace(
                            query_idx=query_indices[idx],
                            agent_idx=agent_idx_counter,
                            agent_name=agent.name,
                            trace=trace_entry,
                        )

                accumulated_agent_names.append(agent.name)

            agent_idx_counter += 1

        # Save accumulated KV cache once per query
        if self.logger and self.save_kv and past_kv is not None:
            for idx in range(batch_size):
                self.logger.save_kv_cache_accumulated(
                    query_idx=query_indices[idx],
                    agent_names=accumulated_agent_names,
                    past_key_values=past_kv,
                    batch_idx=idx,
                )

        # Free GPU memory between batches
        del past_kv, embedding_record
        torch.cuda.empty_cache()

        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            pred = normalize_answer(extract_gsm8k_answer(final_text))
            gold = item["gold"]
            ok = (pred == gold) if (pred and gold) else False
            query_input_tokens = sum(a.get("input_token_count", 0) for a in agent_traces[idx])
            query_output_tokens = sum(a.get("output_token_count", 0) for a in agent_traces[idx])

            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                    "query_total_input_tokens": query_input_tokens,
                    "query_total_output_tokens": query_output_tokens,
                    "query_total_tokens": query_input_tokens + query_output_tokens,
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
