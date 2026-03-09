import os
import csv
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _past_length(past_key_values: Optional[Tuple]) -> int:
    if not past_key_values:
        return 0
    k = past_key_values[0][0]
    return k.shape[-2]


def _compute_max_memory(safety_margin_gb: float = 2.0) -> Dict[int, str]:
    """Compute max_memory dict for device_map='auto' based on actual free GPU memory.

    Call this AFTER any other frameworks (e.g. vLLM) have already allocated their
    GPU memory so that the returned values reflect genuinely available VRAM.

    Args:
        safety_margin_gb: Extra headroom (in GiB) to leave free on each GPU
            for KV caches, activations, and other dynamic allocations.

    Returns:
        Dict mapping GPU index -> available memory string, e.g. {0: '5.2GiB', 1: '22.0GiB'}
    """
    if not torch.cuda.is_available():
        return {}
    max_memory: Dict[int, str] = {}
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        available_gb = (free / (1024 ** 3)) - safety_margin_gb
        if available_gb > 0.5:  # only include GPUs with meaningful free memory
            max_memory[i] = f"{available_gb:.1f}GiB"
    return max_memory


class ModelWrapper:
    def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args = None):
        self.model_name = model_name
        self.device = device
        self.use_vllm = use_vllm and _HAS_VLLM
        self.vllm_engine = None
        self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        # AE enhancement: per-layer alignment matrices and loaded AE models
        self._ae_layer_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._ae_controllers: Dict[int, object] = {}   # signed_layer → QueryAwareAutoencoder
        self.args = args

        # for ablation
        self.pre_aligned = None

        self.device_map_auto = bool(getattr(args, "device_map_auto", False)) if args else False

        if self.use_vllm:
            
            tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
            gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))
            
            print(f"[vLLM] Using vLLM backend for model {model_name}")
            if args.enable_prefix_caching and args.method == "latent_mas": 
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util, enable_prefix_caching=True, enable_prompt_embeds=True)
            else:
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            
            use_second_hf = bool(getattr(args, "use_second_HF_model", False)) if args else False
            if use_second_hf:
                if self.device_map_auto:
                    # Compute available memory AFTER vLLM has allocated on its GPU(s)
                    max_memory = _compute_max_memory(safety_margin_gb=2.0)
                    print(f"[HF Model] Using device_map='auto' with max_memory={max_memory}")
                    self.HF_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                        device_map="auto",
                        max_memory=max_memory,
                    )
                    self.HF_model.eval()
                    # With device_map="auto", use the device of the first parameter
                    self.HF_device = str(self.HF_model.device)
                    print(f"[HF Model] Primary device: {self.HF_device}")
                    if hasattr(self.HF_model, 'hf_device_map'):
                        print(f"[HF Model] Device map: {self.HF_model.hf_device_map}")
                else:
                    self.HF_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                    ).to(args.device2).eval() 
                    self.HF_device = args.device2
                self.embedding_layer = self.HF_model.get_input_embeddings()
                # if self.latent_space_realign:
                self._ensure_latent_realign_matrix(self.HF_model, torch.device(self.HF_device), args)
            elif self.latent_space_realign:
                raise ValueError("latent_space_realign requires --use_second_HF_model when using vLLM backend.")
            _ensure_pad_token(self.tokenizer)
            return  # skip loading transformers model

        # fallback: normal transformers path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _ensure_pad_token(self.tokenizer)
        with torch.no_grad():
            if self.device_map_auto and torch.cuda.device_count() > 1:
                max_memory = _compute_max_memory(safety_margin_gb=2.0)
                print(f"[HF Model] Using device_map='auto' with max_memory={max_memory}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                    device_map="auto",
                    max_memory=max_memory,
                )
                self.device = self.model.device
                if hasattr(self.model, 'hf_device_map'):
                    print(f"[HF Model] Device map: {self.model.hf_device_map}")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                )
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        if not self.device_map_auto or torch.cuda.device_count() <= 1:
            self.model.to(device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        if self.latent_space_realign:
            self._ensure_latent_realign_matrix(self.model, self.device, args)

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts: List[str] = []
        for messages in batch_messages:
            prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt))
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    def vllm_generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        if not self.vllm_engine:
            raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.vllm_engine.generate(prompts, sampling_params)
        generations = [out.outputs[0].text.strip() for out in outputs]
        return generations
    
    def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")
        input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()

        if self.args.latent_space_realign:
            pass
        else:
            # keep the matrix, for further normalization
            realign_matrix = torch.eye(realign_matrix.shape[0], device=realign_matrix.device, dtype=realign_matrix.dtype)

        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(model, target_device, args)
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = target_norm.to(device=target_device, dtype=matrix.dtype) if isinstance(target_norm, torch.Tensor) else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        self._latent_realign_matrices[key] = (matrix, target_norm)

        return matrix, target_norm

    def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device, self.args)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pre_aligned = aligned.detach().clone()
        self.pre_aligned = pre_aligned
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    # ──────────────────────────────────────────────────────────────────────
    # AE-enhanced: per-layer alignment + QAAE reconstruction
    # ──────────────────────────────────────────────────────────────────────

    def load_ae_controllers(
        self,
        checkpoint_dir: str,
        target_layers: List[int],          # signed indices, e.g. [-5,-4,-3,-2,-1]
        device: Optional[str] = None,
    ) -> None:
        """Load trained QueryAwareAutoencoder checkpoints for each target layer.

        Args:
            checkpoint_dir: directory containing ae_layer_{idx}.pt files.
            target_layers:  signed layer indices matching the AE files.
            device:         torch device string; defaults to self.device.
        """
        from methods.query_aware_ae import QueryAwareAutoencoder

        dev = device or str(self.device)
        loaded = []
        for signed_li in target_layers:
            ckpt_path = os.path.join(checkpoint_dir, f"ae_layer_{signed_li}.pt")
            if not os.path.exists(ckpt_path):
                print(f"[ModelWrapper] AE checkpoint not found: {ckpt_path} — skipping.")
                continue
            ckpt = torch.load(ckpt_path, map_location=dev)
            cfg  = ckpt["config"]
            ae   = QueryAwareAutoencoder(
                hidden_dim     = cfg["hidden_dim"],
                bottleneck_dim = cfg["bottleneck_dim"],
                gate_hidden    = cfg.get("gate_hidden"),
                encoder_layers = cfg.get("encoder_layers", 2),
                decoder_layers = cfg.get("decoder_layers", 3),
                mlp_hidden     = cfg.get("mlp_hidden", 2048),
            ).to(dev)
            sd_key = "state_dict" if "state_dict" in ckpt else "model_state_dict"
            ae.load_state_dict(ckpt[sd_key])
            ae.eval()
            self._ae_controllers[signed_li] = ae
            loaded.append(signed_li)
        print(f"[ModelWrapper] Loaded AE controllers for layers: {loaded}")

    def _build_layer_realign_matrix(
        self,
        model,
        abs_layer_idx: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build an alignment matrix mapping hidden states from abs_layer_idx → embedding space.

        Uses least-squares: finds W such that W minimises ||embed_W - h_layer||_F
        over the vocabulary embedding vectors.  Target norm = mean norm of embedding vectors.

        This is analogous to _build_latent_realign_matrix but sources hidden
        activations from an intermediate layer's output projection (we approximate
        this via the lm_head weight, which is tied to the embedding in most LMs).
        """
        # Embedding matrix E: [vocab, D]  (input embedding weights)
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        if input_embeds is None or not hasattr(input_embeds, "weight"):
            raise RuntimeError("Cannot build layer realign matrix: embedding weights not accessible.")

        E = input_embeds.weight.detach().to(device=device, dtype=torch.float32)   # [V, D]
        target_norm = E.norm(dim=1).mean().detach()

        # Layer output distribution proxy: pass a small batch of embedding vectors
        # through the transformer up to abs_layer_idx and take mean/std per feature.
        # For efficiency, we use the embedding weights themselves as representative
        # input — this is a fast, closed-form approximation.
        #
        # The realignment solves: W  s.t.  E ≈ H_layer · W
        # i.e. W = (H_layer^T H_layer)^{-1} H_layer^T E  (least-squares)
        #
        # We approximate H_layer ≈ E  (same space assumption, scaled by layer depth)
        # and just build a lightweight scale+rotation correction:
        gram = torch.matmul(E.T, E)                     # [D, D]
        reg  = 1e-5 * torch.eye(gram.shape[0], device=device, dtype=torch.float32)
        rhs  = gram                                      # target = E maps to E
        W    = torch.linalg.solve(gram + reg, rhs)      # [D, D]  ≈ identity

        # Override to simple scaling correction between layers:
        # We want to shift the distribution of h_layer (which may have different
        # mean norm) to match the embedding distribution.
        # Store as (W=identity, target_norm) — the norm rescaling in
        # _apply_ae_realignment handles the rest.
        W = torch.eye(E.shape[1], device=device, dtype=torch.float32)
        return W, target_norm

    def _ensure_layer_realign_matrix(
        self,
        model,
        abs_layer_idx: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (id(model), abs_layer_idx)
        info = self._ae_layer_realign_matrices.get(key)
        if info is None:
            W, target_norm = self._build_layer_realign_matrix(model, abs_layer_idx, device)
            self._ae_layer_realign_matrices[key] = (W, target_norm)
        else:
            W, target_norm = info
        W          = W.to(device)
        target_norm = target_norm.to(device) if isinstance(target_norm, torch.Tensor) else torch.as_tensor(target_norm, device=device)
        return W, target_norm

    def _apply_ae_realignment(
        self,
        hidden: torch.Tensor,              # [B, D]
        model,
        abs_layer_idx: int,
    ) -> torch.Tensor:
        """Apply per-layer realignment: normalise hidden → embedding norm scale."""
        W, target_norm = self._ensure_layer_realign_matrix(model, abs_layer_idx, hidden.device)
        h = hidden.to(torch.float32)
        aligned = torch.matmul(h, W)       # [B, D]
        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[List[str], Optional[Tuple]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        cache_position = None
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        sequences = outputs.sequences
        generations: List[str] = []
        for idx, length in enumerate(prompt_lengths):
            length = int(length)
            generated_ids = sequences[idx, length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)
        return generations, outputs.past_key_values

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values

        e_t = outputs.hidden_states[0][:, -1, :]          # [B, D]
        last_hidden = outputs.hidden_states[-1][:, -1, :] # [B, D]
        h_t = last_hidden.detach().clone()

        e_t_plus_1 = None
        latent_vecs_all: List[torch.Tensor] = []
        latent_vecs_all.append(e_t.detach().clone())

        for step in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)

            latent_vecs_all.append(latent_vec.detach().clone())

            if step == 0:
                e_t_plus_1 = latent_vec.detach().clone()
            
            latent_embed = latent_vec.unsqueeze(1)

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device,
            )
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        return past
    
    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.HF_device)
        else:
            attention_mask = attention_mask.to(self.HF_device)
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.HF_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        
        curr_output_embedding = [] 
        curr_output_embedding.append(outputs.hidden_states[0])  # input embedding
        
        
        for _ in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)
            latent_embed = latent_vec.unsqueeze(1)
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            outputs = self.HF_model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            curr_output_embedding.append(latent_embed.detach())

        return past, torch.cat(curr_output_embedding, dim=1) # Output input embeddings

    @torch.no_grad()
    def generate_latent_batch_all_layers_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        """Generate latent steps while capturing hidden states at ALL layers.

        Returns:
            (past_key_values, hidden_states_record) where hidden_states_record is a dict:
                - "last_input_token": Tensor of shape [B, num_layers+1, D]
                    Hidden states of the last input token across all layers
                    (index 0 = embedding layer, index L = last transformer layer).
                - "latent_steps": List of Tensors, each of shape [B, num_layers+1, D]
                    Hidden states of each latent step token across all layers.
                    Length equals `latent_steps`.
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        model = self.HF_model if hasattr(self, "HF_model") else self.model
        device = self.HF_device if hasattr(self, "HF_device") else self.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        else:
            attention_mask = attention_mask.to(device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        # --- Forward pass on input tokens ---
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]

        # Collect hidden states of last input token across ALL layers: [B, num_layers+1, D]
        # outputs.hidden_states is a tuple of (num_layers+1,) tensors, each [B, seq_len, D]
        last_input_token_all_layers = torch.stack(
            [hs[:, -1, :].detach().clone() for hs in outputs.hidden_states],
            dim=1,
        )  # [B, num_layers+1, D]

        # --- Latent steps ---
        latent_step_all_layers: List[torch.Tensor] = []

        for _ in range(latent_steps):
            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)
            latent_embed = latent_vec.unsqueeze(1)  # [B, 1, D]

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            outputs = model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            # Collect hidden states of this latent token across ALL layers: [B, num_layers+1, D]
            step_all_layers = torch.stack(
                [hs[:, -1, :].detach().clone() for hs in outputs.hidden_states],
                dim=1,
            )  # [B, num_layers+1, D]
            latent_step_all_layers.append(step_all_layers)

        hidden_states_record = {
            "last_input_token": last_input_token_all_layers,  # [B, num_layers+1, D]
            "latent_steps": latent_step_all_layers,           # list of latent_steps x [B, num_layers+1, D]
        }

        return past, hidden_states_record




    @torch.no_grad()
    def generate_latent_batch_ae_enhanced(
        self,
        input_ids,
        attention_mask=None,
        *,
        latent_steps,
        past_key_values=None,
        ae_target_layers=None,
    ):
        """Generate latent steps using QAAE-enhanced hidden state fusion.

        Strategy (depends on number of loaded AE layers):

        Single AE layer (e.g. AE_TARGET_LAYERS=-1):
            Each latent step = 1 token: QAAE(-1)(last_hidden, h_input) → align → embed

        Multiple AE layers (e.g. AE_TARGET_LAYERS=-3,-2,-1):
            Each latent step = N tokens, one per layer, fed SEQUENTIALLY into KV cache:
              For l in [-3, -2, -1]:
                h_rec_l = QAAE_l(last_hidden_l, h_input_l) → align → embed_l
                model(inputs_embeds=embed_l, past_kv=past) → KV cache grows by 1 token
                last_hidden ← output hidden state of this sub-step
            So total KV tokens = latent_steps × N (multi-perspective latent thoughts)

        Falls back to standard _apply_latent_realignment if no AE loaded.
        """
        import torch
        model  = self.HF_model if hasattr(self, "HF_model") else self.model
        device = self.HF_device if hasattr(self, "HF_device") else str(self.device)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        else:
            attention_mask = attention_mask.to(device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        # ── Initial forward pass on input tokens ──────────────────────────────
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        num_layers = len(outputs.hidden_states) - 1  # exclude embedding layer

        # Hidden states of LAST INPUT TOKEN at every layer: [B, L+1, D]
        last_input_token_all_layers = torch.stack(
            [hs[:, -1, :].detach().clone() for hs in outputs.hidden_states],
            dim=1,
        )
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, D]

        # Resolve signed → abs layer indices (sorted from earliest to latest)
        ae_signed = sorted(
            ae_target_layers or list(self._ae_controllers.keys()),
            key=lambda x: x if x >= 0 else num_layers + x,
        )
        ae_abs = [(num_layers + li if li < 0 else li) for li in ae_signed]

        source_model = self.HF_model if hasattr(self, "HF_model") else self.model

        latent_step_all_layers = []

        # ── steps Latent ─────────────────
        for step in range(latent_steps):

            if not self._ae_controllers or not ae_signed:
                # ── Fallback: standard single-token realignment ────────────────
                latent_vec = self._apply_latent_realignment(last_hidden, source_model)
                latent_embed = latent_vec.unsqueeze(1)  # [B, 1, D]

                past_len = _past_length(past)
                latent_mask = torch.ones(
                    (latent_embed.shape[0], past_len + 1),
                    dtype=torch.long, device=latent_embed.device,
                )
                outputs = model(
                    inputs_embeds=latent_embed,
                    attention_mask=latent_mask,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                past = outputs.past_key_values
                last_hidden = outputs.hidden_states[-1][:, -1, :]

                step_all_layers = torch.stack(
                    [hs[:, -1, :].detach().clone() for hs in outputs.hidden_states], dim=1,
                )
                latent_step_all_layers.append(step_all_layers)

            else:
                # ── AE-enhanced: feed each layer as a SEPARATE token ──────────
                # Iterate layers in order (earliest → latest layer)
                step_last_hs = None  # record hs from last sub-step of this step

                for signed_li, abs_li in zip(ae_signed, ae_abs):
                    ae = self._ae_controllers.get(signed_li)

                    if ae is None:
                        # No AE for this layer → standard realignment token
                        latent_vec = self._apply_latent_realignment(last_hidden, source_model)
                    else:
                        # h_latent: the hidden state at this layer
                        #   step 0  → use input hs (no latent step yet)
                        #   step >0 → use previous step's hs at this layer
                        if step == 0 or step_last_hs is None:
                            h_latent_l = last_input_token_all_layers[:, abs_li, :]
                        else:
                            h_latent_l = step_last_hs[:, abs_li, :]

                        # h_input: query = last input token hs at this layer (fixed)
                        h_input_l = last_input_token_all_layers[:, abs_li, :]

                        h_latent_l = h_latent_l.to(device).float()
                        h_input_l  = h_input_l.to(device).float()

                        ae_device = next(ae.parameters()).device
                        h_rec, _z, _gate = ae(
                            h_latent_l.to(ae_device),
                            h_input_l.to(ae_device),
                            normalize=True,
                        )
                        h_rec = h_rec.to(device)

                        # Align to embedding distribution
                        latent_vec = self._apply_latent_realignment(
                            h_rec.to(last_hidden.dtype), source_model
                        )

                    # Feed this layer's token into KV cache
                    latent_embed = latent_vec.unsqueeze(1)   # [B, 1, D]
                    past_len = _past_length(past)
                    latent_mask = torch.ones(
                        (latent_embed.shape[0], past_len + 1),
                        dtype=torch.long, device=latent_embed.device,
                    )
                    outputs = model(
                        inputs_embeds=latent_embed,
                        attention_mask=latent_mask,
                        past_key_values=past,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    past = outputs.past_key_values
                    last_hidden = outputs.hidden_states[-1][:, -1, :]  # update for next sub-step

                # Record all-layer hs from the LAST sub-step of this latent step
                step_last_hs = torch.stack(
                    [hs[:, -1, :].detach().clone() for hs in outputs.hidden_states], dim=1,
                )
                latent_step_all_layers.append(step_last_hs)

        hidden_states_record = {
            "last_input_token": last_input_token_all_layers,
            "latent_steps":     latent_step_all_layers,
        }
        return past, hidden_states_record

