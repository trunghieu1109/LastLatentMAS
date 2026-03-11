import argparse
import json
import os
import shlex
from typing import Dict, List, Tuple


def _load_dotenv(path: str | None = None) -> None:
    """Load key=value pairs from a .env file into os.environ.

    Values already set in the environment (e.g. via shell export) are NOT
    overridden — the .env file acts only as a fallback default.

    Supports:
      - Blank lines and lines starting with '#' (comments) are skipped.
      - Optional 'export KEY=VALUE' prefix.
      - Inline comments after the value (e.g. KEY=foo  # comment).
      - Quoted values (single or double): KEY="hello world".
    """
    if path is None:
        # Look for .env next to this script
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(path):
        return
    with open(path) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            # Skip blanks and comments
            if not line or line.startswith("#"):
                continue
            # Strip optional 'export ' prefix
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, _, rest = line.partition("=")
            key = key.strip()
            # Strip inline comment (everything after unquoted ' #')
            rest = rest.split(" #")[0].split("\t#")[0].strip()
            # Strip surrounding quotes
            if (rest.startswith('"') and rest.endswith('"')) or \
               (rest.startswith("'") and rest.endswith("'")):
                rest = rest[1:-1]
            # Only set if not already in the environment
            if key and key not in os.environ:
                os.environ[key] = rest

from tqdm import tqdm

from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa
)
from methods.baseline import BaselineMethod
from methods.our_mas import LatentMASMethod
# from methods.our_mas import ComLatMAS
from methods.text_mas import TextMASMethod
from methods.ae_trainer import AETrainer, AEConfig
from models import ModelWrapper
from utils import auto_device, set_seed
from experiment_logger import ExperimentLogger
import time


def _parse_target_layers(spec: str, model) -> list:
    """Parse ``--ae_target_layers`` into a list of integer layer indices.

    Supported formats:
        * ``"all"``            → every layer index ``[0 .. num_hidden_layers]``
        * ``"5-20"``           → range(5, 21)  (inclusive end)
        * ``"-10--1"``         → [-10, -9, ..., -1]  (negative range)
        * ``"-5,-4,-3,-2,-1"`` → explicit comma-separated signed ints
    """
    import re
    spec = spec.strip()

    # --- "all" ----------------------------------------------------------------
    if spec.lower() == "all":
        hf = getattr(model, "HF_model", None) or getattr(model, "model", None)
        n = hf.config.num_hidden_layers + 1  # +1 for embedding layer
        layers = list(range(n))
        print(f"[ae_target_layers] 'all' → {n} layers (0 … {n - 1})")
        return layers

    # --- range  e.g. "5-20" or "-10--1" ----------------------------------------
    # Use regex to match two signed integers separated by a dash.
    # Pattern: optional minus + digits, dash, optional minus + digits
    # e.g.  "5-20"  →  (5, 20)     "-10--1"  →  (-10, -1)
    m = re.match(r'^\s*(-?\d+)\s*-\s*(-?\d+)\s*$', spec)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        layers = list(range(lo, hi + 1))
        print(f"[ae_target_layers] range {lo}..{hi} → {len(layers)} layers")
        return layers

    # --- comma-separated ints -------------------------------------------------
    return [int(x.strip()) for x in spec.split(",")]


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct

# Main processing function for each batch
def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds
    current_batch = batch[:remaining]
    if args.method == "latent_mas" and args.use_vllm: 
        results = method.run_batch_vllm(current_batch) 
    else:
        results = method.run_batch(current_batch)
    if len(results) > remaining:
        results = results[:remaining]
    batch_start = processed
    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1
        print(f"\n==================== Problem #{problem_idx} ====================")
        print("Question:")
        print(res.get("question", "").strip())
        agents = res.get("agents", [])
        for a in agents:
            name = a.get("name", "Agent")
            role = a.get("role", "")
            agent_header = f"----- Agent: {name} ({role}) -----"
            print(agent_header)
            agent_input = a.get("input", "").rstrip()
            agent_output = a.get("output", "").rstrip()
            latent_steps = a.get("latent_steps", None)
            print("[To Tokenize]")
            print(agent_input)
            if latent_steps is not None:
                print("[Latent Steps]")
                print(latent_steps)
            print("[Output]")
            print(agent_output)
            print("----------------------------------------------")
        print(f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}")

    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    return processed, preds


def main():
    # Load .env before building the parser so env vars feed into defaults
    _load_dotenv()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------ #
    # Helper: read a default from os.environ (populated from .env above)  #
    # ------------------------------------------------------------------ #
    def _env(key: str, default):
        """Return os.environ[key] cast to the same type as *default*, or *default*."""
        raw = os.environ.get(key)
        if raw is None:
            return default
        if isinstance(default, bool):
            return raw.lower() in ("1", "true", "yes")
        try:
            return type(default)(raw)
        except (ValueError, TypeError):
            return default

    # core args for experiments
    parser.add_argument("--method",
                        choices=["baseline", "text_mas", "latent_mas", "com_mas",
                                 "collect_and_train_ae", "collect_only"],
                        default=_env("METHOD", None),
                        required=(os.environ.get("METHOD") is None),
                        help="Which multi-agent method to run. Use 'collect_and_train_ae' "
                             "to collect hidden state data and train the Query-Aware AE. "
                             "Use 'collect_only' to only collect hidden states and export "
                             "to a benchmark/ directory (no AE training).")
    parser.add_argument("--model_name", type=str,
                        choices=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B", "Qwen/Qwen3-14B"],
                        default=_env("MODEL_NAME", None),
                        required=(os.environ.get("MODEL_NAME") is None),
                        help="HuggingFace model name.")
    parser.add_argument("--max_samples", type=int,
                        default=_env("MAX_SAMPLES", -1),
                        help="Number of questions to evaluate; -1 = all.")
    parser.add_argument("--task",
                        choices=["gsm8k", "aime2024", "aime2025", "gpqa",
                                 "arc_easy", "arc_challenge", "mbppplus",
                                 "humanevalplus", "medqa"],
                        default=_env("TASK", "gsm8k"),
                        help="Dataset/task to evaluate.")
    parser.add_argument("--prompt", type=str,
                        choices=["sequential", "hierarchical"],
                        default=_env("PROMPT", "sequential"),
                        help="Multi-agent system architecture.")

    # other args
    parser.add_argument("--device", type=str, default=_env("DEVICE", "cuda"))
    parser.add_argument("--split", type=str, default=_env("SPLIT", "test"))
    parser.add_argument("--max_new_tokens", type=int, default=_env("MAX_NEW_TOKENS", 4096))
    parser.add_argument("--latent_steps", type=int, default=_env("LATENT_STEPS", 0),
                        help="Number of latent steps for LatentMAS method.")
    parser.add_argument("--temperature", type=float, default=_env("TEMPERATURE", 0.6))
    parser.add_argument("--top_p", type=float, default=_env("TOP_P", 0.95))
    parser.add_argument("--generate_bs", type=int, default=_env("GENERATE_BS", 5),
                        help="Batch size for generation.")
    parser.add_argument("--text_mas_context_length", type=int,
                        default=_env("TEXT_MAS_CONTEXT_LENGTH", -1),
                        help="TextMAS context length limit.")
    parser.add_argument("--think", action="store_true",
                        default=_env("THINK", False),
                        help="Manually add <think> token in the prompt for LatentMAS.")
    parser.add_argument("--latent_space_realign", action="store_true",
                        default=_env("LATENT_SPACE_REALIGN", False))
    parser.add_argument("--accumulate_latent", action="store_true",
                        default=_env("ACCUMULATE_LATENT", False),
                        help="Accumulate only latent thoughts across agents in KV cache.")
    parser.add_argument("--seed", type=int, default=_env("SEED", 42))

    # vLLM support
    parser.add_argument("--use_vllm", action="store_true",
                        default=_env("USE_VLLM", False),
                        help="Use vLLM backend for generation.")
    parser.add_argument("--enable_prefix_caching", action="store_true",
                        default=_env("ENABLE_PREFIX_CACHING", False),
                        help="Enable prefix caching in vLLM for latent_mas.")
    parser.add_argument("--use_second_HF_model", action="store_true",
                        default=_env("USE_SECOND_HF_MODEL", False),
                        help="Use a second HF model for latent generation in latent_mas.")
    parser.add_argument("--device2", type=str, default=_env("DEVICE2", "cuda:1"))
    parser.add_argument("--tensor_parallel_size", type=int,
                        default=_env("TENSOR_PARALLEL_SIZE", 1),
                        help="How many GPUs vLLM should shard the model across.")
    parser.add_argument("--gpu_memory_utilization", type=float,
                        default=_env("GPU_MEMORY_UTILIZATION", 0.8),
                        help="Target GPU memory utilization for vLLM.")
    parser.add_argument("--device_map_auto", action="store_true",
                        default=_env("DEVICE_MAP_AUTO", False),
                        help="Use device_map='auto' for HF model.")
    parser.add_argument("--kv_flush_interval", type=int,
                        default=_env("KV_FLUSH_INTERVAL", 30),
                        help="Flush KV caches to disk every N queries.")
    parser.add_argument("--save_kv", action="store_true",
                        default=_env("SAVE_KV", False),
                        help="Enable saving KV caches to disk (latent_mas mode).")

    # ── Query-Aware AE args ────────────────────────────────────────────────
    parser.add_argument("--ae_target_layers", type=str,
                        default=_env("AE_TARGET_LAYERS", "-5,-4,-3,-2,-1"),
                        help="Layer indices for hidden-state collection / AE training. "
                             "Accepts: 'all' (every layer), a range like '0-27' or "
                             "'-10--1', or comma-separated signed ints e.g. '-5,-4,-3,-2,-1'.")
    parser.add_argument("--ae_bottleneck_dim", type=int,
                        default=_env("AE_BOTTLENECK_DIM", 512),
                        help="Bottleneck dimension Z for the Query-Aware AE.")
    parser.add_argument("--ae_epochs", type=int,
                        default=_env("AE_EPOCHS", 150),
                        help="Training epochs per layer.")
    parser.add_argument("--ae_batch_size", type=int,
                        default=_env("AE_BATCH_SIZE", 256),
                        help="Mini-batch size for AE training.")
    parser.add_argument("--ae_lr", type=float,
                        default=_env("AE_LR", 1e-3),
                        help="Learning rate for AE optimizer.")
    parser.add_argument("--ae_lambda_sparse", type=float,
                        default=_env("AE_LAMBDA_SPARSE", 0.01),
                        help="Weight for Jacobian sparsity loss.")
    parser.add_argument("--ae_lambda_attn", type=float,
                        default=_env("AE_LAMBDA_ATTN", 0.1),
                        help="Weight for attention entropy loss.")
    parser.add_argument("--ae_num_heads", type=int,
                        default=_env("AE_NUM_HEADS", 0),
                        help="Hidden dim of FeatureGate MLP (0 = auto = hidden_dim//2).")
    parser.add_argument("--ae_checkpoint_dir", type=str,
                        default=_env("AE_CHECKPOINT_DIR", ""),
                        help="Root directory for AE checkpoints and data cache. "
                             "Defaults to ae_checkpoints/paired_hidden_state_cache/.")
    parser.add_argument("--ae_collect_samples", type=int,
                        default=_env("AE_COLLECT_SAMPLES", -1),
                        help="Number of samples to use for hidden state collection "
                             "(-1 = same as --max_samples / all dataset).")
    parser.add_argument("--ae_latent_data_dir", type=str,
                        default=_env("AE_LATENT_DATA_DIR", "latent_data"),
                        help="Directory to save the collected hidden states when using 'collect_only'.")
    parser.add_argument("--ae_non_judger_agents", type=str,
                        default=_env("AE_NON_JUDGER_AGENTS", "Planner,Critic,Refiner"),
                        help="Comma-separated agent names whose hidden states are collected.")

    # ── WCT space alignment args ──────────────────────────────────────────
    parser.add_argument("--wct_params_dir", type=str,
                        default=_env("WCT_PARAMS_DIR", ""),
                        help="Directory with precomputed WCT params (from build_wct_params.py). "
                             "If set, WCT is applied after AE to map hidden states from "
                             "layer_i space to last layer space before realignment.")

    args = parser.parse_args()
    
    if args.method == "latent_mas" and args.use_vllm:
        args.use_second_HF_model = True 
        args.enable_prefix_caching = True
    
    set_seed(args.seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)

    # --- Create experiment logger ---
    logger = ExperimentLogger(
        base_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_logs"),
        task=args.task,
        args=args,
        kv_flush_interval=args.kv_flush_interval,
    )
    
    start_time = time.time()

    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # method selection 
    if args.method == "baseline":
        method = BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            use_vllm=args.use_vllm,
            args=args,
            logger=logger,
        )
    elif args.method == "text_mas":
        method = TextMASMethod(
            model,
            max_new_tokens_each=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
            logger=logger,
        )
    elif args.method == 'latent_mas':
        method = LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
            logger=logger,
        )
        # Auto-load QAAE controllers if checkpoint dir & target layers exist
        _ae_ckpt_dir = args.ae_checkpoint_dir or os.path.join(
            os.path.dirname(__file__), "ae_checkpoints", "paired_hidden_state_cache"
        )
        if os.path.isdir(_ae_ckpt_dir) and args.ae_target_layers:
            _ae_layers = _parse_target_layers(args.ae_target_layers, model)
            # Store parsed layers on args for LatentMASMethod to access
            args.ae_target_layers_parsed = _ae_layers
            _ae_device = args.device2 if args.use_second_HF_model else args.device
            model.load_ae_controllers(
                checkpoint_dir=_ae_ckpt_dir,
                target_layers=_ae_layers,
                device=_ae_device,
            )

            # Auto-load WCT params if specified
            _wct_dir = args.wct_params_dir
            if not _wct_dir:
                # Default: check wct_params/ in project root
                _wct_dir_default = os.path.join(os.path.dirname(__file__), "wct_params")
                if os.path.isdir(_wct_dir_default):
                    _wct_dir = _wct_dir_default
            if _wct_dir and os.path.isdir(_wct_dir):
                model.load_wct_params(_wct_dir)
    elif args.method in ('collect_and_train_ae', 'collect_only'):
        method = LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=1,           # collect one sample at a time
            args=args,
            logger=logger,
        )

    # ── collect_and_train_ae / collect_only: early-exit after AE pipeline ──
    if args.method in ("collect_and_train_ae", "collect_only"):
        # Build dataset list upfront
        if args.task == "gsm8k":
            all_items = list(load_gsm8k(split=args.split))
        elif args.task == "aime2024":
            all_items = list(load_aime2024(split="train"))
        elif args.task == "aime2025":
            all_items = list(load_aime2025(split='train'))
        elif args.task == "gpqa":
            all_items = list(load_gpqa_diamond(split='test'))
        elif args.task == "arc_easy":
            all_items = list(load_arc_easy(split='test'))
        elif args.task == "arc_challenge":
            all_items = list(load_arc_challenge(split='test'))
        elif args.task == "mbppplus":
            all_items = list(load_mbppplus(split='test'))
        elif args.task == "humanevalplus":
            all_items = list(load_humanevalplus(split='test'))
        elif args.task == "medqa":
            all_items = list(load_medqa(split='test'))
        else:
            raise ValueError(f'no {args.task} support')

        n_collect = args.ae_collect_samples if args.ae_collect_samples > 0 else len(all_items)
        collect_items = all_items[:n_collect]
        print(f"[AE] Collecting over {len(collect_items)} samples from {args.task}.")

        # Parse AE config from args
        target_layers = _parse_target_layers(args.ae_target_layers, model)
        non_judger_agents = [x.strip() for x in args.ae_non_judger_agents.split(",")]
        ae_cfg = AEConfig(
            bottleneck_dim=args.ae_bottleneck_dim,
            gate_hidden=args.ae_num_heads,   # reusing the arg slot, 0 = auto
            epochs=args.ae_epochs,
            batch_size=args.ae_batch_size,
            lr=args.ae_lr,
            lambda_sparse=args.ae_lambda_sparse,
            lambda_attn=args.ae_lambda_attn,
            target_layers=target_layers,
            non_judger_agents=non_judger_agents,
            device=str(device),
        )

        ckpt_dir = args.ae_checkpoint_dir if args.ae_checkpoint_dir else None
        ae_trainer = AETrainer(method, args, ae_cfg=ae_cfg, checkpoint_dir=ckpt_dir)

        if args.method == "collect_only":
            ae_trainer.collect_only(collect_items)
            logger.shutdown()
            print(f"[Collect-Only] Done. Data → {ae_trainer.data_dir}")
        else:
            ae_trainer.collect_and_train(collect_items)
            logger.shutdown()
            print(f"[AE] Done. Data → {ae_trainer.data_dir}  |  Models → {ae_trainer.checkpoint_dir}")
        return

    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []
    
    # dataset loading
    if args.task == "gsm8k":
        dataset_iter = load_gsm8k(split=args.split)
    elif args.task == "aime2024":
        dataset_iter = load_aime2024(split="train")
    elif args.task == "aime2025":
        dataset_iter = load_aime2025(split='train')
    elif args.task == "gpqa":
        dataset_iter = load_gpqa_diamond(split='test')
    elif args.task == "arc_easy":
        dataset_iter = load_arc_easy(split='test')
    elif args.task == "arc_challenge":
        dataset_iter = load_arc_challenge(split='test')
    elif args.task == "mbppplus":
        dataset_iter = load_mbppplus(split='test')
    elif args.task == "humanevalplus":
        dataset_iter = load_humanevalplus(split='test')
    elif args.task == "medqa":
        dataset_iter = load_medqa(split='test')
    else:
        raise ValueError(f'no {args.task} support')

    if args.max_samples == -1:
        dataset_iter = list(dataset_iter)  
        args.max_samples = len(dataset_iter)

    progress = tqdm(total=args.max_samples)

    for item in dataset_iter:
        if processed >= args.max_samples:
            break
        batch.append(item)
        if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
            processed, preds = process_batch(
                method,
                batch,
                processed,
                preds,
                progress,
                args.max_samples,
                args,
            )
            batch = []
            if processed >= args.max_samples:
                break

    if batch and processed < args.max_samples:
        processed, preds = process_batch(
            method,
            batch,
            processed,
            preds,
            progress,
            max_samples=args.max_samples,
            args=args,
        )
    progress.close()
    
    total_time = time.time() - start_time

    acc, correct = evaluate(preds)

    # Aggregate token usage stats
    total_input_tokens = sum(p.get("query_total_input_tokens", 0) for p in preds)
    total_output_tokens = sum(p.get("query_total_output_tokens", 0) for p in preds)
    total_tokens = total_input_tokens + total_output_tokens
    num_queries = len(preds) if preds else 1

    # Run summary
    summary = {
        "method": args.method,
        "model": args.model_name,
        "split": args.split,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "accuracy": acc,
        "correct": correct,
        "total_time_sec": round(total_time, 4),
        "time_per_sample_sec": round(total_time / args.max_samples, 4),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "avg_input_tokens_per_query": round(total_input_tokens / num_queries, 2),
        "avg_output_tokens_per_query": round(total_output_tokens / num_queries, 2),
        "avg_total_tokens_per_query": round(total_tokens / num_queries, 2),
    }

    # Wait for any background KV-cache writes to finish
    if args.save_kv:
        logger.flush_kv_cache_queue()

    # Save all per-query results (predictions, agent traces, etc.)
    logger.save_results(preds)

    # Save summary to experiment log
    logger.save_summary(summary)

    # Stop the background writer thread
    logger.shutdown()

    # Load results in JSON format
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
        )
    )



if __name__ == "__main__":
    main()
