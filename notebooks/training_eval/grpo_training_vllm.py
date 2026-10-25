## Cell 1:

# ── Shared config ─────────────────────────────────────────────────────
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0

MAX_SEQ_LEN = 8192
RESET_WEIGHTS = (
    False  # if True, skip loading pretrained adapter; train from fresh LoRA init
)
IN_PROJ_ONLY = False
MOE_TIE_WEIGHTS = True  # if True, tie one side of MoE expert LoRA across all 128 experts (Tinker-style)

# ── GRPO knobs ────────────────────────────────────────────────────
NUM_STEPS = 200
BATCH_SIZE = 4              # unique prompts per optimizer step
GROUP_SIZE = 8              # G — completions sampled per prompt
MICRO_BATCH_SIZE = 4        # full sequences per training forward pass
LEARNING_RATE = 1e-5        # GRPO is much more sensitive than SFT
MAX_PROMPT_LEN = 2048
MAX_NEW_TOKENS = 2560       # MAX_SEQ_LEN must accommodate MAX_PROMPT_LEN + MAX_NEW_TOKENS
ROLLOUT_TEMPERATURE = 1.0
ROLLOUT_TOP_P = 1.0
MICRO_ROLLOUT_BATCH = 32    # sequences per generate() call; lower to 16/8 if rollout OOMs
CLIP_EPS = 0.2              # PPO ratio clip
PPO_EPOCHS = 1              # μ — optimizer iterations per rollout batch (1 = on-policy REINFORCE-with-clip)
ADV_NORM_EPS = 1e-6

# Instruction prepended to every prompt — sets up the output format the
# reward function (and the competition's grader) expects.
SYSTEM_INSTRUCTION = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "in_proj",
    "out_proj",
    "lm_head",
]

# ── vLLM rollout config ──────────────────────────────────────────
USE_VLLM_ROLLOUT = True       # if False, falls back to HF generate (legacy path)
VLLM_GPU_MEM_UTIL = 0.85      # fraction of GPU vLLM may use when awake
VLLM_MAX_LORA_RANK = LORA_RANK
VLLM_ENFORCE_EAGER = False    # set True if CUDA graph capture fails on NemotronH
VLLM_ADAPTER_TMP_DIR = "/kaggle/working/vllm_lora_tmp"

# ── Offline W&B (Kaggle has no internet) ──────────────────────────
USE_WANDB = True
WANDB_MODE = "offline"
WANDB_PROJECT = "nemotron-grpo"
WANDB_RUN_NAME = None              # auto if None
WANDB_DIR = "/kaggle/working/wandb"


## Cell 2:

# ── Kaggle offline install (matches SFT notebook package paths) ──────
import os
import subprocess

subprocess.run(
    "pip install -q --no-index --find-links /kaggle/input/datasets/mayukh18/nemotron-packages/packages "
    "unsloth trl peft transformers datasets accelerate bitsandbytes vllm",
    shell=True,
    check=True,
)
subprocess.run(
    "pip install -q /kaggle/input/datasets/mayukh18/nemotron-packages/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
    shell=True,
    check=True,
)
subprocess.run(
    "pip install -q /kaggle/input/datasets/mayukh18/nemotron-packages/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
    shell=True,
    check=True,
)
for _wd in ["/kaggle/input/datasets/llkh0a/rtx-wheels/wheels"]:
    if os.path.isdir(_wd):
        subprocess.run(
            [
                "pip",
                "install",
                "-q",
                "--no-index",
                "--find-links",
                _wd,
                "protobuf==6.33.5",
                "sentencepiece",
                "safetensors",
                "huggingface_hub",
            ],
            check=False,
        )
subprocess.run("rm -rf /kaggle/tmp/*", shell=True, check=True)

## Cell 3:
from unsloth import FastLanguageModel ## It has to be at the top to perform the patching

def run_training() -> None:
    """Full GRPO training flow on Kaggle (no GRPOTrainer dependency)."""
    import csv
    import gc
    import math
    import os
    import random
    import re
    import shutil
    import sys
    import time
    import zipfile

    import kagglehub
    import torch
    from cut_cross_entropy import linear_cross_entropy
    from peft import LoraConfig
    from peft.tuners.lora import Linear as LoraLinear
    from trl.rewards import think_format_reward as _trl_think_format_reward

    # ── Kaggle paths ─────────────────────────────────────────────────
    TRAIN_CSV_PATH = "/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv"
    ADAPTER_SRC = "/kaggle/input/models/zosov07/sft-nemotron-lora-adapter/transformers/performance_0_84/1"
    MODEL_PATH = kagglehub.model_download(
        "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
    )
    save_dir = "."

    # ── vLLM init (before HF model load so GPU is empty for vLLM weight load) ──
    # vLLM needs ~60GB free to load the 30B bf16 model. We init it here,
    # sleep it immediately (offloads weights to CPU), then load the HF model.
    # The two-way swap (vLLM↔GPU during rollout, HF↔CPU) is managed per step.
    vllm_engine = None
    if USE_VLLM_ROLLOUT:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        print("Initializing vLLM engine (will sleep immediately to free GPU for HF load)...")
        _t_vinit = time.time()
        vllm_engine = LLM(
            model=MODEL_PATH,
            dtype="bfloat16",
            trust_remote_code=True,
            enable_lora=True,
            max_lora_rank=VLLM_MAX_LORA_RANK,
            max_loras=1,
            max_cpu_loras=2,
            enable_sleep_mode=True,
            gpu_memory_utilization=VLLM_GPU_MEM_UTIL,
            max_model_len=MAX_SEQ_LEN,
            enforce_eager=VLLM_ENFORCE_EAGER,
        )
        vllm_engine.sleep(level=2)  # offload bf16 weights to CPU; frees ~60GB
        print(f"vLLM init + sleep done in {time.time() - _t_vinit:.1f}s")
        os.makedirs(VLLM_ADAPTER_TMP_DIR, exist_ok=True)

    # ── GPU + kernel sanity check ────────────────────────────────────
    import causal_conv1d
    import mamba_ssm

    cc = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}, sm_{cc[0] * 10 + cc[1]}")
    print(f"torch={torch.__version__}, cuda={torch.version.cuda}")
    print(
        f"mamba_ssm={mamba_ssm.__version__}, causal_conv1d={causal_conv1d.__version__}"
    )
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    from causal_conv1d import causal_conv1d_fn

    _x = torch.randn(1, 256, 32, device="cuda", dtype=torch.bfloat16)
    _w = torch.randn(256, 4, device="cuda", dtype=torch.bfloat16)
    causal_conv1d_fn(_x, _w, None, activation="silu")
    print("causal_conv1d CUDA kernel: OK")

    # ── Load base model ──────────────────────────────────────────────
    gc.collect()
    torch.cuda.empty_cache()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=True,
        attn_implementation="eager",  # TODO: sdpa investigate why FA2=False despite xformers=0.0.35 being present
        dtype=torch.bfloat16,
    )

    # ── Wrap in LoRA (Unsloth grad checkpointing offloads activations to CPU) ──
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    FastLanguageModel.for_training(model)

    # ── Patch Mamba CUDA fast path (Unsloth disables it on import) ───
    nemotron_mod = None
    for _name, _m in sys.modules.items():
        if "modeling_nemotron_h" in _name and hasattr(_m, "is_fast_path_available"):
            nemotron_mod = _m
            break
    assert nemotron_mod is not None, "Could not find modeling_nemotron_h module"
    print(f"is_fast_path_available was: {nemotron_mod.is_fast_path_available}")
    nemotron_mod.is_fast_path_available = True  # type: ignore[attr-defined]
    print("Patched is_fast_path_available = True")

    # Fix three bugs in the vendored HybridMambaAttentionDynamicCache:
    #  1. conv_states is allocated with `mamba_num_heads * mamba_head_dim` channels,
    #     but the conv operates on `[hidden, B, C]` = that + 2 * n_groups * ssm_state_size.
    #     Wrong dim → causal_conv1d_update raises "weight must have shape (dim, width)".
    #  2. get_seq_length() returns shape[-2] of the empty placeholder `torch.tensor([[]]*B)`,
    #     which is B (the batch size), not 0. Generate then thinks the cache already has
    #     B tokens cached and routes the first forward to the decode branch with the full
    #     prompt as input, instead of doing prefill.
    #  3. The model's prefill path reads `cache_params.conv_kernel_size`, but the upstream
    #     __init__ only uses it as a local variable and never stores it on self.
    _OrigCache = nemotron_mod.HybridMambaAttentionDynamicCache
    class HybridMambaAttentionDynamicCache(_OrigCache):
        def __init__(self, config, batch_size, dtype=torch.float16, device=None):
            super().__init__(config, batch_size, dtype=dtype, device=device)
            intermediate_size = config.mamba_num_heads * config.mamba_head_dim
            ssm_state_size = config.ssm_state_size
            conv_kernel_size = config.conv_kernel
            conv_dim = intermediate_size + 2 * config.n_groups * ssm_state_size
            self.conv_kernel_size = conv_kernel_size
            self.intermediate_size = intermediate_size
            self.ssm_state_size = ssm_state_size
            self.conv_dim = conv_dim
            for i in range(config.num_hidden_layers):
                if self.hybrid_override_pattern[i] == "M":
                    self.conv_states[i] = torch.zeros(
                        batch_size, conv_dim, conv_kernel_size,
                        device=device, dtype=dtype,
                    )

        def get_seq_length(self, layer_idx=0):
            if not self.transformer_layers:
                return 0
            layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
            if len(self.key_cache) <= layer_idx:
                return 0
            kc = self.key_cache[layer_idx]
            if kc.numel() == 0:
                return 0
            return kc.shape[-2]

        def update_conv_state(self, layer_idx, new_conv_state, cache_init=False):
            target = self.conv_states[layer_idx]
            if cache_init:
                self.conv_states[layer_idx] = new_conv_state.to(target.device)
            else:
                rolled = target.roll(shifts=-1, dims=-1)
                rolled[:, :, -1] = new_conv_state[:, 0, :].to(target.device)
                self.conv_states[layer_idx] = rolled
            return self.conv_states[layer_idx]

        def update_ssm_state(self, layer_idx, new_ssm_state):
            target = self.ssm_states[layer_idx]
            self.ssm_states[layer_idx] = new_ssm_state.to(target.device)
            return self.ssm_states[layer_idx]

        def reset(self):
            for t in self.conv_states:
                t.zero_()
            for t in self.ssm_states:
                t.zero_()

    nemotron_mod.HybridMambaAttentionDynamicCache = HybridMambaAttentionDynamicCache
    print("Patched HybridMambaAttentionDynamicCache (conv_dim + get_seq_length fixes)")
    

    # ── Manually add lm_head LoRA (Unsloth drops it for MoE) ─────────
    _causal_lm = model
    while hasattr(_causal_lm, "model"):
        _causal_lm = _causal_lm.model

    # ── Patch prepare_inputs_for_generation: NemotronH forward expects ──
    # `cache_params=`, but its prepare_inputs_for_generation puts the cache
    # under `past_key_values=` in model_inputs (a bug in the upstream remote
    # code — the docstring even says "uses cache_params" but the dict key
    # wasn't renamed). Without this rewrite, generate's decode steps run
    # without cache reuse and every step re-prefills the full prefix.
    _orig_prepare = _causal_lm.prepare_inputs_for_generation
    def _patched_prepare_inputs(*a, **kw):
        inputs = _orig_prepare(*a, **kw)
        if "past_key_values" in inputs:
            inputs["cache_params"] = inputs.pop("past_key_values")
        return inputs
    _causal_lm.prepare_inputs_for_generation = _patched_prepare_inputs
    print("Patched prepare_inputs_for_generation: past_key_values → cache_params")

    # transformers 4.57+ validates model_kwargs more strictly, rejecting `past_key_values`
    # for models that use `cache_params` (NemotronH). The prepare_inputs patch above handles
    # the rename correctly; we just need to stop the pre-rename validation from crashing.
    _causal_lm._validate_model_kwargs = lambda model_kwargs: None
    print("Patched _validate_model_kwargs: disabled (NemotronH uses cache_params)")

    _lm_head = _causal_lm.lm_head
    if not isinstance(_lm_head, LoraLinear):
        _cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)
        model.base_model._create_and_replace(
            _cfg,
            "default",
            target=_lm_head,
            target_name="lm_head",
            parent=_causal_lm,
        )
        print("Manually added LoRA to lm_head")
    else:
        print("lm_head already has LoRA")

    # ── Cast LoRA params to fp32 (base bf16 except MoE router stays fp32) ──
    for name, param in model.named_parameters():
        if ".lora_" in name:
            param.data = param.data.to(torch.float32)

    for name, param in model.named_parameters():
        if ".lora_" in name:
            assert param.dtype == torch.float32, (
                f"LoRA param {name} expected fp32, got {param.dtype}"
            )
            continue
        # NemotronHTopkRouter.weight + e_score_correction_bias are kept fp32 by design
        # (see modeling_nemotron_h.py: _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
        # and the per-forward .type(torch.float32) cast on the gate weight).
        is_router = ".mixer.gate." in name
        if is_router:
            assert param.dtype == torch.float32, (
                f"param {name} expected fp32, got {param.dtype}"
            )
            continue
        assert param.dtype == torch.bfloat16, (
            f"param {name} expected bf16, got {param.dtype}"
        )

    print("Verified: LoRA params fp32, base params bf16 (MoE router fp32)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {trainable:,} trainable / {total:,} total parameters")

    # ── vLLM swap helpers ─────────────────────────────────────────────
    # During vLLM rollout the HF model is idle; we offload its weights to
    # CPU to free ~60GB for vLLM. Timings are recorded and surfaced per step.
    _vllm_device = next(model.parameters()).device  # captured before any .to() call

    def _hf_to_cpu() -> float:
        t = time.time()
        model.to("cpu")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return time.time() - t

    def _hf_to_gpu() -> float:
        t = time.time()
        model.to(_vllm_device)
        torch.cuda.synchronize()
        return time.time() - t

    def _save_adapter_for_vllm(step_id: int) -> str:
        """Serialize current LoRA adapter to a per-step temp dir for LoRARequest."""
        out = os.path.join(VLLM_ADAPTER_TMP_DIR, f"step_{step_id}")
        if os.path.isdir(out):
            shutil.rmtree(out)
        os.makedirs(out, exist_ok=True)
        model.save_pretrained(out)
        # Rename lm_head keys to match the base model's module path so vLLM
        # can resolve them. Mirrors the rename applied at final checkpoint save.
        from safetensors.torch import load_file as _lf, save_file as _sf
        _st = os.path.join(out, "adapter_model.safetensors")
        _tensors = _lf(_st)
        _renamed = {
            k.replace("base_model.model.lm_head.", "base_model.model.backbone.lm_head."): v
            for k, v in _tensors.items()
        }
        _sf(_renamed, _st)
        return out

    # ── Patch forward with Cut Cross-Entropy ─────────────────────────
    # Same trick as SFT: when `labels` is provided, compute per-token
    # −log p(label) via linear_cross_entropy (no logits materialization)
    # and stash it on the model so the GRPO loop can negate it to get
    # per-token logprobs.
    #
    # Crucially, the GRPO rollout uses `model.generate`, which calls
    # forward with `labels=None` and needs real logits back. So we keep
    # a reference to the original forward and dispatch to it on the
    # generation path.
    _base = model
    while hasattr(_base, "model"):
        _base = _base.model
    _orig_forward = _base.forward

    def _patched_causal_forward(
        input_ids=None, attention_mask=None, labels=None, **kwargs
    ):
        if labels is None:
            # Generation / no-loss path → original forward (returns logits)
            return _orig_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                **kwargs,
            )
        backbone_out = _base.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{
                k: v
                for k, v in kwargs.items()
                if k in ("position_ids", "past_key_values", "use_cache")
            },
        )
        hidden_states = backbone_out[0]
        lm_head = _base.lm_head
        base_w = lm_head.base_layer.weight
        lora_A = lm_head.lora_A["default"].weight
        lora_B = lm_head.lora_B["default"].weight
        scaling = lm_head.scaling["default"]
        lm_weight = base_w + scaling * lora_B @ lora_A
        per_token_neg_logp = linear_cross_entropy(
            hidden_states, lm_weight, labels, reduction="none"
        )
        loss = per_token_neg_logp.mean()
        model._cached_per_token_neg_logp = per_token_neg_logp  # type: ignore[attr-defined]
        return loss

    _base.forward = _patched_causal_forward
    print("Patched CausalLM.forward with CCE (training path); generate uses original forward")

    # ── Load adapter weights (unless RESET_WEIGHTS) ──────────────────
    if RESET_WEIGHTS:
        print(
            "RESET_WEIGHTS=True — skipping pretrained adapter load; using fresh LoRA init"
        )
        loaded = 0
        adapter_weights: dict = {}
    else:
        print(f"Loading adapter from {ADAPTER_SRC}...")
        from peft import load_peft_weights

        adapter_weights = load_peft_weights(ADAPTER_SRC)

        model_sd = model.state_dict()
        new_sd: dict = {}
        loaded = 0
        for ak, av in adapter_weights.items():
            if ak in model_sd:
                new_sd[ak] = av
                loaded += 1
                continue
            ak_with_default = ak.replace(
                ".lora_A.weight", ".lora_A.default.weight"
            ).replace(".lora_B.weight", ".lora_B.default.weight")
            if ak_with_default in model_sd:
                new_sd[ak_with_default] = av
                loaded += 1
                continue
            ak_lm = ak.replace(".backbone.lm_head.", ".lm_head.")
            ak_lm_default = ak_lm.replace(
                ".lora_A.weight", ".lora_A.default.weight"
            ).replace(".lora_B.weight", ".lora_B.default.weight")
            if ak_lm_default in model_sd:
                new_sd[ak_lm_default] = av
                loaded += 1
                continue

        model.load_state_dict(new_sd, strict=False)
        assert loaded == len(adapter_weights), (
            f"Not all adapter weights loaded: {loaded}/{len(adapter_weights)}"
        )
        print(f"  Loaded {loaded}/{len(adapter_weights)} weights into model")

    # ── Freeze all LoRA params except in_proj (if IN_PROJ_ONLY) ──
    print(f"{IN_PROJ_ONLY=}")
    if IN_PROJ_ONLY:
        for name, param in model.named_parameters():
            if param.requires_grad and ".in_proj." not in name:
                param.requires_grad = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  {trainable_params:,} trainable / {frozen_params:,} frozen")

    # ── MoE tied-weight params (Tinker convention) ───────────────────
    # gate_up_proj / up_proj / w1 / gate_proj  → tie A (input/hidden side)
    # down_proj / w2                            → tie B (output/hidden side)
    moe_tied_params: list[torch.Tensor] = []
    if MOE_TIE_WEIGHTS:
        w1_proj_names = ("gate_up_proj", "up_proj", "gate_proj", ".w1.")
        w2_proj_names = ("down_proj", ".w2.")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if ".experts." not in name or ".lora_" not in name:
                continue
            is_w1 = any(p in name for p in w1_proj_names)
            is_w2 = any(p in name for p in w2_proj_names)
            is_A = ".lora_A." in name
            is_B = ".lora_B." in name
            should_tie = (is_w1 and is_A) or (is_w2 and is_B)
            if not should_tie:
                continue
            if param.dim() < 2 or param.shape[0] <= 1:
                continue
            moe_tied_params.append(param)

        def _tie_param_init() -> None:
            with torch.no_grad():
                for p in moe_tied_params:
                    mean = p.data.mean(dim=0, keepdim=True)
                    p.data.copy_(mean.expand_as(p.data))

        def _tie_grads() -> None:
            # Sum (not mean) across the expert dim: dL/dW = Σ_i dL/dW_i.
            with torch.no_grad():
                for p in moe_tied_params:
                    if p.grad is None:
                        continue
                    grad_sum = p.grad.sum(dim=0, keepdim=True)
                    p.grad.copy_(grad_sum.expand_as(p.grad))

        print(f"MoE weight tying: {len(moe_tied_params)} params identified for tying")
        if moe_tied_params:
            print(f"  example shapes: {[tuple(p.shape) for p in moe_tied_params[:3]]}")
        _tie_param_init()
    else:
        def _tie_grads() -> None:
            pass

    

    # ── Build prompt list from train.csv (id, prompt, answer) ────────
    examples: list[dict] = []
    with open(TRAIN_CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_msg = row["prompt"] + SYSTEM_INSTRUCTION
            try:
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_msg}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt_text = user_msg
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            if len(prompt_ids) > MAX_PROMPT_LEN:
                prompt_ids = prompt_ids[:MAX_PROMPT_LEN]
            examples.append(
                {
                    "problem_id": row["id"],
                    "prompt_ids": prompt_ids,
                    "ground_truth": row["answer"],
                }
            )
    print(f"Loaded {len(examples)} prompts from {TRAIN_CSV_PATH}")

    # ── Reward helpers ───────────────────────────────────────────────
    def _normalize_answer(s: str) -> str:
        s = (s or "").strip()
        try:
            f = float(s)
            return str(int(f)) if f == int(f) else str(f)
        except (ValueError, OverflowError):
            return s

    _BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}", re.DOTALL)
    _BOXED_FALLBACK_RE = re.compile(r"boxed\{([^}]*)\}", re.DOTALL)

    def _extract_boxed(content: str):
        m = _BOXED_RE.search(content) or _BOXED_FALLBACK_RE.search(content)
        return m.group(1).strip() if m else None

    # ── Reward functions (uniform signature: batch -> list[float]) ───
    # Each takes a list of completion records (dicts) and returns one
    # float per record. Uniform shape is what makes the (fn, weight)
    # registry below work for arbitrary additional rewards.
    def cosine_reward(batch):
        """Token-length-aware cosine-scaled accuracy reward (Light-R1)."""
        rewards = []
        for rec in batch:
            clen = rec["comp_token_len"]
            progress = min(clen / max(MAX_NEW_TOKENS, 1), 1.0)
            cos_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            if rec["boxed_correct"]:
                r = 0.1 + 0.9 * cos_scale
            elif rec["boxed_extracted"] is not None:
                r = -0.1 - 0.9 * (1.0 - cos_scale)
            else:
                r = -0.5 * progress
            rewards.append(r)
        return rewards

    def think_format_reward(batch):
        """TRL's think_format_reward, batched. Returns 1.0 for valid <think>…</think>."""
        completions = [[{"role": "assistant", "content": r["comp_text"]}] for r in batch]
        return _trl_think_format_reward(completions=completions)

    # Drop in additional (fn, weight) tuples to extend the training signal.
    register_rewards = [
        (cosine_reward,       1.0),
        (think_format_reward, 0.15),
    ]

    # ── GRPO training loop ───────────────────────────────────────────
    gc.collect()
    torch.cuda.empty_cache()

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # ── W&B (offline by default; Kaggle has no internet) ─────────────
    wandb_run = None
    if USE_WANDB:
        os.environ["WANDB_MODE"] = WANDB_MODE
        os.environ["WANDB_DIR"] = WANDB_DIR
        os.makedirs(WANDB_DIR, exist_ok=True)
        import wandb

        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            mode=WANDB_MODE,
            dir=WANDB_DIR,
            config={
                "lora_rank": LORA_RANK,
                "lr": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "group_size": GROUP_SIZE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "num_steps": NUM_STEPS,
                "register_rewards": [(fn.__name__, w) for fn, w in register_rewards],
            },
        )

    rng = random.Random(0)
    prompt_indices = list(range(len(examples)))
    rng.shuffle(prompt_indices)

    training_log: list[str] = []

    def _log(msg: str) -> None:
        print(msg, flush=True)
        training_log.append(msg)

    max_steps_avail = len(examples) // BATCH_SIZE
    num_steps = NUM_STEPS
    if num_steps > max_steps_avail:
        _log(
            f"WARNING: NUM_STEPS={NUM_STEPS} exceeds available {max_steps_avail} "
            f"({len(examples)} prompts // {BATCH_SIZE}). Clamping."
        )
        num_steps = max_steps_avail

    _log(
        f"GRPO: {num_steps} steps, batch_size={BATCH_SIZE}, group_size={GROUP_SIZE}, "
        f"micro_batch_size={MICRO_BATCH_SIZE}, lr={LEARNING_RATE}, "
        f"clip_eps={CLIP_EPS}, ppo_epochs={PPO_EPOCHS}"
    )

    optimizer: torch.optim.AdamW | None = None
    step = 0

    try:
        for batch_start in range(0, len(prompt_indices), BATCH_SIZE):
            if step >= num_steps:
                break
            batch_idx = prompt_indices[batch_start : batch_start + BATCH_SIZE]
            if len(batch_idx) < BATCH_SIZE:
                break  # drop incomplete tail batch so group/advantage shapes stay clean
            batch = [examples[i] for i in batch_idx]

            # Reset per-step so logged peak reflects this step only,
            # not the global high-water mark since process start.
            torch.cuda.reset_peak_memory_stats()

            # ── Phase A: rollout ─────────────────────────────────────────
            # Build flattened list: each prompt repeated GROUP_SIZE times.
            flat_examples: list[dict] = [ex for ex in batch for _ in range(GROUP_SIZE)]
            batch_completions: list[dict] = []
            t_roll = time.time()

            if USE_VLLM_ROLLOUT:
                # ── vLLM path ────────────────────────────────────────────
                # (a) Serialize the current LoRA adapter to disk
                save_dt = wake_dt = gen_dt = sleep_dt = hf_off_dt = hf_on_dt = 0.0
                cache_creation_dt = post_dt = 0.0  # not used in this path

                _t = time.time()
                _adapter_dir = _save_adapter_for_vllm(step)
                save_dt = time.time() - _t

                # (b) Offload HF model to CPU to free GPU for vLLM
                hf_off_dt = _hf_to_cpu()

                # (c) Wake vLLM and generate all completions in one call
                _t = time.time()
                vllm_engine.wake_up()
                wake_dt = time.time() - _t

                _sampling = SamplingParams(
                    temperature=ROLLOUT_TEMPERATURE,
                    top_p=ROLLOUT_TOP_P,
                    max_tokens=MAX_NEW_TOKENS,
                    stop_token_ids=[tokenizer.eos_token_id],
                )
                _lora_req = LoRARequest(
                    lora_name=f"grpo_step_{step}",
                    lora_int_id=step + 1,  # 0 is reserved by vLLM
                    lora_path=_adapter_dir,
                )

                _t = time.time()
                _vllm_outputs = vllm_engine.generate(
                    prompt_token_ids=[ex["prompt_ids"] for ex in flat_examples],
                    sampling_params=_sampling,
                    lora_request=_lora_req,
                    use_tqdm=False,
                )
                gen_dt = time.time() - _t

                # (d) Sleep vLLM and restore HF model on GPU
                _t = time.time()
                vllm_engine.sleep(level=2)
                sleep_dt = time.time() - _t
                hf_on_dt = _hf_to_gpu()

                # (e) Convert vLLM outputs → batch_completions schema
                for ex, _vout in zip(flat_examples, _vllm_outputs):
                    comp_ids = list(_vout.outputs[0].token_ids)
                    full_ids = list(ex["prompt_ids"]) + comp_ids
                    comp_text = tokenizer.decode(comp_ids, skip_special_tokens=True)
                    boxed = _extract_boxed(comp_text)
                    batch_completions.append({
                        "prompt_len":      len(ex["prompt_ids"]),
                        "full_ids":        full_ids,
                        "comp_ids":        comp_ids,
                        "comp_text":       comp_text,
                        "gt":              ex["ground_truth"],
                        "comp_token_len":  len(comp_ids),
                        "boxed_extracted": boxed,
                        "boxed_correct":   boxed is not None
                                           and _normalize_answer(boxed)
                                               == _normalize_answer(ex["ground_truth"]),
                    })

                shutil.rmtree(_adapter_dir, ignore_errors=True)

            else:
                # ── Legacy HF generate path ──────────────────────────────
                # Tile prompts in micro-batches with left-padding (safe for
                # NemotronH: no positional embeddings, RMSNorm only).
                FastLanguageModel.for_inference(model)
                cache_creation_dt = 0.0
                gen_dt = 0.0
                post_dt = 0.0
                save_dt = wake_dt = sleep_dt = hf_off_dt = hf_on_dt = 0.0

                with torch.no_grad():
                    for chunk_start in range(0, len(flat_examples), MICRO_ROLLOUT_BATCH):
                        chunk = flat_examples[chunk_start : chunk_start + MICRO_ROLLOUT_BATCH]
                        n_chunk = len(chunk)

                        max_plen = max(len(ex["prompt_ids"]) for ex in chunk)
                        prompt_t = torch.full(
                            (n_chunk, max_plen), pad_id, dtype=torch.long, device=device
                        )
                        for i, ex in enumerate(chunk):
                            pids = ex["prompt_ids"]
                            prompt_t[i, max_plen - len(pids) :] = torch.tensor(
                                pids, dtype=torch.long, device=device
                            )
                        attn = (prompt_t != pad_id).long()

                        _t = time.time()
                        cache = HybridMambaAttentionDynamicCache(
                            config=model.config,
                            batch_size=n_chunk,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        torch.cuda.synchronize()
                        cache_creation_dt += time.time() - _t

                        _t = time.time()
                        out = model.generate(
                            input_ids=prompt_t,
                            attention_mask=attn,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=True,
                            temperature=ROLLOUT_TEMPERATURE,
                            top_p=ROLLOUT_TOP_P,
                            use_cache=True,
                            past_key_values=cache,
                            pad_token_id=pad_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                        torch.cuda.synchronize()
                        gen_dt += time.time() - _t

                        _t = time.time()
                        for i, ex in enumerate(chunk):
                            full_row = out[i].tolist()
                            pad_prefix = max_plen - len(ex["prompt_ids"])
                            full = full_row[pad_prefix:]
                            while len(full) > len(ex["prompt_ids"]) and full[-1] == pad_id:
                                full.pop()
                            comp = full[len(ex["prompt_ids"]) :]
                            comp_text = tokenizer.decode(comp, skip_special_tokens=True)
                            boxed = _extract_boxed(comp_text)
                            batch_completions.append({
                                "prompt_len":      len(ex["prompt_ids"]),
                                "full_ids":        full,
                                "comp_ids":        comp,
                                "comp_text":       comp_text,
                                "gt":              ex["ground_truth"],
                                "comp_token_len":  len(comp),
                                "boxed_extracted": boxed,
                                "boxed_correct":   boxed is not None
                                                   and _normalize_answer(boxed)
                                                       == _normalize_answer(ex["ground_truth"]),
                            })
                        post_dt += time.time() - _t

                # Release fragmented blocks before Phase C/D's different allocation pattern.
                torch.cuda.empty_cache()

            _t = time.time()
            # ── Reward computation: serial loop over registry ────────────
            reward_outputs = {
                fn.__name__: fn(batch_completions) for fn, _ in register_rewards
            }

            # ── Weighted sum into the per-record training signal ─────────
            n_completions = len(batch_completions)
            total_rewards_list = [0.0] * n_completions
            for fn, w in register_rewards:
                rs = reward_outputs[fn.__name__]
                for i, r in enumerate(rs):
                    total_rewards_list[i] += w * r

            # ── Build rollout_records used by Phase B onwards ────────────
            rollout_records: list[dict] = [
                {
                    "prompt_len":      rec["prompt_len"],
                    "full_ids":        rec["full_ids"],
                    "reward":          total_rewards_list[i],
                    "cosine_reward":   reward_outputs["cosine_reward"][i],
                    "think_reward":    reward_outputs["think_format_reward"][i],
                    "boxed_correct":   rec["boxed_correct"],
                    "boxed_present":   rec["boxed_extracted"] is not None,
                    "comp_token_len":  rec["comp_token_len"],
                }
                for i, rec in enumerate(batch_completions)
            ]

            # ── Phase B: group-relative advantages ───────────────────────
            rewards = torch.tensor(
                [r["reward"] for r in rollout_records], dtype=torch.float32
            )
            rewards_g = rewards.view(BATCH_SIZE, GROUP_SIZE)
            adv_g = (rewards_g - rewards_g.mean(dim=1, keepdim=True)) / (
                rewards_g.std(dim=1, keepdim=True) + ADV_NORM_EPS
            )
            advantages = adv_g.view(-1).to(device)  # [BATCH_SIZE * GROUP_SIZE]
            reward_dt = time.time() - _t

            # ── Phase C: cache "old" logprobs (no grad, single batched forward) ──
            # Process rollout_records in micro-batches of MICRO_BATCH_SIZE to
            # keep peak VRAM predictable (same chunk size as Phase D training).
            _t = time.time()
            old_logps_per_record: list[torch.Tensor] = []
            with torch.no_grad():
                for mb_start in range(0, len(rollout_records), MICRO_BATCH_SIZE):
                    mb_recs = rollout_records[mb_start : mb_start + MICRO_BATCH_SIZE]
                    n_mb = len(mb_recs)
                    max_flen = max(len(r["full_ids"]) for r in mb_recs)
                    full_ids_c = torch.full(
                        (n_mb, max_flen), pad_id, dtype=torch.long, device=device
                    )
                    for i, r in enumerate(mb_recs):
                        fids = r["full_ids"]
                        full_ids_c[i, : len(fids)] = torch.tensor(
                            fids, dtype=torch.long, device=device
                        )
                    attn_c = (full_ids_c != pad_id).long()
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        model(
                            input_ids=full_ids_c[:, :-1],
                            attention_mask=attn_c[:, :-1],
                            labels=full_ids_c[:, 1:],
                            use_cache=False,
                        )
                    neg_logp_mb = model._cached_per_token_neg_logp  # type: ignore[attr-defined]
                    model._cached_per_token_neg_logp = None  # type: ignore[attr-defined]
                    for i, r in enumerate(mb_recs):
                        L = len(r["full_ids"]) - 1
                        old_logps_per_record.append((-neg_logp_mb[i, :L].detach()).clone())

            torch.cuda.synchronize()
            old_logps_dt = time.time() - _t

            torch.cuda.empty_cache()

            # ── Phase D: training forward (grad ckpt ON, KV cache OFF) ───
            _t = time.time()
            FastLanguageModel.for_training(model)
            if optimizer is None:
                # 8-bit AdamW: ~4x smaller optimizer state vs fp32 AdamW
                # (here ~1.7 GB vs ~7 GB for 888M trainable params).
                import bitsandbytes as bnb

                optimizer = bnb.optim.AdamW8bit(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=LEARNING_RATE,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=0.0,
                )
            lr = LEARNING_RATE * (1 - step / num_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            for ppo_epoch in range(PPO_EPOCHS):
                loss_sum_acc = 0.0
                token_count_acc = 0.0
                grad_norm = 0.0
                n = len(rollout_records)
                n_accum = math.ceil(n / MICRO_BATCH_SIZE)
                for mb_start in range(0, n, MICRO_BATCH_SIZE):
                    mb = rollout_records[mb_start : mb_start + MICRO_BATCH_SIZE]
                    mb_old = old_logps_per_record[mb_start : mb_start + MICRO_BATCH_SIZE]
                    mb_adv = advantages[mb_start : mb_start + MICRO_BATCH_SIZE]
                    n_mb = len(mb)

                    max_len = max(len(r["full_ids"]) for r in mb)
                    input_ids = torch.full(
                        (n_mb, max_len), pad_id, dtype=torch.long, device=device
                    )
                    for i, r in enumerate(mb):
                        input_ids[i, : len(r["full_ids"])] = torch.tensor(
                            r["full_ids"], dtype=torch.long
                        )
                    attn = (input_ids != pad_id).long()

                    t0 = time.time()
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        model(
                            input_ids=input_ids[:, :-1],
                            attention_mask=attn[:, :-1],
                            labels=input_ids[:, 1:],
                            use_cache=False,  # required when grad checkpointing is on
                        )
                        neg_logp_new = model._cached_per_token_neg_logp  # type: ignore[attr-defined]
                        model._cached_per_token_neg_logp = None  # type: ignore[attr-defined]
                        new_logp = -neg_logp_new

                        # Per-token completion mask (only completion tokens contribute)
                        comp_mask = torch.zeros_like(new_logp)
                        for i, r in enumerate(mb):
                            comp_start = r["prompt_len"] - 1
                            comp_end = len(r["full_ids"]) - 1
                            comp_mask[i, comp_start:comp_end] = 1.0

                        # Pad/align old logp to current micro-batch shape
                        old_logp = torch.zeros_like(new_logp)
                        for i, ol in enumerate(mb_old):
                            L = min(ol.shape[0], new_logp.shape[1])
                            old_logp[i, :L] = ol[:L]

                        log_ratio = new_logp - old_logp.detach()
                        ratio = torch.exp(log_ratio)
                        adv_t = mb_adv.view(-1, 1).expand_as(ratio)
                        surr1 = ratio * adv_t
                        surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t
                        per_token_obj = torch.min(surr1, surr2)
                        masked = per_token_obj * comp_mask
                        denom = comp_mask.sum().clamp(min=1.0)
                        loss = -(masked.sum() / denom)

                    (loss / n_accum).backward()
                    loss_sum_acc += loss.item() * comp_mask.sum().item()
                    token_count_acc += comp_mask.sum().item()

                    t_end = time.time()
                    peak_gb = torch.cuda.max_memory_allocated() / 1e9
                    mem_gb = torch.cuda.memory_allocated() / 1e9
                    mb_idx = mb_start // MICRO_BATCH_SIZE
                    print(
                        f"    [epoch {ppo_epoch}] micro-batch {mb_idx}: {n_mb} seqs, "
                        f"max_len={max_len}, wall={t_end - t0:.1f}s, "
                        f"peak={peak_gb:.1f}GB, mem={mem_gb:.1f}GB"
                    )
                    del loss, new_logp, ratio, surr1, surr2, per_token_obj, masked

                _tie_grads()  # average MoE expert grads before clip+step
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()

            torch.cuda.synchronize()
            train_dt = time.time() - _t

            step += 1
            mean_reward = rewards.mean().item()
            loss_mean = loss_sum_acc / token_count_acc if token_count_acc > 0 else 0.0

            n_rec = len(rollout_records)
            pct_boxed_correct = 100.0 * sum(r["boxed_correct"] for r in rollout_records) / n_rec
            pct_format_compliant = 100.0 * sum(
                r["boxed_present"] and r["think_reward"] >= 1.0 for r in rollout_records
            ) / n_rec
            mean_comp_len_tok = sum(r["comp_token_len"] for r in rollout_records) / n_rec
            mean_cosine_r = sum(r["cosine_reward"] for r in rollout_records) / n_rec
            mean_think_r = sum(r["think_reward"] for r in rollout_records) / n_rec
            grad_norm_v = float(grad_norm)
            rollout_seconds = time.time() - t_roll

            metrics = {
                "step":                       step,
                "rollout_seconds":            rollout_seconds,
                "mean_reward":                mean_reward,
                "mean_cosine_reward":         mean_cosine_r,
                "mean_think_reward":          mean_think_r,
                "boxed_correct_pct":          pct_boxed_correct,
                "format_compliance_pct":      pct_format_compliant,
                "mean_completion_len_tokens": mean_comp_len_tok,
                "loss":                       loss_mean,
                "grad_norm":                  grad_norm_v,
                "lr":                         lr,
            }

            _log(
                f"  step {step}/{num_steps}: rollout={rollout_seconds:.1f}s, "
                f"mean_reward={mean_reward:.4f} (cos={mean_cosine_r:.4f}, think={mean_think_r:.4f}), "
                f"boxed_correct%={pct_boxed_correct:.1f}, format_compliance%={pct_format_compliant:.1f}, "
                f"mean_comp_len_tok={mean_comp_len_tok:.1f}, "
                f"loss={loss_mean:.6f}, grad_norm={grad_norm_v:.4f}, lr={lr:.2e}"
            )
            if USE_VLLM_ROLLOUT:
                _log(
                    f"    times: save_adapter={save_dt:.2f}s, hf_off={hf_off_dt:.2f}s, "
                    f"vllm_wake={wake_dt:.2f}s, gen={gen_dt:.1f}s, vllm_sleep={sleep_dt:.2f}s, "
                    f"hf_on={hf_on_dt:.2f}s, reward={reward_dt:.2f}s, "
                    f"old_logps={old_logps_dt:.1f}s, train={train_dt:.1f}s"
                )
            else:
                _log(
                    f"    times: cache_creation={cache_creation_dt:.2f}s, gen={gen_dt:.1f}s, "
                    f"post={post_dt:.2f}s, reward={reward_dt:.2f}s, "
                    f"old_logps={old_logps_dt:.1f}s, train={train_dt:.1f}s"
                )
            if wandb_run is not None:
                if USE_VLLM_ROLLOUT:
                    wandb_run.log(
                        {
                            **metrics,
                            "time/vllm_save_adapter": save_dt,
                            "time/vllm_hf_off":       hf_off_dt,
                            "time/vllm_wake":         wake_dt,
                            "time/vllm_gen":          gen_dt,
                            "time/vllm_sleep":        sleep_dt,
                            "time/vllm_hf_on":        hf_on_dt,
                            "time/reward":            reward_dt,
                            "time/old_logps":         old_logps_dt,
                            "time/train":             train_dt,
                        },
                        step=step,
                    )
                else:
                    wandb_run.log(
                        {
                            **metrics,
                            "time/cache_creation": cache_creation_dt,
                            "time/gen":            gen_dt,
                            "time/post":           post_dt,
                            "time/reward":         reward_dt,
                            "time/old_logps":      old_logps_dt,
                            "time/train":          train_dt,
                        },
                        step=step,
                    )

        print(
            f"\nTraining complete. Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
        )

        # ── Save adapter + rename lm_head keys (matches SFT submission format) ──
        from safetensors.torch import load_file, save_file

        os.makedirs(save_dir, exist_ok=True)
        for _f in os.listdir(save_dir):
            if _f.startswith("adapter"):
                os.remove(os.path.join(save_dir, _f))
        model.save_pretrained(save_dir)
        st_path = os.path.join(save_dir, "adapter_model.safetensors")
        tensors = load_file(st_path)
        renamed = {
            k.replace("base_model.model.lm_head.", "base_model.model.backbone.lm_head."): v
            for k, v in tensors.items()
        }
        save_file(renamed, st_path)

        # ── Clean unsloth compiled cache ─────────────────────────────────
        _ucache = "unsloth_compiled_cache"
        if os.path.isdir(_ucache):
            shutil.rmtree(_ucache)

        # ── Package adapter as submission.zip ────────────────────────────
        adapter_files = [f for f in os.listdir(save_dir) if f.startswith("adapter")]
        SUBMISSION_ZIP = "submission.zip"
        with zipfile.ZipFile(SUBMISSION_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in adapter_files:
                zf.write(os.path.join(save_dir, fname), fname)
        for fname in adapter_files:
            os.remove(os.path.join(save_dir, fname))
        print(f"Wrote {SUBMISSION_ZIP}")

        # ── Persist training log alongside submission ────────────────────
        with open("training_log.txt", "w") as f:
            f.write("\n".join(training_log) + "\n")
        print("Training complete.")
    finally:
        if vllm_engine is not None:
            try:
                vllm_engine.sleep(level=2)
            except Exception:
                pass
            del vllm_engine
            torch.cuda.empty_cache()
        if wandb_run is not None:
            wandb_run.finish()



## Cell 4:def smoke_test() -> None:
def smoke_test() -> None:
    """End-to-end pipeline check with tiny config.

    Snapshots the config globals that control step count and batch shape,
    overrides them with minimal values, runs run_training(), and asserts
    that all expected artifacts (submission.zip with renamed lm_head keys,
    training_log.txt with a step line, W&B offline run dir) were produced.
    Originals are restored in `finally` so a follow-up full run picks up
    the unchanged config.
    """
    import glob
    import os
    import zipfile

    from safetensors.torch import load_file

    overrides = {
        "NUM_STEPS":          1,
        "BATCH_SIZE":         2,
        "GROUP_SIZE":         2,
        "MICRO_BATCH_SIZE":   2,
        "MICRO_ROLLOUT_BATCH": 4,  # equals BATCH_SIZE*GROUP_SIZE → single generate call
        "MAX_PROMPT_LEN":     256,
        "MAX_NEW_TOKENS":     128,
        "PPO_EPOCHS":         1,
        "WANDB_RUN_NAME":     "smoke-test",
        "USE_VLLM_ROLLOUT":   True,  # explicitly exercise vLLM path; flip to False to test HF fallback
    }
    g = globals()
    snapshot = {k: g[k] for k in overrides}
    print(f"smoke test: overriding {list(overrides)}")
    try:
        for k, v in overrides.items():
            g[k] = v
        run_training()

        # ── Assert artifacts ─────────────────────────────────────────
        sub_zip = "submission.zip"
        assert os.path.isfile(sub_zip), f"{sub_zip} missing"
        assert os.path.getsize(sub_zip) > 0, f"{sub_zip} is empty"

        unzip_dir = "/tmp/smoke_test_unzipped"
        os.makedirs(unzip_dir, exist_ok=True)
        with zipfile.ZipFile(sub_zip) as zf:
            names = zf.namelist()
            assert "adapter_model.safetensors" in names, (
                f"adapter_model.safetensors missing from zip; got {names}"
            )
            assert "adapter_config.json" in names, (
                f"adapter_config.json missing from zip; got {names}"
            )
            zf.extractall(unzip_dir)

        tensors = load_file(os.path.join(unzip_dir, "adapter_model.safetensors"))
        renamed_keys = [
            k for k in tensors if k.startswith("base_model.model.backbone.lm_head.")
        ]
        assert renamed_keys, (
            "no renamed lm_head keys in adapter_model.safetensors; "
            f"sample keys: {list(tensors)[:3]}"
        )

        log_path = "training_log.txt"
        assert os.path.isfile(log_path), f"{log_path} missing"
        with open(log_path) as f:
            log_text = f.read()
        assert "step 1/1:" in log_text, (
            f"'step 1/1:' not found in {log_path}; first 200 chars: {log_text[:200]!r}"
        )
        if USE_VLLM_ROLLOUT:
            assert "vllm_wake=" in log_text, (
                f"vLLM timer line missing from {log_path}; first 400 chars: {log_text[:400]!r}"
            )

        if USE_WANDB:
            assert os.path.isdir(WANDB_DIR), f"{WANDB_DIR} missing"
            wandb_files = glob.glob(
                os.path.join(WANDB_DIR, "**", "*.wandb"), recursive=True
            )
            assert wandb_files, f"no *.wandb files under {WANDB_DIR}"

        print("smoke test OK")
    finally:
        for k, v in snapshot.items():
            g[k] = v
        print("smoke test: config restored")


# Uncomment to run the smoke test instead of the full training cell:
smoke_test()