# Nemotron Finetuning Pipeline

This document specifies the end-to-end LoRA finetuning pipeline for
`Nemotron-3-Nano-30B-A3B` (a 30B MoE + Mamba hybrid) that produced the
LB-0.86 submission. The entire pipeline lives in a single notebook
(`end-to-end-finetuning-for-lb-0-86-custom-repo.ipynb`) that detects its
runtime context and adapts accordingly — the same source file runs as a Kaggle
kernel, a Modal worker, or a Modal launcher with no code changes.

---

## 1. Execution Environments

Environment is detected at import time via two `os.environ` flags:

| role | detection condition | what fires |
|---|---|---|
| **Kaggle kernel** | `KAGGLE_KERNEL_RUN_TYPE` in env | `run_training()` called at module level (cell-6) |
| **Modal worker** | `MODAL_TASK_ID` in env | Modal runtime calls `train_remote()` → `run_training()` |
| **Modal launcher** | neither flag present | `main()` calls `train_remote.remote()` and returns |

```
IS_KAGGLE         = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
IS_MODAL_WORKER   = "MODAL_TASK_ID" in os.environ
IS_MODAL_LAUNCHER = not IS_KAGGLE and not IS_MODAL_WORKER
```

---

## 2. Configuration

All hyperparameters and toggles are defined once in cell-1 (before the env
detection) so they are shared across all three roles.

### Hyperparameters

| name | value | meaning |
|---|---|---|
| `LORA_RANK` | 32 | LoRA rank r for all target modules |
| `LORA_ALPHA` | 32 | LoRA scaling α (effective scale = α/r = 1.0) |
| `LORA_DROPOUT` | 0.0 | no dropout during SFT |
| `MAX_SEQ_LEN` | 8192 | sequences longer than this are truncated |
| `NUM_STEPS` | 1000 | optimizer steps (clamped to `len(examples)//BATCH_SIZE`) |
| `BATCH_SIZE` | 32 | examples per optimizer step (accumulated over micro-batches) |
| `MICRO_BATCH_SIZE` | 4 | examples per forward/backward pass |
| `LEARNING_RATE` | 2e-4 | peak LR; decays linearly to 0 |

### Target Modules

```
q_proj, k_proj, v_proj, o_proj,   # attention projections
up_proj, down_proj,                # MLP / MoE experts
in_proj, out_proj,                 # Mamba SSM projections
lm_head                            # output head (injected manually)
```

### Behaviour Toggles

| flag | default | effect when `True` |
|---|---|---|
| `RESET_WEIGHTS` | `True` | skip loading a pretrained adapter; start from fresh LoRA init |
| `IN_PROJ_ONLY` | `False` | freeze all LoRA params except `in_proj` (Mamba SSM only) |
| `MOE_TIE_WEIGHTS` | `True` | tie one shared LoRA factor across all 128 MoE experts |
| `ORIGINAL_PROBLEMS_ONLY` | `False` | filter corpus to `problem_id`s listed in `train.csv` |
| `SHUFFLE_DATASET` | `False` | shuffle examples before training (default: corpus order) |
| `SKIP_TRAINING` | `False` | skip corpus load + training loop; save the fresh zero-delta LoRA adapter as a base-model baseline (`lora_B` init=0 → adapter contributes nothing) |

---

## 3. Execution Flow

All training logic lives in `run_training()`. Below is the ordered execution
sequence with environment-specific branches noted.

### 3.1 Env-specific paths

| path/source | Kaggle | Modal worker |
|---|---|---|
| base model | `kagglehub.model_download(...)` | `"unsloth/Nemotron-3-Nano-30B-A3B"` |
| corpus | `/kaggle/input/.../tokens/` (one dir per problem) | `/data/corpus_preprocessed.jsonl` |
| train order | `/kaggle/.../logprobs/index.jsonl` | — |
| `train.csv` | `/kaggle/input/competitions/...` | `/data/train.csv` |
| adapter src | `/kaggle/tmp/pretrained_adapter` (from `submission.zip`) | `/merged/weights` |
| adapter dst | `.` (working dir) → `submission.zip` | `/output/weights` → Kaggle dataset |

### 3.2 GPU / CUDA kernel sanity check

- Prints GPU name, `sm_XX`, torch version, VRAM.
- Verifies `mamba_ssm` and `causal_conv1d` versions.
- On Modal worker: asserts `sm_120` (RTX PRO 6000).
- Runs a live `causal_conv1d_fn` call to confirm the CUDA kernel works.

### 3.3 Corpus load → `examples` list

Each entry in `examples`:
```
{
  "problem_id": str,
  "tokens":  list[int],   # input  ids (tokens[:-1])
  "targets": list[int],   # target ids (tokens[1:])
  "weights": list[float], # 1.0 = train on this token, 0.0 = masked
}
```

- **Kaggle:** reads `index.jsonl` to recover epoch-0 problem ordering, then
  loads `{CORPUS_PATH}/{problem_id}/synthetic.json` for each id in that order.
- **Modal worker:** reads one JSON record per line from `corpus_preprocessed.jsonl`.
- Sequences longer than `MAX_SEQ_LEN` are **truncated** (not dropped).
- Entries where all weights are 0 are skipped.

### 3.4 Optional `ORIGINAL_PROBLEMS_ONLY` filter

When enabled: load `problem_id` set from `train.csv`, drop examples not in it.

### 3.5 Base model load

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False, load_in_8bit=False,
    full_finetuning=False,
    trust_remote_code=True,
    unsloth_force_compile=True,
    attn_implementation="eager",
    dtype=torch.bfloat16,
)
```

### 3.6 LoRA wrap + `lm_head` injection

- `FastLanguageModel.get_peft_model(...)` wraps all `TARGET_MODULES` in LoRA.
- Unsloth silently skips `lm_head` for MoE models → the pipeline manually
  calls `model.base_model._create_and_replace(...)` to inject it (see §4).

### 3.7 dtype policy

| param group | dtype |
|---|---|
| LoRA A/B matrices (`.lora_*`) | `float32` |
| Base model weights | `bfloat16` |
| MoE router (`.mixer.gate.*`) | `float32` (kept by transformers) |

Each group is verified with an `assert` loop after casting.

### 3.8 Mamba fast-path patch

Unsloth's compile path leaves `is_fast_path_available = False` on the
`modeling_nemotron_h` module. The pipeline finds the module in `sys.modules`
and patches it to `True` at runtime.

### 3.9 CCE forward patch

The model's `CausalLM.forward` is replaced with a custom function that:
1. Runs only the backbone (`_base.backbone(...)`).
2. Fuses the `lm_head` weight + LoRA delta inline: `W = base_w + scale * B @ A`.
3. Calls `linear_cross_entropy(hidden_states, W, labels, reduction="none")` from
   the `cut-cross-entropy` library — no logit tensor is materialised.
4. Stashes `per_token_ce` on `model._cached_per_token_ce` for the training loop.

### 3.10 Adapter load vs fresh init

- `RESET_WEIGHTS=True` (default): skip loading; LoRA matrices stay at their
  random init.
- `RESET_WEIGHTS=False`: load from `ADAPTER_SRC` using `load_peft_weights`;
  remap three key-name variants (plain, `.default.`, and `backbone.lm_head.`).

### 3.11 Optional `IN_PROJ_ONLY` freeze

When `IN_PROJ_ONLY=True`: every LoRA param not containing `.in_proj.` in its
name has `requires_grad` set to `False`.

### 3.12 MoE weight tying (see §5)

Identify tied params and call `_tie_param_init()`.

### 3.13 Training loop

See §6 for full details.

### 3.14 Save adapter + key rename

```python
model.save_pretrained(save_dir)   # writes adapter_model.safetensors
# rename: base_model.model.lm_head. → base_model.model.backbone.lm_head.
save_file(renamed, st_path)
```

### 3.15 Package & ship

| env | action |
|---|---|
| Kaggle | zip adapter files → `submission.zip`; remove originals |
| Modal worker | write `training_log.txt`; commit volume; upload dir to Kaggle dataset via `kaggle datasets version` |

---

## 4. Key Model-Specific Patches

| patch | where | why |
|---|---|---|
| Manual `lm_head` LoRA injection | after `get_peft_model` | Unsloth skips `lm_head` for MoE architectures; it must be added back via `_create_and_replace` |
| fp32 router exception in dtype assert | dtype verification loop | transformers keeps `NemotronHTopkRouter.weight` + `e_score_correction_bias` in fp32 by design; the assert loop explicitly allows this |
| Mamba fast-path flag | `sys.modules` patch | `is_fast_path_available` is `False` after Unsloth compile; setting it `True` re-enables the CUDA selective-scan kernel |
| CCE forward override | `_base.forward = _patched_causal_forward` | avoids materialising the full `[batch, seq, vocab]` logit tensor (saves ~10 GB VRAM at seq_len 8192) |
| `lm_head` key rename in safetensors | post-save | Unsloth saves the key as `base_model.model.lm_head.*`; the submission format expects `base_model.model.backbone.lm_head.*` |

---

## 5. MoE Weight Tying (Tinker Convention)

Nemotron-H has 128 MoE experts per layer. Unsloth stores their LoRA matrices as
a batched tensor of shape `[128, out, r]` / `[128, r, in]`. Tying means all 128
slices share one logical weight.

### Which side is tied

| projection | tied side |
|---|---|
| `gate_up_proj`, `up_proj`, `gate_proj`, `w1` | lora_**A** (input/hidden side) |
| `down_proj`, `w2` | lora_**B** (output/hidden side) |

Only expert LoRA params (`.experts.` + `.lora_`) with ≥ 2 slices are tied.

### Tying operations

- **`_tie_param_init()`** — called once before training:
  `p.data = p.data.mean(dim=0, keepdim=True).expand_as(p.data)`
  (all 128 slices become identical).

- **`_tie_grads()`** — called after each backward, before `optimizer.step()`:
  `p.grad = p.grad.sum(dim=0, keepdim=True).expand_as(p.grad)`
  Uses **sum** (not mean) because `dL/dW = Σ_i dL/dW_i` under shared weights;
  mean would introduce a silent `1/128` lr rescale incompatible with AdamW's
  eps and weight-decay.

### Note on saving

`model.save_pretrained()` serialises all 128 slices individually, so
`submission.zip` naturally untied — the tying only lives during the training run.

---

## 6. Training Loop Details

```
for batch (size BATCH_SIZE) in examples:
    for micro-batch (size MICRO_BATCH_SIZE) in batch:
        pad inputs/targets/weights to max_len in micro-batch
        forward → model() → per_token_ce (from CCE patch)
        weighted_loss = per_token_ce * padded_weights
        loss = loss_sum / weight_sum
        (loss / n_accum).backward()
    _tie_grads()
    clip_grad_norm_(params, max_norm=1e9)  # effectively no clip
    optimizer.step(); optimizer.zero_grad()
    lr = LEARNING_RATE * (1 - step / num_steps)  # linear decay
```

Key details:
- **Padding:** right-padded to the longest sequence in the micro-batch; attention
  mask set to 0 for pad positions.
- **Weighted loss:** per-token CE is multiplied by the float mask (0/1) from the
  corpus; loss = `sum(ce * w) / sum(w)`. Fully masked batches → loss = 0.
- **Gradient accumulation:** `n_accum = ceil(BATCH_SIZE / MICRO_BATCH_SIZE)`;
  each micro-batch contributes `loss / n_accum` to the gradient.
- **LR schedule:** linear decay from `LEARNING_RATE` to 0 over `num_steps`.
- **`NUM_STEPS` clamping:** if `NUM_STEPS > len(examples) // BATCH_SIZE`, it is
  clamped to `max_steps` with a warning.
- **Optimizer:** `AdamW(betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)`;
  initialised lazily on step 1.

---

## 7. Modal Infrastructure

### Container Image (`train_image`)

Built from `nvidia/cuda:12.8.1-devel-ubuntu22.04` + Python 3.12:

| layer | packages / commands |
|---|---|
| PyTorch | `torch==2.10.0` (cu128 wheel) |
| ML stack | `transformers≥4.56.2`, `peft≥0.15.0`, `accelerate≥1.0.0`, `bitsandbytes≥0.45.0`, `trl`, `datasets`, `sentencepiece`, `xformers` |
| Kernels | `mamba_ssm==2.3.1` + `causal_conv1d==1.6.1` — built from source with `TORCH_CUDA_ARCH_LIST='12.0'` for sm_120 |
| Unsloth | installed from git (`unsloth_zoo[base]` then `unsloth[base]`) |
| CCE | `cut-cross-entropy≥25.1.0` |
| HF | `huggingface_hub≥0.36.2`, `hf-transfer≥0.1.9`; `HF_HOME=/root/.cache/huggingface` |

### Volumes

| volume name | mount | contents |
|---|---|---|
| `huggingface-cache` | `/root/.cache/huggingface` | model weights cached after first download |
| `merged-adapter` | `/merged` | pre-merged adapter weights fed as `ADAPTER_SRC` |
| `corpus-data` | `/data` | `corpus_preprocessed.jsonl` + `train.csv` |
| `post-finetune-output` | `/output` | saved adapter + `training_log.txt` |

### Function config

- **GPU:** `RTX-PRO-6000` (sm_120, 96 GB VRAM)
- **Timeout:** 6 × 60 min = 360 min
- **Secret:** `KAGGLE_API_TOKEN` (from local env, used for dataset upload)

### Output round-trip

```
Modal worker saves adapter → /output/weights/
  → commits output_vol
  → uploads via `kaggle datasets version -p <upload_dir>`
  → lands in KAGGLE_DATASET = "huikang/nemotron-data"
```

---

## 8. Inputs & Outputs

### Inputs

| artefact | Kaggle path | Modal worker path |
|---|---|---|
| Base model | `kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/...")` | `"unsloth/Nemotron-3-Nano-30B-A3B"` (HF Hub) |
| Corpus tokens | `/kaggle/input/.../nemotron/tokens/{problem_id}/synthetic.json` | `/data/corpus_preprocessed.jsonl` |
| Training order | `/kaggle/input/.../nemotron/logprobs/index.jsonl` | — (corpus already ordered) |
| train.csv | `/kaggle/input/competitions/nvidia-nemotron.../train.csv` | `/data/train.csv` |
| Pretrained adapter | `/kaggle/tmp/pretrained_adapter` (extracted from `submission.zip`) | `/merged/weights` |

### Outputs

| artefact | Kaggle | Modal worker |
|---|---|---|
| LoRA adapter | `./submission.zip` (adapter files zipped) | `/output/weights/adapter_model.safetensors` + config |
| Training log | — | `/output/weights/training_log.txt` |
| Remote copy | — | uploaded to `huikang/nemotron-data` on Kaggle |
