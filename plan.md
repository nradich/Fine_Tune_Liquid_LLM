# Plan: Fine-tune LFM2.5-350M on Databricks

## Goal

Fine-tune **LiquidAI/LFM2.5-350M** on a small slice of **nvidia/Nemotron-Personas-USA** using persona roleplay SFT, then show **before vs after** answers so the fine-tuned model stays in character with persona details the base model does not use well.

## Choices

| Item | Value |
|------|--------|
| Model | `LiquidAI/LFM2.5-350M` |
| Dataset | `nvidia/Nemotron-Personas-USA` |
| Format | Persona roleplay SFT (system profile + user Q + grounded answer) |
| Scale | ~3,000 rows, 1 epoch (demo) |
| Method | LoRA + TRL `SFTTrainer` |
| Compute | Databricks serverless GPU **1xH100**, AI v5, Python 3.12 |
| Storage | Unity Catalog volume under `/Volumes/<catalog>/<schema>/liquid_ft/` |

## Notebook path

`notebooks/01_finetune_lfm25_personas.py`

### Cell flow

1. Config widgets + install `trl` / `peft` if needed
2. Create UC volume
3. Stream-load ~3k persona rows; format as chat roleplay
4. Load base model; run **BEFORE** eval on fixed persona prompts
5. LoRA SFT (1 epoch); save adapter to volume
6. Run **AFTER** eval on the same prompts; print ground truth for comparison

### Eval design

- Pick a few training personas
- Same system profile + same user questions before and after
- Print ground-truth field text so you can see the FT model align to dataset context (occupation, hobbies, goals, etc.)

## Out of scope

- Multi-GPU / 8xH100 / multi-node distributed training
- Full 1M-row training
- Model serving / UC model registry
- DeepSpeed / Unsloth

## Deploy / run from terminal

Asset Bundle job (`databricks.yml`):

```powershell
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run finetune_lfm25_personas -t dev
```

Uses serverless GPU `GPU_1xH100` + `databricks_ai_v5`. Notebook still `%pip install`s TRL/PEFT (AI Runtime jobs cannot rely on the Environments panel alone).

## Success criteria

- Notebook / job runs end-to-end on 1xH100
- Adapter saved under the UC volume
- Clear before/after outputs showing stronger persona fidelity after training
