# Fine_Tune_Liquid_LLM

Fine-tune **LiquidAI/LFM2.5-350M** on Databricks GPU using **nvidia/Nemotron-Personas-USA**.

## Experiment

| Item | Value |
|------|--------|
| Notebook | `notebooks/01_finetune_lfm25_personas.py` |
| Model | [LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) |
| Dataset | [nvidia/Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA) |
| Method | Persona roleplay SFT with LoRA (TRL) |
| Compute | Databricks serverless GPU **1xH100**, AI v5 |
| Plan | `plan.md` |

## How to run (recommended: Asset Bundle job)

Requires Databricks CLI auth (`databricks auth profiles`; this repo expects profile `DEFAULT`).

```powershell
# From repo root
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run finetune_lfm25_personas -t dev
```

Optional job parameters (also bundle variables):

```powershell
databricks bundle run finetune_lfm25_personas -t dev --params uc_catalog=main,uc_schema=default,n_samples=3000
```

The job uses serverless GPU **1xH100** + AI v5 (`databricks.yml`).

### Interactive notebook (optional)

1. Open the deployed notebook under your user `.bundle/` path, or import `notebooks/01_finetune_lfm25_personas.py`.
2. Set **Hardware → Accelerator → 1xH100** (AI v5).
3. Set widgets (`uc_catalog`, `uc_schema`, `uc_volume`, `n_samples`) and **Run all**.

## Outputs

- Checkpoints: `/Volumes/<catalog>/<schema>/liquid_ft/checkpoints/lfm25-personas`
- LoRA adapter: `/Volumes/<catalog>/<schema>/liquid_ft/adapters/lfm25-personas-lora`

## Notes

- Defaults write to managed volume `benchmarks.default.liquid_ft` (works in this workspace). Avoid `main` if UC storage credentials are broken.
- Training uses a small demo slice (~3k rows, 1 epoch), not the full 1M dataset.
- The notebook streams the HF dataset so it does not download all ~2.6GB up front.
