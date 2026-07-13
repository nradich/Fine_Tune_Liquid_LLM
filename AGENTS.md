# AGENTS.md

## Approach
- Take your time. Read existing context before editing; prefer verified facts over guesses.
- Keep changes small and intentional. Double-check work before you call it done.

## Project
- Goal: fine-tune LiquidAI LFM2.5-350M on Databricks GPU with Nemotron Personas.
- Entrypoint notebook: `notebooks/01_finetune_lfm25_personas.py`
- Plan: `plan.md`

## Stack (verified for this experiment)
- Model: `LiquidAI/LFM2.5-350M`
- Dataset: `nvidia/Nemotron-Personas-USA` (persona roleplay SFT, ~3k-row demo)
- Training: TRL `SFTTrainer` + PEFT LoRA
- Compute: Databricks serverless GPU **1xH100**, environment **AI v5** (Python 3.12)
- Storage: UC volume `/Volumes/<catalog>/<schema>/liquid_ft/`
- Deploy: Databricks Asset Bundle `databricks.yml` → job `finetune_lfm25_personas`
- CLI (profile `DEFAULT`): `databricks bundle deploy -t dev` then `databricks bundle run finetune_lfm25_personas -t dev`

## Databricks context
- Prefer the serverless 1xH100 path in the notebook (simple single-GPU).
- Reference classic cluster screenshot (if present): `Screenshots/DBX_Compute.png` — multi-node A100 is optional/legacy for this demo.
- Training is on Databricks, not local-first.

## Git / PR workflow (default)
1. Start from up-to-date `main`.
2. Create a dedicated branch for the work (never commit straight to `main`).
3. Do the work on that branch; commit only when asked or when the task clearly requires a commit.
4. When finished, re-check the diff and any applicable checks.
5. Open a PR into `main`; do not merge unless explicitly asked.

## Working rules
- Prefer Databricks notebooks / jobs / workspace artifacts over inventing a local training CLI unless the repo adds one.
- Do not invent stack versions, deps, or run commands not present in the repo.
- When code lands, update this file with real entrypoints and verified commands; drop stale claims.
