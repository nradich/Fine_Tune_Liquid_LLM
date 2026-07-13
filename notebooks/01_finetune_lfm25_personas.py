# Databricks notebook source
# MAGIC %md
# MAGIC # Fine-tune LiquidAI LFM2.5-350M on Nemotron Personas
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | Model | `LiquidAI/LFM2.5-350M` |
# MAGIC | Dataset | `nvidia/Nemotron-Personas-USA` |
# MAGIC | Method | LoRA + TRL SFT (persona roleplay) |
# MAGIC | Compute | Serverless GPU **1xH100**, AI v5 |
# MAGIC
# MAGIC **Goal:** show BEFORE (base) vs AFTER (fine-tuned) answers so the FT model sticks to persona details from the dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install packages
# MAGIC If imports fail after install, use **Detach & re-attach** once, then re-run.

# COMMAND ----------

# MAGIC %pip install -q "transformers>=4.55.0" "trl>=0.9.0" peft accelerate datasets

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Config

# COMMAND ----------

dbutils.widgets.text("uc_catalog", "main")
dbutils.widgets.text("uc_schema", "default")
dbutils.widgets.text("uc_volume", "liquid_ft")
dbutils.widgets.text("n_samples", "3000")
dbutils.widgets.text("num_train_epochs", "1")
dbutils.widgets.text("max_seq_length", "1024")

UC_CATALOG = dbutils.widgets.get("uc_catalog")
UC_SCHEMA = dbutils.widgets.get("uc_schema")
UC_VOLUME = dbutils.widgets.get("uc_volume")
N_SAMPLES = int(dbutils.widgets.get("n_samples"))
NUM_TRAIN_EPOCHS = float(dbutils.widgets.get("num_train_epochs"))
MAX_SEQ_LENGTH = int(dbutils.widgets.get("max_seq_length"))

MODEL_ID = "LiquidAI/LFM2.5-350M"
DATASET_ID = "nvidia/Nemotron-Personas-USA"

VOLUME_ROOT = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}"
CHECKPOINT_DIR = f"{VOLUME_ROOT}/checkpoints/lfm25-personas"
ADAPTER_DIR = f"{VOLUME_ROOT}/adapters/lfm25-personas-lora"

print(f"MODEL_ID      = {MODEL_ID}")
print(f"DATASET_ID    = {DATASET_ID}")
print(f"N_SAMPLES     = {N_SAMPLES}")
print(f"VOLUME_ROOT   = {VOLUME_ROOT}")
print(f"ADAPTER_DIR   = {ADAPTER_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. GPU check + UC volume

# COMMAND ----------

import torch

assert torch.cuda.is_available(), "No GPU. Set Hardware accelerator to 1xH100."
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")

# Catalog/schema should already exist; create volume only.
spark.sql(f"CREATE VOLUME IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}.{UC_VOLUME}")
print(f"Volume ready: {VOLUME_ROOT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load + format dataset (persona roleplay SFT)

# COMMAND ----------

from datasets import Dataset, load_dataset

# (user question, dataset field used as assistant answer)
QUESTION_SPECS = [
    ("What do you do for work, and what is your professional background?", "professional_persona"),
    ("What are your hobbies and interests?", "hobbies_and_interests"),
    ("What are your career goals and ambitions?", "career_goals_and_ambitions"),
    ("What skills and expertise do you have?", "skills_and_expertise"),
]


def system_prompt(row: dict) -> str:
    return (
        "You are the following person. Stay fully in character. "
        "Answer only as this person, using their background and details.\n\n"
        f"Profile: {row.get('persona') or ''}\n"
        f"Age: {row.get('age')}; Sex: {row.get('sex')}; "
        f"Occupation: {row.get('occupation') or ''}; "
        f"Location: {row.get('city') or ''}, {row.get('state') or ''}\n"
        f"Professional summary: {row.get('professional_persona') or ''}"
    )


def make_example(row: dict, q_idx: int) -> dict:
    question, field = QUESTION_SPECS[q_idx % len(QUESTION_SPECS)]
    answer = (row.get(field) or row.get("persona") or "").strip()
    return {
        "messages": [
            {"role": "system", "content": system_prompt(row)},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


print(f"Streaming first {N_SAMPLES} rows from {DATASET_ID} ...")
stream = load_dataset(DATASET_ID, split="train", streaming=True)

raw_rows = []
train_rows = []
for i, row in enumerate(stream):
    if i >= N_SAMPLES:
        break
    raw_rows.append(row)
    train_rows.append(make_example(row, q_idx=i))

train_ds = Dataset.from_list(train_rows)
print(f"Train examples: {len(train_ds)}")

# Fixed eval: first 3 personas, always ask career goals (clear dataset-grounded answer)
EVAL_QUESTION, EVAL_FIELD = QUESTION_SPECS[2]
eval_examples = []
for i in range(3):
    row = raw_rows[i]
    gt = (row.get(EVAL_FIELD) or row.get("persona") or "").strip()
    eval_examples.append(
        {
            "messages_prompt": [
                {"role": "system", "content": system_prompt(row)},
                {"role": "user", "content": EVAL_QUESTION},
            ],
            "ground_truth": gt,
            "meta": {
                "age": row.get("age"),
                "occupation": row.get("occupation"),
                "city": row.get("city"),
                "state": row.get("state"),
            },
        }
    )

for i, ex in enumerate(eval_examples):
    print(f"\nEval {i}: {ex['meta']}")
    print(f"GT preview: {ex['ground_truth'][:180]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load base model

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=dtype,
    device_map="auto",
)
model.eval()
print(f"Loaded {MODEL_ID} ({model.num_parameters():,} params), dtype={dtype}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. BEFORE fine-tune (base model)

# COMMAND ----------

def generate_reply(model, tokenizer, messages, max_new_tokens: int = 180) -> str:
    model.eval()
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_k=50,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


def run_eval(model, tokenizer, tag: str):
    print("=" * 80)
    print(tag)
    print("=" * 80)
    replies = []
    for i, ex in enumerate(eval_examples):
        reply = generate_reply(model, tokenizer, ex["messages_prompt"])
        replies.append(reply)
        print(f"\n### Example {i} | {ex['meta']}")
        print(f"\n[GROUND TRUTH — {EVAL_FIELD}]\n{ex['ground_truth'][:500]}")
        print(f"\n[{tag}]\n{reply}")
        print("-" * 80)
    return replies


before_replies = run_eval(model, tokenizer, "BEFORE (base LFM2.5-350M)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Apply chat template for training

# COMMAND ----------

def to_text(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


train_text_ds = train_ds.map(to_text, remove_columns=train_ds.column_names)
print(train_text_ds[0]["text"][:600])
print("...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. LoRA fine-tune (TRL SFT)

# COMMAND ----------

from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

sft_args = SFTConfig(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    max_length=MAX_SEQ_LENGTH,
    report_to="none",
    dataset_text_field="text",
)

torch.cuda.empty_cache()

trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=train_text_ds,
    processing_class=tokenizer,
    peft_config=peft_config,
)

result = trainer.train()
print(result)

trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"Saved adapter → {ADAPTER_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. AFTER fine-tune (same prompts)

# COMMAND ----------

after_replies = run_eval(trainer.model, tokenizer, "AFTER (LoRA on Nemotron Personas)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Side-by-side summary
# MAGIC The AFTER model should track dataset persona fields (goals, occupation, place) more tightly than the base model.

# COMMAND ----------

print("=" * 80)
print("SIDE-BY-SIDE SUMMARY")
print("=" * 80)

for i, ex in enumerate(eval_examples):
    print(f"\n{'=' * 80}")
    print(f"EXAMPLE {i} | {ex['meta']}")
    print(f"Question: {EVAL_QUESTION}")
    print(f"\n--- GROUND TRUTH ({EVAL_FIELD}) ---\n{ex['ground_truth'][:700]}")
    print(f"\n--- BEFORE (base) ---\n{before_replies[i]}")
    print(f"\n--- AFTER (fine-tuned) ---\n{after_replies[i]}")

print(f"\nAdapter saved at: {ADAPTER_DIR}")
