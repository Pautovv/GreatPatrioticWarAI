import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, GPT2LMHeadModel

os.environ["WANDB_DISABLED"] = "true"
MODEL_PATH = "ai-forever/rugpt3medium_based_on_gpt2"

OUTPUT_PATH = "./fintuned_2gpt"

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="ptvnck/GPW_stories",
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir=OUTPUT_PATH,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

trainer.train()

trainer.save_model(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)