"""Example training script using Hugging Face Transformers Trainer for seq2seq fine-tuning.

This is an illustrative script; in real workloads split dataset loading and config management.
"""
import argparse
from typing import Optional

import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a seq2seq model for translation (example).")
    parser.add_argument("--model_name", type=str, default="facebook/mbart-large-50-many-to-many-mmt")
    parser.add_argument("--train_file", type=str, required=True, help="Path to train JSONL/CSV/TSV file with columns 'src' and 'tgt'")
    parser.add_argument("--validation_file", type=str, required=False, help="Path to validation file")
    parser.add_argument("--output_dir", type=str, default="models/translation")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=256)
    return parser.parse_args()


def load_dataset_from_file(train_file: str, validation_file: Optional[str] = None):
    # datasets will infer format from extension
    data_files = {"train": train_file}
    if validation_file:
        data_files["validation"] = validation_file
    if train_file.endswith(".csv"):
        ds = datasets.load_dataset("csv", data_files=data_files)
    else:
        ds = datasets.load_dataset("json", data_files=data_files, lines=True)
    return ds


def preprocess_function(examples, tokenizer, max_source_length, max_target_length):
    inputs = examples["src"]
    targets = examples["tgt"]
    model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    args = parse_args()
    ds = load_dataset_from_file(args.train_file, args.validation_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    tokenized_ds = ds.map(lambda x: preprocess_function(x, tokenizer, args.max_source_length, args.max_target_length), batched=True, remove_columns=ds["train"].column_names)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        evaluation_strategy="steps" if "validation" in tokenized_ds else "no",
        save_total_limit=3,
        num_train_epochs=args.num_train_epochs,
        logging_steps=50,
        save_steps=200,
        fp16=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
