import os
import yaml
import argparse
from transformers import (
    WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.dataset import prepare_dataset

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def train(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("Loading processor and model...")
    processor = WhisperProcessor.from_pretrained(
        config["model"]["name_or_path"], language=config["model"]["language"], task=config["model"]["task"]
    )
    
    model = WhisperForConditionalGeneration.from_pretrained(config["model"]["name_or_path"])
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    model = prepare_model_for_kbit_training(model)
    
    # LoRA / PEFT setup
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["dropout"],
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Preparing dataset...")
    train_dataset = prepare_dataset(config, processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config["model"]["output_dir"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        warmup_steps=config["training"]["warmup_steps"],
        num_train_epochs=config["training"]["num_epochs"],
        fp16=False, # Disable fp16 on CPU
        use_cpu=True, # Force CPU mode for hardware compatibility test
        save_strategy="epoch",
        eval_strategy="no",
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        report_to=["tensorboard"]
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    model.save_pretrained(config["model"]["output_dir"])
    processor.save_pretrained(config["model"]["output_dir"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
