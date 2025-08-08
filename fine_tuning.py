import os
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from data_preparation import (
    BASE_MODEL,
    DATASET_OUT_DIR,
    print_number_of_trainable_model_parameters,
)

# ---------------------------------------------------------------------------
# Full Fine-Tuning
# ---------------------------------------------------------------------------
# load dataset & model
tokenized_datasets = load_from_disk(DATASET_OUT_DIR)
tokenized_datasets.set_format(type="torch")
original_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# fine-tune the model with the Preprocessed Dataset
output_dir = f'./dialogue-summary-training-{str(int(time.time()))}'  # check points, logs, TensorBoard go there

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5, # initial LR for the optimiser, small since full FT updates every weight, big step sizes blow up training.
    num_train_epochs=1, # train once on this train
    weight_decay=0.01,  # the rate we decay the weight in every update for large weight matrice's regularisation
    logging_steps=1,    # log every step
    max_steps=1         # only trained one step
)

full_trainer = Trainer(
    model=original_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)

full_trainer.train()

# free memory before heading to the next stage
full_trainer = None
tokenizer = None
original_model = None
torch.mps.empty_cache()
import gc; gc.collect()

# ---------------------------------------------------------------------------
# PEFT(Partial Efficient Fine-Tuning)
# ---------------------------------------------------------------------------
# load the original model
original_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# 3-b Perform Parameter Efficient Fine-Tuning (PEFT)
lora_config = LoraConfig(
    r=32, # rank of the lora adapter
    lora_alpha=32, # scaling factor, ΔW = (alpha/r)*(AB), since A and B are randomly initialised, the output can be untsable especially early on.
    target_modules=["q", "v"],  # Q influences where attention is paid, V is the actual content being passed on. K is less benefit shown from fine-tuning so we freeze it for effectiveness
    lora_dropout=0.05, # apply some dropout on A only
    bias="none",   # no bias for the lora to keep the consistency to the frozen base model
    task_type=TaskType.SEQ_2_SEQ_LM  # FLAN-T5
)

peft_model = get_peft_model(original_model, lora_config)  # here we get the original model + LoRA layers injected into the target linear modules.
print(print_number_of_trainable_model_parameters(peft_model))

# 3-c train PEFT Adapter
output_dir = f"./pft-dialogue-summary-training-{str(int(time.time()))}"

peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True, # LoRA needs far less memory than full fine-tune, so we can let the trainer automatically bisect the batch
    learning_rate=1e-3, # higher than full fine-tune, since LoRA only updates tiny low-rank adapters
    num_train_epochs=1,
    logging_steps=1,
    max_steps=1,
) # weight_decay is omitted since LoRA adapters are already tiny, extra decay often hurts more than helps

peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["test"],
)

peft_trainer.train()

peft_model_path="./peft-dialogue-summary-checkpoint-local"
peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

# 3-d free memory before heading to the next stage
peft_trainer = None
tokenizer = None
original_model = None
torch.mps.empty_cache()
import gc; gc.collect()

