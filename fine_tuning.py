import os
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_NAME = "knkarthick/dialogsum"
BASE_MODEL = "google/flan-t5-base"
PEFT_DIR = "./peft-dialogue-summary-checkpoint-from-s3"
INSTRUCT_MODEL_DIR = "./flan-dialogue-summary-checkpoint"
MODELS_RESULTS_DIR = "./data/dialogue-summary-training-results.csv"
EXAMPLE_INDEX = 200
DASH_LINE = '-'.join('' for x in range(100))

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def build_prompt(dialogue):
    return f"""
    Summarize the following conversation.

    {dialogue}

    Summary:
    """

def print_number_of_trainable_model_parameters(model):
    """calculates the total number of parameters in the model"""
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():  # returns an iterator over all trainable parameters of a pytorch model
        all_model_params += param.numel()  # numel(): counts the num of parameters in the tensor
        if param.requires_grad:
            # requires_grad: specifies this parameter needs to compute and store the gradient during backpropagation.
            # which means this tensor is a trainable parameter for the model.
            trainable_model_params += param.numel()

    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

def tokenize_function(example):
    """form the prompt and tokenize it from the input example"""
    start_prompt = "Summarize the following conversation.\n\n"
    end_prompt = "\n\nSummary: "
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors='pt').input_ids  # feed into the model
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors='pt').input_ids  # expected output

    return example

def rouge_scores(predictions, references):
    rouge = evaluate.load('rouge')
    return rouge.compute(
        predictions=predictions,
        references=references,
        use_aggregator=True,
        use_stemmer=True,
    )

def compare_rouge_results(left, right, keys):
    improvement = (np.array(list(left)) - np.array(list(right)))
    for key, value in zip(keys, improvement):
        print(f"{key}: {value*100:.2f}%")


# ---------------------------------------------------------------------------
# 1. Datasets and Model preparation
# ---------------------------------------------------------------------------
print(DASH_LINE + "DATASET AND MODEL PREP" + DASH_LINE + "\n")
# 1-a load dataset
dataset = load_dataset(DATASET_NAME)
print("DATASET DESCRIPTION:")
print(dataset)

# 1-b load LLM: Flan-T5
original_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("\n")
print("TRAINABLE PARAMETERS - ORIGINAL MODEL:")
print(print_number_of_trainable_model_parameters(original_model))

# 1-c test the model with zero shot inferencing
dialogue = dataset["test"][EXAMPLE_INDEX]["dialogue"]
summary = dataset["test"][EXAMPLE_INDEX]["summary"]

prompt = build_prompt(dialogue)
inputs = tokenizer(prompt, return_tensors='pt')
output = tokenizer.decode(
    original_model.generate(
        inputs["input_ids"],
        max_new_tokens=200,
    )[0],
    skip_special_tokens=True
)

print("\n")
print("TEST THE MODEL WITH ZERO-SHOT INFERENCE.")
print(DASH_LINE)
print(f"INPUT PROMPT: {prompt}")
print(DASH_LINE)
print(f"BASELINE HUMAN SUMMARY: \n{summary}")
print(DASH_LINE)
print(f"""MODEL GENERATION - ZERO SHOT:\n{output}""")

# ---------------------------------------------------------------------------
# 2. Full Fine-Tuning Walk Through
# ---------------------------------------------------------------------------
# # 2-a Data preparation
print(DASH_LINE + "FULL FINE-TUNING WALK THROUGH" + DASH_LINE + "\n")
# the dataset actually contains 3 different splits: train, validation, test.
# The tokenize_function is handling all data across all splits in batches.
tokenized_datasets = dataset.map(tokenize_function, batched=True)  # pass a batch of example to the function
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary'])  # remove the original columns, only use input_ids and labels

# extract some sample from the dataset (for testing purpose)
s = tokenized_datasets.filter(lambda example, index: index % 100 == 0, with_indices=True)

print(f"Shape of the tokenized datasets for full fine-tuning:")
print(f"Training: {tokenized_datasets['train'].shape}") #  (12460, 2): each row is a tokenized (prompt,summary) pair, 2 columns: 'input_ids', 'labels'
print(f"Validation: {tokenized_datasets['validation'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")

print(tokenized_datasets)
print("\n")
#
# # 2-b Fine-tune the model with the Preprocessed Dataset
output_dir = f'./dialogue-summary-training-{str(int(time.time()))}'  # check points, logs, TensorBoard go there

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5, # initial LR for the optimiser, small since full FT updates every weight, big step sizes blow up training.
    num_train_epochs=1, # train once on this train
    weight_decay=0.01,  # the rate we decay the weight in every update for large weight matrice's regularisation
    logging_steps=1,    # log every step
    max_steps=1         # only trained one step
)

trainer = Trainer(
    model=original_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)

trainer.train()

# # 2-c free memory before heading to the next stage
trainer = None
tokenizer = None
original_model = None
torch.mps.empty_cache()
import gc; gc.collect()

# ---------------------------------------------------------------------------
# 3. PEFT(Partial Efficient Fine-Tuning) Walk Through
# ---------------------------------------------------------------------------
print(DASH_LINE + "PEFT WALK THROUGH" + DASH_LINE + "\n")
# 3-a load the model
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

# # 3-c train PEFT Adapter
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

# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------
print(DASH_LINE + "MODELS EVALUATION" + DASH_LINE + "\n")
gen_cfg = GenerationConfig(max_new_tokens=200, num_beams=1)

# During the evaluation, we don't train the models from the scratch, we load some well-trained models instead for convenience
# 4-a load the models
models : dict[str, AutoModelForSeq2SeqLM] = {}
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# load the base model — Flan-T5
models["original"] = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
)

# We load a fully fine-tuned model here as full fine-tuning costs a lot of computational resources
models["instruct"] = AutoModelForSeq2SeqLM.from_pretrained(
    INSTRUCT_MODEL_DIR,
    torch_dtype=torch.bfloat16,  # tells the model how to load the weights into memory
)

# Also we load the fine-tuned peft checkpoint to fit in the base model
models["peft"] = PeftModel.from_pretrained(
    models["original"],
    PEFT_DIR,
    torch_dtype=torch.bfloat16,
    is_trainable=False,  # inference only
)

# 4-b Evaluate models qualitively
dialogue = dataset['test'][EXAMPLE_INDEX]['dialogue']
human_baseline_summary = dataset['test'][EXAMPLE_INDEX]['summary']

prompt = build_prompt(dialogue)

input_ids = tokenizer(prompt, return_tensors='pt').input_ids

models_text_outputs : dict[str, str] = {}
for key, model in models.items():
    model_outputs = models[key].generate(input_ids=input_ids, generation_config=gen_cfg)
    models_text_outputs[key] = tokenizer.decode(model_outputs[0], skip_special_tokens=True)


print(DASH_LINE)
print(f"BASELINE HUMAN SUMMARY:\n{human_baseline_summary}")
for key, value in models_text_outputs.items():
    print(DASH_LINE)
    print(f"{key.upper()} MODEL: {value}")

# 4-c Evaluate the model quantitively
dialogues = dataset["test"][0:10]["dialogue"]
human_baseline_summaries = dataset["test"][0:10]["summary"]
model_summaries : dict[str, list] = {}

for index, dialogue in enumerate(dialogues):
    prompt = build_prompt(dialogue)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    for key, model in models.items():
        model_outputs = model.generate(input_ids=input_ids, generation_config=gen_cfg)
        model_text_output = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
        if key in model_summaries.keys():
            model_summaries[key].append(model_text_output)
        else:
            model_summaries[key] = [model_text_output]

zipped_summaries = list(zip(
    human_baseline_summaries,
    model_summaries['original'],
    model_summaries['instruct'],
    model_summaries['peft']
))

print("zipped_summaries\n", zipped_summaries)
df = pd.DataFrame(zipped_summaries, columns=["human_baseline_summaries", "original_model_summaries", "instruct_model_summaries", "peft_model_summaries"])
print("OUTPUTS FOR ALL MODELS:")
print(df.head())

# compute the rouge matrix
models_rouge_scores = {}

for key, model in models.items():
    # load the results and print them
    models_rouge_scores[key] = rouge_scores(model_summaries[key], human_baseline_summaries)
    print(f"{key.upper()} ROUGE SCORE: \n{models_rouge_scores[key]}")

# later on we compare the output with the results which is output by input the whole dataset into the three models
results = pd.read_csv(MODELS_RESULTS_DIR, index_col=0)
print(results.columns)
print(results.shape)

summaries_scores = {}
human_baseline_summaries = results["human_baseline_summaries"]
for key in results.columns:
    if key != "human_baseline_summaries":
        summaries_scores[key] = rouge_scores(results[key], human_baseline_summaries)
        print(f"{key.upper()} MODEL ROUGE SCORE: \n{summaries_scores[key]}")

# INSTRUCT MODEL vs ORIGINAL
print("Absolute percentage improvement of INSTRUCT MODEL over ORIGINAL MODEL")
compare_rouge_results(summaries_scores["instruct_model_summaries"].values(), summaries_scores["original_model_summaries"].values(), summaries_scores["instruct_model_summaries"].keys())

# PEFT MODEL vs ORIGINAL
print("Absolute percentage improvement of PEFT MODEL over ORIGINAL MODEL")
compare_rouge_results(summaries_scores["peft_model_summaries"].values(), summaries_scores["original_model_summaries"].values(), summaries_scores["peft_model_summaries"].keys())

# PEFT MODEL vs INSTRUCT MODEL
print("Absolute percentage improvement of PEFT MODEL over INSTRUCT MODEL")
compare_rouge_results(summaries_scores["peft_model_summaries"].values(), summaries_scores["instruct_model_summaries"].values(), summaries_scores["peft_model_summaries"].keys())


