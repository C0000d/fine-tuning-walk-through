import os
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_NAME = "knkarthick/dialogsum"
BASE_MODEL = "google/flan-t5-base"
DATASET_OUT_DIR = "./data/tokenized_dataset"
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
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True).input_ids  # feed into the model
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True).input_ids  # expected output

    return example

# ---------------------------------------------------------------------------
#  Datasets and Model preparation
# ---------------------------------------------------------------------------
def main():
    # test the model with zero shot inferencing
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

    # the dataset actually contains 3 different splits: train, validation, test.
    # The tokenize_function is handling all data across all splits in batches.
    tokenized_datasets = dataset.map(tokenize_function, batched=True)  # pass a batch of example to the function
    tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary'])  # remove the original columns, only use input_ids and labels
    tokenized_datasets.save_to_disk(DATASET_OUT_DIR)

    print(f"Shape of the tokenized datasets for full fine-tuning:")
    print(f"Training: {tokenized_datasets['train'].shape}") #  (12460, 2): each row is a tokenized (prompt,summary) pair, 2 columns: 'input_ids', 'labels'
    print(f"Validation: {tokenized_datasets['validation'].shape}")
    print(f"Test: {tokenized_datasets['test'].shape}")


if __name__ == "__main__":
    # load dataset
    dataset = load_dataset(DATASET_NAME)

    # load LLM: Flan-T5
    original_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    print("TRAINABLE PARAMETERS - ORIGINAL MODEL:")
    print(print_number_of_trainable_model_parameters(original_model))

    main()