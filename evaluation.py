from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig
)
from peft import PeftModel
import pandas as pd
import numpy as np
import evaluate
import torch

from data_preparation import (
    DATASET_NAME,
    BASE_MODEL,
    EXAMPLE_INDEX,
    build_prompt,
    DASH_LINE,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PEFT_DIR = "./peft-dialogue-summary-checkpoint-from-s3"
INSTRUCT_MODEL_DIR = "./flan-dialogue-summary-checkpoint"
MODELS_RESULTS_DIR = "./data/dialogue-summary-training-results.csv"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
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
# 4. Evaluation
# ---------------------------------------------------------------------------
def main():
    dataset = load_dataset(DATASET_NAME)
    gen_cfg = GenerationConfig(max_new_tokens=200, num_beams=1)

    # During the evaluation, we don't train the models from the scratch, we load some well-trained models instead for convenience
    # load dataset & models
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

    # Evaluate models qualitively
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

    # Evaluate the model quantitively
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

if __name__ == "__main__":
    main()
