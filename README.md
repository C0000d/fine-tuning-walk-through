# Flan‑T5 Dialogue‑Summary ❙ full vs LoRA fine‑tuning comparison 
*Train‑time frugal flavour‑tuning on **DialogSum** with ROUGE tracking*

---

##  What this repo contains

| Area | What I actually coded |
|------|-----------------------|
| **Dataset handling** | Loaded **`knkarthick/dialogsum`** via 🤗 `datasets`; formulated a prompt wrapper and batch‑tokenised every split with the base Flan‑T5 tokenizer. |
| **Baseline check** | Ran zero‑shot generation from **`google/flan‑t5‑base`** to record starting quality and inspect the prompt shape. |
| **Full fine‑tune stub** | One‑epoch / one‑step `Trainer`; shows the wiring for end‑to‑end FT, but keeps compute minimal. |
| **Parameter‑Efficient FT** | Injected **LoRA** adapters (`r=32`, Q+V only) with `peft`; trained on the same data slice using a higher LR and `auto_find_batch_size`. |
| **Model bookkeeping** | Saved LoRA weights locally, plus a separate *fully‑fine‑tuned* checkpoint previously trained on GPU (referenced as `./flan-dialogue-summary-checkpoint`). |
| **Evaluation harness** | • Generates summaries for 10 test items with three systems (base / full‑FT / LoRA).<br>• Computes aggregated **ROUGE‑1/2/L/Lsum** via `evaluate`.<br>• Helper to print absolute % deltas between systems. |

---

## ️  Quick start

```bash
# baseline & tokenised cache
python data_preparation.py

# run fine-tuning pipeline
python fine-tuning.py      # runs sections 1 and 2 − baseline + full‑FT stub

# evaluation
python evaluation.py
```

---

##  Results

### ROUGE scores (higher = better)

| Model            | ROUGE‑1 | ROUGE‑2 | ROUGE‑L | ROUGE‑Lsum |
|------------------|:-------:|:-------:|:-------:|:----------:|
| Original (Flan‑T5) | 0.2334 | 0.0759 | 0.2014 | 0.2014 |
| Full FT (“Instruct”) | **0.4212** | **0.1807** | **0.3384** | **0.3384** |
| LoRA PEFT         | 0.4080 | 0.1637 | 0.3251 | 0.3252 |

### Absolute improvement (percentage points)

| Comparison             | Δ ROUGE‑1  | Δ ROUGE‑2  | Δ ROUGE‑L  | Δ ROUGE‑Lsum |
|------------------------|:----------:|:----------:|:----------:|:------------:|
| Instruct vs Original   | **+18.78** | **+10.48** | **+13.70** |  **+13.70**  |
| PEFT vs Original       |  +17.46    |   +8.78    |  +12.37    |   +12.37     |
| PEFT vs Instruct       |   −1.33    |   −1.70    |   −1.33    |    −1.32     |

### Trainable parameters & compute trade‑off

| Approach               | Trainable Params | Total Params* | % Trainable | Key takeaway |
|------------------------|-----------------:|--------------:|------------:|--------------|
| Full fine‑tune (“Instruct”) | **247 M** | 247 M | **100 %** | Highest accuracy, but you’re updating every weight → chunky GPU, long training time |
| LoRA PEFT              | 3.5 M | 251 M | **1.4 %** | ~98 % fewer weights touched. Tiny VRAM hit and super‑fast to iterate, with only ~1–2 pp ROUGE‑L drop |

\* *Total params stay roughly the same—LoRA just adds a smidge of extra adapter weights on top of the frozen base.*

> **TL;DR:** If you’ve got the hardware and need every last ROUGE point, full fine‑tune wins. If you’re skint on compute (or keen on rapid A/B testing), LoRA gets you 95 % of the gains for 1 % of the effort.
