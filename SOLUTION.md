# SMILES-2026 Hallucination Detection — Solution

## Reproducibility

### Environment
- Python 3.12
- Run on Google Colab with a T4 GPU
- Dependencies in `requirements.txt` (unmodified)

### How to run

```bash
git clone https://github.com/angelinarodionova/SMILES-2026-Hallucination-Detection.git
cd SMILES-2026-Hallucination-Detection
pip install -r requirements.txt
python solution.py
```

Running `python solution.py` will:
1. Download Qwen/Qwen2.5-0.5B (first run only).
2. Extract hidden-state features for the 689 training samples and 100 test samples (about 3 minutes on a T4 GPU).
3. Train a 5-fold cross-validated logistic-regression probe.
4. Write `results.json` and `predictions.csv`.

Random seeds are fixed (`random_state=42` in the splitter and the probe), so the run is deterministic up to small numerical differences.

### Files modified
- `aggregation.py` (feature aggregation strategy)
- `probe.py` (probe classifier)
- `splitting.py` (cross-validation strategy)

Fixed infrastructure (`model.py`, `evaluate.py`, `solution.py`) is unchanged, except for flipping `USE_GEOMETRIC = True` in `solution.py`.

---

## Final solution

### Aggregation (`aggregation.py`)
For each input I take the last 32 real (non-padding) tokens of the final transformer layer and mean-pool them into a single 896-dim vector. Every sample is fed to the model as `prompt + response` with right-padding, so the trailing real tokens are the response tokens. That is where any hallucination signal should live, since the prompt itself is identical across hallucinated and truthful examples. I also enabled the geometric features: per-layer L2 norms (25 dims), cosine drift between consecutive layers (24 dims), and the real sequence length (1 dim). Total feature dimension is 946.

### Probe (`probe.py`)
A logistic regression classifier with `StandardScaler` preprocessing, L2 regularization (`C=0.1`), `class_weight='balanced'` to handle the 70/30 label imbalance, and a decision threshold tuned for F1 on the validation set. The class still inherits from `nn.Module` so it works with the evaluation harness, but the actual classifier inside is sklearn's `LogisticRegression`.

### Splitting (`splitting.py`)
5-fold stratified cross-validation. Each fold reserves 20% of the data as held-out test, and the remaining 80% is split 80/20 into train/val (val is only used for threshold tuning). Stratification keeps the 70/30 class ratio the same in every subset.

### Results (averaged over 5 folds)
| Checkpoint | Accuracy | F1 | AUROC |
|---|---|---|---|
| Majority-class baseline | 70.10% | 82.42% | n/a |
| Probe (train) | 77.55% | 86.54% | 99.97% |
| Probe (val) | 71.71% | 82.91% | 66.21% |
| Probe (test) | 69.08% | 81.20% | 65.63% |

### Why these choices
I picked the last layer because that is the one closest to the model's actual prediction, so it should carry the most information about whether the response is going to be wrong. The earlier layers seem to encode more low-level stuff like syntax. Pooling only over response tokens (instead of the whole prompt) made sense because the prompt is the same regardless of whether the answer is a hallucination, so it just adds noise.

Logistic regression beat the MLP for me. With around 440 training samples per fold and a few hundred to a few thousand features, the MLP just memorized the training set every single time (train AUROC pinned at 100%, test AUROC stuck in the low 60s). A linear classifier with strong L2 regularization handles that situation much better.

I went with 5-fold CV because 689 samples is small. One random split would give me a really noisy estimate, and averaging five folds was less likely to mislead me.

---

## Experiments and failed attempts

I ran four full configurations end-to-end. The number I tracked across runs was test AUROC averaged over the 5 folds.

### v1: MLP probe, 4-layer concatenation (3634 dims)
- Aggregation: mean-pool over all real tokens, concatenated across layers `[-1, -3, -5, -7]`.
- Probe: MLP (3634 -> 256 -> 1), dropout 0.3, Adam, 80 epochs.
- Test AUROC: 64.64%. Test accuracy: 69.81%.
- Train AUROC was already 100%, so the MLP was just memorizing. Feature dim was way too big relative to the training set.

### v2: MLP probe, fewer layers (1842 dims), more regularization
- Aggregation: 2 layers `[-1, -3]` instead of 4.
- Probe: MLP (1842 -> 256 -> 64 -> 1), dropout 0.5, weight decay 1e-2, 30 epochs.
- Test AUROC: 59.28%. This was worse than v1. Cutting layers threw away useful information, and the MLP still overfit even with much stronger regularization.

### v3: Logistic regression, 4-layer features, `C=1.0`
- Same aggregation as v1, but swapped the MLP for sklearn `LogisticRegression(C=1.0)`.
- Test AUROC: 64.20%. Basically the same as v1. Train AUROC was still 100%, which told me even a linear classifier was overfitting with this much feature dimension. The regularization was not strong enough.

### v4 (final): response-token pooling, single layer, strong regularization
- Aggregation: last 32 real tokens of the final layer, plus the geometric features. 946 dims total.
- Probe: logistic regression with `C=0.1`.
- Test AUROC: 65.63%. Test accuracy: 69.08%. Best of all the runs. Train AUROC is still 100% so there is overfitting left, but the generalization gap is smaller than in the earlier runs.

### What I learned from these
All four configurations landed in the 60-66% AUROC range. That is a small spread, and it told me the bottleneck was not the classifier, it was the features. Tuning the probe more wouldn't have moved the number much.

Test accuracy is glued to the majority-class baseline (about 70%) in every run. The dataset is 70/30 imbalanced, so just predicting "hallucinated" for everything gets you 70% accuracy basically for free. The probe is picking up some signal (AUROC above 50% on every fold) but it is not strong enough to consistently beat that 70% threshold.

I think the underlying reason is that the dataset is small (689 samples) and the model is small (Qwen2.5-0.5B, hidden dim 896). There is only so much information you can extract from a 0.5B model's internal representations with 440 training samples per fold.

### Ideas I would try with more time
- Token-level features. Things like the model's per-token logit entropies, max log-prob drop across the response, etc. These do not rely on hidden states.
- Self-consistency features. Generate multiple responses for the same prompt, compare them. There is a lot of literature on this working well.
- A per-layer ablation, training one probe per layer to see which one actually carries the most signal, instead of picking layers heuristically.
- A bigger backbone (Qwen2.5-1.5B or larger), though that is probably outside the scope of this assignment.

---

## Honest reflection

I am new to ML, and this was the first end-to-end project I have done with a transformer. The first few hours went into just understanding what was already in the repo. I had to look up what hidden states are, how attention masks work, and what the ChatML format does. Getting the existing pipeline to run without errors felt like a win in itself, even though I had not actually changed anything yet.

After that, iterating on the probe and aggregation was the easier part. The harder part was figuring out when I was tuning the wrong thing. I spent two full runs adjusting the MLP before I realized that the features themselves were the limit, not the classifier. Once I saw four different setups all landing in the same AUROC range, I stopped trying to squeeze out more numbers and just locked in the best one.

The final result is modest. AUROC around 65% and accuracy at the baseline is not a strong number in absolute terms. But every choice in the final pipeline came from something I observed in an earlier run, and I would rather submit a result I can explain than one I got by tuning blindly.
