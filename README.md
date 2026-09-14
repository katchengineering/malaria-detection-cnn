# Automated Malaria Detection Using Deep Learning

Binary classification of microscopy blood-smear images (*Parasitized* vs. *Uninfected*) with convolutional neural networks, benchmarked across five architectures. Final model reaches **98.42% test accuracy at a 1.08% false-negative rate**.

Capstone project for the MIT Professional Education Applied AI & Data Science program (Deep Learning specialization), September–October 2025.

---

## Why false negatives are the metric that matters

Accuracy is the wrong headline for a diagnostic screen. A missed infection sends an untreated patient home; a false positive sends a healthy patient for confirmatory testing. The two errors are not symmetric, so model selection here optimized for **sensitivity first**, with accuracy as a constraint rather than the objective.

That framing drove every architecture decision below, and it's why the winning model wasn't the one with the most parameters.

---

## Results

| Metric | Value |
|---|---|
| Test accuracy | 98.42% |
| F1 score | 0.98 |
| Precision | 0.98 |
| Recall | 0.98 |
| False positives | 0.50% |
| False negatives | 1.08% |

Trained on ~25,000 cell images with a 2,600-image held-out test set.

---

## Architecture comparison

Five models were trained and evaluated on identical splits:

| Model | Approach | Outcome |
|---|---|---|
| Base CNN | Simple network trained from scratch | Strong accuracy and sensitivity; efficient, stable baseline |
| Extra layers | Added depth to increase capacity | Most sensitive (lowest FN), but longer training with no meaningful accuracy gain |
| BatchNorm + LeakyReLU | Normalization and smoother activation | Most conservative (lowest FP); better regularization, but higher false negatives |
| **Data augmentation** | Rotation, zoom, and flip transforms | **Best overall generalization and robustness — selected** |
| VGG16 transfer learning | Pre-trained ImageNet features, custom head | Fast convergence and stable, but no advantage over the augmented CNN |

**Why augmentation won.** Added depth bought sensitivity at the cost of training time and overfitting risk. BatchNorm bought precision at the cost of the error type that matters most. Transfer learning converged quickly but didn't beat a much smaller purpose-built network on this domain — ImageNet features transfer poorly to stained cell morphology. Augmentation closed the train–validation gap instead of adding capacity, which is the right move when the dataset is moderate and the images are visually homogeneous.

---

## Final pipeline

```
Input            64×64×3 RGB microscopy images, rescaled to [0, 1]
Augmentation     horizontal flip, ±30° rotation, 0.2 zoom (on-the-fly)
Feature blocks   3 × [ Conv2D(32, 3×3, ReLU, same) → MaxPool(2×2) → Dropout(0.2) ]
Classifier       Flatten → Dense(512, ReLU) → Dropout(0.4) → Dense(2, Softmax)
Training         Adam (lr=0.001), binary cross-entropy, early stopping + checkpointing
```

Built in Python with TensorFlow/Keras; NumPy, pandas, scikit-learn, and Matplotlib for preprocessing and evaluation.

---

## Deployment analysis

The project extends past the model into what it would take to use one. Proposed as an AI-assisted screening service — cloud inference for centralized labs, edge inference for clinics with limited connectivity — accelerating triage where lab capacity and trained microscopists are scarce.

Three prerequisites before any clinical use:

1. **Validation and benchmarking** — test against real-world datasets with hospital and lab partners, evaluate sensitivity against expert pathologists, and tune the decision threshold explicitly to suppress false negatives.
2. **Product integration** — a dashboard or mobile client that ingests images from connected slide scanners, returns infection probability with visualization, and lets clinicians override and annotate; the model served as an API.
3. **Deployment architecture** — two-tier cloud and edge inference, with an automated retraining pipeline that folds in newly annotated data.

---

## Limitations

- Trained on a curated public dataset; real-world smears vary in staining protocol, illumination, and scanner optics, and performance on those is unmeasured.
- 64×64 input discards fine morphological detail that a pathologist would use.
- No external clinical validation. The reported metrics describe a held-out split, not diagnostic performance in practice.
- Binary classification only — no parasite species identification or parasitemia quantification, both of which affect treatment.

---

## Repository contents

```
## Repository contents

- [`malaria-detection-code.ipynb`](malaria-detection-code.ipynb) — model training, evaluation, and result plots
- [`malaria-detection-deck.pdf`](malaria-detection-deck.pdf) — project presentation (33 pages)

```

---

**Kathleen Chen** · MS Computer Science candidate, University of Pennsylvania
