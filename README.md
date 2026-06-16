# Emotion Detection in Sinhala Social Media Texts

## Project Overview

This repository implements an end-to-end pipeline for multi-class emotion detection in Sinhala social media text. The project compares classical machine learning baselines (TF-IDF + ML classifiers) against deep learning approaches using SinBERT-large, a RoBERTa-based pretrained model for Sinhala. Two dataset variants are explored — an original 2-annotator dataset and an expanded 4-annotator dataset with majority voting — enabling a systematic comparison of annotation quality and model performance.

## Repository Structure

```
├── raw/
│   ├── Dataset.csv                  # Original 2-annotator labeled dataset
│   └── 4_labeled.csv                # Expanded 4-annotator labeled dataset
│
├── processed/
│   └── processed_dataset.csv        # Cleaned output from original preprocessing
│
├── preprocessing.ipynb              # Original dataset: cleaning, Cohen's Kappa agreement, export
├── Model.ipynb                      # Original dataset: SinBERT + CNN model
├── PlainModel.ipynb                 # Original dataset: plain SinBERT fine-tuning (baseline)
├── ML_models.ipynb                  # Classical ML baselines: TF-IDF + Naive Bayes, Logistic Regression
│
└── New_DataSet_Models/
    ├── preprocessing.ipynb          # New dataset: 4-annotator agreement, majority voting, export
    ├── Model.ipynb                  # New dataset: SinBERT + CNN model
    ├── PlainModel.ipynb             # New dataset: plain SinBERT fine-tuning
    └── processed/
        └── processed_dataset.csv   # Cleaned output from new dataset preprocessing
```

## Datasets

### Original Dataset (`raw/Dataset.csv`)
- **Annotators:** 2
- **Agreement method:** Cohen's Kappa; rows where both annotators agreed are kept
- **Processed output:** `processed/processed_dataset.csv`

### Expanded Dataset (`raw/4_labeled.csv`)
- **Annotators:** 4
- **Agreement method:** Fleiss' Kappa (pairwise Cohen's Kappa across all annotator pairs); majority voting determines the final label
- **Processed output:** `New_DataSet_Models/processed/processed_dataset.csv`
- The 4-annotator setup provides stronger inter-rater reliability and a more robust label set

## Models

### 1. Classical ML Baselines (`ML_models.ipynb`)
Traditional machine learning classifiers over TF-IDF features, run in Google Colab:
- **Feature extraction:** TF-IDF (5,000 features, unigrams + bigrams)
- **Classifiers:** Naive Bayes, Logistic Regression
- Serves as the lower-bound performance reference

### 2. Plain SinBERT Fine-Tuning (`PlainModel.ipynb`, `New_DataSet_Models/PlainModel.ipynb`)
Direct fine-tuning of SinBERT-large using `RobertaForSequenceClassification` — a linear classification head on top of the pooled SinBERT output, no CNN layers:
- **Optimizer:** AdamW with weight decay
- **Regularization:** Dropout on transformer layers, class-weighted cross-entropy loss
- **Tuned hyperparameters:** learning rate, dropout rate, weight decay (grid search)
- **Evaluation:** 5-fold stratified cross-validation, early stopping per fold

### 3. SinBERT + CNN (`Model.ipynb`, `New_DataSet_Models/Model.ipynb`)
Hybrid architecture combining SinBERT contextual embeddings with multi-kernel CNNs:
- **Optimizer:** AdamW with linear warmup scheduler
- **PEFT:** LoRA (Parameter-Efficient Fine-Tuning) applied to SinBERT layers
- **Regularization:** Class-weighted cross-entropy loss for imbalance, dropout
- **Tuned hyperparameters:** learning rate, CNN filters, kernel sizes, dropout (grid search)
- **Evaluation:** 5-fold stratified cross-validation, fresh model per fold, early stopping

## Model Architecture (SinBERT + CNN)

```
[Input text]
     |
[SinBERT-large tokenizer]
     |
[SinBERT encoder — token-level contextual embeddings]
     |
[CNN branches: Conv1D(k1) | Conv1D(k2) | Conv1D(k3)]
     |             (each: Conv1D → ReLU → MaxPool)
     +-------------------+
                         |
                    Concatenate
                         |
                      Dropout
                         |
                  Fully connected
                         |
                       Softmax
                         |
               [Emotion class prediction]
```

**Why SinBERT + CNN?**
- SinBERT provides rich contextual embeddings that understand Sinhala morphology and semantics — critical for a low-resource language.
- CNN layers capture local n-gram patterns (emotion-bearing phrases) from contextual embeddings, complementing the global context from the transformer.
- LoRA reduces the number of trainable parameters while preserving SinBERT's pretrained knowledge.

## Quick Start

1. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install Jupyter widget support (required for progress bars):
   ```
   pip install ipywidgets
   jupyter nbextension enable --py widgetsnbextension
   ```

4. Run notebooks in order:

   **Original dataset pipeline:**
   ```
   preprocessing.ipynb → PlainModel.ipynb / Model.ipynb
   ```

   **Expanded dataset pipeline:**
   ```
   New_DataSet_Models/preprocessing.ipynb → New_DataSet_Models/PlainModel.ipynb / New_DataSet_Models/Model.ipynb
   ```

   **Classical baselines** (`ML_models.ipynb`) — designed to run in Google Colab; update the drive paths at the top of the notebook to match your environment.

## Evaluation

All deep learning models are evaluated using:
- **Accuracy** (overall)
- **Precision, Recall, F1-score** — macro and weighted averages
- **Confusion matrix** — per-class breakdown
- **5-fold stratified cross-validation** — metrics reported as mean ± std across folds

Class imbalance is handled with inverse-frequency class weights in the loss function.

## Training Details

| Setting | Value |
|---|---|
| Pretrained model | `NLPC-UOM/SinBERT-large` |
| Optimizer | AdamW |
| Cross-validation | 5-fold stratified |
| Early stopping | Yes (per fold) |
| Class imbalance | Weighted cross-entropy |
| PEFT | LoRA (SinBERT + CNN model) |
| Hardware | CUDA GPU (CPU fallback supported) |

## Troubleshooting

- **tqdm / ipywidgets ImportError:** install ipywidgets and enable the notebook extension (see Quick Start step 3).
- **`TypeError: new(): invalid data type 'str'`** when converting labels: ensure labels are integer-encoded before creating tensors. In `__getitem__`: `torch.tensor(int(label), dtype=torch.long)`.
- **Pooler weights warning** (`Some weights of RobertaModel were not initialized...`): benign — the pooler is randomly initialized and fine-tuned during training.
- **DataLoader indexing errors:** verify `__len__` and `__getitem__` are implemented correctly and return tensors, not strings.

## Project Credits

- **Authors:** 2021/E/045 and 2021/E/053
- **Pretrained model:** [NLPC-UOM/SinBERT-large](https://huggingface.co/NLPC-UOM/SinBERT-large) (Hugging Face)
