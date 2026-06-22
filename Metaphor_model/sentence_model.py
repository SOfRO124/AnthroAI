"""
SentenceMetaphorBERT — sentence-level metaphor classifier (BERT + classification head).

Extracted from the sentence-level section of notebooks/metaphors.ipynb.
Uses AutoModelForSequenceClassification fine-tuned via HuggingFace Trainer.

Quick-start (inference only):
    from Metaphor_model.sentence_model import load_sentence_metaphor_bert, predict_sentence_metaphor
    model, tokenizer = load_sentence_metaphor_bert("Metaphor_model/sentence_bert_results/best")
    results = predict_sentence_metaphor(sentences, model, tokenizer)
"""

import os

import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from datasets import Dataset as HFDataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class SentenceMetaphorDataset(Dataset):
    """Sentence-level binary dataset for metaphor classification."""

    def __init__(self, df, tokenizer, max_length=128, label_col="label"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        for _, row in df.iterrows():
            sentence = str(row["sentence"])
            label = int(row[label_col]) if label_col in df.columns else 0
            self.samples.append({"sentence": sentence, "label": label})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tokenizer(
            s["sentence"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get(
                "token_type_ids",
                torch.zeros(self.max_length, dtype=torch.long),
            ).squeeze(0),
            "labels": torch.tensor(s["label"], dtype=torch.long),
        }


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"f1": f1_score(labels, preds, average="binary", zero_division=0)}


def train_sentence_metaphor_model(
    train_df,
    val_df,
    output_dir="Metaphor_model/sentence_bert_results",
    base_model_name="google-bert/bert-base-german-cased",
    num_epochs=5,
    lr=5e-5,
    batch_size=16,
    weight_decay=0.1,
    label_col="label",
):
    """
    Fine-tune BERT for sentence-level metaphor detection.

    Parameters
    ----------
    train_df, val_df : pd.DataFrame  (columns: sentence, <label_col>)
    output_dir : str   directory for checkpoints
    base_model_name : str  HuggingFace model ID
    label_col : str  name of the binary label column (0=literal, 1=metaphorical)

    Returns
    -------
    model : AutoModelForSequenceClassification  (best checkpoint, eval mode)
    tokenizer : AutoTokenizer
    log_history : list[dict]  Trainer log entries
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)

    def _tok(examples):
        return tokenizer(examples["sentence"], padding="max_length",
                         truncation=True, max_length=128)

    def _prepare(df):
        hf = HFDataset.from_pandas(df[["sentence", label_col]].reset_index(drop=True))
        if label_col != "label":
            hf = hf.rename_column(label_col, "label")
        return hf.map(_tok, batched=True)

    train_hf = _prepare(train_df)
    val_hf   = _prepare(val_df)

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))
    trainer.train()

    best_dir = os.path.join(output_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"Best model saved to {best_dir}")
    return model, tokenizer, trainer.state.log_history


# ------------------------------------------------------------------
# Loading helper
# ------------------------------------------------------------------

def load_sentence_metaphor_bert(checkpoint_dir, device=None):
    """
    Load a saved SentenceMetaphorBERT checkpoint.

    Parameters
    ----------
    checkpoint_dir : str
        Directory saved via Trainer.save_model() (contains config.json, model weights, etc.)
    device : str | None

    Returns
    -------
    model : AutoModelForSequenceClassification  (eval mode, on device)
    tokenizer : AutoTokenizer
    """
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_dir}. "
            "Call train_sentence_metaphor_model() in notebooks/metaphors.ipynb first."
        )
    if device is None:
        device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu"
        )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()
    return model, tokenizer


# ------------------------------------------------------------------
# Inference helper
# ------------------------------------------------------------------

def predict_sentence_metaphor(sentences, model, tokenizer, batch_size=16, device=None):
    """
    Run SentenceMetaphorBERT inference on a list of sentences.

    Parameters
    ----------
    sentences : list[str]
    model : AutoModelForSequenceClassification
    tokenizer : AutoTokenizer
    batch_size : int
    device : str | None

    Returns
    -------
    results_df : pd.DataFrame
        Columns: sentence, prediction (0/1), score (softmax prob of class 1)
    """
    if device is None:
        device = next(model.parameters()).device

    df_tmp = pd.DataFrame({"sentence": sentences, "label": 0})
    dataset = SentenceMetaphorDataset(df_tmp, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds, all_scores = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="SentenceMetaphorBERT predict"):
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
            ).logits
            probs = F.softmax(logits, dim=-1)
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_scores.extend(probs[:, 1].cpu().tolist())

    return pd.DataFrame({
        "sentence": sentences,
        "prediction": all_preds,
        "score": all_scores,
    })
