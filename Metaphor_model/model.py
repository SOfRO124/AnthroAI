"""
MetaphorBERT — Siamese BERT for per-word metaphor identification.

Extracted from notebooks/metaphors.ipynb.

Quick-start (inference only):
    from Metaphor_model.model import load_metaphor_bert, predict_sentences
    model, tokenizer, domain_names = load_metaphor_bert(
        metaphor_weights="Metaphor_model/metaphor_bert_best.pt",
        domain_ckpt_dir="Metaphor_model/results",
    )
    results = predict_sentences(sentences_ger, model, tokenizer, domain_names, nlp)
"""

import os
import re

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    get_linear_schedule_with_warmup,
)


def _get_last_checkpoint(folder):
    """Return the path to the highest-numbered checkpoint-N dir inside folder."""
    import re
    if not os.path.isdir(folder):
        return None
    ckpts = [
        d for d in os.listdir(folder)
        if re.match(r"checkpoint-\d+$", d) and os.path.isdir(os.path.join(folder, d))
    ]
    if not ckpts:
        return None
    return os.path.join(folder, sorted(ckpts, key=lambda x: int(x.split("-")[-1]))[-1])

# ------------------------------------------------------------------
# POS special tokens (derived from de_core_news_lg over all datasets)
# ------------------------------------------------------------------
GERMAN_POS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
    "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
    "SCONJ", "SPACE", "VERB", "X",
]
POS_SPECIAL_TOKENS = [f"[POS_{t}]" for t in GERMAN_POS]


def add_pos_tokens(tokenizer):
    """Register German POS tags as additional special tokens."""
    tokenizer.add_special_tokens({"additional_special_tokens": POS_SPECIAL_TOKENS})
    return tokenizer


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class MetaphorDataset(Dataset):
    """
    Word-level dataset: one sample per (sentence, target word).

    Expected DataFrame columns: sentence, pos_tags (list[str]), label (list[int], optional).
    """

    def __init__(self, df, tokenizer, domain_names,
                 max_sent_len=128, max_word_len=16, label_col="label"):
        self.tokenizer = tokenizer
        self.domain_names = domain_names
        self.max_sent_len = max_sent_len
        self.max_word_len = max_word_len
        self.samples = []

        for _, row in df.iterrows():
            sentence = str(row["sentence"])
            pos_tags = row.get("pos_tags", [])
            raw_labels = row.get(label_col, None)

            words = sentence.split()
            if not isinstance(pos_tags, list) or len(pos_tags) != len(words):
                pos_tags = ["X"] * len(words)
            labels = (
                raw_labels
                if isinstance(raw_labels, list) and len(raw_labels) == len(words)
                else [0] * len(words)
            )

            for w_idx, (word, pos, label) in enumerate(zip(words, pos_tags, labels)):
                self.samples.append({
                    "sentence": sentence, "word": word,
                    "word_idx": w_idx, "pos": pos, "label": int(label),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        pos_tok = (
            f"[POS_{s['pos']}]"
            if f"[POS_{s['pos']}]" in self.tokenizer.additional_special_tokens
            else "[UNK]"
        )

        enc = self.tokenizer(
            s["sentence"], pos_tok,
            max_length=self.max_sent_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        token_type_ids = enc.get(
            "token_type_ids",
            torch.zeros(self.max_sent_len, dtype=torch.long)
        ).squeeze(0)

        target_mask = self._build_target_mask(s["sentence"], s["word_idx"], input_ids)

        dom_ids, dom_masks = {}, {}
        for domain in self.domain_names:
            we = self.tokenizer(
                s["word"], max_length=self.max_word_len, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            dom_ids[domain] = we["input_ids"].squeeze(0)
            dom_masks[domain] = we["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "target_mask": target_mask,
            "domain_input_ids": dom_ids,
            "domain_attention_masks": dom_masks,
            "label": torch.tensor(s["label"], dtype=torch.long),
        }

    def _build_target_mask(self, sentence, word_idx, input_ids):
        mask = torch.zeros(len(input_ids), dtype=torch.float)
        words = sentence.split()
        pos = 1  # skip [CLS]
        for i, word in enumerate(words):
            subwords = self.tokenizer.tokenize(word)
            n = len(subwords)
            if i == word_idx:
                for j in range(n):
                    if pos + j < len(input_ids):
                        mask[pos + j] = 1.0
                break
            pos += n
        return mask


def collate_metaphor(batch):
    keys = ["input_ids", "attention_mask", "token_type_ids", "target_mask", "label"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    domains = list(batch[0]["domain_input_ids"].keys())
    out["domain_input_ids"] = {
        d: torch.stack([b["domain_input_ids"][d] for b in batch]) for d in domains
    }
    out["domain_attention_masks"] = {
        d: torch.stack([b["domain_attention_masks"][d] for b in batch]) for d in domains
    }
    return out


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

class LatentInteraction(nn.Module):
    """Bridge the semantic gap between basic and contextual embeddings."""
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size * 2, out_size)

    def forward(self, basic, ctx):
        return F.gelu(self.linear(torch.cat([basic, ctx], dim=-1)))


class MetaphorBERT(nn.Module):
    """
    Siamese BERT for per-word metaphor classification.

    Sentence encoder:  [CLS] sentence [SEP] [POS_TAG] → contextual word embedding
    Domain encoders:   target word → basic embedding (one fine-tuned BERT per domain)
    Latent interaction per domain → concatenated → binary classifier

    Returns
    -------
    inference : (logits [B,2], max_domain_idx [B], distances dict)
    training  : (loss, logits [B,2], max_domain_idx [B])
    """

    def __init__(self, sentence_encoder, domain_encoders, bert_config,
                 num_labels=2, dropout=0.2, interaction_hidden=256):
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.domain_encoders = nn.ModuleDict(domain_encoders)
        self.config = bert_config
        self.dropout = nn.Dropout(dropout)
        self.domain_names = list(domain_encoders.keys())
        H = bert_config.hidden_size

        self.interaction_layers = nn.ModuleDict({
            d: LatentInteraction(H, interaction_hidden) for d in self.domain_names
        })
        self.classifier = nn.Linear(interaction_hidden * len(self.domain_names), num_labels)
        self._init_weights()

    def _init_weights(self):
        std = self.config.initializer_range
        for layer in self.interaction_layers.values():
            nn.init.normal_(layer.linear.weight, std=std)
            nn.init.zeros_(layer.linear.bias)
        nn.init.normal_(self.classifier.weight, std=std)
        nn.init.zeros_(self.classifier.bias)

    def _mean_pool(self, seq, mask):
        m = mask.unsqueeze(-1).float()
        return (seq * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, input_ids, attention_mask, token_type_ids, target_mask,
                domain_input_ids, domain_attention_masks, labels=None):
        sent_out = self.sentence_encoder(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        ctx_emb = self._mean_pool(self.dropout(sent_out[0]), target_mask)

        interactions, distances = [], {}
        for domain in self.domain_names:
            dom_out = self.domain_encoders[domain](
                domain_input_ids[domain],
                attention_mask=domain_attention_masks[domain],
            )
            basic_emb = self._mean_pool(
                self.dropout(dom_out[0]), domain_attention_masks[domain]
            )
            distances[domain] = 1.0 - F.cosine_similarity(basic_emb, ctx_emb, dim=-1)
            interactions.append(
                self.dropout(self.interaction_layers[domain](basic_emb, ctx_emb))
            )

        logits = self.classifier(torch.cat(interactions, dim=-1))
        all_dist = torch.stack([distances[d] for d in self.domain_names], dim=-1)
        max_domain_idx = all_dist.argmax(dim=-1)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits, max_domain_idx
        return logits, max_domain_idx, distances


# ------------------------------------------------------------------
# Training utilities
# ------------------------------------------------------------------

def evaluate_metaphor(model, loader, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor)
                else {dk: dv.to(device) for dk, dv in v.items()}
                for k, v in batch.items() if k != "label"
            }
            loss, logits, _ = model(**inputs, labels=labels)
            total_loss += loss.item()
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    return total_loss / max(len(loader), 1), f1


def train_metaphor_model(model, train_loader, val_loader, device,
                         num_epochs=10, warmup_epochs=2, lr=2e-5,
                         save_path=None):
    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) * warmup_epochs

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1, best_state = 0.0, None

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            labels = batch["label"].to(device)
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor)
                else {dk: dv.to(device) for dk, dv in v.items()}
                for k, v in batch.items() if k != "label"
            }
            optimizer.zero_grad()
            loss, _, _ = model(**inputs, labels=labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        val_loss, val_f1 = evaluate_metaphor(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | val_F1={val_f1:.4f} "
            f"| lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        print(f"Best val F1: {best_f1:.4f}")
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Saved to {save_path}")
    return model, history


# ------------------------------------------------------------------
# Loading helpers
# ------------------------------------------------------------------

DOMAIN_NAMES = [
    "arzt", "assistent", "auszubildende", "autor", "dieb",
    "freund", "gott", "kind", "kollege", "künstler", "lehrer", "programmierer",
]


def load_domain_encoders(domain_ckpt_dir, tokenizer):
    """Load all 12 domain-fine-tuned BertModel encoders from checkpoint subdirs."""
    domain_encoders = {}
    for domain in DOMAIN_NAMES:
        save_dir = os.path.join(domain_ckpt_dir, domain)
        last_ckpt = _get_last_checkpoint(save_dir) if os.path.isdir(save_dir) else None
        if last_ckpt is None:
            raise FileNotFoundError(
                f"No checkpoint found for domain '{domain}' in {save_dir}. "
                "Run metaphors.ipynb to train domain models first."
            )
        mlm = AutoModelForMaskedLM.from_pretrained(last_ckpt)
        encoder = mlm.bert if hasattr(mlm, "bert") else mlm
        encoder.resize_token_embeddings(len(tokenizer))
        domain_encoders[domain] = encoder
    return domain_encoders


def load_metaphor_bert(
    metaphor_weights,
    domain_ckpt_dir,
    base_model_name="google-bert/bert-base-german-cased",
    device=None,
):
    """
    Build and return a ready-to-use MetaphorBERT model.

    Parameters
    ----------
    metaphor_weights : str
        Path to the saved state dict (.pt file from training).
    domain_ckpt_dir : str
        Directory containing one subfolder per domain (e.g. Metaphor_model/results).
    base_model_name : str
        HuggingFace model ID used as the sentence encoder.
    device : str | None
        Torch device string. Auto-detected if None.

    Returns
    -------
    model : MetaphorBERT  (eval mode, on device)
    tokenizer : AutoTokenizer (with POS special tokens)
    domain_names : list[str]
    """
    from transformers import AutoTokenizer

    if device is None:
        device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu"
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer = add_pos_tokens(tokenizer)

    bert_config = AutoConfig.from_pretrained(base_model_name)

    sentence_bert = AutoModel.from_pretrained(base_model_name)
    sentence_bert.resize_token_embeddings(len(tokenizer))

    domain_encoders = load_domain_encoders(domain_ckpt_dir, tokenizer)

    model = MetaphorBERT(
        sentence_encoder=sentence_bert,
        domain_encoders=domain_encoders,
        bert_config=bert_config,
    ).to(device)

    if os.path.isfile(metaphor_weights):
        state = torch.load(metaphor_weights, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights from {metaphor_weights}")
    else:
        print(f"Warning: no weights found at {metaphor_weights} — using random init.")

    model.eval()
    return model, tokenizer, DOMAIN_NAMES


# ------------------------------------------------------------------
# Inference helper
# ------------------------------------------------------------------

def predict_sentences(sentences, model, tokenizer, domain_names, nlp,
                      batch_size=8, device=None):
    """
    Run MetaphorBERT inference on a list of raw text sentences.

    Parameters
    ----------
    sentences : list[str]
    model : MetaphorBERT
    tokenizer : transformers tokenizer (with POS tokens added)
    domain_names : list[str]
    nlp : spaCy Language model (for POS tagging)
    batch_size : int
    device : str | None

    Returns
    -------
    results_df : pd.DataFrame
        Columns: sentence, word, word_idx, pos, prediction (0/1), dominant_domain
    """
    if device is None:
        device = next(model.parameters()).device

    rows = []
    for sent in sentences:
        doc = nlp(sent)
        pos_tags = [token.pos_ for token in doc]
        rows.append({"sentence": sent, "pos_tags": pos_tags})
    df = pd.DataFrame(rows)

    dataset = MetaphorDataset(df, tokenizer, domain_names, label_col="label")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_metaphor)

    all_preds, all_domains = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Predicting"):
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor)
                else {dk: dv.to(device) for dk, dv in v.items()}
                for k, v in batch.items() if k != "label"
            }
            logits, max_domain_idx, _ = model(**inputs)
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_domains.extend([domain_names[i] for i in max_domain_idx.cpu().tolist()])

    samples = dataset.samples
    results_df = pd.DataFrame([
        {
            "sentence": s["sentence"], "word": s["word"],
            "word_idx": s["word_idx"], "pos": s["pos"],
            "prediction": pred, "dominant_domain": dom,
        }
        for s, pred, dom in zip(samples, all_preds, all_domains)
    ])
    return results_df
