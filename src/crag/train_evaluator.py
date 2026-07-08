import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from src.utils.logger import setup_logger

logger = setup_logger("rags.crag.train")

class CRAGDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    
    # We want to print confusion matrix for logging
    logger.info(f"\nConfusion Matrix:\n{cm}")
    return {'accuracy': acc}

def train_crag_evaluator(
    data_path: str = "data/crag_bootstrap.jsonl",
    model_out: str = "models/crag_evaluator",
    base_model: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16
):
    if not Path(data_path).exists():
        logger.error(f"Data file {data_path} not found. Run bootstrap_data.py first.")
        return

    queries = []
    chunks = []
    labels = []
    
    label2id = {"Correct": 0, "Ambiguous": 1, "Incorrect": 2}

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            queries.append(d["query"])
            chunks.append(d["chunk_text"])
            labels.append(label2id[d["label"]])

    if not queries:
        logger.error("No data found in JSONL file.")
        return
        
    logger.info(f"Loaded {len(queries)} samples.")

    # 70/30 split
    q_train, q_val, c_train, c_val, l_train, l_val = train_test_split(
        queries, chunks, labels, test_size=0.3, random_state=42, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    train_encodings = tokenizer(q_train, c_train, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(q_val, c_val, truncation=True, padding=True, max_length=512)

    train_dataset = CRAGDataset(train_encodings, l_train)
    val_dataset = CRAGDataset(val_encodings, l_val)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, 
        num_labels=3,
        id2label={v: k for k, v in label2id.items()},
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Evaluating on 30% holdout set...")
    eval_results = trainer.evaluate()
    logger.info(f"Eval Results: {eval_results}")

    logger.info(f"Saving model to {model_out}")
    model.save_pretrained(model_out)
    tokenizer.save_pretrained(model_out)

if __name__ == "__main__":
    train_crag_evaluator()
