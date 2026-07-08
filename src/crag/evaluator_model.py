import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

from src.utils.logger import setup_logger

logger = setup_logger("rags.crag.evaluator")

class CRAGEvaluator:
    def __init__(self, model_path: str = "models/crag_evaluator"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.model.eval()
            self.loaded = True
        except Exception as e:
            logger.warning(f"Could not load CRAG model from {model_path}: {e}")
            self.loaded = False
            
        # Standard mapping for the 3 classes
        self.id2label = {0: "Correct", 1: "Ambiguous", 2: "Incorrect"}

    def evaluate(self, query: str, chunk_text: str) -> str:
        """
        Evaluate if a chunk can answer the query.
        Returns: 'Correct', 'Ambiguous', or 'Incorrect'
        """
        if not self.loaded:
            # Silent fallback if model is not trained yet
            return "Correct"
            
        inputs = self.tokenizer(
            query, 
            chunk_text, 
            truncation=True, 
            padding=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            
        return self.id2label[preds.item()]