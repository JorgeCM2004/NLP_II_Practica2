from sklearn.metrics import accuracy_score, f1_score
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .knn_classifier import KNNClassifier


class HybridClassifier:
    def __init__(self,teacher_model: Any, tokenizer: Any, knn_classifier: KNNClassifier,
    alpha: float = 0.5,device: Optional[str] = None,max_length: int = 512) -> None:

        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.knn_classifier = knn_classifier
        self.alpha = alpha
        self.max_length = max_length
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        
        self.num_classes = knn_classifier.num_classes
    
    def _get_transformer_proba(self,texts: List[str],batch_size: int = 32) -> np.ndarray:
        all_probs = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            encodings = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            all_probs.append(probs.cpu().numpy())
        
        return np.concatenate(all_probs, axis=0)
    
    def predict(self,texts: List[str], return_details: bool = False) -> Any:

        combined_proba = self.get_combined_proba(texts)
        
        predictions = np.argmax(combined_proba, axis=1)
        
        predictions = np.array([self.knn_classifier.idx_to_label[p] for p in predictions])
        
        if return_details:
            transformer_proba = self._get_transformer_proba(texts)
            knn_proba = self.knn_classifier.predict_proba(texts)
            _, neighbor_info = self.knn_classifier.predict(texts)
            
            details = {
                "transformer_proba": transformer_proba,
                "knn_proba": knn_proba,
                "combined_proba": combined_proba,
                "neighbor_info": neighbor_info,
                "alpha": self.alpha,
            }
            return predictions, details
        
        return predictions
    
    def get_combined_proba(self, texts: List[str]) -> np.ndarray:
        p_transformer = self._get_transformer_proba(texts)
        
        p_knn = self.knn_classifier.predict_proba(texts)
        
        p_combined = self.alpha * p_transformer + (1 - self.alpha) * p_knn
        
        return p_combined
    
    def set_alpha(self, alpha: float) -> None:
        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
    
    def experiment_alpha_values(
        self,
        texts: List[str],
        true_labels: List[Any],
        alpha_values: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
    ) -> Dict[float, Dict[str, float]]:

        
        results = {}
        
        p_transformer = self._get_transformer_proba(texts)
        p_knn = self.knn_classifier.predict_proba(texts)
        
        for alpha in alpha_values:
            p_combined = alpha * p_transformer + (1 - alpha) * p_knn
            
            pred_indices = np.argmax(p_combined, axis=1)
            predictions = [
                self.knn_classifier.idx_to_label[p] for p in pred_indices
            ]
            
            accuracy = accuracy_score(true_labels, predictions)
            macro_f1 = f1_score(true_labels, predictions, average='macro')
            
            results[alpha] = {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        
        return results
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "k": self.knn_classifier.k,
            "num_classes": self.num_classes,
            "teacher_model": self.teacher_model.config._name_or_path,
        }
