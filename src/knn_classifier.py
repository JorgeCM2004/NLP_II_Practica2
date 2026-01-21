from typing import List, Tuple, Optional, Dict, Any
from collections import Counter

import numpy as np

from .dense_retriever import DenseRetriever


class KNNClassifier:    
    def __init__(self,retriever: DenseRetriever, k: int = 5, num_classes: Optional[int] = None,weighted: bool = False) -> None:

        self.retriever = retriever
        self.k = k
        self.weighted = weighted
        
        if num_classes is None:
            unique_labels = set(retriever.corpus_labels)
            self.num_classes = len(unique_labels)
            self.label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
            self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        else:
            self.num_classes = num_classes
            self.label_to_idx = {i: i for i in range(num_classes)}
            self.idx_to_label = {i: i for i in range(num_classes)}
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, List[Dict]]:
        
        results = self.retriever.search(texts, k=self.k)
    
        predictions = []
        neighbor_info = []
        
        for result in results:
            if self.weighted:
                weights = 1 / (result.distances + 1e-6)
                label_weights = Counter()
                for label, weight in zip(result.labels, weights):
                    label_weights[label] += weight
                predicted_label = label_weights.most_common(1)[0][0]
            else:
                label_counts = Counter(result.labels)
                predicted_label = label_counts.most_common(1)[0][0]
            
            predictions.append(predicted_label)
            
            neighbor_info.append({
                "indices": result.indices.tolist(),
                "distances": result.distances.tolist(),
                "texts": result.texts,
                "labels": result.labels,
                "label_distribution": dict(Counter(result.labels))
            })
        
        return np.array(predictions), neighbor_info
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        results = self.retriever.search(texts, k=self.k)
        
        probabilities = np.zeros((len(texts), self.num_classes))
        
        for i, result in enumerate(results):
            if self.weighted:
                weights = 1 / (result.distances + 1e-6)
                total_weight = np.sum(weights)
                
                for label, weight in zip(result.labels, weights):
                    if label in self.label_to_idx:
                        idx = self.label_to_idx[label]
                        probabilities[i, idx] += weight / total_weight
            else:
                for label in result.labels:
                    if label in self.label_to_idx:
                        idx = self.label_to_idx[label]
                        probabilities[i, idx] += 1 / self.k
        
        return probabilities
    
    def set_k(self, k: int) -> None:
        self.k = k
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "num_classes": self.num_classes,
            "weighted": self.weighted,
            "metric": self.retriever.metric,
        }
