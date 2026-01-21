from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score


@dataclass
class RetrievalResult:
    indices: np.ndarray
    distances: np.ndarray
    texts: List[str]
    labels: List[Any]


class DenseRetriever:
    def __init__(self, embedding_extractor: Any, metric: str = "cosine", use_faiss: bool = False) -> None:
        self.embedding_extractor = embedding_extractor
        self.metric = metric
        self.use_faiss = use_faiss
        
        self.index: Optional[NearestNeighbors] = None
        self.corpus_embeddings: Optional[np.ndarray] = None
        self.corpus_texts: List[str] = []
        self.corpus_labels: List[Any] = []
        self.corpus_indices: List[int] = []
        
        self._faiss_index = None
        
    def build_index(self, texts: List[str], labels: List[Any], show_progress: bool = True) -> None:
        self.corpus_texts = texts
        self.corpus_labels = labels
        self.corpus_indices = list(range(len(texts)))
        
        print(f"Generating embeddings for {len(texts)} texts...")
        self.corpus_embeddings = self.embedding_extractor.embed_fn(texts,show_progress=show_progress)
        
        if self.use_faiss:
            self._build_faiss_index()
        else:
            self._build_sklearn_index()
        
        print(f"Index built with {len(texts)} documents.")
    
    def _build_sklearn_index(self) -> None:
        self.index = NearestNeighbors(
            metric=self.metric,
            algorithm='auto',
            n_jobs=-1
        )
        self.index.fit(self.corpus_embeddings)
    
    def _build_faiss_index(self) -> None:
        try:
            import faiss
            d = self.corpus_embeddings.shape[1]
            
            if self.metric == "cosine":
                embeddings = self.corpus_embeddings.copy()
                faiss.normalize_L2(embeddings)
                self._faiss_index = faiss.IndexFlatIP(d)
            else:
                self._faiss_index = faiss.IndexFlatL2(d)
                embeddings = self.corpus_embeddings
            
            self._faiss_index.add(embeddings.astype(np.float32))
            
        except ImportError:
            print("FAISS not available, falling back to sklearn.")
            self.use_faiss = False
            self._build_sklearn_index()
    
    def search(self, query_texts: List[str], k: int = 5) -> List[RetrievalResult]:

        query_embeddings = self.embedding_extractor.embed_fn(
            query_texts,
            show_progress=False
        )
        
        if self.use_faiss and self._faiss_index is not None:
            return self._search_faiss(query_embeddings, k)
        else:
            return self._search_sklearn(query_embeddings, k)
    
    def _search_sklearn(self, query_embeddings: np.ndarray, k: int) -> List[RetrievalResult]:
        distances, indices = self.index.kneighbors(
            query_embeddings,
            n_neighbors=k
        )
        
        results = []
        for i in range(len(query_embeddings)):
            neighbor_indices = indices[i]
            neighbor_distances = distances[i]
            neighbor_texts = [self.corpus_texts[j] for j in neighbor_indices]
            neighbor_labels = [self.corpus_labels[j] for j in neighbor_indices]
            
            results.append(RetrievalResult(
                indices=neighbor_indices,
                distances=neighbor_distances,
                texts=neighbor_texts,
                labels=neighbor_labels
            ))
        
        return results
    
    def _search_faiss(
        self,
        query_embeddings: np.ndarray,
        k: int
    ) -> List[RetrievalResult]:
        import faiss
        
        queries = query_embeddings.astype(np.float32)
        
        if self.metric == "cosine":
            faiss.normalize_L2(queries)
        
        distances, indices = self._faiss_index.search(queries, k)
        
        results = []
        for i in range(len(query_embeddings)):
            neighbor_indices = indices[i]
            neighbor_distances = distances[i]
            neighbor_texts = [self.corpus_texts[j] for j in neighbor_indices]
            neighbor_labels = [self.corpus_labels[j] for j in neighbor_indices]
            
            results.append(RetrievalResult(
                indices=neighbor_indices,
                distances=neighbor_distances,
                texts=neighbor_texts,
                labels=neighbor_labels
            ))
        
        return results
    
    def evaluate_retrieval(
        self,
        test_texts: List[str],
        test_labels: List[Any],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Dict[str, float]]:

        results = {}
        
        for k in k_values:
            retrieval_results = self.search(test_texts, k=k)
            
            precision_scores = []
            recall_at_k = []
            
            for i, (result, true_label) in enumerate(
                zip(retrieval_results, test_labels)
            ):
                matching = sum(
                    1 for label in result.labels if label == true_label
                )
                
                precision_scores.append(matching / k)
                
                recall_at_k.append(1.0 if matching > 0 else 0.0)
            
            results[f"k={k}"] = {
                "precision@k": np.mean(precision_scores),
                "recall@k": np.mean(recall_at_k),
            }
        
        return results
    
    def get_corpus_stats(self) -> Dict[str, Any]:
        unique_labels = set(self.corpus_labels)
        label_counts = {
            label: self.corpus_labels.count(label)
            for label in unique_labels
        }
        
        return {
            "num_documents": len(self.corpus_texts),
            "embedding_dim": self.corpus_embeddings.shape[1] if self.corpus_embeddings is not None else 0,
            "num_labels": len(unique_labels),
            "label_distribution": label_counts,
            "metric": self.metric,
        }
