from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)


class EmbeddingExtractor:
    def __init__(self,model_name_or_model: Union[str, PreTrainedModel] = "roberta-base",
    tokenizer: Optional[PreTrainedTokenizer] = None,device: Optional[str] = None,pooling_strategy: str = "cls",
    max_length: int = 512) -> None:

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        if isinstance(model_name_or_model, str):
            self.model = AutoModel.from_pretrained(model_name_or_model)
            self.model_name = model_name_or_model
        else:
            self.model = model_name_or_model
            self.model_name = getattr(model_name_or_model.config, '_name_or_path', 'custom')
        
        self.model.to(self.device)
        self.model.eval()
        
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name if isinstance(model_name_or_model, str) 
                else model_name_or_model.config._name_or_path
            )
        else:
            self.tokenizer = tokenizer
        
        self.pooling_strategy = pooling_strategy
        self.max_length = max_length
        
    def _get_cls_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            last_hidden_state = outputs.last_hidden_state

            cls_embeddings = last_hidden_state[:, 0, :]
        
        return cls_embeddings
    
    def _get_mean_embeddings(self,input_ids: torch.Tensor,attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            last_hidden_state = outputs.last_hidden_state
            
            mask_expanded = attention_mask.unsqueeze(-1).expand(
                last_hidden_state.size()
            ).float()
            
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            
            mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings
    
    def embed_fn(self,texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        all_embeddings = []
        
        n_batches = (len(texts) + batch_size - 1) // batch_size
        iterator = range(0, len(texts), batch_size)
        
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting embeddings", total=n_batches)
        
        for i in iterator:
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
            
            if self.pooling_strategy == "cls":
                embeddings = self._get_cls_embeddings(input_ids, attention_mask)
            else:
                embeddings = self._get_mean_embeddings(input_ids, attention_mask)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.concatenate(all_embeddings, axis=0)
    
    def get_embedding_dimension(self) -> int:
        return self.model.config.hidden_size
    
    def set_pooling_strategy(self, strategy: str) -> None:

        if strategy not in ['cls', 'mean']:
            raise ValueError(f"Invalid pooling strategy: {strategy}")
        self.pooling_strategy = strategy
