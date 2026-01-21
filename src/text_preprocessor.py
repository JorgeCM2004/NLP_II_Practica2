import re
from typing import List, Optional, Dict, Any, Union

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, BatchEncoding


class TextDataset(Dataset):
    def __init__(self,encodings: BatchEncoding, labels: Optional[List[int]] = None) -> None:
        self.encodings = encodings
        self.labels = labels
        
    def __len__(self) -> int:
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        
        return item


class TextPreprocessor:
    def __init__(self,model_name: str = "roberta-base",max_length: int = 512,lowercase: bool = False) -> None:

        self.model_name = model_name
        self.max_length = max_length
        self.lowercase = lowercase
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        text = re.sub(r'http\S+|www\.\S+', '[URL]', text) 
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        if self.lowercase:
            text = text.lower()
        
        return text
    
    def clean_texts(self, texts: List[str]) -> List[str]:
        return [self.clean_text(text) for text in texts]
    
    def tokenize(self, texts: List[str], clean: bool = True, **kwargs: Any) -> BatchEncoding:
        if clean:
            texts = self.clean_texts(texts)

        encodings = self.tokenizer(texts,padding=True, truncation=True, max_length=self.max_length,return_tensors=None, **kwargs)
        
        return encodings
    
    def prepare_dataset(self,texts: List[str],labels: Optional[List[int]] = None,clean: bool = True) -> TextDataset:
        encodings = self.tokenize(texts, clean=clean)
        return TextDataset(encodings, labels)
    
    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self.tokenizer
    
    def decode(self, token_ids: Union[List[int], torch.Tensor],  skip_special_tokens: bool = True) -> str:

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
    
    def get_vocab_size(self) -> int:
        return len(self.tokenizer)
