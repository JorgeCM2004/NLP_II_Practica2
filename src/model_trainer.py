from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import time
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, PreTrainedModel, PreTrainedTokenizer)
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


class ModelTrainer:
    def __init__(self, model_name: str = "roberta-base", num_labels: int = 2, device: Optional[str] = None, label_names: Optional[List[str]] = None) -> None:

        self.model_name = model_name
        self.num_labels = num_labels
        self.label_names = label_names
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels)
        self.model.to(self.device)
        
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "val_macro_f1": [],
        }
    
    def train(self,train_dataset: Any,val_dataset: Any, epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5, warmup_ratio: float = 0.1, weight_decay: float = 0.01, save_best: bool = True, save_path: Optional[str] = None) -> Dict[str, List[float]]:

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = AdamW(self.model.parameters(),lr=learning_rate,weight_decay=weight_decay)
        
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=total_steps)
        
        best_val_f1 = 0.0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, scheduler)
            
            val_loss, val_acc, val_f1 = self._evaluate(val_loader)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_accuracy"].append(val_acc)
            self.history["val_macro_f1"].append(val_f1)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            if save_best and val_f1 > best_val_f1:
                best_val_f1 = val_f1
                if save_path:
                    self.save_model(save_path)
                    print(f"✓ Saved best model (F1: {val_f1:.4f})")
        
        return self.history
    
    def _train_epoch(self,train_loader: DataLoader, optimizer: Any, scheduler: Any) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def _evaluate(self,data_loader: DataLoader) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, accuracy, macro_f1
    
    def evaluate(self, test_dataset: Any, batch_size: int = 32) -> Dict[str, float]:
        test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
        
        loss, accuracy, macro_f1 = self._evaluate(test_loader)
        
        return {
            "test_loss": loss,
            "test_accuracy": accuracy,
            "test_macro_f1": macro_f1,
        }
    
    def predict(self,texts: List[str],batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_preds = []
        all_probs = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            encodings = self.tokenizer(batch_texts,padding=True,truncation=True,max_length=512,return_tensors="pt")
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids,attention_mask=attention_mask)  
                probs = torch.softmax(outputs.logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        return np.array(all_preds), np.concatenate(all_probs, axis=0)
    
    def save_model(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
        print(f"Model loaded from {path}")
    
    def get_model(self) -> PreTrainedModel:
        return self.model
    
    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self.tokenizer
    
    def measure_inference_time(self,texts: List[str], batch_size: int = 32, num_runs: int = 5) -> Dict[str, float]:
        times = []
        
        for _ in range(num_runs):
            start = time.time()
            self.predict(texts, batch_size=batch_size)
            end = time.time()
            times.append(end - start)
        
        return {
            "mean_time_seconds": np.mean(times),
            "std_time_seconds": np.std(times),
            "samples_per_second": len(texts) / np.mean(times),
            "batch_size": batch_size,
            "num_samples": len(texts),
        }
