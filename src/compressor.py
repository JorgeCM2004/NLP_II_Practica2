from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


class ModelCompressor:    
    def __init__(self,teacher_model: Any, student_model_name: str = "distilbert-base-uncased",num_labels: int = 2,device: Optional[str] = None) -> None:
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.teacher_model = teacher_model
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        
        self.student_model_name = student_model_name
        self.num_labels = num_labels
        
        self.student_model = AutoModelForSequenceClassification.from_pretrained(
            student_model_name,
            num_labels=num_labels
        )
        self.student_model.to(self.device)
        
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_macro_f1": [],
        }
    
    def distill(self,train_dataset: Any,val_dataset: Any,epochs: int = 3,batch_size: int = 16,
    learning_rate: float = 2e-5,temperature: float = 2.0,alpha: float = 0.5, save_path: Optional[str] = None) -> Any:

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        optimizer = AdamW(
            self.student_model.parameters(),
            lr=learning_rate
        )
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        best_val_f1 = 0.0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            train_loss = self._distill_epoch(
                train_loader, optimizer, scheduler, temperature, alpha
            )
            
            val_loss, val_acc, val_f1 = self._evaluate(val_loader)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)
            self.history["val_macro_f1"].append(val_f1)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                if save_path:
                    self.save_student(save_path)
                    print(f"✓ Saved best student (F1: {val_f1:.4f})")
        
        return self.student_model
    
    def _distill_epoch(
        self,
        train_loader: DataLoader,
        optimizer: Any,
        scheduler: Any,
        temperature: float,
        alpha: float
    ) -> float:
        self.student_model.train()
        self.teacher_model.eval()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Distilling")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits
            
            student_outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            student_logits = student_outputs.logits
            
            loss = self._distillation_loss(
                student_logits, teacher_logits, labels, temperature, alpha
            )
            
            total_loss += loss.item()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            pbar.set_postfix({"loss": loss.item()})
        
        return total_loss / len(train_loader)
    
    def _distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        alpha: float
    ) -> torch.Tensor:

        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        soft_loss = soft_loss * (temperature ** 2)
        
        hard_loss = F.cross_entropy(student_logits, labels)
        
        loss = alpha * soft_loss + (1 - alpha) * hard_loss
        
        return loss
    
    def _evaluate(
        self,
        data_loader: DataLoader
    ) -> Tuple[float, float, float]:

        self.student_model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.student_model(
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
    
    def compare_models(
        self,
        test_dataset: Any,
        texts_for_timing: Optional[List[str]] = None,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        student_loss, student_acc, student_f1 = self._evaluate(test_loader)
        
        self.teacher_model.eval()
        teacher_preds = []
        teacher_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=-1)
                teacher_preds.extend(preds.cpu().numpy())
                teacher_labels.extend(labels.numpy())
        
        teacher_acc = accuracy_score(teacher_labels, teacher_preds)
        teacher_f1 = f1_score(teacher_labels, teacher_preds, average='macro')
        
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())
        
        comparison = {
            "teacher": {
                "accuracy": teacher_acc,
                "macro_f1": teacher_f1,
                "parameters": teacher_params,
                "size_mb": teacher_params * 4 / (1024 ** 2),  # Approximate
            },
            "student": {
                "accuracy": student_acc,
                "macro_f1": student_f1,
                "parameters": student_params,
                "size_mb": student_params * 4 / (1024 ** 2),
            },
            "compression_ratio": teacher_params / student_params,
            "accuracy_drop": teacher_acc - student_acc,
            "f1_drop": teacher_f1 - student_f1,
        }
        
        if texts_for_timing:
            comparison["teacher"]["inference_time"] = self._measure_inference_time(
                self.teacher_model, None, texts_for_timing, batch_size
            )
            comparison["student"]["inference_time"] = self._measure_inference_time(
                self.student_model, self.student_tokenizer, texts_for_timing, batch_size
            )
            comparison["speedup"] = (
                comparison["teacher"]["inference_time"] /
                comparison["student"]["inference_time"]
            )
        
        return comparison
    
    def _measure_inference_time(
        self,
        model: Any,
        tokenizer: Any,
        texts: List[str],
        batch_size: int
    ) -> float:
        model.eval()
        
        if tokenizer is None:
            tokenizer = self.student_tokenizer
        
        start = time.time()
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            encodings = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                model(
                    input_ids=encodings['input_ids'].to(self.device),
                    attention_mask=encodings['attention_mask'].to(self.device)
                )
        
        return time.time() - start
    
    def save_student(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.student_model.save_pretrained(path)
        self.student_tokenizer.save_pretrained(path)
    
    def get_student(self) -> Any:
        return self.student_model
    
    def get_student_tokenizer(self) -> Any:
        return self.student_tokenizer
