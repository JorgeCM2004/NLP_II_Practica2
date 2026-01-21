"""
SummarizationExplainer Module - Generates explanations using abstractive summarization.

This module provides the SummarizationExplainer class for generating:
- Global class summaries
- Local neighbor-based explanations
"""

from typing import List, Dict, Any, Optional

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)


class SummarizationExplainer:
    def __init__(
        self,
        model_name: str = "t5-small",
        device: Optional[str] = None,
        max_input_length: int = 1024,
        max_output_length: int = 150
    ) -> None:

        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.summarizer = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
    
    def generate_class_summary(
        self,
        texts: List[str],
        class_name: str,
        num_samples: int = 50,
        separator: str = " --- "
    ) -> str:

        if len(texts) > num_samples:
            import random
            texts = random.sample(texts, num_samples)
        
        max_chars_per_text = max(50, self.max_input_length * 4 // num_samples)
        truncated_texts = [
            text[:max_chars_per_text] + "..." if len(text) > max_chars_per_text else text
            for text in texts
        ]
        
        pseudo_doc = separator.join(truncated_texts)
        
        if "t5" in self.model_name.lower():
            prompt = f"summarize: These are examples of '{class_name}' reviews: {pseudo_doc}"
        else:
            prompt = pseudo_doc
        
        tokens = self.tokenizer.encode(prompt, truncation=True, max_length=self.max_input_length)
        prompt = self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        summary = self._generate(prompt)
        
        return summary
    
    def generate_local_explanation(
        self,
        target_text: str,
        neighbors: List[Dict[str, Any]],
        predicted_class: str,
        k: int = 5
    ) -> str:

        neighbors = neighbors[:k]
        
        neighbor_str = ""
        for i, neighbor in enumerate(neighbors, 1):
            label = neighbor.get('label', 'unknown')
            text = neighbor.get('text', '')[:200]  # Truncate long texts
            neighbor_str += f"[{i}] ({label}): {text}\n"
        
        prompt = f"""Target text:
                {target_text[:300]}

                Similar training examples:
                {neighbor_str}

                Write a brief explanation of why this text might belong to class '{predicted_class}' based on the similar examples above:"""
                        
        if "t5" in self.model_name.lower():
            prompt = f"explain: {prompt}"
        
        explanation = self._generate(
            prompt,
            max_length=100,
            min_length=20
        )
        
        return explanation
    
    def generate_comparison_explanation(
        self,
        target_text: str,
        similar_texts: List[str],
        similar_labels: List[str],
        highlight_features: Optional[List[str]] = None
    ) -> str:

        examples = []
        for text, label in zip(similar_texts[:3], similar_labels[:3]):
            examples.append(f"({label}): {text[:150]}...")
        
        prompt = f"""Compare this text with similar examples:

                Text to analyze: {target_text[:200]}

                Similar examples:
                {chr(10).join(examples)}

                Summarize what they have in common:"""
                        
        if "t5" in self.model_name.lower():
            prompt = f"summarize: {prompt}"
        
        return self._generate(prompt, max_length=80)
    
    def _generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        min_length: int = 30
    ) -> str:

        if max_length is None:
            max_length = self.max_output_length
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        generated = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return generated.strip()
    
    def analyze_hallucinations(
        self,
        generated_text: str,
        source_texts: List[str]
    ) -> Dict[str, Any]:

        all_source = " ".join(source_texts).lower()
        generated_lower = generated_text.lower()
        
        import re
        generated_words = set(re.findall(r'\b\w+\b', generated_lower))
        source_words = set(re.findall(r'\b\w+\b', all_source))
        
        novel_words = generated_words - source_words
        common_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                       "being", "have", "has", "had", "do", "does", "did", "will",
                       "would", "could", "should", "may", "might", "can", "this",
                       "that", "these", "those", "it", "its", "they", "them", "their",
                       "and", "or", "but", "if", "then", "else", "when", "where",
                       "which", "who", "what", "how", "why", "all", "each", "every",
                       "both", "few", "more", "most", "other", "some", "such", "no",
                       "nor", "not", "only", "own", "same", "so", "than", "too",
                       "very", "just", "also", "with", "from", "for", "to", "in",
                       "on", "at", "by", "of", "as", "about", "into", "through"}
        
        potential_hallucinations = novel_words - common_words
        
        return {
            "novel_word_count": len(potential_hallucinations),
            "novel_words": list(potential_hallucinations)[:10],
            "source_coverage": 1 - len(potential_hallucinations) / max(len(generated_words), 1),
            "warning": len(potential_hallucinations) > 5
        }
