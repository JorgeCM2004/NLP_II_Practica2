from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .dense_retriever import DenseRetriever
from .summarizer import SummarizationExplainer


@dataclass
class ExplanationResult:
    text: str
    prediction: Any
    true_label: Optional[Any] = None
    neighbors: List[Dict[str, Any]] = field(default_factory=list)
    case_based_explanation: str = ""
    generated_explanation: str = ""
    confidence: float = 0.0
    is_correct: Optional[bool] = None


class ExplainabilityModule:
    def __init__(
        self,
        retriever: DenseRetriever,
        summarizer: Optional[SummarizationExplainer] = None
    ) -> None:

        self.retriever = retriever
        self.summarizer = summarizer
    
    def explain_prediction(
        self,
        text: str,
        prediction: Any,
        k: int = 5,
        true_label: Optional[Any] = None,
        confidence: float = 0.0,
        generate_summary: bool = True
    ) -> ExplanationResult:

        results = self.retriever.search([text], k=k)
        result = results[0]
        
        neighbors = []
        for i in range(len(result.indices)):
            neighbors.append({
                "index": int(result.indices[i]),
                "distance": float(result.distances[i]),
                "text": result.texts[i],
                "label": result.labels[i]
            })
        
        case_explanation = self.generate_case_based_reasoning(
            text, neighbors, prediction
        )
        
        generated_explanation = ""
        if generate_summary and self.summarizer is not None:
            generated_explanation = self.summarizer.generate_local_explanation(
                text, neighbors, str(prediction), k=k
            )
        
        is_correct = None
        if true_label is not None:
            is_correct = prediction == true_label
        
        return ExplanationResult(
            text=text,
            prediction=prediction,
            true_label=true_label,
            neighbors=neighbors,
            case_based_explanation=case_explanation,
            generated_explanation=generated_explanation,
            confidence=confidence,
            is_correct=is_correct
        )
    
    def generate_case_based_reasoning(
        self,
        text: str,
        neighbors: List[Dict[str, Any]],
        prediction: Any
    ) -> str:

        if not neighbors:
            return "No similar examples found in the training set."
        
        from collections import Counter
        label_counts = Counter(n['label'] for n in neighbors)
        
        target_words = set(text.lower().split())
        common_features = []
        
        for neighbor in neighbors[:3]:
            neighbor_words = set(neighbor['text'].lower().split())
            overlap = target_words & neighbor_words
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 
                        'been', 'have', 'has', 'had', 'do', 'does', 'did',
                        'and', 'or', 'but', 'if', 'then', 'in', 'on', 'at',
                        'to', 'for', 'of', 'with', 'by', 'from', 'as', 'it',
                        'this', 'that', 'they', 'them', 'their', 'i', 'we',
                        'you', 'he', 'she', 'my', 'your', 'his', 'her'}
            meaningful_overlap = overlap - stopwords
            common_features.extend(list(meaningful_overlap)[:3])
        
        explanation_parts = []
        
        most_common = label_counts.most_common(3)
        label_str = ", ".join([f"{count} labeled as '{label}'" for label, count in most_common])
        explanation_parts.append(
            f"Among the {len(neighbors)} most similar training examples, {label_str}."
        )
        
        if common_features:
            unique_features = list(set(common_features))[:5]
            features_str = "', '".join(unique_features)
            explanation_parts.append(
                f"Common terms include: '{features_str}'."
            )
        
        if label_counts.get(prediction, 0) > 0:
            support = label_counts[prediction]
            explanation_parts.append(
                f"The prediction '{prediction}' is supported by {support}/{len(neighbors)} similar examples."
            )
        else:
            explanation_parts.append(
                f"Interestingly, none of the nearest neighbors share the predicted label '{prediction}'."
            )
        
        return " ".join(explanation_parts)
    
    def explain_batch(
        self,
        texts: List[str],
        predictions: List[Any],
        k: int = 5,
        true_labels: Optional[List[Any]] = None,
        generate_summaries: bool = False
    ) -> List[ExplanationResult]:

        results = []
        
        for i, (text, pred) in enumerate(zip(texts, predictions)):
            true_label = true_labels[i] if true_labels else None
            result = self.explain_prediction(
                text=text,
                prediction=pred,
                k=k,
                true_label=true_label,
                generate_summary=generate_summaries
            )
            results.append(result)
        
        return results
    
    def select_examples_for_analysis(
        self,
        texts: List[str],
        predictions: List[Any],
        true_labels: List[Any],
        num_correct: int = 10,
        num_incorrect: int = 10
    ) -> Dict[str, List[int]]:

        correct_indices = []
        incorrect_indices = []
        
        for i, (pred, true) in enumerate(zip(predictions, true_labels)):
            if pred == true:
                correct_indices.append(i)
            else:
                incorrect_indices.append(i)
        
        import random
        if len(correct_indices) > num_correct:
            correct_indices = random.sample(correct_indices, num_correct)
        if len(incorrect_indices) > num_incorrect:
            incorrect_indices = random.sample(incorrect_indices, num_incorrect)
        
        return {
            "correct": correct_indices,
            "incorrect": incorrect_indices
        }
    
    def format_explanation_for_display(
        self,
        result: ExplanationResult,
        show_neighbors: int = 3
    ) -> str:

        lines = []
        lines.append("=" * 60)
        
        if result.is_correct is not None:
            status = "✓ CORRECT" if result.is_correct else "✗ INCORRECT"
            lines.append(f"Status: {status}")
        
        text_display = result.text[:200] + "..." if len(result.text) > 200 else result.text
        lines.append(f"\nText: {text_display}")
        
        lines.append(f"\nPrediction: {result.prediction}")
        if result.true_label is not None:
            lines.append(f"True Label: {result.true_label}")
        
        lines.append(f"\nSimilar Training Examples ({show_neighbors}):")
        for i, neighbor in enumerate(result.neighbors[:show_neighbors], 1):
            neighbor_text = neighbor['text'][:100] + "..." if len(neighbor['text']) > 100 else neighbor['text']
            lines.append(f"  [{i}] ({neighbor['label']}): {neighbor_text}")
        
        lines.append(f"\nCase-Based Explanation:")
        lines.append(f"  {result.case_based_explanation}")
        
        if result.generated_explanation:
            lines.append(f"\nGenerated Explanation:")
            lines.append(f"  {result.generated_explanation}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
