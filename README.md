# NLP II Práctica 2

# 1. Instalación

```bash

```

# 3. Probar la practica

Para probar los resultados simplemenete se deberá ejecutar el notebook llamado `Notebook_P2.ipynb`.

## 3.Estructura del proyecto

```
NLP_II_Practica2/
├── src/   
│   ├── __init__.py
│   ├── data_loader.py   
│   ├── text_preprocessor.py   
│   ├── embedding_extractor.py  
│   ├── dense_retriever.py   
│   ├── knn_classifier.py  
│   ├── hybrid_classifier.py   
│   ├── model_trainer.py   
│   ├── compressor.py   
│   ├── summarizer.py  
│   ├── explainability.py   
│   └── evaluator.py  
├── notebooks/
│   └── Notebook_P2.ipynb  
├── models/   
├── data/
└── README.md
└── pyproject.toml
```

## Módulos implementados

| Módulo                    | Descripción                                  |
| -------------------------- | --------------------------------------------- |
| `DataLoader`             | Descarga y armoniza los datos                 |
| `TextPreprocessor`       | Tokenization and text cleaning                |
| `EmbeddingExtractor`     | Dense embeddings from Transformer encoder     |
| `DenseRetriever`         | k-NN index for similarity search              |
| `KNNClassifier`          | Classification via majority voting            |
| `HybridClassifier`       | Transformer + k-NN combination (α parameter) |
| `ModelTrainer`           | Training with logging and checkpoints         |
| `ModelCompressor`        | Knowledge distillation                        |
| `SummarizationExplainer` | T5/BART for generating explanations           |
| `ExplainabilityModule`   | Case-based reasoning + LLM explanations       |
| `Evaluator`              | Metrics, confusion matrices, plots            |

## 🧪 Experiments

The notebook covers:

1. **§4.1 Dense Retrieval**: Build index, evaluate Precision@k, Recall@k
2. **§4.2 k-NN Classifier**: Majority voting, compare with baselines
3. **§4.3 Hybrid RAG**: Experiment with α values (0.0 → 1.0)
4. **§4.4 Explainability**: Case-based reasoning for 20 examples
5. **§4.5 Compression**: DistilBERT vs RoBERTa (speed/quality)
6. **§4.6 Summarization**: Global class summaries, local explanations

## ⏱️ Estimated Runtime

| Phase                               | Time (GPU)           |
| ----------------------------------- | -------------------- |
| Data loading & preprocessing        | ~2 min               |
| Embedding extraction                | ~10-15 min           |
| k-NN index building                 | ~1-2 min             |
| Distilled model training (3 epochs) | ~20-30 min           |
| Summarization                       | ~5-10 min            |
| **Total**                     | **~40-60 min** |

## 🛠️ Models Used

| Role       | Model                         |
| ---------- | ----------------------------- |
| Teacher    | `roberta-base` (fine-tuned) |
| Student    | `distilbert-base-uncased`   |
| Summarizer | `t5-small`                  |

## 📊 Metrics

- Accuracy
- Macro F1
- Per-class F1
- Precision@k / Recall@k (retrieval)
- Inference time (ms/sample)
- Model size (MB)
