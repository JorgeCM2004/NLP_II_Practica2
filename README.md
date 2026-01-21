# NLP II Práctica 2

# 1. Instalación y Configuración

Este proyecto utiliza **uv**, un gestor de paquetes de Python moderno y extremadamente rápido. Se encarga de gestionar tanto la versión de Python como las dependencias del proyecto automáticamente.

### 1. Instalar uv
Si aún no tienes `uv` instalado, ejecuta el siguiente comando en tu terminal dependiendo de tu sistema operativo:

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
```

**macOS / Linux:**
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

> **Nota:** Es posible que debas reiniciar la terminal después de la instalación para que reconozca el comando.

### 2. Sincronizar el entorno
Una vez tengas `uv` instalado y hayas descargado el código del proyecto, navega hasta la carpeta raíz del proyecto y ejecuta:

```bash
uv sync
```

Este comando realizará automáticamente los siguientes pasos:
1. Leerá el archivo `.python-version` y descargará la versión correcta de Python si no la tienes.
2. Creará el entorno virtual (`.venv`) de manera aislada.
3. Instalará todas las dependencias exactas definidas en el archivo `uv.lock`.

### 3. Ejecutar el proyecto
Con el entorno sincronizado, se te creará un entorno virtual:
* **Windows:** `.venv\Scripts\activate`
* **Mac/Linux:** `source .venv/bin/activate`

# 2. Probar la practica

Para probar los resultados simplemenete se deberá ejecutar el notebook llamado `Notebook_P2.ipynb`.

# 3. Estructura del proyecto

```
NLP_II_Practica2/
├── data/
│   └── data.csv
├── src/
│   ├── __init__.py
│   ├── compressor.py
│   ├── data_loader.py
│   ├── dense_retriever.py
│   ├── embedding_extractor.py
│   ├── evaluator.py
│   ├── explainability.py
│   ├── hybrid_classifier.py
│   ├── knn_classifier.py
│   ├── model_trainer.py
│   ├── summarizer.py
│   └── text_preprocessor.py
├── .gitignore
├── .python-version
├── main.py
├── Notebook_P2.ipynb
├── pyproject.toml
├── README.md
└── uv.lock
```

## Módulos implementados

| Módulo | Descripción |
| -|- |
| `DataLoader` | Descarga y unificación de los datos. |
| `TextPreprocessor` | Limpieza del texto y tokenización. |
| `EmbeddingExtractor` | Generación de embeddings densos usando el encoder de un Transformer. |
| `DenseRetriever` | Creación de un índice k-NN para búsquedas por similitud. |
| `KNNClassifier` | Clasificación mediante votación por mayoría. |
| `HybridClassifier` | Combinación de Transformer y k-NN (ajustable con el parámetro α). |
| `ModelTrainer` | Gestión del entrenamiento, registro de logs y checkpoints. |
| `ModelCompressor` | Compresión del modelo mediante *Knowledge Distillation*. |
| `SummarizationExplainer` | Generación de resúmenes explicativos utilizando T5 o BART. |
| `ExplainabilityModule` | Explicabilidad basada en razonamiento por casos y LLMs. |
| `Evaluator` | Cálculo de métricas, matrices de confusión y visualización de gráficos. |

