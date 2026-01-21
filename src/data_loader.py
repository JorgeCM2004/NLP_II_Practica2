
import os
import glob
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import kagglehub

MAX_ROWS = 200000
class DataLoader:

    def __init__(self ,text_column: str = "text", label_column: str = "label", random_state: int = 42) -> None:
        self.text_column = text_column
        self.label_column = label_column
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self._data: Optional[pd.DataFrame] = None


    def load_raw_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        
        if file_path is None:
            file_path = self.find_excel_file()
        
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, engine='openpyxl')

        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path, engine='xlrd')
        
        self._validate_columns(df)
        
        df = self._clean_data(df)
        
        self._data = df
        return df
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        required_cols = [self.text_column, self.label_column]
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            available = ", ".join(df.columns.tolist())
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available columns: {available}"
            )
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.dropna(subset=[self.text_column, self.label_column])
        
        df[self.text_column] = df[self.text_column].astype(str).str.strip()
        
        df = df[df[self.text_column].str.len() > 0]
        
        df = df.reset_index(drop=True)
        return df
    
    def load_data(self, file_path: Optional[str] = None, val_size: float = 0.1, test_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self._data is None:
            self.load_raw_data(file_path)
        
        df = self._data.copy()
        
        df['label_encoded'] = self.label_encoder.fit_transform(
            df[self.label_column]
        )
        
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df['label_encoded'],
            random_state=self.random_state
        )
        
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            stratify=train_val_df['label_encoded'],
            random_state=self.random_state
        )
        
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        return train_df, val_df, test_df
    
    def get_label_encoder(self) -> LabelEncoder:
        return self.label_encoder
    
    def get_label_names(self) -> List[str]:
        return list(self.label_encoder.classes_)
    
    def get_num_labels(self) -> int:
        return len(self.label_encoder.classes_)
    
    def get_data_stats(self) -> Dict[str, Any]:
        if self._data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        df = self._data
        stats = {
            "total_samples": len(df),
            "num_classes": df[self.label_column].nunique(),
            "class_distribution": df[self.label_column].value_counts().to_dict(),
            "avg_text_length": df[self.text_column].str.len().mean(),
            "min_text_length": df[self.text_column].str.len().min(),
            "max_text_length": df[self.text_column].str.len().max(),
        }
        return stats


    def download_data(self, save: bool = True) -> pd.DataFrame:
        datasets_info = [
            ('kritanjalijain/amazon-reviews', 'Amazon Reviews'),
            ('d4rklucif3r/restaurant-reviews', 'Restaurant Reviews'),
            ('lakshmi25npathi/imdb-dataset-of-50k-movie-reviews', 'IMDB Movie Reviews')
        ]

        all_data = []
        
        for kaggle_path, source_name in datasets_info:
            try:
                path_folder = kagglehub.dataset_download(kaggle_path)

                files = [f for f in os.listdir(path_folder) if f.endswith('.csv') or f.endswith('.tsv') or f.endswith('.txt')]
                if not files:
                    print(f" No se encontró archivo CSV/TSV en {source_name}, saltando...")
                    continue
                        
                file_path = os.path.join(path_folder, files[0])

                data = self._arminizate_data(file_path, source_name)
                all_data.append(data)

            except Exception as e:
                print(f"Error al descargar {source_name}: {str(e)}")

        final_df = pd.concat(all_data, ignore_index=True)
        
        if save:
            final_df.to_csv('data/data.csv', index=False)
        
        self._data = final_df
        return final_df


    def _arminizate_data(self, file_path: str, source_name: str) -> pd.DataFrame:
        if source_name == 'Amazon Reviews':
                df = pd.read_csv(file_path, header=None, names=['label', 'title', 'text'])
                df['text'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
                df['label'] = df['label'].apply(lambda x: 0 if x == 1 else 1)

        elif source_name == 'Restaurant Reviews':
            try:
                df = pd.read_csv(file_path, sep='\t')
            except:
                df = pd.read_csv(file_path)
            df = df.rename(columns={'Review': 'text', 'Liked': 'label'})

        elif source_name == 'IMDB Movie Reviews':
            df = pd.read_csv(file_path)
            df = df.rename(columns={'review': 'text', 'sentiment': 'label'})
            
            sentiment_map = {'positive': 1, 'negative': 0}
            df['label'] = df['label'].map(sentiment_map)

        if df is not None:
            df = df[['text', 'label']]
            df['source'] = source_name
            df['label'] = df['label'].astype(int)
            df['text'] = df['text'].astype(str)

            if len(df) > MAX_ROWS:
                df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
            return df
        
        return None



        
            

            
