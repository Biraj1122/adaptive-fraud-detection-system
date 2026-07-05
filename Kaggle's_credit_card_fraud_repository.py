import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =====================================================================
# 1. DATABASE INGESTION LAYER
# =====================================================================

class DatabaseLoader:
    """
    Handles the retrieval and preprocessing of the credit card fraud database
    directly from the host machine's storage disk.
    """
    def __init__(self, explicit_path=r"C:\Users\Asus Tuf\Desktop\3rd Sem AI project\creditcard.csv"):
        self.file_path = explicit_path

    def load_csv(self):
        """
        Ingests the CSV database file, cleans data gaps, 
        and filters down structural feature vectors to optimize memory.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"[CRITICAL ERROR] Database file not found at path: '{self.file_path}'. "
            )
            
        print(f" -> Accessing Storage Database: '{self.file_path}'...")
        
        # Reading first 50,000 rows to optimize execution speeds on standard laptop hardware
        df = pd.read_csv(self.file_path, nrows=50000)
        
        # Target column schema matching the file structure
        required_columns = ['Time', 'Amount', 'V1', 'V2', 'Class']
        
        # Data Preprocessing: Drop row entries with missing/NaN feature attributes
        df = df.dropna(subset=required_columns)
        
        print(f" -> Database Ingest Complete. Successfully loaded {len(df)} entries from disk.")
        return df, required_columns




