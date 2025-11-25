"""  
🎯 BASIC LEVEL CODE
Netflix Data Loader - Simple utility to load and validate Netflix viewing data

This module provides basic data loading functionality for Netflix datasets.
Ideal for beginners learning data processing in Python.

Author: Rishav Raj
Complexity: Basic
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import os


class NetflixDataLoader:
    """
    Basic data loader for Netflix datasets.
    
    Features:
    - Load CSV data
    - Basic validation
    - Handle missing values
    - Simple data inspection
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path (str): Path to the data directory
        """
        self.data_path = data_path
        self.data = None
        
    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filename (str): Name of the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        filepath = os.path.join(self.data_path, filename)
        
        try:
            self.data = pd.read_csv(filepath)
            print(f"✅ Successfully loaded {filename}")
            print(f"📊 Shape: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            print(f"❌ Error: File {filename} not found at {filepath}")
            return None
        except Exception as e:
            print(f"❌ Error loading file: {str(e)}")
            return None
    
    def get_basic_info(self) -> Dict:
        """
        Get basic information about the loaded dataset.
        
        Returns:
            dict: Dictionary containing basic dataset information
        """
        if self.data is None:
            print("⚠️  No data loaded. Please load data first.")
            return {}
        
        info = {
            'rows': len(self.data),
            'columns': len(self.data.columns),
            'column_names': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'dtypes': self.data.dtypes.to_dict()
        }
        
        return info
    
    def show_sample(self, n: int = 5) -> pd.DataFrame:
        """
        Display sample rows from the dataset.
        
        Args:
            n (int): Number of rows to display
            
        Returns:
            pd.DataFrame: Sample rows
        """
        if self.data is None:
            print("⚠️  No data loaded.")
            return None
        
        return self.data.head(n)
    
    def handle_missing_values(self, strategy: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            strategy (str): Strategy to handle missing values ('drop' or 'fill')
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        if self.data is None:
            print("⚠️  No data loaded.")
            return None
        
        if strategy == 'drop':
            cleaned_data = self.data.dropna()
            print(f"🧹 Dropped {len(self.data) - len(cleaned_data)} rows with missing values")
        elif strategy == 'fill':
            cleaned_data = self.data.fillna(0)
            print("🧹 Filled missing values with 0")
        else:
            print("❌ Invalid strategy. Use 'drop' or 'fill'")
            return self.data
        
        return cleaned_data
    
    def filter_by_column(self, column: str, value) -> pd.DataFrame:
        """
        Filter data by column value.
        
        Args:
            column (str): Column name to filter
            value: Value to filter by
            
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        if self.data is None:
            print("⚠️  No data loaded.")
            return None
        
        if column not in self.data.columns:
            print(f"❌ Column '{column}' not found in dataset")
            return None
        
        filtered_data = self.data[self.data[column] == value]
        print(f"🔍 Filtered {len(filtered_data)} rows where {column} = {value}")
        
        return filtered_data


def main():
    """
    Example usage of NetflixDataLoader.
    """
    # Initialize loader
    loader = NetflixDataLoader(data_path="./data")
    
    # Example: Load data
    print("\n" + "="*50)
    print("📥 LOADING NETFLIX DATA")
    print("="*50)
    
    # Get basic info
    print("\n" + "="*50)
    print("ℹ️  BASIC INFORMATION")
    print("="*50)
    info = loader.get_basic_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Show sample
    print("\n" + "="*50)
    print("👁️  SAMPLE DATA")
    print("="*50)
    print(loader.show_sample(3))


if __name__ == "__main__":
    main()
