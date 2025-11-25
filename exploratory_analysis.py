"""  
💡 MEDIUM LEVEL CODE
Netflix Exploratory Data Analysis - Advanced EDA with statistical analysis and visualizations

This module provides intermediate-level data analysis functionality including:
- Statistical analysis and hypothesis testing
- Advanced visualizations
- Trend analysis
- Data quality assessment

Author: Rishav Raj
Complexity: Medium
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class NetflixExploratoryAnalysis:
    """
    Advanced Exploratory Data Analysis for Netflix datasets.
    
    Features:
    - Statistical summaries
    - Distribution analysis
    - Correlation analysis
    - Trend detection
    - Outlier identification
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize EDA class with data.
        
        Args:
            data (pd.DataFrame): Netflix viewing data
        """
        self.data = data.copy()
        self.numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
    def generate_statistical_summary(self) -> pd.DataFrame:
        """
        Generate comprehensive statistical summary.
        
        Returns:
            pd.DataFrame: Statistical summary table
        """
        print("📊 Generating Statistical Summary...\n")
        
        summary = self.data[self.numerical_cols].describe()
        
        # Add additional statistics
        summary.loc['variance'] = self.data[self.numerical_cols].var()
        summary.loc['skewness'] = self.data[self.numerical_cols].skew()
        summary.loc['kurtosis'] = self.data[self.numerical_cols].kurtosis()
        
        return summary.round(2)
    
    def analyze_distributions(self, save_plots: bool = False) -> None:
        """
        Analyze and visualize distributions of numerical features.
        
        Args:
            save_plots (bool): Whether to save plots to disk
        """
        print("📊 Analyzing Distributions...\n")
        
        n_cols = len(self.numerical_cols)
        fig, axes = plt.subplots(n_cols, 2, figsize=(15, 5*n_cols))
        
        if n_cols == 1:
            axes = axes.reshape(1, -1)
        
        for idx, col in enumerate(self.numerical_cols):
            # Histogram
            axes[idx, 0].hist(self.data[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
            axes[idx, 0].set_title(f'Distribution of {col}')
            axes[idx, 0].set_xlabel(col)
            axes[idx, 0].set_ylabel('Frequency')
            
            # Box plot
            axes[idx, 1].boxplot(self.data[col].dropna())
            axes[idx, 1].set_title(f'Box Plot of {col}')
            axes[idx, 1].set_ylabel(col)
            
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('distribution_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ Plots saved to 'distribution_analysis.png'")
        
        plt.show()
    
    def correlation_analysis(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Perform correlation analysis on numerical features.
        
        Args:
            method (str): Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        print(f"🔍 Performing {method.capitalize()} Correlation Analysis...\n")
        
        corr_matrix = self.data[self.numerical_cols].corr(method=method)
        
        # Visualize correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title(f'{method.capitalize()} Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        return corr_matrix
    
    def detect_outliers(self, column: str, method: str = 'iqr') -> Tuple[List, Dict]:
        """
        Detect outliers using IQR or Z-score method.
        
        Args:
            column (str): Column name to analyze
            method (str): Detection method ('iqr' or 'zscore')
            
        Returns:
            tuple: (outlier_indices, statistics_dict)
        """
        if column not in self.numerical_cols:
            print(f"❌ Column '{column}' is not numerical")
            return [], {}
        
        data = self.data[column].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            stats_dict = {
                'method': 'IQR',
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'n_outliers': len(outliers),
                'outlier_percentage': (len(outliers) / len(data)) * 100
            }
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outliers = data[z_scores > 3]
            
            stats_dict = {
                'method': 'Z-score',
                'threshold': 3,
                'n_outliers': len(outliers),
                'outlier_percentage': (len(outliers) / len(data)) * 100
            }
        
        print(f"⚠️  Outliers detected in '{column}':")
        print(f"   Method: {stats_dict['method']}")
        print(f"   Count: {stats_dict['n_outliers']}")
        print(f"   Percentage: {stats_dict['outlier_percentage']:.2f}%\n")
        
        return outliers.index.tolist(), stats_dict
    
    def categorical_analysis(self) -> Dict:
        """
        Analyze categorical variables.
        
        Returns:
            dict: Analysis results for each categorical column
        """
        print("📊 Analyzing Categorical Variables...\n")
        
        results = {}
        
        for col in self.categorical_cols:
            value_counts = self.data[col].value_counts()
            
            results[col] = {
                'unique_values': self.data[col].nunique(),
                'most_common': value_counts.index[0],
                'most_common_count': value_counts.iloc[0],
                'distribution': value_counts.to_dict()
            }
            
            print(f"Column: {col}")
            print(f"  Unique values: {results[col]['unique_values']}")
            print(f"  Most common: {results[col]['most_common']} ({results[col]['most_common_count']} occurrences)\n")
        
        return results
    
    def time_series_analysis(self, date_column: str, value_column: str) -> pd.DataFrame:
        """
        Perform time series analysis.
        
        Args:
            date_column (str): Date column name
            value_column (str): Value column for trend analysis
            
        Returns:
            pd.DataFrame: Aggregated time series data
        """
        print(f"📈 Performing Time Series Analysis...\n")
        
        # Convert to datetime if needed
        self.data[date_column] = pd.to_datetime(self.data[date_column])
        
        # Group by date and aggregate
        ts_data = self.data.groupby(self.data[date_column].dt.date)[value_column].agg(['mean', 'sum', 'count'])
        
        # Plot trend
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        axes[0].plot(ts_data.index, ts_data['mean'])
        axes[0].set_title(f'Average {value_column} Over Time')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel(f'Mean {value_column}')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(ts_data.index, ts_data['sum'])
        axes[1].set_title(f'Total {value_column} Over Time')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel(f'Sum {value_column}')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(ts_data.index, ts_data['count'])
        axes[2].set_title('Activity Count Over Time')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Count')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return ts_data


def main():
    """
    Example usage of Netflix Exploratory Analysis.
    """
    # Generate sample data for demonstration
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'watch_duration': np.random.exponential(30, 1000),
        'user_rating': np.random.normal(4.0, 0.8, 1000),
        'content_genre': np.random.choice(['Drama', 'Comedy', 'Action', 'Documentary'], 1000),
        'device_type': np.random.choice(['TV', 'Mobile', 'Desktop'], 1000)
    })
    
    # Initialize analyzer
    analyzer = NetflixExploratoryAnalysis(sample_data)
    
    # Generate statistical summary
    print("\n" + "="*50)
    print("📊 STATISTICAL SUMMARY")
    print("="*50)
    print(analyzer.generate_statistical_summary())
    
    # Correlation analysis
    print("\n" + "="*50)
    print("🔍 CORRELATION ANALYSIS")
    print("="*50)
    corr_matrix = analyzer.correlation_analysis()
    
    # Outlier detection
    print("\n" + "="*50)
    print("⚠️  OUTLIER DETECTION")
    print("="*50)
    analyzer.detect_outliers('watch_duration', method='iqr')
    
    # Categorical analysis
    print("\n" + "="*50)
    print("📊 CATEGORICAL ANALYSIS")
    print("="*50)
    results = analyzer.categorical_analysis()


if __name__ == "__main__":
    main()
