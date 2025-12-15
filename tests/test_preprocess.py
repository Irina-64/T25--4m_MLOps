import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocess import clean_data

def test_clean_data_fills_nan():
    df = pd.DataFrame({
        'Time_broken_spent_Alone': [4.0, np.nan, 9.0],
        'Stage_fear': ['No', 'Yes', 'No'],
        'Personality': ['Extrovert', 'Introvert', 'Extrovert']
    })
    
    result = clean_data(df)
    assert result['Time_broken_spent_Alone'].isna().sum() == 0
    assert set(result['Stage_fear'].unique()).issubset({0, 1})
    assert 'Personality_encoded' in result.columns
    assert set(result['Personality_encoded'].unique()).issubset({0, 1})

def test_clean_data_preserves_rows():
    df = pd.DataFrame({
        'Time_broken_spent_Alone': [4.0, 9.0],
        'Stage_fear': ['No', 'Yes'],
        'Personality': ['Extrovert', 'Introvert']
    })
    
    result = clean_data(df)
    assert len(result) == 2