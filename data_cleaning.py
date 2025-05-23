import pandas as pd
import numpy as np

def clean_data(df):
    """
    Funksjon for å rense data fra en DataFrame.
    Fjerner duplikate rader og håndterer NaN-verdier.
    
    Args:
        df (pd.DataFrame): Input DataFrame som skal renses
        
    Returns:
        pd.DataFrame: Renset DataFrame
    """
    # Lag en kopi av DataFrame for å unngå å modifisere original
    df_cleaned = df.copy()
    
    # Fjern duplikate rader
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Håndter NaN-verdier
    # For numeriske kolonner: erstatt med median
    numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    
    # For kategoriske kolonner: erstatt med modus (mest vanlige verdien)
    categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
    
    return df_cleaned 