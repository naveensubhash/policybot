import pandas as pd


def load_policies(filepath: str) -> pd.DataFrame:
    """Load policy text data"""
    df = pd.read_csv(filepath)
    return df