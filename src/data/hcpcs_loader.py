import pandas as pd


def load_hcpcs(filepath: str) -> pd.DataFrame:
    """Load HCPCS code reference data"""
    df = pd.read_csv(filepath)
    df["code"] = df["code"].astype(str)
    return df