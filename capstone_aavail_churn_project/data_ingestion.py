import pandas as pd
from .config import config
from .logging_utils import logger

def load_data(path: str | None = None) -> pd.DataFrame:
    path = path or config.data_path
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Running basic data cleaning")
    df = df.dropna(subset=["churn"])
    num_cols = df.select_dtypes(include="number").columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df

if __name__ == "__main__":
    df = basic_clean(load_data())
    print(df.head())
