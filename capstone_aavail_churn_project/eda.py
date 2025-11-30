from .data_ingestion import load_data, basic_clean
from .logging_utils import logger

def main():
    df = basic_clean(load_data())
    logger.info(f"Shape: {df.shape}")
    logger.info("Null counts:\n" + str(df.isna().sum()))
    logger.info("Churn rate by country:\n" + str(df.groupby("country")["churn"].mean()))

if __name__ == "__main__":
    main()
