from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd

from stock_prediction.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "TSLA_preprocessed_dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "TSLA_preprocessed_dataset_with_features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    df = pd.read_csv(input_path)

    # Generate output features
    df["Float_Price"] = df["Close"].shift(-1)
    df["Binary_Price"] = (df["Float_Price"] > df["Close"]).astype(int)

    df["Factor_Price"] = df["Float_Price"] / df["Close"]
    df["Delta_Price"] = df["Float_Price"] - df["Close"]

    df.to_csv(output_path, index=False)

    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
