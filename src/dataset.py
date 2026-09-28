from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# TODO: add more features in the string (time signature, explicit)
# TODO: try different formats for the string
def format_row(row):
    """
    Format a row of the dataset.
    Input: row with Spotify data in Spotify's format
    Output: row with Spotify data in string format
    """
    formatted_row = f"""Song {row['track_id']} named {row['track_name']} by {row['artist_name']} from the album {row['album_name']}.
                    Belongs to the genre {row['genre']}. 
                    It has a popularity score of {row['popularity']} and a duration of {row['duration_ms']} milliseconds.
                    The song has the following audio features: danceability: {row['danceability']}, energy: {row['energy']}, key: {row['key']}, loudness: {row['loudness']}, mode: {row['mode']}, speechiness: {row['speechiness']}, acousticness: {row['acousticness']}, instrumentalness: {row['instrumentalness']}, liveness: {row['liveness']}, valence: {row['valence']}, tempo: {row['tempo']}.
                    """
    return row

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    #logger.info("Processing dataset...")
    #for i in tqdm(range(10), total=10):
    #    if i == 5:
    #        logger.info("Something happened for iteration 5.")

    logger.info(f"Reading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    logger.info("Processing rows with missing values...")
    # in the exploratory data analysis notebook, we found there is a row with 3 missing in key columns, so we will drop it
    output_df = df.dropna()

    # Format rows in the dataset
    logger.info("Formatting rows in the dataset...")
    output_df = output_df.apply(format_row, axis=1)

    logger.info(f"Saving processed dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
