import os
import pandas as pd
import logging
import json
import base64
from bs4 import BeautifulSoup
import argparse

# Init logging configuration
logger = logging.getLogger()
fhandler = logging.FileHandler(filename='/tmp/test.log', mode='a')
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.INFO)
logging.info(f'Logger inited.')

# Define the function to extract useful text from har file
def extract_text_from_har(har_path: str) -> str:
    """
    Extract useful text from a har file
    :param har_path: str, the path of the har file
    :return: str, the extracted text
    """
    try:
        with open(har_path, "r") as f:
            har = json.loads(f.read())
        for entry in har["log"]["entries"]:
            content = entry["response"]["content"]
            if content.get("mimeType", "NoMIMEType")[:9] == "text/html" and "text" in content.keys():
                html_doc = base64.b64decode(content["text"])
                soup = BeautifulSoup(html_doc, "html.parser")
                return " ".join([str(s) for s in soup.strippped_strings])
        return ""
    except Exception as e:
        logging.warning(f"Failed to extract text from {har_path}: {e}")
        return ""

# Define the function to summarize the data into a csv file
def summarize_data(data_dir: str, output_csv: str) -> None:
    """
    Summarize the data in the data directory and save the summary to a csv file
    :param data_dir: str, the directory of the data, generally, the directory contains multiple snapshots
    :param output_csv: str, the output csv file path
        the csv file contains the columns "domain", "image" and "text"
    """
    # Get the list of snapshots
    # A snapshot is a directory containing a screenshot (.png) and a network track file (.har)
    snapshots = os.listdir(data_dir)
    snapshots = [snapshot for snapshot in snapshots if os.path.isdir(os.path.join(data_dir, snapshot))]

    # Initialize the summary dataframe
    summary_df = pd.DataFrame(columns=["domain", "image", "text"])

    # Iterate through the snapshots
    for snapshot in snapshots:
        snapshot_dir = os.path.join(data_dir, snapshot)
        screenshot_path = os.path.join(snapshot_dir, "page_screenshot.png")
        har_path = os.path.join(snapshot_dir, "test.har")
        text = extract_text_from_har(har_path)
        summary_df = pd.concat([
            summary_df,
            pd.DataFrame({
                "domain": [snapshot],
                "image": [screenshot_path],
                "text": [text]
            })
        ])

    # Save the summary dataframe to a csv file
    summary_df.to_csv(output_csv, index=False)
    logging.info(f"Data summarized and saved to {output_csv}.")


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    # data_dir containing the snapshots
    # e.g., data_dir = /tmp/pfs/screenshots/20240101/ngrok/
    parser.add_argument("--data_dir", type=str, help="The directory of the data")
    parser.add_argument("--output_csv", type=str, help="The output csv file path")
    args = parser.parse_args()

    # Summarize the data
    summarize_data(args.data_dir, args.output_csv)

