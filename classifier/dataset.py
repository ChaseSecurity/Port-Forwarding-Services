import argparse
import multiprocessing
import os
import logging
import pandas as pd
import traceback

from transformers import LayoutXLMProcessor, LayoutLMv2ForSequenceClassification, set_seed, TrainingArguments, Trainer
from PIL import Image
from PIL import ImageFile
from copy import deepcopy
import torch
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import EarlyStoppingCallback
import pickle
import shutil
import torch.nn.functional as F


os.environ["WANDB_DISABLED"] = "true"
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Init logging configuration
logger = logging.getLogger()
# fhandler = logging.FileHandler(filename='/tmp/test.log', mode='a')
# formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
# fhandler.setFormatter(formatter)
# logger.addHandler(fhandler)
logger.setLevel(logging.INFO)
logging.info(f'Logger inited.')

parser = argparse.ArgumentParser()
parser.add_argument("--screenshots_dir", type=str, help="the dir of the screenshots", default="/")
parser.add_argument("--target_dir", type=str, help="the dir of the target pkl files", default="/path/to/your/target")
parser.add_argument("--start_idx", type=int, default=0)
parser.add_argument("--end_idx", type=int, default=800, help="No the end one!")
parser.add_argument("--process_num", type=int, default=4)
parser.add_argument("--file", type=str)
args = parser.parse_args()

# Define Dataset format
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, processor, images_path, le, is_pred=False):
        """
        :param df: DataFrame, with columns "image" (relative path) and "text", and "label" if training
        :param processor: LayoutXLMProcessor
        :param images_path: str, the directory of the images, the image path in df is relative to this path
        :param le: LabelEncoder
        :param is_pred: bool, if True, the dataset is for prediction
        """
        super(CustomDataset).__init__()
        images = [Image.open(os.path.join(images_path, item["image"])).convert("RGB") for idx, item in df.iterrows()]
        texts = [str(item["text"]) for idx, item in df.iterrows()]
        # The processor will do OCR on the images
        encodings = processor(images=images, text=texts, return_tensors="pt", truncation=True, padding="max_length", max_length=512, return_attention_mask=True)
        self.encodings = encodings
        self.is_pred = is_pred
        if not is_pred:
            self.labels = torch.tensor(le.transform(df.loc[:, "label"].to_list()))
        self.length = df.shape[0]

    def __getitem__(self, idx):
        """
        Get the item by index
        :param idx: int or slice or list or np.ndarray
        """
        if isinstance(idx, int):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            if not self.is_pred:
                item["labels"] = self.labels[idx]
            return item
        elif isinstance(idx, slice):
            items = []
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else self.length
            step = idx.step if idx.step is not None else 1
            for i in range(start, stop, step):
                item = {key: torch.tensor(val[i]) for key, val in self.encodings.items()}
                if not self.is_pred:
                    item["labels"] = self.labels[i]
                items.append(item)
            return items
        elif isinstance(idx, np.ndarray) or isinstance(idx, list):
            items = []
            for i in idx:
                item = {key: torch.tensor(val[i]) for key, val in self.encodings.items()}
                if not self.is_pred:
                    item["labels"] = self.labels[i]
                items.append(item)
            return items
        else:
            print(f"Please check index type! Now index type: {type(idx)}")

    def __len__(self):
        return self.length
logging.info("CustomDataset defined.")

# Define the processor
processor = LayoutXLMProcessor.from_pretrained("microsoft/layoutxlm-base")
processor.feature_extractor.tesseract_config = "-l eng+chi_sim --psm 11"
processor.feature_extractor.apply_ocr = True
processor.feature_extractor.size = (224, 126)
logging.info("Processor defined.")

# Define the label list
label_list = [
    'Code repository',
    'Data store',
    'Error Page',
    'Industrial Controller',
    'NAS/Server Interface',
    'Network',
    'OA',
    'Others',
    'Remote desktop',
    'Server Default Page',
    'Smart Home Controller',
    'White Page'
]
logging.info(f'label_list defined with length {len(label_list)}')

# Define the LabelEncoder
le = LabelEncoder()
le.fit(label_list)
logging.info("le(LabelEncoder) defined.")

# Define the function to init the dataset
def init_dataset(file_path, target_dir, images_path, idx, lock):
    """
    Init the dataset and write it to pkl file
    :param file_path: str, the path of the csv file
    :param target_dir: str, the directory to save the pkl file
    :param images_path: str, the directory of the images
    :param idx: int, the index/id of the part, used to name the pkl file
    :param lock: multiprocessing.Manager().Lock(), the lock to ensure the writing process is safe
    """
    df = pd.read_csv(file_path, lineterminator="\n")
    df.fillna("", inplace=True)
    if os.path.exists(os.path.join(target_dir, f'{idx}part.pkl')):
        logging.info(f'{idx}part.pkl already exists.')
    else:
        try:
            dataset = CustomDataset(df, processor, images_path, le, is_pred=True)
        except Exception as e:
            logging.warning(f'{e}(type {type(e)}) occurs when initing part {idx} from {file_path}')
        try:
            with lock:
                logging.info(f'Start writing dataset {idx}part to pkl file.')
                with open(os.path.join(target_dir, f'{idx}part.pkl'), "wb") as f:
                    f.write(pickle.dumps(dataset))
        except Exception as e:
            logging.warning(f'{e}(type {type(e)}) occurs when trying to write part {idx} dataset to file.')
    logging.info(f'Process {str(os.getpid())}(part {idx}) finished.')
    return

def task(file_path, target_dir, idx, images_path, lock):
    """
    The task function for multiprocessing
    A wrapper for init_dataset()
    params are the same as init_dataset()
    """
    logging.info(f'Start process {str(os.getpid())} for part {idx}(file: {file_path})')
    try:
        init_dataset(file_path=file_path, target_dir=target_dir, images_path=images_path, idx=idx, lock=lock)
    except Exception as e:
        logging.warning(f'{e}(type {type(e)}) occurs when trying to call init_dataset() for part {idx}')
        logging.warning(f'traceback: {traceback.format_exc()}')
    return

logging.info("Function init_dataset() and task() defined.")


if __name__ == "__main__":
    # Init the lock and the multiprocessing pool
    lock = multiprocessing.Manager().Lock()
    logging.info(f'Num of processes: {args.process_num}')
    pool = multiprocessing.Pool(processes=args.process_num)

    # Get the target files
    if args.file:
        files = [args.file]
    else:
        # For large scale measuremnt, you need to summarize the data by your self
        files = [os.path.join(args.screenshots_dir, f'{i}part', "data.csv") for i in range(args.start_idx, args.end_idx)]
    logging.info(f'Target files: {files}')
    logging.info("Start.")
    for file, idx in zip(files, range(args.start_idx, args.end_idx)):
        pool.apply_async(
            task,
            kwds={
                "file_path": file,
                "target_dir": args.target_dir,
                "idx": idx,
                "images_path": os.path.split(file)[0],
                "lock": lock,
            }
        )
    pool.close()
    pool.join()
    logging.info("Finish.")
