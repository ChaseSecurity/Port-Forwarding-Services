import argparse
import os
import logging
import time
import traceback

from transformers import LayoutXLMProcessor, LayoutLMv2ForSequenceClassification, set_seed, TrainingArguments, Trainer
from PIL import Image
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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import random

os.environ["WANDB_DISABLED"] = "true"
manual_random_seed = 102 # just for program to run, do anything for prediction

logger = logging.getLogger()
# fhandler = logging.FileHandler(filename='/tmp/test.log', mode='a')
# formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
# fhandler.setFormatter(formatter)
# logger.addHandler(fhandler)
logger.setLevel(logging.INFO)
logging.info(f'Logger inited.')

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="/path/to/model/dir", help="model path")
parser.add_argument("--data_pkl", type=str, default="/path/to/data", help="data(in pkl format)")
parser.add_argument("--data_csv", type=str, default="/path/to/data.csv", help="data(in csv format) for domain info")
parser.add_argument("--output_csv", type=str, default="/path/to/output.csv", help="output csv file")
args = parser.parse_args()

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, processor, images_path, le, is_pred=False):
        """
        :param df: DataFrame, with columns "image" and "text", and "label" if training
        """
        super(CustomDataset).__init__()
        images = [Image.open(os.path.join(images_path, item["image"])).convert("RGB") for idx, item in df.iterrows()]
        texts = [str(item["text"]) for idx, item in df.iterrows()]
        encodings = processor(images=images, text=texts, return_tensors="pt", truncation=True, padding="max_length", max_length=512, return_attention_mask=True)
        self.encodings = encodings
        self.is_pred = is_pred
        if not is_pred:
            self.labels = torch.tensor(le.transform(df.loc[:, "label"].to_list()))
        self.length = df.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, np.int64) or isinstance(idx, int):
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

"""
processor = LayoutXLMProcessor.from_pretrained("microsoft/layoutxlm-base")
processor.feature_extractor.tesseract_config = "-l eng+chi_sim --psm 11"
processor.feature_extractor.apply_ocr = True
processor.feature_extractor.size = (224, 126)
logging.info("Processor defined.")
"""

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
label_list.sort()
logging.info(f'label_list defined with length {len(label_list)}')

le = LabelEncoder()
le.fit(label_list)
logging.info("le(LabelEncoder) defined.")

model = LayoutLMv2ForSequenceClassification.from_pretrained(
    args.model_path,
    num_labels=len(label_list)
)
logging.info("model defined.")

# do anything for prediction
training_args = TrainingArguments(
    learning_rate=5e-5,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,
    seed=manual_random_seed,

    # output_dir="/path/to/your/output",
    output_dir="./output",
    logging_steps=100,
    # logging_dir="/path/to/your/logs",
    logging_dir="./logs",
)
logging.info("training_args defined.")

def compute_metrics(pred_labels):    
    pred, labels = pred_labels
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average="weighted")
    precision = precision_score(y_true=labels, y_pred=pred, average="weighted")
    f1 = f1_score(y_true=labels, y_pred=pred, average="weighted")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
logging.info("compute_metrics defined.")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=None,
    compute_metrics=compute_metrics,
)
logging.info("trainer defined.")


if __name__ == "__main__":
    """
    Just a simple example to show how to predict from dataset(pkl) with trained model.
    You can get the pkl dataset by file `dataset.py`.
    """
    # load dataset
    with open(args.data_pkl, 'rb') as f:
        dataset = pickle.load(f)
        logging.info(f'dataset loaded with length {len(dataset)}')
        
    # predict
    res = trainer.predict(test_dataset=dataset)
    logging.info(f'prediction finished with length {len(res.predictions)}')
    # save result
    probabilities = F.softmax(torch.from_numpy(res.predictions), dim=1)
    predication = le.inverse_transform(np.argmax(res.predictions, axis=1))
    df = pd.read_csv(args.data_csv)
    df["label"] = predication
    df.to_csv(args.output_csv, index=False)
    logging.info(f'prediction saved to {args.output_csv}')

