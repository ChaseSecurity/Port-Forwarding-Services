# Port Forwarding Services Are Forwarding Security Risks

## PFW Classifier

### Dependencies

The classifier is developed on CUDA 11. Higher CUDA versions may support it also.

You can use `pip install -r requirements.txt` to install some of the dependencies, and you need to install [Detectron2](https://github.com/facebookresearch/detectron2) (we used v0.6) separately.

### Model

You can get model from [our Hugging Face repo](https://huggingface.co/MirageTurtle/website-classifier/tree/main).

### Code Details

The code is designed as assembly line:
1. `data_summarizer.py`: summarize the data info (domain name, image path, extracting text) into a csv file.
2. `dataset.py`: build dataset pkl file from the previous csv file.
3. `predict.py`: predict the website category by loading dataset pkl into model.

### Usage

```bash
python3 data_summarizer.py --data_dir /path/to/provider/data/dir --output_csv /path/to/your/csv
python3 dataset.py --target_dir /path/to/pkl/target/dir --file /path/to/your/csv
python3 predict.py --model_path /path/to/model/dir --data_pkl /path/to/your/pkl --data_csv /path/to/your/csv --output_csv /path/to/result/csv
```

*An example* (you can get some test cases in seconds by following [collector usage instructions](../collector/README.md#Usage)):

```bash
python3 data_summarizer.py --data_dir /tmp/test/screenshots/20240101/ngrok --output_csv /tmp/test.csv
python3 dataset.py --target_dir /tmp --file /tmp/test.csv
python3 predict.py --data_pkl /tmp/pfs_model --data_pkl /tmp/0part.pkl --data_csv /tmp/test.csv --output_csv /tmp/prediction.csv
```

**Tips/Notes**:
1. The assembly line is designed for large-scale measurement, enabling us to process the dataset efficiently within the assembly line framework, which is more efficient.
2. You can power each component of the assembly line by a multiprocessing/parallelism manner, a strategy that we employ to improve the classifier.
