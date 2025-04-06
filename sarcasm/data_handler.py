from transformers import PreTrainedTokenizer
from datasets import Dataset

from sarcasm.constants import DATA_DIR

class SarcasmDataHandler:
    DATASET_URL = "rmisra/news-headlines-dataset-for-sarcasm-detection"
    DATA_FILE = "Sarcasm_Headlines_Dataset_v2.json"

    dataset: Dataset
    tokenizer: PreTrainedTokenizer

    def __init__(self, test_size=0.2):
        # Convert to Hugging Face dataset format
        df = self.load_data()
        self.dataset = Dataset.from_pandas(df)
        self.dataset = self.dataset.remove_columns(["article_link"])
        self.dataset = self.dataset.train_test_split(test_size=test_size, seed=24)

    def download_dataset(self):
        if not DATA_DIR.exists():
            DATA_DIR.mkdir()

        dataset_path = DATA_DIR / self.DATASET_URL
        if not dataset_path.exists():
            import kagglehub
            import shutil
            
            cache_path = kagglehub.dataset_download(self.DATASET_URL)
            shutil.move(cache_path, dataset_path)

        return dataset_path

    def load_data(self):
        import pandas as pd
        dataset_path = self.download_dataset()
        return pd.read_json(dataset_path / self.DATA_FILE, lines=True)
