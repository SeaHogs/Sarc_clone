from sarcasm.constants import DATA_DIR, CHECKPOINT_DIR
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    AlbertTokenizer,
    AlbertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AutoTokenizer,
    BartForSequenceClassification,
)
from torch.optim import AdamW, Optimizer
from transformers import get_scheduler
from typing import Optional
from tqdm.auto import tqdm
import torch
import evaluate as hf_evaluate
import abc
from datasets import Dataset
from torch.utils.data import DataLoader


class SarcasmClassifier(abc.ABC):
    MODEL_NAME: str
    PRETRAINED_MODEL: str

    device: torch.device
    tokenizer: BertTokenizer
    model: BertForSequenceClassification
    data_collator: DataCollatorWithPadding

    @abc.abstractmethod
    def load_tokenizer(self, model_name):
        pass

    @abc.abstractmethod
    def load_model(self, model_name):
        pass

    @abc.abstractmethod
    def get_embeddings(self, text):
        pass

    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        checkpoint_path = CHECKPOINT_DIR / self.MODEL_NAME
        if not checkpoint_path.exists():
            self.load_tokenizer(self.PRETRAINED_MODEL)
            self.load_model(self.PRETRAINED_MODEL)
        else:
            self.load_tokenizer(checkpoint_path)
            self.load_model(checkpoint_path)

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def train(
        self,
        dataset: Dataset,
        num_epochs: int = 3,
        optimizer: Optional[Optimizer] = None,
        batch_size: int = 8,
    ):
        if optimizer is None:
            optimizer = AdamW(self.model.parameters(), lr=1e-5)

        tokenized_dataset = self.tokenize(dataset)

        train_dataloader = DataLoader(
            tokenized_dataset,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=self.data_collator,
        )

        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        progress_bar = tqdm(range(num_training_steps))
        self.model.train()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

    def tokenize(self, dataset: Dataset):
        def tokenize_function(examples):
            return self.tokenizer(
                examples["headline"],
                truncation=True,
                padding="max_length",
                max_length=64,
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(["headline"])
        tokenized_dataset = tokenized_dataset.rename_column("is_sarcastic", "labels")

        tokenized_dataset.set_format("torch")
        return tokenized_dataset

    def predict(self, text):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits

    def evaluate(self, dataset: Dataset, metric: Optional[str] = None):
        if metric is None:
            metric = "f1"
        self.model.eval()

        task_evaluator = hf_evaluate.evaluator("text-classification")

        results = task_evaluator.compute(
            model_or_pipeline=self.model,
            data=dataset,
            metric=metric,
            label_mapping={"LABEL_0": 0.0, "LABEL_1": 1.0},
            strategy="bootstrap",
            tokenizer=self.tokenizer,
            input_column="headline",
            label_column="is_sarcastic",
            n_resamples=10,
            device=self.device,
        )

        # for batch in data_loader:
        #     batch = {k: v.to(self.device) for k, v in batch.items()}
        #     with torch.no_grad():
        #         outputs = self.model(**batch)

        #     logits = outputs.logits
        #     predictions = torch.argmax(logits, dim=-1)
        #     metric.add_batch(predictions=predictions, references=batch["labels"])
        return results

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = CHECKPOINT_DIR / self.MODEL_NAME
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)


class SarcasmClassifierBertFcnn(SarcasmClassifier):
    """
    A class for classifying sarcasm using a pre-trained BERT model.
    """

    MODEL_NAME = "bert-base-uncased-24-finepruned-sarcasm-detection"
    PRETRAINED_MODEL = "bert-base-uncased"

    def load_tokenizer(self, model_name):
        self.tokenizer = BertTokenizer.from_pretrained(
            model_name, cache_dir=DATA_DIR / model_name
        )

    def load_model(self, model_name):
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.model.to(self.device)

    def get_embeddings(self):
        return self.model.bert.embeddings


class SarcasmClassifierAlbertFcnn(SarcasmClassifier):
    """
    A class for classifying sarcasm using a pre-trained ALBERT model.
    """

    MODEL_NAME = "albert-base-uncased-24-finepruned-sarcasm-detection"
    PRETRAINED_MODEL = "albert/albert-base-v1"

    SPIECE_UNDERLINE = "▁"

    def load_tokenizer(self, model_name):
        self.tokenizer = AlbertTokenizer.from_pretrained(
            model_name, cache_dir=DATA_DIR / model_name
        )

    def load_model(self, model_name):
        self.model = AlbertForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.model.to(self.device)

    def get_embeddings(self):
        return self.model.albert.embeddings


class SarcasmClassifierRobertaFcnn(SarcasmClassifier):
    """
    A class for classifying sarcasm using a pre-trained RoBERTa model.
    """

    MODEL_NAME = "roberta-base-uncased-24-finepruned-sarcasm-detection"
    PRETRAINED_MODEL = "FacebookAI/roberta-base"

    def load_tokenizer(self, model_name):
        self.tokenizer = RobertaTokenizer.from_pretrained(
            model_name, cache_dir=DATA_DIR / model_name
        )

    def load_model(self, model_name):
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.model.to(self.device)

    def get_embeddings(self):
        return self.model.roberta.embeddings


class SarcasmClassifierBart(SarcasmClassifier):
    """
    A class for classifying sarcasm using a pre-trained BART model.
    """

    MODEL_NAME = "bart-large-sst2-24-finepruned-sarcasm-detection"
    PRETRAINED_MODEL = "valhalla/bart-large-sst2"

    def load_tokenizer(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=DATA_DIR / model_name
        )

    def load_model(self, model_name):
        self.model = BartForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.model.to(self.device)
        print(self.model.config.label2id)
