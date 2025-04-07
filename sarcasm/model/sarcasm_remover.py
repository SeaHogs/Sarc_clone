from sarcasm.constants import DATA_DIR, CHECKPOINT_DIR
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    AutoTokenizer,
    BartForConditionalGeneration,
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


class SarcasmRemover(abc.ABC):
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

        train_dataloader = self._get_data_loader(dataset, batch_size, shuffle=True)

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

    def predict(self, text):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def evaluate(
        self, dataset: Dataset, metric: Optional[hf_evaluate.EvaluationModule] = None
    ):
        if metric is None:
            metric = hf_evaluate.load("rouge")
        self.model.eval()

        # task_evaluator = hf_evaluate.evaluator("question-answering")

        # results = task_evaluator.compute(
        #     model_or_pipeline=self.model,
        #     data=dataset,
        #     metric=metric,
        #     strategy="bootstrap",
        #     tokenizer=self.tokenizer,
        #     question_column="sarcastic_headline",
        #     label_column="non_sarcastic_headline",
        #     n_resamples=10,
        #     device=self.device,
        # )

        test_dataloader = self._get_data_loader(dataset, batch_size=1, shuffle=False)

        for batch in test_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model.generate(**batch)

            predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            references = self.tokenizer.batch_decode(
                batch["labels"], skip_special_tokens=True
            )
            metric.add_batch(predictions=predictions, references=references)
        return metric.compute()

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = CHECKPOINT_DIR / self.MODEL_NAME
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

    def _get_data_loader(
        self,
        dataset: Dataset,
        batch_size: int = 8,
        shuffle: bool = True,
    ):
        def tokenize_function(examples):
            inputs = self.tokenizer(
                examples["sarcastic_headline"],
                truncation=True,
                padding="max_length",
                max_length=64,
            )
            targets = self.tokenizer(
                examples["non_sarcastic_headline"],
                truncation=True,
                padding="max_length",
                max_length=64,
            )
            inputs["labels"] = targets["input_ids"]
            return inputs

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(
            [
                "sarcastic_headline",
                "original_label",
                "transformed_label",
                "non_sarcastic_headline",
            ]
        )

        tokenized_dataset.set_format("torch")

        return DataLoader(
            tokenized_dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=self.data_collator,
        )


class SarcasmRemoverBart(SarcasmRemover):
    """
    A class for removing sarcasm using a pre-trained BART model.
    """

    MODEL_NAME = "bart-large-24-finetuned-sarcasm-remover"
    PRETRAINED_MODEL = "facebook/bart-large"

    def load_tokenizer(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=DATA_DIR / model_name
        )

    def load_model(self, model_name):
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
