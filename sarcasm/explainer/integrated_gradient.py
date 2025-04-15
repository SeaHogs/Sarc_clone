import torch
from sarcasm.model.sarcasm_classifier import SarcasmClassifier
from captum.attr import visualization as viz
from captum.attr import LayerIntegratedGradients
import torch.nn as nn


class IGExplainer:
    def __init__(self, classifier: SarcasmClassifier):
        self.classifier = classifier
        self.classifier.model.eval()

        self.model_input = self.classifier.get_embeddings()

        def model_output(*inputs):
            logits = self.classifier.model(*inputs).logits
            return logits[0]

        self.model_output = model_output

        self.lig = LayerIntegratedGradients(self.model_output, self.model_input)

    def construct_input_and_baseline(self, text):
        baseline_token_id = self.classifier.tokenizer.pad_token_id
        sep_token_id = self.classifier.tokenizer.sep_token_id
        cls_token_id = self.classifier.tokenizer.cls_token_id

        text_ids_no_special_tokens = self.classifier.tokenizer(
            text, truncation=True, max_length=64, add_special_tokens=False
        )

        paddings = [baseline_token_id] * (
            64 - len(text_ids_no_special_tokens["input_ids"]) - 2
        )
        token_list = self.classifier.tokenizer.convert_ids_to_tokens(
            text_ids_no_special_tokens["input_ids"]
        )
        baseline_input_ids = (
            [cls_token_id]
            + [baseline_token_id] * len(text_ids_no_special_tokens["input_ids"])
            + [sep_token_id]
            + paddings
        )

        text_ids = self.classifier.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=64,
            add_special_tokens=True,
        )
        input_ids = text_ids["input_ids"]
        other_args = tuple(
            torch.tensor([text_ids[k]], device=self.classifier.device)
            if k in text_ids
            else None
            for k in ["attention_mask", "token_type_ids"]
        )

        return (
            torch.tensor([input_ids], device=self.classifier.device),
            torch.tensor([baseline_input_ids], device=self.classifier.device),
            token_list,
            other_args,
        )

    def explain(self, text):
        input_ids, baseline_input_ids, token_list, other_args = (
            self.construct_input_and_baseline(text)
        )

        attributions, delta = self.lig.attribute(
            inputs=input_ids,
            baselines=baseline_input_ids,
            additional_forward_args=other_args,
            return_convergence_delta=True,
            internal_batch_size=1,
        )

        def summarize_attributions(attributions):
            attributions = attributions.sum(dim=-1).squeeze(0)
            attributions = attributions / torch.norm(attributions)

            return attributions

        attributions_sum = summarize_attributions(attributions)[1 : 1 + len(token_list)]

        prediction = self.model_output(input_ids, *other_args).detach().cpu()
        predcition_probs = nn.functional.softmax(prediction, dim=-1)

        return viz.VisualizationDataRecord(
            word_attributions=attributions_sum,
            pred_prob=torch.max(predcition_probs),
            pred_class=torch.argmax(predcition_probs).numpy(),
            true_class=1,
            attr_class=text,
            attr_score=attributions_sum.sum(),
            raw_input_ids=token_list,
            convergence_score=delta,
        )

    def visualize(self, text):
        score_vis = self.explain(text)
        return viz.visualize_text([score_vis])
