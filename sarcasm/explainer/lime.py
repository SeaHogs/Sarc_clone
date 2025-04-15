from lime.lime_text import LimeTextExplainer
from lime.explanation import Explanation
import torch.nn as nn

from sarcasm.model.sarcasm_classifier import SarcasmClassifier
from dataclasses import dataclass


@dataclass
class LimeExplanation:
    explanation: Explanation
    prediction: float


def predictor(classifier: SarcasmClassifier):
    def predict(text):
        predictions = classifier.predict(text)
        probs = nn.functional.softmax(predictions, dim=1)
        return probs.detach().cpu().numpy()

    return predict


def explain_with_lime(
    classifier: SarcasmClassifier, data, num_features: int = 6, random_state: int = 24
):
    class_names = ["non-sarcastic", "sarcastic"]
    explainer = LimeTextExplainer(class_names=class_names, random_state=random_state)
    prediction_fn = predictor(classifier)
    explanation = explainer.explain_instance(
        data["headline"],
        prediction_fn,
        num_features=num_features,
        num_samples=100,
    )
    return LimeExplanation(
        explanation=explanation,
        prediction=prediction_fn(data["headline"]),
    )
