import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from writing_assistance_core import FormalityDetector

tokenizer = AutoTokenizer.from_pretrained('s-nlp/deberta-large-formality-ranker')
model = AutoModelForSequenceClassification.from_pretrained('s-nlp/deberta-large-formality-ranker')


class Deberta(FormalityDetector):
    """Deberta-large model formality detector"""

    model = model
    tokenizer = tokenizer

    @staticmethod
    def _predict_from_list(list_of_texts: list[str]) -> list[dict[str, float]]:
        inputs = Deberta.tokenizer(
            list_of_texts,
            add_special_tokens=True,
            return_token_type_ids=True,
            padding=True,
            return_tensors='pt',
        )
        with torch.no_grad():
            outputs = Deberta.model(**inputs)

        scores = []
        for text_scores in outputs.logits.softmax(dim=1):
            score = {'formal': text_scores[0].item(), 'informal': text_scores[1].item()}
            scores.append(score)

        return scores

    @staticmethod
    def _predict_from_str(text: str) -> list[dict[str, float]]:
        inputs = Deberta.tokenizer(
            text,
            add_special_tokens=True,
            return_token_type_ids=True,
            padding=True,
            return_tensors='pt',
        )

        with torch.no_grad():
            outputs = Deberta.model(**inputs)

        scores = []
        for text_scores in outputs.logits.softmax(dim=1):
            score = {'formal': text_scores[0].item(), 'informal': text_scores[1].item()}
            scores.append(score)

        return scores
