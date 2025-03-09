import torch
from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification

from writing_assistance_core import FormalityDetector


xlm_roberta_classifier = XLMRobertaForSequenceClassification.from_pretrained('s-nlp/xlmr_formality_classifier')
xlm_roberta_tokenizer = XLMRobertaTokenizerFast.from_pretrained('s-nlp/xlmr_formality_classifier')


class XLMRoberta(FormalityDetector):
    """XLMRoberta model formality detector"""

    model = xlm_roberta_classifier
    tokenizer = xlm_roberta_tokenizer

    @staticmethod
    def _predict_from_list(list_of_texts: list[str]) -> list[dict[str, float]]:
        inputs = XLMRoberta.tokenizer(
            list_of_texts,
            add_special_tokens=True,
            return_token_type_ids=True,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        with torch.no_grad():
            outputs = XLMRoberta.model(**inputs)

        scores = []
        for text_scores in outputs.logits.softmax(dim=1):
            score = {'formal': text_scores[0], 'informal': text_scores[1]}
            scores.append(score)

        return scores

    @staticmethod
    def _predict_from_str(text: str) -> list[dict[str, float]]:
        inputs = XLMRoberta.tokenizer(
            text,
            add_special_tokens=True,
            return_token_type_ids=True,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        with torch.no_grad():
            outputs = XLMRoberta.model(**inputs)

        scores = []
        for text_scores in outputs.logits.softmax(dim=1):
            score = {'formal': text_scores[0], 'informal': text_scores[1]}
            scores.append(score)

        return scores
