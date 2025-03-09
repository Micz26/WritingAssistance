from writing_assistance_core import FormalityDetector

from writing_assistance.utils.gpt_components import gpt_rate_formality


class GPTFormalityDetector(FormalityDetector):
    """GPT (gpt-4o-mini) Formality detector"""

    @staticmethod
    def _predict_from_list(list_of_texts: list[str]) -> list[dict[str, float]]:
        responses = []
        for text in list_of_texts:
            responses.append(GPTFormalityDetector._predict_from_str(text)[0])
        return responses

    @staticmethod
    def _predict_from_str(text: str) -> list[dict[str, float]]:
        score = gpt_rate_formality(text=text)
        return [score]
