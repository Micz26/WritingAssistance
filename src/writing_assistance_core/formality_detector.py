from abc import ABC, abstractmethod


class FormalityDetector(ABC):
    """Base class for Formality Detector models"""

    @classmethod
    def predict(cls, text: str | list[str]) -> list[dict[str, float]]:
        if isinstance(text, str):
            return cls._predict_from_str(text)
        return cls._predict_from_list(text)

    @abstractmethod
    def _predict_from_list(list_of_texts: list[str]) -> list[dict[str, float]]:
        pass

    @abstractmethod
    def _predict_from_str(text: str) -> list[dict[str, float]]:
        pass
