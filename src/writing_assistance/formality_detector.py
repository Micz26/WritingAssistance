from typing import Type, Literal

from writing_assistance_core import FormalityDetector as AbstractFormalityDetector
from writing_assistance.models import Deberta, XLMRoberta, GPTFormalityDetector


class FormalityDetector:
    """Handler for Formality Detectors"""

    _detectors: dict[str, Type[AbstractFormalityDetector]] = {
        'deberta': Deberta,
        'xlm_roberta': XLMRoberta,
        'gpt': GPTFormalityDetector,
    }

    @classmethod
    def get_detector(cls, name: str) -> Type[AbstractFormalityDetector]:
        if name not in cls._detectors:
            raise ValueError(f'Wrong detector provided: {name}')
        return cls._detectors[name]

    @classmethod
    def predict(
        cls, detector_name: Literal['deberta', 'xlm_roberta', 'gpt'], text: str | list[str]
    ) -> list[dict[str, float]]:
        detector_cls = cls.get_detector(detector_name)
        return detector_cls.predict(text)
