import torch

from konta.tasks import (
    KoNTATranslationFactory,
    KoNTAAugmentationFactory
)

SUPPORTED_TASKS = {
    "mt": KoNTATranslationFactory,
    "machine_translation": KoNTATranslationFactory,
    "aug": KoNTAAugmentationFactory,
    "text_augumentation": KoNTAAugmentationFactory
}

LANG_ALIASES = {
    "english": "en",
    "eng": "en",
    "korean": "ko",
    "kor": "ko",
    "kr": "ko"
}

class KoNTA:
    pass