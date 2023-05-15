import torch

from konta.tasks import (
    KoNTATranslationFactory,
    KoNTAAugmentationFactory
)

SUPPORTED_TASKS = {
    "mt": KoNTATranslationFactory,
    "translation": KoNTATranslationFactory,
    "aug": KoNTAAugmentationFactory,
    "augumentation": KoNTAAugmentationFactory
}

LANG_ALIASES = {
    "english": "eng_Latn",
    "eng": "eng_Latn",
    "en": "eng_Latn",
    "ko": "kor_Hang",
    "korean": "kor_Hang",
    "kor": "kor_Hang",
    "kr": "kor_Hang"
}

class KoNTA:
    def __new__(
            cls,
            task: str,
            src: str
            ):
        
        if task not in SUPPORTED_TASKS:
            raise KeyError("Unknown task {}, available tasks are {}".format(
                task,
                list(SUPPORTED_TASKS.keys()),
            ))

        if src not in LANG_ALIASES:
            raise KeyError("Unknown target language {}, available target languages are {}".format(
                task,
                list(LANG_ALIASES.keys()),
            ))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        task_module = SUPPORTED_TASKS[task](
            task,
            LANG_ALIASES[src],
            LANG_ALIASES
        ).load(device)

        return task_module

    @staticmethod
    def available_tasks() -> str:
        """
        Returns available tasks in KoNTA project

        Returns:
            str: Supported task names

        """
        return "Available tasks are {}".format(list(SUPPORTED_TASKS.keys()))
    
    @staticmethod
    def available_lang() -> str:
        """
        Returns available language in KoNTA project

        Returns:
            str: Supported language names

        """
        return "Available tasks are {}".format(list(LANG_ALIASES.keys()))