import torch

from .tasks import (
    KoTANTranslationFactory,
    KoTANAugmentationFactory,
    KoTANStyleConversiontFactory
)

from .const import LANG_ALIASES, LEVEL, STYLE


# Task list
SUPPORTED_TASKS = {
    "mt": KoTANTranslationFactory,
    "translation": KoTANTranslationFactory,
    "aug": KoTANAugmentationFactory,
    "augmentation": KoTANAugmentationFactory,
    "style": KoTANStyleConversiontFactory,
    "convert": KoTANStyleConversiontFactory
}


class KoTAN:
    def __new__(
            cls,
            task,
            tgt="en",
            level="fine",
            style="formal"
            ):
        
        if task not in SUPPORTED_TASKS:
            raise KeyError("Unknown task {}, available tasks are {}".format(
                task,
                list(SUPPORTED_TASKS.keys()),
            ))

        if tgt not in LANG_ALIASES:
            raise KeyError("Unknown target language {}, available target languages are {}".format(
                task,
                list(LANG_ALIASES.keys()),
            ))
        
        if level not in LEVEL:
            raise KeyError("Unknown level {}, available levels are {}".format(
                task,
                list(LEVEL.keys()),
            ))
        
        if style not in STYLE:
            raise KeyError("Unknown style {}, available levels are {}".format(
                task,
                list(STYLE.keys()),
            ))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        task_module = SUPPORTED_TASKS[task](
            task,
            LANG_ALIASES[tgt],
            LEVEL[level],
            STYLE[style]
        ).load(device)

        return task_module

    @staticmethod
    def available_tasks():
        """
        Returns available tasks in KoTAN project

        Returns:
            str: Supported task names

        """
        return "Available tasks are {}".format(list(SUPPORTED_TASKS.keys()))
    
    @staticmethod
    def available_lang():
        """
        Returns available language in KoTAN project

        Returns:
            str: Supported language names

        """
        return "Available tasks are {}".format(list(LANG_ALIASES.keys()))
    
    @staticmethod
    def available_level():
        """
        Returns available level in KoTAN project

        Returns:
            str: Supported level names

        """
        return "Available tasks are {}".format(list(LEVEL.keys()))
    
    @staticmethod
    def available_style():
        """
        Returns available convert style in KoTAN project

        Returns:
            str: Supported style names

        """
        return "Available tasks are {}".format(list(STYLE.keys()))