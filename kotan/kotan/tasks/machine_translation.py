from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from transformers import BatchEncoding

class KoTANTranslationFactory:
    """
    Machine translation using facebook/nllb-200-distilled-600M Meta model

    - dataset: Train

    Args:
        src (str): source language
        
    Returns:
        class: KoTANTranslation class

    Examples:
        >>> mt = KoTAN(task="translation", tgt="en")
    """
    def __init__(
            self,
            task,
            tgt,
            level,
            style
            ):
        super().__init__()
        self.task = task
        self.tgt = tgt

    def load(self, device: str):
        if self.tgt == "kor_Hang":
            tokenizer = AutoTokenizer.from_pretrained("KoJLabs/nllb-finetuned-en2ko")
            model = AutoModelForSeq2SeqLM.from_pretrained("KoJLabs/nllb-finetuned-en2ko").to(device)
            
        if self.tgt == "eng_Latn":
            tokenizer = AutoTokenizer.from_pretrained("KoJLabs/nllb-finetuned-ko2en")
            model = AutoModelForSeq2SeqLM.from_pretrained("KoJLabs/nllb-finetuned-ko2en").to(device)

        return KoTANTranslation(
            model,
            tokenizer,
            device,
            self.tgt
        )
    

class KoTANTranslation:
    def __init__(self, model, tokenizer, device, tgt):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.tgt = tgt

    def predict(self, text):
        """
        Predict a translation result

        Args:
            text (str): input text

        Returns:
            output (list): Translation results
        """
        
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            
        translated_tokens = self.model.generate(
            **inputs.to(self.device), forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt], max_length=128
        )

        output = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            
        return output
    

