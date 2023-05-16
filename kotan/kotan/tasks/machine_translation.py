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
            LANG_ALIASES,
            level
            ):
        super().__init__()
        self.task = task
        self.tgt = tgt
        self.LANG_ALIASES = LANG_ALIASES

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
            self.LANG_ALIASES
        )
    

class KoTANTranslation:
    def __init__(self, model, tokenizer, device, LANG_ALIASES):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.LANG_ALIASES = LANG_ALIASES

    def predict(self, text, tgt):
        """
        Predict a translation result

        Args:
            text (str): input text
            tgt (str): target language

        Returns:
            output (list): Translation results
        """
        
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")

        if isinstance(inputs['input_ids'], list):
            input_dict = {}            
            input_dict['input_ids'] = torch.LongTensor(inputs['input_ids']).reshape(1,len(inputs['input_ids']))
            input_dict['attention_mask'] = torch.LongTensor(inputs['attention_mask']).reshape(1,len(inputs['attention_mask']))
            inputs = BatchEncoding(input_dict)
            
        translated_tokens = self.model.generate(
            **inputs.to(self.device), forced_bos_token_id=self.tokenizer.lang_code_to_id[self.LANG_ALIASES[tgt]], max_length=128
        )

        output = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            
        return output
    

