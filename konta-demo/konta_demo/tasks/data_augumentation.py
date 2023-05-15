from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from transformers import BatchEncoding
import torch

class KoNTAAugmentationFactory():

    def __init__(self, task, src, LANG_ALIASES: dict):
        super().__init__()
        self.task = task
        self.src = src
        self.LANG_ALIASES = LANG_ALIASES

    def load(self, device):
        ko2en_tokenizer = NllbTokenizer.from_pretrained("KoJLabs/nllb-finetuned-ko2en")
        en2ko_tokenizer = NllbTokenizer.from_pretrained("KoJLabs/nllb-finetuned-en2ko")

        ko2en_model = AutoModelForSeq2SeqLM.from_pretrained("KoJLabs/nllb-finetuned-ko2en").to(device)
        en2ko_model = AutoModelForSeq2SeqLM.from_pretrained("KoJLabs/nllb-finetuned-en2ko").to(device)

        return KoNTAAugmentation(
            ko2en_tokenizer, en2ko_tokenizer,
            ko2en_model, en2ko_model,
            device,
            self.LANG_ALIASES
        )


class KoNTAAugmentation:

    def __init__(self, 
                 ko2en_tokenizer, en2ko_tokenizer, 
                 ko2en_model, en2ko_model, 
                 device, LANG_ALIASES):
        self.ko2en_tokenizer = ko2en_tokenizer
        self.en2ko_tokenizer = en2ko_tokenizer

        self.ko2en_model = ko2en_model
        self.en2ko_model = en2ko_model

        self.device = device
        self.LANG_ALIASES = LANG_ALIASES

    def predict(self, text):

        # ko2en
        translation = self._translate(text, 'eng_Latn', self.ko2en_tokenizer, self.ko2en_model)

        # en2ko
        backtranslation = self._translate(translation, 'kor_Hang', self.en2ko_tokenizer, self.en2ko_model)

        return backtranslation


    def _translate(self, text, tgt, tokenizer, model):

        # translation
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        translated_tokens = model.generate(
            inputs['input_ids'], forced_bos_token_id=tokenizer.lang_code_to_id[tgt], max_length=128
        )

        translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        
        return translation


        