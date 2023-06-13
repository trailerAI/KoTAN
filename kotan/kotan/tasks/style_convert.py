from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from transformers import BatchEncoding

class KoTANStyleConversiontFactory:
    """
    Speech-style convert using NHNDQ/bart-speech-style-converter

    - dataset: Train

    Args:
        src (str): source language
        
    Returns:
        class: KoTANConversion class

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
        self.style = style

    def load(self, device: str):
        tokenizer = AutoTokenizer.from_pretrained("NHNDQ/bart-speech-style-converter")
        model = AutoModelForSeq2SeqLM.from_pretrained("NHNDQ/bart-speech-style-converter").to(device)
            
        return KoTANConversion(
            model,
            tokenizer,
            device,
            self.style
        )
    

class KoTANConversion:
    def __init__(self, model, tokenizer, device, style):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.style = style

    def predict(self, text):
        """
        Predict a style convert result

        Args:
            text (str): input text
            style (str): speech style

        Returns:
            output (list): Style conversion results
        """

        if len(text) > 1:
            input_text = []
            
            for txt in text:
                input_text.append(f"{self.style} 형식으로 변환:" + txt)
        else:
            input_text = f"{self.style} 형식으로 변환:" + text

        
        inputs = self.tokenizer(input_text, max_length=128, return_tensors="pt")["input_ids"]
            
        convert_tokens = self.model.generate(
            inputs.to(self.device), max_length=128
        )

        output = self.tokenizer.batch_decode(convert_tokens, 
                                             skip_special_tokens=True, 
                                             clean_up_tokenization_spaces=False
                                             )
            
        return output
    

