from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from konlpy.tag import Twitter
import numpy as np

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
            tokenizer = AutoTokenizer.from_pretrained("NHNDQ/nllb-finetuned-en2ko")
            model = AutoModelForSeq2SeqLM.from_pretrained("NHNDQ/nllb-finetuned-en2ko").to(device)
            
        if self.tgt == "eng_Latn":
            tokenizer = AutoTokenizer.from_pretrained("NHNDQ/nllb-finetuned-ko2en")
            model = AutoModelForSeq2SeqLM.from_pretrained("NHNDQ/nllb-finetuned-ko2en").to(device)

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

        ## text가 '' 일 경우, '' 그대로 출력되도록 후처리
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            
        translated_tokens = self.model.generate(
            **inputs.to(self.device), forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt], max_length=128
        )

        output = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

        post_output = []

        for inp, outp in zip(text, output):
            if (inp == '') or (inp == '!') or (inp == '?') or (inp == '.') or (inp == ','):
                post_output.append(inp)
            else:
                post_output.append(outp)

        return post_output
    
    def _post_process(self, text):
        textList = []
        emojiList = []
        twit = Twitter()

        posText = twit.pos(text)
        posArray = np.array(posText)

        for i in range(len(posArray)):
            if posArray[i][1] == 'KoreanParticle':
                emojiList.append(posArray[i][0])

        for i in range(len(emojiList)):
            splitText = text.split(emojiList[i], maxsplit=1)

            if splitText[0] == '':
                textList.append('')
            else:
                textList.append(splitText[0])

            try:
                if len(splitText[1:]) > 1:
                    text = ''.join(splitText[1:]).strip()
                else:
                    text = splitText[1:][0].strip()

            except:
                break

            try:
                if text in emojiList[i+1]:
                    pass
            except:
                textList.append(splitText[-1])
                emojiList.append('')
                break

        ## 이모지 없는 경우            
        if len(emojiList) < 1:
            emojiList.append('')
            textList.append(text)
                
        return emojiList, textList
    

