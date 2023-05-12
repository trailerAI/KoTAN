from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

class KoNTATranslationFactory():
    """
    Machine translation using NLLN Meta model

    - dataset: Train
    - metric: BLEU score
        +-----------------+-----------------+------------+
        | Source Language | Target Language | BLEU score |
        +=================+=================+============+
        | English         |  Korean         |            |
        +-----------------+-----------------+------------+
        | Korean          |  English        |            |
        +-----------------+-----------------+------------+

    Args:
        text (str): input text to be translated
        src (str): source language
        tgt (str): target language
        
    Returns:
        str: machine translation sentence

    Examples:
        >>> mt = KoNTA(task="translation", src="en")
    """
    def __init__(
            self,
            text: str,
            src: str,
            tgt: str 
            ):
        super().__init__()
        self.text = text
        self.src = src
        self.tgt = tgt

    def predict(self, device:str):
        """
        Load fine tuned NLLB machine translation model

        Args:
            device (str): device information

        Returns:
            object: fine tuned NLLB machine translation model
        """

        tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", legacy_behaviour=True, src_lang=self.src, tgt_lang=self.tgt)
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

        inputs = tokenizer(self.text, return_tensor="pt")

        print(inputs)

        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[self.tgt], max_length=128
        )

        output = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        return output