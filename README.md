# KoTAN: Korean Translation and Augmentation with fine-tuned NLLB

A `KoTAN` package can exercise korean data augmentation task and en->ko, ko->en translation task.
In case of translation model, we are fine-tuning facebook [NLLB](https://arxiv.org/abs/2207.04672) model. About data augmentation task, we processe backtranslation task.
In addition, we also provide speech-style conversion options.

## Package install
* `torch=2.0.0 (cuda 12.0)` and `python>=3.8` are avaliable.
* You can install the package with below command.
```
pip3 install kotan
```

## Usage
* You can use `KoTAN` with below command.
* Import package.
```python
>>> from kotan import KoTAN
```
* Avaliable tasks
```python
>>> KoTAN.available_tasks()
```
* Avaliable languages
```python
>>> KoTAN.available_lang()
```
* Data augmentation options
```python
>>> KoTAN.available_level()
```
  - origin: Before fine-tuning nllb model.
  - fine: After fine-tuning nllb model.
* Speech-style conversion options
```python
>>> KoTAN.available_style()
```
* formal: 문어체
* informal: 구어체
* android: 안드로이드
* azae: 아재
* chat: 채팅
* choding: 초등학생
* emoticon: 이모티콘
* enfp: enfp
* gentle: 신사
* halbae: 할아버지
* halmae: 할머니
* joongding: 중학생
* king: 왕
* naruto: 나루토
* seonbi: 선비
* sosim: 소심한
* translator: 번역기

### Translation
```python
>>> from kotan import KoTAN
>>> mt = KoTAN(task="translation", tgt="en")
>>> inputs = ['나는 온 세상 사람들이 행복해지길 바라', '나는 선한 영향력을 펼치는 사람이 되고 싶어']
>>> mt.predict(inputs)
```

### Data Augmentation

#### Origin nllb model (before fine-tuning)
```python
>>> from kotan import KoTAN
>>> aug = KoTAN(task="augmentation", level="origin")
>>> inputs = ['나는 온 세상 사람들이 행복해지길 바라', '나는 선한 영향력을 펼치는 사람이 되고 싶어']
>>> aug.predict(inputs)
```

#### Fine-tuned nllb model with Aihub datasets.
```python
>>> from kotan import KoTAN
>>> aug = KoTAN(task="augmentation", level="fine")
>>> inputs=['나는 온 세상 사람들이 행복해지길 바라', '나는 선한 영향력을 펼치는 사람이 되고 싶어']
>>> aug.predict(inputs)
```

#### Apply style-convert option.
```python
>>> from kotan import KoTAN
>>> aug = KoTAN(task="augmentation", style="chat")
>>> inputs=['나는 온 세상 사람들이 행복해지길 바라', '나는 선한 영향력을 펼치는 사람이 되고 싶어']
>>> aug.predict(inputs)
```

### Speech-style conversion
```python
>>> from kotan import KoTAN
>>> style = KoTAN(task="augmentation", style="king")
>>> inputs=['나는 온 세상 사람들이 행복해지길 바라', '나는 선한 영향력을 펼치는 사람이 되고 싶어']
>>> style.predict(inputs)
```

## Demo
[Huggingface KoTAN space](https://huggingface.co/spaces/KoJLabs/KoTAN)

## Citation
```
@misc{KoTAN,
  author       = {Juhwan Lee and Jisu Kim},
  title        = {KoTAN: Korean Translation and Augmentation with fine-tuned NLLB},
  howpublished = {\url{https://github.com/KoJLabs/KoTAN}},
  year         = {2023},
}
```

## Contributors
[Jisu, Kim](https://github.com/merry555), [Juhwan, Lee](https://github.com/juhwanlee-diquest)

## License
`KoTAN` project follow **Apache License 2.0 lisence**
