# KoTAN: Korean Translation and Augmentation with fine-tuned NLLB

영->한, 한->영 번역 및 한국어 데이터 셋을 증강할 수 있도록 도와주는 라이브러리 `KoTAN`입니다.
번역 모델의 경우, facebook의 NLLB 모델을 fine-tuning하였고, 데이터 셋 증강의 경우 backtranslation 과정을 거쳐 진행했습니다.

## Package install
* `torch=2.0.0 (cuda 12.0)`과 `python>=3.8` 환경에서 정상적으로 동작합니다.
* 아래 커멘드를 입력하여 패키지를 설치할 수 있습니다.
```
pip3 install kotan
```

## Usage
* 아래와 같은 명령어로 `KoTAN` 을 사용할 수 있습니다.
* 먼저, `KoTAN` 을 import 하기 위해 다음과 같은 명령어를 실행해야 합니다.
```python
>>> from kotan import KoTAN
```
* import후, 현재 `KoTAN` 에서 지원하는 태스크를 확인할 수 있습니다.
```python
>>> KoTAN.available_tasks()
```
* 어떤 언어가 지원되는지 확인하기 위해서는 아래와 같이 명령어를 입력해 확인할 수 있습니다.
```python
>>> KoTAN.available_lang()
```
* 데이터 증강시, 어떤 옵션이 있는지 확인하기 위해서는 아래와 같이 명령어를 입력해 확인할 수 있습니다.
```python
>>> KoTAN.available_level()
```
  - origin: nllb모델을 fine-tuning하기 전 모델 입니다.
  - fine: nllb모델을 fine-tuning한 모델 입니다.

### Translation
* 아래와 같이 명령어를 입력해 `KoTAN` 의 번역 태스크를 수행할 수 있습니다.
```python
>>> from kotan import KoTAN
>>> mt = KoTAN(task="translation", tgt="en")
>>> inputs = ['나는 온 세상 사람들이 행복해지길 바라', '나는 선한 영향력을 펼치는 사람이 되고 싶어']
>>> mt.predict(inputs, 'en')
```

### Data Augmentation
* 아래와 같이 명령어를 입력해 `KoTAN` 의 데이터 증강 태스크를 수행할 수 있습니다.

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

## Citation
KoTAN 라이브러리를 프로젝트 혹은 연구에 활용하신다면 아래 정보를 인용해주시길 바랍니다.
```
@misc{pororo,
  author       = {Juhwan Lee and Jisu Kim},
  title        = {KoTAN: Korean Translation and Augmentation with fine-tuned NLLB},
  howpublished = {\url{https://github.com/KoJLabs/KoTAN}},
  year         = {2023},
}
```

## Contributors
[김지수](https://github.com/merry555), [이주환](https://github.com/juhwanlee-diquest)

## License
`KoTAN` 프로젝트는 **Apache License 2.0 라이센스**를 따릅니다.
