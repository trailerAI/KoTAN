# KoTAN: Korean Translation and Augmentation with fine-tuned NLLB
NLLB 모델을 fine-tuning하여 한국어 번역 및 한국어 데이터 셋을 증강할 수 있는 python 패키지 입니다.
사용법은 아래와 같습니다.

## Usage 1. pip

### Package install
```
pip3 install kotan
```

### Translation
```
from kotan import KoTAN

mt=KoTAN(task="translation", tgt="en")

inputs=['나는 온 세상 사람들이 행복해지길 바라', '나는 선한 영향력을 펼치는 사람이 되고 싶어']
mt.predict(inputs, 'en')
```

### Backtranslation

#### Origin nllb model (before fine-tuning)
```
from kotan import KoTAN

bt=KoTAN(task="augmentation", level="origin")
inputs=['나는 온 세상 사람들이 행복해지길 바라', '나는 선한 영향력을 펼치는 사람이 되고 싶어']
bt.predict(inputs)
```

#### Fine-tuned nllb model with Aihub datasets.
```
from kotan import KoTAN

bt = KoTAN(task="augmentation", level="fine")
inputs=['나는 온 세상 사람들이 행복해지길 바라', '나는 선한 영향력을 펼치는 사람이 되고 싶어']
bt.predict(inputs)
```

