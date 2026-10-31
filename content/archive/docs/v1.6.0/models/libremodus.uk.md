---
title: LibreMODUS
families:
  - libremodus
seo_title: 'LibreMODUS у LibreYOLO: аналіз зображень будь-що-в-що-завгодно'
description: >-
  Використовуйте LibreMODUS у LibreYOLO для глибини, нормалей, країв і виявлення
  та поєднуйте їх через any2any(). Лише інференс; ваги завантажуються з
  EPFL-VILAB.
lead: >-
  LibreMODUS є інтеграцією лише для інференсу контрольної точки MODUS 14B-A7B,
  моделі будь-що-в-що-завгодно, яка перетворює один отриманий із зображення вхід
  на інший: RGB на вході, глибина на виході; глибина на вході, нормалі на
  виході; будь-який із них разом із фразою, рамки на виході. LibreYOLO підтримує
  чотири завдання через стандартний API predict і ширший набір через any2any().
keywords:
  - LibreMODUS
  - MODUS
  - any-to-any
  - оцінювання глибини
  - нормалі поверхні
  - детекція країв
  - виявлення за описом
  - EPFL VILAB
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(size="14b-a7b", task="normal")
        result = model.predict("room.jpg")
        normals = result.normal_map.data

        model.set_task("edge")
        result = model.predict("room.jpg")
        edges = result.edges.data

        # Без власного словника detect декодує токени міток COCO контрольної
        # точки в послідовні ідентифікатори класів COCO-80.
        model.set_task("detect")
        result = model.predict("street.jpg")
        print(result.boxes.xyxy)
    - label: Прив'язування фрази
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(task="detect")
        # set_classes() перемикає виявлення на прив'язування фраз: кожна фраза
        # виконується окремо й повертається через той самий контракт Boxes.
        model.set_classes(["red bus", "cyclist"])
        result = model.predict("street.jpg", conf=0.2)
        print(result.boxes.xyxy, result.boxes.cls)
    - label: any2any()
      language: python
      code: >
        from libreyolo import LibreMODUS


        model = LibreMODUS()


        # Від одного до трьох отриманих із зображення входів (rgb, depth,
        normal,

        # canny/edge) разом із необов'язковим допоміжним текстом до однієї цілі.

        result = model.any2any(
            inputs={"rgb": "room.jpg"},
            target="normal",
            steps=10,
            cfg=2.0,
            seed=0,
        )

        normals = result.normal_map.data


        # Прив'язування через any2any() потребує текстового входу з назвою
        фрази.

        result = model.any2any(
            {"rgb": "street.jpg", "text": "red bus"},
            target="grounding",
        )

        print(result.boxes.xyxy)
source_hash: 7386886d4c36ea9a
---

## Встановлення

LibreMODUS потребує власного набору додаткових залежностей, який установлює
`accelerate` для розподілу великої моделі, потрібного цій контрольній точці.

```bash
pip install "libreyolo[modus]"
```

LibreYOLO не розповсюджує й не дзеркалює ваги MODUS. Типово завантаження моделі
`LibreMODUS` отримує потрібні файли безпосередньо з `EPFL-VILAB/MODUS` у
зафіксованій редакції Hugging Face. Нове завантаження завжди потребує власного
автентифікованого облікового запису Hugging Face користувача, навіть якщо
обмеження початкового хостингу тимчасово відкрите. Перегляньте й прийміть
початкові умови, а потім автентифікуйтеся:

```bash
hf auth login
```

```python
from libreyolo import LibreMODUS

model = LibreMODUS(token="hf_...")
```

Щоб уникнути будь-якого мережевого запиту, укажіть уже наявний знімок:

```python
model = LibreMODUS(checkpoint_path="/models/MODUS")
```

Цей каталог має містити `model.safetensors`, `ae.safetensors`,
`llm_config.json`, `vit_config.json`, `tokenizer_config.json`,
`vocab.json` і `merges.txt`. Дозволи умов контрольної точки пояснено в
розділі «Ліцензування» нижче.

## Передбачення

<code-tabs name="predict" />

Стандартний API завдань охоплює чотири завдання, кожне зіставлене з однією
ціллю MODUS: `depth` із відносною глибиною (`result.depth_map`), `normal`
із нормалями поверхні (`result.normal_map`), `edge` із краями у стилі
Canny (`result.edges`) і `detect` із рамками COCO-80 (`result.boxes`),
якщо `set_classes()` не перемикає його на прив'язування фрази.
`set_task()` перемикає їх у тій самій завантаженій моделі. Випущений рецепт
використовує десять кроків потокового семплювання з текстовим керуванням 4.0
і керуванням зображенням 2.0; перевизначте їх через `inference_steps=`,
`inference_cfg=` і `inference_image_cfg=` під час створення.

`any2any()` відкриває ширшу публічну поверхню аналізу: від одного до трьох
отриманих із зображення входів (`rgb`, `depth`, `normal`,
`canny`/`edge`) разом із необов'язковим допоміжним текстом, поєднаних для
однієї з цілей: глибини, нормалей, країв, отриманих із SAM країв, виявлення
COCO або прив'язування фрази. Усі отримані із зображення входи мають описувати
однакове вирівняне полотно; LibreMODUS відхиляє невідповідні ширини й висоти
замість незалежної зміни їх розміру. `chain=(...)` генерує проміжні цілі й
повертає їх у той самий контекст у межах навчального бюджету контрольної
точки з трьох умов. `verify=N` (`N >= 2`) генерує N кандидатів і зберігає
того, що має найвищу оцінку обмеженої перевірки самоузгодженості, доступну як
`result.verification_score`.

`dtype="bf16"` (типове значення) відповідає точності випущеної контрольної
точки; `dtype="fp8"` зберігає придатні лінійні ваги основи декодера як E4M3
з масштабом на кожен вихідний канал, один раз перетворює їх до локального кешу
в `~/.cache/libreyolo/modus/fp8` і деквантує до типу входу для кожного
матричного множення. Тому це економить пам'ять без зниження точності на рівні
активацій.

`train()`, `val()` і `export()` спричиняють помилку: LibreMODUS
призначена лише для інференсу, валідація датасету не пропонується, а шляху
експорту ONNX, TensorRT або TFLite немає. Батчеве `predict()` і аугментація
під час тестування також не підтримуються; кожен виклик обробляє одне
зображення.

## Ліцензування

<provenance-box>

LibreYOLO ніде не розміщує й не дзеркалює контрольну точку MODUS, зокрема у
власній організації Hugging Face: завантаження завжди отримує зафіксовану
редакцію безпосередньо з EPFL-VILAB/MODUS або читає вже наявний на диску
знімок за `checkpoint_path`.

</provenance-box>

## Цитування

<citation-block />
