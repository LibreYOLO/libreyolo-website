---
title: API моделей зору й мови
seo_title: 'API LibreVLM: псевдоніми, set_classes і chat'
description: >-
  Фабрика LibreVLM, усі псевдоніми моделей, постійний словник set_classes,
  set_task, універсальний вихід chat і причина використання умовної впевненості.
lead: >-
  LibreVLM завантажує генеративну модель зору й мови та керує нею як детектором
  об'єктів. Список класів є підказкою, а не фіксованою головою, і модель
  повертає ті самі Results, що й будь-яке інше сімейство.
keywords:
  - LibreVLM
  - виявлення моделлю зору й мови
  - Qwen3-VL
  - LFM2-VL
  - InternVL3
  - SmolVLM2
  - Florence-2
  - чат LibreYOLO
last_verified: 1.5.0
verification: >-
  Псевдоніми прочитано з libreyolo/models/vlm/__init__.py; репозиторії, розміри
  й списки задач взято з модулів сімейств у libreyolo/models/vlm/ і
  libreyolo/models/sensenova/model.py; правила викликів і помилок взято з
  libreyolo/models/vlm/base.py, усе у версії v1.5.0.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[vlm]'
  usage:
    - label: Виявлення за відкритим словником
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        model.set_classes(["person", "skateboard"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
    - label: Запитання у вільній формі
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        print(model.chat(SAMPLE_IMAGE, "How many people are in this image?"))
source_hash: 57ddac08bc4d4e05
---

## Встановлення

Для цього рівня потрібна додаткова залежність `vlm`.

<code-tabs name="install" />

## Фабрика

```python
LibreVLM(model: str = "qwen3-vl-4b", **kwargs) -> LibreVLMModel
```

`model`, це псевдонім, а не шлях. `**kwargs` передається конструктору
сімейства, який приймає `device`, `names` (початковий словник,
рівнозначний виклику `set_classes` після завантаження), `prompt`
(перевизначення підказки виявлення) і `max_new_tokens`. Невідомий псевдонім
спричиняє `ValueError` зі списком усіх псевдонімів.

<code-tabs name="usage" />

## Псевдоніми

| Сімейство | Псевдоніми | Розміри | Ваги |
|---|---|---|---|
| Qwen3-VL | `qwen3-vl`, `qwen3-vl-2b`, `qwen3-vl-4b`, `qwen3-vl-8b` | `2b`, `4b`, `8b` | `Qwen/Qwen3-VL-2B-Instruct`, `-4B-`, `-8B-` |
| LFM2-VL | `lfm2-vl`, `lfm2-vl-450m`, `lfm2-vl-1.6b` | `450m`, `1.6b` | `LiquidAI/LFM2.5-VL-450M`, `-1.6B` |
| InternVL3 | `internvl3`, `internvl3-1b`, `internvl3-2b`, `internvl3-8b` | `1b`, `2b`, `8b` | `OpenGVLab/InternVL3-1B-hf`, `-2B-hf`, `-8B-hf` |
| SmolVLM2 | `smolvlm2`, `smolvlm2-2.2b`, `smolvlm2-500m` | `2.2b`, `500m` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, `SmolVLM2-500M-Video-Instruct` |
| Florence-2 | `florence-2`, `florence2`, `florence-2-base`, `florence-2-large` | `base`, `large` | `florence-community/Florence-2-base`, `-large` |
| Kosmos-2 | `kosmos-2`, `kosmos2` | `224` | `microsoft/kosmos-2-patch14-224` |
| LocateAnything | `locate-anything`, `locateanything`, `locate-anything-3b`, `locateanything-3b` | `3b` | `nvidia/LocateAnything-3B` |
| SenseNova-Vision | `sensenova-vision`, `sensenova-vision-7b`, `sensenovavision` | `7b` | `LibreYOLO/SenseNovaVision7b` |
| LibreMODUS | `libremodus`, `libremodus-14b-a7b`, `modus`, `modus-14b-a7b` | `14b-a7b` | Закріплений знімок upstream |

Стандартний псевдонім, `qwen3-vl-4b`. Розміри для стандартного псевдоніма
кожного сімейства наведено першими: `qwen3-vl` відповідає `4b`,
`lfm2-vl`, `450m`, `internvl3`, `2b`, `smolvlm2`, `2.2b`,
а `florence-2`, `base`.

`LibreVLM`, `LibreLFM2VL`, `LibreQwen3VL`, `LibreSmolVLM2`,
`LibreInternVL3`, `LibreFlorence2`, `LibreKosmos2`,
`LibreLocateAnything` і `LibreMODUS` (також пишеться `LibreModus`)
експортуються на рівні пакета.

## Задачі

Більшість сімейств обслуговує лише `detect`. Два підтримують більше:

| Сімейство | Підтримувані задачі |
|---|---|
| LocateAnything | `detect`, `point` |
| SenseNova-Vision | `detect`, `segment`, `panoptic`, `pose`, `point`, `depth`, `ocr` |

Оскільки задачу визначає підказка, а не вбудована в контрольну точку
конфігурація, її можна перемкнути в завантаженій моделі:

```python
model.set_task(task: str) -> LibreVLMModel
```

Задача перевіряється за списком підтримки сімейства, зберігається для наступних
викликів `predict()` і `track()`, а модель повертається, щоб виклики можна
було ланцюжити.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreVLMModel
```

Задає відкритий словник. Працюють будь-які слова, оскільки модель отримує їх
у підказці, а не обмежується фіксованою головою. Список має бути непорожнім,
а його елементи унікальними без урахування регістру. Передавання окремого
рядка спричиняє `TypeError`, оскільки інакше його було б розбито на
односивольні класи. Словник зберігається: задайте його один раз після
завантаження, і він діятиме до наступного встановлення.

## chat

```python
model.chat(image, prompt, max_new_tokens=None, color_format="auto") -> str
```

Необроблена мультимодальна генерація: на вході зображення й підказка, на
виході декодований текст без змін. Це універсальний вихід з-під зручної
обгортки виявлення для запитань у вільній формі, підрахунку або формату
виводу, якого не підтримує обгортка. `max_new_tokens` використовує резервне
значення `MAX_NEW_TOKENS` сімейства, яке в базовому класі дорівнює 1024.
Декодування жадібне з невеликим штрафом за повторення.

## Упевненість

Згенерований вивід не має каліброваної впевненості для кожної рамки. Ця версія
призначає постійне умовне значення, щоб працювали `predict`, малювання та
`track`, тому фільтрація `conf=` і mAP є радше формальними, ніж
змістовними. Саме тому `val()` спричиняє помилку: COCO mAP за умовними
оцінками вводив би в оману.

## Передбачення та відстеження

Застосовується стандартний інтерфейс передбачення, і `track()` працює,
тому детектор VLM входить до того самого конвеєра, що й будь-яке інше
сімейство. Дві політики рівня класу відрізняються від згорткового детектора:
аугментацію під час тестування вимкнено, оскільки багатомасштабна аугментація
не має сенсу для генератора з фіксованою роздільною здатністю, а пакетне
передбачення вимкнено, оскільки генерація авторегресійна й попередня обробка
повертає кодування тексту та зображення, а не тензор зображень, який можна
скласти в пакет.

## Не підтримується

`train()`, `val()` і `export()` спричиняють `NotImplementedError`.
Виконайте донавчання в upstream і завантажте отримані ваги.

## Віддалений код

Кожне сімейство в постачанні завантажується через нативний клас моделі, тому
LibreYOLO стандартно не виконує код стороннього репозиторію. Сімейство, якому
він справді потрібен, має явно погодитися й закріпити ревізію знімка;
LocateAnything, єдине таке сімейство, закріплене на коміті
`c32291ca5e996f5a7a485845b4f57a233936bba0`.

LibreMODUS є явним винятком зі схеми контрольних точок: її псевдонім
визначається як каталог закріплених файлів upstream, а не файл LibreYOLO
`.pt`; LibreYOLO не додає до нього метадані v1.0 і не публікує його
повторно.

