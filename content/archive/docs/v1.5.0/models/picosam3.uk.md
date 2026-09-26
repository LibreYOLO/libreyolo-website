---
title: PicoSAM3
families:
  - picosam3
seo_title: 'PicoSAM3: сегментація на edge-пристроях за запитами рамками в LibreYOLO'
description: >-
  Використовуйте PicoSAM3 у LibreYOLO для сегментації областей на edge-сенсорах
  за запитами рамками. Установлюйте, виконуйте передбачення та експорт
  контрольної точки pico під ліцензією Apache-2.0.
lead: >-
  PicoSAM3 є компактною CNN, дистильованою із SAM 2.1 і SAM 3 та створеною для
  сегментації областей інтересу за запитами рамками на сенсорах на кшталт Sony
  IMX500. LibreYOLO підтримує її через окрему фабрику LibreSAM, відмінну від
  фабрики детекторів LibreYOLO(), лише із запитами рамками.
keywords:
  - PicoSAM3
  - Segment Anything
  - сегментація на edge-пристроях
  - область інтересу
  - запит рамкою
  - інференс у сенсорі
  - IMX500
  - дистиляція знань
last_verified: 1.5.0
snippets:
  predict:
    - label: Запит рамкою
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # PicoSAM3 має один розмір "pico", тому інші псевдоніми не потрібні.

        model = LibreSAM("picosam3")


        # bboxes= є єдиним підтримуваним запитом: [x1, y1, x2, y2] або список

        # рамок, по одній масці на рамку. Кожна рамка розширюється на 10%,

        # стає квадратною, обрізається до зображення й масштабується до 96x96
        перед CNN.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])

        print(result.masks.xy)      # Полігон для кожної маски

        print(result.boxes.xyxy)    # Щільна рамка, отримана з маски
    - label: 'Одне кодування, багато запитів'
      language: python
      code: |
        from libreyolo import LibrePicoSAM3, SAMPLE_IMAGE

        model = LibrePicoSAM3()

        # set_image() кешує початкове зображення; PicoSAM3 запускає один повний
        # прямий прохід CNN на рамку, тому це заощаджує завантаження/декодування
        # зображення, а не прохід енкодера, як в інших сімействах SAM.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(bboxes=[300, 200, 900, 700])
        b = model.predict(bboxes=[100, 100, 400, 400])
        model.reset_image()
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibrePicoSAM3

        model = LibrePicoSAM3()
        model.export(format="onnx", output_path="LibrePicoSAM3pico.onnx")

        # opset (типово 13) і dynamic (типово True, лише вісь батча) є
        # єдиними аргументами експорту, які приймає це сімейство.
    - label: Використання експортованого файлу
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # PicoSAM3 експортує необроблену CNN ROI 96x96: roi_image ->
        mask_logits.

        # Тут немає попередньої/подальшої обробки LibreYOLO для повторного
        використання,

        # оскільки export() не маршрутизується назад через LibreYOLO(), як
        контрольна

        # точка детектора.

        session = ort.InferenceSession("LibrePicoSAM3pico.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 96, 96),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 5d60ff14fe61ba29
---

## Встановлення

PicoSAM3 потребує додаткового набору залежностей `sam`: власне завантаження
ваг LibreYOLO і далі використовує інструменти Hugging Face із
`transformers`, хоча інференс працює в нативній CNN без `transformers`.

```bash
pip install "libreyolo[sam]"
```

## Передбачення

`LibreSAM(...)` (або фабрика конкретного сімейства `LibrePicoSAM3(...)`)
є окремою точкою входу від `LibreYOLO(...)`: вона повертає сегментатор за
запитами, а не детектор, оскільки прямий прохід тут не має сенсу без запиту.
Для цього сімейства немає команди CLI `libreyolo predict`; використовуйте
API Python.

<code-tabs name="predict" />

PicoSAM3 приймає лише `bboxes=`; передавання `points=`, `labels=`,
`masks=`, `text=`, `multimask=True` або пропуск рамки для сегментації
всього зображення спричиняє зрозумілий `ValueError`, оскільки цих режимів
немає в початковій моделі. `conf` фільтрує за передбаченою якістю маски
(IoU), а не за впевненістю виявлення, і має бути між `0.0` та `1.0`.
Кожна маска має ідентифікатор класу `0` із назвою `"object"`. `train()`,
`val()` і `track()` спричиняють `NotImplementedError`; використовуйте
LibreSAM2 або LibreSAM3 для запитів точками, текстом, масками чи для
сегментації всього. Типи джерел описано в розділі
[передбачення](/docs/predict).

## Варіанти

Доступний один розмір pico із фіксованим входом ROI 96 px: PicoSAM3 виконує
один повний прямий прохід CNN на рамку замість одноразового кодування всього
зображення.

## Експорт

<export-matrix />

PicoSAM3 є єдиним сімейством рівня SAM, яке експортується: воно записує
необроблену CNN ROI 96x96 до ONNX, `roi_image -> mask_logits`, без вбудованих
NMS чи подальшої обробки масок. Інші сімейства SAM спричиняють
`NotImplementedError` для `export()`, оскільки їхній поділ
енкодера/декодера ще не має визначеного контракту експорту середовища
виконання. Експортований граф PicoSAM3 не завантажується назад через
`LibreYOLO()`; запускайте його безпосередньо в середовищі виконання на
кшталт `onnxruntime`, застосовуючи ту саму попередню обробку квадратного ROI
з доповненням 10%, показану вище.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box>

PicoSAM3 дистильовано із SAM 2.1 і SAM 3 як учительських моделей. LibreYOLO не
включає й не розповсюджує код або ваги жодного вчителя в цьому сімействі;
постачаються лише компактна CNN-учень та її перетворена контрольна точка.

</provenance-box>

## Цитування

<citation-block />
