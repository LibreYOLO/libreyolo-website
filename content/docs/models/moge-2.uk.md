---
title: MoGe-2
families:
  - moge2
seo_title: 'MoGe-2: передбачення, валідація та експорт нормалей поверхні'
description: >-
  Використовуйте MoGe-2 у LibreYOLO для щільного передбачення нормалей поверхні.
  Установлюйте, виконуйте передбачення, валідацію та експорт офіційних
  контрольних точок ViT-S, ViT-B і ViT-L.
lead: >-
  MoGe-2 є монокулярною геометричною моделлю з одним прямим проходом, яка
  передбачає щільне поле нормалей поверхні з одного зображення RGB. LibreYOLO
  підтримує її лише для оцінювання нормалей через офіційні контрольні точки
  ViT-S, ViT-B і ViT-L.
keywords:
  - MoGe-2
  - MoGe 2
  - оцінювання нормалей поверхні
  - монокулярна геометрія
  - карта нормалей
  - щільне передбачення
  - DINOv2
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE, save=True)

        normal = result.normal_map
        print(normal.array.shape)   # Одиничні вектори float32 (H, W, 3)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMoGe2s-normal.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])   # Градуси
        print(metrics["metrics/median_angular_error"])
        print(metrics["metrics/within_11_25"])          # Відсоток пікселів
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMoGe2s-normal.pt data=my-dataset.yaml imgsz=518
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
        model.export(format="tensorrt", imgsz=518, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreMoGe2s-normal.pt format=onnx imgsz=518

        libreyolo export model=LibreMoGe2s-normal.pt format=tensorrt imgsz=518
        half=True
    - label: Використання експортованого файлу
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.normal_map.array.shape)
source_hash: ddfacf6b7e9729f6
---

## Встановлення

MoGe-2 не потребує додаткових залежностей. Усе, що вона імпортує, входить до
базового встановлення.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються автоматично: LibreYOLO
отримує відповідний розмір безпосередньо з офіційних контрольних точок і кешує
його локально.

<code-tabs name="predict" />

MoGe-2 повертає щільне поле, а не набір виявлень, тому `result.boxes`
порожній, а `conf`, `iou` і `max_det` не впливають на результат.
`result.normal_map` містить результат: масив одиничних векторів
`(H, W, 3)` у системі координат камери OpenCV, де `+x` спрямовано
праворуч, `+y` вниз, `+z` углиб сцени, а поверхня, повернута до камери,
має значення `(0, 0, -1)`. Передбачення для списку зображень виконує один
прямий прохід на кожне зображення; це сімейство не має швидкого шляху зі
складеним батчем. Типи джерел, потокове передбачення та обробку результатів
описано в розділі [передбачення](/docs/predict).

## Варіанти

Три розміри енкодера постачаються як окремі контрольні точки: ViT-S, ViT-B і
ViT-L, усі з однаковою вхідною роздільною здатністю. Засоби бенчмарків
LibreYOLO не вимірювали це сімейство, тому опублікованих показників
правильності для їх порівняння немає. Вибирайте розмір відповідно до власного
бюджету обчислень.

## Валідація

`val()` вимірює кутову похибку на парному датасеті карт нормалей: зображення
поруч із 16-бітними PNG нормалей з однаковою основою назви й необов'язковою
маскою дійсності, щоб доповнені та недійсні пікселі ніколи не враховувалися.
Метод повертає середню й медіанну кутову похибку в градусах, а також відсоток
пікселів у межах 11.25, 22.5 і 30 градусів.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експорт нормалей використовує контракт середовища виконання з фіксованою
роздільною здатністю та розміром батча 1: `dynamic` і `batch` зі значенням,
відмінним від 1, відхиляються, а `imgsz` має ділитися на розмір патча
енкодера ViT, що LibreYOLO перевіряє до початку запуску. Експортований артефакт
знову завантажується через `LibreYOLO()` відповідно до суфікса файлу, тому
файл `.onnx` поводиться як контрольна точка й повертає той самий об'єкт
`Results`.

<code-tabs name="export" />

## Ліцензування

<provenance-box>

LibreYOLO не копіює ці контрольні точки до власної організації.
`LibreYOLO("LibreMoGe2s-normal.pt")` завантажує відповідний розмір
безпосередньо з офіційних репозиторіїв Hugging Face у зафіксованій редакції
та перед використанням перевіряє файл за записаною контрольною сумою SHA-256.

</provenance-box>

## Цитування

<citation-block />
