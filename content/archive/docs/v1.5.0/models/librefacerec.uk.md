---
title: LibreFaceRec
families:
  - facerec
seo_title: 'LibreFaceRec: розпізнавання та верифікація облич'
description: >-
  Використовуйте LibreFaceRec у LibreYOLO для виявлення облич, створення
  ембедингів і верифікації. Установлюйте та виконуйте передбачення; ваги
  ембедингів мають ліцензію Apache-2.0.
lead: >-
  LibreFaceRec є завданням LibreYOLO для створення ембедингів облич: детектор
  знаходить і вирівнює обличчя, а голова розпізнавання створює L2-нормалізований
  ембединг особи для верифікації або пошуку.
keywords:
  - LibreFaceRec
  - розпізнавання облич
  - ембединг обличчя
  - верифікація облич
  - ArcFace
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Назви librefacerec-* спрямовуються до цього сімейства незалежно від

        # суфікса файлу й під час першого використання завантажуються з

        # організації LibreYOLO на Hugging Face разом із типовим детектором
        облич.

        model = LibreYOLO("librefacerec-l.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.embeddings.data.shape)   # (N, D), L2-нормалізовано
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=face.jpg
    - label: Верифікація
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        # Порівнює найпомітніше обличчя на кожному зображенні за косинусною
        # подібністю їхніх L2-нормалізованих ембедингів.
        result = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)
        print(result["similarity"], result["same_person"])
    - label: Пошук у галереї
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("librefacerec-l.onnx")


        query = model("query.jpg").embeddings          # обличчя на цьому
        зображенні

        gallery = model.embed(["a.jpg", "b.jpg", "c.jpg"])   # (N_total, D)


        # Косинусні подібності (query_faces, N_total).

        scores = query.similarity(gallery)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")
        model.export(format="onnx")
source_hash: f1a345bb96e32f12
---

## Встановлення

Голова розпізнавання LibreFaceRec працює через `onnxruntime`, якого немає в
базовому встановленні.

```bash
pip install "libreyolo[onnx]"
```

## Передбачення

<code-tabs name="predict" />

За одним викликом приховано два окремі графи ONNX для виявлення та
розпізнавання: детектор знаходить і вирівнює кожне обличчя до канонічного
кропа, а голова розпізнавання повертає один L2-нормалізований ембединг на
обличчя. Без додаткових налаштувань `predict()` автоматично завантажує та
під'єднує комплектний типовий детектор. `face_detector` приймає викличний
об'єкт, модель виявлення LibreYOLO або екземпляр `FaceDetector`;
`face_boxes` повністю оминає виявлення й використовує надані рамки.
`result.embeddings` містить один рядок на кожне виявлене обличчя, узгоджений
із `result.boxes`. Метод `.similarity()` одним викликом обчислює косинусну
подібність до іншого ембедингу або цілої галереї. Щоб безпосередньо порівняти
два зображення замість двох уже обчислених ембедингів, виклик
`model.verify(image_a, image_b)` виконує виявлення та створення ембедингів
для обох, а потім порівнює їхні обличчя з найвищою впевненістю. Можна
підставити будь-яку іншу модель розпізнавання ONNX за домовленостями ArcFace
(вирівняний кроп на вході, ембединги `(N, D)` на виході), передавши шлях до
її файлу замість назви `librefacerec-*`. Типи джерел, потокове передбачення
та обробку результатів описано в розділі
[передбачення](/docs/predict).

## Експорт

<export-matrix />

LibreFaceRec уже обгортає попередньо експортований граф ONNX; повторний експорт
до іншого формату не реалізовано.

## Ліцензування

<provenance-box>

Комплектний типовий детектор облич є другим артефактом з іншою ліцензією:
YuNet з OpenCV Zoo, MIT, авторське право Shiqi Yu. Код архітектури не перенесено
з жодного з двох проєктів; обидва графи непрозоро використовуються через
`onnxruntime`, тому власна обгортка LibreYOLO не містить стороннього коду й
повністю ліцензована за MIT.

</provenance-box>

## Цитування

<citation-block />
