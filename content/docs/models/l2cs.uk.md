---
title: L2CS-Net
families:
  - l2cs
seo_title: 'L2CS-Net: оцінювання погляду в LibreYOLO'
description: >-
  Використовуйте L2CS-Net у LibreYOLO для двостадійного оцінювання тангажу й
  рискання погляду. Установлюйте, виконуйте передбачення та експорт; контрольна
  точка Gaze360 призначена лише для досліджень.
lead: >-
  L2CS-Net є двостадійним оцінювачем погляду: детектор облич знаходить обличчя,
  а основа ResNet із двома класифікаційними головами кутових інтервалів
  передбачає тангаж і рискання для кожного обличчя. LibreYOLO обгортає її лише
  для інференсу.
keywords:
  - L2CS-Net
  - оцінювання погляду
  - відстеження очей
  - pitch yaw погляду
  - Gaze360
  - детекція облич
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Якщо face_detector не задано, використовується комплектний детектор
        # облич OpenCV (Haar в OpenCV 4, YuNet в OpenCV 5), тому для запуску
        # не потрібне додаткове завантаження понад саму контрольну точку L2CS.
        model = LibreYOLO("LibreL2CSr50.pt")
        result = model(SAMPLE_IMAGE)

        print(result.gaze.pitch, result.gaze.yaw)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreL2CSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Джерело облич
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # Передайте L2CS рамки з уже запущеного детектора.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # Або вкажіть конкретний комплектний детектор облич.
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
    - label: Використання експортованого файлу
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # Експортований граф містить лише основу ResNet і дві голови кутових
        інтервалів:

        # він приймає попередньо оброблений кроп обличчя 448x448 і повертає
        необроблені

        # (yaw_logits, pitch_logits), а не декодовані кути. Softmax, математичне
        сподівання

        # інтервалів і перетворення на градуси залишаються в Python; див.

        # libreyolo.models.l2cs.utils.bin_logits_to_angles.

        session = ort.InferenceSession("LibreL2CSr50.onnx")

        name = session.get_inputs()[0].name

        yaw_logits, pitch_logits = session.run(
            None, {name: np.zeros((1, 3, 448, 448), dtype=np.float32)}
        )
source_hash: 4ec43f4673b4be3e
---

## Встановлення

L2CS-Net не потребує додаткових залежностей для створення, передбачення чи
експорту моделі, контрольна точка якої вже є у вас.

```bash
pip install libreyolo
```

Єдина контрольна точка, яку LibreYOLO може отримати автоматично, модель
ResNet-50, навчену на Gaze360, завантажується через `gdown`, а не звичайне
HTTP-дзеркало, оскільки вона розміщена на Google Drive автора, а не в
організації LibreYOLO. Для цього шляху потрібен додатковий набір залежностей
`gaze`:

```bash
pip install "libreyolo[gaze]"
```

Без нього LibreYOLO показує інструкції з ручного завантаження замість
непомітної помилки.

## Передбачення

<code-tabs name="predict" />

L2CS-Net є двостадійним оцінювачем: спочатку запускається детектор облич, а
голова погляду зчитує тангаж і рискання з кожного поверненого кропа обличчя.
Без додаткових налаштувань передбачення використовує комплектний детектор
OpenCV, тому після отримання самої контрольної точки L2CS достатньо звичайного
виклику без додаткового завантаження. `face_boxes` приймає рамки з уже
запущеного детектора; `face_detector` приймає `"auto"`, `"haar"`,
`"yunet"`, модель виявлення LibreYOLO або звичайний викличний об'єкт.
`result.gaze` містить тангаж і рискання в радіанах, рядок за рядком узгоджені
з `result.boxes`, рамками виявлених облич. Типи джерел, потокове
передбачення та обробку результатів описано в розділі
[передбачення](/docs/predict).

## Варіанти

П'ять глибин бекбона мають однакову вхідну роздільну здатність і приймають ті
самі аргументи. На Gaze360, датасеті єдиної опублікованої контрольної точки,
навчено ResNet-50; інші чотири глибини підтримуються архітектурно, але не
мають опублікованих ваг.

## Експорт

<export-matrix />

<code-tabs name="export" />

## Ліцензування

<provenance-box>

LibreYOLO не розміщує й не дзеркалює жодної контрольної точки L2CS: на відміну
від більшості інших сімейств цього сайту, для цього сімейства нічого немає в
організації LibreYOLO на Hugging Face. Єдина контрольна точка, яку бібліотека
може отримати автоматично, надходить безпосередньо з власного розповсюдження
автора на Google Drive. Перед початком передавання показується повідомлення
про ліцензію Gaze360; це не копія, «повторно опублікована на
huggingface.co/LibreYOLO», як помилково натякає резюме вище.

</provenance-box>
