---
title: HRNet
families:
  - hrnet
seo_title: 'HRNet: нисхідне оцінювання пози в LibreYOLO'
description: >-
  Використовуйте HRNet у LibreYOLO для нисхідного оцінювання пози COCO-17.
  Установлюйте, передбачайте, валідуйте й експортуйте контрольні точки W32 і W48
  за ліцензією MIT.
lead: >-
  HRNet, це згорткова мережа, яка зберігає потік ознак високої роздільної
  здатності через повторюване багатомасштабне злиття замість відновлення
  роздільної здатності після зменшення. LibreYOLO обгортає офіційний нисхідний
  варіант оцінювання пози для інференсу й валідації.
keywords:
  - HRNet
  - оцінювання пози людини
  - нисхідне оцінювання пози
  - ключові точки COCO-17
  - мережа високої роздільної здатності
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Джерело людей не задано: HRNet автоматично поєднується з легким
        # детектором LibreYOLO9t і один раз реєструє цей вибір.
        model = LibreYOLO("LibreHRNetw32-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreHRNetw32-pose.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Джерело людей
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreHRNetw32-pose.pt")

        # Повністю пропустіть виявлення: вважайте все зображення однією людиною.
        result = model(SAMPLE_IMAGE, cropped=True)

        # Або передайте HRNet рамки з уже запущеного детектора.
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        # Або поєднайте її з конкретним детектором LibreYOLO замість
        # стандартного LibreYOLO9t.
        result = model(SAMPLE_IMAGE, person_detector="rfdetr")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreHRNetw32-pose.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreHRNetw32-pose.pt format=onnx
    - label: Використання експортованого файла
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # Експортований граф містить лише голову теплової карти з фіксованим
        полотном:

        # він приймає пакет уже обрізаних і нормалізованих фрагментів людей

        # та повертає необроблені теплові карти. Виявлення людей, геометрія

        # обрізання, декодування теплової карти й пригнічення OKS не входять до
        графа;

        # поза LibreYOLO цей крок декодування потрібно реалізувати самостійно.

        session = ort.InferenceSession("LibreHRNetw32-pose.onnx")

        name = session.get_inputs()[0].name

        heatmaps = session.run(
            None, {name: np.zeros((1, 3, 256, 192), dtype=np.float32)}
        )[0]
source_hash: 5a5540fd54ee6f23
---

## Встановлення

HRNet не потребує нічого понад базовий пакет.

\`\`\`bash
pip install libreyolo
\`\`\`

Її стандартний детектор людей, легка контрольна точка LibreYOLO9t,
завантажується автоматично, коли HRNet уперше поєднується з ним.

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

HRNet є нисхідним оцінювачем пози: до запуску голови пози їй потрібна рамка
людини, тому кожен виклик визначає її джерело. Якщо нічого не вказано, під
час першого виклику вона поєднується з детектором LibreYOLO9t і реєструє цей
вибір. \`cropped=True\` пропускає виявлення та вважає все зображення однією
людиною; \`person_boxes\` приймає рамки з уже запущеного детектора;
\`person_detector\` приймає \`"auto"\`, \`"rfdetr"\`, будь-яку модель
виявлення LibreYOLO або звичайну викличну сутність. \`flip_test=True\` також
запускає модель на горизонтально віддзеркаленому фрагменті й усереднює дві
теплові карти, що є власною аугментацією HRNet під час тестування; загальний
\`augment=True\` тут не визначено. Джерела з кількома зображеннями
обробляються послідовно: детектор HRNet і змінна кількість людей на
зображення не підтримують пакетне передбачення. Джерела, потокову обробку й
роботу з результатами описано в розділі [передбачення](/docs/predict).

## Варіанти

Два розміри, \`w32\` і \`w48\`, обидва передбачають стандартний набір
ключових точок COCO-17 із фрагмента людини фіксованої роздільної здатності;
\`w48\` має ширший бекбон.

Зоопарк моделей upstream повідомляє точність пози кожного розміру з власним
детектором людей, власним налаштуванням перевірки з віддзеркаленням та
офіційним протоколом оцінювання COCO. Стандартне поєднання LibreYOLO
використовує інший детектор, тому валідація тут вимірює саме це поєднання,
а не upstream. Для відтворення показників upstream потрібні ті самі рамки
людей, оцінки детектора й налаштування віддзеркалення, що в початковому
оцінюванні.

## Валідація

\`val()\` запускає OKS-AP ключових точок у стилі COCO та приймає
\`data.yaml\` YOLO-pose або JSON ключових точок COCO разом із каталогом
зображень. Стандартний бекенд метрик, faster-coco-eval; якщо його не
встановлено, автоматично використовується \`pycocotools\`, а
\`faster_coco_eval=False\` примусово вибирає шлях \`pycocotools\`.

<code-tabs name="val" />

Валідація внутрішньо викликає власний \`predict()\` HRNet, тому використовує
детектор людей, з яким модель було створено або викликано. Створіть модель
з явним \`person_detector=\`, щоб зафіксувати це джерело між запусками й
не визначати стандартний варіант заново в кожному виклику.

## Експорт

<export-matrix />

Контракт експорту HRNet охоплює лише ONNX, TorchScript, OpenVINO і TensorRT;
будь-який інший формат спричиняє помилку до початку трасування. Кожен експорт
містить лише голову теплової карти з фіксованим полотном, пакетом 1 і FP32:
вона приймає фрагмент людини та повертає необроблені теплові карти. Геометрія
афінного обрізання перед нею, декодування теплових карт, відновлення
віддзеркалення та пригнічення OKS після неї залишаються в Python, тому для
повного конвеєра від зображення до ключових точок на іншому кінці й далі
потрібна LibreYOLO.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг для цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />

