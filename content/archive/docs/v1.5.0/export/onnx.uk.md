---
title: ONNX
seo_title: Експорт у ONNX із LibreYOLO
description: >-
  Експорт моделі LibreYOLO у ONNX: opset, який LibreYOLO вибирає для кожного
  сімейства, динамічні осі, вбудований NMS, INT8 та повторне завантаження графа.
lead: >-
  ONNX є переносним форматом графа. LibreYOLO трасує модель за допомогою
  torch.onnx.export, за потреби спрощує граф і записує сімейство, завдання,
  назви класів та розмір вхідних даних у власні метадані файлу, щоб будь-який
  бекенд LibreYOLO міг відтворити подальшу обробку.
keywords:
  - експорт yolo onnx
  - onnxruntime
  - torch.onnx.export
  - onnx opset
  - динамічні осі onnx
  - вбудований nms onnx
  - onnx int8 qdq
  - onnx metadata_props
last_verified: 1.5.0
meta:
  - label: Параметр
    value: export(format="onnx")
    mono: true
  - label: Результат
    value: 'Один файл .onnx із метаданими, вбудованими у граф'
  - label: Додатково
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: Повторне завантаження
    value: LibreYOLO("weights/LibreYOLO9t.onnx")
    mono: true
  - label: Форми
    value: Типово динамічний батч у Python; винятки для завдань наведено нижче
  - label: Точність
    value: 'FP32, FP16 (half=True), INT8 (int8=True, виявлення YOLO9)'
verification: >-
  Перевірено за файлами libreyolo/export/onnx.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/onnx.py та
  libreyolo/cli/commands/export.py у гілці dev.
snippets:
  install:
    - label: Встановлення
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Записує weights/LibreYOLO9t.onnx
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx
    - label: Аргументи
      language: python
      code: |
        model.export(
            format="onnx",
            imgsz=640,        # ціле число або (висота, ширина)
            batch=1,
            dynamic=True,     # типове значення Python; у CLI типовим є False
            simplify=True,    # запуск onnxsim для графа
            opset=None,       # None вибирає 13 або 17 для сімейств у стилі DETR
            half=False,       # ваги та активації FP16
            int8=False,       # QDQ INT8, лише виявлення YOLO9
            data=None,        # калібрувальний data.yaml, лише INT8
            device=None,      # пристрій трасування; None використовує пристрій моделі
            output_path=None, # None записує weights/<stem>.onnx
        )
  nms:
    - label: Вбудувати NMS у граф
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Лише виявлення YOLO9, батч 1. dynamic примусово стає False.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            nms=True,
            conf=0.25,
            iou=0.45,
            max_det=300,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx --nms \
          --conf 0.25 --iou 0.45 --max-det 300
  int8:
    - label: INT8 із калібрувальними даними
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            int8=True,
            data="coco128.yaml",   # кількасот репрезентативних зображень
            fraction=1.0,
        )
  run:
    - label: Через LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.onnx")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Безпосередньо в ONNX Runtime
      language: python
      code: >
        import numpy as np

        import onnx

        import onnxruntime as ort


        session = ort.InferenceSession(
            "weights/LibreYOLO9t.onnx",
            providers=["CPUExecutionProvider"],
        )


        # На цьому шляху попередня та подальша обробка покладаються на вас.

        batch = np.zeros((1, 3, 640, 640), dtype=np.float32)

        outputs = session.run(None, {session.get_inputs()[0].name: batch})

        print([out.shape for out in outputs])


        # Граф містить сімейство, завдання, назви класів і розмір вхідних даних.

        meta = {p.key: p.value for p in
        onnx.load("weights/LibreYOLO9t.onnx").metadata_props}

        print(meta["model_family"], meta["task"], meta["imgsz"])
  support:
    - label: Перевірити сімейство й завдання перед експортом
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cee78250fc7189a3
---

## Встановлення

<code-tabs name="install" />

Додатковий пакет встановлює `onnx`, `onnxsim` та `onnxruntime`. Для запису файлу
достатньо лише `onnx`; `onnxsim` виконує прохід спрощення, а `onnxruntime` запускає
артефакт і виконує калібрування INT8.

## Експорт

<code-tabs name="export" />

Без `output_path` файл потрапляє до `weights/` із основою назви контрольної точки,
до якої додається `_fp16` або `_int8`, якщо запитано відповідну точність.

Типовим значенням `dynamic` є `True` у Python і `False` у CLI. Коли параметр
увімкнено, вісь батча стає символьною, а кілька завдань отримують додаткову
динамічність: для семантичної сегментації змінними стають також висота й ширина
маски, для відновлення Real-ESRGAN змінними стають просторові осі, а двостадійні
детектори зберігають динамічними початкові висоту й ширину, оскільки змінюють
розмір усередині графа.

Якщо `opset` не зазначено, його вибирають для кожного сімейства окремо. Сімейства
у стилі DETR (`detr`, `deformable_detr`, `dinodetr`, `dfine`, `deim`, `deimv2`,
`ec`, `lwdetr`, `rfdetr`, `rtdetr`, `rtdetrv2`, `rtdetrv4`), а також `deit`,
`midas` і `moge2` отримують opset 17, де знижується `aten::scaled_dot_product`.
Усі інші отримують 13. Для matting незалежно від налаштувань вибирається 19,
оскільки декодеру BiRefNet потрібен оператор `DeformConv`, визначений у ONNX
починаючи з opset 19.

Параметр `simplify=True` запускає `onnxsim` і в разі невдачі зберігає початковий
граф, тому помилка спрощення створює попередження, а не зупиняє експорт. У macOS
arm64 із `onnx` 1.22 або новішим та `onnxsim` 0.6.5 або старішим прохід повністю
пропускається, оскільки це поєднання може аварійно завершити процес Python.

### Вбудований NMS

<code-tabs name="nms" />

Параметр `nms=True` підтримується лише для виявлення YOLO9 і потребує батча 1;
якщо запитати його з `dynamic=True`, буде записано попередження й вимкнено
динамічність. Після цього граф має два виходи: `output` форми
`(batch, max_det, 6)` і `raw`, недекодований тензор детектора, який використовує
власний бекенд LibreYOLO, щоб подальша обробка залишалася ідентичною шляху PyTorch.

### DeepStream

Параметр `deepstream=True` доступний лише для ONNX. Він експортує граф у
компонуванні, якого очікує синтаксичний аналізатор NVIDIA DeepStream, і записує
поряд два допоміжні файли, `config_infer_primary_<stem>.txt` та
`<stem>_labels.txt`, щоб артефакт можна було додати до пайплайна без написаної
вручну конфігурації.

Він несумісний із `nms=True`, а запит обох параметрів спричиняє `ValueError`:
DeepStream виконує пригнічення на власному етапі кластеризації. Передавання цього
параметра до будь-якого формату, крім ONNX, також спричиняє помилку. Сітку
підтримуваних сімейств і завдань та збирання аналізатора наведено на сторінці
[DeepStream](/docs/export/deepstream).

### INT8

<code-tabs name="int8" />

Параметр `int8=True` запускає статичне квантування ONNX Runtime та записує граф
QDQ із вхідними й вихідними даними float32. Квантуються лише вузли `Conv` і `Gemm`.
Декодування голови детектора навмисно залишається у float32: ця конкатенація
поєднує координати рамок у масштабі пікселів з оцінками класів у діапазоні від
0 до 1, а єдиний масштаб активації для тензора, у якому переважає величина
координат рамки, звів би всі оцінки до нуля.

Наразі цей параметр застосовується лише до виявлення YOLO9, а для всього іншого
попередня перевірка спричиняє `NotImplementedError`. Якщо `data` не зазначено,
із попередженням використовується `coco8.yaml`; вісім зображень не є
репрезентативним калібрувальним набором. Модель, яку вже квантовано в PyTorch,
використовує інший шлях, описаний на сторінці
[Квантування](/docs/export/quantization).

## Запуск артефакту

<code-tabs name="run" />

`LibreYOLO()` виконує диспетчеризацію за суфіксом `.onnx` і повертає такий самий
об'єкт `Results`, як контрольна точка `.pt`, оскільки під час експорту назви
класів, завдання, розмір вхідних даних і схему пози записано у `metadata_props`
графа. З `device="auto"` сеанс використовує `CUDAExecutionProvider`, якщо про
нього повідомляє ONNX Runtime, а інакше переходить на CPU.

Другий фрагмент призначено для читачів без встановленої LibreYOLO. На цьому шляху
попередня обробка, декодування, NMS і повторне масштабування координат покладаються
на вас; блок метаданих усе одно доступний для читання.

## Обмеження

Назви вихідних тензорів фіксовані для кожного завдання, і споживач без метаданих
має зіставити саме їх:

| Завдання | Назви виходів |
|---|---|
| Виявлення, сіткові та якірні голови | `output` |
| Виявлення, стиль DETR | `pred_logits`, `pred_boxes` |
| Виявлення, RF-DETR | `dets`, `labels` |
| Класифікація | `output` |
| Семантична сегментація | `semantic_logits` |
| Глибина | `depth` |
| Нормалі поверхні | `normal` |
| Контури | `edges` |
| Відновлення | `restored` |
| Matting | `matte` |
| Погляд | `yaw_logits`, `pitch_logits` |

RF-DETR також є єдиним сімейством, вхідний тензор якого називається `input`,
а не `images`.

У цій версії кілька завдань мають контракт середовища виконання з фіксованою
роздільною здатністю. Завдання глибини, нормалей поверхні та контурів відхиляють
`batch != 1` і примусово встановлюють `dynamic=False`. Matting вимагає початкового
квадратного розміру 1024, оскільки таблиці відносних позицій Swin у BiRefNet
прив'язані до своєї роздільної здатності. Відновлення вимагає фіксованого полотна
для всіх сімейств, крім Real-ESRGAN, генератор якого є повністю згортковим.

Прямокутний `imgsz` працює для сімейств YOLO9, HRNet, NAFNet та Real-ESRGAN.
Сімейства з контрактом фіксованої квадратної форми (`clip`, `deformable_detr`,
`detr`, `dinodetr`, `dfine`, `deim`, `deimv2`, `ec`, `lwdetr`, `moge2`,
`rtdetr`, `rtdetrv2`, `rtdetrv4`, `rfdetr`, `siglip2`, `ssd`) одразу його
відхиляють.

Два поєднання відхиляються ще до трасування: сегментація YOLO9, оскільки YOLO9
підтримує в LibreYOLO лише виявлення, і сегментація RTMDet-Ins, декодування масок
із динамічним ядром якої не має контракту експортованого середовища виконання.

Повну сітку сімейств і завдань наведено в
[матриці експорту](/docs/reference/export-matrix). Для окремого поєднання можна
запитати безпосередньо бібліотеку:

<code-tabs name="support" />

