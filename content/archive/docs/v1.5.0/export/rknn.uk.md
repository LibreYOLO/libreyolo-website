---
title: RKNN
seo_title: Експорт до RKNN для NPU Rockchip
description: >-
  Скомпілюйте детектор LibreYOLO в артефакт .rknn для Rockchip: SDK виробника
  для самостійного встановлення, чотири валідовані варіанти RK3588 і паритет у
  симуляторі.
lead: >-
  RKNN є скомпільованим форматом для NPU Rockchip. LibreYOLO експортує проміжний
  файл ONNX з opset 19, компілює його за допомогою SDK RKNN Toolkit2 і може
  порівняти скомпільований граф з ONNX Runtime у симуляторі Toolkit2 на хості
  без плати.
keywords:
  - експорт yolo rknn
  - rockchip npu
  - rk3588
  - rknn-toolkit2
  - паритет симулятора rknn
  - інференс orange pi rockchip
last_verified: 1.5.0
meta:
  - label: Прапорець
    value: 'export(format="rknn", name="rk3588")'
    mono: true
  - label: Створює
    value: >-
      Один файл .rknn, супровідний файл .rknn.metadata.json і звіт
      .rknn.parity.json із verify=True
  - label: Додатково
    value: >-
      Нічого в PyPI. rknn-toolkit2 є SDK виробника, який потрібно встановити
      самостійно.
  - label: Повторне завантаження
    value: >-
      Не через LibreYOLO. Артефакт працює на платі із середовищем виконання
      Rockchip.
  - label: Форми
    value: >-
      Фіксовані квадратні, батч 1, opset 19. Усі три умови примусово
      застосовуються.
  - label: Точність
    value: Збірка з рухомою комою від виробника. half=True та int8=True відхиляються.
  - label: Охоплення
    value: >-
      Чотири варіанти виявлення на RK3588: YOLO9-t, YOLO9-E2E-t, PicoDet-s і
      YOLO-NAS-s
verification: >-
  Перевірено за файлами libreyolo/export/rknn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py і docs/rknn.md у гілці dev. Виміряні значення
  паритету взято із запису валідації від 2026-08-04 у docs/rknn.md.
snippets:
  install:
    - label: На боці LibreYOLO
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 'SDK виробника, який установлюєте ви'
      language: bash
      code: >
        # rknn-toolkit2 є SDK Rockchip з окремою ліцензією. LibreYOLO

        # не постачає й не встановлює його. Лише x86_64 Linux; у Windows

        # використовуйте WSL2 або контейнер Linux.

        #

        # Toolkit2 2.3.2 потребує setuptools<81 і не працює з ONNX 1.19 або
        новішою,

        # оскільки його компілятор досі імпортує вилучений модуль onnx.mapping.

        pip install "setuptools==80.9.0" "onnx==1.18.0"


        # Потім установіть відповідний пакунок rknn-toolkit2 із власного

        # репозиторію Rockchip і перевірте імпорт:

        python -c "import rknn.api; print('rknn-toolkit2 ready')"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # Створює weights/LibreYOLO9t.rknn і
        weights/LibreYOLO9t.rknn.metadata.json

        path = model.export(format="rknn", name="rk3588", imgsz=640,
        verify=True)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format rknn --name rk3588 \
          --imgsz 640 --verify
    - label: Аргументи
      language: python
      code: |
        model.export(
            format="rknn",
            name="rk3588",     # цільова платформа; target= і target_platform= також працюють
            imgsz=640,         # має відповідати записаному полотну варіанта
            batch=1,           # будь-яке інше значення спричиняє NotImplementedError
            dynamic=False,     # True спричиняє ValueError
            opset=19,          # будь-яке інше значення спричиняє NotImplementedError
            verify=False,      # True запускає симулятор на ПК і перевіряє пороги паритету
        )
  parity:
    - label: Паритет на основі наявного артефакту ONNX без плати
      language: python
      code: |
        import numpy as np
        from libreyolo.export import verify_rknn_simulator_parity

        input_tensor = np.random.default_rng(0).standard_normal(
            (1, 3, 640, 640), dtype=np.float32
        )
        metrics = verify_rknn_simulator_parity(
            "weights/LibreYOLO9t.onnx",
            input_tensor,
            target_platform="rk3588",
            rtol=1e-3,
            atol=1e-4,
            raise_on_failure=False,
        )
        print(metrics)
  support:
    - label: Перевірити сімейство й завдання перед компіляцією
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: c659713cc3c8cc9e
---

## Встановлення

Для компіляції потрібен RKNN Toolkit2 від Rockchip, який поширюється як SDK
виробника за власною ліцензією Rockchip і не є залежністю LibreYOLO. Додаткового
набору `libreyolo[rknn]` немає, і для цього формату не існує однорядкової команди
встановлення всього потрібного.

<code-tabs name="install" />

Плата не потрібна для компіляції або перевірки числового паритету. Плата RK3588
потрібна для вимірювання затримки, потужності й теплових характеристик, але таких
вимірювань не записано.

## Експорт

<code-tabs name="export" />

Перед компіляцією запит перевіряється за переліком точних варіантів моделей,
також перевіряється полотно. Якщо передати `imgsz`, відмінний від записаного для
варіанта, виникне помилка замість непомітної компіляції неперевіреної комбінації.
LibreYOLO записує проміжний файл ONNX з opset 19, компілює його, за бажанням
симулює, а потім видаляє проміжний файл.

Метадані зберігаються в супровідному файлі `<model>.rknn.metadata.json`, оскільки
формат RKNN не має переносного поля метаданих.

`verify=True` запускає симулятор Toolkit2 на ПК у тому самому сеансі, де
скомпільовано артефакт, порівнює кожен вихід з ONNX Runtime на тому самому вході
та записує `<model>.rknn.parity.json` із метриками помилок для кожного виходу.
Порогами є косинусна подібність щонайменше 0.9999 і нормалізована RMSE не більше
0.02. Їх застосовано до кожного виходу, який ще не є поелементно близьким. Збірка
виробника з рухомою комою знижує внутрішні тензори до половинної точності, тому
суворе `allclose` не виконується, навіть коли декодовані рамки стабільні. У разі
невдачі записується `<model>.rknn.failed.parity.json`, кандидат відкидається, а
попередній успішний експорт за цим шляхом залишається без змін.

Щоб порівняти вже наявний артефакт ONNX без повторного експорту:

<code-tabs name="parity" />

Симулятор Toolkit2 запускає граф у пам'яті, створений за допомогою `load_onnx` і
`build`. Він не може повторно завантажити цільовий файл `.rknn` без плати, тому
`verify=True` виконує компіляцію, експорт і симуляцію за один сеанс.

## Запуск артефакту

У `libreyolo/backends` немає запису RKNN, тому `LibreYOLO()` не завантажує файл
`.rknn`. Скомпільований артефакт розгортається на платі й виконується у власному
середовищі Rockchip. Попередня обробка, декодування, NMS і масштабування координат
там покладаються на застосунок.

`<model>.rknn.metadata.json` містить назви класів, розмір входу, завдання й
цільову платформу, тобто все потрібне застосунку для відтворення подальшої
обробки LibreYOLO. Постачайте його разом зі скомпільованою моделлю.

Для перевірки на хості без плати збережіть артефакт ONNX із такою самою
фіксованою формою й порівняйте його в симуляторі, як показано вище.

## Обмеження

Компілюються чотири комбінації, визначені варіантами моделей, а не сімействами:

| Варіант | Завдання | Полотно | Ціль |
|---|---|---:|---|
| YOLO9-t | detect | 640 | RK3588 |
| YOLO9-E2E-t | detect | 640 | RK3588 |
| PicoDet-s | detect | 320 | RK3588 |
| YOLO-NAS-s | detect | 640 | RK3588 |

Усе інше відхиляється перед компіляцією з повідомленням, що RKNN у цій версії
обмежено точними варіантами виявлення, перевіреними в симуляторі. Результати
лише компіляції для інших моделей існують, але навмисно не подаються як
підтримувані. У тому самому вимірюванні в RF-DETR залишилися два незнижені вузли
декодера `GridSample`, а D-FINE, RT-DETR, RT-DETRv2, RT-DETRv4, DEIM, DEIMv2 і EC
скомпілювалися та симулювалися з істотно неправильними декодованими виходами.

Батч 1, статичні форми, opset 19. `half=True` відхиляється, оскільки RKNN не
надає контракт `half` бібліотеки LibreYOLO, а `int8=True` відхиляється, доки не
з'являться результати для репрезентативного калібрування й правильності завдань.

Інші цільові платформи Rockchip відхиляються: `rk3588` є єдиною валідованою
платформою.

Повну таблицю сімейств і завдань наведено в
[матриці експорту](/docs/reference/export-matrix). Для однієї комбінації:

<code-tabs name="support" />
