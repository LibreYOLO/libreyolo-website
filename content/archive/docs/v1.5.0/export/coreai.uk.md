---
title: Core AI
seo_title: Експорт до Apple Core AI з LibreYOLO
description: >-
  Експорт моделі LibreYOLO у ресурс .aimodel для Apple Core AI: лише macOS,
  фіксоване полотно, FP32 і контракт порядку іменованих виходів, якого мають
  дотримуватися споживачі.
lead: >-
  Core AI є стеком інференсу на пристрої від Apple. LibreYOLO захоплює модель
  через torch.export, знижує її конвертером Core AI і записує ресурс .aimodel з
  метаданими моделі та іменами експортованих виходів.
keywords:
  - експорт libreyolo у core ai
  - aimodel
  - coreai-torch
  - torch.export apple
  - інференс на пристрої apple
  - coreai_output_names
last_verified: 1.5.0
meta:
  - label: Прапорець
    value: export(format="coreai")
    mono: true
  - label: Записує
    value: Один ресурс .aimodel з доданими метаданими
  - label: Extra
    value: 'pip install "libreyolo[coreai]"'
    mono: true
  - label: Зворотне завантаження
    value: >-
      Не через LibreYOLO. Споживачі використовують середовище виконання Core AI
      напряму.
  - label: Форми
    value: Фіксоване полотно. dynamic=True викликає NotImplementedError.
  - label: Точність
    value: Лише FP32. half=True та int8=True відхиляються.
  - label: Потребує
    value: >-
      macOS. Набір інструментів ніде більше не конвертує і не запускає моделі, а
      coreai-torch закріплює torch на 2.11.x.
verification: >-
  Прочитано з libreyolo/export/coreai.py, libreyolo/export/coreai_compat.py,
  libreyolo/export/exporter.py, libreyolo/export/support.py і pyproject.toml у
  гілці dev.
snippets:
  install:
    - label: 'Встановлення, на macOS'
      language: bash
      code: |
        # Свідомо не входить до жодного зведеного extra: coreai-torch закріплює
        # torch на 2.11.x і перетягнув би все середовище на цю версію.
        pip install "libreyolo[coreai]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Записує weights/LibreYOLO9t.aimodel
        path = model.export(format="coreai", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreai --imgsz 640
    - label: Аргументи
      language: python
      code: |
        model.export(
            format="coreai",
            imgsz=640,        # int або (висота, ширина); це полотно запуску
            batch=1,
            output_path=None, # None записує weights/<stem>.aimodel
        )

        # dynamic=True викликає NotImplementedError.
        # half=True та int8=True відхиляються під час валідації.
  outputs:
    - label: Прочитати порядок виходів перед підключенням споживача
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="coreai", imgsz=640)

        # Метадані ресурсу зберігають імена експортованих виходів у порядку
        # графа під ключем "coreai_output_names". Словник, який повертає
        # Core AI, відображайте за іменами з цього списку; ніколи не
        # зіставляйте його позиційно з кортежем eager-режиму.
  support:
    - label: Перевірити одне сімейство та задачу перед експортом
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: a35bfeafac6d6966
---

## Встановлення

Цей формат працює лише на macOS. Залежність `coreai-torch` має маркер
`sys_platform == 'darwin'`, а набір інструментів ніде більше не конвертує і не
запускає моделі.

<code-tabs name="install" />

Цей extra перебуває поза всіма зведеними extra, зокрема `libreyolo[all]`, бо
`coreai-torch` закріплює torch на серії 2.11. Встановлюйте його в середовище,
яке ви готові обмежити цією парою.

## Експорт

<code-tabs name="export" />

Захоплення виконує `torch.export`: це справжнє захоплення графа з перевірками
(guards), а не один записаний трейс. Такий шлях суворіший за Core ML: читання
скалярів на боці хоста та потік керування, залежний від даних, відхиляються,
замість того щоб мовчки запікатися в граф, тому кілька сімейств тут заблоковано
із записаною помилкою захоплення.

Три підготовчі кроки виконуються всередині області, яка відновлює живу модель
викликача незалежно від того, успішний експорт чи ні. У сімействах, похідних від
Darknet, батч-нормалізацію в режимі інференсу точно згортають у попередні
згортки, бо Core AI 0.4.1 не зберігає формулу Darknet з епсилоном після
квадратного кореня. У сіткових та якірних сімействах якорі заморожують під
фіксоване полотно. У RF-DETR позиційний ембединг перезапікають під запитане
полотно, повторно запускаючи власний шлях запікання моделі, бо конвертер не має
зниження для `aten._upsample_bicubic2d_aa`.

На етапі зниження еталонну декомпозицію PyTorch для `aten.grid_sampler_2d`
вносять у таблицю декомпозицій, бо конвертер Core AI не має зниження для
семплера deformable attention, який використовують сімейства DETR.

Ресурси оголошують мінімальну версію ОС v27, і це єдине значення, яке пропонує
набір інструментів. Це обмежує розгортання, а не конвертацію: конвертація та
виконання на боці Python працюють на давніших macOS завдяки середовищу
виконання всередині пакета wheel, але числові результати різняться між версіями
ОС, тому записаний паритет вимірюють на macOS 27.

## Запуск артефакту

У `libreyolo/backends` немає запису для Core AI, тому `LibreYOLO()` не
завантажує файл `.aimodel`. Споживачі використовують середовище виконання
Core AI напряму, а препроцесинг, декодування, NMS і перемасштабування координат
лишаються на їхньому боці. Валідований рядок у матриці підтримки стверджує, що
експортований граф обчислює ті самі числа, що й еталон, а не що його запустить
`predict`.

Єдине, чого споживач не може відтворити самостійно, це порядок виходів:

<code-tabs name="outputs" />

Core AI повертає іменований словник, порядок ключів у якому не збігається ні з
порядком кортежу прямого проходу в eager-режимі, ні з чимось передбачуваним.
Саме тому експортовані імена записуються в метадані ресурсу як
`coreai_output_names`. Зіставляйте за іменем.

## Обмеження

Фіксоване полотно, FP32, батч такий, яким його експортовано. `dynamic=True`
викликає `NotImplementedError`, а `half=True` та `int8=True` відхиляються під
час валідації.

Покриття з боку конвертації широке. Валідовані комбінації охоплюють виявлення в
сімействах YOLO9, YOLOX, YOLO7, чотирьох детекторах епохи Darknet, YOLO-NAS,
PicoDet, RTMDet, RT-DETR, RT-DETRv2, RT-DETRv4, D-FINE, DEIM, DEIMv2, EC і
RF-DETR; чотири сімейства класифікації на CNN, а також CLIP і SigLIP2 із
замороженими класами; Depth Anything V2 і ZipDepth; відновлення зображень
NAFNet і Real-ESRGAN; семантичну сегментацію PIDNet і LingBotVision; а також
виявлення точок FOMO. Кожна має власний записаний контекст, який виводить
`libreyolo formats`.

Заблоковані, з причиною, записаною для кожної комбінації:

| Комбінація | Причина |
|---|---|
| Семантична сегментація EoMT | Строге захоплення завершується помилкою `GuardOnDataDependentSymNode`: щось у шляху масок читає значення з тензора й розгалужується за ним |
| Семантична сегментація SegFormer | Шлях захоплення не оцінювали, а опубліковані ваги все одно некомерційні, незалежно від формату |
| Оцінювання погляду L2CS | Сама модель підтримує лише ONNX, TorchScript, ExecuTorch, TensorRT і OpenVINO, і це рішення на боці моделі |
| Оцінювання глибини Depth Anything 3 | Сімейство відхиляє експорт для всіх форматів |

З RF-DETR пов'язане одне застереження, яке варто прочитати перед порівнянням
артефактів. Паритет тут записано щодо графа, який готує сам експортер Core AI, а
не щодо ONNX, і на полотні 640 артефакт ONNX для RF-DETR розходиться з цим
підготовленим графом. Повторне запікання в Core AI зберігає масштабування зі
згладжуванням, яке виконує модель в eager-режимі, тоді як шлях ONNX згладжування
вимикає. Тому ONNX не є коректним еталоном для цього сімейства на полотні,
відмінному від нативного.

Про попередній формат Apple читайте в [Core ML](/docs/export/coreml). Повну
сітку сімейств і задач містить [матриця експорту](/docs/reference/export-matrix).
Для однієї комбінації:

<code-tabs name="support" />
