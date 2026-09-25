---
title: "RT-DETR без Ultralytics: v1, v2 і v4 з Apache-2.0"
description: "Запускайте й донавчайте RT-DETR, RT-DETRv2 і RT-DETRv4 з вагами Apache-2.0 у бібліотеці MIT: один API для predict, train, val та export."
date: 2026-09-24
author: Xuban
tags: [LibreYOLO, rt-detr, rtdetr, rt-detrv4, object-detection, license, tutorial]
faq:
  - q: "Чи можна використовувати RT-DETR безкоштовно для комерційних цілей?"
    a: "Так, якщо використовувати реалізацію з дозвільною ліцензією. Оригінальний код RT-DETR і RT-DETRv2 (lyuwenyu/RT-DETR), а також RT-DETRv4 (RT-DETRs/RT-DETRv4) поширюються за Apache-2.0, а LibreYOLO запускає всі три версії з вагами Apache-2.0 у бібліотеці MIT. Реалізація RT-DETR у пакеті ultralytics поширюється за AGPL-3.0; альтернатива для використання із закритим кодом це платна Enterprise License. Це загальна інформація, а не юридична консультація."
  - q: "За якою ліцензією поширюється RT-DETRv4?"
    a: "Apache-2.0. Файл LICENSE у репозиторії github.com/RT-DETRs/RT-DETRv4 містить ліцензію Apache-2.0. Під час навчання RT-DETRv4 використовує модель-учителя DINOv3, але випущені ваги моделі-учня не містять параметрів DINOv3, тому ліцензія DINOv3 від Meta на них не поширюється."
  - q: "Чи можна донавчати RT-DETR без Ultralytics?"
    a: "Так. LibreYOLO донавчає контрольні точки виявлення RT-DETR, RT-DETRv2 і RT-DETRv4 за допомогою model.train() або команди libreyolo train у командному рядку, починаючи з опублікованих ваг COCO. Моделі орієнтованих рамок для RT-DETRv2 підтримують лише інференс."
  - q: "Які ваги RT-DETR постачає LibreYOLO?"
    a: "RT-DETR у варіантах r18, r34, r50, r50m, r101, l і x; RT-DETRv2 у варіантах r18, r34, r50, r50m і r101; RT-DETRv4 у варіантах s, m, l і x, усі для виявлення об'єктів COCO на 640 px. Для RT-DETRv2 також доступні моделі орієнтованих рамок n, s, m, l і x, навчені на DOTA v1.0 на 1024 px. Кожен файл поширюється за Apache-2.0."
---

RT-DETR є трансформером для виявлення об'єктів у реальному часі від Baidu. Він декодує фіксований набір запитів замість щільної сітки, тож не потрібно налаштовувати етап NMS. Якщо пошукати RT-DETR, однією з перших знайдених реалізацій буде та, що входить до пакета Python `ultralytics`. Тут виникає питання ліцензії, якого сама модель ніколи не породжувала.

## Ліцензія RT-DETR залежить від репозиторію

Ліцензія стосується репозиторію, а не моделі. Репозиторіїв RT-DETR кілька, і їхні ліцензії різняться:

| Репозиторій | Що охоплює | Ліцензія |
|:---|:---|:---|
| [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | RT-DETR і RT-DETRv2, PyTorch і Paddle | Apache-2.0 |
| [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | RT-DETRv4 | Apache-2.0 |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | RT-DETR у пакеті `ultralytics` | AGPL-3.0 або Enterprise License |
| [LibreYOLO](https://github.com/LibreYOLO/libreyolo) | RT-DETR, RT-DETRv2 і RT-DETRv4 | код MIT, ваги Apache-2.0 |

Реалізація Ultralytics якісна. На [сторінці RT-DETR](https://docs.ultralytics.com/models/rtdetr/) наведено попередньо навчені `rtdetr-l.pt` і `rtdetr-x.pt`, а також підтримку навчання, валідації, інференсу й експорту. Вона входить до пакета з ліцензією AGPL-3.0, а Ultralytics продає Enterprise License командам, які не хочуть відкривати вихідний код усього проєкту. Якщо жоден варіант не підходить, архітектура доступна за Apache-2.0 від її авторів. У статті [пояснення ліцензій YOLO](/articles/yolo-licenses-explained) розглянуто той самий сценарій для сімейства YOLO, а [посібник із комерційної ліцензії](/articles/yolo-commercial-license) пояснює, що AGPL-3.0 означає для продукту із закритим кодом.

## Запуск RT-DETR з LibreYOLO

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreRTDETRr18.pt")  # downloads from Hugging Face on first use
result = model(SAMPLE_IMAGE, save=True)

for box in result.boxes:
    print(box.cls, box.conf, box.xyxy)
```

Той самий виклик працює з командного рядка:

```bash
libreyolo predict model=LibreRTDETRr18.pt source=image.jpg save=True
```

`conf` і `max_det` фільтрують декодування top-k за запитами й класами. Параметр `iou` приймається, але не використовується, оскільки NMS немає. Повертається об'єкт `Results`, який використовують усі моделі LibreYOLO, тому для заміни детектора достатньо змінити один рядок.

## RT-DETR, RT-DETRv2 і RT-DETRv4 в одному завантажувачі

Версію вказано в імені файлу, а `LibreYOLO()` вибирає її за контрольної точкою, тож усі три версії завантажуються однаково:

* **RT-DETR:** `LibreRTDETR` у варіантах r18, r34, r50, r50m, r101 (бекбони ResNet) та l, x (HGNetv2).
* **RT-DETRv2:** `LibreRTDETRv2` у варіантах r18, r34, r50, r50m і r101. Архітектура версії 1 збережена, змінено спосіб вибірки деформівною увагою.
* **RT-DETRv4:** `LibreRTDETRv4` у варіантах s, m, l і x. Тут повторно використано архітектуру D-FINE, а ваги отримано дистиляцією моделі-учителя DINOv3 у модель-учня HGNetv2.

Усі ваги для виявлення об'єктів навчено на COCO, інференс виконується на 640 px. Для довідки: у файлах README вихідних репозиторіїв вказано 46.5 AP для RT-DETR-R18 і 53.0 AP для RT-DETR-L на COCO, а для RT-DETRv4-S показник сягає 49.8 AP, для RT-DETRv4-X 57.0 AP. На [сторінці документації RT-DETR](/docs/models/rt-detr) наведено власну таблицю бенчмарків LibreYOLO для кожної контрольної точки.

RT-DETRv2 також підтримує орієнтовані рамки. Контрольні точки від `LibreRTDETRv2n-obb.pt` до `LibreRTDETRv2x-obb.pt` включно є офіційними контрольними точками DOTA v1.0: 15 класів аерознімків на 1024 px, конвертовані у формат LibreYOLO:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv2n-obb.pt")
result = model("aerial.png", save=True)
print(result.obb.xywhr)  # (N, 5): cx, cy, w, h, radians
```

У розділі [виявлення з орієнтованими рамками](/docs/tasks/oriented-detection) описано формат міток і метрики.

## Донавчання RT-DETR на власних даних

Навчання починається з опублікованої контрольної точки:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRr18.pt")
model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
```

Для повноцінного запуску вкажіть у `data` YAML-файл власного датасету. Для кожної версії задано власне типове значення швидкості навчання, а у версіях 1 і 2 швидкість навчання бекбона дорівнює одній двадцятій від `lr0`, відповідно до оригінального рецепта. Для кількох GPU вкажіть у CLI `device=0,1`, а для донавчання адаптерами [LoRA](/docs/train/lora) встановіть `lora=True` і додатковий компонент `lora`. Моделі орієнтованих рамок для RT-DETRv2 підтримують лише інференс. Інформацію про датасети, аугментацію та логери наведено в розділі [навчання](/docs/train).

## Валідація й експорт

```python
metrics = model.val(data="coco128.yaml")
print(metrics["metrics/mAP50-95"])

path = model.export(format="onnx")  # needs: pip install "libreyolo[onnx]"
```

`val()` повертає звичайний словник. Серед доступних форматів експорту є ONNX, TorchScript, ExecuTorch, TensorRT і OpenVINO; у матриці експорту на [сторінці моделі](/docs/models/rt-detr) вказано, які з них перевірено для кожної задачі. Експортовані файли `.onnx` або `.engine` завантажуються через `LibreYOLO()` і повертають той самий об'єкт `Results`. Посібники для кожного формату дивіться в розділі [експорт](/docs/export).

## Коли варто використовувати оригінальні репозиторії

Якщо потрібно відтворити результат зі статті з точними конфігураціями авторів, потрібна версія для Paddle або ви хочете самостійно виконати дистиляцію моделі-учителя DINOv3 для RT-DETRv4, скористайтеся вихідними репозиторіями. LibreYOLO донавчає випущену модель-учня v4, але не запускає модель-учителя. Якщо ж ваш проєкт уже має ліцензію AGPL-3.0 або на нього поширюється Enterprise License Ultralytics, ліцензійних причин для переходу немає.

## Примітка щодо ліцензії

Код LibreYOLO поширюється за MIT. Кожен файл ваг RT-DETR має ліцензію Apache-2.0, а в кожному репозиторії ваг на Hugging Face є файл LICENSE і NOTICE. Під час повторного поширення ваг Apache-2.0 вимагає зберігати ці файли. Ваги, навчені на власних даних, належать вам. RT-DETRv4 використовує DINOv3 лише як модель-учителя під час навчання, а випущені моделі-учні не містять параметрів DINOv3. Це загальна інформація, а не юридична консультація. Перед випуском перевірте актуальні файли LICENSE.

## Спробуйте

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv4s.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO поширюється за ліцензією MIT, працює на Linux, Mac і Windows, а також на GPU, Apple Silicon і звичайному CPU без змін у коді.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Документація RT-DETR](/docs/models/rt-detr)
