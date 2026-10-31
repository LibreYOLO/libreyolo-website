---
title: NAFNet
families:
  - nafnet
seo_title: 'NAFNet: шумоподавление, обучение и экспорт под MIT'
description: >-
  Используйте NAFNet в LibreYOLO для шумоподавления и восстановления
  изображений. Установка, предсказание, обучение, валидация и экспорт чекпойнта
  SIDD под лицензией MIT.
lead: >-
  NAFNet — свёрточная сеть для восстановления изображений, в которой из
  типичного блока UNet убраны нелинейные функции активации, а на их месте стоит
  поэлементное умножение. В LibreYOLO он поддерживается для одной задачи —
  восстановления, с опубликованным чекпойнтом для шумоподавления на реальных
  снимках, обученным на SIDD.
keywords:
  - NAFNet
  - восстановление изображений
  - убрать шум с фото python
  - шумоподавление изображений
  - устранение размытия на фото
  - denoising нейросеть
  - SIDD
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model("noisy.jpg", save=True)

        restored = result.restored
        print(restored.array.shape)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreNAFNetl-restore-sidd.pt source=noisy.jpg
        save=True
    - label: Сохранение восстановленного изображения
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model.predict("noisy.jpg")

        result.restored.save("denoised.png")
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 imgsz=256 batch=16 lr0=1e-3
    - label: Происхождение чекпойнта
      language: python
      code: |
        from libreyolo import LibreYOLO

        # degradation и dataset записываются в сохранённый чекпойнт; на то,
        # что именно обучается, они не влияют.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
    - label: Multi-GPU
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val() возвращает обычный dict, а не объект
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.export(format="onnx", imgsz=256)
        model.export(format="tensorrt", imgsz=256, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=onnx
        imgsz=256

        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=tensorrt
        imgsz=256 half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Фабрика выбирает загрузчик по суффиксу файла, поэтому
        # экспортированный артефакт загружается как любой чекпойнт и
        # возвращает тот же объект Results.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")
        result = model("noisy.jpg")

        result.restored.save("denoised.png")
source_hash: 9bae9f82bee741bf
---

## Установка

NAFNet не требует установки дополнительных extra-пакетов. Всё, что он
импортирует, входит в базовую установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Возвращаемый объект `Results` содержит для этого семейства одно поле,
`restored`, — плотное RGB-изображение HWC типа uint8 на исходном холсте;
перебирать здесь нечего, рамок нет. `save=True` записывает это восстановленное
изображение прямо на диск, а не рисует разметку поверх входного. `conf`, `iou`
и `max_det` принимаются ради совпадения сигнатуры с остальными семействами, но
ни на что не влияют: восстановление не порождает детекций, которые нужно было
бы фильтровать. Об источниках, стриминге и обработке результатов — в разделе
[предсказание](/docs/predict).

## Варианты

Эту архитектуру делят две ширины: `s` (ширина 32) и `l` (ширина 64), обе
построены вокруг обучающего патча 256 px. Предсказание и валидация идут в
родном разрешении изображения независимо от размера и дополняют его только до
кратности коэффициенту понижения разрешения сети. Сейчас опубликована только
ширина `l` — чекпойнт для шумоподавления на реальных снимках, обученный на
SIDD.

## Обучение

NAFNet дообучается на ваших собственных парах «испорченное изображение —
чистое»: YAML датасета указывает на папку `inputs/<split>/` с испорченными
изображениями и папку `targets/<split>/` с чистыми целевыми, которые
сопоставляются по имени файла без расширения. `degradation` и `dataset` —
необязательные строки, которые записываются в сохранённый чекпойнт как история
происхождения; в обучении они не участвуют.

<code-tabs name="train" />

Если ничего не менять, обучение идёт 100 эпох с AdamW при `lr0=1e-3`, размером
батча 16, вырезками 256 px и ранней остановкой после 50 эпох без роста PSNR.
Пути через LoRA у этого семейства нет: `lora=True` вызывает ошибку, а не
запускает обучение, потому что `NAFNetTrainer` не подключает дообучение с
адаптерами.

Во время обучения сеть работает с обычным глобальным усредняющим пулингом.
Оконный локальный пулинг NAFNet, который применяется только на инференсе
(Test-time Local Converter), отсоединяется перед первой эпохой и подключается
обратно после окончания обучения: обратное распространение через локальный
пулинг с фиксированным окном не соответствовало бы тому, как чекпойнт
используется на инференсе.

Про датасеты, аугментацию, обучение на нескольких GPU и логгеры —
[обучение](/docs/train).

## Валидация

`val()` возвращает словарь с `metrics/PSNR` и `metrics/SSIM`, посчитанными в
RGB по всему полезному холсту: SSIM использует гауссово окно 11x11 с сигмой
1.5, а `fitness` для выбора лучшего чекпойнта — это значение PSNR. `data`
указывает на тот же формат датасета из пар изображений, что и при обучении.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`, где выходное изображение лежит в `restored`. NAFNet
экспортируется с фиксированным пространственным разрешением: `imgsz` должен
делиться на коэффициент понижения разрешения сети (16 для обеих ширин
архитектуры), и при `dynamic=True` динамическим остаётся только измерение
батча — высота и ширина фиксируются на момент экспорта.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
