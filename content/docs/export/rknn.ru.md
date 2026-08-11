---
title: RKNN
seo_title: Экспорт в RKNN для NPU Rockchip
description: >-
  Компиляция детектора LibreYOLO в артефакт Rockchip .rknn: SDK производителя,
  который вы ставите сами, четыре проверенных варианта для RK3588 и паритет в
  симуляторе.
lead: >-
  RKNN — скомпилированный формат для NPU от Rockchip. LibreYOLO экспортирует
  промежуточный ONNX с opset 19, компилирует его через SDK RKNN Toolkit2 и может
  сравнить скомпилированный граф с ONNX Runtime в хост-симуляторе Toolkit2, без
  платы.
keywords:
  - экспорт yolo в rknn
  - rockchip npu
  - rk3588
  - rknn-toolkit2
  - паритет симулятора rknn
  - инференс на orange pi rockchip
last_verified: 1.5.0
meta:
  - label: Флаг
    value: 'export(format="rknn", name="rk3588")'
    mono: true
  - label: Записывает
    value: >-
      Один файл .rknn, сайдкар .rknn.metadata.json и отчёт .rknn.parity.json при
      verify=True
  - label: Дополнительно
    value: >-
      Ничего на PyPI. rknn-toolkit2 — это SDK производителя, который вы ставите
      сами.
  - label: Загружается обратно
    value: >-
      Не через LibreYOLO. Артефакт работает на плате в собственной среде
      выполнения Rockchip.
  - label: Формы
    value: 'Фиксированная квадратная, батч 1, opset 19. Все три ограничения обязательны.'
  - label: Точность
    value: >-
      Сборка производителя с плавающей точкой. half=True и int8=True
      отклоняются.
  - label: Охват
    value: >-
      Четыре варианта детекции на RK3588: YOLO9-t, YOLO9-E2E-t, PicoDet-s и
      YOLO-NAS-s
verification: >-
  Прочитано из libreyolo/export/rknn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py и docs/rknn.md в ветке dev. Измеренные показатели
  паритета взяты из записи о валидации от 2026-08-04 в docs/rknn.md.
snippets:
  install:
    - label: Со стороны LibreYOLO
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 'SDK производителя, который вы ставите сами'
      language: bash
      code: |
        # rknn-toolkit2 — это SDK от Rockchip под отдельной лицензией. LibreYOLO
        # не поставляет и не устанавливает его. Только Linux x86_64; на Windows
        # используйте WSL2 или Linux-контейнер.
        #
        # Toolkit2 2.3.2 требует setuptools<81 и ломается на ONNX 1.19 и новее,
        # где убрали onnx.mapping, который его компилятор всё ещё импортирует.
        pip install "setuptools==80.9.0" "onnx==1.18.0"

        # Затем установите подходящий wheel rknn-toolkit2 из собственного
        # репозитория wheel-пакетов Rockchip и убедитесь, что он импортируется:
        python -c "import rknn.api; print('rknn-toolkit2 ready')"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Записывает weights/LibreYOLO9t.rknn и weights/LibreYOLO9t.rknn.metadata.json
        path = model.export(format="rknn", name="rk3588", imgsz=640, verify=True)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format rknn --name rk3588 \
          --imgsz 640 --verify
    - label: Аргументы
      language: python
      code: |
        model.export(
            format="rknn",
            name="rk3588",     # целевая платформа; target= и target_platform= тоже работают
            imgsz=640,         # должен совпадать с записанным холстом варианта
            batch=1,           # любое другое значение вызывает NotImplementedError
            dynamic=False,     # True вызывает ValueError
            opset=19,          # любое другое значение вызывает NotImplementedError
            verify=False,      # True запускает симулятор на PC и блокирует экспорт при плохом паритете
        )
  parity:
    - label: Проверка паритета без платы на готовом ONNX-артефакте
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
    - label: Проверка одного семейства и задачи перед компиляцией
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: c659713cc3c8cc9e
---

## Установка

Для компиляции нужен RKNN Toolkit2 от Rockchip: он распространяется как SDK
производителя под собственной лицензией Rockchip и не входит в зависимости
LibreYOLO. Extra `libreyolo[rknn]` не существует, и ничего в этом формате не
ставится одной строкой.

<code-tabs name="install" />

Плата не нужна ни для компиляции, ни для проверки численного паритета. Плата
RK3588 нужна для измерений задержки, энергопотребления и нагрева, ни одно из
которых не зафиксировано.

## Экспорт

<code-tabs name="export" />

Запрос сверяется со списком точных вариантов моделей ещё до того, как что-либо
компилируется, и холст проверяется тоже: `imgsz`, отличный от того, с которым
записан вариант, вызывает ошибку, а не молча компилирует что-то непроверенное.
LibreYOLO записывает промежуточный ONNX с opset 19, компилирует его, при
необходимости прогоняет через симулятор и потом удаляет промежуточный файл.

Метаданные лежат в сайдкар-файле `<model>.rknn.metadata.json`, потому что в
формате RKNN нет переносимого поля для метаданных.

`verify=True` запускает симулятор Toolkit2 на PC в той же сессии, где был
скомпилирован артефакт, сравнивает каждый выход с ONNX Runtime на одном и том же
входе и записывает `<model>.rknn.parity.json` с метриками ошибки по каждому
выходу. Пороги — косинусная близость не ниже 0.9999 и нормализованный RMSE не
выше 0.02, и применяются они к любому выходу, который ещё не совпадает
поэлементно; сборка производителя с плавающей точкой понижает внутренние тензоры
до половинной точности, поэтому строгий `allclose` не выполняется даже тогда,
когда декодированные рамки стабильны. Неудачный прогон записывает
`<model>.rknn.failed.parity.json`, отбрасывает кандидата и оставляет нетронутым
любой более ранний успешный экспорт по этому пути.

Чтобы сравнить уже имеющийся ONNX-артефакт, не экспортируя заново:

<code-tabs name="parity" />

Симулятор Toolkit2 работает с графом в памяти, который создают `load_onnx` и
`build`. Загрузить обратно `.rknn`-файл под конкретную платформу без платы он не
может — поэтому `verify=True` выполняет компиляцию, экспорт и симуляцию в одной
сессии.

## Запуск артефакта

В `libreyolo/backends` нет записи для RKNN, поэтому `LibreYOLO()` не загружает
`.rknn`-файл. Скомпилированный артефакт разворачивают на плате и запускают в
собственной среде выполнения Rockchip, а предобработка, декодирование, NMS и
пересчёт координат остаются там задачей приложения.

`<model>.rknn.metadata.json` несёт имена классов, размер входа, задачу и целевую
платформу — это то, что нужно приложению, чтобы воспроизвести постобработку
LibreYOLO. Поставляйте его вместе со скомпилированной моделью.

Для проверки на хосте, которой не нужна плата, держите ONNX-артефакт с той же
фиксированной формой и сравнивайте его в симуляторе, как выше.

## Ограничения

Компилируются четыре комбинации, и это варианты моделей, а не семейства:

| Вариант | Задача | Холст | Платформа |
|---|---|---:|---|
| YOLO9-t | detect | 640 | RK3588 |
| YOLO9-E2E-t | detect | 640 | RK3588 |
| PicoDet-s | detect | 320 | RK3588 |
| YOLO-NAS-s | detect | 640 | RK3588 |

Всё остальное отклоняется до компиляции, с сообщением о том, что RKNN в этой
версии ограничен ровно теми вариантами детекции, которые проверены в симуляторе.
Результаты «только компиляция» для других моделей есть, но намеренно не выдаются
за поддержку: в том же прогоне измерений RF-DETR оставил два узла `GridSample` в
декодере без понижения, а D-FINE, RT-DETR, RT-DETRv2, RT-DETRv4, DEIM, DEIMv2 и
EC скомпилировались и отработали в симуляторе с декодированными выходами,
которые были существенно неверны.

Батч 1, статические формы, opset 19. `half=True` отклоняется, потому что RKNN не
предоставляет контракт `half` из LibreYOLO, а `int8=True` отклоняется до тех пор,
пока не появятся репрезентативная калибровка и результаты по точности на задаче.

Другие платформы Rockchip отклоняются: `rk3588` — единственная проверенная
платформа.

Полную сетку семейств и задач смотрите в
[матрице экспорта](/docs/reference/export-matrix). Для одной комбинации:

<code-tabs name="support" />
