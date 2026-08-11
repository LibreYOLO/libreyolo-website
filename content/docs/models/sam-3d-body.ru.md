---
title: SAM 3D Body
families: [sam3dbody]
seo_title: "SAM 3D Body: восстановление меша всего тела в LibreYOLO"
description: "Используйте SAM 3D Body в LibreYOLO для восстановления 3D-меша всего тела человека. Установка и предсказание; доступ к чекпойнтам ограничен лицензией Meta SAM License, нужна CUDA."
lead: "SAM 3D Body — управляемая промптами модель Meta, которая восстанавливает 3D-меш всего тела, включая кисти и стопы, по одному изображению и рамкам людей. LibreYOLO оборачивает upstream-пакет, а не портирует его."
keywords: [SAM 3D Body, MHR, Momentum Human Rig, human mesh recovery, 3d меш человека, восстановление 3d модели тела по фото, 3d поза человека python]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # Это семейство не зарегистрировано в фабрике LibreYOLO(), поэтому
        # объект создаётся напрямую. model_path=None запускает скачивание
        # с Hugging Face с ограниченным доступом; строка же трактуется как
        # путь к уже существующему локальному чекпойнту и никогда не
        # скачивается автоматически.
        # Инференсу нужно устройство с CUDA; пути через CPU нет.
        model = LibreSAM3DBody(None, size="d3", device="cuda")
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        meshes = result.meshes
        print(meshes.vertices.shape)    # (N, V, 3), система камеры, метры
        print(meshes.joints3d.shape)    # (N, J, 3)
    - label: С детектором людей
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # Здесь нет сокращения строкой-именем: передайте готовый детектор
        # LibreYOLO, обычный callable или экземпляр PersonDetector.
        detector = LibreYOLO("LibreRFDETRn.pt")
        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_detector=detector)
---

## Установка

```bash
pip install libreyolo
```

Так ставится только адаптер LibreYOLO. Сам SAM 3D Body в поставку не входит,
потому что из его лицензии собственный код LibreYOLO выводить нельзя:
клонируйте upstream-репозиторий и установите его зависимости самостоятельно, а
затем укажите LibreYOLO путь к клону.

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

```python
from libreyolo.models.sam3dbody import LibreSAM3DBody

model = LibreSAM3DBody(
    None,
    size="d3",
    sam_3d_body_path="/path/to/sam-3d-body",
    device="cuda",
)
```

или задайте переменную окружения `SAM_3D_BODY_PATH`, чтобы не передавать
`sam_3d_body_path` при каждом вызове. Тот, кто никогда не создаёт объект этого
семейства, никогда не запускает импорт и никогда не сталкивается с SAM License.
Это семейство не подключено ни к фабрике `LibreYOLO()`, ни к CLI-команде
`libreyolo predict`; `LibreSAM3DBody` — единственная точка входа.

## Предсказание

<code-tabs name="predict" />

Скачивание чекпойнта ограничено: чтобы первая загрузка прошла, нужно принять
лицензию Meta на странице модели в Hugging Face и авторизоваться командой
`hf auth login`. Самому инференсу устройство с CUDA нужно безусловно:
upstream-оценщик переносит батч на GPU без всяких проверок, поэтому машина
только с CPU выбрасывает ошибку, а не откатывается на запасной путь.
`result.meshes` — payload `Meshes`, построчно выровненный с `result.boxes` (по
строке на каждого найденного человека): `vertices` и `joints3d` заданы в
метрических единицах и уже включают оценённый перенос камеры, `joints2d` — в
пикселях исходного изображения, а вращения следуют соглашению MHR: углы Эйлера,
а не axis-angle. Про источники, стриминг и обработку результатов см.
[предсказание](/docs/predict).

## Варианты

Два бэкбона за одной и той же моделью тела MHR: `d3` использует энкодер DINOv3
ViT-H/16+, а `h` — исходный энкодер ViT-H.

## Экспорт

<export-matrix />

Экспорт меша тела не реализован: LibreYOLO пока не определил контракт
экспортируемого графа для задачи меша, в том числе то, как представлять
раскладку параметров MHR за пределами PyTorch.

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box>

Модель тела, которой управляют эти чекпойнты, — MHR (Momentum Human Rig) —
отдельный релиз Meta под лицензией Apache-2.0. LibreYOLO во время работы
скачивает её TorchScript-ассет из собственного публичного релиза MHR и
кэширует локально; этот файл LibreYOLO у себя не зеркалирует, и на него
распространяются его собственные условия Apache-2.0, а не SAM License.

</provenance-box>

## Цитирование

<citation-block />
