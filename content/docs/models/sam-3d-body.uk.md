---
title: SAM 3D Body
families:
  - sam3dbody
seo_title: 'SAM 3D Body: відновлення сітки всього тіла в LibreYOLO'
description: >-
  Використовуйте SAM 3D Body у LibreYOLO для відновлення тривимірної сітки
  всього тіла людини. Встановлення й передбачення; доступ до контрольних точок
  регулює ліцензія SAM від Meta, потрібна CUDA.
lead: >-
  SAM 3D Body, це модель Meta з підказками для відновлення тривимірної сітки
  всього тіла, зокрема рук і стоп, з одного зображення та рамок людей. LibreYOLO
  обгортає пакет upstream, а не переносить його код.
keywords:
  - SAM 3D Body
  - відновлення сітки людини
  - сітка тіла
  - MHR
  - Momentum Human Rig
  - поза 3D
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import SAMPLE_IMAGE

        from libreyolo.models.sam3dbody import LibreSAM3DBody


        # Це сімейство не зареєстроване у фабриці LibreYOLO(), тому

        # його створюють безпосередньо. Саме model_path=None запускає

        # завантаження із закритим доступом із Hugging Face; натомість рядок

        # вважається шляхом до наявної локальної контрольної точки й ніколи не
        завантажується автоматично.

        # Для інференсу потрібен пристрій CUDA; режиму CPU немає.

        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])


        meshes = result.meshes

        print(meshes.vertices.shape)    # (N, V, 3), система координат камери,
        метри

        print(meshes.joints3d.shape)    # (N, J, 3)
    - label: З детектором людей
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        from libreyolo.models.sam3dbody import LibreSAM3DBody


        # Тут немає скорочення у вигляді іменованого рядка: передайте створений
        детектор

        # LibreYOLO, звичайну викличну сутність або екземпляр PersonDetector.

        detector = LibreYOLO("LibreRFDETRn.pt")

        model = LibreSAM3DBody(None, size="d3", device="cuda")


        result = model(SAMPLE_IMAGE, person_detector=detector)
source_hash: 8edc8d7872f3f875
---

## Встановлення

```bash
pip install libreyolo
```

Так ви отримаєте лише адаптер LibreYOLO. Сам SAM 3D Body не входить до
пакета, оскільки його ліцензія не дозволяє створювати на його основі власний
код LibreYOLO: клонуйте репозиторій upstream і самостійно встановіть його
залежності, а потім укажіть LibreYOLO шлях до клону.

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

Або задайте змінну середовища `SAM_3D_BODY_PATH`, щоб не передавати
`sam_3d_body_path` під час кожного виклику. Користувач, який не створює це
сімейство, не запускає імпорт і не стикається з ліцензією SAM. Це сімейство
не підключене до фабрики `LibreYOLO()` чи команди CLI
`libreyolo predict`; єдина точка входу, це `LibreSAM3DBody`.

## Передбачення

<code-tabs name="predict" />

Завантаження контрольної точки має обмежений доступ: до першого успішного
завантаження потрібно прийняти ліцензію Meta на сторінці моделі Hugging Face
та автентифікуватися за допомогою `hf auth login`. Для самого інференсу
безумовно потрібен пристрій CUDA: оцінювач upstream переносить пакет даних на
GPU без перевірки, тому на машині лише з CPU виникає помилка, а резервного
режиму немає. `result.meshes`, це корисне навантаження `Meshes`, рядки
якого узгоджені з `result.boxes` (по одному рядку на виявлену людину):
`vertices` і `joints3d` подано в метричних одиницях, і вони вже містять
оцінене перенесення камери; `joints2d` подано в пікселях вихідного
зображення, а обертання відповідають угоді MHR, тобто це кути Ейлера, а не
вісь-кут. Джерела, потокову обробку й роботу з результатами описано в розділі
[передбачення](/docs/predict).

## Варіанти

Два бекбони для тієї самої моделі тіла MHR: `d3` використовує енкодер
DINOv3 ViT-H/16+, а `h` використовує оригінальний енкодер ViT-H.

## Експорт

<export-matrix />

Експорт сітки тіла не реалізовано: LibreYOLO ще не визначила контракт
експортованого графа для задачі сітки, зокрема спосіб подання структури
параметрів MHR поза PyTorch.

## Контрольні точки

Усі опубліковані файли ваг для цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box>

Модель тіла, яку використовують контрольні точки, MHR (Momentum Human Rig),
є окремим випуском Meta за ліцензією Apache-2.0. Під час виконання LibreYOLO
отримує її ресурс TorchScript із власного загальнодоступного випуску MHR та
кешує локально; LibreYOLO не створює дзеркало цього файла, і на нього
поширюються власні умови Apache-2.0, а не ліцензія SAM.

</provenance-box>

## Цитування

<citation-block />

