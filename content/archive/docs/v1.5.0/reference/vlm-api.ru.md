---
title: API vision-language моделей
seo_title: 'API LibreVLM: алиасы, set_classes и chat'
description: >-
  Фабрика LibreVLM, все алиасы моделей, постоянный словарь set_classes,
  set_task, запасной выход chat и почему уверенность здесь — заглушка.
lead: >-
  LibreVLM загружает генеративную vision-language модель и использует её как
  детектор объектов. Список классов здесь — это промпт, а не фиксированная
  голова, и модель возвращает такие же Results, как любое другое семейство.
keywords:
  - LibreVLM
  - vision-language модель детекция python
  - Qwen3-VL
  - LFM2-VL
  - InternVL3
  - SmolVLM2
  - Florence-2
  - libreyolo chat
last_verified: 1.5.0
verification: >-
  Алиасы прочитаны из libreyolo/models/vlm/__init__.py; репозитории, размеры и
  списки задач — из модулей семейств в libreyolo/models/vlm/ и из
  libreyolo/models/sensenova/model.py; правила вызова и выбрасываемые исключения
  — из libreyolo/models/vlm/base.py, всё на версии v1.5.0.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[vlm]'
  usage:
    - label: Детекция с открытым словарём
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        model.set_classes(["person", "skateboard"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
    - label: Произвольный вопрос к модели
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        print(model.chat(SAMPLE_IMAGE, "How many people are in this image?"))
source_hash: 57ddac08bc4d4e05
---

## Установка

Этому уровню нужен extra `vlm`.

<code-tabs name="install" />

## Фабрика

```python
LibreVLM(model: str = "qwen3-vl-4b", **kwargs) -> LibreVLMModel
```

`model` — это алиас, а не путь. `**kwargs` передаётся в конструктор семейства,
который принимает `device`, `names` (начальный словарь, то же самое, что вызвать
`set_classes` после загрузки), `prompt` (переопределяет промпт детекции) и
`max_new_tokens`. Неизвестный алиас вызывает `ValueError` со списком всех алиасов.

<code-tabs name="usage" />

## Алиасы

| Семейство | Алиасы | Размеры | Веса |
|---|---|---|---|
| Qwen3-VL | `qwen3-vl`, `qwen3-vl-2b`, `qwen3-vl-4b`, `qwen3-vl-8b` | `2b`, `4b`, `8b` | `Qwen/Qwen3-VL-2B-Instruct`, `-4B-`, `-8B-` |
| LFM2-VL | `lfm2-vl`, `lfm2-vl-450m`, `lfm2-vl-1.6b` | `450m`, `1.6b` | `LiquidAI/LFM2.5-VL-450M`, `-1.6B` |
| InternVL3 | `internvl3`, `internvl3-1b`, `internvl3-2b`, `internvl3-8b` | `1b`, `2b`, `8b` | `OpenGVLab/InternVL3-1B-hf`, `-2B-hf`, `-8B-hf` |
| SmolVLM2 | `smolvlm2`, `smolvlm2-2.2b`, `smolvlm2-500m` | `2.2b`, `500m` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, `SmolVLM2-500M-Video-Instruct` |
| Florence-2 | `florence-2`, `florence2`, `florence-2-base`, `florence-2-large` | `base`, `large` | `florence-community/Florence-2-base`, `-large` |
| Kosmos-2 | `kosmos-2`, `kosmos2` | `224` | `microsoft/kosmos-2-patch14-224` |
| LocateAnything | `locate-anything`, `locateanything`, `locate-anything-3b`, `locateanything-3b` | `3b` | `nvidia/LocateAnything-3B` |
| SenseNova-Vision | `sensenova-vision`, `sensenova-vision-7b`, `sensenovavision` | `7b` | `LibreYOLO/SenseNovaVision7b` |
| LibreMODUS | `libremodus`, `libremodus-14b-a7b`, `modus`, `modus-14b-a7b` | `14b-a7b` | Зафиксированный снапшот из апстрима |

Алиас по умолчанию — `qwen3-vl-4b`. Для алиаса по умолчанию каждого семейства
берётся размер, указанный первым: `qwen3-vl` разрешается в `4b`, `lfm2-vl` — в
`450m`, `internvl3` — в `2b`, `smolvlm2` — в `2.2b`, `florence-2` — в `base`.

`LibreVLM`, `LibreLFM2VL`, `LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`,
`LibreFlorence2`, `LibreKosmos2`, `LibreLocateAnything` и `LibreMODUS`
(также пишется `LibreModus`) экспортируются на уровне пакета.

## Задачи

Большинство семейств поддерживают только `detect`. Два умеют больше:

| Семейство | Поддерживаемые задачи |
|---|---|
| LocateAnything | `detect`, `point` |
| SenseNova-Vision | `detect`, `segment`, `panoptic`, `pose`, `point`, `depth`, `ocr` |

Задача задаётся промптом, а не запечена в чекпойнт, поэтому её можно переключить
на уже загруженной модели:

```python
model.set_task(task: str) -> LibreVLMModel
```

Задача проверяется по списку поддерживаемых семейством задач, запоминается для
последующих вызовов `predict()` и `track()`, а модель возвращается, чтобы вызовы
можно было соединять в цепочку.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreVLMModel
```

Задаёт открытый словарь. Подойдут любые слова, потому что модель получает их в
промпте, а не ограничена фиксированной головой. Список должен быть непустым, а
его элементы — уникальными при сравнении без учёта регистра. Передача голой
строки вызывает `TypeError`, потому что она развернулась бы в набор
односимвольных классов. Словарь запоминается: задайте его один раз после загрузки, и он
действует, пока вы не зададите его снова.

## chat

```python
model.chat(image, prompt, max_new_tokens=None, color_format="auto") -> str
```

Сырая мультимодальная генерация: на входе изображение и промпт, на выходе —
декодированный текст, дословно. Это запасной выход под удобной обёрткой
детекции — для произвольных вопросов, подсчёта объектов или формата вывода,
который обёртка детекции не покрывает. `max_new_tokens` по умолчанию берёт
значение `MAX_NEW_TOKENS` семейства, а в базовом классе оно равно 1024.
Декодирование жадное, с мягким штрафом за повторы.

## Уверенность

У сгенерированного вывода нет калиброванной уверенности для каждой рамки. В этой
версии проставляется постоянная заглушка, чтобы работали `predict`, отрисовка и
`track`, из-за чего фильтрация по `conf=` и mAP получаются условными, а не
осмысленными. По этой же причине `val()` выбрасывает исключение: COCO mAP по
оценкам-заглушкам вводил бы в заблуждение.

## Предсказание и трекинг

Стандартный набор аргументов predict работает как обычно, `track()` тоже
работает, поэтому VLM-детектор встраивается в тот же пайплайн, что и любое
другое семейство. Две политики на уровне класса отличаются от свёрточного
детектора: аугментация на этапе инференса отключена, потому что многомасштабная
аугментация бессмысленна для генератора с фиксированным разрешением, а батчевый
predict выключен, потому что генерация авторегрессионная, а препроцессинг
возвращает кодировку из текста и изображения, а не тензор изображения, который
можно сложить в батч.

## Что не поддерживается

`train()`, `val()` и `export()` выбрасывают `NotImplementedError`. Дообучайте в
апстриме и загружайте полученные веса.

## Удалённый код

Каждое поставляемое семейство загружается через нативный класс модели, поэтому
LibreYOLO по умолчанию не выполняет код из сторонних репозиториев. Семейство,
которому это действительно нужно, должно явно это включить и зафиксировать
ревизию снапшота; LocateAnything — единственное такое семейство, зафиксировано
на коммите `c32291ca5e996f5a7a485845b4f57a233936bba0`.

LibreMODUS — явное исключение из схемы чекпойнтов: его алиас разрешается в
каталог зафиксированных файлов апстрима, а не в `.pt` от LibreYOLO, и LibreYOLO
не добавляет к нему метаданные v1.0 и не публикует его заново.
