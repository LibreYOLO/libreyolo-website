---
title: API de visión y lenguaje
seo_title: "API de LibreVLM: alias, set_classes y chat"
description: "La factoría LibreVLM, todos sus alias, el vocabulario persistente de set_classes, set_task, la vía de escape de chat y por qué la confianza es un marcador de posición."
lead: "LibreVLM carga un modelo generativo de visión y lenguaje y lo maneja como un detector de objetos. La lista de clases es un prompt en lugar de una cabeza fija, y el modelo devuelve los mismos Results que devuelve cualquier otra familia."
keywords:
  - LibreVLM
  - detección con modelos de visión y lenguaje
  - Qwen3-VL
  - LFM2-VL
  - InternVL3
  - SmolVLM2
  - Florence-2
  - libreyolo chat
last_verified: "1.5.0"
verification: "Alias leídos de libreyolo/models/vlm/__init__.py; repositorios, tamaños y listas de tareas de los módulos de familia bajo libreyolo/models/vlm/ más libreyolo/models/sensenova/model.py; reglas de llamada y excepciones de libreyolo/models/vlm/base.py, todo en la v1.5.0."
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[vlm]'
  usage:
    - label: Detectar un vocabulario abierto
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        model.set_classes(["person", "skateboard"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
    - label: Hacer una pregunta libre
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        print(model.chat(SAMPLE_IMAGE, "How many people are in this image?"))
---

## Instalación

El tier necesita el extra `vlm`.

<code-tabs name="install" />

## La factoría

```python
LibreVLM(model: str = "qwen3-vl-4b", **kwargs) -> LibreVLMModel
```

`model` es un alias, no una ruta. `**kwargs` llega al constructor de la familia,
que acepta `device`, `names` (el vocabulario inicial, equivalente a llamar a
`set_classes` después de la carga), `prompt` (sobrescribe el prompt de detección)
y `max_new_tokens`. Un alias desconocido lanza `ValueError` enumerando todos los
alias.

<code-tabs name="usage" />

## Alias

| Familia | Alias | Tamaños | Pesos |
|---|---|---|---|
| Qwen3-VL | `qwen3-vl`, `qwen3-vl-2b`, `qwen3-vl-4b`, `qwen3-vl-8b` | `2b`, `4b`, `8b` | `Qwen/Qwen3-VL-2B-Instruct`, `-4B-`, `-8B-` |
| LFM2-VL | `lfm2-vl`, `lfm2-vl-450m`, `lfm2-vl-1.6b` | `450m`, `1.6b` | `LiquidAI/LFM2.5-VL-450M`, `-1.6B` |
| InternVL3 | `internvl3`, `internvl3-1b`, `internvl3-2b`, `internvl3-8b` | `1b`, `2b`, `8b` | `OpenGVLab/InternVL3-1B-hf`, `-2B-hf`, `-8B-hf` |
| SmolVLM2 | `smolvlm2`, `smolvlm2-2.2b`, `smolvlm2-500m` | `2.2b`, `500m` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, `SmolVLM2-500M-Video-Instruct` |
| Florence-2 | `florence-2`, `florence2`, `florence-2-base`, `florence-2-large` | `base`, `large` | `florence-community/Florence-2-base`, `-large` |
| Kosmos-2 | `kosmos-2`, `kosmos2` | `224` | `microsoft/kosmos-2-patch14-224` |
| LocateAnything | `locate-anything`, `locateanything`, `locate-anything-3b`, `locateanything-3b` | `3b` | `nvidia/LocateAnything-3B` |
| SenseNova-Vision | `sensenova-vision`, `sensenova-vision-7b`, `sensenovavision` | `7b` | `LibreYOLO/SenseNovaVision7b` |
| LibreMODUS | `libremodus`, `libremodus-14b-a7b`, `modus`, `modus-14b-a7b` | `14b-a7b` | Snapshot upstream fijado |

El alias por defecto es `qwen3-vl-4b`. Los tamaños del alias por defecto de cada
familia son los que aparecen primero: `qwen3-vl` resuelve a `4b`, `lfm2-vl` a
`450m`, `internvl3` a `2b`, `smolvlm2` a `2.2b`, `florence-2` a `base`.

`LibreVLM`, `LibreLFM2VL`, `LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`,
`LibreFlorence2`, `LibreKosmos2`, `LibreLocateAnything` y `LibreMODUS`
(también escrito `LibreModus`) se exportan a nivel de paquete.

## Tareas

La mayoría de familias sirven solo `detect`. Dos sirven más:

| Familia | Tareas soportadas |
|---|---|
| LocateAnything | `detect`, `point` |
| SenseNova-Vision | `detect`, `segment`, `panoptic`, `pose`, `point`, `depth`, `ocr` |

Como la tarea la marca el prompt en lugar de venir fijada dentro de un
checkpoint, se puede cambiar sobre un modelo ya cargado:

```python
model.set_task(task: str) -> LibreVLMModel
```

La tarea se valida contra la lista de tareas soportadas de la familia, es
persistente en las llamadas posteriores a `predict()` y `track()`, y se devuelve
el modelo para que las llamadas se puedan encadenar.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreVLMModel
```

Define el vocabulario abierto. Vale cualquier palabra, porque al modelo se le
pasan en el prompt en lugar de restringirlo a una cabeza fija. La lista no puede
estar vacía y sus entradas deben ser únicas cuando se comparan sin distinguir
mayúsculas de minúsculas. Pasar una cadena suelta lanza `TypeError`, porque se
enumeraría en clases de un solo carácter. El vocabulario es persistente:
defínelo una vez después de cargar y sigue vigente hasta que lo vuelvas a
definir.

## chat

```python
model.chat(image, prompt, max_new_tokens=None, color_format="auto") -> str
```

Generación multimodal en crudo: entran imagen y prompt, sale el texto
decodificado, literal. Es la vía de escape que hay debajo de la comodidad de la
detección, para preguntas libres, para contar o para un formato de salida que el
envoltorio de detección no cubre. `max_new_tokens` cae por defecto en el
`MAX_NEW_TOKENS` de la familia, que es 1024 en la clase base. La decodificación
es greedy con una penalización de repetición suave.

## Confianza

La salida generada no tiene una confianza calibrada por caja. Esta versión
asigna un marcador de posición constante para que `predict`, el dibujado y
`track` se comporten, lo que hace que el filtrado con `conf=` y el mAP sean
blandos en lugar de significativos. Es también la razón por la que `val()` lanza
excepción: un mAP de COCO sobre puntuaciones de marcador de posición sería
engañoso.

## Predict y track

Se aplica la superficie de predicción estándar, y `track()` funciona, así que un
detector VLM encaja en el mismo pipeline que cualquier otra familia. Dos
políticas a nivel de clase difieren de un detector convolucional: el test-time
augmentation está desactivado, porque el aumento multiescala no tiene sentido
para un generador de resolución fija, y el predict por batches está apagado,
porque la generación es autorregresiva y el preprocesado devuelve una
codificación de texto e imagen en lugar de un tensor de imagen apilable.

## No soportado

`train()`, `val()` y `export()` lanzan `NotImplementedError`. Haz el fine-tuning
upstream y carga los pesos resultantes.

## Código remoto

Todas las familias que se distribuyen cargan a través de una clase de modelo
nativa, así que LibreYOLO no ejecuta por defecto código de repositorios de
terceros. Una familia que lo necesite de verdad tiene que optar por él
explícitamente y fijar una revisión de snapshot; LocateAnything es la que lo
hace, fijada al commit `c32291ca5e996f5a7a485845b4f57a233936bba0`.

LibreMODUS es una excepción explícita al esquema de checkpoints: su alias
resuelve a un directorio de archivos upstream fijados en lugar de a un `.pt` de
LibreYOLO, y LibreYOLO ni le añade metadatos v1.0 ni lo republica.
