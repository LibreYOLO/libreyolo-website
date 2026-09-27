---
title: API de vocabulario abierto
seo_title: 'API de LibreOpenVocab: alias y argumentos'
description: >-
  La factoría LibreOpenVocab, sus cuatro familias y todos sus alias,
  set_classes, los valores de conf por defecto de cada familia y las reglas de
  text_threshold e iou.
lead: >-
  LibreOpenVocab es la factoría de detectores condicionados por texto. La lista
  de clases es un prompt en lugar de una cabeza fija, así que el vocabulario se
  fija con set_classes y el modelo devuelve Results de detección normales frente
  a ella.
keywords:
  - LibreOpenVocab
  - detección de vocabulario abierto python
  - Grounding DINO
  - OWLv2
  - OMDet-Turbo
  - OV-DEIM
  - detectar objetos por texto sin entrenar
last_verified: 1.5.0
verification: >-
  Alias leídos de libreyolo/models/openvocab/__init__.py; repositorios, tamaños
  y umbrales de grounding_dino.py, owlv2.py, omdet_turbo.py y ov_deim.py; reglas
  de llamada de libreyolo/models/openvocab/base.py, todo en la v1.5.0. Intención
  de diseño de docs/adr/0008-open-vocab-detector-contract.md.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[openvocab]'
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-tiny")
        model.set_classes(["person", "skateboard", "handrail"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
source_hash: 64e4c641c6f8cde0
---

## Instalación

El tier necesita el extra `openvocab`.

<code-tabs name="install" />

## La factoría

```python
LibreOpenVocab(model: str = "grounding-dino-tiny", **kwargs) -> LibreOpenVocabDetector
```

`model` es un alias, no una ruta. Los guiones bajos se convierten en guiones
antes de la búsqueda, así que los nombres cualificados por familia que imprime
el inventario de la CLI, como `omdet_turbo-t` y `grounding_dino-t`, se cargan
tal cual. Un alias desconocido lanza `ValueError` con la lista de todos los
alias conocidos.

El constructor acepta `size`, `nb_classes=80`, `names=None`,
`device="auto"`, `task=None` y `text_threshold=None`. Pasar `names` equivale a
llamar a `set_classes` justo después de cargar. Pasar `text_threshold` a una
familia que no lo admite lanza `TypeError`.

<code-tabs name="usage" />

## Familias y alias

| Familia | Alias | Tamaños | Pesos |
|---|---|---|---|
| Grounding DINO | `grounding-dino`, `groundingdino`, `grounding-dino-tiny`, `groundingdino-tiny`, `grounding-dino-t`, `groundingdino-t`, `grounding-dino-base`, `groundingdino-base`, `grounding-dino-b`, `groundingdino-b` | `t`, `b` | `LibreYOLO/LibreGroundingDINOt`, `LibreYOLO/LibreGroundingDINOb` |
| OWLv2 | `owlv2`, `owl-v2`, `owlv2-base`, `owl-v2-base`, `owlv2-b16`, `owl-v2-b16`, `owlv2-large`, `owl-v2-large`, `owlv2-l14`, `owl-v2-l14` | `b16`, `l14` | `LibreYOLO/LibreOWLv2b16`, `LibreYOLO/LibreOWLv2l14` |
| OMDet-Turbo | `omdet-turbo`, `omdet`, `omdetturbo`, `omdet-turbo-tiny`, `omdet-turbo-swin-tiny`, `omdet-turbo-t` | `t` | `LibreYOLO/LibreOMDetTurbot` |
| OV-DEIM | `ov-deim`, `ovdeim`, `ov-deim-s`, `ovdeim-s`, `ov-deim-m`, `ovdeim-m`, `ov-deim-l`, `ovdeim-l` | `s`, `m`, `l` | `LibreYOLO/LibreOVDEIMs`, `LibreYOLO/LibreOVDEIMm`, `LibreYOLO/LibreOVDEIMl` |

El alias por defecto es `grounding-dino-tiny`.

`LibreGroundingDINO`, `LibreOWLv2` y `LibreOMDetTurbo` se exportan a nivel de
paquete y se pueden construir directamente con `size=`. A OV-DEIM se llega a
través de los alias de la factoría de arriba.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreOpenVocabDetector
```

Fija el vocabulario para todas las llamadas posteriores a `predict()` y
devuelve el modelo, de modo que las llamadas se pueden encadenar. La lista no
puede estar vacía, solo puede contener strings y sus entradas deben ser únicas
al compararlas sin distinguir mayúsculas y minúsculas; las etiquetas en blanco
se rechazan. Pasar una string suelta lanza `TypeError`, porque se enumeraría
en clases de un solo carácter.

Tras la llamada, `model.names` asigna `0..N-1` a las etiquetas en el orden
dado, y `model.nb_classes` es `N`.

## Argumentos de llamada

El tier reutiliza la superficie estándar de predict con tres diferencias.

`conf` toma por defecto el valor propio de cada familia en lugar del 0.25
compartido:

| Familia | conf por defecto | Supresión |
|---|---|---|
| Grounding DINO | 0.25 | |
| OWLv2 | 0.1 | |
| OMDet-Turbo | 0.3 | Su propio post-procesado, umbral 0.5, respeta `iou=` |
| OV-DEIM | 0.25 | Emparejamiento uno a uno con selección top-K, sin supresión |

`iou=` solo significa algo en una familia que ejecuta supresión. OMDet-Turbo
toma el umbral como argumento y usa 0.5 por defecto cuando `iou=` no se indica.
Las otras tres no suprimen nada, así que pasar `iou=` ahí emite un aviso y se
ignora.

`text_threshold=` es exclusivo de Grounding DINO, donde vale 0.25 por defecto.
Se puede pasar en la construcción para tener un valor persistente, o llamada a
llamada. Un valor por llamada no se puede combinar con `stream=True`, porque
los resultados en streaming se generan de forma perezosa; en ese caso, fíjalo
en el constructor. Cualquier otra familia lanza `TypeError` con él.

`imgsz=` lanza `ValueError`: en este tier, el pipeline de preprocesado es el
dueño del redimensionado. `augment=True` también lanza, porque el aumento en
tiempo de test queda fuera de alcance aquí. Los tamaños de entrada se registran
por familia solo como referencia: Grounding DINO 800, OWLv2 960 y 1008,
OMDet-Turbo 640, OV-DEIM 640.

## No soportado

`train()`, `val()`, `track()` y `export()` lanzan todos
`NotImplementedError`. Haz el fine-tuning upstream y carga los pesos
resultantes; ejecuta `predict()` por frame en lugar de hacer tracking. La
validación necesitaría un validador dedicado, porque el validador de detección
compartido llama al modelo con tensores de imagen mientras que este tier
requiere entradas condicionadas por texto.
