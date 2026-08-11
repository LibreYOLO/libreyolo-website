---
title: Open-Vocabulary-API
seo_title: 'LibreOpenVocab-API: Aliasse und Argumente'
description: >-
  Die LibreOpenVocab-Factory, ihre vier Familien und alle Aliasse, set_classes,
  familienspezifische conf-Standardwerte sowie die Regeln für text_threshold und
  iou.
lead: >-
  LibreOpenVocab ist die Factory für textkonditionierte Detektoren. Die
  Klassenliste ist ein Prompt und kein fester Head. Das Vokabular wird daher mit
  set_classes festgelegt, und das Modell gibt dafür normale Erkennungs-Results
  zurück.
keywords:
  - LibreOpenVocab
  - open-vocabulary-erkennung
  - Grounding DINO
  - OWLv2
  - OMDet-Turbo
  - OV-DEIM
  - set_classes
last_verified: 1.5.0
verification: >-
  Aliasse aus libreyolo/models/openvocab/__init__.py; Repositorys, Größen und
  Schwellenwerte aus grounding_dino.py, owlv2.py, omdet_turbo.py und ov_deim.py;
  Aufrufregeln aus libreyolo/models/openvocab/base.py, jeweils für v1.5.0.
  Designabsicht aus docs/adr/0008-open-vocab-detector-contract.md.
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

## Installation

Die Stufe benötigt das Extra `openvocab`.

<code-tabs name="install" />

## Die Factory

```python
LibreOpenVocab(model: str = "grounding-dino-tiny", **kwargs) -> LibreOpenVocabDetector
```

`model` ist ein Alias und kein Pfad. Unterstriche werden vor dem Nachschlagen
in Bindestriche umgewandelt. Dadurch können die vom CLI-Inventar ausgegebenen
familienqualifizierten Namen wie `omdet_turbo-t` und `grounding_dino-t`
unverändert geladen werden. Ein unbekannter Alias löst einen `ValueError` aus,
der alle bekannten Aliasse aufführt.

Der Konstruktor akzeptiert `size`, `nb_classes=80`, `names=None`,
`device="auto"`, `task=None` und `text_threshold=None`. Die Übergabe von
`names` entspricht einem Aufruf von `set_classes` direkt nach dem Laden. Wird
`text_threshold` an eine nicht unterstützende Familie übergeben, löst sie
`TypeError` aus.

<code-tabs name="usage" />

## Familien und Aliasse

| Familie | Aliasse | Größen | Gewichte |
|---|---|---|---|
| Grounding DINO | `grounding-dino`, `groundingdino`, `grounding-dino-tiny`, `groundingdino-tiny`, `grounding-dino-t`, `groundingdino-t`, `grounding-dino-base`, `groundingdino-base`, `grounding-dino-b`, `groundingdino-b` | `t`, `b` | `LibreYOLO/LibreGroundingDINOt`, `LibreYOLO/LibreGroundingDINOb` |
| OWLv2 | `owlv2`, `owl-v2`, `owlv2-base`, `owl-v2-base`, `owlv2-b16`, `owl-v2-b16`, `owlv2-large`, `owl-v2-large`, `owlv2-l14`, `owl-v2-l14` | `b16`, `l14` | `LibreYOLO/LibreOWLv2b16`, `LibreYOLO/LibreOWLv2l14` |
| OMDet-Turbo | `omdet-turbo`, `omdet`, `omdetturbo`, `omdet-turbo-tiny`, `omdet-turbo-swin-tiny`, `omdet-turbo-t` | `t` | `LibreYOLO/LibreOMDetTurbot` |
| OV-DEIM | `ov-deim`, `ovdeim`, `ov-deim-s`, `ovdeim-s`, `ov-deim-m`, `ovdeim-m`, `ov-deim-l`, `ovdeim-l` | `s`, `m`, `l` | `LibreYOLO/LibreOVDEIMs`, `LibreYOLO/LibreOVDEIMm`, `LibreYOLO/LibreOVDEIMl` |

Der Standardalias lautet `grounding-dino-tiny`.

`LibreGroundingDINO`, `LibreOWLv2` und `LibreOMDetTurbo` werden auf Paketebene
exportiert und können direkt mit `size=` erstellt werden. OV-DEIM ist über die
oben aufgeführten Factory-Aliasse erreichbar.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreOpenVocabDetector
```

Legt das Vokabular für jeden späteren Aufruf von `predict()` fest und gibt das
Modell zurück, damit Aufrufe verkettet werden können. Die Liste darf nicht leer
sein, muss ausschließlich Strings enthalten und ihre Einträge müssen beim
Vergleich ohne Berücksichtigung der Groß- und Kleinschreibung eindeutig sein.
Leere Labels werden abgelehnt. Ein einfacher String löst `TypeError` aus, da er
sonst in Klassen aus einzelnen Zeichen aufgeteilt würde.

Nach dem Aufruf ordnet `model.names` die Werte `0..N-1` in der angegebenen
Reihenfolge den Labels zu, und `model.nb_classes` ist `N`.

## Aufrufargumente

Die Stufe verwendet die normale Vorhersageschnittstelle mit drei Unterschieden.

`conf` verwendet den eigenen Standardwert der Familie und nicht den
gemeinsamen Wert 0.25:

| Familie | Standard-`conf` | Unterdrückung |
|---|---|---|
| Grounding DINO | 0.25 | |
| OWLv2 | 0.1 | |
| OMDet-Turbo | 0.3 | Eigene Nachverarbeitung mit Schwellenwert 0.5, berücksichtigt `iou=` |
| OV-DEIM | 0.25 | Eindeutige Zuordnung mit Top-K-Auswahl, keine Unterdrückung |

`iou=` ist nur für eine Familie mit Unterdrückung sinnvoll. OMDet-Turbo nimmt
den Schwellenwert als Argument entgegen und verwendet standardmäßig 0.5, wenn
`iou=` fehlt. Die anderen drei Familien unterdrücken nichts. Dort wird die
Übergabe von `iou=` mit einer Warnung ignoriert.

`text_threshold=` wird nur von Grounding DINO unterstützt und hat dort den
Standardwert 0.25. Du kannst beim Erstellen einen dauerhaft gültigen Wert oder
pro Aufruf einen Wert übergeben. Ein aufrufbezogener Wert kann nicht mit
`stream=True` kombiniert werden, weil gestreamte Ergebnisse verzögert erzeugt
werden. Setze ihn stattdessen im Konstruktor. Jede andere Familie löst dafür
`TypeError` aus.

`imgsz=` löst `ValueError` aus, da die Vorverarbeitungspipeline dieser Stufe
die Größenänderung steuert. `augment=True` löst ebenfalls einen Fehler aus,
weil Test-Time-Augmentierung hier außerhalb des Funktionsumfangs liegt. Die
Eingabegrößen werden nur zu Referenzzwecken pro Familie gespeichert: Grounding
DINO 800, OWLv2 960 und 1008, OMDet-Turbo 640, OV-DEIM 640.

## Nicht unterstützte Funktionen

`train()`, `val()`, `track()` und `export()` lösen jeweils
`NotImplementedError` aus. Führe das Fine-Tuning im Upstream-Projekt durch und
lade die resultierenden Gewichte. Verwende statt Tracking für jeden Frame
`predict()`. Die Validierung würde einen eigenen Validator benötigen, da der
gemeinsame Erkennungsvalidator das Modell mit Bildtensoren aufruft, diese Stufe
aber textkonditionierte Eingaben benötigt.

