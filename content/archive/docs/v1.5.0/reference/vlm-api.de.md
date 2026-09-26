---
title: Vision-Language-API
seo_title: 'LibreVLM-API: Aliasse, set_classes und chat'
description: >-
  Die LibreVLM-Factory, alle Modellaliasse, das dauerhafte
  set_classes-Vokabular, set_task, der chat-Ausweg und der Grund für die
  Platzhalter-Confidence.
lead: >-
  LibreVLM lädt ein generatives Vision-Language-Modell und steuert es als
  Objektdetektor. Die Klassenliste ist ein Prompt und kein fester Head. Das
  Modell gibt dieselben Results wie jede andere Familie zurück.
keywords:
  - LibreVLM
  - vision-language-modell objekterkennung
  - Qwen3-VL
  - LFM2-VL
  - InternVL3
  - SmolVLM2
  - Florence-2
  - libreyolo chat
last_verified: 1.5.0
verification: >-
  Aliasse aus libreyolo/models/vlm/__init__.py; Repositorys, Größen und
  Aufgabenlisten aus den Familienmodulen unter libreyolo/models/vlm/ sowie
  libreyolo/models/sensenova/model.py; Aufrufregeln und Ausnahmen aus
  libreyolo/models/vlm/base.py, jeweils für v1.5.0.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[vlm]'
  usage:
    - label: Mit offenem Vokabular erkennen
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        model.set_classes(["person", "skateboard"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
    - label: Freie Frage stellen
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        print(model.chat(SAMPLE_IMAGE, "How many people are in this image?"))
source_hash: 57ddac08bc4d4e05
---

## Installation

Die Stufe benötigt das Extra `vlm`.

<code-tabs name="install" />

## Die Factory

```python
LibreVLM(model: str = "qwen3-vl-4b", **kwargs) -> LibreVLMModel
```

`model` ist ein Alias und kein Pfad. `**kwargs` erreicht den Konstruktor der
Familie. Dieser nimmt `device`, `names` (das anfängliche Vokabular,
gleichbedeutend mit einem Aufruf von `set_classes` nach dem Laden), `prompt`
(Überschreibung des Erkennungs-Prompts) und `max_new_tokens` entgegen. Ein
unbekannter Alias löst einen `ValueError` aus, der alle Aliasse aufführt.

<code-tabs name="usage" />

## Aliasse

| Familie | Aliasse | Größen | Gewichte |
|---|---|---|---|
| Qwen3-VL | `qwen3-vl`, `qwen3-vl-2b`, `qwen3-vl-4b`, `qwen3-vl-8b` | `2b`, `4b`, `8b` | `Qwen/Qwen3-VL-2B-Instruct`, `-4B-`, `-8B-` |
| LFM2-VL | `lfm2-vl`, `lfm2-vl-450m`, `lfm2-vl-1.6b` | `450m`, `1.6b` | `LiquidAI/LFM2.5-VL-450M`, `-1.6B` |
| InternVL3 | `internvl3`, `internvl3-1b`, `internvl3-2b`, `internvl3-8b` | `1b`, `2b`, `8b` | `OpenGVLab/InternVL3-1B-hf`, `-2B-hf`, `-8B-hf` |
| SmolVLM2 | `smolvlm2`, `smolvlm2-2.2b`, `smolvlm2-500m` | `2.2b`, `500m` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, `SmolVLM2-500M-Video-Instruct` |
| Florence-2 | `florence-2`, `florence2`, `florence-2-base`, `florence-2-large` | `base`, `large` | `florence-community/Florence-2-base`, `-large` |
| Kosmos-2 | `kosmos-2`, `kosmos2` | `224` | `microsoft/kosmos-2-patch14-224` |
| LocateAnything | `locate-anything`, `locateanything`, `locate-anything-3b`, `locateanything-3b` | `3b` | `nvidia/LocateAnything-3B` |
| SenseNova-Vision | `sensenova-vision`, `sensenova-vision-7b`, `sensenovavision` | `7b` | `LibreYOLO/SenseNovaVision7b` |
| LibreMODUS | `libremodus`, `libremodus-14b-a7b`, `modus`, `modus-14b-a7b` | `14b-a7b` | Festgelegter Upstream-Snapshot |

Der Standardalias lautet `qwen3-vl-4b`. Der jeweilige Standardalias einer
Familie verwendet die zuerst aufgeführte Größe: `qwen3-vl` wird zu `4b`,
`lfm2-vl` zu `450m`, `internvl3` zu `2b`, `smolvlm2` zu `2.2b` und
`florence-2` zu `base`.

`LibreVLM`, `LibreLFM2VL`, `LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`,
`LibreFlorence2`, `LibreKosmos2`, `LibreLocateAnything` und `LibreMODUS` (auch
`LibreModus` geschrieben) werden auf Paketebene exportiert.

## Aufgaben

Die meisten Familien unterstützen nur `detect`. Zwei unterstützen mehr:

| Familie | Unterstützte Aufgaben |
|---|---|
| LocateAnything | `detect`, `point` |
| SenseNova-Vision | `detect`, `segment`, `panoptic`, `pose`, `point`, `depth`, `ocr` |

Da die Aufgabe durch einen Prompt gesteuert und nicht fest in einen Checkpoint
integriert ist, kann sie bei einem geladenen Modell gewechselt werden:

```python
model.set_task(task: str) -> LibreVLMModel
```

Die Aufgabe wird anhand der von der Familie unterstützten Liste validiert,
bleibt für spätere Aufrufe von `predict()` und `track()` aktiv und gibt das
Modell zurück, damit Aufrufe verkettet werden können.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreVLMModel
```

Legt das offene Vokabular fest. Beliebige Wörter funktionieren, da das Modell
mit ihnen gepromptet und nicht durch einen festen Head eingeschränkt wird. Die
Liste darf nicht leer sein, und ihre Einträge müssen beim Vergleich ohne
Berücksichtigung der Groß- und Kleinschreibung eindeutig sein. Ein einfacher
String löst `TypeError` aus, da er sonst in Klassen aus einzelnen Zeichen
aufgeteilt würde. Das Vokabular bleibt aktiv. Lege es einmal nach dem Laden
fest, und es gilt bis zur nächsten Änderung.

## chat

```python
model.chat(image, prompt, max_new_tokens=None, color_format="auto") -> str
```

Unverarbeitete multimodale Generierung: Bild und Prompt hinein, decodierter
Text unverändert heraus. Dies ist der Ausweg unterhalb der komfortablen
Erkennung für freie Fragen, Zählaufgaben oder ein Ausgabeformat, das der
Erkennungs-Wrapper nicht abdeckt. `max_new_tokens` fällt auf den Wert
`MAX_NEW_TOKENS` der Familie zurück, der in der Basisklasse 1024 beträgt. Die
Decodierung verwendet Greedy Decoding mit einer leichten Wiederholungsstrafe.

## Confidence

Die generierte Ausgabe besitzt keine kalibrierte Confidence pro Box. Diese
Version weist einen konstanten Platzhalter zu, damit `predict`, Zeichnen und
`track` funktionieren. Dadurch ist die Filterung mit `conf=` und mAP eher
formal als aussagekräftig. Aus demselben Grund löst `val()` einen Fehler aus:
Eine COCO-mAP auf Platzhalter-Scores wäre irreführend.

## Vorhersage und Tracking

Die normale Vorhersageschnittstelle gilt, und `track()` funktioniert. Ein
VLM-Detektor kann daher in dieselbe Pipeline wie jede andere Familie eingesetzt
werden. Zwei Richtlinien auf Klassenebene unterscheiden sich von einem
Convolutional Detector. Test-Time-Augmentierung ist deaktiviert, weil eine
Multi-Scale-Augmentierung für einen Generator mit fester Auflösung nicht
sinnvoll ist. Vorhersagen in Batches sind ebenfalls deaktiviert, da die
Generierung autoregressiv ist und die Vorverarbeitung eine Text-und-Bild-
Codierung statt eines stapelbaren Bildtensors zurückgibt.

## Nicht unterstützte Funktionen

`train()`, `val()` und `export()` lösen `NotImplementedError` aus. Führe das
Fine-Tuning im Upstream-Projekt durch und lade die resultierenden Gewichte.

## Remote-Code

Jede ausgelieferte Familie wird über eine native Modellklasse geladen.
LibreYOLO führt daher standardmäßig keinen Code eines Drittanbieter-Repositorys
aus. Eine Familie, die ihn wirklich benötigt, muss sich ausdrücklich dafür
entscheiden und eine Snapshot-Revision festlegen. LocateAnything ist der
einzige solche Fall und verwendet den Commit
`c32291ca5e996f5a7a485845b4f57a233936bba0`.

LibreMODUS ist eine ausdrückliche Ausnahme vom Checkpoint-Schema. Sein Alias
wird zu einem Verzeichnis festgelegter Upstream-Dateien statt einer
LibreYOLO-`.pt` aufgelöst. LibreYOLO ergänzt weder v1.0-Metadaten noch
veröffentlicht es die Dateien erneut.

