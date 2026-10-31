---
title: LoRA-Fine-Tuning
seo_title: LoRA-Fine-Tuning in LibreYOLO
description: >-
  Führe mit lora=True ein Fine-Tuning eines Transformer-Detektors bei wenig VRAM
  durch. Erfahre, welche neun Familien dies unterstützen, welches Adapterrezept
  jede Familie nutzt und wie sich die Checkpoints verhalten.
lead: >-
  LoRA friert die rechenintensiven vortrainierten Teile eines Modells ein und
  trainiert daneben kleine Low-Rank-Adapter sowie die Schichten, die dicht
  bleiben müssen. In LibreYOLO besteht die gesamte öffentliche Schnittstelle aus
  einem booleschen Wert.
keywords:
  - lora fine tuning
  - parametereffizientes fine tuning
  - peft
  - dora
  - training mit wenig vram
  - rf-detr lora
  - d-fine lora
  - adapter zusammenführen
last_verified: 1.5.0
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install "libreyolo[lora]"
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 lora=true
  merge:
    - label: Export führt die Adapter zusammen
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        model.export(format="onnx")
    - label: Direkt zusammenführen
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training.lora import merge_lora_adapters

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        merged = merge_lora_adapters(model.model)

        print(f"{merged} adapter layers folded into dense weights")
source_hash: 603fdddf5ec0c316
---

## Installation

LoRA verwendet die optionale Abhängigkeit `peft`.

<code-tabs name="install" />

Ohne diese Abhängigkeit löst `lora=True` einen `ImportError` aus, der den
Befehl nennt, anstatt versehentlich ein vollständiges Fine-Tuning zu starten.

## Verwendung

<code-tabs name="train" />

`lora=True` bildet die gesamte Schnittstelle. Rang, Alpha, Dropout und
Zielmodule sind für jede Familie passend zur jeweiligen Upstream-Referenz fest
vorgegeben und können nicht durch Nutzeroptionen angepasst werden.

Wenn eine Familie LoRA nicht unterstützt, wird bei der Einrichtung ein Fehler
ausgelöst, anstatt das Flag zu ignorieren:

```text
LoRA fine-tuning (lora=True) is not supported for yolo9. LoRA targets
transformer components with nn.Linear layers (e.g. RF-DETR, D-FINE, DEIM).
```

Die CLI lehnt die Option bereits vor dem Erstellen des Modells anhand ihrer
eigenen Allowlist derselben neun Familien ab.

## Unterstützte Familien

RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 und v4, EC und ConvNeXt. Die
Sperre wird durch das Attribut `supports_lora` der Trainerklasse jeder Familie
gesteuert. Die CLI enthält eine entsprechende Allowlist.

Die Aufgabenabdeckung ist enger als die Familienabdeckung. D-FINE und EC
unterstützen nur die Erkennung. Ihre Segmentierungs- und Pose-Pfade lösen einen
Fehler aus. Der semantische Pfad von RF-DETR löst ebenfalls einen Fehler aus.
ConvNeXt ist für die Klassifikation bestimmt.

Alle anderen Fälle lösen einen Fehler aus. Einen partiellen oder
stillschweigenden Modus gibt es nicht.

## Funktionsweise der einzelnen Rezepte

Die Rezepte unterscheiden sich, weil sich auch die Architekturen unterscheiden.
Ein Rezept für ein ViT-Backbone findet in einem Convolutional Backbone keine
passenden Ansatzpunkte.

RF-DETR verwendet DoRA, also Weight-Decomposed LoRA, mit Rang 16 und Alpha 16
für die Attention-Projektionen `query`, `key` und `value` des DINOv2-Backbones.
Dies entspricht der RF-DETR-Referenz. Das ViT-Backbone wird eingefroren,
während Projector, Decoder und Erkennungs-Head normal weitertrainiert werden.

D-FINE, DEIM und RT-DETR v1, v2 und v4 kombinieren ein Convolutional Backbone
mit einem hybriden Transformer-Encoder und einem deformierbaren Decoder. Daher
verschiebt sich die Aufteilung. Das Convolutional Backbone wird vollständig
eingefroren, wodurch auch sein Backward Pass entfällt. Die Transformer-Blöcke
frieren ihre Basisgewichte ein und trainieren einfache LoRA-Adapter mit Rang 16
und Alpha 16 auf ihren linearen Schichten: den Feed-Forward-Schichten `linear1`
und `linear2`, dem Gate und den Projektionen der deformierbaren Attention. Alle
anderen Teile, also die Convolution-Fusion des Encoders, die
Eingabeprojektionen, die Vorhersage-Heads und die Query-Embeddings, werden
weiterhin dicht trainiert.

Zwei Details dieses Rezepts sind beabsichtigt. Die Self-Attention des Decoders
bleibt ohne Adapter eingefroren, weil `nn.MultiheadAttention` von PyTorch
`out_proj.weight` direkt liest und einen injizierten Adapter unbemerkt umgehen
würde. Außerdem wird einfaches LoRA statt DoRA verwendet, weil mehrere lineare
Decoder-Schichten absichtlich mit Nullen initialisiert werden und die
Betragsnormalisierung von DoRA durch die Gewichtsnorm teilt.

DEIMv2 übernimmt dasselbe Rezept, wobei seine SwiGLU-Feed-Forward-Schichten
`w12` und `w3` als Ziele dienen. Die Größen S, M, L und X besitzen außerdem ein
DINOv3-ViT-Backbone. Dessen ViT-Basis wird eingefroren und seine fusionierten
Attention-Schichten `qkv` erhalten Adapter, während die Convolution-Pyramide
des Spatial Tuning Adapter als Gegenstück zum Projector weitertrainiert wird.
Diese `qkv`-Adapter werden auch dann eingefügt, wenn die ausgelieferte
Konfiguration das ViT bereits eingefroren hat, denn die Anpassung eines
eingefrorenen Backbones ist gerade der Zweck. Die Größen unterhalb von S
verwenden ein Convolutional Backbone und folgen dem einfachen Rezept.

EC ist ein DETR mit einem ViT-Backbone, das von einer trainierbaren
Convolution-Projector-Pyramide umgeben ist. Die ViT-Basis wird eingefroren und
ihre `qkv`-Schichten erhalten Adapter. Die Transformer-Blöcke folgen dem
gemeinsamen Rezept, während Projector und Heads dicht bleiben.

ConvNeXt-Blöcke besitzen Channels-Last-Linear-MLPs namens `fc1` und `fc2`, die
einfache Adapter erhalten. Die Depthwise Convolutions, Normalisierungen und
Layer-Scale-Parameter werden eingefroren. Der Klassifikations-Head bleibt
dicht, damit benutzerdefinierte Klassenzahlen weiterhin funktionieren.

Die Erkennungs- und Klassifikations-Heads bleiben bei jedem Rezept stets
trainierbar, weil eine benutzerdefinierte Klassenzahl einen neu trainierten
Head erfordert.

## Checkpoints und Export

`best.pt` und `last.pt` behalten die Adapter-Tensoren. Dadurch lässt sich ein
LoRA-Lauf wie jeder andere fortsetzen oder untersuchen. Zum Laden eines dieser
Checkpoints muss das Extra `lora` installiert sein. Der Loader wiederholt die
Adapterinjektion, damit die Schlüssel übereinstimmen.

`export()` führt die Adapter mit den dichten Gewichten zusammen. Ein
exportiertes Artefakt ist daher nicht von `peft` abhängig. Dieselbe
Zusammenführung kann direkt für ein Modell im Speicher ausgeführt werden.

<code-tabs name="merge" />

Nach einer Zusammenführung ist der Modulbaum vollständig dicht. Eine zweite
Zusammenführung hat keine Wirkung.

## Einsparungen und Grenzen

LoRA reduziert den Speicherbedarf von Optimizer und Gradienten. Bei den
Familien, deren Backbone vollständig eingefroren wird, entfällt außerdem der
Backward Pass dieses Backbones.

Der Aktivierungsspeicher bleibt unverändert. Die Forward-Aktivierungen müssen
weiterhin für alle trainierbaren Bestandteile aufbewahrt werden, was in der
Regel den Spitzenbedarf bestimmt. Verringere bei einem besonders knappen
VRAM-Budget zusätzlich `batch` oder `imgsz`.

## Verwandte Themen

- [Einfrieren von Schichten](/docs/train/layer-freezing) beschreibt die andere
  Möglichkeit, nur einen Teil der Gewichte zu trainieren. Sie funktioniert mit
  jeder Familie und benötigt keine zusätzliche Abhängigkeit. `freeze` und
  `lora=True` lassen sich kombinieren: Adapterparameter bleiben auch dann
  trainierbar, wenn ihre übergeordnete Backbone-Gruppe eingefroren ist.
- Unter [Hyperparameter](/docs/train/hyperparameters) findest du `batch`,
  `imgsz` und die übrigen Optionen von `train()`.

