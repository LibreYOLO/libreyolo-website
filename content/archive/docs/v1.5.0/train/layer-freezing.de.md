---
title: Einfrieren von Schichten
seo_title: Schichten während des Trainings in LibreYOLO einfrieren
description: >-
  Friere einen Teil eines Modells für Transfer Learning ein: mit einer
  ganzzahligen Anzahl familienspezifischer Freeze-Gruppen, einer expliziten
  Indexliste oder Selektoren für Modul- und Parameternamen.
lead: >-
  Beim Einfrieren bleiben ausgewählte Gewichte konstant, während der Rest des
  Modells trainiert wird. Selektoren beziehen sich auf die geordneten
  Freeze-Gruppen oder Modulnamen einer Familie und nicht auf rohe Schichtnummern
  eines YAML-Graphen.
keywords:
  - schichten einfrieren
  - transfer learning
  - backbone einfrieren
  - frozen batchnorm
  - freeze gruppen
  - nur head nachtrainieren
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Die ersten 10 Gruppen bilden das gesamte YOLOv9-Backbone.
        model.train(data="my-dataset.yaml", epochs=50, freeze=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=50 freeze=10
    - label: Nach Namen
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, freeze="backbone")
    - label: Mehrere Selektoren
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", freeze=["backbone", "neck"])
  groups:
    - label: Freeze-Gruppen einer Familie der Reihe nach auflisten
      language: python
      code: |
        from libreyolo import LibreYOLO9
        from libreyolo.models.yolo9.trainer import YOLO9Trainer

        model = LibreYOLO9("LibreYOLO9s.pt", size="s")
        trainer = YOLO9Trainer(model=model.model, wrapper_model=model, size="s")

        for index, (name, _module) in enumerate(trainer.get_freeze_groups()):
            print(index, name)
source_hash: 9f1e7551af6b16fe
---

## Einfrieren von Komponenten

`freeze` ist optional. Standardmäßig wird nichts eingefroren.

<code-tabs name="train" />

Das Einfrieren erfolgt nach dem Erstellen des Modells und nach einem möglichen
Neuaufbau des Heads für eine neue Klassenzahl, aber vor dem Erstellen des
Optimizers. Der Optimizer erhält daher ausschließlich trainierbare Parameter.

## Mögliche Selektoren

| Wert | Bedeutung |
|---|---|
| `None`, `False`, `""`, `"none"` | Alle Parameter trainieren |
| `10` oder `"10"` | Die ersten zehn Freeze-Gruppen der Familie einfrieren |
| `[0, 3, 7]` | Diese nullbasierten Gruppen einfrieren |
| `"backbone"` | Die passende Gruppe, das passende Modul oder das passende Parameterpräfix einfrieren |
| `["backbone", "neck"]` | Jeden aufgeführten Selektor einfrieren |
| `["backbone", 3]` | Gemischte Listen funktionieren ebenfalls |

Ein String wird vor seiner Interpretation geparst. Daher akzeptieren die CLI
und eine YAML-Konfiguration dieselben Formen wie Python.
`freeze="[0, 3, 'head']"` wird als literale Liste geparst,
`freeze="backbone,neck"` am Komma geteilt und ein einfacher Dezimalstring in
eine Anzahl umgewandelt.

`freeze=True` wird abgelehnt, weil die Angabe mehrdeutig ist.

Namensselektoren können mit dem Namen einer Freeze-Gruppe, einem Modulnamen oder
einem Parameternamen-Präfix übereinstimmen. Die Glob-Zeichen `*`, `?` und `[`
werden unterstützt. Ein vorangestelltes `model.` wird flexibel behandelt.
Dadurch treffen `backbone` und `model.backbone` jeweils die intern von der
Familie verwendete Schreibweise.

## Familienspezifische Gruppen

Eine Ganzzahl bezeichnet die familieneigene geordnete Liste von Freeze-Gruppen
und keine Position in einem gemeinsamen Graphen. Die Familien von LibreYOLO
bestehen nicht alle aus demselben sequenziellen, per YAML indizierten Modell.
Eine rohe Schichtnummer hätte deshalb bei jeder Familie eine andere Bedeutung.

YOLOv9 ordnet seine Gruppen von der Eingabeseite aus: zehn Backbone-Stufen,
anschließend sechs Neck-Stufen und zuletzt den Head. Deshalb entspricht
`freeze=10` genau dem Backbone. Darüber bilden `backbone`, `neck` und `head`
stabile Namensselektoren.

Die Gruppen von RF-DETR heißen `backbone.encoder`, `backbone.projector`,
`decoder`, `queries`, `transformer.encoder_output` und `head`. Namen sind hier
die bessere Wahl, weil Transformer-Komponenten keiner Schichtanzahl zugeordnet
werden können. `backbone` trifft per Präfix beide Backbone-Gruppen.

Familien ohne semantisch definierte Gruppen greifen auf einen konservativen
Standard zurück: Jedes direkte Kind des Modells, das mindestens einen Parameter
besitzt, bildet in Deklarationsreihenfolge eine Gruppe. Dies ist normalerweise
eine kurze Liste. Eine große Ganzzahl findet daher nicht genügend Gruppen:

```text
freeze index 10 is out of range for 3 available freeze groups.
```

Zeige die tatsächliche Liste an, anstatt zu raten:

<code-tabs name="groups" />

## Deutliche Fehlermeldungen

Jede falsche Angabe löst einen Fehler aus, damit nicht unbemerkt etwas anderes
trainiert wird als angefordert.

Ein Selektor ohne Treffer löst einen Fehler aus, der die erfolglosen Selektoren
nennt:

```text
freeze selector(s) matched no parameters: 'backbon'
```

Wenn nach dem Einfrieren keine trainierbaren Parameter mehr übrig wären, wird
sowohl beim Einfrieren als auch erneut beim Aufbau des Optimizers ein Fehler
ausgelöst:

```text
freeze would leave no trainable parameters. Use a smaller freeze value or
target a narrower module.
```

Dies geschieht bei `freeze="all"`, da `all` mit jedem Parameter übereinstimmt.

Bei erfolgreichem Einfrieren hält eine Zeile den Vorgang fest:

```text
Layer freezing: selectors=[10], tensors=124, params=2103776, trainable=1863456/3967232
```

## Keine Aktualisierung eingefrorener BatchNorm-Statistiken

Ein eingefrorener Parameter befindet sich weiterhin in einem Modul, dessen
laufende Statistiken sich sonst weiter ändern würden. Jedes BatchNorm-artige
Modul, dessen Parameter zur eingefrorenen Menge gehören, wird in den
Evaluierungsmodus versetzt. Der Trainer wendet dies nach jedem Aufruf von
`model.train()` zu Beginn einer Epoche erneut an, damit die Statistiken über
den gesamten Lauf konstant bleiben.

Dieses Verhalten ist standardmäßig aktiviert und sorgt dafür, dass beim
Einfrieren eines Backbones tatsächlich alle seine Zustände unverändert bleiben.

## Kombination mit LoRA

`freeze` und `lora=True` können gemeinsam verwendet werden. Bei RF-DETR, DEIM
und ConvNeXt bleiben die Adapterparameter auch dann trainierbar, wenn ihre
übergeordnete Gruppe eingefroren ist. Diese Kombination ist erwünscht: ein
eingefrorenes Backbone, auf dem Adapter lernen. Siehe
[LoRA-Fine-Tuning](/docs/train/lora).

## Umfang

Hierbei handelt es sich um statisches Einfrieren, das beim Start festgelegt
wird. Zeitgesteuertes Auftauen und progressives Einfrieren gehören nicht zur
Schnittstelle.

## Verwandte Themen

- Unter [Hyperparameter](/docs/train/hyperparameters) findest du die übrigen
  Optionen von `train()`.
- [Distillation](/docs/train/distillation) beschreibt eine weitere Möglichkeit,
  Wissen eines großen Modells in einen Trainingslauf zu übertragen.

