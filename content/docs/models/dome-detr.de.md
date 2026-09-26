---
title: Dome-DETR
families:
  - domedetr
seo_title: 'Dome-DETR: Erkennung winziger Objekte in LibreYOLO'
description: "Nutze Dome-DETR für Erkennung winziger Objekte, Training und Validierung. Gespiegelte vortrainierte Checkpoints bleiben auf akademische Forschung beschränkt."
lead: >-
  Ein auf D-FINE basierender Spezialist für winzige Objekte: Ein Dichte-Head
  bestimmt, wo sich Objekte befinden, die Encoder-Attention beschränkt sich auf
  die Fenster, die sie enthalten, und die Query-Anzahl wird aus dieser Dichte
  abgeleitet, statt fest zu sein. LibreYOLO unterstützt ihn für die
  Objekterkennung.
keywords:
  - dome-detr
  - winzige objekte erkennen
  - kleine objekte erkennen
  - luftbilder objekterkennung
  - drohnenbilder erkennung
  - fernerkundung
  - visdrone
  - ai-tod
  - detr
  - dichteadaptive queries
last_verified: "1.6.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Vortrainierte Gewichte sind auf akademische Forschung beschränkt.
        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt", device="cpu")
        print(model(SAMPLE_IMAGE).boxes)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")

        model.train(data="my-dataset.yaml", epochs=160, imgsz=800, batch=4,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml \
          epochs=160 imgsz=800 batch=4 lr0=2e-4
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml
source_hash: 8482301790a9b8d9
---

## Installation

Dome-DETR benötigt kein optionales Zusatzpaket. Alle Importe sind in der
Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Die sechs konvertierten Checkpoints werden automatisch von LibreYOLO-Mirrors heruntergeladen. Ihre Upstream-Bedingungen beschränken die Nutzung auf akademische Forschung.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt entspricht dem aller anderen Familien. Der
Wechsel zu einem anderen Detektor erfordert daher nur eine Änderung in einer
Zeile. `conf` und `max_det` filtern die Auswahl der Queries. `iou` wird aus
Gründen der API-Parität akzeptiert, hat aber keine Wirkung, weil der Decoder
Mengen vorhersagt und keinen NMS-Schritt verwendet. Unter
[Vorhersage](/docs/predict) findest du Quellen, Streaming und die Verarbeitung
von Ergebnissen.

Zwei Funktionen sind für diese Familie abgeschaltet. Die CUDA-Graph-Erfassung
ist deaktiviert, weil die Query-Anzahl von PAQI datenabhängig ist und sich die
Form des Vorwärtslaufs deshalb von Bild zu Bild ändert. Genau das kann die
Graph-Erfassung nicht abbilden. Die Test-Time Augmentation läuft mit einer
einzelnen festen quadratischen Größe, sodass eine mehrskalige TTA-Anforderung
keine Wirkung hat.

## Varianten

Es gibt drei Größen s, m und l, alle mit 800 mal 800. Die Größe wählt das
Backbone. Der Datensatz, aus dem die Gewichte stammen, bestimmt die Tiefe des
Decoders und das Query-Budget. Ein Größencode allein identifiziert daher keinen
Graphen. AI-TOD-V2-Gewichte wählen pro Bild zwischen 300 und 1500 Queries,
VisDrone-Gewichte zwischen 250 und 500. Das große Modell verwendet auf AI-TOD-V2
vier Decoder-Schichten, auf VisDrone dagegen sechs.

Dome-DETR erweitert D-FINE um drei Bestandteile. DeFE sagt eine Dichtekarte
vorher. MWAS beschränkt mithilfe dieser Karte die Encoder-Attention auf die
Fenster, die tatsächlich Objekte enthalten, statt alle Bereiche zu beachten.
PAQI leitet die Größe der Query-Menge aus derselben Dichte ab, statt stets 300
Queries zu dekodieren. Der Gewinn konzentriert sich auf die kleinsten Objekte
und nimmt mit deren Größe ab: In der eigenen Ablation des Upstreams steigt AP
für sehr winzige Objekte von 14.0 auf 17.8, während AP für mittelgroße Objekte
nur von 45.4 auf 46.4 steigt. Betrachte das Modell als Ergänzung zu
[D-FINE](/docs/models/d-fine) für Luft-, Drohnen- und Fernerkundungsbilder,
nicht als Ersatz.

Für diese Familie sind keine Vision-Analysis-Benchmarkzeilen erfasst.

## Training

Dome-DETR kann trainiert werden. Beim Training wird die vollständige
Upstream-Zielfunktion ausgeführt: die D-FINE-Losses sowie die Dichte- und
Anzahlüberwachung von DeFE. Aufgefüllte Queries werden aus den
Klassifizierungstermen ausgeschlossen. Bildweise Denoising-Attention-Masken
verhindern außerdem, dass die Auffüllung eines Bildes in ein anderes gelangt.

<code-tabs name="train" />

Die Konfiguration übernimmt das D-FINE-Rezept und ändert die von MWAS
benötigten Werte. `imgsz` ist 800, `lr0` ist `2e-4`, die Backbone-Parametergruppe
wird mit `backbone_lr_mult=0.1` skaliert, und `multi_scale` wird abgeschaltet,
weil die MWAS-Fenster eine durch Stride 8 teilbare Eingabe benötigen. `batch`
ist standardmäßig 4 statt 16 wie bei D-FINE: PAQI füllt jeden Batch bis zu
seinem breitesten Element auf, sodass der Speicherbedarf vom umfangreichsten
Bild im Batch abhängt und nicht vom Durchschnitt.

Zur Accuracy gilt eine wichtige Einschränkung. Upstream trainiert 160 Epochen
mit `MultiStepLR(milestones=[80, 120], gamma=0.8)`. Diese Standardwerte verwenden
für dieselben 160 Epochen dagegen den Flat-Cosine-Zeitplan von D-FINE. Dieser
Zeitplan wurde hier nicht reproduziert, ebenso wenig wie die AP-Werte der
Veröffentlichung. Betrachte sie daher als Ergebnisse der Upstream-Autoren und
nicht als Zusage, dass dieses Rezept sie erreicht. Nutze den Upstream-Zeitplan,
wenn du die Veröffentlichung reproduzieren möchtest.

Unter [Training](/docs/train) findest du Datensätze, Datenaugmentierung,
Multi-GPU und Logger.

## Validierung

`val()` gibt ein Dictionary mit den Metriknamen als Schlüsseln zurück und
druckt klassenspezifische Ergebnisse, solange `verbose` aktiviert bleibt.

<code-tabs name="val" />

Die Validierung erfolgt auf deinem eigenen Datensatz in dem Format, das du für
das Training verwendet hast. Die COCO-Validierungsschranke der Bibliothek gilt
hier nicht, weil kein COCO-Checkpoint dieser Familie für eine Messung vorhanden
ist.

## Export

Der Export wird in keinem Format unterstützt. Eine Exportanforderung löst einen
Fehler aus, statt eine Datei zu erzeugen.

Der Grund dafür ist PAQI. Es bestimmt die Query-Anzahl pro Bild aus
dichtegefilterten Vorschlägen und einer greedy dichteadaptiven
Unterdrückungsschleife. Die Ausgabelänge des Decoders hängt somit von der
Eingabe statt vom Graphen ab. Beim Tracing wird die zufällige Anzahl des
Tracing-Bildes festgeschrieben. Das erzeugte Artefakt liefert für jedes andere
Bild unbemerkt falsche Ergebnisse. Eine statische Formulierung müsste die
Unterdrückung über alle 250 bis 1500 Kandidaten ausrollen. Eine Reduzierung auf
ein festes Top-k würde genau den Recall für winzige Objekte entfernen, für den
diese Familie existiert. Wenn du einen exportierbaren Detection Transformer
benötigst, verwende [D-FINE](/docs/models/d-fine).

## Checkpoints

<checkpoint-table />

## Lizenzierung

<provenance-box>

Die sechs Mirrors behalten die Upstream-Beschränkung auf akademische Forschung bei. Der Code ist gesondert lizenziert. Das Upstream-Repository steht unter Apache-2.0, die LibreYOLO-Portierung unter MIT, und Gewichte, die du selbst mit deinen eigenen Daten trainierst, gehören dir.

</provenance-box>
