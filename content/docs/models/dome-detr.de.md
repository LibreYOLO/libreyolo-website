---
title: Dome-DETR
families:
  - domedetr
seo_title: 'Dome-DETR: Erkennung winziger Objekte in LibreYOLO'
description: >-
  Nutze Dome-DETR in LibreYOLO zur Erkennung winziger Objekte in Luft- und
  Drohnenbildern. Konvertiere die Upstream-Gewichte, sage vorher, führe
  Fine-Tuning durch und validiere mit MIT-lizenziertem Code.
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
last_verified: 1.5.0
snippets:
  predict:
    - label: Konvertieren und vorhersagen
      language: bash
      code: |
        # LibreYOLO hostet keine Dome-DETR-Gewichte, daher wird der Checkpoint
        # aus dem Upstream-Repository abgerufen und einmalig konvertiert.
        hf download RicePasteM/Dome-DETR --include 'best_ckpts_dome_2026/*' \
          --local-dir dome-ckpts

        python weights/convert_domedetr_weights.py \
          dome-ckpts/best_ckpts_dome_2026/dome-s-visdrone_converted.pth \
          LibreDOMEDETRs-visdrone.pt --size s --variant visdrone
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Ein lokaler Pfad statt eines einfachen Namens: Für diese Familie wird
        nichts geladen.

        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")

        result = model("drone-frame.jpg", save=True)


        for box in result.boxes:
            print(result.names[int(box.cls)], box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDOMEDETRs-visdrone.pt
        source=drone-frame.jpg save=True
    - label: Klassennamen
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Da es keinen COCO-Checkpoint gibt, stammen die Klassen aus dem
        Trainingsdatensatz

        # der Gewichte und werden aus den Checkpoint-Metadaten gelesen.

        aitod = LibreYOLO("LibreDOMEDETRs-aitod.pt")

        print(aitod.model.names)     # 9 AI-TOD-V2-Klassen


        visdrone = LibreYOLO("LibreDOMEDETRs-visdrone.pt")

        print(visdrone.model.names)  # 12 VisDrone-Klassen
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
source_hash: 381f01d769e7c420
---

## Installation

Dome-DETR benötigt kein optionales Zusatzpaket. Alle Importe sind in der
Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Es gibt keinen automatischen Download. LibreYOLO hostet diese Gewichte nicht.
Der Ablauf lautet daher: Rufe den Upstream-Checkpoint ab, konvertiere ihn
einmalig und lade anschließend die konvertierte Datei über ihren Pfad.
[Lizenzierung](#licensing) erklärt den Grund dafür.

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

LibreYOLO veröffentlicht für diese Familie keine Benchmark-Zeilen, weil es
keine Checkpoints für Benchmarks veröffentlicht.

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

Es gibt keine Einträge. LibreYOLO veröffentlicht keine Dome-DETR-Gewichte, und
kein Name der Form `LibreDOMEDETR<size>-<dataset>.pt` wird zu einem Download
aufgelöst.

Upstream veröffentlicht sechs Checkpoints: s, m und l für jeweils zwei
Datensätze, AI-TOD-V2 mit 9 Klassen und VisDrone mit 12. Es gibt keinen
COCO-Checkpoint. Ein kanonischer Dateiname enthält daher immer das
Datensatzsuffix, und die Klassennamen werden in den Checkpoint-Metadaten
gespeichert, statt aus einer Familienkonstante zu stammen. Eine Anfrage nach
`LibreDOMEDETRs.pt` löst sofort einen Fehler aus, der die beiden echten
Dateinamen und den Konvertierungsbefehl nennt. Sie versucht keinen Download,
der den Status 404 zurückgeben würde.

`weights/convert_domedetr_weights.py` übernimmt die Konvertierung. Das Skript
baut den LibreYOLO-Graphen neu auf, lädt die Upstream-Tensoren hinein und
schreibt nichts, wenn auch nur ein Schlüssel fehlt, unerwartet ist oder die
falsche Form hat. Eine konvertierte Datei entspricht daher entweder exakt dem
Graphen oder existiert nicht. Übergib eine Upstream-Datei mit der Endung `.pth`
sowie Größe und Variante:

```bash
python weights/convert_domedetr_weights.py \
    dome-ckpts/best_ckpts_dome_2026/aitod-s-best.pth \
    LibreDOMEDETRs-aitod.pt --size s --variant aitod
```

Zur numerischen Übereinstimmung vergleicht `weights/parity_domedetr.py` diese
Portierung über alle sechs Checkpoints mit der Upstream-Implementierung und
meldet `max_abs_diff == 0.0` sowohl für `pred_logits` als auch für `pred_boxes`.
Zuvor prüft es die MWAS-Fenstermaske Bit für Bit und vergleicht außerdem jeden
Loss-Term einzeln mit dem Upstream-Kriterium. Die Einordnung ist wichtig: Es
handelt sich um ein manuelles Skript, das den Upstream-Checkout und die
veröffentlichten Checkpoints auf dem Datenträger benötigt und von Hand
ausgeführt wird. Es ist kein Bestandteil der Continuous Integration, und kein
CI-Job reproduziert es.

## Lizenzierung

<provenance-box>

Die Gewichte sind der Grund, warum diese Familie nicht gespiegelt wird. Die
Upstream-Modellkarte enthält in ihren Metadaten kein Lizenzfeld. In ihrem Text
bezeichnet sie das Projekt als Apache-2.0, beschränkt das Material aber zugleich
auf akademische Forschungszwecke. Diese beiden Lesarten widersprechen sich,
und die strengere stellt keine Genehmigung zur Weitergabe dar. LibreYOLO
verlinkt daher bis zu einer Klärung das Upstream-Repository, statt die Dateien
zu kopieren. Dieselbe Begründung gilt hier für
[YOLO-NAS](/docs/models/yolo-nas).

Der Code ist eine separate und eindeutigere Frage. Das Upstream-Repository
steht unter Apache-2.0, die LibreYOLO-Portierung unter MIT, und Gewichte, die
du selbst mit deinen eigenen Daten trainierst, gehören dir.

</provenance-box>

## Zitieren

Dome-DETR wurde auf der ACM Multimedia 2025 unter dem Titel „Dome-DETR: DETR
with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object
Detection“ veröffentlicht. Der Preprint ist unter
[arxiv.org/abs/2505.05741](https://arxiv.org/abs/2505.05741) verfügbar. Die
Autoren veröffentlichen in ihrem Repository keinen BibTeX-Block. Daher wird
hier keiner von Hand zusammengestellt und wiedergegeben.

<citation-block />
