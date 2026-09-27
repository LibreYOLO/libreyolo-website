---
title: Schwellenwerte und Filterung
seo_title: 'conf, iou und max_det in LibreYOLO'
description: >-
  Was conf, iou, max_det und classes bei der Vorhersage tatsächlich bewirken,
  welche Familien iou ohne NMS ignorieren und warum agnostic_nms wirkungslos
  ist.
lead: >-
  Vier Argumente bestimmen, welche Vorhersagen erhalten bleiben: conf, iou,
  max_det und classes. Nur zwei davon gelten für jede Familie, da ein
  Set-Predictor eine feste Menge von Abfragen decodiert und nie NMS ausführt.
keywords:
  - YOLO conf Schwellenwert
  - IoU Schwellenwert NMS
  - max_det
  - Klassen filtern Objekterkennung Python
  - agnostic NMS
  - NMS-freies DETR
  - Konfidenzschwellenwert Objekterkennung
  - Klassenfilter Inferenz
last_verified: 1.5.0
verification: >-
  Standardwerte aus InferenceRunner.__call__ in
  libreyolo/models/base/inference.py. Familienspezifisches NMS-Verhalten aus
  jedem Modul in libreyolo/postprocess/ gelesen und mit _is_nms_free_family in
  libreyolo/backends/base.py abgeglichen. Klassenfilterung aus
  InferenceRunner._apply_classes_filter und _wrap_results. Status von
  agnostic_nms aus NOOP_PREDICT_KWARGS in libreyolo/utils/predict_args.py.
  Open-Vocabulary-Verhalten aus NMS_THRESHOLD in
  libreyolo/models/openvocab/base.py. Validierungsstandardwerte aus
  BaseModel.val.
snippets:
  basic:
    - label: Die vier Argumente
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(
            SAMPLE_IMAGE,
            conf=0.25,      # Vorhersagen ab dieser Bewertung behalten
            iou=0.45,       # NMS-Überlappungsschwellenwert, sofern NMS ausgeführt wird
            max_det=300,    # Begrenzung je Bild
            classes=None,   # oder eine Liste von Klassen-IDs
        )
        print(len(result.boxes))
    - label: conf-Werte durchlaufen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        for conf in (0.1, 0.25, 0.5, 0.75):
            result = model(SAMPLE_IMAGE, conf=conf)
            print(conf, len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt conf=0.4 iou=0.5 max_det=100 \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  classes:
    - label: Nach bestimmten Klassen filtern
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # Klassen-IDs indizieren model.names. Bei COCO steht 0 für person.
        result = model(SAMPLE_IMAGE, classes=[0])

        print({result.names[int(c)] for c in result.boxes.cls.tolist()})
    - label: ID für einen Namen finden
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        wanted = {"person", "backpack"}
        ids = [i for i, name in result.names.items() if name in wanted]
        print(ids)

        filtered = model(SAMPLE_IMAGE, classes=ids)
        print(len(filtered.boxes))
  nmsfree:
    - label: iou bei einer Familie ohne NMS
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # RF-DETR decodiert eine feste Abfragemenge, daher ändert iou hier
        nichts.

        model = LibreYOLO("LibreRFDETRs.pt")


        loose = model(SAMPLE_IMAGE, iou=0.9)

        tight = model(SAMPLE_IMAGE, iou=0.1)


        # In beiden Fällen dieselbe Anzahl. Die wirksamen Einstellungen sind
        conf und max_det.

        print(len(loose.boxes), len(tight.boxes))
source_hash: 0b978963c356027d
---

## Die vier Argumente

| Argument | Standard | Gilt für |
|---|---|---|
| `conf` | `0.25` | Jede Familie |
| `iou` | `0.45` | Familien mit Non-Maximum Suppression |
| `max_det` | `300` | Jede Familie |
| `classes` | `None` | Jede Familie |

<code-tabs name="basic" />

Zwei davon gelten universell und zwei nicht. Dies ist die wichtigste Information, bevor du Einstellungen optimierst.

Die Validierung verwendet bewusst andere Standardwerte: `val()` wird mit `conf=0.001` und `iou=0.6` ausgeführt, da die Average Precision über eine vollständige Precision-Recall-Kurve berechnet wird und ein Grenzwert von 0,25 sie abschneiden würde.

## conf

`conf` ist der Schwellenwert, unter dem eine Vorhersage verworfen wird. Er gilt für jede Familie, einschließlich derjenigen ohne NMS. Verwende ihn zuerst, wenn es zu viele oder zu wenige Erkennungen gibt.

Der Standardwert `0.25` eignet sich zum Betrachten von Bildern. Ein nachgelagertes System benötigt meist einen höheren Wert, während eine Genauigkeitsmessung einen deutlich niedrigeren benötigt.

## iou

`iou` ist der Überlappungsschwellenwert, oberhalb dessen Non-Maximum Suppression die niedriger bewertete von zwei Boxen derselben Klasse entfernt. Er hat nur Bedeutung, wenn die Familie überhaupt eine Unterdrückung ausführt.

Ein Set-Predictor decodiert eine feste Anzahl von Abfragen und übernimmt die am höchsten bewerteten. Duplikate werden während des Trainings innerhalb der Architektur unterdrückt und nicht in einem Nachverarbeitungsschritt. Es gibt daher keinen einstellbaren Schwellenwert. Die folgenden Familien akzeptieren `iou` aus Gründen der API-Parität, ignorieren es aber:

CenterNet, DEIM, DETR, Deformable DETR, D-FINE, DINO-DETR, EdgeCrafter, Faster R-CNN, LW-DETR, Mask R-CNN, RF-DETR, RT-DETR und der End-to-End-Kopf von YOLOv9. Varianten auf Basis dieser Decoder übernehmen das Verhalten.

<code-tabs name="nmsfree" />

Die meisten weisen in den Docstrings ihrer Nachverarbeitung darauf hin. Zur Laufzeit wird jedoch keine Warnung ausgelöst. Ein Durchlauf verschiedener `iou`-Werte für RF-DETR erzeugt daher eine flache Linie statt eines Fehlers. Faster R-CNN und Mask R-CNN sind ein etwas anderer Fall: Beide haben NMS bereits innerhalb des Modells mit einem festen Upstream-Schwellenwert ausgeführt, den `iou` nicht ändern kann.

Diese Familien verwenden den Wert: YOLOv1 bis YOLOv4, YOLOv7, YOLOv9, YOLOX, YOLO-NAS, RTMDet, PicoDet, EfficientDet, FCOS, RetinaNet und SSD.

Zwei Optionen zum Vorhersagezeitpunkt machen `iou` selbst für einen Set-Predictor relevant, da beide Boxen nach Abschluss des Modells zusammenführen:

- `tiling=True` gleicht überlappende Kacheln mit klassenweisem NMS bei `iou` ab
- `augment=True` führt gespiegelte Ansichten mit klassenweisem NMS bei `iou` zusammen

Beide werden unter [Inferenzleistung](/docs/predict/performance) behandelt.

Für Open-Vocabulary-Detektoren gilt eine eigene Regel. Eine Familie, deren Prozessor NMS ausführt, definiert einen eigenen Standardschwellenwert und berücksichtigt `iou`. Dies trifft auf OMDet-Turbo zu. Familien ohne Unterdrückung, also Grounding DINO, OWLv2 und OV-DEIM, geben bei Übergabe von `iou` eine Warnung aus. Diese Warnung ist einzigartig in der Bibliothek.

## max_det

`max_det` begrenzt die Anzahl der Vorhersagen je Bild. Es gilt überall, wird jedoch über unterschiedliche Mechanismen umgesetzt. Eine NMS-Familie kürzt nach der Unterdrückung, ein Set-Predictor verwendet den Wert als Größe seiner Top-k-Auswahl.

Einige Familien begrenzen unterhalb des angeforderten Werts, da dies ihrer Upstream-Referenzkonfiguration entspricht. SSD begrenzt auf 200, RTMDet bei der Instanzsegmentierung auf 100 und FCOS auf sein eigenes Erkennungslimit je Bild. Eine Erhöhung von `max_det` darüber hinaus hat keine Wirkung.

Nur bei der Kachelinferenz wird `max_det` zentral statt familienbezogen angewendet. Dort wird die zusammengeführte Liste nach dem Abgleich der Kacheln gekürzt.

## Klassenfilterung

<code-tabs name="classes" />

`classes` akzeptiert eine Liste von Klassen-IDs und behält nur Vorhersagen, deren Klasse darin enthalten ist. Die IDs indizieren `result.names`. Am sichersten liest du die Namen aus einem Ergebnis, statt eine Datensatzreihenfolge anzunehmen.

Die Filterung erfolgt zentral nach der Nachverarbeitung jeder Familie in dem gemeinsamen Pfad, den jede Vorhersage durchläuft. Das hat zwei wichtige Folgen. Sie funktioniert bei jeder Familie, einschließlich derjenigen ohne NMS. Außerdem filtert sie die an Boxen ausgerichteten Nutzlasten, sodass Masken, Keypoints und orientierte Boxen gemeinsam gekürzt werden und nicht unpassend zurückbleiben.

In der Befehlszeile akzeptiert `classes` eine einzelne Ganzzahl, eine Liste oder einen durch Kommas getrennten String:

```bash
libreyolo predict model=LibreYOLO9s.pt classes=0 source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
libreyolo predict model=LibreYOLO9s.pt classes="[0,2,5]" source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
```

Filterung erzeugt keine kostenlose Genauigkeit. Das Modell verwendet weiterhin seine Kapazität für die Vorhersage von Klassen, die du anschließend verwirfst. `max_det` wird von der Familie vor dem Filter angewendet. Ein Bild mit vielen unerwünschten Klassen kann daher das Limit erreichen, bevor deine Klasse berücksichtigt wird. Senke in diesem Fall `conf` oder erhöhe `max_det`.

## agnostic_nms

`agnostic_nms` wird akzeptiert, hat aber keine Wirkung. Bei der Übergabe erscheint eine Warnung, dass es zur Kompatibilität mit der Befehlszeile wirkungslos ist. Anschließend wird das Argument verworfen.

Es gibt keinen klassenunabhängigen Unterdrückungsmodus. Jeder NMS-Aufruf in der Bibliothek ist klassenbewusst. Zwei überlappende Boxen unterschiedlicher Klassen bleiben daher bei jedem Wert von `iou` erhalten. Wenn dies problematisch ist, filtere zuerst mit `classes` oder unterdrücke klassenübergreifend direkt auf `result.boxes`.

## Von predict abgelehnte Argumente

Zwei Argumente lösen statt einer Warnung einen Fehler aus: `visualize` und `embed` lösen beide `NotImplementedError` aus. Für Embeddings lädst du das Modell mit `task="embed"` und rufst `predict` oder `embed` wie gewohnt auf.

Jedes unbekannte Argument löst `TypeError` mit den unterstützten Optionen aus. Ein Tippfehler schlägt dadurch sofort fehl, statt unbemerkt ignoriert zu werden.

Die folgenden Argumente werden akzeptiert, mit einer Warnung versehen und verworfen: `agnostic_nms`, `boxes`, `dnn`, `half`, `line_width`, `retina_masks`, `show_conf`, `show_labels` und `verbose`.
