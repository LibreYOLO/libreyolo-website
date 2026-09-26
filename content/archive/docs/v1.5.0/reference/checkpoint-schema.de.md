---
title: Checkpoint-Schema
seo_title: LibreYOLO-Schema v1.0 für Checkpoint-Metadaten
description: >-
  Metadaten jedes LibreYOLO-.pt-Checkpoints: erforderliche Schlüssel,
  aufgabenspezifische Ergänzungen, Export-Laufzeitschlüssel,
  Quantisierungsmanifeste und Trainingsfelder.
lead: >-
  Eine LibreYOLO-.pt-Datei ist ein mit torch.save gespeichertes flaches
  Dictionary. Der Schlüssel model enthält das State-Dictionary. Die übrigen
  Schlüssel auf oberster Ebene sind Metadaten, die den Checkpoint ohne Analyse
  des Dateinamens oder State-Dictionarys identifizieren.
keywords:
  - LibreYOLO Checkpoint Schema
  - schema_version 1.0
  - model_family
  - LibreYOLO Checkpoint Metadaten
  - Quantisierungsmanifest
  - wrap_libreyolo_checkpoint
last_verified: 1.5.0
verification: >-
  Entspricht docs/checkpoint_schema.md im LibreYOLO-Repository in v1.5.0,
  abgeglichen mit libreyolo/utils/serialization.py und BaseModel.save.
snippets:
  usage:
    - label: Metadaten eines Checkpoints lesen
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.utils.serialization import unwrap_libreyolo_checkpoint

        import torch


        # Einen Checkpoint herunterladen und erneut speichern, damit ein lokaler
        Pfad vorhanden ist.

        LibreYOLO("LibreYOLO9t.pt").save("roundtrip.pt")


        loaded = torch.load("roundtrip.pt", map_location="cpu",
        weights_only=False)

        state_dict, metadata = unwrap_libreyolo_checkpoint(loaded)


        print(metadata["schema_version"], metadata["model_family"])

        print(metadata["size"], metadata["task"], metadata["nc"],
        metadata["imgsz"])

        print(len(state_dict), "tensors")
source_hash: ce760f1bed97bfd0
---

## Schema v1.0

Jeder offizielle LibreYOLO-Checkpoint mit der Endung `.pt` enthält:

```python
{
    "model": state_dict,
    "schema_version": "1.0",
    "libreyolo_version": "0.x.y",
    "model_family": "yolo9",
    "size": "t",
    "task": "detect",
    "nc": 80,
    "names": {0: "cat", 1: "dog"},
    "imgsz": 640,
}
```

| Schlüssel | Typ | Bedeutung |
|---|---|---|
| `model` | State-Dictionary | Modellgewichte |
| `schema_version` | str | Version des Metadatenvertrags; v1.0 verwendet den String `"1.0"` |
| `libreyolo_version` | str | Version, die den Checkpoint erzeugt hat |
| `model_family` | str | Registrierte Familie wie `yolo9`, `rfdetr`, `dfine`, `ec` |
| `size` | str | Variante innerhalb der Familie wie `t`, `s`, `r18`, `atto` |
| `task` | str | Kanonischer Aufgabenname |
| `nc` | int | Positive Klassenanzahl |
| `names` | dict | `dict[int, str]` mit Schlüsseln in `0..nc-1` |
| `imgsz` | int | Positive quadratische Eingabeauflösung oder veralteter Skalar für einen rechteckigen Vertrag |

`task` ist einer der Werte `detect`, `segment`, `semantic`, `panoptic`, `pose`, `classify`, `gaze`, `obb`, `point`, `depth`, `edge`, `normal`, `restore`, `matte`, `ocr`, `embed` oder `mesh`.

Offizielle Checkpoints schreiben jeden Schlüssel von `names`. Leser dürfen fehlende Schlüssel aus veralteten lückenhaften Zuordnungen mit Labels `class_i` auffüllen. Schlüssel außerhalb des Bereichs sind ungültig.

Rechteckige Checkpoints behalten für ältere Leser einen skalaren Wert `imgsz`, der auf `max(imgsz_h, imgsz_w)` gesetzt wird. Zusätzlich schreiben sie die tatsächlichen Dimensionen als `imgsz_h` und `imgsz_w`. Ein Leser mit Unterstützung für rechteckige Felder muss diese gegenüber dem Skalar bevorzugen. Familien mit festem rechteckigem Vertrag wie HRNet Pose lehnen inkompatible Laufzeitgrößen ab.

Das Schema ist bewusst flach und `model` bewusst ein State-Dictionary.

<code-tabs name="usage" />

## Ergänzungen für Posenschätzung

Posenschätzung ist üblicherweise einklassig mit `nc: 1` und `person`. Der Pose-Kopf von YOLO-NAS unterstützt jedoch auch mehrklassige Posenschätzung mit einem gemeinsamen Keypoint-Skelett. In diesem Fall beschreiben `nc` und `names` die Klassen wie bei der Objekterkennung. Pose-Exporte für Laufzeitumgebungen geben `scores` mit der Form `[batch, anchors, nc]` aus.

| Schlüssel | Bedeutung |
|---|---|
| `num_keypoints` | Positive Anzahl der vom Pose-Kopf verwendeten Keypoints |
| `keypoint_dim` | `2` für Labels aus `x,y` oder `3` für Labels aus `x,y,visibility`; Modellausgaben stellen immer `x,y,visibility` bereit |
| `oks_sigmas` | Optionale OKS-Sigmas je Keypoint; bei Auslassung wird der Aufgabenstandard für `num_keypoints` verwendet |
| `num_keypoints_per_class` | Optionale Keypoint-Anzahl je Klasse für GroupPose-artige Köpfe mit nach Klasse aufgefülltem Keypoint-Tensor; `0` für Klassen ohne Keypoints |

## Ergänzungen für Meshes

Mesh-Checkpoints verwenden `task: "mesh"`, `nc: 1` und `names: {0: "person"}`. Die Parameterlayouts unterscheiden sich nach Körpermodell. Deshalb werden Dimensionen aufgezeichnet statt angenommen.

| Schlüssel | Bedeutung |
|---|---|
| `body_model` | Parametrisierung wie `mhr`; erforderlich und zur Interpretation aller folgenden Felder verwendet |
| `num_betas` | Anzahl der Identitäts- und Formkoeffizienten; 45 bei MHR |
| `num_body_pose` | Breite des Körperpose-Parameterblocks; 130 bei MHR. Ein flacher Vektor statt eines Tripels je Gelenk, da Rig-Gelenke unterschiedliche Freiheitsgrade besitzen |
| `num_vertices` | Vom Decoder ausgegebene Eckpunktanzahl; 18439 bei MHR |
| `num_joints` | Vom Decoder ausgegebene Gelenkanzahl; 127 bei MHR |
| `rotation_format` | Codierung der Drehungen, beispielsweise `euler_zyx` bei MHR oder `axis_angle`. Wird nie aus der Tensorform abgeleitet, da ein Dreiervektor mehrdeutig ist |

## Platzhalter für dichte Aufgaben

Mehrere Aufgaben sagen dichte Karten statt Klassen vorher. Klassenartige Slots bestehen dort nur zur Schemakompatibilität.

| Aufgabe | `nc` | `names` |
|---|---|---|
| `depth` | 1 | `{0: "depth"}` |
| `edge` | 1 | `{0: "edge"}` |
| `restore` | 1 | `{0: "image"}` |
| `ocr` | 1 | `{0: "text"}` |

Kantenvorhersagen sind dichte float32-Wahrscheinlichkeitskarten in `[0, 1]`.

Restauration-Checkpoints dürfen `degradation` als kurze Beeinträchtigungsbezeichnung wie `deblur`, `denoise` oder `super-resolution`, `dataset` als Herkunftsbezeichnung wie `GoPro` oder `SIDD` sowie `scale` als positiven ganzzahligen Hochskalierungsfaktor zwischen Ausgabe und Eingabe ergänzen, beispielsweise `4` für ein Modell mit 4-facher Super-Resolution. Bei Auslassung oder `1` behält das restaurierte Bild die Eingabeauflösung. Die Laufzeit leitet die Skalierung auch aus Familie und Größe ab. `scale` ist daher eine Herkunftsmetadatenangabe und keine Voraussetzung beim Laden.

## Ergänzungen für OCR

Die Familie `ppocr` stellt einen zusammengesetzten Checkpoint je Stufe bereit. Dessen State-Dictionary unter `model` enthält zwei Teilmodelle in den Schlüsselnamensräumen `det.*` und `rec.*`.

| Schlüssel | Bedeutung |
|---|---|
| `charset` | Vollständiges CTC-Alphabet in Reihenfolge der Ausgabeindizes: Index 0 ist das CTC-Blank, danach folgen Erkennungs-Dictionary und Leerzeichen. Loader müssen es aus dem Checkpoint und nie aus einer Nebendatei lesen |
| `pipeline` | Bei der Konvertierung eingebettete Pipeline-Standardwerte: `det_limit_side_len`, `det_db_thresh`, `det_db_box_thresh`, `det_db_unclip_ratio`, `rec_image_shape`. Laufzeitargumente dürfen sie je Aufruf überschreiben |
| `components` | Reserviert für optionale Pipeline-Stufen wie Dokumentorientierung, Entzerrung und Textzeilendrehung. In v1 leer |

## Exportmetadaten für Laufzeitumgebungen

Exportierte Artefakte verwenden dieselbe doppelte Schreibweise für Rechtecke. `imgsz_h` und `imgsz_w` werden neben dem veralteten Skalar `imgsz` geschrieben. Ein Leser ohne Verständnis der rechteckigen Felder darf den Skalar nicht unbemerkt als quadratischen Vertrag behandeln.

Rechteckige Laufzeitunterstützung ist familien- und formatbezogen. Exporte der YOLO9-Familie, von HRNet, NAFNet und Real-ESRGAN dürfen in unterstützten Formaten nicht quadratische `imgsz_h` und `imgsz_w` verwenden. Familien oder Formate ohne ausdrückliche Rechteckunterstützung lehnen die Metadaten ab, statt solche Artefakte quadratisch vorzuverarbeiten. HRNet-Exporte sind feste FP32-Personenausschnitt-Köpfe mit Batchgröße 1. W32 akzeptiert 256x192, W48 akzeptiert 384x288. Der Personendetektor ist nicht in den Graphen eingebettet.

Exporte mit eingebettetem NMS dürfen diese flachen Schlüssel ergänzen:

| Schlüssel | Bedeutung |
|---|---|
| `nms` | Boolescher String; `"true"` bedeutet, dass der Graph eine eingebettete Nachverarbeitungsausgabe enthält |
| `nms_conf` | In die eingebettete Ausgabe übernommener Konfidenzschwellenwert |
| `nms_iou` | In die eingebettete Ausgabe übernommener IoU-Schwellenwert |
| `max_det` | Maximale Anzahl von Erkennungszeilen nach NMS in der eingebetteten Ausgabe |
| `nms_raw_output` | Boolescher String; `"true"` bedeutet, dass der Graph zusätzlich eine rohe Detektorausgabe bereitstellt |

Bei ONNX-Exporten für die YOLO9-Objekterkennung mit `nms=true` ist Ausgabe `0` namens `output` der eigenständige Tensor nach NMS mit den Schwellenwerten des Exports. Bei `nms_raw_output=true` ist Ausgabe `1` namens `raw` für LibreYOLO-Backends reserviert. Damit wenden sie natives Beschneiden auf dem Original-Canvas und die Laufzeitsemantik von `predict(conf=..., iou=..., max_det=...)` an. Drittanbieter sollten die erste Ausgabe verwenden.

Pose-Exporte dürfen `num_keypoints`, `keypoint_dim`, `num_keypoints_per_class` und `pose_input` ergänzen. Bei rohen GroupPose-artigen Exporten kann `keypoint_dim` größere Werte wie `8` verwenden, wenn der Tensor Felder für Genauigkeit oder Klassen-Logits enthält. `num_keypoints_per_class` ist eine JSON-codierte Liste. Klassenslots ohne Keypoints müssen erhalten bleiben, da sie das Schema definieren. Bei `pose_input` bedeutet `"person_crop"`, dass der Graph einen bereits extrahierten Ausschnitt verarbeitet und keinen Detektor enthält. HRNet-Laufzeitexporte benötigen diesen Wert.

Klassifizierungsexporte dürfen `crop_pct` als Gleitkommawert für den mittigen Beschnitt ergänzen. Das Skalierungsziel vor dem Beschnitt lautet `round(imgsz / crop_pct)`. Bei Auslassung ist der Standard `0.875`. `interpolation` ist `"bilinear"` oder `"bicubic"` und standardmäßig `"bilinear"`.

ExecuTorch-Exporte schreiben die flachen Metadaten in die erforderliche Nebendatei `<program>.pte.json`. Der v1-Vertrag lautet CPU, FP32, Batchgröße 1 und fester Eingabe-Canvas. Zusätzlich benötigt er `executorch_version`, `executorch_delegate` gleich `"xnnpack"` und eine positive Zahl `executorch_delegate_partitions`. Der Loader lehnt Nebendateien ab, die einen anderen Delegate, dynamische Formen oder eine andere Genauigkeit als FP32 angeben.

MNN-Exporte schreiben die flachen Metadaten in die erforderliche Nebendatei `<model>.mnn.json`. Der v1-Vertrag lautet CPU, FP32, ausschließlich Objekterkennung und feste NCHW-Eingabeform. Zusätzlich benötigt er `mnn_version`, `mnn_backend` gleich `"cpu"`, geordnete nicht leere `mnn_input_names` und `mnn_output_names`, `mnn_input_shape` als vier positive Ganzzahlen in der Reihenfolge `[batch, channels, height, width]` sowie `mnn_batch` gleich `mnn_input_shape[0]`. Der Loader lehnt dynamische Metadaten, andere Genauigkeit als FP32, andere Aufgaben als Objekterkennung, nicht unterstützte Familien oder inkonsistente Formen ab.

Eine `.pte`- oder `.mnn`-Datei ist ein Backend-spezifisches Artefakt und kein PyTorch-Checkpoint.

## Quantisierte Checkpoints

Ein quantisiertes Modell ergänzt den optionalen flachen Schlüssel `quant`. Er enthält ein Manifest-Dictionary mit `schema`, `recipe`, `keep_high_precision`, `execution`, Kalibrierungsherkunft, `module_count` und `state`. FP8-Manifeste dürfen außerdem `fp8_tensorwise_weights` enthalten, die genaue Liste der `QuantLinear`-Modulnamen, deren Gewichtsskalierung tensorweise statt je Ausgabekanal erfolgt. Ein Loader, der `quant` vorfindet, baut die quantisierte Modulstruktur und Skalierungsrichtlinie vor `load_state_dict` neu auf.

`state` unterscheidet die beiden Artefaktformen.

`"prepared"` ist der Standard, enthält FP32-Mastergewichte sowie `_q_*`-Skalenpuffer und ist trainierbar. Ein Leser ohne Quantisierungsunterstützung darf den Schlüssel `quant` ignorieren und die Mastergewichte als Gleitkommamodell laden.

`"finalized"` ist die von `export(format="pt")` geschriebene Deployment-Form. Mastergewichte werden entfernt und jedes quantisierte Modul enthält stattdessen gepackte Gewichte:

| Rezept | Gepackte Tensoren | Dequantisierung |
|---|---|---|
| int8 | `weight_packed` als int8 in ursprünglicher Gewichtsform, `_q_w_scale` als FP32 je Kanal | `weight_packed * scale` |
| fp8 | `weight_packed` als float8_e4m3fn in ursprünglicher Form, `_q_w_scale` als FP32 mit einem Eintrag je Ausgabekanal | `weight_packed * scale` |
| w4a16, w4a8 | `weight_packed` als uint8, zwei 4-Bit-Codes je Byte, niedriges Nibble zuerst, Code `q + 8`; `_q_w_gscale` als FP32 `[out, ngroups]`, Gruppe 128 entlang in_features | Gruppenweise Skalierung |
| int2 | Vier 2-Bit-Codes je Byte, Code `q + 2`, Gruppe 64 | Gruppenweise Skalierung |
| nvfp4 | `weight_packed` als uint8 `[out, ceil(in/16)*8]`, Code `sign<<3 \| E2M1 level`; `weight_block_scale` als float8_e4m3fn `[out, ceil(in/16)]`; `_q_w_amax` als FP32 je Tensor | `block_scale * amax / (448 * 6)` |
| mxfp4 | Wie nvfp4, aber mit Blöcken aus 32 Elementen und zusätzlichem `weight_block_exp` als int8 `[out, ceil(in/32)]` | `2 ** exponent` |

Aktivierungsbereichspuffer `_q_act_lo`, `_q_act_hi` und `_q_calibrated` bleiben bei int8 erhalten. Das Manifest zeichnet für die nicht quantisierten Tensoren unter `remainder` `"fp16"` oder `"fp32"` auf. Das Entpacken reproduziert die Simulation bitgenau. Die finalisierte Inferenz entspricht auf dem finalisierenden Gerät daher exakt der vorbereiteten Inferenz. Dieses Layout ist der stabile Vertrag für externe Exporter und Laufzeitumgebungen.

## Trainings-Checkpoints

Trainer-Checkpoints verwenden denselben erforderlichen Metadatenkern und dürfen flache Felder für Training und Wiederaufnahme ergänzen:

```python
{
    "model": state_dict,
    "epoch": 42,
    "optimizer": optimizer_state_dict,
    "config": {},
    "loss": 1.23,
    "best_metric_key": "metrics/mAP50-95",
    "best_metric_value": 0.51,
    "best_epoch": 39,
    "is_ema_weights": True,
    "train_model": raw_state_dict,
    "ema": ema_state_dict,
    "ema_updates": 12345,
}
```

`is_ema_weights` gibt an, ob `model` auf oberster Ebene mit EMA geglättet ist. Bei aktiviertem EMA bewahren `train_model`, `ema` und `ema_updates` den Zustand zur Wiederaufnahme. Veröffentlichte Inferenzgewichte sollten schlank sein und weder Optimierer, Epoche, Konfiguration, Loss noch EMA-Wiederaufnahmezustand enthalten, sofern sie nicht bewusst als Trainings-Checkpoints verteilt werden.

Aus Kompatibilitätsgründen akzeptieren Leser die veralteten Aliasse `best_mAP50_95`, `best_mAP50`, `best_metric` und `best_metric_name` für die beste Metrik.

## Externe Snapshots

Das Schema gilt für von LibreYOLO erstellte `.pt`-Dateien. Es benennt mehrteilige Upstream-Snapshots der separaten Modellstufen weder um noch bindet es sie ein.

LibreMODUS in der Größe `14b-a7b` ist eine ausdrückliche Ausnahme. Der Alias wird über `LibreVLM(...)` zu einem Verzeichnis festgeschriebener Upstream-Dateien aufgelöst. LibreYOLO ergänzt weder v1.0-Metadaten noch veröffentlicht es die Dateien erneut als `.pt`.

## Veraltete und fremde Gewichte

Neue Writer prüfen streng und müssen v1.0-Metadaten ausgeben. Bei fehlenden oder unvollständigen Metadaten werden ältere LibreYOLO-artige Checkpoints über den Kompatibilitätspfad mit Warnung und Konvertierungsanweisungen geladen. Fremde Upstream-Checkpoints werden zur automatischen Konvertierung weitergeleitet. Weitere Informationen findest du unter [Upstream-Checkpoints](/docs/reference/upstream-checkpoints).

## Hilfsfunktionen

Die Schema-Hilfsfunktionen befinden sich in `libreyolo.utils.serialization`:

```python
wrap_libreyolo_checkpoint(
    state_dict,
    *,
    model_family,
    size,
    task,
    nc,
    names=None,
    imgsz=None,
    libreyolo_version=None,
    schema_version="1.0",
    **extra_metadata,
) -> dict

validate_checkpoint_metadata(checkpoint, *, strict=False) -> list[str]

unwrap_libreyolo_checkpoint(loaded, *, strict=False) -> tuple[dict, dict]
```

`validate_checkpoint_metadata` verändert nichts und gibt eine Fehlerliste zurück. Mit `strict=True` löst es stattdessen `CheckpointMetadataError` aus. `model.save(path)` ist der unterstützte Weg zum Schreiben eines konformen Checkpoints.
