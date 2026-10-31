---
title: Upgrade auf 1.7.0
seo_title: LibreYOLO von 1.6.0 auf 1.7.0 aktualisieren
description: >-
  Migrationsschritte für YOLO9-Checkpoints, Modulnamen und Trainingsrezepte,
  CLI-Laufordner, die Konfusionsmatrix und neue Fehler in LibreYOLO 1.7.0.
lead: >-
  Version 1.7.0 lädt ältere YOLO9-Checkpoints weiterhin mit denselben
  Float-Boxen. Sie ändert das Training der YOLO9-Familie, die Namen, die Code im
  Netzwerk sieht, und einige Verträge von Validierung und Vorhersage.
keywords:
  - libreyolo upgrade
  - libreyolo 1.6.0 migration
  - libreyolo breaking changes
  - yolo9 modulnamen
  - clip_max_norm
  - konfusionsmatrix
last_verified: 1.7.0
source_hash: 895bcd7425f0052f
---

## 1.6.0 auf 1.7.0

- Von 1.7.0 gespeicherte Checkpoints von YOLO9, YOLO9-E2E und YOLO9-P2 verwenden neue Tensornamen und lassen sich in 1.6.x nicht laden: 1.6.x lädt die von YOLO9 und YOLO9-P2 mit einer Warnung über fehlende Schlüssel und sagt keine Boxen vorher, und bei denen von YOLO9-E2E löst es einen Fehler wegen nicht passender `model_family` aus. Aktualisiere jede Umgebung, die sie liest. Float-, quantisierte und Training-Checkpoints früherer Releases lassen sich in 1.7.0 laden, und ONNX-Exporte sind nicht betroffen.

- Code, der auf `model.model` eines Modells der YOLO9-Familie zugreift, sieht die Modulnamen von MultimediaTechLab/YOLO, ohne Aliasse. `DDetect` heißt `YOLO9Head` (`anchor_convs` / `class_convs`), `RepConvN` heißt `RepConv`, `RepNBottleneck` heißt `Bottleneck`, `DFL` heißt `Anchor2Vec` (`anchor2vec`), `nc` heißt `num_classes`, und die Unterschichten der Blöcke heißen `conv1`/`conv2`/`bottleneck`. In YOLO9-E2E heißt `YOLO9E2EDetect` jetzt `YOLO9E2EHead`, und `one2one_cv2` / `one2one_cv3` heißen `one_to_one_anchor_convs` / `one_to_one_class_convs`. `LibreYOLO9Model.fuse()`, `RepConvN.fuse_convs()`, `Conv.forward_without_bn()`, `Concat` und `yolo9.utils.decode_boxes` wurden entfernt. Das [Changelog](/docs/changelog) listet jede Umbenennung auf.

- ONNX-Knotennamen folgen in Head, Backbone und Neck den neuen Modulnamen, zum Beispiel `/head/anchor_convs.0/...`. Aktualisiere Werkzeuge, die auf Knotennamen basieren, etwa die End-Node-Konfigurationen für Hailo. Eingaben, Ausgaben und Ausgabenamen des Modells sind unverändert.

- Ein YOLO9-Fine-Tuning von Gewichten ohne PGI-Tensoren, wozu die veröffentlichten `LibreYOLO9{t,s,m,c}.pt` gehören, trainiert nur den Haupt-Head, wie in 1.5.0; 1.6.0 hängte bei `aux_weight=0.25` einen zufällig initialisierten PGI-Zweig an. Gewichte mit PGI-Tensoren und Läufe von Grund auf neu trainieren weiterhin mit PGI. Übergib `aux_weight=0.25`, um trotzdem einen neuen Zweig anzuhängen, oder `aux_weight=0`, um ihn auszuschalten.

- Ist PGI angehängt, bauen YOLO9-m und YOLO9-c den Hilfszweig von MultimediaTechLab: 12.8M (m) und 25.7M (c) Parameter, die nur im Training existieren, bei 80 Klassen, gegenüber 7.5M und 10.2M zuvor. Das PGI-Training dieser Größen braucht daher mehr Speicher und Zeit pro Schritt. Ein Training-Checkpoint von m/c aus 1.6.0 lässt sich mit dem Zweig fortsetzen und nachtrainieren, mit dem er trainiert wurde. Inferenz und YOLO9-t/s sind unverändert.

- YOLO9, YOLO9-E2E und YOLO9-P2 beschneiden die L2-Norm des Gradienten auf 10, und ihr Trainings-Assigner bewertet Anchors in fp32, sodass unter fp16-AMP gewählte positive Anchors abweichen können. Übergib `clip_max_norm=0` für das Clipping von 1.6.0; ein in 1.6.0 gestarteter und in 1.7.0 fortgesetzter Lauf beschneidet ebenfalls, sofern du es nicht übergibst. Die fp32-Zuweisung hat keinen Schalter.

- Ein `perspective` ungleich null verzerrt über die Verschiebung der Ecken, in jeder Familie, die die gemeinsame affine Transformation nutzt (YOLO9 samt E2E- und P2-Varianten, YOLOX, YOLOv7, YOLO-NAS, PP-YOLOE, RTMDet, PicoDet, RT-DETR und RT-DETRv2), auch bei Läufen, die aus einem Checkpoint von 1.6.0 fortgesetzt werden und ihn gesetzt hatten. Werte ab `0.002` ergeben dieselbe maximale Verzerrung, und negative Werte lösen `ValueError` aus. Keine Einstellung reproduziert die Verzerrung von 1.6.0; der Standard `perspective=0` bleibt unverändert.

- Die YOLO9-Mosaic-Pipeline zieht das MixUp-Verhältnis aus Beta(1, 1) statt aus Beta(32, 32), ohne Einstellung für den alten Wert. Das Standardtraining von YOLOv7 ändert sich, weil YOLOv7 MixUp standardmäßig aktiviert. YOLO9, YOLO9-E2E, YOLO9-P2, RT-DETR und RT-DETRv2 ändern sich nur, wenn MixUp aktiviert ist.

- `libreyolo train` ohne `name=` schreibt Läufe von EC, D-FINE, SegFormer und LingBot-Vision nach `runs/train/ec_exp`, `dfine_exp`, `segformer_exp` und `lingbotvision_exp` statt nach `runs/train/exp`, V-JEPA-2-Probes nach `runs/train/train` und YOLO9-P2-Läufe nach `runs/train/yolo9_exp`. Übergib `name=exp` für den alten Pfad oder `name=yolo9_p2_exp` für YOLO9-P2.

- `libreyolo train` startet D-FINE, SegFormer und LingBot-Vision mit ihren eigenen Rezepten. Bei D-FINE sind das 132 Epochen, `lr0=2e-4`, `weight_decay=1e-4`, 2 Warmup-Epochen und keine Mosaic-, MixUp-, HSV- oder affine Augmentation. Um die Werte zu senden, die 1.6.0 sendete, übergib `epochs=300 optimizer=sgd lr0=0.01 weight_decay=5e-4 warmup_epochs=5 mosaic=1.0 mixup=1.0 hsv_prob=1.0 degrees=10 translate=0.1 shear=2.0`.

- `libreyolo.validation.ConfusionMatrix` indiziert `matrix[predicted, true]` (vorher `[actual, predicted]`), zählt Erkennungen bei `conf_thres=0.25` (vorher 0.15), nimmt `names=` und `task=` nur als Schlüsselwortargumente und lässt sich in `matrix` nicht mehr neu zuweisen. Transponiere Code, der die alte Matrix indizierte, und übergib `conf_thres=0.15` für die alten Zählungen. `plots=True` schreibt die Zählungen nach `confusion_matrix.png` und das normalisierte Diagramm nach `confusion_matrix_normalized.png`.

- `predict(agnostic_nms=True)` unterdrückt überlappende Boxen klassenübergreifend, statt zu warnen. Aufgaben ohne Boxen lösen `ValueError` aus, `LibreEnsemble` und HRNet lösen `NotImplementedError` aus, und die SAM-Stufe lehnt das Schlüsselwort mit `TypeError` ab; entferne das Argument aus diesen Aufrufen.

- Eine unbekannte `interpolation` eines Klassifikators löst `ValueError` aus, auch bei den torchvision-Modi `LANCZOS`, `BOX`, `HAMMING` und `NEAREST_EXACT`. Verwende `bilinear`, `bicubic`, `nearest` oder `bilinear_no_antialias`.

- Wiederhole die Validierung, bevor du mit Zahlen aus 1.6.0 vergleichst, bei `val(single_cls=True)` auf einem mit mehreren Klassen trainierten Modell, bei Segmentierungsdatensätzen, die Zeilen nur mit Box und mit Polygon mischen, und bei RTMDet-Ins, das standardmäßig bis zu `max_det=300` Masken pro Bild behält; übergib `max_det=100` für die Einstellung von MMDetection.

- Ohne `libreyolo[rfdetr]` enden CLI-Aufrufe mit einem RF-DETR- oder DINOv2-Modellnamen mit dem JSON-Fehlercode `model_load_failed` statt `model_not_found`, und die Meldung nennt `pip install libreyolo[rfdetr]`.

Das vollständige Release findest du im [Changelog](/docs/changelog).

## 1.5.0 auf 1.6.0

- Aktualisiere Umgebungen mit einer festgelegten ONNX-Runtime-Version unter 1.18 auf `onnxruntime>=1.18.0`, bevor du das ONNX-Extra installierst.

- SAM 3D Body akzeptiert den geprüften Snapshot-Dateisatz. Installiere `libreyolo[hf]`, beschaffe den erforderlichen Zugang und verwende den automatischen Bezug oder übergib das unveränderte Snapshot-Verzeichnis und die festgelegte MHR-Datei auf einem vertrauenswürdigen lokalen Dateisystem.

- Wiederhole die Validierung nach den Skalierungskorrekturen für RF-DETR ohne Pose und den Normalisierungskorrekturen für Klassifikatoren. Prüfe die verwendeten Confidence-Schwellenwerte erneut, bevor du mit Ergebnissen aus 1.5.0 vergleichst; es gibt keinen Schalter für die bisherige Vorverarbeitung.

- D-FINE, DEIM, RT-DETRv4 und YOLO-NAS-Erkennung aktivieren FP16-AMP standardmäßig. Übergib `amp=False`, um FP32 beizubehalten.

- Für bisherige YOLO9-Fine-Tuning-Einstellungen setze `aux_weight=0`, `max_labels=100` und `warmup_momentum=0.937`. Setze `letterbox_pad="topleft"`, wenn eine neue Konvertierung mit Center-Eintrag die alte Geometrie reproduzieren muss. Alte Checkpoints mit einem Head werden mit ihrem ursprünglichen Graphen fortgesetzt. Neue YOLO9-Läufe benötigen mehr GPU-Speicher, senke also `batch`, wenn eine Batch-Größe aus 1.5.0 den Speicher überschreitet.

- RF-DETR und DINOv2 erstellen hochgezählte Laufverzeichnisse mit Familiennamen. Aktualisiere Programme, die Artefaktpfade verwenden, oder setze `output_dir="runs/train", exist_ok=True`, um den bisherigen Ort und das Wiederverwendungsverhalten beizubehalten.

- Halte bei der Klassifikation `mixup + cutmix <= 1` ein; ungültige Kombinationen lösen jetzt bei der Einrichtung einen Fehler aus.

- QAT deaktiviert EMA, SyncBatchNorm und Checkpoint-Mittelung. Verwende QAT-Best-/Last-Checkpoints, ohne dich auf diese Zustände zu verlassen.

- Eigene Loader mit Hooks für Datensatzänderungen müssen `persistent_workers=False` verwenden oder Worker nach einer Änderung neu erstellen. Inkompatible persistente Multi-Worker-Kopien lösen jetzt einen Fehler aus.

- NumPy-Bildarrays werden als BGR gelesen, in der Reihenfolge von OpenCV. Übergib `color_format="rgb"`, wo du ein RGB-Array wie `np.asarray(pil_image)` übergibst; die Ausgabe von `cv2.imread()` und Video-Frames brauchen keine Änderung.

- Ein 4D-NumPy-Array oder -Tensor ist ein Batch: `predict()` gibt eine Liste mit einem `Results` pro Bild zurück, statt nur das erste Bild zu verwenden.

- `train(resume=True)` und `train(resume="path/to/last.pt")` stellen die gespeicherten Trainingsargumente des Laufs wieder her und schreiben weiter in sein Laufverzeichnis; ausdrücklich übergebene Argumente haben Vorrang. Das Fortsetzen veröffentlichter Gewichte oder eines Laufs, der seine `epochs` bereits erreicht hat, löst `ValueError` aus.

- Bei aktivierter Validierung wird die letzte Epoche immer validiert, sodass Läufe, die kürzer als `eval_interval` sind, jetzt Metriken melden und `best.pt` schreiben. `val=False` schaltet die Validierung ab, die letzte Epoche eingeschlossen. Die Validierung während des Trainings schreibt nach `<run>/val` statt nach `runs/val/`.

- Fehler von CLI-`train` melden ihren Fehlertyp, sodass sich der Befehl bei Konfigurationsfehlern mit Code 2 statt mit Code 1 und `io_error` beendet.

Siehe das [Changelog](/docs/changelog) für das vollständige Release und [Gewichte importieren](/docs/migrate) für die Checkpoint-Konvertierung.

## 1.4.0 auf 1.5.0

Diese Seite behandelt das Upgrade von LibreYOLO selbst. Informationen zum
Laden eines Checkpoints aus einem Upstream-Projekt findest du unter
[Vorhandene Gewichte importieren](/docs/migrate). Dies ist ein anderes Thema.

Der vollständige Eintrag zum Release steht im [Changelog](/docs/changelog).
Nachfolgend werden nur die Änderungen beschrieben, die eine Anpassung durch
dich erfordern.

### Erforderliche Codeänderungen

#### `allow_experimental=True` ist nicht mehr vorhanden

Die Bestätigungssperre und der dahinterliegende Mechanismus
`ddp_aware(experimental_key=...)` wurden entfernt. Training und Export von EC,
RTMDet, PicoDet und FOMO benötigten dieses Argument zuvor. Alle Skripte, die
eine dieser Familien trainieren, sind daher betroffen.

```python
# 1.4.0
model.train(data="data.yaml", epochs=100, allow_experimental=True)

# 1.5.0: Argument löschen
model.train(data="data.yaml", epochs=100)
```

Es gibt keinen veralteten Kompatibilitätspfad. Ein Aufruf, der das Argument
weiterhin übergibt, erhält die Warnung `Unknown training config keys (ignored)`,
und das Argument bleibt wirkungslos. Gleichzeitig wurde
`BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES` entfernt. Der Hook
`get_download_notice()` bleibt bestehen und wird weiterhin von MiDaS,
SegFormer und YOLO9-P2 überschrieben.

Support-Stufen werden weiterhin veröffentlicht, sind aber kein Argument mehr.
Siehe [Stabilitätsstufen](/docs/reference/stability-tiers).

#### Die Exportstufe `"experimental"` ist nicht mehr vorhanden

```python
from libreyolo.export.support import Tier

# 1.4.0: Literal["validated", "experimental", "blocked"]
# 1.5.0: Literal["validated", "available", "blocked"]
```

Code, der anhand des Stufenstrings verzweigt, muss `"available"` verwenden,
wo zuvor `"experimental"` stand. `BaseExporter` gibt für diese Formate keinen
`RuntimeWarning` mehr aus. Der Zustand pro Format ist in der
[Exportmatrix](/docs/reference/export-matrix) aufgeführt.

#### `pretrained=False` zusammen mit `resume` wird jetzt abgelehnt

Die Kombination wurde zuvor in einem widersprüchlichen Zustand fortgesetzt.
Jetzt löst sie einen Fehler aus:

```
ValueError: pretrained=False cannot be combined with resume.
```

Wähle eine der beiden Optionen. `pretrained=False` beginnt mit einer neuen,
reproduzierbar initialisierten Konfiguration. Dies funktioniert in 1.5.0 für
jede trainierbare Familie statt nur für drei. `resume` setzt einen
unterbrochenen Lauf aus seinem Checkpoint fort. Beide werden unter
[Training](/docs/train) beschrieben.

#### CLI-`--imgsz` ist ein String und kein int

Die Änderung ist enger begrenzt, als sie klingt. Diese beiden Formen bleiben
unverändert:

```bash
libreyolo predict --model yolo9-t --source img.jpg --imgsz 640   # funktioniert weiterhin
```

```python
model.predict("img.jpg", imgsz=640)   # funktioniert weiterhin
```

Nur Code, der die Befehlsfunktionen der [CLI](/docs/cli) direkt aus Python
aufruft, muss geändert werden. `predict`, `train` und `val` haben `--imgsz` von
`int` auf `str` erweitert, damit rechteckige Größen akzeptiert werden:

```python
from libreyolo.cli.commands.predict import predict_cmd

predict_cmd(..., imgsz=640)      # 1.4.0
predict_cmd(..., imgsz="640")    # 1.5.0, auch "480x640" funktioniert jetzt
```

Der Standardwert von `train` ist jetzt der String `"640"`. `export --imgsz`
war bereits ein String. `profile` bleibt unverändert.

### Veränderte Zahlen

Drei Änderungen beeinflussen Metriken mit Standardeinstellungen. Lies diesen
Abschnitt, bevor du einen Lauf unter 1.5.0 mit einem Lauf unter 1.4.0 vergleichst.

#### faster-coco-eval ist das Standard-Backend für COCO-Metriken

`val()` und die Validierung pro Trainingsepoche berechnen COCO-Metriken jetzt
mit dem C++-Backend faster-coco-eval statt mit pycocotools.

Die Umstellung basiert auf gemessener Parität über alle 100 Test-Splits von
RF100-VL: 1381 von 1400 Metrikwerten waren bitidentisch, die maximale Abweichung
betrug 2.22e-16 und die wichtigsten Werte unterschieden sich überhaupt nicht.
Gleichzeitig war die Berechnung insgesamt 15.6x und bei Datensätzen mit vielen
Erkennungen 56x schneller. Deine Werte sollten sich nicht verändern. Sie
werden dennoch von einer anderen Implementierung erzeugt, weshalb die Änderung
hier aufgeführt ist.

pycocotools bleibt der automatische Fallback, wenn faster-coco-eval nicht
installiert ist. So erzwingst du es:

```bash
libreyolo val --model yolo9-t --data coco.yaml --no-faster-coco-eval
```

```python
model.val(data="coco.yaml", faster_coco_eval=False)
```

`LIBREYOLO_FASTER_COCO_EVAL=0` bewirkt dasselbe global. Das tatsächlich
verwendete Backend wird auf INFO-Ebene protokolliert, steht nach `val()` als
`model.last_eval_backend` zur Verfügung und ist als `eval_backend` in der
JSON-Payload der [CLI](/docs/cli/val) enthalten. Installiere den schnellen Pfad
mit `pip install libreyolo[fast-eval]`.

#### Vor 1.5.0 trainierte YOLOX-Checkpoints benötigen einen eps-Override

Dies ist der wichtigste Fallstrick des Releases. Lies ihn, wenn du
[YOLOX](/docs/models/yolox) nachtrainiert hast.

YOLOX spezifiziert für BatchNorm `eps=1e-3` und `momentum=0.03`. Bis 1.5.0
wurden diese Werte nachträglich korrigiert. Die Korrektur überstand jedoch
nicht den Neuaufbau des Heads, den `train()` durchführt, wenn sich `nc` deines
Datensatzes von dem des Checkpoints unterscheidet. Ein solches Fine-Tuning
trainierte und validierte während des Trainings mit dem torch-Standardwert
`eps=1e-5`. Nach dem erneuten Laden für die Inferenz verwendete es `1e-3`:
dieselben Tensoren unter einer anderen Normalisierung.

Größen mit normalen Convolutions verändern sich kaum. Die Depthwise-Größe `n`
verändert sich stark, weil ihre kanalweise `running_var` so klein ist, dass
eps dominiert. Auf dem RF100-VL-Split `ball` erreicht derselbe Nano-Checkpoint
bei Evaluierung mit seinem Trainings-eps **0.566** mAP50-95 und nach normalem
erneutem Laden **0.151**.

Ein vor 1.5.0 trainierter Checkpoint besitzt die Semantik von eps=1e-5. Für
treue Werte musst du entweder BN-eps bei der Evaluierung auf 1e-5 setzen:

```python
import torch
from libreyolo import LibreYOLO

model = LibreYOLO("my-yolox-finetune.pt")
for module in model.model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eps = 1e-5

model.val(data="data.yaml")
```

oder einmal `sqrt((var + 1e-3) / (var + 1e-5))` in die BN-Gewichte einrechnen
und das Ergebnis speichern. Checkpoints, die mit 1.5.0 oder neuer trainiert
wurden, benötigen keine der beiden Maßnahmen.

#### D-FINE-Multi-Scale-Training verwendet das Upstream-Rezept pro Größe

`base_size_repeat` war für jede Größe fest auf 3 gesetzt. Jetzt wird der Wert
wie im Upstream-Projekt pro Größe bestimmt: **n** trainiert bei fester Größe
mit deaktiviertem Multi-Scale, **s** verwendet 20, **m** 6, **l** 4 und **x**
3. Zuvor entsprach nur x dem Rezept. n, s, m und l sehen daher eine andere
Skalenverteilung und konvergieren zu anderen Metriken.

Setze den Wert explizit, um das alte Verhalten wiederherzustellen:

```python
from libreyolo.training.config import DFINEConfig

config = DFINEConfig(base_size_repeat=3)
```

DEIM verwendet weiterhin den festen Wert 3. Familiendetails findest du unter
[D-FINE](/docs/models/d-fine).

### Wissenswertes ohne Handlungsbedarf

- **Ergebnisse mit rechteckigem `imgsz` haben sich geändert, weil sie zuvor
  falsch waren.** Boxkoordinaten, die Größenänderung von RTMDet-Masken, die
  Rückskalierung von YOLO-NAS und die Ground-Truth-Skalierung des Validators
  verwenden jetzt getrennte Höhen- und Breitenwerte statt eines einzelnen
  Skalars. Quadratisches `imgsz` bleibt bitidentisch. Rechteckige Inferenz oder
  Validierung war unter 1.4.0 falsch skaliert. YOLO-NAS lehnt ein rechteckiges
  `imgsz` jetzt ab, statt unbemerkt falsche Ausgaben zu erzeugen.
- **Metrik-Dictionaries besitzen neue Schlüssel.** Hinzugekommen sind
  `max_det`, `ar_max_det` und `AR_max_det` des COCO-Evaluators sowie
  `metrics/loss` und `metrics/loss/ce` von FOMO. Die Standardwerte bleiben
  unverändert. Code, der über Metrikschlüssel iteriert, darunter eigene
  [Logger](/docs/train/loggers) und CSV-Header, sieht jedoch neue Spalten.
- **Reproduzierbar initialisierte YOLO9-Läufe mit Neuaufbau des Heads** beginnen
  mit einer anderen Initialisierung, weil der Seed jetzt vor statt nach dem
  Neuaufbau angewendet wird. Ein unter 1.4.0 reproduzierbar initialisiertes
  Fine-Tuning mit anderer Klassenzahl lässt sich unter 1.5.0 nicht bitgenau
  reproduzieren.
- **`libreyolo[hub-kernels]` verwendet unter CUDA jetzt tatsächlich den nativen
  MS-Deform-Attn-Kernel.** Unter 1.4.0 war er hinter einer Bedingung gesperrt,
  die RF-DETR nie erfüllte. Der Kernel wurde daher nie ausgeführt. Vorhersagen
  von RF-DETR und anderen Familien mit deformierbarer Attention können sich
  innerhalb der Float-Toleranz verschieben. Normale Installationen sind nicht
  betroffen. `LIBREYOLO_HUB_KERNELS=0` deaktiviert die Funktion.
- **`libreyolo predict` verwirft nicht unterstützte Optionen, statt einen Fehler
  auszulösen.** Die CLI filtert Keyword-Argumente anhand der `__call__`-Signatur
  des Modells. Eine von der Familie nicht akzeptierte Option wird ignoriert,
  statt `TypeError` auszulösen. Ein unbekannter Flag-Name wird weiterhin mit
  `No such option` abgelehnt.
- **Live-Quellen verändern die Form der JSON-Ausgabe.** Webcams, RTSP-Streams
  und Bildschirmaufnahmen aktivieren implizit Streaming. Dadurch wird ein
  Datensatz pro Frame statt pro Aufruf ausgegeben. Diese
  [Quellen](/docs/predict/sources) sind neu in 1.5.0. Skripte für 1.4.0 sind
  daher nicht betroffen.
- **Beim erneuten ONNX-Export von `rfdetr-pose` oder `yolonas-pose` entstehen
  andere Ausgabenamen.** 1.4.0 interpretierte ihre Pose-Heads mit mehreren
  Tensoren durch eine Heuristik anhand der Ausgabeanzahl fälschlich als
  Segmentierung. Vorhandene `.onnx`-Dateien auf dem Datenträger bleiben
  unverändert.
- **Bei einer torch-freien Installation** enthalten Results numpy-Arrays statt
  `torch.Tensor`. `.boxes.data` gibt daher einen anderen Typ zurück, und das
  Tie-Breaking der NMS kann von torchvision abweichen. Mit installiertem torch
  bleibt das Verhalten bytegenau unverändert. Siehe
  [Schlanke Installation](/docs/lightweight-install).
- **Konfigurationsobjekte validieren beim Erstellen mehr.** `TrainConfig`
  besitzt jetzt ein `__post_init__`, wo zuvor keines vorhanden war. Eine schon
  immer ungültige Konfiguration löst daher sofort einen Fehler aus, statt tief
  in einem Lauf zu scheitern. Die Serialisierung von `ValidationConfig`
  enthält jetzt einen Schlüssel `edge_thresholds`. Dadurch funktioniert ein
  strikter Roundtrip `ValidationConfig(**dump)` mit einem Dump aus 1.4.0 nicht.
- **Gewichtsdateinamen für Familien mit Aufgabensuffix werden anders aufgelöst.**
  `segformer-b0` wird jetzt zu `LibreSegformerb0-sem.pt` aufgelöst. Dies behebt
  404-Fehler bei automatischen Downloads und betrifft Skripte, die den alten
  Namen ohne Suffix fest codiert haben.
- **Der pytest-Marker `experimental_backend` heißt jetzt `extended_backend`.**
  Dies ist nur relevant, wenn du die Testsuite mit `-m` ausführst.

### Checkpoints und Datensätze

Mit 1.4.0 geschriebene Checkpoints werden unverändert geladen. Das
[Schema](/docs/reference/checkpoint-schema) ergänzt `imgsz_h` und `imgsz_w` für
rechteckige Modelle und schreibt für ältere Reader weiterhin den Skalar
`imgsz = max(h, w)`. Exporte für [ExecuTorch](/docs/export/executorch) und
[MNN](/docs/export/mnn) benötigen jetzt jeweils eine Sidecar-Datei namens
`<program>.pte.json` beziehungsweise `<model>.mnn.json`. Exporte von HRNet
enthalten `pose_input: "person_crop"`. Datensatzformate bleiben unverändert.
