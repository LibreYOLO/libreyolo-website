---
title: Upgrade auf 1.5.0
seo_title: LibreYOLO 1.4.0 auf 1.5.0 aktualisieren
description: >-
  Die vier erforderlichen Codeänderungen in 1.5.0, drei Änderungen mit
  Auswirkungen auf Metriken und kleinere Verhaltensänderungen, die du vor dem
  Vergleich von Läufen kennen solltest.
lead: >-
  Aus der öffentlichen Modell-API wurde nichts entfernt: Jede Klasse und
  Funktion, die in 1.4.0 funktionierte, kann weiterhin importiert werden. Vier
  Argumente haben ihre Form geändert, und drei Standardwerte beeinflussen
  möglicherweise deine Vergleichswerte.
keywords:
  - libreyolo upgrade
  - libreyolo 1.5.0 migration
  - allow_experimental entfernt
  - libreyolo breaking changes
  - yolox bn eps
  - faster-coco-eval standard
last_verified: 1.5.0
meta:
  - label: Gilt für
    value: 1.4.0 bis 1.5.0
  - label: Erforderliche Codeänderungen
    value: 'Vier, alle eng begrenzt'
  - label: Veränderte Ergebnisse
    value: 'COCO-Backend, YOLOX-BN-eps, D-FINE-Multi-Scale'
  - label: Entfernte öffentliche APIs
    value: Keine
source_hash: ab38d8ef7b53f596
---

Diese Seite behandelt das Upgrade von LibreYOLO selbst. Informationen zum
Laden eines Checkpoints aus einem Upstream-Projekt findest du unter
[Vorhandene Gewichte importieren](/docs/migrate). Dies ist ein anderes Thema.

Der vollständige Eintrag zum Release steht im [Changelog](/docs/changelog).
Nachfolgend werden nur die Änderungen beschrieben, die eine Anpassung durch
dich erfordern.

## Erforderliche Codeänderungen

### `allow_experimental=True` ist nicht mehr vorhanden

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
weiterhin übergibt, löst `TypeError` aus. Gleichzeitig wurde
`BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES` entfernt. Der Hook
`get_download_notice()` bleibt bestehen und wird weiterhin von MiDaS,
SegFormer und YOLO9-P2 überschrieben.

Support-Stufen werden weiterhin veröffentlicht, sind aber kein Argument mehr.
Siehe [Stabilitätsstufen](/docs/reference/stability-tiers).

### Die Exportstufe `"experimental"` ist nicht mehr vorhanden

```python
from libreyolo.export.support import Tier

# 1.4.0: Literal["validated", "experimental", "blocked"]
# 1.5.0: Literal["validated", "available", "blocked"]
```

Code, der anhand des Stufenstrings verzweigt, muss `"available"` verwenden,
wo zuvor `"experimental"` stand. `BaseExporter` gibt für diese Formate keinen
`RuntimeWarning` mehr aus. Der Zustand pro Format ist in der
[Exportmatrix](/docs/reference/export-matrix) aufgeführt.

### `pretrained=False` zusammen mit `resume` wird jetzt abgelehnt

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

### CLI-`--imgsz` ist ein String und kein int

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

## Veränderte Zahlen

Drei Änderungen beeinflussen Metriken mit Standardeinstellungen. Lies diesen
Abschnitt, bevor du einen Lauf unter 1.5.0 mit einem Lauf unter 1.4.0 vergleichst.

### faster-coco-eval ist das Standard-Backend für COCO-Metriken

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

### Vor 1.5.0 trainierte YOLOX-Checkpoints benötigen einen eps-Override

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
from libreyolo import LibreYOLOX

model = LibreYOLOX("my-yolox-finetune.pt")
for module in model.model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eps = 1e-5

model.val(data="data.yaml")
```

oder einmal `sqrt((var + 1e-3) / (var + 1e-5))` in die BN-Gewichte einrechnen
und das Ergebnis speichern. Checkpoints, die mit 1.5.0 oder neuer trainiert
wurden, benötigen keine der beiden Maßnahmen.

### D-FINE-Multi-Scale-Training verwendet das Upstream-Rezept pro Größe

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

## Wissenswertes ohne Handlungsbedarf

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
  statt `TypeError` auszulösen. Ein Tippfehler in einem Flag-Namen wird jetzt
  unbemerkt ignoriert.
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

## Checkpoints und Datensätze

Mit 1.4.0 geschriebene Checkpoints werden unverändert geladen. Das
[Schema](/docs/reference/checkpoint-schema) ergänzt `imgsz_h` und `imgsz_w` für
rechteckige Modelle und schreibt für ältere Reader weiterhin den Skalar
`imgsz = max(h, w)`. Exporte für [ExecuTorch](/docs/export/executorch) und
[MNN](/docs/export/mnn) benötigen jetzt jeweils eine Sidecar-Datei namens
`<program>.pte.json` beziehungsweise `<model>.mnn.json`. Exporte von HRNet
enthalten `pose_input: "person_crop"`. Datensatzformate bleiben unverändert.

