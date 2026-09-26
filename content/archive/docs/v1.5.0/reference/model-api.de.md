---
title: Modell-API
seo_title: Methoden und Signaturen des LibreYOLO-Modellobjekts
description: >-
  Alle Methoden eines geladenen LibreYOLO-Modells: predict, embed, track, val,
  train, export, save, quantize, info und die CUDA-Graph-Steuerung mit
  tatsächlichen Standardwerten.
lead: >-
  Ein geladenes LibreYOLO-Modell ist eine Instanz von BaseModel. Diese Seite
  führt die Methoden dieser Instanz mit den aus libreyolo/models/base/model.py
  gelesenen Signaturen und Standardwerten auf.
keywords:
  - LibreYOLO Modellmethoden
  - LibreYOLO predict Argumente
  - LibreYOLO val Argumente
  - LibreYOLO export Argumente
  - model.track
  - model.quantize
  - capture_graph
last_verified: 1.5.0
verification: >-
  Signaturen und Standardwerte aus libreyolo/models/base/model.py und
  libreyolo/models/base/inference.py in v1.5.0 gelesen. Familienklassen können
  diese einschränken oder erweitern. train() wird je Familie definiert, daher
  ist hier nur sein gemeinsamer Wrapper cfg= dokumentiert.
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        model.info()
        result = model(SAMPLE_IMAGE, conf=0.25, iou=0.45)

        print(result.boxes.xyxy)
        print(result.speed)
  stream:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9t.pt")


        # stream=True gibt einen Generator mit einem Results-Objekt je Frame
        oder Bild zurück.

        for result in model([SAMPLE_IMAGE, SAMPLE_IMAGE], stream=True):
            print(len(result))
source_hash: da0776970ded8716
---

## Erstellung

Die Factory gibt eine Instanz der Familienklasse zurück. Bei direkter Erstellung akzeptiert diese Klasse dieselben Argumente, allerdings ist `size` erforderlich:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`device="auto"` wählt zuerst verfügbares CUDA, dann MPS und schließlich CPU. Eine Ganzzahl oder Ziffernfolge wird als CUDA-Index gelesen. `device=0` und `device="0"` bedeuten daher beide `cuda:0`. `task` wird gegen `SUPPORTED_TASKS` der Familie geprüft. `model_path=None` erstellt die Architektur und belässt sie im Trainingsmodus. Ein `dict` lädt dieses State-Dictionary direkt.

## predict und \_\_call\_\_

`predict` ist ein Alias für `__call__`.

```python
model(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    output_path=None,
    color_format="auto",
    tiling=False,
    overlap_ratio=0.2,
    output_file_format=None,
    cuda_graph=False,
    **kwargs,
)
```

| Argument | Standard | Bedeutung |
|---|---|---|
| `source` | `None` | Bild, Liste oder Tupel von Bildern im Arbeitsspeicher, Verzeichnis, Videodatei oder Bildschirmquelle wie `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` |
| `conf` | `0.25` | Konfidenzschwellenwert |
| `iou` | `0.45` | IoU-Schwellenwert für NMS |
| `imgsz` | `None` | Überschreibung der Eingabegröße; `None` verwendet die native Modellgröße |
| `device` | `None` | Geräteüberschreibung für diesen Aufruf |
| `classes` | `None` | Nur diese Klassen-IDs behalten |
| `max_det` | `300` | Maximale Erkennungen je Bild |
| `augment` | `False` | Test-Time Augmentation |
| `save` | `False` | Annotiertes Bild oder Video schreiben |
| `batch` | `1` | Bilder je Vorwärtsdurchlauf für Verzeichnis- und Listenquellen |
| `stream` | `False` | Generator statt einer aufgebauten Liste zurückgeben |
| `stream_buffer` | `False` | Alle aufgenommenen Live-Frames statt nur des neuesten behalten |
| `vid_stride` | `1` | Jeden N-ten Video- oder Bildschirm-Frame verarbeiten |
| `show` | `False` | Annotierte Frames in einem Fenster anzeigen |
| `output_path` | `None` | Ausgabepfad bei `save=True` |
| `color_format` | `"auto"` | Hinweis zum Farbformat von Arrays im Arbeitsspeicher |
| `tiling` | `False` | Kachelinferenz für große Bilder |
| `overlap_ratio` | `0.2` | Überschneidungsanteil der Kacheln |
| `output_file_format` | `None` | `"jpg"`, `"png"` oder `"webp"` |
| `cuda_graph` | `False` | `True` zeichnet bei der ersten Verwendung jeder Eingabeform auf, `"auto"` wartet auf eine Wiederholung |

Eine einzelne Bildquelle gibt ein `Results`-Objekt zurück. Eine Liste, ein Tupel oder ein Verzeichnis gibt eine Liste solcher Objekte zurück. `stream=True` liefert in jedem Fall einen Generator.

Live-Stream-Quellen sind unbegrenzt und benötigen `stream=True`. `tiling` und `augment` lassen sich nicht kombinieren. Test-Time Augmentation löst bei den Aufgaben `embed`, `point` und `edge` einen Fehler aus.

<code-tabs name="usage" />

Mit `batch > 1` führen Familien mit wahrem `SUPPORTS_BATCHED_PREDICT` je Gruppe einen gestapelten Vorwärtsdurchlauf aus. `batch=1` behält einen Vorwärtsdurchlauf je Bild bei.

<code-tabs name="stream" />

## embed

```python
model.embed(source=None, **kwargs) -> torch.Tensor
```

Ein komfortabler Wrapper um `predict`, der jede Embedding-Zeile zu einem einzelnen Tensor der Form `(N_total, D)` stapelt. Das Modell muss mit `task="embed"` erstellt worden sein. Andernfalls löst die Methode `NotImplementedError` aus.

## track

```python
model.track(
    source,
    *,
    track_conf=0.25,
    iou=0.45,
    imgsz=None,
    classes=None,
    max_det=300,
    save=False,
    show=False,
    vid_stride=1,
    output_path=None,
    tracker="bytetrack",
    tracker_config=None,
    augment=False,
    **tracker_kwargs,
) -> Generator[Results, None, None]
```

Gibt je Frame ein `Results`-Objekt mit gesetzter `track_id` aus. `tracker` ist `"bytetrack"`, `"botsort"`, `"ocsort"` oder `"deepocsort"`. Wenn `tracker_config` angegeben ist, wird der Wert ignoriert, da der Konfigurationstyp den Tracker auswählt. `track_conf` wird bei ByteTrack und BoT-SORT auf `track_high_thresh`, bei OC-SORT und Deep OC-SORT auf `det_thresh` abgebildet. `output_path` lautet standardmäßig `runs/track/<video_stem>.mp4`.

## val

```python
model.val(
    data=None,
    batch=16,
    imgsz=None,
    conf=0.001,
    iou=0.6,
    workers=4,
    allow_download_scripts=False,
    device=None,
    split="val",
    augment=False,
    save_json=False,
    verbose=True,
    *,
    plots=None,
    **kwargs,
) -> Dict
```

Gibt ein Metrik-Dictionary zurück, dessen Schlüssel von der Aufgabe abhängen. Die Objekterkennung liefert `metrics/precision`, `metrics/recall`, `metrics/mAP50` und `metrics/mAP50-95`. `imgsz` akzeptiert eine Ganzzahl für ein Quadrat oder ein Tupel `(height, width)` und verwendet standardmäßig die native Eingabegröße des Modells. `plots` ist ein Alias für `save_plots`. `allow_download_scripts` steuert die Ausführung eingebetteten Python-Codes, den eine Datensatz-YAML-Datei im Feld `download` enthalten kann.

`faster_coco_eval` wird über `**kwargs` akzeptiert und ist standardmäßig `True`. Ist das Paket nicht installiert, wird auf pycocotools zurückgegriffen. Das verwendete Backend steht in `model.last_eval_backend`.

Augmentierte Validierung löst für die Aufgaben `obb` und `pose` einen Fehler aus.

## train

`train` wird je Familie definiert und besitzt deshalb unterschiedliche Argumente. Zwei Verhaltensweisen sind gemeinsam, da die Basisklasse jede familienspezifische Methode `train` umschließt:

- `cfg=` akzeptiert einen YAML-Pfad, dessen Schlüssel in den Aufruf übernommen werden. Ausdrückliche Schlüsselwortargumente haben Vorrang vor der Datei.
- `pretrained=False` initialisiert bei einer Familie aus Abdeckungsgruppe `g0` oder `g1` das Modell vor dem Training vollständig neu und lässt sich nicht mit `resume=True` kombinieren.

Welche Augmentationseinstellungen eine Familie tatsächlich berücksichtigt, ist familienbezogen. Weitere Informationen findest du in der [Augmentationsmatrix](/docs/reference/augmentation-matrix).

## export

```python
model.export(format="onnx", **kwargs) -> str
```

Gibt den Pfad zum geschriebenen Artefakt zurück. `format` wird über die Exporter-Registry aufgelöst. `engine` ist dort ein Alias für `tensorrt`, `litert` für `tflite`. Alle Exporter akzeptieren diese gemeinsamen Argumente:

| Argument | Standard | Bedeutung |
|---|---|---|
| `output_path` | `None` | Ausgabedateipfad; wird bei Auslassung unter `weights/` erzeugt |
| `imgsz` | `None` | Tupel `(height, width)` oder einzelne Ganzzahl; standardmäßig native Größe |
| `opset` | `None` | ONNX-Opset-Version |
| `simplify` | `True` | Vereinfachung des ONNX-Graphen ausführen |
| `dynamic` | `True` | Dynamische Achsen aktivieren |
| `half` | `False` | FP16-Genauigkeit |
| `int8` | `False` | INT8-Genauigkeit |
| `batch` | `1` | Im Artefakt eingebettete Batchgröße |
| `device` | `None` | Gerät für die Aufzeichnung |
| `data` | `None` | data.yaml für INT8-Kalibrierung |
| `fraction` | `1.0` | Verwendeter Anteil des Kalibrierungsdatensatzes |
| `allow_download_scripts` | `False` | Eingebettetes Python in Datensatz-YAML-Downloads erlauben |
| `verbose` | `False` | Ausführliche Exporter-Protokollierung |

Gesperrte Kombinationen lösen bei der Vorabprüfung und vor der Aufzeichnung `NotImplementedError` aus. Abdeckung und Regeln stehen auf der Seite zur [Exportmatrix](/docs/reference/export-matrix). Bei vorhandenen aktiven LoRA-Adaptern werden diese in dichte Gewichte integriert. Die Zusammenführung erfolgt erst nach allen Prüfungen, die eine Anfrage ablehnen können.

## save

```python
model.save(path) -> str
```

Schreibt einen LibreYOLO-Checkpoint nach Schema v1.0: das State-Dictionary sowie die unter [Checkpoint-Schema](/docs/reference/checkpoint-schema) beschriebenen Metadaten. Ein quantisiertes Modell enthält zusätzlich sein Manifest `quant`, sodass `LibreYOLO(path)` die quantisierte Struktur und ihre Skalen wiederherstellt.

## quantize, quant_info und dequantize

```python
model.quantize(
    recipe,
    calib="coco128.yaml",
    samples=128,
    batch=8,
    algorithm="auto",
    keep_high_precision=None,
    allow_download_scripts=False,
    verbose=True,
)
```

Quantisiert direkt und gibt das Modell zurück. `recipe` ist eine der Umwandlungen `fp16` und `bf16`, eines der Conv- und Linear-Rezepte `int8` und `fp8` oder eines der ausschließlich linearen Rezepte `w4a16`, `w4a8`, `nvfp4`, `mxfp4` und `int2`, die Transformer-Familien wie RF-DETR unterstützen. `int2` erfordert QAT. `calib` akzeptiert einen data.yaml-Pfad oder einen integrierten Datensatznamen und liest Bilder ausschließlich vorwärts. Labels werden nie gelesen. Übergib `calib=None`, um die Kalibrierung zu überspringen. `algorithm` ist `"minmax"`, `"percentile"` oder `"auto"`.

`model.quant_info()` gibt eine Zusammenfassung des Quantisierungszustands zurück, bei einem Gleitkommamodell `None`. `model.dequantize()` stellt Gleitkommamodule direkt wieder her und behält die mit Quantisierung trainierten Mastergewichte. Dies bildet die Brücke von QAT zu `export(format="onnx", int8=True, data=...)`.

## info und Schichten

```python
model.info(detailed=False, verbose=True) -> Dict[str, Any]
model.get_available_layer_names() -> List[str]
model.get_distill_config() -> Dict
```

`info` gibt ein JSON-kompatibles Dictionary zurück und protokolliert bei wahrem `verbose` eine lesbare Zusammenfassung. `get_available_layer_names` führt die Schichten auf, die eine Konfiguration zur Destillation oder Merkmalsextraktion benennen kann.

## CUDA-Graphen

Verfügbar bei Familien, deren Klassenattribut `SUPPORTS_CUDA_GRAPH` wahr ist. Die Wiedergabe ist bitidentisch zur Eager-Ausführung.

```python
model.capture_graph(imgsz=None, batch=1, dtype=None) -> None
model.cuda_graph_scope(mode=True)          # Kontextmanager
model.graph_info() -> Dict[str, Any]
model.release_graphs() -> None
```

Ein aufgezeichneter Graph gilt nur für exakt die Form, bei der er aufgezeichnet wurde. `batch` und `imgsz` müssen deshalb zum späteren Aufruf von `predict` passen. `capture_graph` verschiebt die Aufzeichnungskosten aus der ersten Anfrage. `mode` akzeptiert `True` oder `"on"` für die Aufzeichnung bei der ersten Verwendung, `"auto"` für das Warten auf eine wiederholte Form und `False` für Wirkungslosigkeit. `capture_graph` löst `NotImplementedError` aus, wenn sich die Familie nicht aktiviert hat, und `CudaGraphUnavailable`, wenn die Aufzeichnung fehlschlägt.

## Gerät und dtype

`Results`-Objekte besitzen `.to()`, `.cpu()`, `.cuda()` und `.numpy()`. Weitere Informationen findest du unter [Ergebnistypen](/docs/reference/results-types). Das Modell selbst wird durch `device=` bei `predict` oder bei seiner Erstellung verschoben.
