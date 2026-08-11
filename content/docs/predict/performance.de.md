---
title: Inferenzleistung
seo_title: Schnellere Inferenz in LibreYOLO
description: >-
  CUDA-Graphen, halbe Genauigkeit, Batching, Kachelinferenz und Test-Time
  Augmentation bei der Vorhersage mit den tatsächlichen Standardwerten und
  unterstützten Familien.
lead: >-
  Fünf Einstellungen zum Vorhersagezeitpunkt beeinflussen Durchsatz oder
  Genauigkeit: CUDA-Graph-Wiedergabe, Genauigkeit, Batching, Kacheln und
  Test-Time Augmentation. Jede gilt für eine bestimmte Gruppe von Familien. Zwei
  davon kosten Genauigkeit oder Latenz, statt sie einzusparen.
keywords:
  - CUDA Graphs PyTorch Inferenz
  - YOLO Batch Inferenz Python
  - FP16 Inferenz
  - Kachelinferenz kleine Objekte
  - Sliced Inference große Bilder
  - Test Time Augmentation Objekterkennung
  - capture_graph
  - Ordner Batch Vorhersage
last_verified: 1.5.0
verification: >-
  Argumentstandardwerte aus InferenceRunner.__call__ in
  libreyolo/models/base/inference.py. CUDA-Graph-API aus
  BaseModel.capture_graph, graph_info, release_graphs und cuda_graph_scope in
  libreyolo/models/base/model.py; familienspezifische Aktivierung über die
  Klassenvariable SUPPORTS_CUDA_GRAPH. Verhalten der halben Genauigkeit aus
  NOOP_PREDICT_KWARGS in libreyolo/utils/predict_args.py, der CLI-Warnung in
  libreyolo/cli/commands/predict.py sowie CAST_RECIPES und SUPPORTED_FAMILIES in
  libreyolo/quant/api.py. Batching-Bedingungen aus
  InferenceRunner._process_in_batches und _predict_batch. Kachelverarbeitung aus
  _predict_tiled und _merge_tile_detections. Test-Time Augmentation aus
  BaseModel._predict_augment und _merge_tta; TTA_ENABLED, TTA_SCALES und
  TTA_FIXED_SIZE familienübergreifend aus libreyolo/models/ gelesen.
snippets:
  batch:
    - label: Batch-Inferenz für einen Ordner
      language: python
      code: >
        from pathlib import Path

        from PIL import Image


        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        folder = Path("batch_demo")

        folder.mkdir(exist_ok=True)

        image = Image.open(SAMPLE_IMAGE)

        for index in range(8):
            image.save(folder / f"frame_{index}.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")


        # Ein gestapelter Vorwärtsdurchlauf je Gruppe von 4 bei unterstützten
        Familien.

        results = model(str(folder), batch=4)

        print(len(results), "results")
    - label: Streaming ohne Aufbau der vollständigen Liste
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("batch_demo", batch=4, stream=True):
            print(len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt source=batch_demo batch=4
  graphs:
    - label: Vorab aufzeichnen und anschließend wiedergeben (CUDA erforderlich)
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")


        # Warmup und Aufzeichnung einmalig außerhalb der ersten Anfrage
        ausführen.

        model.capture_graph()


        result = model(SAMPLE_IMAGE, cuda_graph=True)

        print(len(result.boxes))

        print(model.graph_info())
    - label: Erst bei wiederholter Form aufzeichnen (CUDA erforderlich)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # "auto" wartet, bis eine Form zweimal aufgetreten ist, sodass einmalige
        # Aufgaben keine Kosten für die Aufzeichnung verursachen.
        for _ in range(3):
            model(SAMPLE_IMAGE, cuda_graph="auto")

        print(model.graph_info())
        model.release_graphs()
  precision:
    - label: Export-Extra installieren
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: Exportieren und mit Standardgenauigkeit erneut laden
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: FP16-Export (auf einem CUDA-Rechner erstellen und ausführen)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")
        path = model.export(format="onnx", half=True)

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: FP16 in PyTorch über ein Cast-Rezept (CUDA erforderlich)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # Ein Cast-Rezept liest keine Kalibrierungsdaten.
        model.quantize(recipe="fp16", calib=None)

        result = model(SAMPLE_IMAGE)
        print(len(result.boxes))
  tiling:
    - label: Kachelinferenz für ein großes Bild
      language: python
      code: >
        from PIL import Image


        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Kacheln werden nur verwendet, wenn das Bild größer als die
        Eingabegröße ist.

        large = Image.open(SAMPLE_IMAGE).resize((2048, 1536))

        large.save("large.jpg")


        model = LibreYOLO("LibreYOLO9s.pt")


        result = model("large.jpg", tiling=True, overlap_ratio=0.2)

        print(result.num_tiles, "tiles", len(result.boxes), "detections")
  tta:
    - label: Test-Time Augmentation
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        plain = model(SAMPLE_IMAGE)
        flipped = model(SAMPLE_IMAGE, augment=True)

        print(len(plain.boxes), "->", len(flipped.boxes))
source_hash: 3914665d0e7f892c
---

## Einstellungen und Standardwerte

Jede dieser Einstellungen ist ein Argument von `predict`. Standardmäßig sind alle deaktiviert.

| Argument | Standard | Wirkung |
|---|---|---|
| `batch` | `1` | Bilder je Vorwärtsdurchlauf für Ordner- und Listenquellen |
| `cuda_graph` | `False` | Vorwärtsdurchlauf aus einem aufgezeichneten CUDA-Graphen wiedergeben |
| `tiling` | `False` | Ein großes Bild in überlappende Kacheln aufteilen |
| `overlap_ratio` | `0.2` | Kachelüberschneidung bei aktiviertem `tiling` |
| `augment` | `False` | Gespiegelte Ansichten ausführen und zusammenführen |
| `half` | | Wird akzeptiert, erzeugt eine Warnung und wird ignoriert |
| `device` | `None` | Modell vor der Vorhersage auf das Gerät verschieben |

`imgsz` beeinflusst ebenfalls den Rechenaufwand, da es die Modellauflösung festlegt. Es ist jedoch in erster Linie ein Genauigkeitsargument und gehört zum Modell statt in diese Übersicht.

## Batching

<code-tabs name="batch" />

`batch` gilt für Ordner- und Listenquellen. Mit `batch=1` erhält jedes Bild einen eigenen Vorwärtsdurchlauf. Bei einem Wert über `1` wird jede Gruppe vorverarbeitet, zu einem Tensor gestapelt und einmal ausgeführt. Danach wird sie wieder aufgeteilt, damit die vorhandene Nachverarbeitung für Einzelbilder jeder Familie die erwartete Eingabe erhält.

Der gestapelte Pfad wird nur verwendet, wenn alle folgenden Bedingungen erfüllt sind:

- `batch` ist größer als `1`
- `tiling` ist deaktiviert
- Test-Time Augmentation ist nicht aktiv
- die Familie setzt `SUPPORTS_BATCHED_PREDICT`
- das zugrunde liegende Netzwerk befindet sich nicht im Trainingsmodus

Die letzte Bedingung ist wesentlich. Ein Netzwerk im Trainingsmodus würde die gestapelte Gruppe anhand bildübergreifender Batch-Statistiken normalisieren. Dadurch könnten Bilder derselben Gruppe die Vorhersagen der anderen verändern. Solche Läufe bleiben deshalb sequenziell.

`SUPPORTS_BATCHED_PREDICT` ist standardmäßig wahr. Die folgenden Familien deaktivieren es und führen unabhängig von `batch` je Vorwärtsdurchlauf ein Bild aus: Depth Anything V2, Depth Anything 3, EoMT, Faster R-CNN, FCOS, HRNet, L2CS-Net, LibreMODUS, MiDaS, MoGe-2, PP-OCRv5, Real-ESRGAN, RetinaNet, SAM 3D Body, SwinIR, YOLOv1, ZipDepth, alle Open-Vocabulary-Detektoren und alle Vision-Language-Modelle.

Es gibt einen weiteren Fallback. Wenn die Vorverarbeitung in einer Gruppe keine einheitlichen Tensoren der Form `(1, C, H, W)` mit übereinstimmender Form, dtype und Gerät zurückgibt, wird die Gruppe sequenziell statt gestapelt ausgeführt. Die Korrektheit hängt daher nie davon ab, dass Bilder zufällig dieselbe Größe besitzen.

Kombiniere `batch` bei einem großen Ordner mit `stream=True`, um Vorwärtsdurchläufe im Batch auszuführen, ohne alle Ergebnisse gleichzeitig im Arbeitsspeicher zu halten.

## CUDA-Graphen

<code-tabs name="graphs" />

Ein CUDA-Graph zeichnet einen Vorwärtsdurchlauf einmal auf und gibt ihn als einzelnen Start wieder. Kleine Detektoren verbringen einen großen Teil der Zeit bei Batchgröße 1 mit dem Starten von Kernels. Das Zusammenfassen dieser Starts erhöht daher den Durchsatz. Die Wiedergabeausgabe ist bitidentisch zur Eager-Ausführung.

`cuda_graph` akzeptiert drei Werte. `False` ist der Standard und bewirkt nichts. `True` zeichnet bei der ersten Verwendung jeder Eingabeform auf. `"auto"` wartet, bis sich eine Form wiederholt. Einmalige Aufgaben und Aufgaben mit wechselnden Formen tragen dadurch keine Aufzeichnungskosten.

`capture_graph(imgsz=None, batch=1, dtype=None)` verschiebt diese Kosten aus der ersten Anfrage heraus. Ein Graph gilt nur für exakt die aufgezeichnete Form, daher muss `batch` hier dem späteren Aufruf von `predict` entsprechen.

`graph_info()` meldet aufgezeichnete Graphen, Wiedergabezähler und alle Gründe für einen Rückfall auf Eager. `release_graphs()` gibt die Graphen und ihre statischen Puffer frei.

Die Aufzeichnung benötigt CUDA und eine Familie, die sich über `SUPPORTS_CUDA_GRAPH` dafür aktiviert hat. Sie erfordert einen Vorwärtsdurchlauf ohne für den Host sichtbare Arbeit, was für jede Familie geprüft wird. Die Anforderung für eine nicht aktivierte Familie löst `NotImplementedError` aus, statt unbemerkt Eager auszuführen.

Ein Graph zeichnet Speicheradressen und keine Werte auf. Jede Aktion, die Parameter verschiebt, verwirft ihn. Dazu gehören ein Gerätewechsel über `predict(device=...)`, Quantisierung und Dequantisierung.

Die vollständige familienbezogene Unterstützungsmatrix, die Aufteilung der Schnittstellen und der Vertrag zur Numerik stehen unter [CUDA-Graphen](/docs/reference/cuda-graphs).

## Genauigkeit

<code-tabs name="precision" />

`half=True` hat zum Vorhersagezeitpunkt keine Wirkung. Das Argument wird zur Kompatibilität mit der Befehlszeile akzeptiert, erzeugt eine Warnung über seine Wirkungslosigkeit und wird verworfen, bevor es eine Familie erreicht. Das CLI-Flag `--half` gibt für ein `.pt`-Modell dieselbe Warnung aus.

Es gibt zwei tatsächliche Wege zu geringerer Genauigkeit.

Bei einem exportierten Artefakt wird die Genauigkeit beim Export mit `export(format=..., half=True)` festgelegt. Die resultierende Datei wird unverändert wieder durch `LibreYOLO()` geladen.

Für die Ausführung mit PyTorch wandelt `model.quantize(recipe="fp16")` das Modell in float16 um und installiert Hooks, die Ein- und Ausgaben des Modells in float32 halten. `"bf16"` verfährt ebenso mit bfloat16. Beide Umwandlungen lesen keine Kalibrierungsdaten, weshalb `calib` für sie ignoriert wird. Die Quantisierung unterstützt derzeit vier Familien: YOLOv9, RF-DETR, BiRefNet und FeyNobg. Eine Umwandlung auf einem CPU-Gerät protokolliert eine Warnung vor der geringen Geschwindigkeit. Diese Rezepte sind daher für eine GPU vorgesehen.

Beide Wege verändern die Numerik. Keiner garantiert ohne weitere Prüfung dieselben Erkennungen. Validiere vor dem Deployment.

## Kachelinferenz

<code-tabs name="tiling" />

Die Kachelverarbeitung beschneidet ein großes Bild zu überlappenden quadratischen Kacheln, führt für jede eine Vorhersage aus und führt die Ergebnisse zusammen. Sie eignet sich für kleine Objekte in hochauflösenden Bildern, bei denen eine Skalierung des Gesamtbilds die Ziele unter die vom Modell erkennbare Größe verkleinert.

Die Kachelgröße entspricht der Eingabegröße des Modells oder dem angegebenen `imgsz` und muss quadratisch sein. `overlap_ratio` ist standardmäßig `0.2`. Überlappende Kacheln werden mit klassenweiser Non-Maximum Suppression beim Schwellenwert `iou` abgeglichen. Anschließend wird die zusammengeführte Liste auf `max_det` beschränkt. Dadurch wirkt sich `iou` auch bei Familien auf Kachelvorhersagen aus, die selbst kein NMS ausführen.

Wenn das Bild bereits passt, ist die Kachelverarbeitung nicht nur günstig, sondern wird vollständig übersprungen. Liegen beide Dimensionen höchstens bei der Eingabegröße, wird ein gewöhnlicher Vorwärtsdurchlauf ausgeführt. Bei Klassifizierung, semantischer Segmentierung und der Aufgabe `embed` wird sie ebenfalls übersprungen. Dort fällt sie auf einen einzelnen Durchlauf zurück, da Kacheln für diese Aufgaben keine Bedeutung haben.

Für Aufgaben, deren Nutzlast nicht wieder zusammengesetzt werden kann, löst sie einen Fehler aus: Instanzsegmentierungsmasken, orientierte Boxen, Punkte, Tiefe, Kanten und Normalen. Mit `augment` lässt sie sich nicht kombinieren.

Das Ergebnis enthält `result.tiled` und `result.num_tiles`. Bei `save=True` schreiben Kachelläufe ein Verzeichnis unter `runs/tiled_detections`. Es enthält jede Kachel, das annotierte Bild, eine Rastervisualisierung und eine Datei `metadata.json` mit Kachelgröße, Überschneidung und Schwellenwerten. `result.tiles_path` und `result.grid_path` verweisen darauf.

## Test-Time Augmentation

<code-tabs name="tta" />

`augment=True` führt das Bild mehrfach aus und verbindet die Erkennungen mit klassenweiser Non-Maximum Suppression beim Schwellenwert `iou`. Wie die Kachelverarbeitung macht dies `iou` auch für Familien relevant, die es sonst ignorieren.

In der Praxis handelt es sich um horizontales Spiegeln. Die Skalenliste `TTA_SCALES` enthält standardmäßig nur `1.0` und keine ausgelieferte Familie überschreibt sie. Jede Familie führt daher zwei Durchläufe aus: das Originalbild und sein Spiegelbild. Mit `TTA_FIXED_SIZE` markierte Familien skalieren auf ein festes Quadrat, wodurch mehrere Skalen für sie ohnehin wirkungslos wären.

Semantische und panoptische Segmentierung verwenden eine andere Zusammenführung. Ihre gespiegelte Ansicht wird zurückgespiegelt und die beiden Softmax-Verteilungen werden vor dem Argmax gemittelt, statt sie wie Boxen zusammenzuführen.

Test-Time Augmentation ist nicht für jede Aufgabe verfügbar. Sie löst bei orientierten Boxen, Posenschätzung, Punkten, Tiefe, Normalen, Kanten, Restauration, OCR und Embedding-Modellen einen Fehler aus und lässt sich nicht mit der Kachelverarbeitung kombinieren.

Die folgenden Familien deaktivieren sie vollständig, sodass `augment=True` einen einzelnen gewöhnlichen Durchlauf ausführt: BiRefNet, CenterNet, CLIP, DexiNed, FOMO, HRNet, L2CS-Net, LibreMODUS, NAFNet, PP-OCRv5, Real-ESRGAN, RetinaNet, SAM 3D Body, SigLIP2, SwinIR, TEED, jede SAM-Variante, jeder Open-Vocabulary-Detektor und jedes Vision-Language-Modell.

## Messung

Diese Seite enthält keine Latenzwerte, da Millisekunden ohne Angabe von Hardware, Laufzeitumgebung, Genauigkeit und Batchgröße keine belastbare Aussage sind. Gemessene Werte für verschiedene Hardware- und Laufzeitumgebungen werden unter [visionanalysis.org](https://www.visionanalysis.org) veröffentlicht. `libreyolo profile` misst ein bestimmtes Modell auf dem aktuell verwendeten Rechner.
