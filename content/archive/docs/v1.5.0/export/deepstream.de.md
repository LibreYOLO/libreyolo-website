---
title: NVIDIA DeepStream
seo_title: YOLO-Modelle auf NVIDIA DeepStream ausführen
description: >-
  Exportiere ein LibreYOLO-Modell für NVIDIA DeepStream: einen ONNX-Graph plus
  eine generierte nvinfer-Config. Exakte Befehle für den Parser-Build und die
  Pipeline.
lead: >-
  NVIDIA DeepStream führt die Inferenz über sein nvinfer-Element aus, das einen
  ONNX-Graph, eine passende Config-Datei und einen Bounding-Box-Parser braucht.
  deepstream=True beim ONNX-Export schreibt die ersten beiden und verdrahtet sie
  mit dem dritten.
keywords:
  - nvidia deepstream yolo
  - deepstream yolo einbinden
  - nvinfer config erstellen
  - deepstream bounding box parser
  - config_infer_primary
  - NvDsInferParseYolo
  - deepstream-app
  - tensorrt engine bauen
  - yolo auf jetson ausführen
meta:
  - label: Flag
    value: 'export(format="onnx", deepstream=True)'
    mono: true
  - label: Schreibt
    value: 'Einen ONNX-Graph, config_infer_primary_<stem>.txt und <stem>_labels.txt'
  - label: Abdeckung
    value: 43 Kombinationen aus Familie und Aufgabe über neun Aufgaben
  - label: Parser
    value: >-
      NvDsInferParseYolo, aus dem MIT-lizenzierten Projekt DeepStream-Yolo von
      Marcos Luciano. Einmal pro Gerät gebaut.
    links:
      - label: github.com/marcoslucianops/DeepStream-Yolo
        href: 'https://github.com/marcoslucianops/DeepStream-Yolo'
  - label: Verfügbarkeit
    value: Enthalten in v1.5.0. Am 2026-08-08 mit Pull Request 728 in dev übernommen.
    links:
      - label: Pull Request 728
        href: 'https://github.com/LibreYOLO/libreyolo/pull/728'
      - label: Issue 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
  - label: Zur Laufzeit validiert
    value: 'DeepStream 8.0.0 auf einer RTX 5070 Ti, nur Detektion, 2026-08-08'
verification: >-
  Geschrieben nach der Laufzeit-Validierung vom 2026-08-08. Familienlisten,
  Config-Schlüssel und Defaults gelesen aus libreyolo/export/deepstream.py und
  libreyolo/export/exporter.py bei Commit 5f81e11e, der am selben Tag mit Pull
  Request 728 in dev übernommen wurde.
snippets:
  install:
    - label: Installation
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO9, LibreDFINE


        # Schreibt libreyolo9s.onnx, config_infer_primary_libreyolo9s.txt

        # und libreyolo9s_labels.txt ins Arbeitsverzeichnis.

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="onnx",
        deepstream=True)


        # Jedes Detektionsmodell in ein eigenes Verzeichnis: jede Detektions-

        # Config nennt dieselbe Engine-Cache-Datei. Siehe „Bekannte Fallen“.

        LibreDFINE("LibreDFINEs.pt", size="s").export(format="onnx",
        deepstream=True)
    - label: Argumente
      language: python
      code: >
        model.export(
            format="onnx",     # deepstream=True wird für jedes andere Format abgelehnt
            deepstream=True,
            conf=0.25,         # setzt pre-cluster-threshold (und classifier-threshold,
                               # segmentation-threshold bei diesen Aufgaben)
            iou=0.45,          # setzt nms-iou-threshold, entfällt bei cluster-mode=4
            batch=1,           # setzt batch-size und den Engine-Cache-Dateinamen
            half=False,        # True setzt in der Config network-mode=2 (fp16-Build)
            int8=False,        # True setzt in der Config network-mode=1
            dynamic=True,      # dynamische Batch-Achse im ONNX-Graph
            imgsz=640,         # setzt infer-dims=3;H;W
        )


        # deepstream=True und nms=True schließen sich aus: DeepStream
        unterdrückt

        # in seiner Clustering-Stufe, im Graph ist deshalb nichts eingebettet.
    - label: Zuerst die D-FINE-Gewichte holen
      language: bash
      code: |
        curl -L -o LibreDFINEs.pt \
          https://huggingface.co/LibreYOLO/LibreDFINEs/resolve/main/LibreDFINEs.pt
  gpu:
    - label: Vor allem anderen den GPU-Durchgriff prüfen
      language: bash
      code: |
        docker run --rm --gpus all nvcr.io/nvidia/tritonserver:26.04-py3 \
          nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
      expect: |
        name, driver_version, compute_cap
        NVIDIA GeForce RTX 5070 Ti, 591.86, 12.0
  parser:
    - label: 'build_parser.sh, im DeepStream-Container ausführen'
      language: bash
      code: >
        set -e

        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo.git


        # /usr/local/cuda-12 ist in diesem Image ein Stub, der Build stirbt
        daran mit

        # "fatal error: crt/host_defines.h: No such file or directory". Nimm ein

        # Toolkit, das den Header wirklich hat; im 8.0-Image ist das cuda-12.5.

        CUDA_DIR=$(readlink -f /usr/local/cuda)

        [ -f "$CUDA_DIR/include/crt/host_defines.h" ] || \
          CUDA_DIR=$(ls -d /usr/local/cuda-*.* | sort -Vr | \
                     while read d; do [ -f "$d/include/crt/host_defines.h" ] && echo "$d" && break; done)

        # Das Image liefert libcublas.so.12 und libcublas.so.12.8.4.1, aber
        nicht das

        # unversionierte libcublas.so, das -lcublas braucht, der Link-Schritt
        scheitert

        # mit "/usr/bin/ld: cannot find -lcublas". Gib dem Linker die passenden
        Namen.

        mkdir -p /tmp/cudalibs

        for lib in cublas cublasLt cudart; do
          real=$(find /usr/local -name "lib${lib}.so.1*" | grep -v stubs | sort -V | tail -1)
          ln -sf "$real" "/tmp/cudalibs/lib${lib}.so"
        done

        export LIBRARY_PATH="/tmp/cudalibs:$LIBRARY_PATH"


        make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
        CUDA_VER="${CUDA_DIR##*/cuda-}"
    - label: Die Instanzsegmentierung nutzt einen anderen Parser
      language: bash
      code: >
        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo-Seg.git

        make -C DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg \
          CUDA_VER="${CUDA_DIR##*/cuda-}"
  run:
    - label: deepstream_app_config.txt
      language: text
      code: >
        [application]

        enable-perf-measurement=1

        perf-measurement-interval-sec=5

        gie-kitti-output-dir=kitti


        [tiled-display]

        enable=0


        [source0]

        enable=1

        type=3

        uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4

        num-sources=1

        gpu-id=0


        [streammux]

        gpu-id=0

        batch-size=1

        batched-push-timeout=40000

        width=1920

        height=1080

        live-source=0


        [primary-gie]

        enable=1

        gpu-id=0

        gie-unique-id=1

        config-file=config_infer_primary_libreyolo9s.txt


        [osd]

        enable=1

        border-width=2

        text-size=15


        [sink0]

        enable=1

        type=1

        sync=0


        [tests]

        file-loop=0
    - label: Ausführen
      language: bash
      code: |
        deepstream-app -c deepstream_app_config.txt
      expect: |
        App run successful
    - label: Beide Schritte in einem Container
      language: bash
      code: |
        docker run --rm --gpus all -v "$PWD:/work" -w /work \
          nvcr.io/nvidia/deepstream:8.0-samples-multiarch \
          bash -c "bash build_parser.sh && deepstream-app -c deepstream_app_config.txt"
source_hash: 1ee91c265753dd9a
---

## Verfügbarkeit

Der DeepStream-Export ist in v1.5.0 enthalten. Er wurde am 2026-08-08 mit Pull
Request 728 in `dev` übernommen, eine aktuelle Installation hat ihn also, und
ein Branch-Pin ist nicht nötig.

<code-tabs name="install" />

Wenn du den Branch `deepstream-export` vor dem 2026-08-08 geklont hast, ersetze
ihn. Dieser Branch wurde rebased und force-gepusht, und der älteren Historie
fehlt der Fix, der diese Exporte auf einer CUDA-Maschine überhaupt laufen lässt.

## Was der Export schreibt

`model.export(format="onnx", deepstream=True)` schreibt drei Dateien
nebeneinander. Für `libreyolo9s.pt`:

- `libreyolo9s.onnx`, der Detektions-Graph, ein Ausgabetensor der Form
  `(batch, num_detections, 6)`, jede Zeile `[x1, y1, x2, y2, score, class_id]`
  in Pixelkoordinaten des Netz-Inputs.
- `config_infer_primary_libreyolo9s.txt`, eine `nvinfer`-Konfiguration mit den
  Preprocessing-Konstanten der Familie, der Klassenanzahl, den Schwellenwerten
  und der Parser-Verdrahtung.
- `libreyolo9s_labels.txt`, ein Klassenname pro Zeile.

Eine Labels-Datei entsteht immer dann, wenn der Checkpoint Klassennamen trägt.
Tiefenmodelle haben keine, sie bekommen also weder die Datei noch einen
`labelfile-path`-Schlüssel.

LibreYOLO erzeugt keine `.so`. Die `.so`, die DeepStream lädt, ist der
Bounding-Box-Parser aus `marcoslucianops/DeepStream-Yolo`, einmal pro Gerät
kompiliert, und es ist dieselbe Binärdatei, auf welchen LibreYOLO-Detektor du
sie auch ansetzt. Das Modell ist die ONNX. Klassifikation und semantische
Segmentierung brauchen überhaupt keinen Parser, weil `nvinfer` diese selbst
nachverarbeitet.

## Export des Modells

<code-tabs name="export" />

`LibreDFINE._load_weights` wirft `FileNotFoundError`, wenn die Datei nicht schon
auf der Platte liegt, ohne einen Download zu versuchen, hol dir `LibreDFINEs.pt`
also zuerst selbst. Diese Lücke wird als
[Issue #727](https://github.com/LibreYOLO/libreyolo/issues/727) verfolgt. Die
YOLO9-Gewichte laden beim ersten Gebrauch herunter.

Das Flag gibt es nur in Python. `libreyolo export` hat auf diesem Branch keine
`deepstream`-Option, und die CLI baut ihre Export-Argumente aus einer festen
Liste, statt unbekannte Schlüssel durchzureichen.

## Bau des Bounding-Box-Parsers

Die Detektion braucht die Parser-Bibliothek, die Instanzsegmentierung braucht
eine andere, und die übrigen Aufgaben brauchen keine. Zwei Dinge im
DeepStream-8.0-Image brechen den dokumentierten Build-Befehl, und beide sind
Probleme der Umgebung, nicht von LibreYOLO.

Das Image liefert `cuda`, `cuda-12`, `cuda-12.5`, `cuda-12.8` und `cuda-12.9`
unter `/usr/local`. Nur `cuda-12.5` hat ein vollständiges Toolkit. Es liefert
außerdem `libcublas.so.12` und `libcublas.so.12.8.4.1`, aber nicht das
unversionierte `libcublas.so`, gegen das `-lcublas` auflöst. Das Skript unten
umgeht beides.

<code-tabs name="parser" />

Richte danach `custom-lib-path` in der generierten Config auf die gebaute
`libnvdsinfer_custom_impl_Yolo.so`. Der generierte Wert ist der relative Pfad
`nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so`, der aufgeht, wenn
`deepstream-app` aus dem `DeepStream-Yolo`-Checkout heraus läuft, und sonst
angepasst werden muss.

## Ausführen der Pipeline

Prüfe, ob der Container die GPU sieht, bevor du Zeit in irgendetwas anderes
steckst. Das ist die Prüfung, die der Validierungslauf zuerst gemacht hat, auf
einer Blackwell-Karte unter WSL2.

<code-tabs name="gpu" />

Der Validierungslauf steuerte `deepstream-app` mit einer Dateiquelle, ohne
Display-Sink, mit eingeschaltetem On-Screen-Display und gesetztem
`gie-kitti-output-dir`, sodass die Detektionen jedes Frames als KITTI-Text auf
der Platte landeten. Eine Config mit diesen Einstellungen:

<code-tabs name="run" />

`nvinfer` baut die TensorRT-Engine beim ersten Lauf aus der ONNX und legt sie
neben dem Modell im Cache ab, der erste Lauf bezahlt also den Engine-Build und
spätere laden den Cache.

## Die generierte Config

Beide Configs unten wurden vom Exporter für den Validierungslauf geschrieben und
danach nicht bearbeitet.

| Schlüssel | YOLO9-s | D-FINE-s |
|---|---|---|
| `net-scale-factor` | 0.003921568627 | 0.003921568627 |
| `model-color-format` | 0 | 0 |
| `infer-dims` | 3;640;640 | 3;640;640 |
| `maintain-aspect-ratio` | 1 | 0 |
| `symmetric-padding` | 0 | 0 |
| `network-type` | 0 | 0 |
| `num-detected-classes` | 80 | 80 |
| `cluster-mode` | 2 | 4 |
| `parse-bbox-func-name` | NvDsInferParseYolo | NvDsInferParseYolo |
| `pre-cluster-threshold` | 0.25 | 0.25 |
| `nms-iou-threshold` | 0.45 | |
| `topk` | 300 | 300 |

Die beiden Configs unterscheiden sich an drei Stellen: `maintain-aspect-ratio`,
`cluster-mode`, und ob `nms-iou-threshold` überhaupt vorhanden ist. Die Config
von D-FINE lässt diesen Schlüssel komplett weg, genau das verlangt
`cluster-mode=4`.

Heads, die höchstens eine Vorhersage pro Objekt ausgeben, bekommen
`cluster-mode=4`, DeepStream clustert über ihnen also nicht; Clustering würde
tatsächlich verschiedene Detektionen zusammenlegen. Das betrifft `rfdetr`,
`dfine`, `deim`, `deimv2`, `ec`, `rtdetr`, `rtdetrv2`, `rtdetrv4` und
`yolo9_e2e`. Grid- und Anchor-Heads bekommen `cluster-mode=2` plus
`nms-iou-threshold`.

Detektions-Configs tragen außerdem
`engine-create-func-name=NvDsInferYoloCudaEngineGet`, was den Engine-Build an die
Parser-Bibliothek übergibt. Genau das legt den Dateinamen des Engine-Caches fest,
und daher stammt die Kollision, die unter den bekannten Fallen beschrieben ist.

## Unterstützte Aufgaben und Familien

Dreiundvierzig Kombinationen aus Familie und Aufgabe exportieren.
`deepstream_supported_tasks()` und `deepstream_supported_families(task)` in
`libreyolo/export/deepstream.py` liefern zur Laufzeit dieselben Listen.

| Aufgabe | `network-type` | Parser-Bibliothek | Familien |
|---|---|---|---|
| Detektion | 0 | DeepStream-Yolo | yolo9, yolo9_p2, yolo9_e2e, yolo1, yolo2, yolo3, yolo4, yolo7, yolox, yolonas, rtmdet, picodet, rfdetr, dfine, deim, deimv2, ec, rtdetr, rtdetrv2, rtdetrv4 |
| Klassifikation | 1 | Keine nötig | mobilenetv4, convnext, efficientnetv2, resnet, dinov2 |
| Semantische Segmentierung | 2 | Keine nötig | pidnet, eomt, dinov2, lingbotvision |
| Instanzsegmentierung | 3 | DeepStream-Yolo-Seg | rfdetr, dfine, ec |
| Pose | 100 | Keine nötig | yolo9, yolonas, rfdetr, ec |
| Tiefe | 100 | Keine nötig | depth_anything, zipdepth |
| Restauration | 100 | Keine nötig | nafnet, realesrgan, swinir |
| Matting | 100 | Keine nötig | birefnet |
| Blickrichtung | 100 | Keine nötig | l2cs |

`network-type=100` heißt, dass DeepStream keinen Post-Prozessor für die Aufgabe
hat. Diese Configs setzen `output-tensor-meta=1`, die nativen Ausgaben des
Graphen laufen unangetastet durch, und die Anwendung dekodiert sie aus den
Tensor-Metadaten. Graphen mit mehreren Ausgaben sind dort kein Problem: jeder
Output-Layer erreicht die Metadaten mit denselben Ausgabenamen und dynamischen
Achsen wie bei einem einfachen ONNX-Export.

Zeilen der Instanzsegmentierung sind die Detektionszeile, gefolgt von der Maske
dieser Instanz, flach ausgerollt bei `(netH / 4, netW / 4)`, der Auflösung, die
der Seg-Parser fest verdrahtet, als Wahrscheinlichkeiten für
`segmentation-threshold`.

Klassifikation und Blickrichtung laufen als sekundäre Inferenz. Setze
`process-mode=2` und `operate-on-gie-id` in der generierten Config, um einen
Klassifikator hinter einen Detektor zu hängen. Blickrichtung ist ein reiner
Head-Vertrag, ein Gesichtsausschnitt pro Eingabe, davor braucht es also einen
Gesichtsdetektor.

Drei Familien fehlen mit Absicht. `segformer` ist nicht an den gemeinsamen
semantischen Export-Vertrag angeschlossen und kann in keinem Format nach ONNX
exportieren. Bei RTMDet-Ins und YOLO9 ist der Export der Instanzsegmentierung in
LibreYOLO selbst blockiert. `depth_anything3` hat keine Export-Implementierung.

Hinter zwei Zeilen der Tabelle stecken Lücken bei den Checkpoints. Vom
semantischen EoMT ist nur der Checkpoint `l` veröffentlicht, und für die
DINOv2-Klassifikation gibt es überhaupt keinen veröffentlichten Checkpoint,
diese Kombination braucht also deine eigenen nachtrainierten Gewichte.

## Unterschiede beim Preprocessing

`nvinfer` berechnet `net-scale-factor * (x - offsets)` pro Kanal mit einem
skalaren Faktor, der keine kanalweise Standardabweichung ausdrücken kann.
Familien, die eine brauchen (`rfdetr`, `ec`, die DINO-Backbone-Größen von
`deimv2`, `rtmdet`, `picodet` und jede Klassifikationsfamilie), haben die
Normalisierung im exportierten Graphen eingebacken, und die generierte Config
füttert den Graphen mit dem passenden rohen Eingaberaum.

Bei der Geometrie gehen die eigenen Python-Pipelines von LibreYOLO und `nvinfer`
weiterhin auseinander:

- Letterbox-Familien (`yolo9`, `yolox`, `yolonas`, `rtmdet`, `yolo2`, `yolo3`,
  `yolo4`, `yolo7`) füllen nativ mit Grau auf. `nvinfer` füllt mit Schwarz.
- Die `yolonas`-Detektion skaliert nativ die längste Seite auf 636 innerhalb
  ihrer 640er-Leinwand. Das `maintain-aspect-ratio` von `nvinfer` nutzt die
  vollen 640.
- Die Klassifikation skaliert nativ die kürzeste Seite und schneidet dann mittig
  zu. `nvinfer` streckt das Frame oder die Objekt-ROI auf den Netz-Input, eng
  zugeschnittene Motive fallen also anders aus.
- EoMT fährt für die semantische Segmentierung nativ Sliding-Window-Kacheln. Der
  exportierte Graph ist eine einzige gestreckte Leinwand, das ist schneller und
  ungenauer.
- `pidnet` gibt eine Klassenkarte mit 1/8 der Eingabeauflösung aus,
  `lingbotvision` mit 1/16. DeepStream skaliert die Klassenkarte für die Anzeige
  hoch.

Das ONNX-Paritäts-Gate füttert bereits vorverarbeitete Tensoren, prüft also
Graph-Ausgaben und kann eine falsche Farbreihenfolge oder Padding-Regel in der
Config nicht fangen. Validiere auf deinen eigenen Daten, bevor du eine Last mit
exakter Parität ausrollst.

## Bekannte Fallen

### Zwei Detektionsmodelle in einem Verzeichnis laden gegenseitig ihre Engine

Jede Detektions-Config trägt dieselbe Zeile:

```ini
model-engine-file=model_b1_gpu0_fp32.engine
```

Der Engine-Builder des Parsers verlangt genau diesen Basisnamen, und er variiert
nicht pro Modell. Exportiere ein zweites Detektionsmodell in dasselbe
Verzeichnis, und der zweite Lauf lädt die gecachte Engine des ersten Modells.
Nichts stürzt ab; die Boxen sind einfach falsch. Gib jedem Detektionsmodell ein
eigenes Verzeichnis. Der Validierungslauf musste D-FINE erst in eines isolieren,
bevor es überhaupt getestet werden konnte.

### Eine Box kann nur eine Klasse tragen

Das Zeilenformat von `nvinfer` ist `[x1, y1, x2, y2, score, class_id]`, eine
Klasse pro Box, der Export legt die Klassen-Scores also auf ihr Argmax zusammen.
Eine Box, die `predict` unter zwei Klassen meldet, überlebt unter einer.
Gemessener Fall: LibreYOLO meldet `vase 0.773` und `bottle 0.383` auf derselben
Box, und der DeepStream-Graph behält `vase`. Das folgt aus dem Zeilenformat des
Parsers und lässt sich nicht ändern, ohne diesen Vertrag zu verlassen, es ist
also erwartetes Verhalten und keine Regression.

## Validiert

`deepstream-app` lief auf beiden Detektor-Head-Typen bis EOS durch, mit
`App run successful`, über NVIDIAs mitgeliefertes `sample_1080p_h264.mp4`
(1443 Frames), mit aktivierten KITTI-Dumps pro Frame.

| | YOLO9-s | D-FINE-s |
|---|---|---|
| Head-Typ | grid | one-to-one |
| `cluster-mode` | 2 | 4 |
| `maintain-aspect-ratio` | 1 | 0 |
| Frames mit Detektionen | 1443 | 1443 |
| Detektionen gesamt | 18031 | 71105 |

Klassenhistogramme über alle 1443 Frames setzen bei beiden Modellen Autos an die
erste und Personen an die zweite Stelle, was für eine Straßenszene stimmt. Der
vierfache Unterschied in der Zahl der Detektionen ist der
`cluster-mode`-Unterschied bei der Arbeit: D-FINE clustert bei `cluster-mode=4`
nicht, jede Query über dem Schwellenwert überlebt also, Beinahe-Duplikate
eingeschlossen.

Zwei unabhängig trainierte Modelle setzen das dominante Objekt an dieselbe
Stelle:

```text
YOLO9  bus  [706.72,  0.82, 1916.34, 1062.97]  conf 0.965
D-FINE bus  [702.73,  2.93, 1916.24, 1069.32]  conf 0.965
```

Dieser Lauf belegt fünf Dinge: TensorRT baut auf sm_120 eine Engine aus der
exportierten ONNX, `nvinfer` akzeptiert jeden Schlüssel in der generierten
Config, `NvDsInferParseYolo` liest das Tensor-Layout korrekt, Boxen landen in
Koordinaten der Quellauflösung 1920x1080, und Labels lösen gegen die generierte
Labels-Datei auf.

Die Umgebung, in der er lief:

| Komponente | Wert |
|---|---|
| Host-OS | Windows 11 Pro 26200 |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB |
| Treiber | 591.86 |
| Compute Capability | 12.0 (Blackwell, sm_120) |
| Container-Runtime | Docker Desktop 29.4.3, WSL2-Backend |
| DeepStream-Image | `nvcr.io/nvidia/deepstream:8.0-samples-multiarch` |
| DeepStream-Version | 8.0.0 |
| CUDA im Container | 12.8.1 |
| Parser | `marcoslucianops/DeepStream-Yolo` bei HEAD |

Neben dem Pipeline-Lauf deckt `tests/unit/test_deepstream_export.py` die
Graph-Adapter und die generierten Config-Schlüssel ab, und seine 35 Tests laufen
auf diesem Commit durch.

## Nicht validiert

Hier festgehalten, damit der Umfang oben nicht weiter gelesen wird, als er ist.

- Jetson und aarch64. Der Export-Vertrag hängt nicht von der Architektur ab,
  aber die Pipeline lief bisher nur auf einer diskreten x86-GPU.
- Einundvierzig der 43 Kombinationen. Nur die Detektion mit `yolo9` und die
  Detektion mit `dfine` gingen durch DeepStream. Klassifikation, semantische
  Segmentierung, Instanzsegmentierung und die Roh-Tensor-Aufgaben sind durch
  Unit-Tests und ONNX-Paritätsprüfungen abgedeckt, nicht durch einen
  Pipeline-Lauf.
- FP16 und INT8. Nur `network-mode=0` wurde ausgeführt.
- Multi-Stream und Batching. Eine Quelle, `batch-size=1`.
- Accuracy gegen einen Ground-Truth-Datensatz. Detektionen wurden auf semantische
  Plausibilität und auf Übereinstimmung zwischen den Modellen geprüft, nicht als
  mAP durch DeepStream bewertet.
