---
title: Hailo
seo_title: LibreYOLO-Modelle auf Hailo-Beschleunigern ausführen
description: >-
  Ein LibreYOLO-Modell auf einem Hailo-8 oder Hailo-8L ausrollen: der statische
  ONNX-Export, die selbst ausgeführte Dataflow-Compiler-Phase und die
  kompilierbaren Architekturen.
lead: >-
  Hailo-Beschleuniger werden mit dem Hailo Dataflow Compiler kompiliert, einem
  proprietären SDK aus der Developer Zone von Hailo. LibreYOLO übernimmt in
  diesem Ablauf den einfachen statischen ONNX-Export. Anschließend erfolgen
  Parsing, Quantisierung und Kompilierung zu einer HEF im DFC.
keywords:
  - libreyolo hailo
  - hailo-8
  - hailo-8l
  - raspberry pi ai kit
  - ai hat+
  - hailo dataflow compiler
  - hef kompilieren
  - hailortcli
last_verified: 1.5.0
meta:
  - label: LibreYOLO-Schritt
    value: 'export(format="onnx", imgsz=640, dynamic=False)'
    mono: true
  - label: Kein Format
    value: Es gibt kein format="hef". Der DFC kann keine pip-Abhängigkeit sein.
  - label: Zusatzpaket
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: Kompilierungshost
    value: >-
      Linux x86_64 einschließlich WSL2 Ubuntu 22.04. Die Kompilierung kann nicht
      auf ARM ausgeführt werden.
  - label: Kompilierbar
    value: >-
      Reine CNN-Graphen mit fester Form. Attention, dynamische Formen und von
      LayerNorm dominierte Designs sind nicht kompilierbar.
  - label: Status
    value: >-
      Noch keine LibreYOLO-Familie wurde vollständig mit dem DFC in eine
      lauffähige HEF überführt.
verification: >-
  Geprüft anhand von skills/libreyolo-export-hailo/SKILL.md,
  libreyolo/export/onnx.py und libreyolo/cli/commands/export.py im dev-Branch.
  Die DFC-Einschränkungen entsprechen den Angaben in diesem Skill. Noch keine
  LibreYOLO-HEF wurde kompiliert und gemessen.
snippets:
  install:
    - label: LibreYOLO-Seite
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 'Hailo-Seite, von dir installiert'
      language: text
      code: >
        Prerequisites, none of them installable from PyPI:


        - A Linux x86_64 machine. WSL2 Ubuntu 22.04 works. The Raspberry Pi is a
          runtime target, never the compile host.
        - The Dataflow Compiler wheel (hailo_sdk_client) from the Hailo
        Developer
          Zone, which is free to register for.
        - For Hailo-8 and Hailo-8L, the Hailo Model Zoo v2.x line, for its
          recipes and NMS configurations.
        - A GPU on the compile host is strongly recommended: the quantization
          step takes hours without one.
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Hailo benötigt Batch 1, eine feste Auflösung und keine dynamischen
        Achsen.

        # Die Python-API nutzt standardmäßig dynamic=True, daher explizit
        abschalten.

        model = LibreYOLO("LibreYOLOXs.pt")

        model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    - label: CLI
      language: bash
      code: |
        # Die CLI verwendet bereits standardmäßig statische Formen.
        libreyolo export --model LibreYOLOXs.pt --format onnx --imgsz 640
    - label: Statischen Graphen vor dem Kompilieren bestätigen
      language: python
      code: |
        import onnx

        graph = onnx.load("weights/LibreYOLOXs.onnx").graph
        shape = graph.input[0].type.tensor_type.shape
        print([d.dim_value or d.dim_param for d in shape.dim])
  compile:
    - label: 'Parsen, quantisieren und kompilieren'
      language: python
      code: >
        from pathlib import Path


        import numpy as np

        from hailo_sdk_client import ClientRunner

        from PIL import Image


        ONNX = "weights/LibreYOLOXs.onnx"

        HW_ARCH = "hailo8"     # hailo8 | hailo8l | hailo10h

        IMGSZ = 640


        runner = ClientRunner(hw_arch=HW_ARCH)


        # YOLOX zunächst ohne end_node_names übersetzen: Das DFC-Protokoll zeigt

        # die vorgeschlagenen Endknoten. Danach mit diesen erneut ausführen.

        runner.translate_onnx_model(ONNX)


        # Die Normalisierung muss zur LibreYOLO-Vorverarbeitung passen. YOLOX
        und YOLO9

        # benötigen weder Mittelwert noch Standardabweichung, nur die Skalierung
        0-255 auf 0-1.

        script = "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0,
        255.0])\n"


        # Optional: NMS von Hailo ausführen lassen. Die Konfiguration hängt
        sowohl

        # von Klassenanzahl als auch Eingabegröße ab, daher ist COCO-80 für ein

        # auf drei Klassen nachtrainiertes Modell falsch. Ohne diese Zeile gibt
        die HEF rohe

        # Head-Tensoren aus, die von der Anwendung dekodiert werden.

        # script += 'nms_postprocess("yolox_nms_config.json", meta_arch=yolox,
        engine=cpu)\n'


        runner.load_model_script(script)


        # Kalibrierungsbilder müssen die Deployment-Daten repräsentieren.

        # Zufallsbilder lassen sich kompilieren, zerstören aber unbemerkt die
        Accuracy.

        calib_paths = sorted(Path("calib_images").glob("*.jpg"))[:128]

        calib = np.stack([
            np.asarray(
                Image.open(p).convert("RGB").resize((IMGSZ, IMGSZ)),
                dtype=np.float32,
            )
            for p in calib_paths
        ])


        runner.optimize(calib)

        Path("libreyoloxs.hef").write_bytes(runner.compile())
    - label: YOLO9-Endknoten
      language: python
      code: >
        # LibreYOLO-Graphen verwenden das Präfix "/head/...", nicht das Präfix
        "model.N"

        # aus Konfigurationen für andere Exporte. Eine kopierte Konfiguration

        # passt nicht. Prüfe bei Parsing-Fehlern die Namen in deinem eigenen
        Graphen.

        END_NODES = [
            "/head/cv2.0/cv2.0.2/Conv", "/head/cv3.0/cv3.0.2/Conv",
            "/head/cv2.1/cv2.1.2/Conv", "/head/cv3.1/cv3.1.2/Conv",
            "/head/cv2.2/cv2.2.2/Conv", "/head/cv3.2/cv3.2.2/Conv",
        ]

        runner.translate_onnx_model(ONNX, end_node_names=END_NODES)
  device:
    - label: Raspberry Pi 5 mit AI Kit oder AI HAT+
      language: bash
      code: >
        sudo apt install dkms hailo-all

        hailortcli fw-control identify       # Gerät prüfen und Architektur
        anzeigen

        hailortcli run libreyoloxs.hef       # Smoke-Test und Durchsatz
source_hash: 33b077f1c23d5535
---

## Installation

In LibreYOLO gibt es kein `format="hef"`, und es wird auch keines geben. Der Hailo
Dataflow Compiler ist ein proprietäres SDK. Es wird nach einer Registrierung in
der Developer Zone als privates Wheel bereitgestellt und kann daher weder eine
Abhängigkeit noch ein Zusatzpaket sein. Das Deployment besteht aus zwei Phasen:
LibreYOLO schreibt eine statische ONNX-Datei, die du anschließend mit dem DFC verarbeitest.

```text
Libre<Model>.pt  ->  ONNX  ->  HAR (parse)  ->  HAR (quantize INT8)  ->  HEF
                 [libreyolo]           [Hailo DFC, installed by you]
```

<code-tabs name="install" />

## Export

<code-tabs name="export" />

Übergib nicht `half=True`. Der DFC nimmt FP32-ONNX entgegen und führt seine
eigene INT8-Quantisierung durch. Übergib auch nicht `nms=True`: Entweder übernimmt
Hailo die NMS über `nms_postprocess` oder die Anwendung führt sie aus. Hinter den
Endknoten ist ein NMS-Untergraph nur Ballast. Der standardmäßige Opset funktioniert.
Falls der DFC-Parser Einwände hat, exportiere erneut mit `opset=11`.

Der DFC schneidet den Graphen an den von dir angegebenen Endknoten ab. Das sind
die Faltungen des Detection-Heads. Alles Nachfolgende wird verworfen. Der reguläre
dekodierte ONNX-Export von LibreYOLO eignet sich daher als Eingabe: Der Parser
ignoriert einfach den Dekodierungsabschnitt am Ende.

## Kompilierung

<code-tabs name="compile" />

Wähle `hw_arch` passend zum Ziel: `hailo8` für Hailo-8, das AI HAT+ mit 26 TOPS
sowie die M.2- und PCIe-Module, `hailo8l` für Hailo-8L, das Raspberry Pi AI Kit
und das AI HAT+ mit 13 TOPS sowie `hailo10h` für Hailo-10H, das einen passenden
neueren DFC und ein neueres Model Zoo benötigt. Wenn du dir unsicher bist,
beantwortet `hailortcli fw-control identify` auf dem Gerät diese Frage.

Zwei Familien lassen sich einer HailoRT-NMS-Meta-Architektur zuordnen. Hailo kann
die Unterdrückung daher innerhalb der kompilierten Pipeline übernehmen: YOLOX über
`meta_arch=yolox` und YOLO9 über Hailos Meta-Architektur mit entkoppeltem Head,
deren Head-Aufbau identisch ist. Verwende die passende `nms_postprocess`-Konfiguration
aus dem Hailo Model Zoo und passe sie an die Anzahl deiner Klassen und deine
Eingabegröße an. Jeder andere faltungsbasierte Detektor wird als Graph ohne
passende Meta-Architektur kompiliert: Die HEF gibt rohe Head-Tensoren aus, und
die Anwendung führt Dekodierung und NMS auf der CPU aus.

Bewahre das Kompilierungsprotokoll auf, wenn etwas fehlschlägt. Jede Korrektur
hängt vom Namen der betroffenen Schicht oder des betroffenen Operators ab.

## Ausführung des Artefakts

<code-tabs name="device" />

Die Anwendungsinferenz verwendet die Python-API `hailo_platform`. Mit einkompiliertem
`nms_postprocess` hat die Ausgabe die Form `(batch, num_classes, max_dets, 5)`
und enthält `[y1, x1, y2, x2, score]` in Modellkoordinaten. Du skalierst diese
selbst auf das Quellbild zurück. Die `Results`-Pipeline von LibreYOLO ist zur
Laufzeit nicht beteiligt. Die HEF ist ein eigenständiges Artefakt, und die
Anwendung übernimmt die Vor- und Nachverarbeitung.

## Einschränkungen

Ob ein Modell auf Hailo-8 oder Hailo-8L ausgeführt werden kann, hängt von seiner
Architektur und nicht von seinem Namen ab. Die folgende Regel gilt daher auch
für Familien, die erst nach dem Schreiben dieser Seite hinzugefügt wurden.

Ein Modell lässt sich nicht kompilieren, wenn es eine dieser Eigenschaften aufweist:

- Attention beliebiger Art, ob Self-, Cross-, Deformable- oder Window-Attention.
  Dadurch scheiden alle Detektoren im DETR-Stil, alle Open-Vocabulary- oder
  textkonditionierten Detektoren, alle ViT-Backbones sowie alle Sprach- oder
  Vision-Language-Tower aus. Hailos eigenes Model Zoo liefert einige speziell
  optimierte Transformer-HEFs. Das ist maßgeschneiderte Arbeit des Anbieters und
  kein Beleg dafür, dass sich ein beliebiger Attention-Graph kompilieren lässt.
- Dynamische Formen oder datenabhängiger Kontrollfluss. Der DFC kompiliert eine
  feste Eingabeform und einen statischen Graphen. Daher sind variable Query-Anzahlen,
  Text-Prompts, dynamisches Top-k, `NonZero`, `Gather` oder `TopK` mit dynamischen
  Indizes sowie `grid_sample` ausgeschlossen.
- Ein von LayerNorm oder GELU dominiertes Design. BatchNorm lässt sich sauber in
  Faltungen integrieren. LayerNorm wird nur eingeschränkt unterstützt, und GELU
  ist keine native Aktivierung. Ein ConvNeXt-artiger Stack eignet sich daher
  schlecht, obwohl er formal faltungsbasiert ist.
- Bild-zu-Bild-Verarbeitung in nativer Auflösung. Wiederherstellungsmodelle laufen
  mit der vollen Eingabeauflösung und überschreiten die praktischen SRAM-Grenzen
  von Hailo.

Eine Familie kommt infrage, wenn sie ausschließlich Faltungen nutzt, BatchNorm
mit ReLU oder SiLU verwendet und eine feste Eingabegröße hat. In dieser Bibliothek
betrifft das einstufige CNN-Detektoren, wobei YOLOX und YOLO9 die primären Ziele
sind, andere faltungsbasierte Detektoren wie PicoDet, YOLO-NAS und RTMDet mit
anwendungsseitiger Dekodierung, die CNN-Klassifikatoren ResNet, MobileNetV4-conv
und EfficientNetV2, wobei ResNet am besten unterstützt wird, weil Hailos Model
Zoo dafür Rezepte bereitstellt, sowie kleine faltungsbasierte Aufgaben-Heads wie
FOMO-Punkterkennung und L2CS-Blickrichtung mit einem ResNet-Backbone. Letztere
sind grundsätzlich kompilierbar, besitzen aber kein Hailo-Rezept.

Eine Einschränkung beim Status erklärt, warum diese Seite nichts als unterstützt
ausweist: Noch keine LibreYOLO-Familie wurde vollständig mit dem DFC in eine
lauffähige HEF überführt. Die obigen Regeln leiten die Kompilierbarkeit aus der
Architektur ab. Parser-Verhalten, Quantisierung und Accuracy bleiben ungeprüft,
bis eine HEF kompiliert und gemessen wurde. Behandle daher jeden Kandidaten als
Fall, der eigene dokumentierte Nachweise benötigt: eine kompilierte HEF aus dem
exakten Checkpoint mit aufgezeichneten DFC-, Model-Zoo- und HailoRT-Versionen,
dokumentierter Kalibrierung und einem Accuracy-Vergleich auf dem Gerät gegenüber
der FP32-Baseline statt nur einer Durchsatzzahl.

Wenn das Modell ausscheidet, stehen Runtimes mit dokumentierter Parität als
Alternativen bereit: [ONNX](/docs/export/onnx), [TensorRT](/docs/export/tensorrt)
und [OpenVINO](/docs/export/openvino).
