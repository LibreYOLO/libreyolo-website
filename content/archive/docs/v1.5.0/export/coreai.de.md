---
title: Core AI
seo_title: Export nach Apple Core AI aus LibreYOLO
description: >-
  Exportiere ein LibreYOLO-Modell in ein .aimodel-Asset für Apple Core AI: nur
  macOS, festes Canvas, FP32 und der Vertrag über die Reihenfolge der benannten
  Ausgaben, an den sich Konsumenten halten müssen.
lead: >-
  Core AI ist Apples Inferenz-Stack für die Ausführung auf dem Gerät. LibreYOLO
  erfasst das Modell mit torch.export, senkt es über den Core-AI-Konverter ab
  und schreibt ein .aimodel-Asset, das die Modell-Metadaten und die exportierten
  Ausgabenamen mitbringt.
keywords:
  - libreyolo nach core ai exportieren
  - aimodel
  - coreai-torch
  - torch.export apple
  - apple inferenz auf dem gerät
  - coreai_output_names
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="coreai")
    mono: true
  - label: Schreibt
    value: Ein .aimodel-Asset mit angehängten Metadaten
  - label: Extra
    value: 'pip install "libreyolo[coreai]"'
    mono: true
  - label: Zurückladen
    value: Nicht über LibreYOLO. Konsumenten nutzen die Core-AI-Runtime direkt.
  - label: Shapes
    value: Festes Canvas. dynamic=True löst NotImplementedError aus.
  - label: Präzision
    value: Nur FP32. half=True und int8=True werden abgelehnt.
  - label: Voraussetzung
    value: >-
      macOS. Die Toolchain konvertiert weder auf anderen Plattformen noch läuft
      sie dort, und coreai-torch nagelt torch auf 2.11.x fest.
verification: >-
  Gelesen aus libreyolo/export/coreai.py, libreyolo/export/coreai_compat.py,
  libreyolo/export/exporter.py, libreyolo/export/support.py und pyproject.toml
  im dev-Branch.
snippets:
  install:
    - label: 'Installation, unter macOS'
      language: bash
      code: |
        # Absichtlich aus jedem Sammel-Extra herausgehalten: coreai-torch nagelt
        # torch auf 2.11.x fest und zieht die ganze Umgebung auf diese Version.
        pip install "libreyolo[coreai]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Schreibt weights/LibreYOLO9t.aimodel
        path = model.export(format="coreai", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreai --imgsz 640
    - label: Argumente
      language: python
      code: |
        model.export(
            format="coreai",
            imgsz=640,        # int oder (Höhe, Breite); das ist das Run-Canvas
            batch=1,
            output_path=None, # None schreibt weights/<stem>.aimodel
        )

        # dynamic=True löst NotImplementedError aus.
        # half=True und int8=True werden bei der Validierung abgelehnt.
  outputs:
    - label: 'Die Reihenfolge der Ausgaben lesen, bevor du einen Konsumenten anbindest'
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="coreai", imgsz=640)

        # Die Asset-Metadaten halten die exportierten Ausgabenamen in Graph-
        # Reihenfolge unter "coreai_output_names" fest. Ordne das von Core AI
        # zurückgegebene Dictionary über diese Liste zu, nie über die Position
        # im Tupel des Eager-Modells.
  support:
    - label: Eine Familie und Aufgabe vor dem Export prüfen
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: a35bfeafac6d6966
---

## Installation

Dieses Format läuft nur unter macOS. Die Anforderung `coreai-torch` trägt einen
`sys_platform == 'darwin'`-Marker, und die Toolchain konvertiert weder auf
anderen Plattformen noch läuft sie dort.

<code-tabs name="install" />

Das Extra steht außerhalb jedes Sammel-Extras, auch außerhalb von
`libreyolo[all]`, weil `coreai-torch` torch auf die 2.11-Reihe festnagelt.
Installiere es in eine Umgebung, die du bereit bist auf dieses Paar
einzuschränken.

## Export

<code-tabs name="export" />

Die Erfassung läuft über `torch.export`, eine echte Graph-Erfassung mit Guards,
und nicht über einen einzelnen aufgezeichneten Trace. Das ist strenger als der
Core-ML-Weg: Host-seitige Skalar-Zugriffe und datenabhängiger Kontrollfluss
werden abgelehnt, statt still eingebacken zu werden. Deshalb sind hier ein paar
Familien blockiert, mit einem vermerkten Fehler bei der Erfassung.

Drei Vorbereitungsschritte laufen in einem Scope, der das aktive Modell des
Aufrufers wiederherstellt, ob der Export nun gelingt oder scheitert. Bei den von
Darknet abgeleiteten Familien wird die Batch Normalization der Inferenz exakt in
die vorangehenden Convolutions gefaltet, weil Core AI 0.4.1 Darknets Formel mit
Epsilon nach der Quadratwurzel nicht erhält. Bei Grid- und Anchor-Familien
werden die Anchors für das feste Canvas eingefroren. Bei RF-DETR wird das
Position Embedding für das angeforderte Canvas neu gebacken, indem der
modelleigene Baking-Pfad noch einmal läuft, weil der Konverter kein Lowering für
`aten._upsample_bicubic2d_aa` hat.

Das Lowering nimmt PyTorchs Referenz-Decomposition für `aten.grid_sampler_2d` in
die Decomposition-Table auf, denn der Core-AI-Konverter hat kein Lowering für
den Deformable-Attention-Sampler, den die DETR-Familien nutzen.

Assets deklarieren ein Mindest-OS von v27, den einzigen Wert, den die Toolchain
anbietet. Das begrenzt das Deployment, nicht die Konvertierung: Konvertierung
und Ausführung auf der Python-Seite funktionieren über die Runtime im Wheel auch
auf älterem macOS, aber die Numerik unterscheidet sich zwischen den
OS-Versionen, deshalb wird die vermerkte Parität auf macOS 27 gemessen.

## Ausführung des Artefakts

In `libreyolo/backends` gibt es keinen Core-AI-Eintrag, deshalb lädt
`LibreYOLO()` keine `.aimodel`-Datei. Konsumenten nutzen die Core-AI-Runtime
direkt, und Vorverarbeitung, Decoding, NMS und die Umrechnung der Koordinaten
liegen bei ihnen. Eine validierte Zeile in der Support-Matrix behauptet, dass
der exportierte Graph dieselben Zahlen berechnet wie die Referenz, nicht dass
`predict` ihn ausführt.

Das Einzige, was ein Konsument nicht selbst herleiten kann, ist die Reihenfolge
der Ausgaben:

<code-tabs name="outputs" />

Core AI gibt ein benanntes Dictionary zurück, dessen Schlüsselreihenfolge weder
zur Tupel-Reihenfolge des Eager-Forward-Passes noch zu irgendetwas Erratbarem
passt. Genau deshalb stehen die exportierten Namen als `coreai_output_names` in
den Asset-Metadaten. Ordne über die Namen zu.

## Einschränkungen

Festes Canvas, FP32, Batch wie exportiert. `dynamic=True` löst
`NotImplementedError` aus, und `half=True` sowie `int8=True` werden bei der
Validierung abgelehnt.

Auf der Konvertierungsseite ist die Abdeckung breit. Zu den validierten
Kombinationen gehören die YOLO9-Familien, YOLOX, YOLO7, die vier Detektoren aus
der Darknet-Ära, YOLO-NAS, PicoDet, RTMDet, RT-DETR, RT-DETRv2, RT-DETRv4,
D-FINE, DEIM, DEIMv2, EC und RF-DETR für die Erkennung; die vier
CNN-Klassifikationsfamilien plus CLIP und SigLIP2 mit eingefrorenen Klassen;
Depth Anything V2 und ZipDepth; NAFNet und Real-ESRGAN für die Restauration;
PIDNet und LingBotVision für die semantische Segmentierung; und FOMO für die
Punkterkennung. Jede bringt ihren eigenen vermerkten Kontext mit, den
`libreyolo formats` ausgibt.

Blockiert, mit dem je Kombination vermerkten Grund:

| Kombination | Warum |
|---|---|
| EoMT, semantische Segmentierung | Die strikte Erfassung scheitert mit `GuardOnDataDependentSymNode`: Irgendetwas im Masken-Pfad liest einen Wert aus einem Tensor und verzweigt darauf |
| SegFormer, semantische Segmentierung | Der Erfassungspfad wurde nicht bewertet, und die veröffentlichten Gewichte sind unabhängig vom Format nicht kommerziell nutzbar |
| L2CS, Blickschätzung | Das Modell selbst unterstützt nur ONNX, TorchScript, ExecuTorch, TensorRT und OpenVINO, was eine Entscheidung auf Modellseite ist |
| Depth Anything 3, Tiefenschätzung | Die Familie lehnt den Export für jedes Format ab |

Bei RF-DETR gibt es einen Vorbehalt, den du lesen solltest, bevor du Artefakte
vergleichst. Die Parität ist gegen den Graphen vermerkt, den der
Core-AI-Exporter selbst vorbereitet, und nicht gegen ONNX, und bei einem Canvas
von 640 weicht das ONNX-Artefakt von RF-DETR von diesem vorbereiteten Graphen
ab. Das Rebaking in Core AI erhält den Resize mit Antialiasing, den auch das
Eager-Modell durchführt, während der ONNX-Weg das Antialiasing abschaltet. ONNX
ist deshalb für diese Familie bei einem nicht nativen Canvas keine gültige
Referenz.

Für Apples älteres Format siehe [Core ML](/docs/export/coreml). Für das
vollständige Raster aus Familien und Aufgaben siehe
[die Export-Matrix](/docs/reference/export-matrix). Für eine einzelne
Kombination:

<code-tabs name="support" />
