---
title: RKNN
seo_title: Export nach RKNN für Rockchip-NPUs
description: >-
  Kompiliere einen LibreYOLO-Detektor zu einem Rockchip-Artefakt .rknn: das
  Vendor-SDK, das du selbst installierst, die vier validierten RK3588-Varianten
  und die Parität im Simulator.
lead: >-
  RKNN ist Rockchips kompiliertes NPU-Format. LibreYOLO exportiert ein
  ONNX-Zwischenformat mit Opset 19, kompiliert es mit dem SDK RKNN Toolkit2 und
  kann den kompilierten Graphen im Host-Simulator von Toolkit2 ohne Board gegen
  ONNX Runtime vergleichen.
keywords:
  - yolo nach rknn exportieren
  - rockchip npu
  - rk3588
  - rknn-toolkit2
  - rknn simulator parität
  - orange pi rockchip inferenz
last_verified: 1.5.0
meta:
  - label: Flag
    value: 'export(format="rknn", name="rk3588")'
    mono: true
  - label: Schreibt
    value: >-
      Eine .rknn-Datei, ein Sidecar .rknn.metadata.json und einen Bericht
      .rknn.parity.json bei verify=True
  - label: Extra
    value: >-
      Keins auf PyPI. rknn-toolkit2 ist ein Vendor-SDK, das du selbst
      installierst.
  - label: Zurückladen
    value: >-
      Nicht über LibreYOLO. Das Artefakt läuft auf dem Board mit Rockchips
      Runtime.
  - label: Formen
    value: 'Fest quadratisch, Batch 1, Opset 19. Alle drei werden erzwungen.'
  - label: Präzision
    value: >-
      Der Floating-Build des Herstellers. half=True und int8=True werden
      abgelehnt.
  - label: Umfang
    value: >-
      Vier Erkennungsvarianten auf RK3588: YOLO9-t, YOLO9-E2E-t, PicoDet-s und
      YOLO-NAS-s
verification: >-
  Gelesen aus libreyolo/export/rknn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py und docs/rknn.md im dev-Branch. Die gemessenen
  Paritätswerte stammen aus dem Validierungsprotokoll vom 2026-08-04 in
  docs/rknn.md.
snippets:
  install:
    - label: Auf LibreYOLO-Seite
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 'Vendor-SDK, von dir installiert'
      language: bash
      code: >
        # rknn-toolkit2 ist ein Rockchip-SDK unter eigener Lizenz. LibreYOLO

        # liefert es weder mit noch installiert es. Nur Linux x86_64; unter

        # Windows nutze WSL2 oder einen Linux-Container.

        #

        # Toolkit2 2.3.2 braucht setuptools<81 und scheitert an ONNX 1.19 oder

        # neuer: dort ist onnx.mapping entfernt, der Compiler importiert es
        noch.

        pip install "setuptools==80.9.0" "onnx==1.18.0"


        # Dann installiere das passende rknn-toolkit2-Wheel aus Rockchips

        # eigenem Wheel-Repository und prüfe, ob es sich importieren lässt:

        python -c "import rknn.api; print('rknn-toolkit2 ready')"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # Schreibt weights/LibreYOLO9t.rknn und
        weights/LibreYOLO9t.rknn.metadata.json

        path = model.export(format="rknn", name="rk3588", imgsz=640,
        verify=True)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format rknn --name rk3588 \
          --imgsz 640 --verify
    - label: Argumente
      language: python
      code: |
        model.export(
            format="rknn",
            name="rk3588",     # Zielplattform; target= und target_platform= gehen auch
            imgsz=640,         # muss dem erfassten Canvas der Variante entsprechen
            batch=1,           # jeder andere Wert löst NotImplementedError aus
            dynamic=False,     # True löst ValueError aus
            opset=19,          # jeder andere Wert löst NotImplementedError aus
            verify=False,      # True startet den PC-Simulator und erzwingt Parität
        )
  parity:
    - label: Parität ohne Board gegen ein vorhandenes ONNX-Artefakt
      language: python
      code: |
        import numpy as np
        from libreyolo.export import verify_rknn_simulator_parity

        input_tensor = np.random.default_rng(0).standard_normal(
            (1, 3, 640, 640), dtype=np.float32
        )
        metrics = verify_rknn_simulator_parity(
            "weights/LibreYOLO9t.onnx",
            input_tensor,
            target_platform="rk3588",
            rtol=1e-3,
            atol=1e-4,
            raise_on_failure=False,
        )
        print(metrics)
  support:
    - label: Eine Familie und Aufgabe vor dem Kompilieren prüfen
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: c659713cc3c8cc9e
---

## Installation

Zum Kompilieren brauchst du Rockchips RKNN Toolkit2, das als Vendor-SDK unter
Rockchips eigener Lizenz vertrieben wird und keine Abhängigkeit von LibreYOLO
ist. Es gibt kein Extra `libreyolo[rknn]`, und an diesem Format installiert sich
nichts mit einer einzigen Zeile.

<code-tabs name="install" />

Zum Kompilieren oder zum Prüfen der numerischen Parität brauchst du kein Board.
Ein RK3588-Board brauchst du für Messungen von Latenz, Leistungsaufnahme und
Temperaturverhalten, von denen keine einzige erfasst wurde.

## Export

<code-tabs name="export" />

Die Anfrage wird gegen eine Liste exakter Modellvarianten validiert, bevor
irgendetwas kompiliert wird, und das Canvas wird ebenfalls validiert: ein `imgsz`
zu übergeben, das nicht dem entspricht, mit dem die Variante erfasst wurde, löst
einen Fehler aus, statt stillschweigend etwas Ungetestetes zu kompilieren.
LibreYOLO schreibt ein ONNX-Zwischenformat mit Opset 19, kompiliert es, simuliert
es optional und entfernt das Zwischenformat danach.

Die Metadaten liegen in einem Sidecar namens `<model>.rknn.metadata.json`, weil
das RKNN-Format kein portables Metadatenfeld hat.

`verify=True` startet den PC-Simulator von Toolkit2 in derselben Session, die das
Artefakt kompiliert hat, vergleicht jede Ausgabe mit ONNX Runtime auf derselben
Eingabe und schreibt `<model>.rknn.parity.json` mit Fehlermetriken pro Ausgabe.
Die Schwellen sind eine Kosinusähnlichkeit von mindestens 0.9999 und ein
normalisierter RMSE von höchstens 0.02, angewendet auf jede Ausgabe, die nicht
ohnehin elementweise nah dran ist; der Floating-Build des Herstellers senkt
interne Tensoren auf Half-Präzision ab, sodass ein striktes `allclose` selbst
dann nicht hält, wenn die dekodierten Boxen stabil sind. Ein fehlgeschlagener
Lauf schreibt `<model>.rknn.failed.parity.json`, verwirft den Kandidaten und
lässt einen früheren erfolgreichen Export an diesem Pfad unangetastet.

Um ein ONNX-Artefakt zu vergleichen, das du bereits hast, ohne erneut zu
exportieren:

<code-tabs name="parity" />

Der Simulator von Toolkit2 führt den In-Memory-Graphen aus, den `load_onnx` und
`build` erzeugen. Er kann eine zielspezifische `.rknn`-Datei ohne Board nicht
neu laden, weshalb `verify=True` Kompilierung, Export und Simulation in einer
Session erledigt.

## Ausführung des Artefakts

In `libreyolo/backends` gibt es keinen RKNN-Eintrag, also lädt `LibreYOLO()`
keine `.rknn`-Datei. Das kompilierte Artefakt wird auf dem Board ausgerollt und
von Rockchips eigener Runtime ausgeführt, und Preprocessing, Dekodierung, NMS
und die Umrechnung der Koordinaten liegen dort in der Verantwortung der
Anwendung.

`<model>.rknn.metadata.json` enthält die Klassennamen, die Eingabegröße, die
Aufgabe und die Zielplattform, also genau das, was eine Anwendung braucht, um
LibreYOLOs Postprocessing nachzubilden. Liefere es zusammen mit dem kompilierten
Modell aus.

Für eine Prüfung auf dem Host, die das Board nicht braucht, halte ein
ONNX-Artefakt mit derselben festen Form bereit und vergleiche es im Simulator,
wie oben.

## Einschränkungen

Vier Kombinationen kompilieren, und es sind Modellvarianten statt Familien:

| Variante | Aufgabe | Canvas | Target |
|---|---|---:|---|
| YOLO9-t | detect | 640 | RK3588 |
| YOLO9-E2E-t | detect | 640 | RK3588 |
| PicoDet-s | detect | 320 | RK3588 |
| YOLO-NAS-s | detect | 640 | RK3588 |

Alles andere wird vor der Kompilierung abgelehnt, mit der Meldung, dass RKNN in
dieser Version auf die exakt im Simulator getesteten Erkennungsvarianten
beschränkt ist. Ergebnisse, die nur die Kompilierung abdecken, gibt es auch für
andere Modelle, sie werden aber bewusst nicht als Unterstützung dargestellt: im
selben Messlauf ließ RF-DETR zwei `GridSample`-Knoten des Decoders nicht
abgesenkt, und D-FINE, RT-DETR, RT-DETRv2, RT-DETRv4, DEIM, DEIMv2 und EC
kompilierten und simulierten mit dekodierten Ausgaben, die deutlich falsch waren.

Batch 1, statische Formen, Opset 19. `half=True` wird abgelehnt, weil RKNN
LibreYOLOs `half`-Vertrag nicht abbildet, und `int8=True` wird abgelehnt, solange
keine Ergebnisse zu repräsentativer Kalibrierung und Aufgaben-Accuracy vorliegen.

Andere Rockchip-Targets werden abgelehnt: `rk3588` ist die einzige validierte
Plattform.

Das vollständige Raster aus Familien und Aufgaben findest du in
[der Export-Matrix](/docs/reference/export-matrix). Für eine einzelne
Kombination:

<code-tabs name="support" />
