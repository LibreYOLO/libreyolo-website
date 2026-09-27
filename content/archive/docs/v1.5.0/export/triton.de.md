---
title: Triton Inference Server
seo_title: Ein LibreYOLO-Modell auf NVIDIA Triton bereitstellen
description: >-
  Einen LibreYOLO-ONNX-Export über NVIDIA Triton bereitstellen: das Layout des
  Model Repositorys, die erzeugte config.pbtxt und Vorhersagen gegen eine
  HTTP-Modell-URL.
lead: >-
  Triton Inference Server hostet ein Model Repository und beantwortet
  Inferenzanfragen über HTTP. LibreYOLO exportiert den ONNX-Graphen, erzeugt
  eine config.pbtxt, die die Export-Metadaten als einen einzelnen
  Triton-Parameter transportiert, und behandelt eine Modell-URL wie einen
  ladbaren Modellpfad.
keywords:
  - libreyolo triton
  - triton inference server
  - config.pbtxt erzeugen
  - tritonclient http
  - triton model repository
  - yolo inferenz über http
last_verified: 1.5.0
meta:
  - label: Aufruf
    value: 'LibreYOLO("http://127.0.0.1:8000/yolo9")'
    mono: true
  - label: Hilfsfunktion
    value: >-
      create_triton_config(onnx_path, config_path, model_name=...,
      max_batch_size=8)
    mono: true
  - label: Extra
    value: 'pip install "libreyolo[onnx,triton]"'
    mono: true
  - label: Protokoll
    value: >-
      Nur HTTP- und HTTPS-V2-Inferenz. Kein gRPC, keine Authentifizierung, kein
      Shared Memory, kein Laden und Entladen von Modellen.
  - label: Timeouts
    value: Verbindungs- und Netzwerk-Timeouts liegen standardmäßig bei 30 Sekunden
verification: >-
  Gelesen aus libreyolo/backends/triton.py, libreyolo/models/__init__.py,
  docs/triton.md und pyproject.toml im dev-Branch. Die Container-Befehle sind
  die gepinnten aus docs/triton.md.
snippets:
  install:
    - label: Installation
      language: bash
      code: |
        pip install "libreyolo[onnx,triton]"
  repo:
    - label: In das Repository-Layout exportieren
      language: python
      code: |
        from pathlib import Path

        from libreyolo import LibreYOLO

        model_dir = Path("triton_repo/yolo9/1")
        model_dir.mkdir(parents=True, exist_ok=True)

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            output_path=str(model_dir / "model.onnx"),
            dynamic=True,
            simplify=False,
        )
    - label: config.pbtxt erzeugen
      language: python
      code: |
        from libreyolo import create_triton_config

        create_triton_config(
            "triton_repo/yolo9/1/model.onnx",
            "triton_repo/yolo9/config.pbtxt",
            model_name="yolo9",
            max_batch_size=8,
        )
    - label: Resultierendes Layout
      language: text
      code: |
        triton_repo/
          yolo9/
            config.pbtxt
            1/
              model.onnx
  serve:
    - label: Server starten
      language: bash
      code: |
        docker run --rm --name libreyolo-triton \
          -p 8000:8000 -p 8002:8002 \
          -v "$(pwd)/triton_repo:/models:ro" \
          nvcr.io/nvidia/tritonserver:26.04-py3 \
          tritonserver --model-repository=/models --exit-on-error=true
    - label: Auf Bereitschaft warten
      language: bash
      code: >
        until curl --fail --silent http://127.0.0.1:8000/v2/health/ready; do
        sleep 1; done
    - label: Stoppen
      language: bash
      code: |
        docker stop libreyolo-triton
  run:
    - label: Gegen das bereitgestellte Modell vorhersagen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9")
        result = remote.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Mit dem lokalen Modell vergleichen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9").predict(SAMPLE_IMAGE)
        native = LibreYOLO("LibreYOLO9t.pt").predict(SAMPLE_IMAGE)

        print(len(remote.boxes), len(native.boxes))
        print(remote.boxes.xyxy[:3])
        print(native.boxes.xyxy[:3])
    - label: Version fixieren oder Timeout ändern
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.backends.triton import TritonBackend

        # Ein zweites Pfadsegment wählt die Modellversion. Ohne es
        # entscheidet Tritons konfigurierte Version Policy.
        pinned = LibreYOLO("http://127.0.0.1:8000/yolo9/1")

        # Verbindungs- und Netzwerk-Timeout: standardmäßig 30 Sekunden.
        patient = TritonBackend("http://127.0.0.1:8000/yolo9", timeout=120)
source_hash: 0652e4faf0224df3
---

## Installation

<code-tabs name="install" />

Das Extra `triton` installiert `tritonclient[http]`. Die Extras für gRPC und
Shared Memory sind absichtlich ausgeschlossen: Diese Integration macht
ausschließlich HTTP- und HTTPS-V2-Inferenz. `onnx` wird gebraucht, weil sowohl
das bereitgestellte Artefakt als auch der Config-Generator von einem
ONNX-Graphen ausgehen.

## Aufbau des Model Repositorys

Exportiere mit dynamischer Batch-Achse, in das Verzeichnis-Layout, das Triton
erwartet.

<code-tabs name="repo" />

Triton behält die ONNX-Custom-Metadaten in seiner Model-Config-Antwort nicht
bei, deshalb müssen die vollständigen exportierten Metadaten auf einem anderen
Weg mitreisen. `create_triton_config` kodiert sie als einen einzelnen
JSON-String-Parameter namens `libreyolo_metadata` in `config.pbtxt`, gibt die
Input- und Output-Deklarationen in Graph-Reihenfolge aus, kümmert sich um das
JSON-Escaping und fixiert das Modell auf `KIND_CPU`.

Die Hilfsfunktion validiert vor dem Schreiben. Sie verlangt genau einen
ONNX-Graph-Input, mindestens einen Output, auflösbare Tensor-Shapes und
Metadaten, deren `names`-Map jeden Klassenindex von 0 bis `nc - 1` definiert.
Ein Modell, das eine dieser Prüfungen nicht besteht, wird zur Config-Zeit
abgelehnt statt beim ersten Request.

`max_batch_size: 8` passt zu einem dynamischen Export und lässt den Server bis
zu acht Bilder pro Request bündeln. Für einen ONNX-Graphen mit fester
Batch-Größe 1 nimm `max_batch_size=0`; LibreYOLO schickt die Bilder dann
nacheinander.

## Start des Servers

<code-tabs name="serve" />

Die Befehle fixieren Triton Server 26.04 und lassen die GPU-Flags von Docker
bewusst weg, da `KIND_CPU` in der erzeugten Config eine GPU-Platzierung ohnehin
verhindert.

## Ausführung des Artefakts

Eine Triton-Modell-URL ist ein Modellpfad. `LibreYOLO()` prüft auf ein `http`-
oder `https`-Schema, bevor irgendeine lokale Pfadverarbeitung greift, und gibt
ein Backend zurück, das mit dem Server spricht. Die Aufrufstelle ist damit
identisch mit der für einen lokalen Checkpoint, und das zurückkommende
`Results`-Objekt ebenfalls.

<code-tabs name="run" />

Die URL-Form ist `http(s)://host:port/model` mit einem optionalen
Versionssegment. Der Port muss explizit angegeben sein. Eingebettete
Zugangsdaten, ein Query-String und ein Fragment werden allesamt abgelehnt,
ebenso ein Pfad mit mehr als zwei Segmenten.

`device` wird angenommen und mit einer Log-Zeile ignoriert, weil die
Platzierung die Entscheidung des Servers ist.

## Einschränkungen

Das Backend scheitert mit einem direkten Fehler statt mit einem
verschlechterten Ergebnis, wenn der Vertrag nicht erfüllt ist: fehlende
LibreYOLO-Metadaten in der Model Config, mehr als ein Modell-Input, eine
Abweichung zwischen den konfigurierten Outputs und den Modell-Metadaten, ein
nicht unterstützter Input-Datentyp oder ein Server beziehungsweise Modell, das
nicht bereit ist.

Außerhalb des Vertrags in dieser Version: gRPC, Authentifizierung, Shared
Memory sowie das Laden oder Entladen von Modellen über die API.

Jedes Format, das Triton selbst unterstützt, lässt sich bereitstellen, aber der
Metadaten-Parameter und die erzeugte Config sind hier auf ONNX zugeschnitten,
also führt der LibreYOLO-Weg über [ONNX](/docs/export/onnx) in das Repository.
Für eine vollständige Video-Pipeline statt eines Request-Response-Servers siehe
[DeepStream](/docs/export/deepstream).
