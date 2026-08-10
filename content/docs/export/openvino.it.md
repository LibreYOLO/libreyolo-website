---
title: OpenVINO
seo_title: "Esportare in OpenVINO IR da LibreYOLO"
description: "Converti un modello LibreYOLO in OpenVINO IR: la coppia model.xml e model.bin, la compressione dei pesi in FP16, INT8 con NNCF e inferenza su CPU, GPU o NPU."
lead: "OpenVINO IR è il formato di runtime di Intel: un grafo model.xml accanto a un blob di pesi model.bin. LibreYOLO esporta un ONNX intermedio, lo converte con ov.convert_model e scrive un metadata.yaml nella stessa directory."
keywords:
  - esportare yolo openvino
  - openvino ir
  - model.xml model.bin
  - ov.convert_model
  - quantizzazione int8 nncf
  - openvino npu
  - compress_to_fp16
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="openvino")'
    mono: true
  - label: Scrive
    value: "Una directory con model.xml, model.bin e metadata.yaml"
  - label: Extra
    value: 'pip install "libreyolo[onnx,openvino]"'
    mono: true
  - label: Si ricarica con
    value: 'LibreYOLO("weights/LibreYOLO9t_openvino")'
    mono: true
  - label: Forme
    value: "Segue l'ONNX intermedio: batch dinamico quando dynamic=True"
  - label: Precisione
    value: "FP32, compressione dei pesi in FP16 (half=True), INT8 tramite NNCF (int8=True con data=)"
verification: "Letto da libreyolo/export/openvino.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/openvino.py e pyproject.toml sul branch dev."
snippets:
  install:
    - label: Installazione
      language: bash
      code: |
        # L'IR viene convertito da un ONNX intermedio, quindi servono entrambi gli extra.
        pip install "libreyolo[onnx,openvino]"
    - label: INT8 richiede in più NNCF
      language: bash
      code: |
        pip install nncf
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Scrive la directory weights/LibreYOLO9t_openvino
        path = model.export(format="openvino")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format openvino
    - label: Argomenti
      language: python
      code: |
        model.export(
            format="openvino",
            imgsz=640,
            batch=1,
            dynamic=False,    # True mantiene un asse di batch dinamico nell'IR
            half=False,       # True salva i pesi in FP16
            int8=False,       # True esegue la quantizzazione post-training di NNCF
            data=None,        # obbligatorio quando int8=True
            output_path=None, # None scrive weights/<stem>_openvino
        )
  int8:
    - label: INT8 con dati di calibrazione
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="openvino",
            int8=True,
            data="coco128.yaml",   # obbligatorio: per questo formato non c'è un valore predefinito
            fraction=1.0,
        )
  run:
    - label: Tramite LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_openvino")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Selezionare il dispositivo
      language: python
      code: |
        from libreyolo import LibreYOLO

        # "auto" e "cpu" corrispondono alla CPU, "gpu" e "cuda" alla GPU,
        # qualsiasi altro valore viene passato in maiuscolo, per esempio "npu" -> NPU.
        model = LibreYOLO("weights/LibreYOLO9t_openvino", device="gpu")
    - label: OpenVINO puro
      language: python
      code: |
        import numpy as np
        import openvino as ov
        import yaml

        core = ov.Core()
        print(core.available_devices)

        compiled = core.compile_model("weights/LibreYOLO9t_openvino/model.xml", "CPU")
        outputs = compiled(np.zeros((1, 3, 640, 640), dtype=np.float32))
        print([tensor.shape for tensor in outputs.values()])

        # I nomi delle classi, il task e la dimensione di input stanno in metadata.yaml accanto all'IR.
        meta = yaml.safe_load(open("weights/LibreYOLO9t_openvino/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # Il preprocessing e il postprocessing su questo percorso sono a tuo carico.
  support:
    - label: Controllare una famiglia e un task prima di esportare
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Installazione

<code-tabs name="install" />

La conversione passa per un ONNX intermedio, quindi l'extra `onnx` fa parte del
requisito e non è un accessorio opzionale. NNCF si installa a parte e serve solo
per `int8=True`.

## Esportazione

<code-tabs name="export" />

L'artefatto è una directory, non un file. `weights/LibreYOLO9t_openvino` contiene
`model.xml`, `model.bin` e `metadata.yaml`, e `_fp16` viene inserito prima del
suffisso quando `half=True`. Sposta o copia l'intera directory: i tre file sono un
unico artefatto.

`half=True` imposta `compress_to_fp16` al salvataggio. È una compressione dei pesi
nell'IR, non un cambiamento della precisione di inferenza che il dispositivo
sceglie in esecuzione.

### INT8

<code-tabs name="int8" />

`int8=True` esegue la quantizzazione post-training di NNCF su un loader di
calibrazione LibreYOLO con il preset mixed, e `data` è obbligatorio: questo formato
non ha un ripiego a otto immagini. Se NNCF manca viene sollevato un `ImportError`
che indica il comando di installazione.

## Eseguire l'artefatto

<code-tabs name="run" />

`LibreYOLO()` riconosce qualsiasi directory che contenga `model.xml` e restituisce
lo stesso oggetto `Results` del checkpoint, leggendo i nomi delle classi, il task,
la dimensione di input e lo schema della posa da `metadata.yaml`.

La stringa del dispositivo viene mappata, non passata direttamente. `auto` e `cpu`
compilano entrambi per la CPU, `gpu` e `cuda` compilano entrambi per la GPU, e
qualsiasi altro valore viene convertito in maiuscolo e consegnato a OpenVINO: è
così che si raggiunge un target NPU.

Il terzo snippet è per chi non ha LibreYOLO installato. Lì il preprocessing, il
decoding, l'NMS e il riscalamento delle coordinate diventano a tuo carico, e i nomi
delle classi esistono solo in `metadata.yaml`.

## Vincoli

Un IR senza il suo `metadata.yaml` si carica comunque, ma il backend ripiega su 80
classi e sul task di rilevamento, il che è sbagliato per qualsiasi altro caso.
Mantieni la directory intatta.

Bloccati prima del tracing: la segmentazione YOLO9, la segmentazione RTMDet-Ins,
SSD, il rilevamento con Faster R-CNN e RetinaNet, e il matting con BiRefNet o
FeyNobg, dove OpenVINO 2026.2 non riesce ad abbassare l'operazione ONNX standard
`DeformConv-19` del decoder di matte condiviso.

Quando una combinazione non è né validata né bloccata, il percorso del converter è
disponibile e il progetto non ha registrato una parità di runtime OpenVINO per
essa. Diverse combinazioni sono validate con un contesto esplicito allegato, per
esempio la segmentazione semantica DeepLabV3 a un input fisso di 520 per 520 su
OpenVINO 2026.2 con la precisione di inferenza predefinita della CPU, e lo sguardo
L2CS a un ritaglio del volto fisso di 448 per 448. `libreyolo formats` stampa
quel contesto per ogni combinazione.

Per la griglia completa di famiglie e task, vedi
[la matrice di esportazione](/docs/reference/export-matrix). Per una sola
combinazione:

<code-tabs name="support" />
