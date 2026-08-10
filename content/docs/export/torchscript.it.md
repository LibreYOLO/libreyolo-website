---
title: TorchScript
seo_title: Esportare in TorchScript da LibreYOLO
description: >-
  Esporta un modello LibreYOLO in TorchScript: un archivio .torchscript
  tracciato con i metadati LibreYOLO all'interno, caricabile da Python o da
  libtorch.
lead: >-
  TorchScript è il formato a grafo serializzato di PyTorch. LibreYOLO traccia il
  modello con torch.jit.trace e salva il risultato insieme a un file extra
  libreyolo_metadata.json, così l'archivio porta con sé la famiglia, il task, i
  nomi delle classi e la dimensione di input.
keywords:
  - esportare yolo torchscript
  - torch.jit.trace
  - torch.jit.load
  - deployment libtorch
  - metadati torchscript
  - extra_files
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="torchscript")
    mono: true
  - label: Scrive
    value: Un archivio .torchscript con un file extra libreyolo_metadata.json
  - label: Extra
    value: Nessuno. TorchScript è incluso in PyTorch.
  - label: Si ricarica con
    value: LibreYOLO("weights/LibreYOLO9t.torchscript")
    mono: true
  - label: Forme
    value: Fisse. Il grafo viene tracciato con una sola forma di input.
  - label: Precisione
    value: 'FP32, FP16 (half=True). Nessun INT8.'
verification: >-
  Letto da libreyolo/export/torchscript.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py e libreyolo/backends/torchscript.py sul branch
  dev.
snippets:
  install:
    - label: Installazione
      language: bash
      code: |
        pip install libreyolo
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Scrive weights/LibreYOLO9t.torchscript
        path = model.export(format="torchscript")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format torchscript
    - label: Argomenti
      language: python
      code: |
        model.export(
            format="torchscript",
            imgsz=640,        # int, oppure (altezza, larghezza)
            batch=1,
            half=False,       # pesi e attivazioni in FP16
            device=None,      # None traccia su CPU per questo formato
            output_path=None, # None scrive weights/<stem>.torchscript
        )

        # dynamic viene accettato, ma l'archivio è sempre un tracing a forma
        # fissa, e i metadati incorporati registrano dynamic=False in ogni caso.
  run:
    - label: Tramite LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.torchscript")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: PyTorch puro
      language: python
      code: |
        import json

        import torch

        extra_files = {"libreyolo_metadata.json": ""}
        module = torch.jit.load(
            "weights/LibreYOLO9t.torchscript",
            map_location="cpu",
            _extra_files=extra_files,
        )
        module.eval()

        metadata = json.loads(extra_files["libreyolo_metadata.json"])
        print(metadata["model_family"], metadata["task"], metadata["imgsz"])

        # Su questa via il preprocessing e il postprocessing sono a tuo carico.
        with torch.no_grad():
            out = module(torch.zeros(1, 3, 640, 640))
        print(out.shape if torch.is_tensor(out) else [t.shape for t in out])
  support:
    - label: Controllare una famiglia e un task prima di esportare
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 286a082969ccd604
---

## Installazione

<code-tabs name="install" />

TorchScript non ha bisogno di nulla oltre all'installazione base, perché
`torch.jit` è incluso in PyTorch. È l'unico target di esportazione senza
dipendenze opzionali e senza convertitore esterno, il che ne fa una prima
verifica utile quando una toolchain più lunga fallisce.

## Esportazione

<code-tabs name="export" />

Il tracing gira su CPU a meno che non venga indicato un dispositivo, e l'archivio
viene scritto in `weights/` con lo stem del checkpoint quando `output_path` è
omesso.

Il controllo di ri-tracing che `torch.jit.trace` esegue di norma è disattivato.
Diversi wrapper di esportazione mettono in cache anchor dipendenti dalla forma
durante il primo forward, quindi un secondo tracing osserva un percorso Python
diverso anche se il grafo a forma fissa registrato è corretto. I test di parità
validano invece direttamente il modulo salvato.

I metadati non vivono in un sidecar. `torch.jit.save` salva
`libreyolo_metadata.json` dentro l'archivio, e `torch.jit.load` li restituisce
attraverso `_extra_files`.

## Eseguire l'artefatto

<code-tabs name="run" />

`LibreYOLO()` instrada in base al suffisso `.torchscript` e restituisce lo stesso
oggetto `Results` del checkpoint da cui proviene. Con `device="auto"` il modulo
viene mappato su CUDA quando è disponibile, poi su MPS, poi su CPU.

Il secondo snippet è la via per chi non ha LibreYOLO installato, e per il
deployment in C++ tramite libtorch, dove lo stesso archivio si carica con
`torch::jit::load`. Lì il preprocessing, il decoding, l'NMS e il riscalamento
delle coordinate diventano a tuo carico. Il file extra dei metadati resta
leggibile, ed è l'unico posto in cui esistono i nomi delle classi.

## Vincoli

Il grafo è un tracing con una sola forma di input. `dynamic=True` viene accettato
per simmetria di interfaccia ma non cambia nulla, e i metadati incorporati
riportano `dynamic=False` così un backend non assume mai un asse che non può
usare. Esporta un secondo archivio per una seconda risoluzione.

`half=True` converte in FP16 il modello e l'input del tracing. Non esiste una via
INT8: `int8=True` solleva `NotImplementedError` durante la validazione.

Un `imgsz` rettangolare funziona per le famiglie YOLO9, HRNet, NAFNet e
Real-ESRGAN, ed è rifiutato per le famiglie con un contratto quadrato fisso.

Cinque combinazioni vengono rifiutate prima del tracing. La segmentazione con
YOLO9, perché in LibreYOLO YOLO9 fa solo rilevamento. La segmentazione con
RTMDet-Ins, il cui decoding delle maschere a kernel dinamico non ha un contratto
di runtime esportato. Il rilevamento con SSD, Faster R-CNN e RetinaNet, i cui
grafi a lunghezza variabile o ad anchor dinamici hanno evidenze di parità solo
attraverso il contratto di ONNX Runtime.

Per la griglia completa di famiglie e task, vedi
[la matrice di esportazione](/docs/reference/export-matrix). Per una
sola combinazione:

<code-tabs name="support" />
