---
title: ExecuTorch
seo_title: Esportare in ExecuTorch da LibreYOLO
description: >-
  Esporta un modello LibreYOLO in un programma .pte di ExecuTorch con delega a
  XNNPACK: forma fissa, batch 1, FP32 e il sidecar di metadati di cui ha
  bisogno.
lead: >-
  ExecuTorch esegue programmi PyTorch su dispositivi edge. LibreYOLO cattura il
  modello con torch.export in modalità strict, fa il lowering su XNNPACK e
  scrive il programma .pte insieme a un sidecar di metadati JSON come una cosa
  sola.
keywords:
  - esportare yolo executorch
  - programma .pte
  - xnnpack partitioner
  - torch.export strict
  - executorch runtime
  - inferenza pytorch edge
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="executorch")
    mono: true
  - label: Scrive
    value: Un programma .pte più un sidecar di metadati .pte.json
  - label: Extra
    value: 'pip install "libreyolo[executorch]"'
    mono: true
  - label: Si ricarica con
    value: LibreYOLO("weights/LibreYOLO9t.pte")
    mono: true
  - label: Forme
    value: Fisse. dynamic=True e batch != 1 vengono rifiutati.
  - label: Precisione
    value: Solo FP32. half=True e int8=True vengono rifiutati.
  - label: Delegato
    value: 'XNNPACK, CPU. delegate=''xnnpack'' è l''unico valore accettato.'
verification: >-
  Letto da libreyolo/export/executorch.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/executorch.py e pyproject.toml
  sul branch dev.
snippets:
  install:
    - label: Installazione
      language: bash
      code: |
        # Escluso da libreyolo[all] di proposito: ExecuTorch limita con quale
        # versione di Torch può essere abbinato.
        pip install "libreyolo[executorch]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Scrive weights/LibreYOLO9t.pte e weights/LibreYOLO9t.pte.json
        path = model.export(format="executorch", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format executorch --imgsz 640
    - label: Argomenti
      language: python
      code: |
        model.export(
            format="executorch",
            imgsz=640,             # int, oppure (altezza, larghezza)
            batch=1,               # qualsiasi altro valore solleva ValueError
            dynamic=False,         # True solleva ValueError
            delegate="xnnpack",    # l'unico valore accettato
            device="cpu",          # qualsiasi altro dispositivo solleva ValueError
            output_path=None,      # None scrive weights/<stem>.pte
        )
  run:
    - label: Tramite LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.pte")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Runtime ExecuTorch diretto
      language: python
      code: >
        import json

        from pathlib import Path


        import torch

        from executorch.runtime import Runtime


        runtime = Runtime.get()

        print(runtime.backend_registry.is_available("XnnpackBackend"))


        program =
        runtime.load_program(Path("weights/LibreYOLO9t.pte").read_bytes())

        method = program.load_method("forward")


        # Su questa via il preprocessing e il postprocessing sono affare tuo.

        outputs = method.execute((torch.zeros(1, 3, 640, 640),))

        print([tensor.shape for tensor in outputs])


        meta = json.load(open("weights/LibreYOLO9t.pte.json"))

        print(meta["model_family"], meta["task"], meta["executorch_delegate"])
  support:
    - label: Controllare una famiglia e un task prima di esportare
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: c2c354a76ee33157
---

## Installazione

<code-tabs name="install" />

Questo extra sta deliberatamente fuori da `libreyolo[all]`, perché ExecuTorch fissa
con quale versione di Torch funziona e installarlo trascinerebbe l'intero ambiente
su quella coppia. Installalo in un ambiente che sei disposto a vincolare.

Su Windows il passo di lowering chiama l'eseguibile `flatc` distribuito con
ExecuTorch. Se non è nel `PATH`, l'esportazione solleva un `RuntimeError` che lo
segnala, e la soluzione è eseguirla da una Developer PowerShell di Visual Studio 2022.

## Esportazione

<code-tabs name="export" />

La cattura è `torch.export.export(..., strict=True)`, cioè una vera cattura del
grafo con guard invece di un trace registrato. Le letture di scalari sull'host e il
controllo di flusso dipendente dai dati vengono rifiutati invece di essere fissati
in silenzio, quindi qui falliscono diverse famiglie che altrove si tracciano senza
problemi; i motivi sono registrati per combinazione nella matrice di supporto.

Il lowering esegue `to_edge_transform_and_lower` con il partitioner XNNPACK. Se il
risultato non contiene nessuna partizione delegata, l'esportazione solleva un errore
invece di etichettare come XNNPACK un programma che usa solo kernel portabili.

Il programma e il sidecar vengono scritti insieme. Entrambi vengono preparati,
entrambi vengono sostituiti, e un errore riporta tutto a com'era prima, così una
coppia incompleta non arriva mai sul disco.

## Eseguire l'artefatto

<code-tabs name="run" />

`LibreYOLO()` sceglie in base al suffisso `.pte` e restituisce lo stesso oggetto
`Results` del checkpoint. Il sidecar è obbligatorio al caricamento: senza
`<program>.pte.json` il backend solleva `FileNotFoundError`, perché il programma non
porta con sé nomi di classe, task né dimensione di input. Il backend controlla anche
che il runtime installato offra `XnnpackBackend` prima di caricare, e legge il
programma dai byte invece di mappare il file, il che evita di tenere un lock di file
di Windows per tutta la vita del backend.

Il secondo snippet è la via del runtime diretto. Lì il preprocessing, la decodifica,
l'NMS e il riscalamento delle coordinate diventano affare tuo.

## Vincoli

Batch 1, forma fissa, FP32, CPU. Sia `batch != 1` sia `dynamic=True` sollevano
`ValueError` prima che l'esportazione modifichi qualcosa, `half=True` e `int8=True`
vengono rifiutati durante la validazione, e un dispositivo diverso dalla CPU viene
respinto.

`delegate` accetta `"xnnpack"` e nient'altro in questa versione.

Le esportazioni di classificazione portano due chiavi di metadati extra, `crop_pct` e
`interpolation`, così il runtime può riprodurre la politica di ridimensionamento e
ritaglio centrale della famiglia.

Le voci bloccate indicano il fallimento concreto invece di una categoria. Il
rilevamento e la segmentazione di D-FINE arrivano a una lettura di `ContextVar` non
supportata nell'attenzione deformabile sotto cattura strict, e forzare la via manuale
del grid-sample serializza ma poi fallisce a runtime per un ordine di dimensioni non
valido su un tensore delegato. DEIM e DEIMv2 vengono catturati, passano il lowering e
si serializzano, poi falliscono durante l'esecuzione. La segmentazione semantica di
EoMT fallisce su un'espressione simbolica dipendente dai dati nel percorso delle
maschere. Il matting di BiRefNet viene catturato a 1024 per 1024 ma non ha una
variante out per `torchvision::deform_conv2d`. Il restauro di SwinIR si ricarica e poi
fallisce in `aten::alias_copy.out` per ordini di dimensioni non corrispondenti.

Per la griglia completa di famiglie e task, vedi
[la matrice di esportazione](/docs/reference/export-matrix). Per una singola combinazione:

<code-tabs name="support" />
