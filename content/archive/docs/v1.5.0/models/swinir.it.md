---
title: SwinIR
families:
  - swinir
seo_title: 'SwinIR: super-risoluzione 4x delle immagini in LibreYOLO'
description: >-
  Usa SwinIR in LibreYOLO per la super-risoluzione 4x delle immagini. Installa,
  fai predizioni, valida ed esporta i checkpoint lightweight, medium e large.
lead: >-
  Una rete Swin Transformer per il restauro delle immagini. LibreYOLO include
  inferenza e validazione per i suoi checkpoint di super-risoluzione 4x: il
  generatore lightweight ufficiale e i generatori real-world medium e large.
keywords:
  - SwinIR
  - Swin Transformer
  - image super resolution
  - image restoration
  - residual Swin Transformer block
  - super risoluzione immagini python
  - aumentare la risoluzione di un'immagine
  - ingrandire immagine senza perdere qualità
  - restauro delle immagini
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwinIRm-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 'A tasselli, per immagini grandi'
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRl-restore.pt")

        # tile divide il forward pass in tasselli sovrapposti e ricompone le
        # giunture; tile_pad è l'alone aggiunto attorno a ogni tassello
        # prima che venga ritagliato via. Sono entrambi argomenti keyword
        # solo Python, non flag della CLI.
        result = model("large-photo.jpg", tile=512, tile_pad=16, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwinIRm-restore.pt data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")

        # se omesso, imgsz assume una piccola dimensione di patch interna,
        # non la tua risoluzione di lavoro, quindi passa la dimensione che
        # il tuo deployment dà davvero in pasto al modello.
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwinIRm-restore.pt format=onnx imgsz=512
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base all'estensione del file, quindi un
        # artefatto esportato si carica come qualsiasi checkpoint e
        # restituisce lo stesso oggetto Results.
        model = LibreYOLO("LibreSwinIRm-restore.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.restored.array.shape)
source_hash: 87fc3d5524480eec
---

## Installazione

SwinIR non richiede nessun extra opzionale. Tutto quello che importa è
nell'installazione base.

```bash
pip install libreyolo
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella
cache locale.

<code-tabs name="predict" />

Un risultato di restauro non porta box; `result.restored` è un'immagine RGB
uint8 densa `(H, W, 3)`, su un canvas grande 4 volte l'input in ciascuna
dimensione. `save=True` scrive direttamente quell'immagine invece di un plot
annotato. L'input viene riempito con padding fino a un multiplo di 8 invece di
essere ridimensionato, quindi la predizione gira alla risoluzione originale
della foto; una sorgente più grande di quanto la memoria consenta si può
dividere con `tile` e `tile_pad`, che ricompongono le giunture dei tasselli
nell'output. Vedi [predizione](/docs/predict) per sorgenti, streaming e
gestione dei risultati.

## Varianti

Tre dimensioni, tutte fisse a un upscaling 4x. `s` è il generatore lightweight
ufficiale, con quattro stage di residual Swin Transformer block (RSTB) e
upsampling pixel-shuffle diretto. `m` e `l` sono i generatori real-world medium
e large, con sei e nove stage RSTB e un upsampler nearest-neighbor più
convoluzione, costruito per le degradazioni del mondo reale e non solo per la
riduzione bicubica.

## Validazione

`val()` misura PSNR e SSIM tra l'output restaurato e un'immagine target pulita,
entrambi calcolati in RGB sul canvas originale, senza ritaglio dei bordi e
senza ridimensionamento. SSIM usa una finestra gaussiana 11x11 con sigma 1.5,
mediata sui tre canali di colore.

<code-tabs name="val" />

L'argomento dataset è un YAML che accoppia una cartella di immagini di input
degradate con una cartella di immagini target pulite alla stessa risoluzione;
vedi [formati dei dataset](/docs/reference/dataset-formats) per le chiavi
esatte.

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica con `LibreYOLO()` in base all'estensione del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. ExecuTorch e tutti i formati che la matrice
segna come bloccati non sono disponibili per questa famiglia; ONNX, TorchScript,
TensorRT, OpenVINO e TFLite invece sì. [Esportazione](/docs/export) elenca gli
argomenti che ogni formato accetta e gli extra che qualcuno di essi aggiunge.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenza

<provenance-box></provenance-box>

## Citazione

<citation-block />
