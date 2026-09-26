---
title: Real-ESRGAN
families:
  - realesrgan
seo_title: 'Real-ESRGAN: super-risoluzione delle immagini in LibreYOLO'
description: >-
  Usa Real-ESRGAN in LibreYOLO per la super-risoluzione pratica delle immagini a
  4x, 2x e con un livello 4x veloce. Installa, fai predizioni, valida ed
  esporta.
lead: >-
  Un upscaler pratico di super-risoluzione blind, addestrato su degradazioni
  sintetiche invece che solo sulla riduzione bicubica. LibreYOLO include
  inferenza e validazione per i suoi checkpoint 4x, 2x e 4x veloce.
keywords:
  - Real-ESRGAN
  - RRDBNet
  - SRVGGNetCompact
  - image super resolution
  - migliorare la qualità di un'immagine python
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

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRealESRGANx4-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 'A tasselli, per immagini grandi'
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")

        # tile divide il forward pass in tasselli sovrapposti e ricompone le
        # giunture; tile_pad è l'alone aggiunto attorno a ogni tassello
        # prima che venga ritagliato via. Sono entrambi argomenti keyword
        # solo Python, non flag della CLI.
        result = model("large-photo.jpg", tile=512, tile_pad=10, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: >
        libreyolo val model=LibreRealESRGANx4-restore.pt
        data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")

        # se omesso, imgsz assume una piccola dimensione di patch interna,
        # non la tua risoluzione di lavoro, quindi passa la dimensione che
        # il tuo deployment dà davvero in pasto al modello.
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRealESRGANx4-restore.pt format=onnx
        imgsz=512
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base all'estensione del file, quindi un
        # artefatto esportato si carica come qualsiasi checkpoint e
        # restituisce lo stesso oggetto Results.
        model = LibreYOLO("LibreRealESRGANx4-restore.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.restored.array.shape)
source_hash: f0efb4f65d38e22d
---

## Installazione

Real-ESRGAN non richiede nessun extra opzionale. Tutto quello che importa è
nell'installazione base.

```bash
pip install libreyolo
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella
cache locale.

<code-tabs name="predict" />

Un risultato di restauro non porta box; `result.restored` è un'immagine RGB
uint8 densa `(H, W, 3)`, su un canvas grande `Results.restore_scale` volte
l'input in ciascuna dimensione. `save=True` scrive direttamente quell'immagine
invece di un plot annotato. L'input viene convertito in RGB e l'eventuale
canale alfa viene scartato. Una sorgente più grande di quanto la memoria
consenta si può dividere con `tile` e `tile_pad`, che ricompongono le giunture
dei tasselli nell'output. Vedi [predizione](/docs/predict) per sorgenti,
streaming e gestione dei risultati.

## Varianti

Tre checkpoint, che prendono il nome dal loro fattore di upscaling. `x4` è
RRDBNet (`RealESRGAN_x4plus`), 23 blocchi densi residual-in-residual, la scelta
predefinita per la qualità a 4x. `x2` è la stessa architettura RRDBNet a 2x.
`x4t` è SRVGGNetCompact (`realesr-general-x4v3`), un generatore più piccolo e
più veloce, costruito per il video e per usi a bassa latenza a 4x. Il modello
general-purpose del progetto upstream include anche una rete accoppiata di
denoise strength, miscelata al momento dell'inferenza; quella manopola di
intensità non fa parte di questo port, che esegue il generatore `x4t` di base.

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
restituisce lo stesso `Results`. [Esportazione](/docs/export) elenca gli
argomenti che ogni formato accetta e gli extra che qualcuno di essi aggiunge.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenza

<provenance-box></provenance-box>

## Citazione

<citation-block />
