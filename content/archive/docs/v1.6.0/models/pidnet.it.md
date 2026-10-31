---
title: PIDNet
families:
  - pidnet
seo_title: >-
  PIDNet: fai predizioni ed esporta la segmentazione in tempo reale con licenza
  MIT
description: >-
  Usa PIDNet in LibreYOLO per la segmentazione semantica in tempo reale.
  Installa, fai predizioni, valida ed esporta i checkpoint s/m/l per Cityscapes
  con licenza MIT.
lead: >-
  Una rete di segmentazione semantica a tre rami che aggiunge un ramo dedicato
  ai contorni a un'architettura ispirata al controllo
  proporzionale-integrale-derivativo, pensata per l'inferenza in tempo reale.
  LibreYOLO la distribuisce solo per la segmentazione semantica.
keywords:
  - PIDNet
  - real-time semantic segmentation
  - segmentazione semantica tempo reale
  - Cityscapes
  - segmentazione semantica python
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibrePIDNets-sem.pt")

        result = model(SAMPLE_IMAGE, save=True)


        mask = result.semantic_mask

        print(mask.data.shape)   # (H, W) id delle classi

        print(mask.classes)      # id delle classi presenti nell'immagine,
        ordinati
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePIDNets-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePIDNets-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibrePIDNets-sem.pt format=onnx
        libreyolo export model=LibrePIDNets-sem.pt format=tensorrt half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibrePIDNets-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 489db64a39e3a61a
---

## Installazione

PIDNet non richiede nessun extra opzionale. Tutto ciò che importa è già
nell'installazione di base.

```bash
pip install libreyolo
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella
cache locale. Il suffisso `-sem` nel nome del file è obbligatorio per questa
famiglia.

<code-tabs name="predict" />

La segmentazione semantica restituisce un id di classe per pixel, non dei
bounding box, quindi `result.semantic_mask` porta un array `(H, W)` in `.data`
e la lista degli id di classe presenti nell'immagine in `.classes`. `conf`,
`iou` e `max_det` sono accettati per parità di API ma non hanno alcun effetto:
il modello assegna una classe a ogni pixel per argmax, senza soglia di
confidenza né passo di NMS. Vedi [predizione](/docs/predict) per sorgenti,
streaming e gestione dei risultati.

## Varianti

Tre dimensioni, tutte con input fisso a 1024 px. I checkpoint pubblicati sono
conversioni dei pesi ufficiali di PIDNet per Cityscapes, 19 classi.

LibreYOLO non addestra PIDNet: `train()` solleva `NotImplementedError` per
questa famiglia, cosa che il [livello di supporto](/docs/models) qui sopra
segnala come solo inferenza.

## Validazione

`val()` restituisce `metrics/mIoU` e `metrics/pixel_accuracy`, misurati su
qualsiasi dataset nel formato con cui hai addestrato.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica con `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. [Esportazione](/docs/export) elenca gli
argomenti accettati da ogni formato.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
