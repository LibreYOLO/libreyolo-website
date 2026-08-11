---
title: FCN
families:
  - fcn
seo_title: 'FCN: fai predizioni ed esporta un FCN ResNet con licenza BSD-3-Clause'
description: >-
  Usa FCN in LibreYOLO per la segmentazione semantica. Installa, fai predizioni,
  valida ed esporta i checkpoint FCN con ResNet dilatata di torchvision.
lead: >-
  Un classificatore denso pixel per pixel che sostituisce i layer fully
  connected di un detector con delle convoluzioni, così da produrre una mappa di
  classi a piena risoluzione invece dei box. LibreYOLO lo distribuisce solo per
  la segmentazione semantica.
keywords:
  - FCN
  - fully convolutional network
  - semantic segmentation python
  - segmentazione semantica immagini
  - ResNet
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreFCNr50.pt")

        result = model(SAMPLE_IMAGE, save=True)


        mask = result.semantic_mask

        print(mask.data.shape)   # (H, W) id delle classi

        print(mask.classes)      # id delle classi presenti nell'immagine,
        ordinati
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCNr50.pt format=onnx
        libreyolo export model=LibreFCNr50.pt format=tensorrt half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreFCNr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 7776b0fc85a208fb
---

## Installazione

FCN non richiede nessun extra opzionale. Tutto ciò che importa è già
nell'installazione di base.

```bash
pip install libreyolo
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella
cache locale.

<code-tabs name="predict" />

La segmentazione semantica restituisce un id di classe per pixel, non dei
bounding box, quindi `result.semantic_mask` porta un array `(H, W)` in `.data`
e la lista degli id di classe presenti nell'immagine in `.classes`. `conf`,
`iou` e `max_det` sono accettati per parità di API ma non hanno alcun effetto:
il modello assegna una classe a ogni pixel per argmax, senza soglia di
confidenza né passo di NMS. Vedi [predizione](/docs/predict) per sorgenti,
streaming e gestione dei risultati.

## Varianti

Due profondità di ResNet, entrambe con input fisso a 520 px. Il grafo di
inferenza della libreria è l'FCN con ResNet dilatata di torchvision, non la
rete FCN-8s basata su VGG con skip connection del paper originale.

LibreYOLO non addestra FCN: `train()` solleva `NotImplementedError` per questa
famiglia, cosa che il [livello di supporto](/docs/models) qui sopra segnala
come solo inferenza. I due checkpoint pubblicati sono i pesi di torchvision
addestrati su COCO, convertiti per il loader di LibreYOLO.

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
