---
title: DeepLabv3
families: [deeplabv3]
seo_title: "DeepLabv3: fai predizioni ed esporta la segmentazione semantica ASPP"
description: "Usa DeepLabv3 in LibreYOLO per la segmentazione semantica. Installa, fai predizioni, valida ed esporta i checkpoint ResNet e MobileNetV3 di torchvision."
lead: "Una rete di segmentazione semantica che aggrega le feature a diversi tassi di dilatazione in parallelo (atrous spatial pyramid pooling) prima di classificare ogni pixel. LibreYOLO la distribuisce solo per la segmentazione semantica."
keywords: [DeepLabv3, ASPP, "atrous spatial pyramid pooling", "semantic segmentation python", "segmentazione semantica immagini"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) id delle classi
        print(mask.classes)      # id delle classi presenti nell'immagine, ordinati
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDeepLabv3r50-sem.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeepLabv3r50-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDeepLabv3r50-sem.pt format=onnx
        libreyolo export model=LibreDeepLabv3r50-sem.pt format=tensorrt half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreDeepLabv3r50-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
---

## Installazione

DeepLabv3 non richiede nessun extra opzionale. Tutto ciò che importa è già
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

Tre backbone: ResNet-50 dilatata, ResNet-101 dilatata e MobileNetV3-Large
dilatata. Questo è DeepLabv3, non DeepLabv3+, quindi non c'è nessuno stadio di
decoder né raffinamento con CRF, in linea con l'implementazione di torchvision
e non con il codice di riferimento del paper originale.

LibreYOLO non addestra DeepLabv3: `train()` solleva `NotImplementedError` per
questa famiglia, cosa che il [livello di supporto](/docs/models) qui sopra
segnala come solo inferenza. I tre checkpoint pubblicati sono i pesi di
torchvision addestrati su COCO con le etichette di VOC, convertiti per il
loader di LibreYOLO.

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
