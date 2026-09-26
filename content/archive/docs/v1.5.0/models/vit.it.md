---
title: ViT
families:
  - vit
seo_title: 'ViT: esegui i classici classificatori Vision Transformer in LibreYOLO'
description: >-
  Fai predizioni, validazione ed esportazione con i classificatori ViT in
  LibreYOLO. Pesi AugReg con licenza Apache-2.0; il fine-tuning non è ancora
  supportato.
lead: >-
  Il classico Vision Transformer: un transformer puro applicato a patch di
  immagine di dimensione fissa, con un class token appreso e senza convoluzioni.
  LibreYOLO include quattro dimensioni preaddestrate con AugReg per la
  classificazione di immagini.
keywords:
  - ViT
  - Vision Transformer
  - AugReg
  - image classification python
  - classificare immagini python
  - transformer classifier
  - classificazione immagini deep learning
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreViTti-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreViTti-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")

        # data è la cartella radice con gli split train/ e val/ organizzati in
        # cartelle per classe (struttura ImageFolder), non uno YAML di dataset.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreViTti-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreViTti-cls.pt format=onnx
        libreyolo export model=LibreViTti-cls.pt format=tensorrt half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file: un artefatto esportato
        # si carica come qualsiasi checkpoint e restituisce lo stesso Results.
        model = LibreYOLO("LibreViTti-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: f63e98454913765a
---

## Installazione

ViT non richiede nessun extra opzionale. Tutto quello che importa è già
nell'installazione base.

```bash
pip install libreyolo
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella
cache locale.

<code-tabs name="predict" />

Un classificatore restituisce `result.probs` invece di `result.boxes`: `top1`
e `top5` danno gli indici delle classi, `top1conf` e `top5conf` le rispettive
confidenze. Il preprocessing ridimensiona e ritaglia al centro fino a un input
fisso da 224px, seguendo la ricetta di valutazione AugReg di timm:
interpolazione bicubica con una crop fraction di 0.9. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Varianti

Quattro dimensioni, da tiny a large, che condividono un unico grafo fisso a
224px con patch 16 e differiscono per larghezza dell'embedding e profondità del
transformer. LibreYOLO include questa famiglia solo per l'inferenza: predizione,
validazione top-1/top-5 in stile ImageNet ed esportazione sono supportate,
mentre la ricetta di fine-tuning di AugReg non è implementata.

## Validazione

`val()` gira su uno split in stile ImageFolder (una cartella con le
sottocartelle `train/` e `val/`, una cartella per classe) e restituisce
l'accuratezza top-1 e top-5.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica con `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. [Esportazione](/docs/export) elenca gli
argomenti accettati da ogni formato e quelli in più che alcuni aggiungono.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
