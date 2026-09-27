---
title: VGG
families:
  - vgg
seo_title: 'VGG: esegui i classificatori di immagini VGG-16/19 in LibreYOLO'
description: >-
  Fai predizioni, valida ed esporta i classificatori VGG con LibreYOLO. Pesi
  torchvision con licenza BSD-3-Clause; il fine-tuning non è ancora supportato.
lead: >-
  VGG è un classificatore di immagini convoluzionale costruito con pile uniformi
  di piccole convoluzioni 3x3 al posto di filtri più grandi. LibreYOLO
  distribuisce le dimensioni a 16 e 19 layer, semplici e con batch
  normalization, per la classificazione di immagini.
keywords:
  - VGG
  - VGG-16
  - VGG-19
  - image classification python
  - classificazione immagini python
  - rete neurale convoluzionale
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreVGG16-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreVGG16-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")

        # data è la directory radice con gli split train/ e val/ in cartelle
        # per classe (layout ImageFolder), non uno YAML del dataset.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreVGG16-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreVGG16-cls.pt format=onnx
        libreyolo export model=LibreVGG16-cls.pt format=tensorrt half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreVGG16-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 26eb6ff5811533fd
---

## Installazione

VGG non richiede nessun extra opzionale. Tutto ciò che importa è già
nell'installazione di base.

```bash
pip install libreyolo
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella
cache locale.

<code-tabs name="predict" />

Un classificatore restituisce `result.probs` invece di `result.boxes`: `top1`
e `top5` danno gli indici delle classi, `top1conf` e `top5conf` le rispettive
confidenze. La predizione gira a un input fisso di 224px e solleva un errore se
passi un `imgsz` diverso. Vedi [predizione](/docs/predict) per sorgenti,
streaming e gestione dei risultati.

## Varianti

Quattro dimensioni: 16 e 19 layer convoluzionali, ognuna con una variante
semplice e una con batch normalization. I pesi distribuiti sono quelli
dell'addestramento da zero su ImageNet fatto in seguito da torchvision, non
conversioni della release Caffe originale del gruppo di Oxford del 2014.
LibreYOLO distribuisce questa famiglia solo per l'inferenza: predizione,
validazione top-1/top-5 in stile ImageNet ed esportazione sono supportate,
mentre il fine-tuning non è implementato.

## Validazione

`val()` gira su uno split in stile ImageFolder (una directory con sottocartelle
`train/` e `val/`, una cartella per classe) e restituisce l'accuratezza top-1 e
top-5.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica con `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. [Esportazione](/docs/export) elenca gli
argomenti accettati da ogni formato e gli extra che alcuni di essi aggiungono.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box></provenance-box>
