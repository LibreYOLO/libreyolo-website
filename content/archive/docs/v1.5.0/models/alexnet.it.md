---
title: AlexNet
families:
  - alexnet
seo_title: 'AlexNet: esegui il classico classificatore ImageNet in LibreYOLO'
description: >-
  Fai predizioni, valida ed esporta AlexNet con LibreYOLO. Pesi torchvision con
  licenza BSD-3-Clause; il fine-tuning non è ancora supportato.
lead: >-
  AlexNet è la rete convoluzionale che ha vinto ILSVRC 2012 e ha contribuito ad
  aprire l'era del deep learning nella computer vision. LibreYOLO distribuisce
  la revisione successiva dell'architettura, a torre singola, per la
  classificazione di immagini.
keywords:
  - AlexNet
  - ImageNet
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

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreAlexNetb-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")

        # data è la directory radice con gli split train/ e val/ in cartelle
        # per classe (layout ImageFolder), non uno YAML del dataset.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreAlexNetb-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreAlexNetb-cls.pt format=onnx
        libreyolo export model=LibreAlexNetb-cls.pt format=tensorrt half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreAlexNetb-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 68c09f080c74bb87
---

## Installazione

AlexNet non richiede nessun extra opzionale. Tutto ciò che importa è già
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
confidenze. Vedi [predizione](/docs/predict) per sorgenti, streaming e
gestione dei risultati.

## Varianti

Una sola dimensione. Il grafo distribuito è la revisione successiva a torre
singola rilasciata da torchvision, con 64 filtri nel primo layer e senza local
response normalization, non l'architettura originale del 2012 a due GPU.
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
