---
title: Swin Transformer
families:
  - swin
seo_title: 'Swin Transformer: classifica immagini con LibreSwin di LibreYOLO'
description: >-
  Fai predizioni, valida ed esporta i classificatori Swin Transformer con
  LibreYOLO. Pesi MIT; il fine-tuning non è ancora supportato.
lead: >-
  Swin Transformer V1: un vision transformer gerarchico che calcola l'attenzione
  dentro finestre locali spostate invece che sull'intera immagine. LibreYOLO
  include quattro dimensioni per la classificazione di immagini.
keywords:
  - Swin Transformer
  - Swin tiny
  - vision transformer gerarchico
  - shifted window attention
  - image classification python
  - classificazione immagini python
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwint-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwint-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")

        # data è la radice di una cartella con split train/ e val/ organizzati
        # in cartelle per classe (layout ImageFolder), non un YAML di dataset.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwint-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwint-cls.pt format=onnx
        libreyolo export model=LibreSwint-cls.pt format=tensorrt half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreSwint-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: faa6bbacae62d88e
---

## Installazione

Swin non richiede nessun extra opzionale. Tutto ciò che importa è già
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
confidenze. Ogni dimensione è fissata a un input di 224px, perché lo stadio
finale di attenzione è costruito per quella risoluzione; predizione,
validazione ed esportazione sollevano un errore se passi un `imgsz` diverso.
Vedi [predizione](/docs/predict) per sorgenti, streaming e gestione dei
risultati.

## Varianti

Quattro dimensioni, da tiny a large, costruite sulla stessa torre a finestre
spostate e differenti per larghezza degli embedding e profondità degli stadi.
La large è preaddestrata su ImageNet-22k e sottoposta a fine-tuning su
ImageNet-1k; le altre tre sono addestrate direttamente su ImageNet-1k.
LibreYOLO distribuisce questa famiglia solo per l'inferenza: predizione,
validazione top-1/top-5 in stile ImageNet ed esportazione sono supportate,
mentre la ricetta di addestramento ImageNet del progetto originale non è
implementata.

## Validazione

`val()` viene eseguito su uno split in stile ImageFolder (una cartella con
sottocartelle `train/` e `val/`, una cartella per classe) e restituisce
l'accuratezza top-1 e top-5.

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

## Citazione

<citation-block />
