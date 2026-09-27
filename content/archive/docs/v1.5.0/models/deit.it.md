---
title: DeiT
families:
  - deit
seo_title: 'Classificatore di immagini DeiT: predizione, validazione, esportazione'
description: >-
  Esegui i classificatori di immagini DeiT in LibreYOLO: una famiglia da museo
  congelata e solo per l'inferenza, nelle dimensioni tiny, small e base, con
  licenza Apache-2.0.
lead: >-
  DeiT (Data-efficient image Transformer) è un classificatore Vision Transformer
  puro, addestrato solo su ImageNet-1k e senza dati di preaddestramento
  aggiuntivi. LibreYOLO include le dimensioni tiny, small e base con patch 16
  come pezzo da museo congelato e solo per l'inferenza.
keywords:
  - DeiT
  - Vision Transformer
  - ViT
  - image classification python
  - classificare immagini python
  - ImageNet
  - transformer visione artificiale
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeiTb-cls.pt")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeiTb-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeiTb-cls.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDeiTb-cls.pt format=onnx
        libreyolo export model=LibreDeiTb-cls.pt format=tensorrt half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreDeiTb-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 9c67c8554b2af5c6
---

## Installazione

DeiT non richiede nessun extra oltre al pacchetto base.

```bash
pip install libreyolo
```

## Predizione

Questa famiglia è solo per l'inferenza: `train()` solleva `NotImplementedError`,
quindi questa pagina non ha una sezione sull'addestramento. Predizione,
validazione ed esportazione sono invece tutte supportate. I pesi vengono
scaricati da Hugging Face al primo utilizzo e restano nella cache locale. Il
suffisso `-cls` nel nome del file è obbligatorio e seleziona il task di
classificazione.

<code-tabs name="predict" />

L'oggetto `Results` restituito porta un tensore `probs` invece di `boxes`;
`top1` e `top5` indicizzano le 1.000 classi di ImageNet-1k e `top1conf` è il
punteggio softmax della predizione principale. Ogni dimensione ha una
risoluzione di input fissa, che deriva dal suo positional embedding: il
preprocessing ridimensiona e ritaglia al centro fino a quella risoluzione, e
passare un `imgsz` diverso solleva un errore invece di ricampionare in silenzio.
Vedi [predizione](/docs/predict) per sorgenti, streaming e gestione dei
risultati.

## Validazione

`val()` restituisce un dizionario con l'accuratezza top-1 e top-5, misurata su
un dataset organizzato nella struttura di cartelle convenzionale
`train/<class>/` e `val/<class>/`.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica con `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. È supportata anche l'esecuzione del grafo in un
runtime nudo, senza LibreYOLO installato, ma in quel caso il preprocessing e il
postprocessing li scrivi tu.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
