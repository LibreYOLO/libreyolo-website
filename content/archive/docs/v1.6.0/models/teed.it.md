---
title: TEED
families:
  - teed
seo_title: 'TEED: rilevamento dei bordi, porta il tuo checkpoint'
description: >-
  Usa TEED in LibreYOLO per predire una mappa densa di probabilità di bordo.
  Converti un checkpoint di cui hai la licenza, poi fai predizioni, valida ed
  esporta.
lead: >-
  TEED (Tiny and Efficient Edge Detector) è una piccola rete convoluzionale che
  predice una mappa densa di probabilità di bordo a partire da una sola immagine
  RGB. LibreYOLO ne include l'architettura solo per il rilevamento dei bordi; la
  libreria non distribuisce nessun checkpoint.
keywords:
  - TEED
  - Tiny and Efficient Edge Detector
  - edge detection python
  - BIPED
  - rilevare bordi immagini deep learning
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        result = model(SAMPLE_IMAGE, save=True)

        edges = result.edges
        print(edges.array.shape)        # (H, W) float32 in [0, 1]
        print(edges.binary(0.5).sum())  # n. di pixel di bordo dopo la soglia
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=weights/LibreTEEDt-edge.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("weights/LibreTEEDt-edge.pt")

        metrics = model.val(data="my-dataset.yaml", imgsz=352)


        print(metrics["metrics/ODS"])   # F-measure a scala ottimale sul dataset

        print(metrics["metrics/OIS"])   # F-measure a scala ottimale
        sull'immagine
    - label: CLI
      language: bash
      code: >
        libreyolo val model=weights/LibreTEEDt-edge.pt data=my-dataset.yaml
        imgsz=352
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        model.export(format="onnx", imgsz=352)
        model.export(format="tensorrt", imgsz=352, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=weights/LibreTEEDt-edge.pt format=onnx imgsz=352

        libreyolo export model=weights/LibreTEEDt-edge.pt format=tensorrt
        imgsz=352 half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreTEEDt-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: c7203b254e460258
---

## Installazione

TEED non richiede nessun extra opzionale. Tutto ciò che importa è già
nell'installazione di base.

```bash
pip install libreyolo
```

## Predizione

LibreYOLO non distribuisce nessun checkpoint di TEED. I pesi rilasciati
ufficialmente sono addestrati su BIPED, i cui termini pubblicati del dataset
limitano l'uso a scopi non commerciali, quindi LibreYOLO non li rende
disponibili sui propri mirror. Converti un checkpoint per cui hai la licenza
d'uso con `weights/convert_teed_weights.py`, che controlla le chiavi dei
tensori rispetto all'architettura del runtime prima di scrivere un file che
LibreYOLO può caricare direttamente:

```bash
python weights/convert_teed_weights.py upstream.pth weights/LibreTEEDt-edge.pt --verify
```

<code-tabs name="predict" />

`result.edges` contiene il risultato: un array float32 `(H, W)` in `[0, 1]`,
con `.binary(threshold)` che restituisce una maschera booleana dei bordi. Non
ci sono box, quindi `conf`, `iou` e `max_det` non hanno alcun effetto. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Varianti

TEED è disponibile in un'unica taglia in LibreYOLO. Il sistema di benchmark di
LibreYOLO non ha misurato questa famiglia, quindi non ci sono numeri pubblicati
con cui confrontarla.

## Validazione

`val()` riporta le F-measure ODS e OIS in stile BSDS rispetto a un dataset di
bordi accoppiato: immagini affiancate a mappe di bordi con lo stesso nome base,
con una maschera di validità opzionale in modo che i pixel di padding non
contino mai. `imgsz` deve essere divisibile per lo stride di downsample della
rete, e LibreYOLO solleva un errore chiaro se non lo è.

<code-tabs name="val" />

## Esportazione

<export-matrix />

L'esportazione dei bordi usa un contratto di runtime a risoluzione fissa e
batch 1: `dynamic` e un `batch` diverso da 1 vengono rifiutati, e il grafo
esportato restituisce un'unica mappa di probabilità fusa. Un artefatto
esportato si ricarica con `LibreYOLO()` in base al suffisso del file, quindi un
file `.onnx` si comporta come un checkpoint e restituisce lo stesso `Results`.

<code-tabs name="export" />

## Licenze

<provenance-box>

LibreYOLO non pubblica nessun checkpoint di TEED. Non c'è niente sui mirror
dell'organizzazione LibreYOLO; converti invece un checkpoint per cui possiedi
una licenza con `weights/convert_teed_weights.py`.

</provenance-box>

## Citazione

<citation-block />
