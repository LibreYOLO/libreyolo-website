---
title: FeyNobg
families:
  - feynobg
seo_title: 'FeyNobg: rimozione dello sfondo in LibreYOLO'
description: >-
  Usa FeyNobg in LibreYOLO per rimuovere lo sfondo e per l'alpha matting, una
  variante di BiRefNet resa più profonda da Feyn Inc. Installa, fai predizioni e
  valida.
lead: >-
  Un modello per la rimozione dello sfondo di Feyn Inc. che rende più profonda
  l'architettura di BiRefNet e la riaddestra. LibreYOLO include inferenza e
  validazione per il task matte di FeyNobg.
keywords:
  - FeyNobg
  - rimuovere lo sfondo da un'immagine
  - background removal python
  - dichotomous image segmentation
  - alpha matte
  - image matting
  - ritaglio con sfondo trasparente
  - nobg
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFeyNobgl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Ritaglio
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8: l'RGB di origine più il matte come canale alfa.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFeyNobgl-matte.pt")

        # Anche una cartella che contiene images/ e una cartella di matte
        # rilevata automaticamente (mattes/, matte/, gt/, masks/, mask/ o
        # alpha/) va bene al posto di un YAML del dataset.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
source_hash: 45de3b578d7ebbf2
---

## Installazione

FeyNobg non richiede nessun extra opzionale. Tutto quello che importa è
nell'installazione base.

```bash
pip install libreyolo
```

## Predizione

Il checkpoint viene scaricato dall'organizzazione LibreYOLO su Hugging Face al
primo utilizzo e resta nella cache locale, come per qualsiasi altra famiglia,
anche se non è ancora elencato nella tabella dei Checkpoint di questa pagina.

<code-tabs name="predict" />

Un risultato di tipo matte non porta box; `result.matte` è un array denso
`(H, W)` float32 in `[0, 1]`, dove 1 è primo piano pieno e 0 sfondo pieno. A
differenza di una maschera binaria, il matte morbido conserva il dettaglio dei
bordi con antialiasing, come capelli e pelo. `result.cutout()` compone
l'immagine di origine con quel canale alfa in un array RGBA, e
`result.save(path)` (oppure `save=True` nella chiamata di predizione) lo scrive
direttamente in un PNG con sfondo trasparente. Il modello lavora su un canvas
nativo fisso di 1024x1024; un'altra risoluzione non è supportata, perché le
tabelle di posizione relativa del backbone Swin sono legate a quella
dimensione, e una discrepanza le interpola male invece di sollevare un errore.
Vedi [predizione](/docs/predict) per sorgenti, streaming e gestione dei
risultati.

## Varianti

Una sola dimensione pubblicata, `l`, un backbone di livello Swin-L. FeyNobg
prende l'architettura di BiRefNet e porta il suo terzo stage Swin da 18 a 24
blocchi prima di riaddestrarla, quindi il port per LibreYOLO riusa il forward
path, il preprocessing e il contratto di output a logit singolo di BiRefNet;
predizione, validazione e gestione dei checkpoint si comportano come nella
famiglia `birefnet`.

## Validazione

`val()` riporta due metriche su una cartella accoppiata di immagini e matte,
entrambe in `[0, 1]` e indipendenti dalla risoluzione: MAE, l'errore assoluto
medio rispetto all'alfa del ground truth (più basso è meglio), e S-measure
(Fan et al., ICCV 2017), una similarità strutturale che premia la
conservazione della forma e dei buchi del soggetto, cosa che il MAE per pixel
da solo si perde (più alto è meglio). La validazione passa per il `predict` del
modello stesso, quindi usa esattamente il preprocessing della famiglia.

<code-tabs name="val" />

La validazione è solo inferenza. La libreria `nobg` a monte include codice di
addestramento Apache-2.0; oggi fare fine-tuning significa addestrare lì e
convertire il risultato con lo script di conversione di LibreYOLO, non chiamare
`train()` su questa famiglia, che solleva un errore invece di eseguire un
trainer parziale.

## Licenza

<provenance-box></provenance-box>

## Citazione

<citation-block />
