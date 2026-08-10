---
title: BiRefNet
families:
  - birefnet
seo_title: 'BiRefNet: rimozione dello sfondo e matting in LibreYOLO'
description: >-
  Usa BiRefNet in LibreYOLO per rimuovere lo sfondo e per la segmentazione
  dicotomica delle immagini. Installa, fai predizioni, valida ed esporta il
  checkpoint general.
lead: >-
  Una rete a riferimento bilaterale che predice un alpha matte morbido, capace
  di separare il soggetto dallo sfondo. LibreYOLO include inferenza e
  validazione per il task matte di BiRefNet.
keywords:
  - BiRefNet
  - rimuovere lo sfondo da un'immagine
  - background removal python
  - dichotomous image segmentation
  - alpha matte
  - image matting
  - ritaglio con sfondo trasparente
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreBiRefNetl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Ritaglio
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8: l'RGB di origine più il matte come canale alfa.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # Anche una cartella che contiene images/ e una cartella di matte
        # rilevata automaticamente (mattes/, matte/, gt/, masks/, mask/ o
        # alpha/) va bene al posto di un YAML del dataset.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreBiRefNetl-matte.pt format=onnx
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory instrada in base all'estensione del file, quindi un
        # artefatto esportato si carica come qualsiasi checkpoint e
        # restituisce lo stesso oggetto Results.
        model = LibreYOLO("LibreBiRefNetl-matte.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
source_hash: 1af1bd7f4f905081
---

## Installazione

BiRefNet non richiede nessun extra opzionale. Tutto quello che importa è
nell'installazione base.

```bash
pip install libreyolo
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella
cache locale.

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

Un solo checkpoint pubblicato, `l`, il modello BiRefNet-general del livello
Swin-L e la scelta predefinita per la qualità nel progetto originale. Il codice
della famiglia supporta anche un livello lite Swin-T, `t`, ma non è ancora
pubblicata nessuna sua conversione per LibreYOLO.

## Validazione

`val()` riporta due metriche su una cartella accoppiata di immagini e matte,
entrambe in `[0, 1]` e indipendenti dalla risoluzione: MAE, l'errore assoluto
medio rispetto all'alfa del ground truth (più basso è meglio), e S-measure
(Fan et al., ICCV 2017), una similarità strutturale che premia la
conservazione della forma e dei buchi del soggetto, cosa che il MAE per pixel
da solo si perde (più alto è meglio). La validazione passa per il `predict`
del modello stesso, quindi usa esattamente il preprocessing della famiglia.

<code-tabs name="val" />

La validazione è solo inferenza; il fine-tuning è un seguito documentato, non
una funzionalità già inclusa (vedi Predizione per il vincolo esatto di
risoluzione che qualsiasi trainer futuro erediterebbe).

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica con `LibreYOLO()` in base all'estensione del
file, quindi un file `.onnx` si comporta come un checkpoint e restituisce lo
stesso `Results`. TorchScript è il percorso validato; la conversione in ONNX
funziona, ma non ha superato la stessa asticella di parità.
[Esportazione](/docs/export) elenca gli argomenti che ogni formato accetta e
gli extra che qualcuno di essi aggiunge.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenza

<provenance-box></provenance-box>

## Citazione

<citation-block />
