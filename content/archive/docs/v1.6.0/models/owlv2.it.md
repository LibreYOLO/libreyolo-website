---
title: OWLv2
families:
  - owlv2
seo_title: 'OWLv2 in LibreYOLO: rilevamento di oggetti zero-shot'
description: >-
  Usa OWLv2 in LibreYOLO per rilevare qualsiasi oggetto descritto a parole.
  Installa l'extra openvocab e fai predizioni con un vocabolario in testo
  libero.
lead: >-
  OWLv2 è un rilevatore di oggetti a vocabolario aperto, sviluppato da Google
  Research, che valuta le regioni dell'immagine rispetto agli embedding testuali
  di un encoder in stile CLIP. LibreYOLO lo avvolge come famiglia di sola
  predizione nel suo livello di rilevatori a vocabolario aperto.
keywords:
  - OWLv2
  - OWL-ViT
  - open-vocabulary object detection
  - zero-shot detection
  - rilevare oggetti da testo
  - owlv2 python
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("owlv2-b16")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.1)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Vocabolario predefinito
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        # Saltare set_classes() mantiene il vocabolario COCO-80 predefinito del
        livello.

        model = LibreOpenVocab("owlv2-l14")

        result = model.predict(SAMPLE_IMAGE, conf=0.1)

        print(result.names)
source_hash: 2d0ce68af0daabb7
---

## Installazione

OWLv2 si carica attraverso il livello dei rilevatori a vocabolario aperto di
LibreYOLO, che richiede l'extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Quell'extra porta con sé `transformers` e `timm`, le librerie di Hugging Face
che questo livello richiama.

## Predizione

OWLv2 non è un checkpoint che LibreYOLO carica con `LibreYOLO()`. Si carica con
la factory gemella `LibreOpenVocab`, che al primo utilizzo scarica uno snapshot
di Hugging Face e lo mette in cache sotto `weights/`.

<code-tabs name="predict" />

`set_classes()` imposta un vocabolario testuale persistente: chiamalo di nuovo
per sostituire la lista, oppure saltalo per tenere le etichette COCO-80
predefinite. Ogni etichetta viene avvolta in un template di prompt fisso prima
di arrivare alla torre testuale, in linea con il modo in cui
`Owlv2ForObjectDetection` di `transformers` è stato addestrato.

OWLv2 non ha una soglia sui token di testo: solo `conf` filtra i rilevamenti, e
passare `text_threshold` solleva un errore. `iou` viene accettato per
compatibilità di API ma emette un avviso e non fa nulla, dato che qui non c'è
niente che esegua la non-maximum suppression. `imgsz` e `augment=True` vengono
rifiutati del tutto: il ridimensionamento spetta al processor di
`transformers`, e la test-time augmentation è fuori dallo scopo di questo
livello. `predict()` su una singola immagine restituisce un solo `Results`, non
una lista; passa una cartella, una lista di immagini o `stream=True` per una
sorgente video per averne diversi. Non esiste una strada da CLI per questa
famiglia, `libreyolo predict` carica solo checkpoint `.pt` attraverso
`LibreYOLO()`, quindi le famiglie `LibreOpenVocab` si eseguono da Python. Vedi
[predizione](/docs/predict) per i tipi di sorgente e lo streaming.

## Varianti

Due checkpoint, `b16` (base, patch size 16) e `l14` (large, patch size 14).
`b16` è la dimensione predefinita di questo livello quando non se ne indica
nessuna. Entrambi rispecchiano la release ufficiale di Google Research
attraverso `Owlv2ForObjectDetection` di `transformers`, scaricata una volta
sola in uno snapshot Hugging Face ospitato da LibreYOLO che conserva i file
originali. Per questa famiglia non sono ancora pubblicati numeri di accuratezza
o di latenza.

L'addestramento, la validazione su dataset e l'esportazione sono tutti fuori
dallo scopo di questo livello: `train()`, `val()` ed `export()` sollevano tutti
`NotImplementedError` senza condizioni. Questo è un wrapper di sola predizione
attorno a un checkpoint pubblicato.

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
