---
title: OV-DEIM
families:
  - ov_deim
seo_title: 'OV-DEIM in LibreYOLO: rilevamento a vocabolario aperto'
description: >-
  Usa OV-DEIM in LibreYOLO per il rilevamento a vocabolario aperto in tempo
  reale, in stile DETR. Installa l'extra openvocab e fai predizioni con un
  vocabolario in testo libero.
lead: >-
  OV-DEIM è un rilevatore di oggetti a vocabolario aperto in stile DETR, che
  abbina le query del decoder agli embedding testuali di un text tower
  MobileCLIP incluso. LibreYOLO lo porta nativamente come famiglia di sola
  predizione nel suo livello di rilevatori a vocabolario aperto.
keywords:
  - OV-DEIM
  - DEIMv2
  - open-vocabulary object detection
  - real-time detection
  - zero-shot detection
  - rilevamento vocabolario aperto python
  - ov-deim python
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("ov-deim-s")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Sostituire il vocabolario
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("ov-deim-l")
        model.set_classes(["traffic light", "bicycle"])
        first = model.predict(SAMPLE_IMAGE, conf=0.3)

        # Una seconda chiamata a set_classes() sostituisce del tutto il
        # vocabolario e ne ricalcola gli embedding con il text tower; un
        # risultato vuoto è un esito valido e non un errore.
        model.set_classes(["giraffe"])
        second = model.predict(SAMPLE_IMAGE, conf=0.5)
        print(second.names, len(second))
source_hash: 0c295f555a9eb303
---

## Installazione

OV-DEIM si carica attraverso il livello dei rilevatori a vocabolario aperto di
LibreYOLO, che richiede l'extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

A differenza del resto di questo livello, OV-DEIM è un port nativo di LibreYOLO
e non un wrapper di `transformers`, non esiste una classe di modello
`transformers` per questa famiglia, ma lo stesso extra copre i pacchetti
`huggingface_hub`, `safetensors`, `regex` e `ftfy` che le servono al momento
della predizione.

## Predizione

OV-DEIM non è un checkpoint che LibreYOLO carica con `LibreYOLO()`. Si carica
con la factory gemella `LibreOpenVocab`, che al primo utilizzo scarica uno
snapshot di Hugging Face e lo mette in cache sotto `weights/`.

<code-tabs name="predict" />

`set_classes()` imposta un vocabolario testuale persistente: chiamalo di nuovo
per sostituire del tutto la lista, oppure saltalo per tenere le etichette
COCO-80 predefinite, e un risultato vuoto è un esito valido e non un errore.
Ogni query del decoder viene valutata per similarità coseno rispetto agli
embedding testuali di un text tower MobileCLIP-B(LT) incluso, calcolati al
momento per qualunque vocabolario sia impostato e tenuti in cache finché non
cambia, così qualsiasi prompt funziona senza nessun file di embedding
precalcolato.

OV-DEIM non ha una soglia sui token di testo: solo `conf` filtra i rilevamenti,
e passare `text_threshold` solleva un errore. Il matching è una selezione top-K
uno a uno, quindi qui non gira nessuna non-maximum suppression, e `iou` viene
accettato per compatibilità di API ma emette un avviso e non fa nulla. `imgsz` e
`augment=True` vengono rifiutati del tutto: il modello ha un suo input
letterboxed fisso, e la test-time augmentation è fuori dallo scopo di questo
livello. `predict()` su una singola immagine restituisce un solo `Results`, non
una lista; passa una cartella, una lista di immagini o `stream=True` per una
sorgente video per averne diversi. Non esiste una strada da CLI per questa
famiglia, `libreyolo predict` carica solo checkpoint `.pt` attraverso
`LibreYOLO()`, quindi le famiglie `LibreOpenVocab` si eseguono da Python. Vedi
[predizione](/docs/predict) per i tipi di sorgente e lo streaming.

Ogni chiamata a `predict()` esegue anche il text tower MobileCLIP-B(LT) incluso
per calcolare gli embedding del vocabolario corrente; vedi Licenze per cosa
aggiunge ai termini d'uso.

## Varianti

Tre checkpoint, `s`, `m` e `l`. `s` è la dimensione predefinita di questo
livello quando non se ne indica nessuna. A differenza del resto di questo
livello, OV-DEIM è un port nativo e non un wrapper di `transformers`: LibreYOLO
incorpora i moduli del rilevatore sotto la stessa licenza Apache-2.0 del codice
upstream e riusa l'adattatore di backbone DINOv3 già costruito per la famiglia
DEIMv2. Il backbone del checkpoint `l` è un fine-tune di DINOv3-S, con licenza a
parte sotto la DINOv3 License di Meta. Per questa famiglia non sono ancora
pubblicati numeri di accuratezza o di latenza.

L'addestramento, la validazione su dataset e l'esportazione sono tutti fuori
dallo scopo di questo livello: `train()`, `val()` ed `export()` sollevano tutti
`NotImplementedError` senza condizioni. Questo è un wrapper di sola predizione
attorno a un checkpoint pubblicato.

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box>

OV-DEIM sovrappone tre licenze upstream a ogni chiamata di predizione: i pesi
del rilevatore sotto la CC BY-NC 4.0 di OV-DEIM stesso, il text tower eseguito
al momento sotto la Machine Learning Research Model license di Apple (solo per
uso di ricerca) e, per il checkpoint `l`, un fine-tune del backbone DINOv3-S
sotto la DINOv3 License di Meta. I testi di tutte e tre le licenze sono
distribuiti dentro il repository dei pesi di LibreYOLO.

</provenance-box>

## Citazione

<citation-block />
