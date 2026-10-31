---
title: OMDet-Turbo
families:
  - omdet_turbo
seo_title: 'OMDet-Turbo in LibreYOLO: rilevamento zero-shot in tempo reale'
description: >-
  Usa OMDet-Turbo in LibreYOLO per il rilevamento a vocabolario aperto in tempo
  reale. Installa l'extra openvocab e fai predizioni con un vocabolario in testo
  libero.
lead: >-
  OMDet-Turbo è un rilevatore di oggetti a vocabolario aperto in tempo reale,
  sviluppato da Om AI Lab, che disaccoppia gli embedding delle classi dal prompt
  testuale del task. LibreYOLO lo avvolge come famiglia di sola predizione nel
  suo livello di rilevatori a vocabolario aperto.
keywords:
  - OMDet-Turbo
  - OmDet
  - open-vocabulary object detection
  - real-time detection
  - zero-shot detection
  - rilevamento oggetti tempo reale
  - omdet turbo python
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.3)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Soglia NMS personalizzata
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["traffic light", "bicycle"])

        # OMDet-Turbo è l'unica famiglia di questo livello che rispetta iou=: il
        # suo post-processing prende la soglia di soppressione come argomento,
        # e vale 0.5 se iou= non viene impostato.
        result = model.predict(SAMPLE_IMAGE, conf=0.3, iou=0.7)
        print(result.names, len(result))
source_hash: c2a375d234341b7e
---

## Installazione

OMDet-Turbo si carica attraverso il livello dei rilevatori a vocabolario aperto
di LibreYOLO, che richiede l'extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Quell'extra porta con sé `transformers` e `timm`, le librerie di Hugging Face
che questo livello richiama; il backbone Swin di OMDet-Turbo si carica
attraverso il wrapper `TimmBackbone` di `transformers`.

## Predizione

OMDet-Turbo non è un checkpoint che LibreYOLO carica con `LibreYOLO()`. Si
carica con la factory gemella `LibreOpenVocab`, che al primo utilizzo scarica
uno snapshot di Hugging Face e lo mette in cache sotto `weights/`.

<code-tabs name="predict" />

`set_classes()` imposta un vocabolario testuale persistente: chiamalo di nuovo
per sostituire del tutto la lista, oppure saltalo per tenere le etichette
COCO-80 predefinite, e un risultato vuoto è un esito valido e non un errore. A
differenza di Grounding DINO, OMDet-Turbo disaccoppia gli embedding delle classi
dal prompt testuale del task, così il post-processing di `transformers`
restituisce etichette che si rimappano direttamente sulla lista di classi
richiesta, senza alcun passaggio di disambiguazione delle frasi.

OMDet-Turbo non ha una soglia sui token di testo: solo `conf` filtra i
rilevamenti, e passare `text_threshold` solleva un errore. È l'unica famiglia di
questo livello che esegue la propria non-maximum suppression dentro
`post_process_grounded_object_detection`, quindi qui `iou` viene rispettato
invece che segnalato con un avviso. `imgsz` e `augment=True` vengono rifiutati
del tutto: il ridimensionamento spetta al processor di `transformers`, e la
test-time augmentation è fuori dallo scopo di questo livello. `predict()` su una
singola immagine restituisce un solo `Results`, non una lista; passa una
cartella, una lista di immagini o `stream=True` per una sorgente video per
averne diversi. Non esiste una strada da CLI per questa famiglia, `libreyolo
predict` carica solo checkpoint `.pt` attraverso `LibreYOLO()`, quindi le
famiglie `LibreOpenVocab` si eseguono da Python. Vedi
[predizione](/docs/predict) per i tipi di sorgente e lo streaming.

## Varianti

Un solo checkpoint, `t`, l'unica dimensione di questo livello. Rispecchia
`omlab/omdet-turbo-swin-tiny-hf` a una revisione upstream fissata, attraverso
`OmDetTurboForObjectDetection` di `transformers`; il file di pesi replicato è
identico byte per byte a quello snapshot upstream. Per questa famiglia non sono
ancora pubblicati numeri di accuratezza o di latenza.

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
