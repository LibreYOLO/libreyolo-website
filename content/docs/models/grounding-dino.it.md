---
title: Grounding DINO
families:
  - grounding_dino
seo_title: 'Grounding DINO in LibreYOLO: rilevamento open-set'
description: >-
  Usa Grounding DINO in LibreYOLO per rilevare qualsiasi oggetto descritto a
  parole. Installa l'extra openvocab e fai predizioni con un vocabolario in
  testo libero.
lead: >-
  Grounding DINO è un rilevatore di oggetti open-set, sviluppato da IDEA
  Research, che valuta un'immagine rispetto a un prompt in testo libero invece
  che rispetto a una lista di classi fissa. LibreYOLO lo avvolge come famiglia
  di sola predizione nel suo livello di rilevatori a vocabolario aperto.
keywords:
  - Grounding DINO
  - open-vocabulary object detection
  - open-set detection
  - zero-shot detection
  - rilevare oggetti da testo
  - grounding dino python
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-t")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Soglia di testo
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-b")
        model.set_classes(["remote control", "school bus"])

        # conf filtra per punteggio del box, text_threshold per il punteggio
        # dei token della frase decodificata. Se non impostati valgono 0.25.
        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)
        print(result.names)
source_hash: 06bd13b8e6a66038
---

## Installazione

Grounding DINO si carica attraverso il livello dei rilevatori a vocabolario
aperto di LibreYOLO, che richiede l'extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Quell'extra porta con sé `transformers` e `timm`, le librerie di Hugging Face
che questo livello richiama.

## Predizione

Grounding DINO non è un checkpoint che LibreYOLO carica con `LibreYOLO()`. Si
carica con la factory gemella `LibreOpenVocab`, che al primo utilizzo scarica
uno snapshot di Hugging Face e lo mette in cache sotto `weights/`.

<code-tabs name="predict" />

`set_classes()` imposta un vocabolario testuale persistente: chiamalo di nuovo
per sostituire la lista, oppure saltalo per tenere le etichette COCO-80
predefinite. Grounding DINO decodifica frasi in forma libera dal proprio output
testuale e le rimappa da sé su quel vocabolario, vince una corrispondenza
esatta normalizzata, si accetta una corrispondenza su token interi, e una frase
ambigua o senza corrispondenza viene scartata invece che indovinata, così
`school bus` non finisce mai mappato su `bus` o `school` da soli. Un vocabolario
abbastanza lungo da superare il limite di token dell'encoder di testo viene
diviso in più prompt, eseguito come passate forward separate e riunito in un
unico insieme di rilevamenti limitato da `max_det`.

`iou` viene accettato per compatibilità di API ma emette un avviso e non fa
nulla, dato che qui non c'è niente che esegua la non-maximum suppression.
`imgsz` e `augment=True` vengono rifiutati del tutto: il ridimensionamento
spetta al processor di `transformers`, e la test-time augmentation è fuori
dallo scopo di questo livello. `predict()` su una singola immagine restituisce
un solo `Results`, non una lista; passa una cartella, una lista di immagini o
`stream=True` per una sorgente video per averne diversi. Non esiste una strada
da CLI per questa famiglia, `libreyolo predict` carica solo checkpoint `.pt`
attraverso `LibreYOLO()`, quindi le famiglie `LibreOpenVocab` si eseguono da
Python. Vedi [predizione](/docs/predict) per i tipi di sorgente e lo streaming.

## Varianti

Due checkpoint, `t` e `b`. `t` è la dimensione predefinita di questo livello
quando non se ne indica nessuna. Entrambi rispecchiano la release ufficiale di
IDEA Research attraverso `GroundingDinoForObjectDetection` di `transformers`,
scaricata una volta sola in uno snapshot Hugging Face ospitato da LibreYOLO che
conserva i file originali. Per questa famiglia non sono ancora pubblicati
numeri di accuratezza o di latenza.

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
