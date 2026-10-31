---
title: Florence-2
families:
  - florence2
seo_title: 'Florence-2 in LibreYOLO: rilevamento a vocabolario aperto'
description: >-
  Florence-2 in LibreYOLO: installazione, definizione di un vocabolario aperto e
  predizione di box con il modello di visione di Microsoft con licenza MIT.
lead: >-
  Florence-2 è il modello fondazionale di visione di Microsoft, che si guida con
  un token di task invece di passare per una testa di rilevamento fissa.
  LibreYOLO lo avvolge come rilevatore di oggetti a vocabolario aperto: la lista
  di classi si indica al momento della predizione.
keywords:
  - Florence-2
  - vision-language model
  - open-vocabulary detection
  - rilevare oggetti senza addestramento
  - florence 2 python
  - grounding
  - Microsoft
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Video
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])

        # Qualsiasi sorgente accettata dalla libreria: file, cartella, URL,
        # indice della webcam, stream RTSP o una lista .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
source_hash: ad26d9056465d662
---

## Installazione

Florence-2 appartiene al livello VLM-come-detector di LibreYOLO, una superficie
di prodotto separata dalle famiglie basate su checkpoint e con una factory
propria. Richiede l'extra `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella cache
locale. LibreYOLO scarica il re-upload del checkpoint pubblicato da
florence-community invece del repository originale `microsoft/Florence-2-*`;
vedi Licenze per il motivo.

<code-tabs name="predict" />

Questa famiglia si carica con la factory `LibreVLM()`, non con `LibreYOLO()`: le
famiglie VLM non dichiarano nessun caricatore di checkpoint, quindi il routing
per suffisso di file descritto nelle altre pagine dei modelli qui non si applica.
`set_classes()` definisce il vocabolario che si chiede a Florence-2 di trovare
nell'immagine; è persistente, quindi resta in vigore in tutte le chiamate
successive a `predict()`/`track()` finché non lo imposti di nuovo. L'oggetto
`Results` restituito porta `boxes` nella stessa forma di qualsiasi altra
famiglia, ma ogni rilevamento porta la stessa confidenza segnaposto, quindi
filtrare per `conf` è tutto o niente invece che un ordinamento, e `iou` non ha
effetto: il wrapper di Florence-2 costruisce la lista dei rilevamenti
direttamente a partire dall'output del token di task già parsato, senza nessun
passaggio di deduplicazione. Qui `chat()` solleva `NotImplementedError`, perché
Florence-2 si guida con il token di task `<OPEN_VOCABULARY_DETECTION>` e non con
un template di chat. La CLI di LibreYOLO non copre questo livello: non esiste una
forma `libreyolo predict model=...` per esso. Vedi [predizione](/docs/predict)
per sorgenti, streaming e gestione dei risultati.

## Varianti

Due dimensioni: Florence-2-base e Florence-2-large, entrambe a 768 px, caricate
come `LibreVLM("florence-2-base")` o `LibreVLM("florence-2-large")`. LibreYOLO
non ha pubblicato nessun benchmark che confronti l'accuratezza tra le due.

LibreYOLO non addestra, non valida e non esporta Florence-2: `train()`, `val()` e
`export()` sollevano tutti `NotImplementedError` per ogni famiglia di questo
livello (vedi il livello di supporto qui sopra). Fai fine-tuning di Florence-2
upstream e carica i pesi risultanti se ti serve un vocabolario personalizzato
integrato; controlla a occhio l'output di `predict()` invece di una passata di
validazione in stile COCO, dato che ogni rilevamento porta la stessa confidenza
segnaposto.

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
