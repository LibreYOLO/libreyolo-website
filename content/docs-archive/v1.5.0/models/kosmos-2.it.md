---
title: Kosmos-2
families:
  - kosmos2
seo_title: 'Kosmos-2 in LibreYOLO: rilevamento di oggetti con grounding'
description: >-
  Kosmos-2 in LibreYOLO: installazione, definizione di un vocabolario aperto e
  predizione di box con grounding con il modello di Microsoft con licenza MIT.
lead: >-
  Kosmos-2 è il modello di grounding di Microsoft: genera una didascalia
  dell'immagine e poi individua con un box ogni frase nominale di quella
  didascalia. LibreYOLO lo avvolge come rilevatore di oggetti a vocabolario
  aperto: la lista di classi si indica al momento della predizione.
keywords:
  - Kosmos-2
  - vision-language model
  - grounding
  - open-vocabulary detection
  - kosmos 2 python
  - rilevare oggetti senza addestramento
  - Microsoft
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("kosmos-2")
        model.set_classes(["boat", "person"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Video
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("kosmos-2")
        model.set_classes(["boat", "person"])

        # Qualsiasi sorgente accettata dalla libreria: file, cartella, URL,
        # indice della webcam, stream RTSP o una lista .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
source_hash: 60e0796f34be6d59
---

## Installazione

Kosmos-2 appartiene al livello VLM-come-detector di LibreYOLO, una superficie di
prodotto separata dalle famiglie basate su checkpoint e con una factory propria.
Richiede l'extra `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella cache
locale. LibreYOLO carica direttamente il repository ufficiale di Microsoft
`microsoft/kosmos-2-patch14-224`; a differenza di Florence-2, qui non serve
nessun re-upload della community.

<code-tabs name="predict" />

Questa famiglia si carica con la factory `LibreVLM()`, non con `LibreYOLO()`: le
famiglie VLM non dichiarano nessun caricatore di checkpoint, quindi il routing
per suffisso di file descritto nelle altre pagine dei modelli qui non si applica.
`set_classes()` definisce il vocabolario che si chiede a Kosmos-2 di trovare; è
persistente, quindi resta in vigore in tutte le chiamate successive a
`predict()`/`track()` finché non lo imposti di nuovo. Kosmos-2 fa il grounding di
frasi nominali invece di confrontare esattamente un'etichetta, quindi il wrapper
di LibreYOLO accetta una corrispondenza parziale: una classe chiamata `"boat"`
corrisponde anche a una frase generata come "the boats". Ogni rilevamento porta
la stessa confidenza segnaposto, quindi filtrare per `conf` è tutto o niente
invece che un ordinamento, e `iou` qui non ha effetto, dato che il wrapper
costruisce la lista dei rilevamenti direttamente dalle entità con grounding,
senza nessun passaggio di deduplicazione. `chat()` solleva `NotImplementedError`,
perché Kosmos-2 si guida con un prompt `<grounding>` e non con un template di
chat. La CLI di LibreYOLO non copre questo livello: non esiste una forma
`libreyolo predict model=...` per esso. Vedi [predizione](/docs/predict) per
sorgenti, streaming e gestione dei risultati.

## Varianti

Una sola dimensione: `kosmos-2-patch14-224`, a 224 px, caricata come
`LibreVLM("kosmos-2")`. È un modello del 2023, e il wrapper di LibreYOLO segnala
che il suo grounding è più grossolano di quello dei rilevatori più recenti di
questo livello.

LibreYOLO non addestra, non valida e non esporta Kosmos-2: `train()`, `val()` e
`export()` sollevano tutti `NotImplementedError` per ogni famiglia di questo
livello (vedi il livello di supporto qui sopra). Fai fine-tuning di Kosmos-2
upstream e carica i pesi risultanti se ti serve un vocabolario personalizzato
integrato; controlla a occhio l'output di `predict()` invece di una passata di
validazione in stile COCO, dato che ogni rilevamento porta la stessa confidenza
segnaposto.

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
