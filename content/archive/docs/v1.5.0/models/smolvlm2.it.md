---
title: SmolVLM2
families:
  - smolvlm2
seo_title: 'SmolVLM2 in LibreYOLO: rilevamento a vocabolario aperto'
description: >-
  SmolVLM2 in LibreYOLO: installazione, definizione di un vocabolario aperto e
  predizione o chat con il modello vision-language Apache-2.0 di Hugging Face.
lead: >-
  SmolVLM2 è il piccolo modello vision-language di Hugging Face. LibreYOLO lo
  avvolge come rilevatore di oggetti a vocabolario aperto ed espone direttamente
  la sua chat libera: indica una lista di classi da rilevare, oppure fagli una
  domanda.
keywords:
  - SmolVLM2
  - vision-language model
  - open-vocabulary detection
  - modello multimodale piccolo
  - rilevare oggetti con il testo
  - Hugging Face
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("smolvlm2-500m")
        model.set_classes(["cat", "dog"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("smolvlm2-500m")

        # La via di fuga sotto la comodità del rilevamento: qualsiasi domanda,
        # non solo una query sui bounding box.
        answer = model.chat(SAMPLE_IMAGE, "What is the cat doing?")
        print(answer)
source_hash: b30823b62d6347b5
---

## Installazione

SmolVLM2 appartiene al livello VLM-come-detector di LibreYOLO, una superficie di
prodotto separata dalle famiglie basate su checkpoint e con una factory propria.
Richiede l'extra `vlm`, che porta con sé anche `num2words`, una dipendenza del
processor di SmolVLM2 stesso.

```bash
pip install "libreyolo[vlm]"
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella cache
locale.

<code-tabs name="predict" />

Questa famiglia si carica con la factory `LibreVLM()`, non con `LibreYOLO()`: le
famiglie VLM non dichiarano nessun caricatore di checkpoint, quindi il routing
per suffisso di file descritto nelle altre pagine dei modelli qui non si applica.
`set_classes()` definisce il vocabolario che si chiede a SmolVLM2 di trovare; è
persistente, quindi resta in vigore in tutte le chiamate successive a
`predict()`/`track()` finché non lo imposti di nuovo. SmolVLM2 non richiede
nessun override del parser in LibreYOLO: segue lo stesso output a template di
chat più JSON del default condiviso del livello, quindi il suo prompt di
rilevamento e il formato dei box non sono specifici della famiglia. Ogni
rilevamento porta la stessa confidenza segnaposto, quindi filtrare per `conf` è
tutto o niente invece che un ordinamento; `iou` invece ha effetto, e scarta un
box successivo della stessa classe quando si sovrappone oltre la soglia a uno
già tenuto, dato che un generatore che si ripete può altrimenti emettere box
quasi duplicati per uno stesso oggetto. SmolVLM2 risponde anche a domande libere
tramite `chat()`, la stessa via di fuga documentata sulla factory `LibreVLM`. La
CLI di LibreYOLO non copre questo livello: non esiste una forma
`libreyolo predict model=...` per esso. Vedi [predizione](/docs/predict) per
sorgenti, streaming e gestione dei risultati.

## Varianti

Una sola dimensione nel registro: SmolVLM2-500M-Video-Instruct, caricata come
`LibreVLM("smolvlm2-500m")`. SmolVLM2 è un rilevatore più debole dei modelli di
grounding costruiti apposta in questo livello; il wrapper di LibreYOLO stesso lo
descrive come una dimostrazione che una famiglia nuova non ha bisogno di un
parsing speciale per funzionare qui, non come la sua opzione a vocabolario
aperto più forte.

LibreYOLO non addestra, non valida e non esporta SmolVLM2: `train()`, `val()` e
`export()` sollevano tutti `NotImplementedError` per ogni famiglia di questo
livello (vedi il livello di supporto qui sopra). Fai fine-tuning di SmolVLM2
upstream e carica i pesi risultanti se ti serve un vocabolario personalizzato
integrato; controlla a occhio l'output di `predict()` invece di una passata di
validazione in stile COCO, dato che ogni rilevamento porta la stessa confidenza
segnaposto.

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
