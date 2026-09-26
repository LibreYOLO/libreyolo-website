---
title: Qwen3-VL
families:
  - qwen3vl
seo_title: 'Qwen3-VL in LibreYOLO: rilevamento a vocabolario aperto'
description: >-
  Qwen3-VL in LibreYOLO: installazione, definizione di un vocabolario aperto e
  predizione o chat con il modello vision-language di Alibaba con licenza
  Apache-2.0.
lead: >-
  Qwen3-VL è il modello vision-language di Alibaba con grounding 2D nativo.
  LibreYOLO lo avvolge come rilevatore di oggetti a vocabolario aperto ed espone
  direttamente la sua chat libera: passa una lista di classi da rilevare, oppure
  fagli una domanda.
keywords:
  - Qwen3-VL
  - vision-language model
  - open-vocabulary detection
  - rilevare oggetti senza addestramento
  - qwen3 vl python
  - grounding
  - Alibaba
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("qwen3-vl-4b")
        model.set_classes(["forklift", "pallet", "safety vest"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat
      language: python
      code: >
        from libreyolo import LibreVLM, SAMPLE_IMAGE


        model = LibreVLM("qwen3-vl-4b")


        # La via di fuga sotto la comodità del rilevamento: qualsiasi domanda,

        # non solo una query su bounding box.

        answer = model.chat(SAMPLE_IMAGE, "How many people are wearing a safety
        vest?")

        print(answer)
source_hash: ee225b6221d624d9
---

## Installazione

Qwen3-VL appartiene al livello VLM-come-detector di LibreYOLO, una superficie di
prodotto separata dalle famiglie basate su checkpoint e con una factory propria.
Richiede l'extra `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella cache
locale. `LibreVLM()` chiamata senza argomenti usa per default Qwen3-VL-4B.

<code-tabs name="predict" />

Questa famiglia si carica con la factory `LibreVLM()`, non con `LibreYOLO()`: le
famiglie VLM non dichiarano nessun caricatore di checkpoint, quindi il routing
per suffisso di file descritto nelle altre pagine dei modelli qui non si applica.
`set_classes()` definisce il vocabolario che si chiede a Qwen3-VL di trovare; è
persistente, quindi resta in vigore in tutte le chiamate successive a
`predict()`/`track()` finché non lo imposti di nuovo. Ogni rilevamento porta la
stessa confidenza segnaposto, quindi filtrare per `conf` è tutto o niente invece
che un ordinamento; `iou` invece ha effetto per questa famiglia, perché scarta un
box successivo della stessa classe quando si sovrappone oltre la soglia a uno già
tenuto, dato che un generatore ripetitivo può altrimenti emettere box quasi
duplicati per uno stesso oggetto. A differenza di Florence-2 e Kosmos-2, Qwen3-VL
risponde anche a domande libere tramite `chat()`, la stessa via di fuga
documentata sulla factory `LibreVLM`. La CLI di LibreYOLO non copre questo
livello: non esiste una forma `libreyolo predict model=...` per esso. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Varianti

Tre dimensioni: Qwen3-VL-2B-Instruct, Qwen3-VL-4B-Instruct e Qwen3-VL-8B-Instruct,
caricate come `LibreVLM("qwen3-vl-2b")`, `LibreVLM("qwen3-vl-4b")` e
`LibreVLM("qwen3-vl-8b")`. Tutte e tre dichiarano un input nominale di 1024 px,
ma è lo smart-resize del processore di Qwen a decidere il canvas effettivo che
arriva alla rete, quindi quella cifra non è una risoluzione operativa fissa come
lo è per le altre famiglie di questo sito. LibreYOLO non ha pubblicato nessun
benchmark che confronti l'accuratezza tra le tre dimensioni.

LibreYOLO non addestra, non valida e non esporta Qwen3-VL: `train()`, `val()` e
`export()` sollevano tutti `NotImplementedError` per ogni famiglia di questo
livello (vedi il livello di supporto qui sopra). Fai fine-tuning di Qwen3-VL
upstream e carica i pesi risultanti se ti serve un vocabolario personalizzato
integrato; controlla a occhio l'output di `predict()` invece di una passata di
validazione in stile COCO, dato che ogni rilevamento porta la stessa confidenza
segnaposto.

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
