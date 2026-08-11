---
title: Rilevamento a vocabolario aperto
seo_title: Rilevamento a vocabolario aperto in LibreYOLO
description: >-
  Rileva oggetti a partire da un vocabolario testuale in LibreYOLO. Carica
  Grounding DINO, OWLv2, OMDet-Turbo o OV-DEIM con LibreOpenVocab e imposta le
  classi a runtime.
lead: >-
  Il rilevamento a vocabolario aperto sostituisce la lista fissa di classi di un
  checkpoint con le parole che scegli al momento della chiamata. In LibreYOLO
  non è un task a parte: è il task detect servito da una famiglia di modelli
  separata, caricata attraverso la factory LibreOpenVocab invece che LibreYOLO.
keywords:
  - open vocabulary detection
  - zero shot object detection python
  - rilevamento oggetti con prompt testuale
  - grounding dino python
  - owlv2
  - omdet turbo
  - rilevamento a vocabolario aperto
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
        print(result.names)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Cambiare il vocabolario
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("owlv2-b16")

        # set_classes è persistente: resta valido fino alla chiamata successiva.
        # Le etichette devono essere uniche una volta rese minuscole e private
        # degli articoli.
        model.set_classes(["a red backpack", "traffic cone"])
        result = model.predict(SAMPLE_IMAGE)

        model.set_classes(["bicycle wheel"])
        result = model.predict(SAMPLE_IMAGE)
    - label: Soglia sul testo di Grounding DINO
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-b")
        model.set_classes(["remote control", "school bus"])

        # conf filtra in base al punteggio del box, text_threshold in base al
        # punteggio dei token della frase decodificata. Se non li imposti,
        # entrambi valgono 0.25. Solo Grounding DINO accetta text_threshold;
        # gli altri sollevano un'eccezione.
        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)
source_hash: 17197cf4d80f3d6f
---

## Definizione

Il rilevamento a vocabolario aperto restituisce normali `Results` di rilevamento:
box, confidenze e indici di classe, con `result.names` che riporta quegli indici
alle stringhe che hai chiesto. Quello che cambia è da dove arriva la lista delle
classi. Un rilevatore convenzionale viene addestrato su un insieme fisso di
categorie e non può mai emettere una categoria che ne stia fuori. Questi modelli
prendono il vocabolario come testo al momento dell'inferenza, quindi
`set_classes(["forklift", "safety cone"])` basta a rendere quelle le classi.

LibreYOLO non ha una chiave di task `open-vocabulary`. Questi modelli dichiarano
`SUPPORTED_TASKS = ("detect",)` come qualsiasi altro rilevatore. A distinguerli è
il percorso di caricamento: sono snapshot di Hugging Face invece che checkpoint
LibreYOLO in formato state-dict, quindi restano fuori dalla factory `LibreYOLO()`
e si costruiscono con `LibreOpenVocab()`. Quella factory è una sorella di
`LibreSAM()` e `LibreVLM()`, non un sostituto di `LibreYOLO()`.

I punteggi sono veri punteggi di rilevamento, non una didascalia generata e poi
interpretata a posteriori. Ogni famiglia confronta le regioni dell'immagine con
l'embedding testuale di ogni prompt.

## Modelli

Quattro famiglie compongono questa fascia, tutte solo in predizione. Puoi
caricarle tutte per alias attraverso `LibreOpenVocab`.

[Grounding DINO](/docs/models/grounding-dino), di IDEA Research, nelle taglie `t`
e `b`. È il default della fascia e l'unica famiglia che accetta
`text_threshold`, una seconda soglia sul punteggio dei token della frase
decodificata.

[OWLv2](/docs/models/owlv2), di Google Research, nelle taglie `b16` e `l14`.
Confronta le regioni dell'immagine con embedding testuali prodotti da un encoder
in stile CLIP.

[OMDet-Turbo](/docs/models/omdet-turbo), di Om AI Lab, in un'unica taglia `t`.
Separa gli embedding di classe da un prompt testuale di task ed è l'unica
famiglia qui che sopprime i box sovrapposti nel proprio post-processing, quindi
`iou=` viene rispettato.

[OV-DEIM](/docs/models/ov-deim), nelle taglie `s`, `m` e `l`, è un rilevatore in
stile DETR che associa le query del decoder a embedding testuali prodotti da una
text tower MobileCLIP inclusa. Usa un matching uno-a-uno con selezione top-K,
quindi non gira NMS da nessuna parte.

I pesi di OV-DEIM sono il caso vincolato di questa fascia. I pesi del rilevatore
sono CC BY-NC 4.0, non commerciali. La text tower inclusa è coperta dalla Machine
Learning Research Model license di Apple, solo per uso di ricerca. Il checkpoint
`l` aggiunge un fine-tune del backbone DINOv3-S sotto la DINOv3 License di Meta.
Tutti e tre i testi di licenza sono distribuiti dentro il repository dei pesi, e
la libreria registra lo stesso riepilogo quando risolve i pesi, prima che il
modello venga costruito. Leggi [OV-DEIM](/docs/models/ov-deim) prima di metterlo
in produzione.

La fascia richiede un extra:

```bash
pip install "libreyolo[openvocab]"
```

Copre `transformers` e `timm` per le tre famiglie incapsulate, più i pacchetti
`huggingface_hub`, `safetensors`, `regex` e `ftfy` che servono a OV-DEIM in
quanto port nativo.

Anche una seconda fascia accetta un vocabolario testuale: `LibreVLM()` carica
modelli generativi vision-language, come [Qwen3-VL](/docs/models/qwen3-vl) e
[Florence-2](/docs/models/florence-2), e trasforma il loro output negli stessi
`Results`. Condivide la stessa superficie `set_classes()`. La differenza è che
cosa produce i box: le famiglie di questa pagina sono rilevatori discriminativi
che emettono punteggi direttamente, mentre la fascia VLM li genera.

## Predizione

<code-tabs name="predict" />

`set_classes()` prende una lista non vuota di stringhe di etichetta e resta
valido fino alla chiamata successiva. Le etichette devono essere uniche una volta
rese minuscole e private degli articoli iniziali, quindi `"a bus"` e `"bus"` non
possono coesistere in uno stesso vocabolario. Le frasi di più parole sono
etichette come tutte le altre, e ogni famiglia trasforma la lista nel proprio
input testuale prima di tokenizzarla, quindi `"traffic cone"` è una query diversa
da `"cone"`.

Tre argomenti di predizione si comportano qui in modo diverso rispetto a un
rilevatore nativo. `imgsz=` viene rifiutato, perché per queste famiglie è il
processor a gestire il ridimensionamento. `augment=True` viene rifiutato, dato
che la data augmentation in fase di test è fuori dallo scopo della fascia. `iou=`
si applica solo alla famiglia il cui processor esegue una propria soppressione;
dove non viene soppresso nulla, passarlo produce un avviso e viene ignorato.

Se non lo imposti, `conf` prende il default della famiglia caricata invece del
consueto 0.25 di `predict()`, e quel default non è lo stesso in tutta la fascia.
Impostalo esplicitamente quando confronti due famiglie sulla stessa immagine.

`track()` solleva un'eccezione su tutta la fascia. Esegui invece `predict()` su
ogni frame. Vedi [predizione](/docs/predict) per sorgenti, streaming e gestione
dei risultati.

## Addestramento

Nessuna famiglia di questa fascia si addestra dentro LibreYOLO. `train()` solleva
un'eccezione: fai fine-tuning a monte e carica i pesi risultanti. Il vocabolario
passato a `set_classes()` è l'unica impostazione che cambia cosa rileva un
modello caricato.

## Validazione

Per questa fascia non esiste un validatore, e `val()` solleva un'eccezione. La
validazione a vocabolario aperto ne richiede uno dedicato, perché il validatore
di rilevamento standard passa i tensori delle immagini direttamente al modello,
mentre queste famiglie richiedono input condizionati sul testo costruiti insieme
a loro.

## Esportazione

L'esportazione è fuori dallo scopo della fascia e `export()` solleva
un'eccezione. Questi modelli girano attraverso `predict()` in PyTorch.
