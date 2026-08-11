---
title: SAM
families:
  - sam
seo_title: 'SAM (Segment Anything): predire maschere in LibreYOLO'
description: >-
  Usa SAM in LibreYOLO per la segmentazione guidata da prompt di punto e di box.
  Installa e fai predizioni con i checkpoint base, large e huge sotto licenza
  Apache-2.0.
lead: >-
  SAM (Segment Anything) trasforma un clic di punto o di box in una maschera
  dell'oggetto. LibreYOLO lo carica attraverso una factory LibreSAM dedicata,
  separata dalla factory di detector LibreYOLO(), perché un modello guidato da
  prompt ha bisogno di una forma di chiamata diversa.
keywords:
  - SAM
  - Segment Anything
  - promptable segmentation
  - segmentazione interattiva python
  - segmentare un oggetto con un clic
  - point prompt
  - box prompt
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompt di punto e di box
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # "base" scarica automaticamente facebook/sam-vit-base al primo
        utilizzo.

        # Altre dimensioni: "large", "huge" (anche "b"/"l"/"h").

        model = LibreSAM("base")


        # Un prompt di punto: [x, y] in coordinate pixel, label 1 = primo piano.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # un poligono per maschera

        print(result.boxes.xyxy)    # box aderente derivato dalla maschera


        # Un prompt di box invece di un punto.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])


        # Senza nessun prompt viene segmentata l'intera immagine (un generatore

        # automatico di maschere semplificato, non quello esaustivo di
        riferimento).

        result = model.predict(SAMPLE_IMAGE)
    - label: 'Codifica una volta, poi lancia tutti i prompt che vuoi'
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        model = LibreSAM("base")


        # L'encoder dell'immagine è la parte costosa. set_image() lo esegue una
        volta;

        # ogni chiamata a predict() successiva riusa l'embedding in cache.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: f8904d241ef8a929
---

## Installazione

SAM richiede l'extra `sam`, che porta con sé `transformers` e `timm`.

```bash
pip install "libreyolo[sam]"
```

## Predizione

`LibreSAM(...)` è un punto di ingresso separato da `LibreYOLO(...)`:
restituisce un segmentatore guidato da prompt invece di un detector, perché qui
un forward pass non significa niente senza un prompt spaziale. Per questa
famiglia non esiste il comando CLI `libreyolo predict`; usa l'API Python.

<code-tabs name="predict" />

Un prompt di punto accetta `[x, y]` per un oggetto, `[[x, y], ...]` per
diversi, oppure array numpy; `labels` marca ogni punto con `1` (primo piano) o
`0` (sfondo) e per default sono tutti primo piano. Un prompt di box prende
`[x1, y1, x2, y2]` o una lista di box, una maschera per box. Omettere entrambi
i prompt segmenta l'intera immagine lanciando una griglia densa di prompt e
tenendo le maschere confidenti che non si sovrappongono; questa modalità
«segmenta tutto» è semplificata rispetto al generatore automatico di maschere
di riferimento e può sotto-segmentare le scene affollate, quindi un vero prompt
di punto o di box è la via precisa. `conf` filtra in base alla qualità predetta
della maschera (IoU), non a una confidenza di rilevamento: passa `0.0` per
tenere ogni candidato. `multimask=True` restituisce tutte e tre le maschere di
ambiguità intero-contro-parte di SAM per ogni prompt, invece della sola
migliore. `device=` sposta il modello e, se è attiva una sessione
`set_image()`, il suo embedding in cache. Ogni maschera porta l'id di classe
`0`, di nome `"object"`, dato che una maschera guidata da prompt non ha un
insieme fisso di classi. `train()`, `val()`, `export()` e `track()` sollevano
tutti `NotImplementedError` per questa famiglia: in LibreYOLO SAM serve solo a
fare predizioni, e il tracking video è fuori ambito. Vedi
[predizione](/docs/predict) per i tipi di sorgente.

## Varianti

Tre dimensioni di image encoder ViT: base, large e huge, tutte con input fisso
a 1024 px. Per questa famiglia non è ancora pubblicato nessun benchmark di
accuratezza o di latenza, quindi scegliere una dimensione significa scambiare
direttamente peso dell'encoder con qualità della maschera: base è il più veloce
da codificare, huge il più pesante.

## Licenza

<provenance-box>

LibreYOLO non ospita una propria copia dei pesi di SAM-1. `LibreSAM("base")`,
`"large"` e `"huge"` scaricano direttamente dai repository
`facebook/sam-vit-base`, `facebook/sam-vit-large` e `facebook/sam-vit-huge` di
Meta su Hugging Face, ognuno dei quali è etichettato lì come Apache-2.0
indipendentemente da LibreYOLO.

</provenance-box>

## Citazione

<citation-block />
