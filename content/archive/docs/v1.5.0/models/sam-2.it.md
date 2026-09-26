---
title: SAM 2
families:
  - sam2
seo_title: 'SAM 2: segmentazione di immagini guidata da prompt in LibreYOLO'
description: >-
  Usa SAM 2 in LibreYOLO per la segmentazione guidata da prompt di punto e di
  box. Installa e fai predizioni con i checkpoint tiny, small, base-plus e
  large, Apache-2.0.
lead: >-
  SAM 2 estende SAM con un'architettura a memoria in streaming pensata per il
  video, e trasforma un clic di punto o di box in una maschera dell'oggetto.
  LibreYOLO supporta il suo percorso di segmentazione di immagini attraverso una
  factory LibreSAM dedicata, separata dalla factory di detector LibreYOLO().
keywords:
  - SAM 2
  - Segment Anything
  - promptable segmentation
  - segmentazione interattiva python
  - segmentare un oggetto con un clic
  - point prompt
  - box prompt
  - Meta AI
  - Hiera
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompt di punto e di box
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # Alias di dimensione: "sam2-tiny", "sam2-small", "sam2-base-plus",

        # "sam2-large" (anche le forme brevi
        "sam2-t"/"sam2-s"/"sam2-bp"/"sam2-l").

        model = LibreSAM("sam2-large")


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
        from libreyolo import LibreSAM2, SAMPLE_IMAGE


        # La classe specifica della famiglia prende la dimensione senza il
        prefisso "sam2-".

        model = LibreSAM2("large")


        # L'encoder dell'immagine è la parte costosa. set_image() lo esegue una
        volta;

        # ogni chiamata a predict() successiva riusa l'embedding in cache.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: 2a3090d7ecd533b0
---

## Installazione

SAM 2 richiede l'extra `sam`, che porta con sé `transformers` e `timm`.

```bash
pip install "libreyolo[sam]"
```

## Predizione

`LibreSAM(...)` (o il `LibreSAM2(...)` specifico della famiglia) è un punto di
ingresso separato da `LibreYOLO(...)`: restituisce un segmentatore guidato da
prompt invece di un detector, perché qui un forward pass non significa niente
senza un prompt spaziale. Per questa famiglia non esiste il comando CLI
`libreyolo predict`; usa l'API Python. È supportata solo la segmentazione di
immagini; il tracking con memoria video di SAM 2 qui è fuori ambito.

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
tutti `NotImplementedError` per questa famiglia: qui quello che LibreYOLO
supporta è l'inferenza sulle immagini. Vedi [predizione](/docs/predict) per i
tipi di sorgente.

## Varianti

Quattro dimensioni con backbone Hiera: tiny, small, base-plus e large, tutte
alla stessa risoluzione di input. Per questa famiglia non è ancora pubblicato
nessun benchmark di accuratezza o di latenza, quindi scegliere una dimensione
significa scambiare direttamente peso dell'encoder con qualità della maschera:
tiny è il più veloce da codificare, large il più pesante.

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenza

<provenance-box></provenance-box>

## Citazione

<citation-block />
