---
title: EdgeTAM
families:
  - edgetam
seo_title: 'EdgeTAM: segmentazione con prompt on-device in LibreYOLO'
description: >-
  Usa EdgeTAM in LibreYOLO per la segmentazione con prompt a punti e box,
  pensata per la velocità on-device. Installa e fai predizioni con il checkpoint
  sotto Apache-2.0.
lead: >-
  EdgeTAM è una variante on-device di SAM 2, costruita per la velocità di
  inferenza su mobile mantenendo lo stesso flusso di lavoro con prompt a punti e
  box. LibreYOLO supporta il suo percorso di segmentazione di immagini
  attraverso una factory LibreSAM dedicata, separata dalla factory del detector
  LibreYOLO().
keywords:
  - EdgeTAM
  - SAM 2
  - promptable segmentation
  - segmentazione interattiva python
  - on-device segmentation
  - prompt a punti
  - prompt a box
  - Meta Reality Labs
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompt a punti e box
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # EdgeTAM ha una sola taglia, "edge". Alias: "edgetam", "edge-tam",

        # "edgetam-edge".

        model = LibreSAM("edgetam")


        # Un prompt a punto: [x, y] in coordinate pixel, etichetta 1 = primo
        piano.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # poligono per maschera

        print(result.boxes.xyxy)    # box aderente derivato dalla maschera


        # Un prompt a box invece di un punto.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])


        # Nessun prompt segmenta l'intera immagine (un generatore automatico

        # di maschere semplificato, non quello esaustivo di riferimento).

        result = model.predict(SAMPLE_IMAGE)
    - label: 'Codifica una volta, usa molti prompt'
      language: python
      code: |
        from libreyolo import LibreEdgeTAM, SAMPLE_IMAGE

        model = LibreEdgeTAM()

        # L'encoder dell'immagine è la parte costosa. set_image() lo esegue una
        # volta; ogni predict() successiva riusa l'embedding in cache.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
source_hash: e6cce8faad18e73d
---

## Installazione

EdgeTAM richiede l'extra `sam`, che porta con sé `transformers` e `timm`.

```bash
pip install "libreyolo[sam]"
```

## Predizione

`LibreSAM(...)` (o il `LibreEdgeTAM(...)` specifico della famiglia) è un entry
point separato da `LibreYOLO(...)`: restituisce un segmentatore guidato da
prompt invece di un detector, perché qui un forward pass non ha senso senza un
prompt spaziale. Per questa famiglia non esiste un comando CLI
`libreyolo predict`; usa l'API Python. È supportata solo la segmentazione di
immagini; il tracking video di EdgeTAM è fuori dallo scopo di questa pagina.

<code-tabs name="predict" />

Un prompt a punto accetta `[x, y]` per un solo oggetto, `[[x, y], ...]` per più
oggetti, oppure array numpy; `labels` marca ogni punto con `1` (primo piano) o
`0` (sfondo) e per default li considera tutti primo piano. Un prompt a box
prende `[x1, y1, x2, y2]` oppure una lista di box, una maschera per box.
Omettere entrambi i prompt segmenta l'intera immagine interrogando una griglia
densa e tenendo le maschere sicure e non sovrapposte; questa modalità "segmenta
tutto" è semplificata rispetto al generatore automatico di maschere di
riferimento e può sotto-segmentare le scene affollate, quindi un vero prompt a
punto o a box è la strada precisa. `conf` filtra in base alla qualità predetta
della maschera (IoU), non a una confidenza di rilevamento: passa `0.0` per
tenere ogni candidato. `multimask=True` restituisce per ogni prompt tutte e tre
le maschere di ambiguità intero-contro-parte di SAM invece della sola migliore.
`device=` sposta il modello e, se è attiva una sessione `set_image()`, anche il
suo embedding in cache. Ogni maschera porta l'id di classe `0`, di nome
`"object"`, perché una maschera guidata da prompt non ha un insieme di classi
fisso. `train()`, `val()`, `export()` e `track()` sollevano tutti
`NotImplementedError` per questa famiglia: qui LibreYOLO supporta l'inferenza
su immagini. Vedi [predizione](/docs/predict) per i tipi di sorgente.

## Varianti

Una sola taglia, edge, a risoluzione di input fissa, quindi scegliere questa
famiglia al posto del resto del livello SAM è una decisione hardware più che di
dimensionamento: EdgeTAM esiste proprio per l'inferenza on-device su risorse
limitate.

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenza

<provenance-box></provenance-box>

## Citazione

<citation-block />
