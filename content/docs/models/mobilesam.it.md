---
title: MobileSAM
families:
  - mobilesam
seo_title: 'MobileSAM: segmentazione leggera guidata da prompt in LibreYOLO'
description: >-
  Usa MobileSAM in LibreYOLO per la segmentazione con prompt di punto e di box
  con un encoder TinyViT. Installa e fai predizioni con il checkpoint tiny sotto
  Apache-2.0.
lead: >-
  MobileSAM sostituisce l'encoder di immagini ViT-H di SAM con un encoder
  TinyViT distillato, così lo stesso flusso di lavoro con prompt di punto e di
  box gira su hardware più leggero. LibreYOLO include un port nativo tramite una
  factory LibreSAM dedicata, separata dalla factory di detector LibreYOLO().
keywords:
  - MobileSAM
  - Segment Anything
  - TinyViT
  - segmentazione con prompt
  - interactive segmentation python
  - segmentare un oggetto con un clic
  - point prompt
  - segmentazione leggera
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompt di punto e di box
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # MobileSAM ha un'unica taglia, "tiny", quindi non serve nessun altro
        alias.

        model = LibreSAM("mobilesam")


        # Un prompt di punto: [x, y] in coordinate in pixel, label 1 = primo
        piano.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # un poligono per maschera

        print(result.boxes.xyxy)    # box aderente derivato dalla maschera


        # Un prompt di box invece di un punto.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])


        # Senza nessun prompt viene segmentata l'intera immagine (un generatore

        # automatico di maschere semplificato, non quello esaustivo di
        riferimento).

        result = model.predict(SAMPLE_IMAGE)
    - label: 'Codifica una volta, lancia molti prompt'
      language: python
      code: >
        from libreyolo import LibreMobileSAM, SAMPLE_IMAGE


        model = LibreMobileSAM()


        # L'encoder di immagini è la parte costosa. set_image() lo esegue una

        # volta; ogni chiamata a predict() successiva riusa l'embedding in
        cache.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: f96e885d93f72bdd
---

## Installazione

MobileSAM richiede l'extra `sam`: il download dei pesi di LibreYOLO passa
ancora dagli strumenti di snapshot di Hugging Face di `transformers`, anche se
l'inferenza gira su un decoder nativo che non usa `transformers`.

```bash
pip install "libreyolo[sam]"
```

## Predizione

`LibreSAM(...)` (oppure il `LibreMobileSAM(...)` specifico della famiglia) è un
punto di ingresso separato da `LibreYOLO(...)`: restituisce un segmentatore
guidato da prompt invece di un detector, perché qui un forward pass non ha
senso senza un prompt spaziale. Non esiste un comando CLI `libreyolo predict`
per questa famiglia; usa l'API Python.

<code-tabs name="predict" />

Un prompt di punto accetta `[x, y]` per un oggetto, `[[x, y], ...]` per più
oggetti, oppure array numpy; `labels` marca ogni punto con `1` (primo piano) o
`0` (sfondo) e per default sono tutti di primo piano. Un prompt di box prende
`[x1, y1, x2, y2]` o una lista di box, con una maschera per box. Omettendo
entrambi i prompt viene segmentata l'intera immagine lanciando una griglia
densa di prompt e tenendo le maschere sicure e non sovrapposte; questa modalità
"segmenta tutto" è semplificata rispetto al generatore automatico di maschere
di riferimento e può sottosegmentare le scene affollate, quindi un vero prompt
di punto o di box è la strada precisa. `conf` filtra in base alla qualità
predetta della maschera (IoU), non a una confidenza di rilevamento: passa `0.0`
per tenere tutti i candidati. `multimask=True` restituisce tutte e tre le
maschere di ambiguità intero-contro-parte di SAM per ogni prompt, invece della
sola migliore. `device=` sposta il modello e, se è attiva una sessione
`set_image()`, il suo embedding in cache. Ogni maschera porta l'id di classe
`0`, di nome `"object"`, dato che una maschera guidata da prompt non ha un
insieme fisso di classi. `train()`, `val()`, `export()` e `track()` sollevano
tutti `NotImplementedError` per questa famiglia: in LibreYOLO MobileSAM è solo
di predizione. Vedi [predizione](/docs/predict) per i tipi di sorgente.

## Varianti

Un'unica taglia, tiny, con input fisso a 1024 px: MobileSAM viene distribuito
con un solo encoder TinyViT, invece della scala base/large/huge offerta da
SAM-1.

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenza

<provenance-box></provenance-box>

## Citazione

<citation-block />
