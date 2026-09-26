---
title: SAM 3
families:
  - sam3
seo_title: 'SAM 3: segmentazione guidata da prompt e per concetto in LibreYOLO'
description: >-
  Usa SAM 3 in LibreYOLO per la segmentazione per punto, per box e per concetto
  testuale. Installa e fai predizioni con il checkpoint large, ad accesso
  limitato sotto la SAM License di Meta.
lead: >-
  SAM 3 estende SAM con un prompt di concetto testuale oltre ai soliti punti e
  box, così una frase come "yellow school bus" restituisce tutte le istanze
  corrispondenti. LibreYOLO supporta il suo percorso su immagini tramite una
  factory LibreSAM dedicata, separata dalla factory di detector LibreYOLO().
keywords:
  - SAM 3
  - Segment Anything
  - promptable segmentation
  - segmentazione con prompt
  - segmentare con testo
  - concept segmentation
  - prompt di testo
  - prompt di punto
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompt di punto e di box
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # "sam3" è l'unica taglia ("large"); alias: "sam3", "sam-3",
        "sam3-large".

        model = LibreSAM("sam3")


        # Un prompt di punto: [x, y] in coordinate in pixel, label 1 = primo
        piano.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # un poligono per maschera

        print(result.boxes.xyxy)    # box aderente derivato dalla maschera


        # Un prompt di box invece di un punto.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: Prompt di testo (concetto)
      language: python
      code: >
        from libreyolo import LibreSAM3, SAMPLE_IMAGE


        model = LibreSAM3("large")


        # Trova tutte le istanze che corrispondono alla frase, non un solo
        oggetto.

        # text= è mutuamente esclusivo con points, bboxes, labels e masks.

        result = model.predict(SAMPLE_IMAGE, text="a person")

        print(result.names)         # {0: "a person"}

        print(result.boxes.conf)    # il punteggio di rilevamento PCS per
        istanza
    - label: 'Codifica una volta, lancia molti prompt'
      language: python
      code: >
        from libreyolo import LibreSAM3, SAMPLE_IMAGE


        model = LibreSAM3("large")


        # L'encoder di immagini è la parte costosa. set_image() lo esegue una

        # volta; ogni chiamata a predict() successiva riusa l'embedding in
        cache.

        # Una chiamata con text= ricodifica internamente, dato che il tracker e

        # l'encoder di segmentazione per concetto non condividono la cache.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: c4fb6d5a622f99ff
---

## Installazione

SAM 3 richiede l'extra `sam`, che porta con sé `transformers` e `timm`.

```bash
pip install "libreyolo[sam]"
```

I pesi sono ad accesso limitato: visita
[huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), accetta
la SAM License di Meta, poi esegui `hf auth login` (oppure imposta `HF_TOKEN`)
prima del primo download. LibreYOLO registra un avviso di licenza la prima
volta che scarica questa famiglia.

## Predizione

`LibreSAM(...)` (oppure il `LibreSAM3(...)` specifico della famiglia) è un
punto di ingresso separato da `LibreYOLO(...)`: restituisce un segmentatore
guidato da prompt invece di un detector, perché qui un forward pass non ha
senso senza un prompt. Non esiste un comando CLI `libreyolo predict` per questa
famiglia; usa l'API Python. È supportata solo l'inferenza su immagini; i
modelli video di SAM 3 restano fuori dall'ambito di questa pagina.

<code-tabs name="predict" />

Il percorso di punto e box coincide con il resto della famiglia SAM: un prompt
di punto accetta `[x, y]` per un oggetto o `[[x, y], ...]` per più oggetti,
`labels` marca ogni punto con `1` (primo piano) o `0` (sfondo) e un prompt di
box prende `[x1, y1, x2, y2]` o una lista di box. Su questo percorso `conf`
filtra in base alla qualità predetta della maschera (IoU), non a una confidenza
di rilevamento.

Il percorso `text=` è ciò che SAM 3 aggiunge: una stringa di concetto
restituisce tutte le istanze corrispondenti nell'immagine tramite la Promptable
Concept Segmentation, e non può essere combinata con punti, box, labels o
maschere. Lì `conf` è il punteggio di rilevamento PCS invece dell'IoU della
maschera; lasciarlo al valore predefinito applica la soglia di 0.3 propria del
modello, e `conf=0.0` tiene tutti i candidati. Il `names` restituito associa
l'id di classe `0` alla stringa di concetto richiesta, dato che altrimenti una
maschera guidata da prompt non ha un insieme fisso di classi. `device=` sposta
il modello e, se è attiva una sessione `set_image()`, il suo embedding in cache.
`train()`, `val()`, `export()` e `track()` sollevano tutti
`NotImplementedError` per questa famiglia: in LibreYOLO SAM 3 è solo di
predizione, e il tracking video resta fuori ambito. Vedi
[predizione](/docs/predict) per i tipi di sorgente.

## Varianti

Un'unica taglia, large, con input fisso a 1008 px. SAM 3.1 non è supportato: la
sua implementazione porta con sé una licenza personalizzata che non può essere
inclusa in questo repository MIT, e la versione di Transformers da cui dipende
LibreYOLO non carica ancora il formato del suo checkpoint.

## Licenza

<provenance-box>

LibreYOLO non ospita una propria copia dei pesi di SAM 3 e non li
ridistribuisce. `LibreSAM("sam3")` scarica direttamente dal repository
`facebook/sam3` di Meta ad accesso limitato su Hugging Face, che richiede di
accettare la SAM License di Meta e di autenticarsi prima del primo download.

</provenance-box>

## Citazione

<citation-block />
