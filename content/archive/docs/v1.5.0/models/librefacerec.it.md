---
title: LibreFaceRec
families:
  - facerec
seo_title: 'LibreFaceRec: riconoscimento e verifica dei volti'
description: >-
  Usa LibreFaceRec in LibreYOLO per il rilevamento dei volti, l'embedding e la
  verifica. Installa e fai una predizione; i pesi di embedding sono Apache-2.0.
lead: >-
  LibreFaceRec è il task di face embedding di LibreYOLO: un rilevatore di volti
  individua e allinea i volti, e una testa di riconoscimento produce un
  embedding di identità normalizzato L2 per la verifica o la ricerca.
keywords:
  - LibreFaceRec
  - face recognition
  - riconoscimento facciale python
  - face embedding
  - embedding volti
  - face verification
  - verifica identità volto
  - ArcFace
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # I nomi librefacerec-* puntano a questa famiglia a prescindere dal
        # suffisso del file e al primo uso vengono scaricati dall'org Hugging
        # Face di LibreYOLO, insieme al rilevatore di volti predefinito.
        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (N, D), normalizzati L2
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=face.jpg
    - label: Verifica
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        # Confronta il volto più prominente di ciascuna immagine tramite la
        # similarità coseno dei loro embedding normalizzati L2.
        result = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)
        print(result["similarity"], result["same_person"])
    - label: Ricerca in una galleria
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("librefacerec-l.onnx")


        query = model("query.jpg").embeddings          # i volti di questa
        immagine

        gallery = model.embed(["a.jpg", "b.jpg", "c.jpg"])   # (N_total, D)


        # Similarità coseno (query_faces, N_total).

        scores = query.similarity(gallery)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")
        model.export(format="onnx")
source_hash: f1a345bb96e32f12
---

## Installazione

La testa di riconoscimento di LibreFaceRec passa per `onnxruntime`, che non fa
parte dell'installazione base.

```bash
pip install "libreyolo[onnx]"
```

## Predizione

<code-tabs name="predict" />

Il rilevamento e il riconoscimento sono due grafi ONNX separati dietro a una
sola chiamata: un rilevatore di volti individua e allinea ogni volto a un
ritaglio canonico, e la testa di riconoscimento restituisce un embedding
normalizzato L2 per ogni volto. Se non intervieni, `predict()` scarica e abbina
automaticamente il rilevatore predefinito incluso. `face_detector` accetta un
callable, un modello di rilevamento LibreYOLO o un'istanza di `FaceDetector`;
`face_boxes` salta del tutto il rilevamento usando i box che hai già.
`result.embeddings` contiene una riga per ogni volto rilevato, allineata con
`result.boxes`; il suo metodo `.similarity()` calcola la similarità coseno
rispetto a un altro embedding o a un'intera galleria in una sola chiamata. Per
confrontare direttamente due immagini invece di due embedding già calcolati,
`model.verify(image_a, image_b)` esegue rilevamento ed embedding su entrambe e
confronta il volto con la confidenza più alta. Qualsiasi altro modello di
riconoscimento ONNX che segua la convenzione ArcFace (in ingresso un ritaglio
allineato, in uscita embedding `(N, D)`) può essere sostituito passando il
percorso del suo file al posto di un nome `librefacerec-*`. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Esportazione

<export-matrix />

LibreFaceRec incapsula già un grafo ONNX pre-esportato; la riesportazione verso
un altro formato non è implementata.

## Licenze

<provenance-box>

Il rilevatore di volti predefinito incluso è un secondo artefatto con una
seconda licenza: YuNet di OpenCV Zoo, MIT, copyright Shiqi Yu. Non è stato
portato codice di architettura da nessuno dei due progetti; entrambi i grafi
vengono consumati in modo opaco tramite `onnxruntime`, quindi il wrapper di
LibreYOLO non contiene codice di terze parti ed è MIT ovunque.

</provenance-box>

## Citazione

<citation-block />
