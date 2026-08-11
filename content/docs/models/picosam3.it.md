---
title: PicoSAM3
families:
  - picosam3
seo_title: 'PicoSAM3: segmentazione edge guidata da box in LibreYOLO'
description: >-
  Usa PicoSAM3 in LibreYOLO per la segmentazione di regioni guidata da box su
  sensori edge. Installa, fai predizioni ed esporta il checkpoint pico sotto
  licenza Apache-2.0.
lead: >-
  PicoSAM3 è una CNN compatta distillata da SAM 2.1 e SAM 3, pensata per la
  segmentazione di regioni di interesse guidata da box su sensori come il Sony
  IMX500. LibreYOLO la supporta tramite una factory LibreSAM dedicata, separata
  dalla factory di detector LibreYOLO(), e solo con prompt di box.
keywords:
  - PicoSAM3
  - Segment Anything
  - edge segmentation
  - segmentazione con box python
  - region of interest
  - box prompt
  - IMX500
  - knowledge distillation
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompt di box
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # PicoSAM3 ha una sola taglia, "pico", quindi non serve nessun altro
        alias.

        model = LibreSAM("picosam3")


        # bboxes= è l'unico prompt supportato: [x1, y1, x2, y2] oppure una lista

        # di box, una maschera per box. Ogni box viene espanso del 10%, reso

        # quadrato, ritagliato sull'immagine e portato a 96x96 prima della CNN.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])

        print(result.masks.xy)      # un poligono per maschera

        print(result.boxes.xyxy)    # box aderente derivato dalla maschera
    - label: 'Codifica una volta, prompt multipli'
      language: python
      code: >
        from libreyolo import LibrePicoSAM3, SAMPLE_IMAGE


        model = LibrePicoSAM3()


        # set_image() mette in cache l'immagine di origine; PicoSAM3 esegue un

        # forward completo della CNN per box, quindi risparmia il caricamento e
        la

        # decodifica, non un passaggio di encoder come nelle altre famiglie SAM.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(bboxes=[300, 200, 900, 700])

        b = model.predict(bboxes=[100, 100, 400, 400])

        model.reset_image()
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibrePicoSAM3


        model = LibrePicoSAM3()

        model.export(format="onnx", output_path="LibrePicoSAM3pico.onnx")


        # opset (default 13) e dynamic (default True, solo l'asse del batch)
        sono

        # gli unici argomenti di esportazione accettati da questa famiglia.
    - label: Usare il file esportato
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # PicoSAM3 esporta la sua CNN ROI grezza a 96x96: roi_image ->
        mask_logits.

        # Qui non c'è nessun pre/postprocessing lato LibreYOLO da riutilizzare,

        # perché export() non viene instradato di nuovo attraverso LibreYOLO()

        # come succede per il checkpoint di un detector.

        session = ort.InferenceSession("LibrePicoSAM3pico.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 96, 96),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 5d60ff14fe61ba29
---

## Installazione

PicoSAM3 richiede l'extra `sam`: il download dei pesi di LibreYOLO passa ancora
dagli strumenti Hugging Face di `transformers`, anche se l'inferenza gira su una
CNN nativa che non usa `transformers`.

```bash
pip install "libreyolo[sam]"
```

## Predizione

`LibreSAM(...)` (o il `LibrePicoSAM3(...)` specifico della famiglia) è un punto
di ingresso distinto da `LibreYOLO(...)`: restituisce un segmentatore guidato da
prompt invece di un detector, perché qui un forward pass non ha senso senza un
prompt. Per questa famiglia non esiste il comando CLI `libreyolo predict`; usa
l'API Python.

<code-tabs name="predict" />

PicoSAM3 accetta solo `bboxes=`; passare `points=`, `labels=`, `masks=`,
`text=`, `multimask=True` oppure omettere il box per segmentare tutto solleva
in ogni caso un `ValueError` esplicito, perché nessuna di queste modalità
esiste nel modello originale. `conf` filtra in base alla qualità predetta della
maschera (IoU), non a una confidenza di rilevamento, e deve stare tra `0.0` e
`1.0`. Ogni maschera porta l'id di classe `0`, di nome `"object"`. `train()`,
`val()` e `track()` sollevano `NotImplementedError`; usa LibreSAM2 o LibreSAM3
per prompt di punto, di testo, di maschera o per segmentare tutto. Vedi
[predizione](/docs/predict) per i tipi di sorgente.

## Varianti

Una sola taglia, pico, con un input ROI fisso di 96 px: PicoSAM3 esegue un
forward completo della CNN per ogni box invece di codificare una volta sola
l'intera immagine.

## Esportazione

<export-matrix />

PicoSAM3 è l'unica famiglia del livello SAM che si esporta: porta in ONNX la sua
CNN ROI grezza a 96x96, `roi_image -> mask_logits`, senza NMS né postprocessing
delle maschere incorporati. Le altre famiglie SAM sollevano
`NotImplementedError` su `export()`, perché la loro separazione tra encoder e
decoder non ha ancora un contratto di esportazione definito per il runtime. Un
grafo PicoSAM3 esportato non si ricarica tramite `LibreYOLO()`; eseguilo
direttamente con un runtime come `onnxruntime`, applicando lo stesso
preprocessing di ROI quadrata con margine del 10% mostrato sopra.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenza

<provenance-box>

PicoSAM3 è distillato da SAM 2.1 e SAM 3 usati come modelli teacher. LibreYOLO
non incorpora né ridistribuisce il codice o i pesi di nessuno dei due teacher in
questa famiglia; vengono distribuiti solo la CNN student compatta e il suo
checkpoint convertito.

</provenance-box>

## Citazione

<citation-block />
