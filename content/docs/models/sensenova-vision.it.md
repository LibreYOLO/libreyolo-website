---
title: SenseNova-Vision
families:
  - sensenovavision
seo_title: 'SenseNova-Vision in LibreYOLO: 7 task, un solo checkpoint'
description: >-
  Usa SenseNova-Vision in LibreYOLO per rilevamento, segmentazione, panottica,
  posa, punti, profondità e OCR da un unico checkpoint generativo guidato da
  prompt.
lead: >-
  SenseNova-Vision è un modello multimodale unificato che imposta i task di
  visione come generazione guidata da prompt su un decoder condiviso: box,
  punti, keypoint e parole dell'OCR escono come testo etichettato, mentre le
  mappe di profondità, di maschera e panottiche escono come immagini che un
  decoder disegna. LibreYOLO lo carica tramite LibreVLM e supporta sette task a
  partire dall'unico checkpoint da 7B.
keywords:
  - SenseNova-Vision
  - SenseTime
  - modello multimodale unificato
  - Bagel
  - prompted detection
  - dense perception
  - referring segmentation
  - segmentazione panottica python
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="detect")
        model.set_classes(["bird", "boat"])
        result = model.predict("image.jpg")
        print(result.boxes.xyxy)

        # set_task() cambia task sullo stesso modello già caricato.
        model.set_task("depth")
        result = model.predict("image.jpg")
        depth = result.depth_map.data
    - label: Segmentazione referring e panottica
      language: python
      code: >
        from libreyolo import LibreVLM


        model = LibreVLM("sensenova-vision", task="segment")

        # La segmentazione è referring: serve una frase target, non una lista di
        classi.

        model.set_classes(["the person furthest to the right"])

        result = model.predict("street.jpg")

        mask = result.masks.data[0]


        model.set_task("panoptic")

        # Senza un vocabolario personalizzato, la panottica ripiega sulle
        categorie

        # panottiche COCO su cui il checkpoint è stato messo a punto.

        result = model.predict("street.jpg")

        segment_map = result.panoptic.data

        for segment in result.panoptic.segments_info:
            print(segment)
    - label: 'Punti, posa e OCR'
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="point")
        model.set_classes(["screw"])
        result = model.predict("board.jpg")
        print(result.points.xy)

        # Senza un vocabolario impostato, la posa ripiega su "person".
        model.set_task("pose")
        result = model.predict("gym.jpg")
        print(result.boxes.xyxy, result.keypoints.data.shape)

        model.set_task("ocr")
        result = model.predict("sign.jpg")
        print(result.ocr.texts)
source_hash: 8749277e1910baa4
---

## Installazione

SenseNova-Vision richiede un extra proprio, che porta con sé `accelerate` per il dispatch dei modelli grandi di cui questo checkpoint ha bisogno e, sulle piattaforme diverse da macOS, `bitsandbytes` per il caricamento a 4 bit.

```bash
pip install "libreyolo[sensenova]"
```

Il checkpoint è replicato su Hugging Face sotto l'organizzazione di LibreYOLO e si scarica automaticamente al primo utilizzo; è CC BY-NC 4.0, solo per uso non commerciale, e il caricatore stampa quell'avviso prima di ogni download automatico. Vedi Licenze qui sotto.

## Predizione

<code-tabs name="predict" />

Ogni predizione è una decodifica per diffusione sul backbone Bagel-MoT condiviso, quindi è un modello di capacità e non un modello in tempo reale: aspettati una latenza per immagine decisamente più alta di quella di un rilevatore o di un segmentatore fatti su misura. `dtype="auto"` (il valore predefinito) carica in bf16 su una GPU con memoria sufficiente e altrove ripiega sulla quantizzazione NF4 a 4 bit, che richiede `bitsandbytes`; passa `dtype="bf16"` per forzare la precisione piena su una GPU abbastanza grande. `noise_seed=42` alla costruzione fissa il seed del sampler di diffusione per ottenere output densi riproducibili; passa `noise_seed=None` per disattivare il seeding.

I sette task condividono un unico checkpoint caricato: `set_task()` passa dall'uno all'altro senza ricaricarlo. `set_classes()` imposta il vocabolario attivo; il rilevamento, i punti, la posa e la panottica accettano una lista di classi, mentre la segmentazione è referring e ha bisogno esattamente della frase da isolare. Ogni task restituisce il consueto oggetto `Results` con un contenuto diverso valorizzato: `boxes` per detect, `points` per point, `boxes` e `keypoints` per pose, `ocr` per OCR, `depth_map` per depth, `masks` per segment e `panoptic` (con `segments_info`) per panoptic. Vedi [predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Checkpoint

<checkpoint-table />

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
