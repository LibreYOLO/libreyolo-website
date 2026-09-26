---
title: CLIP
families:
  - clip
seo_title: 'CLIP in LibreYOLO: classificazione zero-shot ed embedding'
description: >-
  Usa CLIP in LibreYOLO per la classificazione zero-shot di immagini e per
  l'embedding di immagini e testo. Nessun addestramento: set_classes() definisce
  l'insieme di etichette a runtime.
lead: >-
  CLIP è un modello a due torri che confronta un'immagine con dei prompt
  testuali invece che con un insieme fisso di etichette. LibreYOLO lo supporta
  per la classificazione zero-shot e per l'embedding di immagini e testo, senza
  nessun passaggio di addestramento.
keywords:
  - CLIP
  - OpenCLIP
  - zero-shot classification
  - classificazione zero-shot python
  - image embedding
  - embedding immagini testo
  - open vocabulary
  - LAION-2B
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: >
        # Senza una chiamata a set_classes(), predict da CLI usa i 1.000 nomi

        # di classe ImageNet con cui il modello viene caricato di default.

        libreyolo predict model=LibreCLIPb32-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Embedding di immagini e testo
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        image_embed = model(SAMPLE_IMAGE).embeddings.data

        text_embed = model.embed_text("a photo of a forklift")


        # Entrambi sono normalizzati L2, quindi un semplice prodotto scalare è
        la similarità coseno.

        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreCLIPb32-cls.pt")


        # data è una directory radice ImageFolder con uno split train/; i nomi

        # delle cartelle diventano i prompt di classe zero-shot per questa
        esecuzione.

        metrics = model.val(data="imagenette160")


        print(metrics["metrics/accuracy_top1"])

        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCLIPb32-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        model.export(format="onnx")

        # Le etichette correnti di set_classes() e la risoluzione di input
        # vengono fissate nel grafo. Riesporta dopo aver cambiato una delle due.
    - label: CLI
      language: bash
      code: >
        # Qui non c'è nessuna chiamata a set_classes(), quindi vengono fissate

        # le 1.000 classi ImageNet predefinite con cui il modello viene
        caricato.

        libreyolo export model=LibreCLIPb32-cls.pt format=onnx
    - label: Esportazione degli embedding
      language: python
      code: |
        from libreyolo import LibreYOLO

        # task="embed" traccia solo la torre delle immagini; non servono classi.
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        model.export(format="onnx")
source_hash: ac7cfd75ad6c0fa7
---

## Installazione

CLIP richiede un extra dedicato, che installa i pacchetti usati dal suo
tokenizer BPE incorporato per riprodurre esattamente gli id dei token.

```bash
pip install "libreyolo[clip]"
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella
cache locale.

<code-tabs name="predict" />

`set_classes()` è l'unica primitiva che rende questo modello un classificatore
a vocabolario aperto: inserisce ogni etichetta in ciascun template di prompt,
codifica e media i risultati e memorizza nella cache la matrice `[K, D]`
risultante come testa del classificatore, così non viene ricalcolata per ogni
immagine. Chiamala di nuovo per cambiare classi in qualsiasi momento. Senza
nessuna chiamata, LibreCLIP viene caricato con i 1.000 nomi di classe di
ImageNet-1k già impostati.

Con `task="embed"`, la predizione restituisce un vettore immagine normalizzato
L2 per ogni input invece delle probabilità di classe, e `embed_text()`
restituisce righe di testo normalizzate nello stesso spazio vettoriale, quindi
un semplice prodotto scalare tra i due è la similarità coseno. `iou` non ha
effetto su nessuno dei due task; non c'è nessun passaggio di NMS. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Validazione

`val()` legge i nomi delle cartelle di classe sotto lo split `train/` di un
ImageFolder, li passa a `set_classes()` e poi misura l'accuratezza zero-shot
top-1 e top-5. L'accuratezza dipende da come i nomi delle classi funzionano
come prompt, non da un aggiornamento dei pesi, dato che non c'è niente da
addestrare. La validazione copre solo `task="classify"`; `task="embed"` non ha
un validatore su dataset.

<code-tabs name="val" />

## Esportazione

<export-matrix />

L'esportazione fissa lo stato corrente del modello in un grafo statico. Per
`task="classify"`, le ultime etichette impostate da `set_classes()` e la
risoluzione al momento dell'esportazione vengono fissate in un layer lineare
finale, quindi il grafo ONNX o TensorRT esportato è un normale classificatore
di immagini `[B, K]` senza torre testuale e senza tokenizer; riesporta dopo
aver cambiato le classi o la dimensione. L'esportazione con `task="embed"`
traccia solo la torre delle immagini. Entrambe richiedono l'opset ONNX 14 o
superiore, che l'esportatore imposta di default.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia. Entrambi sono convertiti
dai checkpoint di OpenCLIP addestrati su LAION-2B (`ViT-B-32` e `ViT-B-16`),
non da un addestramento su COCO.

<checkpoint-table />

I dati di addestramento LAION-2B hanno una storia documentata di contenuti CSAM
(Stanford Internet Observatory, dicembre 2023). Da allora LAION ha pubblicato
Re-LAION, una nuova release ripulita; se ridistribuisci ulteriormente questi
pesi, preferisci dove disponibili i checkpoint derivati da Re-LAION.

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
