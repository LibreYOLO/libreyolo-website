---
title: SigLIP2
families:
  - siglip2
seo_title: 'SigLIP2 in LibreYOLO: classificazione zero-shot ed embedding'
description: >-
  Usa SigLIP2 in LibreYOLO per la classificazione zero-shot di immagini e per
  l'embedding di immagini e testo, con punteggi sigmoid multi-etichetta. Nessun
  addestramento richiesto.
lead: >-
  SigLIP2 è un modello a due torri che confronta un'immagine con dei prompt
  testuali usando un sigmoid indipendente per classe, invece di un softmax
  condiviso su un insieme fisso di etichette. LibreYOLO lo supporta per la
  classificazione zero-shot e per l'embedding di immagini e testo, senza nessun
  passaggio di addestramento.
keywords:
  - SigLIP2
  - SigLIP 2
  - zero-shot classification
  - classificazione zero-shot immagini
  - image embedding
  - embedding immagini testo
  - open vocabulary
  - sigmoid loss
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: >
        # Senza una chiamata a set_classes(), predict da CLI usa i 1.000 nomi

        # di classe ImageNet con cui il modello viene caricato di default.

        libreyolo predict model=LibreSigLIP2b16-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Punteggi sigmoid multi-etichetta
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreSigLIP2b16-cls.pt")

        model.set_classes(["a dog", "a cat", "outdoors"], multi_label=True)

        r = model(SAMPLE_IMAGE)


        # Probabilità indipendenti per classe: più di una, o nessuna, possono

        # avere un punteggio alto insieme. Il softmax (predefinito) invece le

        # normalizza in una distribuzione a etichetta singola, come fa
        LibreCLIP.

        for i, name in model.names.items():
            print(name, float(r.probs.data[i]))
    - label: Embedding di immagini e testo
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")

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


        model = LibreYOLO("LibreSigLIP2b16-cls.pt")


        # data è una directory radice ImageFolder con uno split train/; i nomi

        # delle cartelle diventano i prompt di classe zero-shot per questa
        esecuzione.

        metrics = model.val(data="imagenette160")


        print(metrics["metrics/accuracy_top1"])

        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSigLIP2b16-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        model.export(format="onnx")

        # Le etichette correnti di set_classes() e la risoluzione di input
        # vengono fissate nel grafo. Riesporta dopo aver cambiato una delle due.
        # multi_label deve essere False (il valore predefinito) al momento
        # dell'esportazione.
    - label: CLI
      language: bash
      code: >
        # Qui non c'è nessuna chiamata a set_classes(), quindi vengono fissate

        # le 1.000 classi ImageNet predefinite con cui il modello viene
        caricato.

        libreyolo export model=LibreSigLIP2b16-cls.pt format=onnx
    - label: Esportazione degli embedding
      language: python
      code: |
        from libreyolo import LibreYOLO

        # task="embed" traccia solo la torre delle immagini; non servono classi.
        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")
        model.export(format="onnx")
source_hash: f992655747fd8819
---

## Installazione

SigLIP2 richiede un extra dedicato, che installa il pacchetto SentencePiece usato dal suo tokenizer multilingue.

```bash
pip install "libreyolo[siglip2]"
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella cache locale.

<code-tabs name="predict" />

`set_classes()` è l'unica primitiva che rende questo modello un classificatore a vocabolario aperto: inserisce ogni etichetta in ciascun template di prompt, codifica e media i risultati e memorizza nella cache la matrice `[K, D]` risultante come testa del classificatore, così non viene ricalcolata per ogni immagine. Chiamala di nuovo per cambiare classi in qualsiasi momento. Senza nessuna chiamata, LibreSigLIP2 viene caricato con i 1.000 nomi di classe di ImageNet-1k già impostati.

SigLIP assegna un punteggio a ogni classe in modo indipendente: `logit = scale * (image . text) + bias`. Di default quell'insieme di logit passa comunque per un softmax, che dà una distribuzione a etichetta singola equivalente al comportamento di `top1`/`top5` di LibreCLIP. Passando `multi_label=True` a `set_classes()` (o alla costruzione del modello) si passa invece a probabilità sigmoid indipendenti, così più di una classe, o nessuna, può avere un punteggio alto sulla stessa immagine. Il tokenizer è un modello SentencePiece multilingue (vocabolario Gemma), quindi i nomi di classe in lingue diverse dall'inglese funzionano allo stesso modo.

Con `task="embed"`, la predizione restituisce un vettore immagine normalizzato L2 per ogni input invece delle probabilità di classe, e `embed_text()` restituisce righe di testo normalizzate nello stesso spazio vettoriale, quindi un semplice prodotto scalare tra i due è la similarità coseno. `iou` non ha effetto su nessuno dei due task; non c'è nessun passaggio di NMS. Vedi [predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Validazione

`val()` legge i nomi delle cartelle di classe sotto lo split `train/` di un ImageFolder, li passa a `set_classes()` e poi misura l'accuratezza zero-shot top-1 e top-5 con punteggio softmax. L'accuratezza dipende da come i nomi delle classi funzionano come prompt, non da un aggiornamento dei pesi, dato che non c'è niente da addestrare. La validazione copre solo `task="classify"`; `task="embed"` non ha un validatore su dataset.

<code-tabs name="val" />

## Esportazione

<export-matrix />

L'esportazione fissa lo stato corrente del modello in un grafo statico. Per `task="classify"`, le ultime etichette impostate da `set_classes()` e la risoluzione al momento dell'esportazione vengono fissate in un layer lineare finale con lo scale e il bias appresi, quindi il grafo esportato è un normale classificatore di immagini `[B, K]` senza torre testuale e senza tokenizer; riesporta dopo aver cambiato le classi o la dimensione. L'esportazione in modalità `multi_label=True` non è implementata; riportalo prima a `False`. L'esportazione con `task="embed"` traccia solo la torre delle immagini. Entrambe richiedono l'opset ONNX 14 o superiore, che l'esportatore imposta di default.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia. Entrambi sono convertiti dai checkpoint Apache-2.0 di Google `siglip2-base-patch16-256` e `siglip2-so400m-patch14-384`, non da un addestramento su COCO.

<checkpoint-table />

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
