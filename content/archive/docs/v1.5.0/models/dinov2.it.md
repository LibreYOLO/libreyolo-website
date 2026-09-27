---
title: DINOv2
families:
  - dinov2
seo_title: 'DINOv2 in LibreYOLO: semantica, classificazione ed embedding'
description: >-
  Usa DINOv2 in LibreYOLO per la segmentazione semantica, la classificazione e
  l'embedding dell'immagine intera sul backbone DINOv2-with-Registers.
  Apache-2.0 dall'inizio alla fine.
lead: >-
  DINOv2 è un vision transformer autosupervisionato addestrato da Meta AI per
  produrre feature di immagine generiche senza etichette. LibreYOLO incapsula il
  suo backbone DINOv2-with-Registers per tre task: segmentazione semantica,
  classificazione ed embedding dell'immagine intera.
keywords:
  - DINOv2
  - DINOv2 with registers
  - self-supervised learning
  - apprendimento autosupervisionato
  - vision transformer
  - segmentazione semantica python
  - embedding immagini
  - estrazione di feature
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: Semantica
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # Per questa famiglia non esiste nessun checkpoint ospitato da
        # LibreYOLO: questo scarica il backbone Apache-2.0
        # DINOv2-with-Registers-small dall'org Hugging Face di Meta. La testa
        # densa parte da un'inizializzazione casuale finché non la addestri
        # (vedi Addestramento più sotto).
        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        result = model(SAMPLE_IMAGE)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: Classificazione
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # nb_classes= è il numero di classi del tuo dataset; la testa lineare
        # parte da un'inizializzazione casuale finché non la addestri.
        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
    - label: Embedding
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # Salta tutte le teste di task: basta il backbone da solo, quindi non
        # serve nessun fine-tuning perché sia utile.
        model = LibreDINOv2(size="s", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (1, D), normalizzato L2
    - label: Embedding di un batch
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # Wrapper di comodo: esegue predict() e impila ogni riga in un unico
        # tensore (N, D).
        features = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(features.shape)
  train:
    - label: Semantica
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: Classificazione
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: Multi-GPU
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(
            data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4,
            device="0,1",
        )
  val:
    - label: Semantica
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: Classificazione
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
  export:
    - label: Semantica
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.export(format="onnx")
    - label: Classificazione
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.export(format="onnx")
    - label: Embedding
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        model.export(format="tflite")
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results. L'esportazione dà il nome al file a partire dal
        # task, qui LibreDINOv2s-sem.onnx.
        model = LibreYOLO("LibreDINOv2s-sem.onnx")
        result = model(SAMPLE_IMAGE)
source_hash: 4256e0a0398e5aaf
---

## Installazione

LibreDINOv2 si registra solo quando `transformers` è installato, la stessa
dipendenza opzionale che serve a RF-DETR per il suo backbone DINOv2, quindi
richiede lo stesso extra.

```bash
pip install "libreyolo[rfdetr]"
```

## Predizione

LibreYOLO non pubblica nessun checkpoint LibreDINOv2. Costruisci il wrapper
direttamente invece di caricare un file: `model_path=None` (il valore
predefinito) scarica al primo utilizzo il backbone Apache-2.0
`facebook/dinov2-with-registers-small` di Meta da Hugging Face. `task=`
seleziona che cosa gira sopra di esso.

<code-tabs name="predict" />

`task="semantic"` e `task="classify"` aggiungono una testa densa o lineare
sopra il backbone; quella testa è inizializzata in modo casuale ed è utile solo
dopo che l'hai addestrata (vedi [Addestramento](#train)). `task="embed"` salta
tutte le teste e restituisce il token CLS finale normalizzato del backbone come
un'unica riga per l'immagine intera in `result.embeddings`, quindi non ha
bisogno di nessun addestramento. `result.boxes` è sempre `None`: nessuno dei
tre task produce rilevamenti per istanza. Vedi [predizione](/docs/predict) per
sorgenti, streaming e gestione dei risultati.

## Varianti

`size` seleziona la larghezza del proiettore in stile RF-DETR sovrapposto al
backbone, non il backbone stesso: tutte le dimensioni condividono lo stesso
encoder DINOv2-S (small). La segmentazione semantica gira alla griglia di patch
quadrata nativa di DINOv2; la classificazione e l'embedding girano alla
risoluzione di classificazione, più piccola, usata per addestrare il linear
probe.

## Addestramento

`task="semantic"` e `task="classify"` si addestrano entrambi; `task="embed"`
non ha nessuna testa dipendente dalle classi da adattare e solleva
`NotImplementedError` se ci chiami `train()` sopra.

<code-tabs name="train" />

Gli argomenti a parola chiave principali qui sono `batch_size` e `lr`, non
`batch` e `lr0` usati dalla maggior parte delle altre famiglie; `batch` e `lr0`
sono ancora accettati e mappati su di essi, ma passarli entrambi solleva un
errore di conflitto. `output_dir=` (predefinito `"runs/train"`) sostituisce
`project=`/`name=` come modo principale per collocare un'esecuzione, anche se
passare `project=`/`name=` direttamente continua a funzionare. Vedi
[addestramento](/docs/train) per dataset, data augmentation, multi-GPU e
logger.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/`: mIoU e accuratezza per
pixel per `task="semantic"`, accuratezza top-1 e top-5 per `task="classify"`.
`task="embed"` non ha nessun ground truth con cui misurarsi e solleva
`NotImplementedError` se ci chiami `val()` sopra.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Ogni task supporta un sottoinsieme diverso di formati, mostrato qui sopra. Un
artefatto esportato si ricarica tramite `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce gli stessi `Results`. [Esportazione](/docs/export) elenca gli
argomenti che ogni formato accetta.

<code-tabs name="export" />

## Licenze

<provenance-box>

La riga "Pesi" qui sopra indica la licenza che si applica, Apache-2.0, ma per
questa famiglia non viene ripubblicato niente sotto l'org Hugging Face di
LibreYOLO: LibreYOLO non ospita nessun checkpoint LibreDINOv2 proprio. Quello
che `LibreDINOv2(model_path=None)` scarica è il repository
`facebook/dinov2-with-registers-small` di Meta, intatto.

</provenance-box>

## Citazione

<citation-block />
