---
title: Segmentazione semantica
seo_title: Segmentazione semantica in LibreYOLO
description: >-
  Assegna una classe a ogni pixel in LibreYOLO: le famiglie che coprono il task,
  il formato delle maschere dense e le chiamate di predizione, addestramento,
  validazione ed esportazione.
lead: >-
  La segmentazione semantica assegna una classe a ogni pixel di un'immagine e
  non distingue tra istanze della stessa classe. La chiave del task è semantic.
keywords:
  - semantic segmentation python
  - segmentazione semantica python
  - classificazione dei pixel
  - dense prediction
  - addestrare modello di segmentazione
  - mIoU
  - libreria segmentazione MIT
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Il suffisso -sem nel nome del file seleziona il task, quindi non
        # serve l'argomento task.
        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) id di classe sul canvas originale
        print(mask.classes)      # id di classe presenti, ordinati, 255 escluso
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSegformerb0-sem.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Una classe alla volta
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreSegformerb0-sem.pt")(SAMPLE_IMAGE)
        mask = result.semantic_mask

        for class_id in mask.classes:
            pixels = mask.class_mask(class_id)   # booleana (H, W)
            print(result.names[class_id], int(pixels.sum()))
    - label: 'Un''altra famiglia, stessa chiamata'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePIDNets-sem.pt")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: Su ADE20K
      language: bash
      code: |
        # ade20k.yaml incorpora uno script di download per l'archivio da ~1 GB,
        # quindi serve un permesso esplicito se i dati non sono già locali.
        libreyolo train model=LibreSegformerb0-sem.pt data=ade20k.yaml \
          epochs=160 imgsz=512 batch=8 allow_download_scripts=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")

        # val() restituisce un dict semplice, non un oggetto.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory sceglie in base al suffisso del file, quindi un artefatto
        # esportato si carica come un checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreSegformerb0-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 44b92d8ba6062f04
---

## Definizione

La segmentazione semantica etichetta i pixel, non gli oggetti. Ogni pixel
riceve un id di classe, e due auto che si toccano nell'immagine diventano
un'unica regione della classe auto, senza alcun confine tra loro. Contare le
istanze è la [segmentazione di istanze](/docs/tasks/instance-segmentation);
etichettare ogni pixel e separare al tempo stesso le istanze è la
[segmentazione panottica](/docs/tasks/panoptic-segmentation).

`semantic` è la chiave canonica del task, e il suffisso `-sem` nel nome del
file di un checkpoint la seleziona, quindi `task=` non serve quando carichi
pesi pubblicati.

`predict()` riempie `result.semantic_mask`. `.data` è una mappa `(H, W)` di id
di classe interi sul canvas dell'immagine originale, `.classes` elenca gli id
presenti in ordine crescente e `.class_mask(id)` restituisce la selezione
booleana `(H, W)` per una classe. Il valore `255` è l'etichetta ignore: non è
mai una classe, è escluso da loss e metriche, e `.classes` lo lascia fuori.

## Modelli

Tre famiglie sia addestrano sia predicono:
[SegFormer](/docs/models/segformer),
[LingBot-Vision](/docs/models/lingbot-vision) e
[DINOv2](/docs/models/dinov2). SegFormer e LingBot-Vision funzionano con il
pacchetto base e hanno pesi pubblicati. DINOv2 richiede
`pip install "libreyolo[rfdetr]"` e non ha un checkpoint ospitato da
LibreYOLO: carica il backbone originale e la sua testa densa parte da
un'inizializzazione casuale, quindi è un punto di partenza per l'addestramento
più che un predittore pronto all'uso.

Altre quattro fanno predizione, validazione ed esportazione, ma il loro
`train()` solleva `NotImplementedError`: [FCN](/docs/models/fcn),
[DeepLabv3](/docs/models/deeplabv3), [PIDNet](/docs/models/pidnet) ed
[EoMT](/docs/models/eomt).

Gli insiemi di classi cambiano da checkpoint a checkpoint, non da famiglia a
famiglia. I pesi pubblicati vengono da dataset i cui spazi di etichette hanno
poco in comune, tra cui le 150 classi di ADE20K contro le 19 di Cityscapes,
quindi il campo `names` di un checkpoint è ciò che ti dice che cosa sa
etichettare, e due checkpoint sono confrontabili solo se sono stati addestrati
sullo stesso dataset.

## Predizione

I pesi si scaricano da Hugging Face al primo utilizzo e restano in cache
localmente.

<code-tabs name="predict" />

La mappa è un argmax per pixel, quindi non c'è alcun passaggio di NMS e `iou`
non ha mai effetto. `conf` e `max_det` sono accettati per parità di API e non
fanno nulla su SegFormer, PIDNet e gli altri predittori densi; EoMT è
l'eccezione, dove `conf` filtra la selezione delle query. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Formato del dataset

Ogni immagine è abbinata a una maschera densa a canale singolo invece che a un
file di etichette `.txt`, trovata sostituendo `images` con la directory delle
maschere nel percorso dell'immagine.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  masks/
    train/000001.png
    val/000101.png
```

Le maschere sono immagini a canale singolo senza perdita di qualità,
normalmente PNG, e i PNG in modalità palette vengono letti come indici di
palette. Ogni valore di pixel è un id di classe in `0..nc-1`, il valore `255`
significa ignore, e la risoluzione della maschera deve essere uguale a quella
dell'immagine abbinata.

Lo YAML accetta due chiavi in più rispetto al contratto condiviso:

```yaml
path: dataset
train: images/train
val: images/val
masks_dir: masks
nc: 19
names:
  0: road
  1: sidewalk
```

`masks_dir` è il nome della directory sostituito a `images`, con valore
predefinito `masks`. `label_mapping` è un rimappaggio opzionale
`{source_id: train_id}` applicato ai valori dei pixel della maschera al
caricamento, ed è così che un dataset numerato da 1 a 150 diventa da 0 a 149;
qualsiasi valore di origine non mappato diventa ignore, e ogni train id deve
ricadere in `0..nc-1`.

Omettere `masks_dir` fa passare il loader a un percorso di ripiego: le maschere
vengono rasterizzate al caricamento a partire dalle etichette poligonali
risolte con la solita convenzione da `images` a `labels`, e una classe
`background` viene aggiunta dopo le classi degli oggetti, quindi `nc` cresce di
uno.

Il loader canonico è `libreyolo.data.SemanticDataset`.

## Addestramento

<code-tabs name="train" />

Qui `imgsz` è vincolato in un modo che non vale per un detector. Ogni famiglia
dichiara un divisore di cui il suo input deve essere un multiplo, fissato dalla
griglia di patch o dallo stride di output, e sia l'addestramento sia la
validazione sollevano un `ValueError` prima che l'esecuzione inizi quando
`imgsz` non è divisibile esattamente. Il divisore è 32 per SegFormer, 16 per
LingBot-Vision ed EoMT, 14 per DINOv2 e 8 per FCN e PIDNet. Vedi
[addestramento](/docs/train) per dataset, augmentation, multi-GPU e logger.

## Validazione

`val()` restituisce un dizionario semplice di chiavi `metrics/`, calcolate
sullo split indicato da `val` nello YAML del dataset.

<code-tabs name="val" />

`metrics/mIoU` è la media dell'intersection over union: per ogni classe, la
sovrapposizione tra i pixel predetti e quelli veri divisa per la loro unione,
mediata sulle classi. È il numero di riferimento, ed è quello usato per
scegliere l'epoca migliore durante l'addestramento. `metrics/pixel_accuracy` è
la quota di pixel a cui è stata assegnata la classe corretta, che una classe di
sfondo estesa può gonfiare, quindi mIoU è il valore su cui fare i confronti. I
pixel marcati `255` non contano né per l'una né per l'altra. Il dizionario
porta anche `fitness`, una copia del valore di mIoU.

## Esportazione

<code-tabs name="export" />

Un artefatto esportato si ricarica tramite `LibreYOLO()` in base al suffisso
del file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. La copertura dei formati cambia da famiglia a
famiglia; la matrice su ogni pagina di modello è generata dall'insieme
validato invece che scritta a mano. Vedi
[esportazione e deployment](/docs/export) per i formati, i loro extra e i loro
vincoli.
