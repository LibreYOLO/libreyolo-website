---
title: Rilevamento di punti
seo_title: Rilevamento di punti e conteggio in LibreYOLO
description: >-
  Localizza gli oggetti come singoli punti invece che come box in LibreYOLO.
  Predici i centroidi, conta gli oggetti, addestra FOMO e leggi le metriche dei
  punti.
lead: >-
  Il rilevamento di punti restituisce una posizione x, y per oggetto invece di
  un bounding box. LibreYOLO lo espone come task point, e una predizione porta
  una riga con x, y, classe e confidenza per ogni oggetto.
keywords:
  - point detection python
  - contare oggetti in un'immagine python
  - rilevamento dei centroidi
  - FOMO point detection
  - conteggio di oggetti computer vision
  - localizzazione a punti
last_verified: 1.5.0
snippets:
  predict:
    - label: Predire i punti e contarli
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # I pesi di LibreFOMO non vengono scaricati automaticamente. Scarica
        prima

        # un checkpoint da https://huggingface.co/LibreYOLO e caricalo dal
        percorso locale.

        model = LibreYOLO("./LibreFOMOs-point.pt")

        result = model(SAMPLE_IMAGE, save=True)


        points = result.points

        print(len(points))     # numero di oggetti

        print(points.xy)       # (N, 2) centri in pixel dell'immagine originale

        print(points.cls, points.conf)
    - label: Coordinate normalizzate e conteggi per classe
      language: python
      code: |
        from collections import Counter

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("./LibreFOMOs-point.pt")
        result = model(SAMPLE_IMAGE)

        points = result.points.numpy()
        print(points.xyn)                          # gli stessi centri in [0, 1]
        print(Counter(points.cls.astype(int).tolist()))
  train:
    - label: Addestrare FOMO su un dataset YOLO
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(data="my-dataset.yaml", epochs=40, batch=32, lr0=3e-4)
    - label: Predire con il checkpoint addestrato
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("./LibreFOMOs-point.pt")

        results = model.train(data="my-dataset.yaml", epochs=40)


        # train() ricarica il checkpoint migliore nello stesso oggetto, quindi
        al

        # ritorno della chiamata il modello predice con i pesi addestrati.

        print(results["best_checkpoint"])

        print(model(SAMPLE_IMAGE).points.xy)
  val:
    - label: Validare e leggere le chiavi delle metriche
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("./LibreFOMOs-point.pt")

        metrics = model.val(data="my-dataset.yaml")


        print(metrics["metrics/precision"], metrics["metrics/recall"])

        print(metrics["metrics/f1"])

        print(metrics["metrics/mAP@[0.01:0.10]"])   # fitness

        print(metrics["metrics/MLE"])               # errore medio di
        localizzazione

        print(metrics["metrics/MAE"], metrics["metrics/RMSE"])   # errore di
        conteggio
    - label: Cambiare le soglie di distanza
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("./LibreFOMOs-point.pt")


        # I limiti dello sweep fanno parte del testo della chiave, quindi uno
        sweep

        # personalizzato rinomina le chiavi mAP che produce.

        metrics = model.val(data="my-dataset.yaml", dist_thresholds=[0.02,
        0.05])


        print(metrics["metrics/mAP@0.02"])

        print(metrics["metrics/mAP@[0.02:0.05]"])
  export:
    - label: Esportare
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
    - label: Eseguire il file esportato
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory smista in base al suffisso del file, quindi un artefatto
        esportato

        # si carica come qualsiasi checkpoint e restituisce lo stesso oggetto
        Results.

        model = LibreYOLO("./LibreFOMOs-point.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.points.xy)
source_hash: 932153c8870d1c7c
---

## Definizione

Il task `point` localizza ogni oggetto con una singola coordinata x, y e una
classe, senza larghezza, altezza o maschera. Poiché una predizione è un elenco
piatto di oggetti, il numero di righe è il numero di oggetti, ed è questo che ne
fa il task di conteggio.

Una predizione riempie `result.points`, un payload `Points` che incapsula un array
`(N, 4)` di righe `x, y, class, confidence` in pixel dell'immagine originale.
`.xy` restituisce le coordinate, `.xyn` le stesse coordinate divise per la
dimensione dell'immagine, `.cls` gli indici di classe e `.conf` i punteggi;
`len()` restituisce il numero di punti. `result.boxes` resta vuoto, quindi `iou`
e `max_det` non hanno nulla su cui agire.

## Modelli

Tre famiglie coprono `point`, e non sono intercambiabili.

[FOMO](/docs/models/fomo) è l'opzione a vocabolario fisso: un classificatore a
griglia che etichetta ogni cella di una griglia a bassa risoluzione come sfondo o
come centro di un oggetto. È l'unica famiglia point che LibreYOLO può addestrare,
e l'unica che si esporta.

[LocateAnything](/docs/models/locate-anything) accetta testo invece di un
indice di classe, quindi il vocabolario è qualunque frase tu scriva. Richiede
l'extra `vlm`, si costruisce come `LibreLocateAnything` invece che tramite la
factory `LibreYOLO()`, e i suoi pesi sono limitati all'uso non commerciale. I
termini esatti, e le altre due licenze che il checkpoint compone, sono sulla sua
pagina.

[SenseNova-Vision](/docs/models/sensenova-vision) arriva a `point` attraverso lo
stesso checkpoint a generazione guidata da prompt che usa per altri sei task,
caricato con `LibreVLM("sensenova-vision", task="point")`. Richiede l'extra
`sensenova`, e ogni predizione è un passaggio di generazione su un modello da 7B,
quindi aspettati una latenza per immagine nettamente più alta rispetto a un
rilevatore dedicato. I suoi pesi sono non commerciali; la licenza è sulla sua
pagina.

## Predizione

I pesi di LibreFOMO sono l'unica eccezione al download automatico su questo sito.
`LibreYOLO("LibreFOMOs-point.pt")` cerca quel file su disco e solleva un
`ValueError` che lo nomina invece di scaricarlo. Scarica prima un checkpoint
dall'[organizzazione LibreYOLO](https://huggingface.co/LibreYOLO) su Hugging Face
e caricalo dal percorso locale, oppure addestrane uno tuo.

<code-tabs name="predict" />

Il nome del file deve portare il suffisso di task `-point` perché il loader lo
riconosca. `predict(..., nms_radius=1)` controlla di quante celle della griglia
devono essere distanti due rilevamenti FOMO perché sopravvivano entrambi. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Formato del dataset

`point` non ha un formato di etichette proprio. Le famiglie point leggono il
layout standard di rilevamento YOLO e ricavano un centro da ogni riga di box,
quindi `cx cy` è il punto e `w h` decidono soltanto se la riga è valida.

```text
dataset/
  data.yaml
  images/
    train/scene.jpg
    val/scene.jpg
  labels/
    train/scene.txt
    val/scene.txt
```

Ogni file di etichette contiene una riga per oggetto, con coordinate
normalizzate:

```text
<class_id> <cx> <cy> <w> <h>
```

```yaml
path: dataset
train: images/train
val: images/val
nc: 1
names: {0: seedling}
```

Un file di etichette mancante o vuoto significa che non ci sono oggetti. Vedi
[formati dei dataset](/docs/reference/dataset-formats) per il contratto completo.

## Addestramento

FOMO è l'unica famiglia point con un'implementazione di addestramento. `train()`
su LocateAnything e su SenseNova-Vision solleva `NotImplementedError`; fai
fine-tuning di quei modelli a monte e carica il risultato.

<code-tabs name="train" />

`imgsz` non è una scelta libera con FOMO: il valore predefinito è la risoluzione
nativa del checkpoint caricato, e passare un valore diverso solleva un
`ValueError` che indica la dimensione attesa. Vedi [addestramento](/docs/train)
per dataset, logger e multi-GPU, e la [pagina di FOMO](/docs/models/fomo) per i
valori predefiniti di questa famiglia.

## Validazione

`val()` associa i punti predetti ai punti ground truth uno a uno con l'algoritmo
ungherese, su uno sweep di soglie di distanza. Una soglia è una distanza euclidea
in coordinate immagine normalizzate, e lo sweep predefinito è di dieci valori da
0.01 a 0.10.

<code-tabs name="val" />

`metrics/precision`, `metrics/recall` e `metrics/f1` sono mediate a livello macro
sulle classi alla soglia più severa dello sweep, 0.01 per impostazione
predefinita. `metrics/mAP@0.01` è l'average precision a quella stessa soglia, e
`metrics/mAP@[0.01:0.10]` è la media sull'intero sweep. Quel valore dello sweep è
anche `fitness`, il numero letto dalla selezione del checkpoint migliore.
Entrambe le chiavi mAP sono costruite a partire dalle soglie in uso, quindi
passare `dist_thresholds=` le rinomina.

`metrics/MLE` è la distanza media tra le coppie associate alla soglia più severa,
nelle stesse unità normalizzate. `metrics/MAE` e `metrics/RMSE` sono metriche di
conteggio più che di localizzazione: misurano la differenza, immagine per
immagine, tra il numero di punti predetti e quello dei punti ground truth.

A queste FOMO aggiunge un secondo gruppo, a livello di griglia. Fa uno sweep
su confidenza e `nms_radius` e pubblica la combinazione con l'F1 migliore come
`metrics/grid_F1`, `metrics/grid_precision`, `metrics/grid_recall`,
`metrics/grid_mean_distance`, `metrics/grid_TP`, `metrics/grid_FP` e
`metrics/grid_FN`, con le impostazioni che l'hanno prodotta sotto
`decode/threshold` e `decode/nms_radius`.

## Esportazione

FOMO si esporta attraverso il percorso di esportazione condiviso, e un artefatto
esportato si ricarica tramite `LibreYOLO()` in base al suffisso del file, quindi
un file `.onnx` o `.engine` si comporta come un checkpoint e restituisce lo
stesso `Results`.

<code-tabs name="export" />

La copertura per formato è sulla [pagina di FOMO](/docs/models/fomo) e nella
[matrice completa delle esportazioni](/docs/reference/export-matrix).
LocateAnything e SenseNova-Vision non si esportano: `export()` solleva
un'eccezione in entrambi i casi, perché un modello generativo non ha un grafo di
rilevamento tracciabile.
