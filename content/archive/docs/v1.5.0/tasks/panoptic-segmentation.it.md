---
title: Segmentazione panottica
seo_title: Segmentazione panottica in LibreYOLO
description: >-
  Assegna a ogni pixel un solo segmento in LibreYOLO: le famiglie che servono il
  task, il formato di dataset COCO-panoptic e le chiamate di predizione e
  validazione.
lead: >-
  La segmentazione panottica assegna ogni pixel a esattamente un segmento, senza
  sovrapposizioni, unificando le istanze di oggetti numerabili con le regioni di
  sfondo amorfe. La chiave del task è panoptic.
keywords:
  - panoptic segmentation python
  - segmentazione panottica python
  - panoptic quality PQ
  - things and stuff segmentation
  - formato COCO panoptic
  - metrica PQ segmentazione
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Il suffisso -panoptic nel nome del file seleziona il task, quindi non
        # serve l'argomento task.
        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # (H, W) id dei segmenti
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreEoMTl-panoptic.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Un segmento alla volta
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreEoMTl-panoptic.pt")(SAMPLE_IMAGE)
        pan = result.panoptic

        for segment in pan.segments_info:
            pixels = pan.segment_mask(segment["id"])   # booleana (H, W)
            print(result.names[segment["category_id"]], int(pixels.sum()))
    - label: Un checkpoint più piccolo
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTs-panoptic.pt")
        result = model(SAMPLE_IMAGE)

        print(len(result.panoptic.segment_ids))
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")

        # val() restituisce un semplice dict, non un oggetto.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
        print(metrics["metrics/PQ_things"], metrics["metrics/PQ_stuff"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-panoptic.pt data=my-dataset.yaml
source_hash: b8adc9ccde7a4e6c
---

## Definizione

La segmentazione panottica è l'unione degli altri due task di segmentazione.
Ogni pixel riceve esattamente un segmento, i segmenti non si sovrappongono mai e
un segmento è o una thing, cioè un'istanza di oggetto numerabile, oppure stuff,
cioè una regione amorfa come il cielo o la strada. Questo la rende più rigorosa
della [segmentazione di istanze](/docs/tasks/instance-segmentation), che lascia
i pixel di sfondo non assegnati e permette alle maschere di sovrapporsi, e più
rigorosa della [segmentazione semantica](/docs/tasks/semantic-segmentation), che
etichetta ogni pixel ma unisce le istanze adiacenti di una stessa classe.

`panoptic` è la chiave canonica del task, e il suffisso `-panoptic` nel nome del
file di un checkpoint la seleziona, quindi `task=` non serve quando carichi i
pesi pubblicati.

`predict()` riempie `result.panoptic`. `.data` è una mappa `(H, W)` di id di
segmento interi sul canvas dell'immagine originale. `.segments_info` è una lista
di dict, uno per segmento, ciascuno con almeno `{"id", "category_id"}`, dove `id`
corrisponde a un valore nella mappa e `category_id` indicizza `result.names`.
`.segment_ids` elenca in ordine gli id presenti e `.segment_mask(id)` restituisce
la selezione booleana `(H, W)` di un singolo segmento. L'id di segmento `0` è il
valore void: pixel non etichettati, esclusi dalla metrica e lasciati fuori da
`.segment_ids`.

La distinzione tra thing e stuff è una proprietà della categoria, non del
singolo segmento. È registrata nei metadati di categoria dell'insieme di
etichette, e un payload di predizione può copiarla su ogni segmento come
`"isthing"` per comodità, ma sono i metadati di categoria a fare fede.

## Modelli

[EoMT](/docs/models/eomt) è la famiglia che serve questo task tramite
`LibreYOLO()`. Gira con il pacchetto base e distribuisce checkpoint panottici in
tre taglie, s, b e l, addestrati su COCO.

[SenseNova-Vision](/docs/models/sensenova-vision) emette anch'esso mappe
panottiche. È un modello generativo guidato da prompt, con una factory propria,
`LibreVLM`, e un extra proprio; senza un vocabolario impostato ricade sulle
categorie panottiche COCO su cui è stato messo a punto. I suoi pesi non sono
utilizzabili commercialmente. La latenza per immagine è molto più alta di quella
di un segmentatore dedicato, perché ogni predizione è una decodifica per
diffusione.

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e messi in cache in
locale.

<code-tabs name="predict" />

`conf` filtra la selezione delle query. Vedi [predizione](/docs/predict) per
sorgenti, streaming e gestione dei risultati.

## Formato del dataset

LibreYOLO adotta alla lettera il formato COCO-panoptic, di Kirillov et al.,
CVPR 2019. Non esiste un layout panottico specifico di LibreYOLO.

```text
dataset/
  data.yaml
  images/
    val/000000000139.jpg
  annotations/
    panoptic_val.json
    panoptic_val/000000000139.png
```

Ogni immagine è accoppiata a un PNG RGB alla stessa risoluzione, dove il colore
di ciascun pixel codifica l'id del segmento a cui appartiene:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

L'id di segmento `0`, il nero RGB, è void: pixel non etichettati che non premiano
né penalizzano una predizione. Ogni altro pixel appartiene esattamente a un
segmento.

Il JSON elenca, per ogni immagine, il PNG con gli id dei segmenti e i segmenti
che contiene:

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1}]
}
```

`annotations[].file_name` indica il nome del PNG dentro la directory panottica, e
`segments_info[].id` corrisponde a un valore in quel PNG. `iscrowd` marca le
regioni di gruppo: non vengono mai contate come falsi negativi, e una predizione
che ne copre gran parte non è un falso positivo. `isthing` sta su `categories` e
mai su un singolo segmento.

Lo YAML punta a entrambi:

```yaml
path: dataset
val: images/val
annotations:
  val: annotations/panoptic_val.json
panoptic_dir:
  val: annotations/panoptic_val
names:
  0: person
  1: bicycle
```

`annotations` e `panoptic_dir` accettano ciascuno un singolo percorso oppure una
mappatura per split. Gli id di categoria COCO grezzi sono di solito non
contigui, mentre i modelli predicono un intervallo contiguo `0..nc-1`, quindi gli
id vengono rimappati attraverso `names` per nome di categoria. Una categoria del
JSON assente da `names` è un errore, non uno scarto silenzioso, perché
scartarla varrebbe come un falso negativo permanente.

Il loader canonico è `libreyolo.data.PanopticDataset`.

## Addestramento

Oggi in LibreYOLO nessuna famiglia addestra la segmentazione panottica: il
`train()` di EoMT solleva `NotImplementedError`, quindi i checkpoint panottici si
usano così come sono pubblicati.

## Validazione

`val()` restituisce un semplice dizionario di chiavi `metrics/`, calcolate alla
risoluzione del ground truth sullo split indicato da `val` nello YAML del dataset.
Un segmento predetto e uno vero della stessa categoria corrispondono quando il
loro IoU supera 0.5, e quella corrispondenza è unica.

<code-tabs name="val" />

`metrics/PQ` è la Panoptic Quality, il numero di riferimento. All'interno di una
categoria è il prodotto di due fattori. La qualità di segmentazione è l'IoU
medio sui segmenti corrispondenti e dice quanto bene combaciano le forme messe
in corrispondenza. La qualità di riconoscimento è `TP / (TP + 0.5 FP + 0.5 FN)`,
l'F1 score della corrispondenza stessa, e dice quanti segmenti sono stati
effettivamente trovati.
Tutti e tre i valori vengono poi mediati sulle categorie che sono comparse, e
riportati come `metrics/PQ`, `metrics/SQ` e `metrics/RQ`, quindi la PQ riportata
è la media dei prodotti per categoria e non il prodotto delle due medie
riportate.

`metrics/PQ_things` e `metrics/PQ_stuff` mediano la stessa PQ per categoria
separatamente sulle categorie thing e sulle categorie stuff, e
`metrics/categories` conta le categorie che sono comparse e su cui quindi si è
mediato. Il dizionario porta anche `fitness`, una copia del valore di PQ.

## Esportazione

I checkpoint panottici non si esportano. `export()` solleva
`NotImplementedError` per questo task, perché l'output query-mask non ha ancora
un contratto di esportazione a runtime. Il task semantico di EoMT si esporta;
vedi [segmentazione semantica](/docs/tasks/semantic-segmentation) e
[esportazione e deployment](/docs/export).
