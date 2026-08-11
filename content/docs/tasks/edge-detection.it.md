---
title: Rilevamento dei bordi
seo_title: Rilevamento dei bordi in LibreYOLO
description: >-
  Predici una mappa densa di probabilità dei bordi da un'immagine in LibreYOLO.
  Converti un checkpoint, applica una soglia alla mappa, valida con ODS e OIS,
  ed esporta.
lead: >-
  Il rilevamento dei bordi predice quanto è probabile che ogni pixel si trovi
  sul contorno di un oggetto. LibreYOLO lo espone come task edge, che
  restituisce una mappa densa di probabilità sul canvas dell'immagine originale
  invece di un insieme di segmenti.
keywords:
  - edge detection python
  - rilevamento bordi immagini python
  - mappa probabilità bordi
  - ODS OIS F-measure
  - DexiNed python
last_verified: 1.5.0
snippets:
  predict:
    - label: Predire una mappa di bordi
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # LibreYOLO non include nessun checkpoint edge; convertine uno prima
        (sotto).

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")

        result = model(SAMPLE_IMAGE, save=True)


        edges = result.edges

        print(edges.array.shape)          # (H, W) float32 in [0, 1]

        print(edges.binary(0.5).sum())    # conteggio dei pixel di bordo a 0.5
    - label: Scegliere la propria soglia
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")

        result = model(SAMPLE_IMAGE)


        # La mappa continua viene conservata perché la soglia resti una tua
        scelta.

        for t in (0.3, 0.5, 0.7):
            print(t, int(result.edges.binary(t).sum()))
    - label: Salvare la visualizzazione
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE)

        # plot() disegna la mappa; è definito per i risultati edge e normal.
        result.plot().save("edges.png")
  val:
    - label: Validare e leggere le chiavi delle metriche
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])              # fitness
        print(metrics["metrics/OIS"])
        print(metrics["metrics/best_threshold"])
    - label: Cambiare lo sweep e la tolleranza di corrispondenza
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(
            data="my-dataset.yaml",
            imgsz=352,
            edge_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5),
            edge_max_dist=0.0075,
        )

        print(metrics["metrics/ODS"], metrics["metrics/best_threshold"])
  export:
    - label: Esportazione
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        model.export(format="onnx", imgsz=352)
    - label: Eseguire il file esportato
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory instrada in base al suffisso del file, così un artefatto
        esportato

        # si carica come un checkpoint qualsiasi e restituisce lo stesso oggetto
        Results.

        model = LibreYOLO("weights/LibreDexiNedb-edge.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.edges.array.shape)
source_hash: bc286345540ed966
---

## Definizione

Il task `edge` predice una probabilità per pixel a partire da una singola
immagine RGB: `0` significa non-bordo e `1` significa bordo. La mappa resta
continua, quindi la scelta della soglia che la trasforma in un'immagine binaria
di contorni è lasciata a chi chiama, e la soglia giusta dipende dal dataset e
dall'uso a valle.

Una predizione riempie `result.edges`, un payload `EdgeMap` che contiene un
array float32 `(H, W)` in `[0, 1]` sul canvas dell'immagine originale. `.array`
restituisce quella mappa come NumPy e `.binary(threshold)` restituisce una
maschera booleana. `result.boxes` resta vuoto, quindi `conf`, `iou` e `max_det`
non hanno effetto. `Results.plot()` copre questo task e disegna la mappa
direttamente.

## Modelli

Tre famiglie servono `edge`.

[DexiNed](/docs/models/dexined), la Dense Extreme Inception Network, fonde
diverse uscite laterali in un'unica mappa di probabilità e gira a una
risoluzione nativa di 352 px.

[TEED](/docs/models/teed), il Tiny and Efficient Edge Detector, è una rete
piccola alla stessa risoluzione nativa di 352 px, con uno stride di downsample
di 4 contro i 16 di DexiNed, quindi accetta più valori di `imgsz`.

[LibreMODUS](/docs/models/libremodus) produce bordi in stile Canny come uno dei
target di un modello any-to-any. Richiede l'extra `modus` e un tuo account
Hugging Face autenticato, e non offre né `val()` né `export()`, quindi non
compare nelle sezioni di validazione ed esportazione qui sotto.

## Predizione

LibreYOLO non pubblica nessun checkpoint edge. I pesi DexiNed e TEED rilasciati
ufficialmente sono addestrati su BIPED, i cui termini di pubblicazione del
dataset limitano l'uso a scopi non commerciali, quindi LibreYOLO non ne fa il
mirror. Converti un checkpoint che sei autorizzato a usare, poi carica il file
convertito indicandone il percorso:

```bash
python weights/convert_dexined_weights.py upstream.pth weights/LibreDexiNedb-edge.pt --verify
```

<code-tabs name="predict" />

Il nome del file deve portare il suffisso di task `-edge` perché il loader lo
riconosca. `imgsz` deve essere divisibile per lo stride di downsample della
rete, e quando non lo è LibreYOLO solleva un errore chiaro che indica il
divisore. Vedi [predizione](/docs/predict) per sorgenti, streaming e gestione
dei risultati.

## Formato del dataset

La validazione edge abbina ogni immagine RGB a una mappa a canale singolo con lo
stesso nome base e la stessa risoluzione, più una maschera di validità
opzionale.

```text
dataset/
  data.yaml
  images/
    val/scene.jpg
  edges/
    val/scene.png
  masks/
    val/scene.png
```

```yaml
path: dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

Il target è un PNG o TIF a canale singolo, non una visualizzazione RGB. Le mappe
intere vengono divise per il massimo del loro dtype; le mappe float devono già
essere finite e in `[0, 1]`. I pixel della maschera contano come validi quando
sono diversi da zero, e i pixel di padding non contribuiscono mai a una metrica.
`edge_invert: true` copre le sorgenti che memorizzano bordi neri su bianco. Vedi
[formati dei dataset](/docs/reference/dataset-formats) per il contratto
completo.

## Addestramento

Nessuna famiglia edge in LibreYOLO ha un'implementazione dell'addestramento:
`train()` solleva `NotImplementedError` su tutte e tre. La pagina di ogni
modello indica lo script di conversione che trasforma un checkpoint addestrato
altrove in uno che LibreYOLO può caricare.

## Validazione

`val()` riporta le F-measure in stile BSDS. Le predizioni continue vengono prima
assottigliate con una non-maximum suppression del gradiente a quattro direzioni,
poi i pixel di bordo predetti e quelli del ground truth vengono abbinati uno a
uno entro una tolleranza di distanza.

<code-tabs name="val" />

`metrics/ODS` è la F-measure optimal-dataset-scale: i conteggi delle
corrispondenze vengono aggregati su tutto il dataset a ogni soglia, e viene
riportata la migliore di quelle F-measure aggregate. È anche `fitness`, il
numero che legge la selezione del miglior checkpoint. `metrics/OIS` è la
F-measure optimal-image-scale, la media sulle immagini della migliore F-measure
di ciascuna immagine, quindi lascia che ogni immagine scelga la propria soglia.
`metrics/best_threshold` è la singola soglia che ha prodotto ODS, ed è quella da
riusare in `edges.binary()` durante l'inferenza.

Due argomenti danno forma allo sweep. `edge_thresholds` è l'insieme delle soglie
provate, che per default va da 0.01 a 0.99 a passi di un centesimo.
`edge_max_dist` è la tolleranza di corrispondenza espressa come frazione della
diagonale dell'immagine, con default `0.0075`; una coppia più distante di così
non è una corrispondenza.

## Esportazione

Un modello edge esportato si ricarica tramite `LibreYOLO()` in base al suffisso
del file, quindi un file `.onnx` si comporta come un checkpoint e restituisce
gli stessi `Results`.

<code-tabs name="export" />

L'esportazione edge usa un contratto di runtime a risoluzione fissa e batch 1:
`dynamic` e un `batch` diverso da 1 vengono rifiutati, e il grafo esportato
emette un'unica mappa di probabilità fusa. La copertura per formato è nelle
pagine [DexiNed](/docs/models/dexined) e [TEED](/docs/models/teed) e nella
[matrice completa di esportazione](/docs/reference/export-matrix).
[Esportazione](/docs/export) elenca gli argomenti che ogni formato accetta.
