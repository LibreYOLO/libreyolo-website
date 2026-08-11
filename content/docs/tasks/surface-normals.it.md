---
title: Normali di superficie
seo_title: Stima delle normali di superficie in LibreYOLO
description: >-
  Predici un campo denso di normali di superficie da una sola immagine in
  LibreYOLO. Leggi la convenzione del sistema di riferimento della fotocamera,
  valida l'errore angolare ed esporta un modello.
lead: >-
  La stima delle normali di superficie predice la direzione verso cui è
  orientata ogni superficie visibile. LibreYOLO la espone come task normal, che
  restituisce un campo denso di vettori unitari sul canvas dell'immagine
  originale.
keywords:
  - surface normal estimation python
  - normal map da immagine
  - geometria monocolare
  - metrica errore angolare
  - predizione densa di normali
last_verified: 1.5.0
snippets:
  predict:
    - label: Predire un campo di normali
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreMoGe2s-normal.pt")

        result = model(SAMPLE_IMAGE, save=True)


        normals = result.normal_map

        print(normals.data.shape)      # (H, W, 3) vettori unitari float32

        normals.assert_normalized()    # solleva un errore se un pixel non ha
        lunghezza unitaria
    - label: Leggere un singolo pixel
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreMoGe2s-normal.pt")

        result = model(SAMPLE_IMAGE)


        # Riferimento fotocamera OpenCV: +x a destra, +y in basso, +z verso la
        scena.

        # Una superficie rivolta verso la fotocamera si legge vicino a (0, 0,
        -1).

        field = result.normals.data

        h, w = field.shape[:2]

        print(field[h // 2, w // 2])
    - label: Salvare la visualizzazione
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE)

        # plot() disegna il campo; è definito per i risultati normal ed edge.
        result.plot().save("normals.png")
  val:
    - label: Validare e leggere le chiavi delle metriche
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])     # gradi
        print(metrics["metrics/median_angular_error"])   # gradi
        print(metrics["metrics/within_11_25"])           # percentuale di pixel
        print(metrics["metrics/within_22_5"], metrics["metrics/within_30"])
  export:
    - label: Esportare
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
    - label: Eseguire il file esportato
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory instrada in base al suffisso del file, quindi un artefatto

        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        Results.

        model = LibreYOLO("LibreMoGe2s-normal.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.normal_map.data.shape)
source_hash: d26d26d894b436ff
---

## Definizione

Il task `normal` predice un vettore unitario a tre componenti per pixel a partire da
una singola immagine RGB: la direzione verso cui è orientata la superficie in quel
pixel. A differenza della profondità, l'output non ha una scala libera, quindi due
predizioni sono direttamente confrontabili senza allineamento.

Una predizione riempie `result.normal_map`, un payload `NormalMap` che contiene un
array float32 `(H, W, 3)` sul canvas dell'immagine originale, raggiungibile anche
come `result.normals`. I vettori usano il sistema di riferimento della fotocamera
OpenCV di LibreYOLO, con `+x` a destra, `+y` in basso e `+z` verso la scena, e sono
rivolti verso la fotocamera, quindi una superficie fronto-parallela si legge come
`(0, 0, -1)`. `.assert_normalized()` controlla che ogni pixel sia finito e di
lunghezza unitaria entro una tolleranza. `result.boxes` resta vuoto, quindi `conf`,
`iou` e `max_det` non hanno effetto, e `Results.plot()` copre questo task.

## Modelli

Due famiglie servono `normal`.

[MoGe-2](/docs/models/moge-2) è quella dedicata: un modello di geometria monocolare a
singolo forward in tre taglie di encoder. LibreYOLO non copia questi checkpoint nella
propria organizzazione; caricarne uno scarica la taglia corrispondente dai repository
ufficiali a una revisione fissata e la verifica contro uno SHA-256 registrato.

[LibreMODUS](/docs/models/libremodus) produce le normali come uno dei target di un
modello any-to-any, e può prendere in input una mappa di profondità invece di
un'immagine RGB. Richiede l'extra `modus` e un tuo account Hugging Face autenticato, e
non offre né `val()` né `export()`, quindi non partecipa alle sezioni di validazione
ed esportazione qui sotto.

## Predizione

I pesi di MoGe-2 si scaricano al primo uso e restano in cache in locale.

<code-tabs name="predict" />

`imgsz` deve essere divisibile per la dimensione delle patch dell'encoder ViT, cosa
che LibreYOLO controlla prima che l'esecuzione inizi. Predire una lista di immagini
esegue un forward pass per immagine; questo task non ha un percorso rapido a batch
impilati. Vedi [predizione](/docs/predict) per sorgenti, streaming e gestione dei
risultati.

## Formato del dataset

La validazione delle normali accoppia ogni immagine con un PNG a 16 bit e tre canali
con lo stesso nome base e la stessa risoluzione, più una maschera di validità
opzionale.

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  normals/
    val/room.png
  masks/
    val/room.png
```

```yaml
path: dataset
train: images/train
val: images/val
normals_dir: normals
masks_dir: masks
nc: 1
names: {0: normal}
```

Il PNG di target è esattamente `uint16` a tre canali, con i canali memorizzati come
RGB. La decodifica è `n = png / 65535 * 2 - 1` seguita dalla rinormalizzazione di ogni
vettore, e i vettori decodificati usano lo stesso sistema di riferimento della
fotocamera OpenCV delle predizioni. Un pixel della maschera conta come valido quando è
diverso da zero; senza un file di maschera, ogni vettore decodificato finito e diverso
da zero è valido. I pixel di target non validi e quelli di padding sono tenuti
internamente a `(0, 0, 0)` e non contribuiscono mai a una metrica. Vedi
[formati dei dataset](/docs/reference/dataset-formats) per il contratto completo.

## Addestramento

Nessuna delle due famiglie normal ha un'implementazione dell'addestramento: `train()`
solleva `NotImplementedError` su entrambe. La pagina di MoGe-2 indica i suoi
checkpoint ufficiali fissati per predizione, validazione ed esportazione.

## Validazione

`val()` misura l'angolo tra ogni vettore predetto e il suo vettore di ground truth,
sui pixel che il dataset segna come validi.

<code-tabs name="val" />

`metrics/mean_angular_error` e `metrics/median_angular_error` sono quell'angolo in
gradi, e più basso è meglio. `metrics/within_11_25`, `metrics/within_22_5` e
`metrics/within_30` sono la percentuale di pixel validi il cui errore angolare sta
entro 11.25, 22.5 e 30 gradi, quindi più alto è meglio. Attenzione all'unità: quei tre
valori sono percentuali, non frazioni. `fitness` è `metrics/within_11_25` diviso 100,
il che porta la selezione del checkpoint migliore sulla stessa scala `[0, 1]` di ogni
altro task.

## Esportazione

Un modello normal esportato si ricarica tramite `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` si comporta come un checkpoint e restituisce lo stesso
`Results`.

<code-tabs name="export" />

L'esportazione delle normali usa un contratto di runtime a risoluzione fissa e batch 1:
`dynamic` e un `batch` diverso da 1 vengono rifiutati, e `imgsz` deve essere divisibile
per la dimensione delle patch dell'encoder. La copertura per formato è nella
[pagina di MoGe-2](/docs/models/moge-2) e nella
[matrice completa delle esportazioni](/docs/reference/export-matrix).
[Esportazione](/docs/export) elenca gli argomenti che ogni formato accetta.
