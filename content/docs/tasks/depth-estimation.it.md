---
title: Stima della profondità
seo_title: Stima monoculare della profondità in LibreYOLO
description: >-
  Predici una mappa densa di profondità relativa da una sola immagine in
  LibreYOLO. Confronta le famiglie depth, leggi le metriche di profondità ed
  esporta un modello depth.
lead: >-
  La stima della profondità predice quanto ogni pixel dista dalla fotocamera
  usando una sola immagine. LibreYOLO la espone come task depth, che restituisce
  una mappa densa di profondità inversa relativa sul canvas dell'immagine
  originale.
keywords:
  - monocular depth estimation python
  - mappa di profondità da una singola immagine
  - modello di profondità relativa
  - depth anything libreyolo
  - stima della profondità python
last_verified: 1.6.0
snippets:
  predict:
    - label: Predire una mappa di profondità
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.data.shape)              # (H, W) sul canvas originale
        print(depth.min, depth.max, depth.mean)
    - label: Lavorare con i valori
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")

        result = model(SAMPLE_IMAGE)


        depth = result.depth_map

        raw = depth.data          # più alto è più vicino; nessuna unità
        metrica, nessuna scala

        gray = depth.normalized() # riscalato a [0, 1] per la visualizzazione

        print(raw.shape, float(gray.max()))
    - label: Un'alternativa compatta
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Stesso contratto di task, con una rete molto più piccola pensata per i
        runtime edge.

        model = LibreYOLO("LibreZipDepthb-depth.pt")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
  val:
    - label: Validare e leggere le chiavi delle metriche
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])   # fitness
        print(metrics["metrics/delta2"], metrics["metrics/delta3"])
  export:
    - label: Esportare
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
    - label: Eseguire il file esportato
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory instrada in base al suffisso del file, quindi un artefatto

        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        Results.

        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
source_hash: f0afab6b9b451075
---

## Definizione

Il task `depth` predice un valore per pixel a partire da una singola immagine RGB.
LibreYOLO definisce quel valore come profondità inversa relativa: più alto significa
più vicino alla fotocamera, e i numeri non hanno unità metrica né una scala che
valga tra due immagini. Confrontare la profondità tra due pixel della stessa
predizione ha senso; confrontare un valore con quello di un'altra immagine no.

Una predizione riempie `result.depth_map`, un payload `DepthMap` che contiene un
array `(H, W)` sul canvas dell'immagine originale. `.min`, `.max` e `.mean` leggono
i valori finiti, e `.normalized()` riscala la mappa a `[0, 1]` per la
visualizzazione. `result.boxes` resta vuoto, quindi `conf`, `iou` e `max_det` non
hanno effetto, e `save=True` scrive un'immagine della mappa con una colormap invece
di una foto annotata.

## Modelli

Le seguenti famiglie supportano `depth`.

[Depth Anything V2](/docs/models/depth-anything-v2) abbina un encoder DINOv2 a un
decoder DPT ed è qui l'opzione predefinita per l'uso generico. La licenza decide la
taglia tanto quanto l'accuratezza: il checkpoint Small è Apache-2.0 mentre Base e
Large sono non commerciali, quindi controlla la tabella dei checkpoint sulla sua
pagina prima di sceglierne uno.

[Depth Anything 3](/docs/models/depth-anything-3) è il porting del checkpoint
DA3MONO-LARGE, un transformer semplice senza specializzazioni architetturali per la
profondità.

[ZipDepth](/docs/models/zipdepth) è la fascia compatta: una CNN riparametrizzabile
distillata da Depth Anything V2 Large, con un secondo checkpoint il cui decoder
evita le operazioni gather e unfold per i compilatori NPU che non le supportano.

[MiDaS](/docs/models/midas) è il filone di ricerca che ha stabilito il protocollo di profondità relativa zero-shot con cui vengono misurate le altre famiglie. I checkpoint s e l vengono scaricati dai mirror di LibreYOLO con la licenza MIT del distributore.

[LibreMODUS](/docs/models/libremodus) copre la profondità come uno dei target di
un modello any-to-any, non con una testa dedicata. Richiede l'extra `modus` e
un tuo account Hugging Face autenticato, e non offre né `val()` né `export()`.

[SenseNova-Vision](/docs/models/sensenova-vision) genera la mappa di profondità come
immagine tramite una decodifica per diffusione, dallo stesso checkpoint da 7B che
copre gli altri sei task. Richiede l'extra `sensenova`, e i suoi pesi sono limitati a un
uso non commerciale; la licenza è nella sua pagina.

[Marigold V2](/docs/models/marigold-v2) aggiunge adattatori di profondità basati sulla diffusione con codifiche esplicite della profondità.

## Predizione

I pesi vengono scaricati al primo utilizzo e memorizzati nella cache locale. Le pagine dei modelli descrivono i requisiti di autenticazione e runtime.

<code-tabs name="predict" />

La risoluzione di input ha vincoli specifici per famiglia. Depth Anything V2 e Depth Anything 3 si basano su una griglia di patch DINOv2, quindi `imgsz` deve essere divisibile per 14, condizione che LibreYOLO verifica prima dell'esecuzione. `Results.plot()` visualizza i risultati di profondità. Vedi [predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

`DepthMap.encoding` è `inverse_depth` di default e può essere `depth` o `log_depth`. La validazione interpreta la codifica prima dell'allineamento affine. La codifica non attribuisce una scala metrica alle predizioni relative.

## Formato del dataset

La validazione della profondità abbina a ogni immagine una mappa di profondità densa
a singolo canale della stessa risoluzione, individuata sostituendo la directory
delle profondità nel percorso dell'immagine.

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  depths/
    val/room.png
```

```yaml
path: dataset
val: images/val
depths_dir: depths
nc: 1
names: {0: depth}
```

Le mappe sono PNG o TIF a singolo canale, oppure `.npy`. I valori esprimono la
profondità in un'unità che il dataset mantiene coerente, e i pixel a `0`, negativi,
NaN e infiniti segnalano campioni non validi, esclusi dalle metriche. Le mappe intere
vengono divise per `depth_scale`, che di default vale `256.0`, la convenzione dei PNG
a 16 bit; le mappe `.npy` in float sono usate così come sono. `depth_stem_suffix` e
`depth_mask_suffix` coprono i dataset che nominano diversamente i loro file di
profondità o le maschere di validità. Vedi
[formati dei dataset](/docs/reference/dataset-formats) per il contratto completo.

## Addestramento

Nessuna famiglia di profondità in LibreYOLO implementa l'addestramento: `train()` genera `NotImplementedError` per queste famiglie. Ogni pagina del modello indica lo script di conversione che trasforma un checkpoint addestrato upstream in uno caricabile da LibreYOLO.

## Validazione

`val()` esegue il validatore di profondità condiviso. La profondità relativa non ha
una scala assoluta, quindi ogni predizione viene prima adattata all'inverso del suo
ground truth con una scala e uno shift ai minimi quadrati calcolati per immagine, poi
invertita di nuovo in profondità. Ogni metrica qui sotto è calcolata per immagine su
quella mappa allineata e mediata sul dataset, contando solo i pixel che il dataset
segna come validi.

<code-tabs name="val" />

`metrics/abs_rel` è l'errore relativo assoluto medio, il residuo diviso per la
profondità di ground truth, e valori più bassi sono migliori. `metrics/rmse` è la
radice dell'errore quadratico medio nell'unità di profondità del dataset stesso,
anche qui valori più bassi sono migliori. `metrics/delta1`, `metrics/delta2` e
`metrics/delta3` sono le accuratezze a soglia: la frazione di pixel validi il cui
rapporto con il ground truth, preso nella direzione in cui è maggiore, resta sotto
1.25, 1.25 al quadrato e 1.25 al cubo, quindi valori più alti sono migliori.
`metrics/delta1` è anche `fitness`, il numero su cui si basa la selezione del
checkpoint migliore.

## Esportazione

Un modello depth esportato si ricarica tramite `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e restituisce
lo stesso `Results`, con `depth_map` al posto dei box.

<code-tabs name="export" />

La copertura cambia da famiglia a famiglia, e Depth Anything 3 rifiuta qualsiasi
formato fuori dal suo insieme validato invece di tentare una conversione non
validata. Controlla la pagina del modello e la
[matrice completa delle esportazioni](/docs/reference/export-matrix) prima di
impegnarti su un target. LibreMODUS e SenseNova-Vision non esportano affatto.
[Esportazione](/docs/export) elenca gli argomenti che ogni formato accetta.
