---
title: ZipDepth
families:
  - zipdepth
seo_title: 'ZipDepth: profondità monoculare leggera in LibreYOLO'
description: >-
  Usa ZipDepth in LibreYOLO per la stima leggera della profondità monoculare.
  Installa, fai predizioni, valida ed esporta due checkpoint con licenza MIT.
lead: >-
  ZipDepth è una CNN compatta e riparametrizzabile distillata da Depth Anything
  V2 Large che predice una mappa densa di profondità inversa relativa. LibreYOLO
  lo supporta per il task depth: predizione e validazione zero-shot, senza
  percorso di addestramento.
keywords:
  - ZipDepth
  - monocular depth estimation
  - stima della profondità monoculare
  - modello depth per edge
  - profondità relativa
  - depth map python
  - CNN riparametrizzabile
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreZipDepthb-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Checkpoint NPU/edge
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Stesso encoder, con una testa di upsampling senza unfold per i
        # compilatori che non supportano gather/unfold. L'output è visivamente
        # equivalente a quello del checkpoint b.
        model = LibreYOLO("LibreZipDepthbnpu-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreZipDepthb-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        model.export(format="onnx")
        model.export(format="ncnn")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreZipDepthb-depth.pt format=onnx
        libreyolo export model=LibreZipDepthbnpu-depth.pt format=ncnn
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory sceglie in base al suffisso del file, quindi un artefatto
        # esportato si carica come un qualsiasi checkpoint e restituisce lo
        # stesso oggetto Results.
        model = LibreYOLO("LibreZipDepthb-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: 891eaa1a42795a4c
---

## Installazione

ZipDepth non richiede nessun extra opzionale. Tutto ciò che importa è già
nell'installazione base.

```bash
pip install libreyolo
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano in cache
in locale.

<code-tabs name="predict" />

`result.depth_map` contiene una mappa densa di profondità inversa relativa:
valori più alti indicano una maggiore vicinanza alla camera, e i valori non
hanno unità metrica né scala confrontabile fra immagini. `save=True` scrive su
disco una visualizzazione di quella mappa con una colormap; `Results.plot()`
non copre questa famiglia, perché è definito solo per le normali di superficie
e i contorni. Vedi [predizione](/docs/predict) per sorgenti, streaming e
gestione dei risultati.

## Varianti

Due checkpoint, entrambi con la stessa capacità di encoder, che differiscono
solo nella testa di upsampling addestrata. `b` usa l'upsampling convesso e gira
su GPU o CPU. `bnpu` sostituisce quella testa con un decoder senza unfold per
NPU e compilatori edge che non supportano gather/unfold; il suo output è
documentato come visivamente equivalente a quello di `b`. Scegli `bnpu` quando
il target dell'esportazione è un runtime limitato, `b` negli altri casi.

Entrambi i checkpoint sono stati distillati da pseudo-etichette di Depth
Anything V2 Large, quindi questa famiglia è il livello compatto e orientato
all'edge del task depth di LibreYOLO, accanto agli encoder più grandi di Depth
Anything V2.

L'addestramento non è offerto per questa famiglia. `LibreZipDepth.train()`
solleva sempre `NotImplementedError`: la ricetta upstream distilla
pseudo-etichette su un grande insieme di immagini che non è riproducibile come
un addestramento LibreYOLO. Addestra upstream su
[fabiotosi92/ZipDepth](https://github.com/fabiotosi92/ZipDepth) e converti il
risultato con `weights/convert_zipdepth_weights.py`.

## Validazione

`val()` esegue il validatore depth condiviso: allinea ogni predizione al suo
ground truth con una scala e uno shift ai minimi quadrati calcolati per
immagine, poi riporta le metriche standard di profondità relativa zero-shot,
AbsRel, RMSE e le tre soglie delta.

<code-tabs name="val" />

## Esportazione

<export-matrix />

L'esportazione segue un contratto denso a risoluzione fissa: l'immagine di
origine viene ridimensionata per stiramento sul canvas esportato, e la mappa di
profondità restituita viene poi riportata al canvas originale. Un artefatto
esportato si ricarica con `LibreYOLO()` in base al suffisso del file, quindi un
file `.onnx` o `.ncnn` si comporta come un checkpoint e restituisce lo stesso
`Results`, con `depth_map` al posto dei box.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
