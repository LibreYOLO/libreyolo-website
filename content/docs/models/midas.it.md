---
title: MiDaS
families:
  - midas
seo_title: 'MiDaS: stima della profondità monoculare in LibreYOLO'
description: >-
  Usa MiDaS in LibreYOLO per la stima della profondità monoculare. Installa, fai
  predizioni, valida ed esporta due varianti con licenza MIT, scaricate da
  isl-org.
lead: >-
  MiDaS è la stima della profondità relativa monoculare addestrata con una loss
  invariante a scala e shift su dataset misti, la linea di lavoro che ha
  definito il protocollo di trasferimento zero-shot della profondità che le
  famiglie successive riutilizzano. LibreYOLO lo supporta per il task depth:
  predizione e validazione zero-shot, senza percorso di addestramento.
keywords:
  - MiDaS
  - monocular depth estimation
  - DPT
  - stima della profondità monoculare
  - mappa di profondità relativa
  - depth map python
  - zero-shot depth
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Non ancora su disco: LibreYOLO lo scarica dalla release GitHub
        # ufficiale isl-org/MiDaS e lo verifica contro uno SHA-256 fissato.
        model = LibreYOLO("LibreMiDaSl-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMiDaSl-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Variante Small
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Encoder EfficientNet-Lite3, più piccolo e veloce della dimensione l
        DPT-Large.

        model = LibreYOLO("LibreMiDaSs-depth.pt")

        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMiDaSl-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMiDaSl-depth.pt format=onnx
        libreyolo export model=LibreMiDaSl-depth.pt format=tensorrt half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory sceglie in base al suffisso del file, quindi un artefatto
        # esportato si carica come un qualsiasi checkpoint e restituisce lo
        # stesso oggetto Results.
        model = LibreYOLO("LibreMiDaSl-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: ce2fbf3ae43e9be4
---

## Installazione

MiDaS non richiede nessun extra opzionale. Tutto ciò che importa è già nell'installazione base.

```bash
pip install libreyolo
```

## Predizione

MiDaS è l'unica famiglia depth che LibreYOLO non ripubblica sulla propria
organizzazione Hugging Face. Richiedere un checkpoint con il suo nome file
LibreYOLO scarica l'asset ufficiale corrispondente direttamente dalle release
GitHub di `isl-org/MiDaS`, lo verifica contro uno SHA-256 fissato e lo avvolge
nei metadati di checkpoint di LibreYOLO prima del primo utilizzo; le esecuzioni
successive riusano il file locale in cache. Vedi Licenze per il motivo.

<code-tabs name="predict" />

`result.depth_map` contiene una mappa densa di profondità inversa relativa:
valori più alti indicano una maggiore vicinanza alla camera, e i valori non
hanno unità metrica né scala confrontabile fra immagini. `save=True` scrive su
disco una visualizzazione della mappa con una colormap; `Results.plot()` non
copre questa famiglia, perché è definito solo per le normali di superficie e i
contorni. Vedi [predizione](/docs/predict) per sorgenti, streaming e gestione
dei risultati.

## Varianti

Due varianti con encoder diversi, non semplicemente scale diverse dello stesso
encoder. `s` è MiDaS v2.1 Small, un encoder EfficientNet-Lite3. `l` è
DPT-Large, un encoder ViT-L/16 con il decoder DPT che MiDaS ha introdotto per
la predizione densa. Anche il preprocessing è diverso: `s` usa un
ridimensionamento a proporzioni mantenute con vincolo superiore e
normalizzazione mean/std di ImageNet, `l` usa un ridimensionamento minimo a
proporzioni mantenute con media e deviazione standard di 0.5. Scegli `s` per
una CNN più leggera, `l` per l'accuratezza del decoder transformer.

L'addestramento non è offerto per questa famiglia. `LibreMiDaS.train()` solleva
sempre `NotImplementedError`.

## Validazione

`val()` esegue il validatore depth condiviso: allinea ogni predizione al suo
ground truth con una scala e uno shift ai minimi quadrati calcolati per
immagine, poi riporta le metriche standard di profondità relativa zero-shot,
AbsRel, RMSE e le tre soglie delta.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica con `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`, con `depth_map` al posto dei box.

<code-tabs name="export" />

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
