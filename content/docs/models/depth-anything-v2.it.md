---
title: Depth Anything V2
families: [depth_anything]
seo_title: "Depth Anything V2: predire e validare la profondità monoculare"
description: "Usa Depth Anything V2 in LibreYOLO per la stima della profondità monoculare. Installa, fai predizioni e valida; Small è Apache-2.0, Base e Large sono CC-BY-NC-4.0."
lead: "Depth Anything V2 è un encoder DINOv2 abbinato a un decoder DPT che predice una mappa densa di profondità inversa relativa a partire da una sola immagine. LibreYOLO lo supporta per il task depth: predizione e validazione zero-shot, senza percorso di addestramento."
keywords: [Depth Anything V2, "monocular depth estimation", DPT, DINOv2, "stima della profondità da una singola immagine", "depth map python", "mappa di profondità"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDepthAnythingV2s-depth.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Leggere la mappa di profondità
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE)

        depth = result.depth_map    # DepthMap: densa (H, W), più alto = più vicino
        raw = depth.data                # tensore, senza unità metrica né scala fra immagini
        normalized = depth.normalized() # riscalata in [0, 1] per la visualizzazione
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnythingV2s-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=onnx
        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=tensorrt half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory sceglie in base al suffisso del file, quindi un artefatto esportato
        # si carica come un qualsiasi checkpoint e restituisce lo stesso oggetto Results.
        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
---

## Installazione

Depth Anything V2 non richiede nessun extra opzionale. Tutto ciò che importa è già nell'installazione base.

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
disco una visualizzazione della mappa con una colormap; `Results.plot()` non
copre questa famiglia, perché è definito solo per le normali di superficie e i
contorni. La risoluzione di input deve essere divisibile per 14, la griglia di
patch DINOv2 su cui si appoggia la testa DPT; LibreYOLO lo verifica prima di
eseguire e solleva un errore se non lo è. Vedi [predizione](/docs/predict) per
sorgenti, streaming e gestione dei risultati.

## Varianti

Quattro dimensioni di encoder, s/b/l/g, corrispondenti a ViT-S/B/L/G. La
tabella dei checkpoint qui sotto elenca solo s, b e l; nessun checkpoint Giant
è stato pubblicato. Tutte e quattro condividono la stessa risoluzione di input,
quindi scegliere una dimensione significa scambiare capacità dell'encoder, non
dimensione dell'immagine. Anche la licenza conta: il checkpoint Small è
Apache-2.0, mentre Base e Large sono CC-BY-NC-4.0, vedi Licenze più sotto.

L'addestramento e il fine-tuning non sono offerti per questa famiglia.
`LibreDepthAnythingV2.train()` solleva sempre `NotImplementedError`; converti
invece un checkpoint upstream compatibile, con
`weights/convert_depth_anything_v2_weights.py`.

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
[Esportazione](/docs/export) elenca gli argomenti che ogni formato accetta.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
