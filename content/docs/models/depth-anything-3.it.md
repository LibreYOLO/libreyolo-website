---
title: Depth Anything 3
families: [depth_anything3]
seo_title: "Depth Anything 3: predice la profondità monoculare in LibreYOLO"
description: "Usa Depth Anything 3 in LibreYOLO per la stima della profondità monoculare. Installa, fai predizioni, valida ed esporta il checkpoint DA3MONO-LARGE, Apache-2.0."
lead: "Depth Anything 3 è un semplice transformer DINOv2 addestrato a predire la profondità e la geometria della fotocamera da una o più viste, senza alcuna specializzazione architetturale. LibreYOLO porta il suo checkpoint DA3MONO-LARGE per il task di profondità: predizione e validazione zero-shot, senza percorso di addestramento."
keywords: [Depth Anything 3, DA3, "monocular depth estimation", DINOv2, "stima della profondità da immagine", "mappa di profondità python", "profondità relativa"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDepthAnything3l-depth.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Leggere la mappa di profondità
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        result = model(SAMPLE_IMAGE)

        depth = result.depth_map    # DepthMap: densa (H, W), più alto = più vicino
        raw = depth.data                # tensor, senza unità metrica né scala tra immagini
        normalized = depth.normalized() # riscalato a [0, 1] per la visualizzazione
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnything3l-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDepthAnything3l-depth.pt format=onnx
        libreyolo export model=LibreDepthAnything3l-depth.pt format=tensorrt half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory instrada in base al suffisso del file, quindi un artefatto esportato
        # si carica come qualsiasi checkpoint e restituisce lo stesso oggetto Results.
        model = LibreYOLO("LibreDepthAnything3l-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
---

## Installazione

Depth Anything 3 non richiede alcun extra opzionale. Tutti i moduli che importa
sono già nell'installazione base.

```bash
pip install libreyolo
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella cache
locale.

<code-tabs name="predict" />

`result.depth_map` contiene una mappa densa di profondità inversa relativa: i
valori più alti indicano una maggiore vicinanza alla fotocamera, e i valori non
hanno unità metrica né scala comune tra immagini. Il checkpoint originale
produce profondità relativa positiva; il wrapper di rete di LibreYOLO la inverte
e riproduce la gestione ufficiale del cielo, così l'output rispetta il contratto
di profondità condiviso di LibreYOLO. `save=True` scrive su disco una
visualizzazione di quella mappa con mappa di colori; `Results.plot()` non copre
questa famiglia, perché è definito solo per le normali di superficie e i bordi.
Vedi [predizione](/docs/predict) per sorgenti, streaming e gestione dei
risultati.

## Varianti

Una sola taglia, `l`, a risoluzione di input fissa. Il progetto originale di DA3
pubblica anche checkpoint any-view Small e Base, un checkpoint di profondità
metrica e checkpoint Nested e Giant; LibreYOLO non ne espone nessuno. La
profondità metrica richiede un contratto pubblico diverso da quello del task di
profondità inversa relativa di LibreYOLO, e i checkpoint any-view e Nested
richiedono un'API multi-immagine per la fotocamera che LibreYOLO non offre. I
checkpoint any-view Large e Giant sono inoltre CC-BY-NC-4.0 e nessun percorso di
download di LibreYOLO li referenzia.

Per questa famiglia non è previsto l'addestramento.
`LibreDepthAnything3.train()` solleva `NotImplementedError` in modo
incondizionato; addestra con il progetto originale e converti un checkpoint
DA3MONO-LARGE compatibile con `weights/convert_depth_anything3_weights.py`.

## Validazione

`val()` esegue il validatore di profondità condiviso: allinea ogni predizione al
suo ground truth con una scala e uno spostamento ai minimi quadrati calcolati
per immagine, poi riporta le metriche standard di profondità relativa zero-shot:
AbsRel, RMSE e le tre soglie delta.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Per questa famiglia l'esportazione è limitata a cinque formati: ONNX,
TorchScript, ExecuTorch, TensorRT e OpenVINO. Richiedere qualsiasi altro formato
solleva `NotImplementedError` invece di tentare una conversione non validata. Un
artefatto esportato si ricarica tramite `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`, con `depth_map` al posto dei box.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenza

<provenance-box></provenance-box>

## Citazione

<citation-block />
