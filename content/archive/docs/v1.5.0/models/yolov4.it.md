---
title: YOLOv4
families:
  - yolo4
seo_title: 'YOLOv4: esegui, valida ed esporta in LibreYOLO'
description: >-
  Esegui YOLOv4 in LibreYOLO: una famiglia da museo congelata e solo per
  inferenza, con backbone CSPDarknet-53. Fai predizioni, valida ed esporta, con
  licenza di pubblico dominio.
lead: >-
  YOLOv4 combina un backbone CSPDarknet-53, un blocco SPP e un neck PANet con
  attivazioni Mish. LibreYOLO lo include come pezzo da museo congelato e solo
  per inferenza, nelle taglie tiny e base.
keywords:
  - YOLOv4
  - Darknet
  - CSPDarknet-53
  - PANet
  - object detection python
  - rilevamento oggetti python
  - Mish activation
  - famiglia da museo
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO4b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO4b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO4b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO4b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO4b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO4b.pt format=onnx
        libreyolo export model=LibreYOLO4b.pt format=tensorrt half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory instrada in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreYOLO4b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 6070bb4a09d75416
---

## Installazione

YOLOv4 non richiede nessun extra oltre al pacchetto base.

```bash
pip install libreyolo
```

## Predizione

Questa famiglia è solo per inferenza: `train()` solleva `NotImplementedError`,
quindi questa pagina non ha una sezione di addestramento. Predizione,
validazione ed esportazione sono tutte supportate. I pesi si scaricano da
Hugging Face al primo utilizzo e restano in cache in locale.

<code-tabs name="predict" />

L'oggetto `Results` restituito è lo stesso che restituisce ogni famiglia,
quindi passare a un rilevatore diverso è una modifica di una riga. `conf`
filtra la soglia di confidenza e `iou` la soglia di NMS, applicate dopo lo
scaling dei centri `scale_x_y` proprio di ogni testa. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurate su qualsiasi dataset nel formato con cui
validi.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica tramite `LibreYOLO()` in base al suffisso
del file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. È supportata anche l'esecuzione del grafo in
un runtime spoglio, senza LibreYOLO installato, ma in quel caso il
preprocessing e il postprocessing tocca a te scriverli.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenza

<provenance-box></provenance-box>

## Citazione

<citation-block />
