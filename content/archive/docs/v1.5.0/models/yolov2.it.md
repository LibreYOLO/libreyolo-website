---
title: YOLOv2
families:
  - yolo2
seo_title: 'YOLOv2 in LibreYOLO: predizione, validazione, esportazione'
description: >-
  Esegui YOLOv2 (YOLO9000) in LibreYOLO: una famiglia da museo, congelata e solo
  per inferenza. Fai predizioni, valida ed esporta, con licenza di pubblico
  dominio.
lead: >-
  YOLOv2, pubblicato anche come YOLO9000, è il rilevatore Darknet-19 che ha
  introdotto le anchor box e un layer passthrough nella linea YOLO. LibreYOLO lo
  conserva come pezzo da museo, congelato e solo per inferenza.
keywords:
  - YOLOv2
  - YOLO9000
  - Darknet
  - Darknet-19
  - object detection
  - anchor boxes
  - rilevamento oggetti python
  - esportare yolov2 onnx
  - modelli YOLO storici
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO2b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO2b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO2b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO2b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO2b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO2b.pt format=onnx
        libreyolo export model=LibreYOLO2b.pt format=tensorrt half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory instrada in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreYOLO2b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: ba2884a2f6e1b0da
---

## Installazione

YOLOv2 non richiede nessun extra oltre al pacchetto base.

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
quindi passare a un rilevatore diverso è una modifica di una riga. `conf` filtra
la soglia di confidenza e `iou` la soglia di NMS, applicate alle predizioni
basate su anchor della testa `region`. Vedi [predizione](/docs/predict) per
sorgenti, streaming e gestione dei risultati.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurate su qualsiasi dataset nel formato con cui
validi.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica tramite `LibreYOLO()` in base al suffisso
del file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. Anche eseguire il grafo in un runtime nudo,
senza LibreYOLO installato, è supportato, ma in quel caso il preprocessing e il
postprocessing li devi scrivere tu.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenza

<provenance-box></provenance-box>
