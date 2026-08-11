---
title: YOLOv3
families:
  - yolo3
seo_title: 'YOLOv3 in LibreYOLO: predizione, validazione, esportazione'
description: >-
  Esegui YOLOv3 in LibreYOLO: una famiglia da museo congelata e solo per
  inferenza, con le dimensioni tiny, base e SPP. Fai predizioni, valida ed
  esporta, con una licenza di pubblico dominio.
lead: >-
  YOLOv3 è il rilevatore Darknet-53 che ha portato nella linea YOLO la
  predizione multi-scala e i classificatori logistici indipendenti. LibreYOLO lo
  conserva come pezzo da museo congelato e solo per inferenza, nelle dimensioni
  tiny, base e SPP.
keywords:
  - YOLOv3
  - Darknet
  - Darknet-53
  - object detection
  - rilevamento oggetti python
  - yolov3 python
  - yolov3 pesi preaddestrati
  - darknet yolov3
  - modelli yolo storici
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO3b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO3b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Dimensione SPP
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La variante SPP aggiunge un blocco di spatial pyramid pooling prima
        # delle teste di rilevamento e usa la sua dimensione di input nativa.
        model = LibreYOLO("LibreYOLO3spp.pt")
        result = model(SAMPLE_IMAGE)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO3b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO3b.pt format=onnx
        libreyolo export model=LibreYOLO3b.pt format=tensorrt half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory instrada in base al suffisso del file, quindi un artefatto
        # esportato si carica come un checkpoint qualsiasi e restituisce lo
        # stesso oggetto Results.
        model = LibreYOLO("LibreYOLO3b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: a4c652bb2707fc8f
---

## Installazione

YOLOv3 non richiede nessun extra oltre al pacchetto base.

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
filtra sulla soglia di confidenza e `iou` sulla soglia di NMS, applicate a ogni
scala prima che i box delle tre teste vengano uniti. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurate su qualsiasi dataset nel formato con cui
esegui la validazione.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica con `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. È supportata anche l'esecuzione del grafo in un
runtime nudo, senza LibreYOLO installato, ma in quel caso il preprocessing e il
postprocessing li scrivi tu.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box></provenance-box>
