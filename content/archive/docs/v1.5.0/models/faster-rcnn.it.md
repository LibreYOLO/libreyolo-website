---
title: Faster R-CNN
families:
  - faster_rcnn
seo_title: 'Faster R-CNN in LibreYOLO: predizione, validazione ed esportazione'
description: >-
  Esegui Faster R-CNN in LibreYOLO per il rilevamento di oggetti su quattro
  backbone. Installa, fai predizioni, valida ed esporta il port di torchvision
  con licenza BSD-3-Clause.
lead: >-
  Faster R-CNN rileva gli oggetti con una region proposal network che alimenta
  un classificatore a due stadi, l'architettura che ha reso le region proposal
  parte della stessa rete addestrata invece di un passaggio separato. LibreYOLO
  include il port dell'implementazione di torchvision per il rilevamento.
keywords:
  - Faster R-CNN
  - Faster R-CNN pytorch
  - object detection python
  - rilevamento oggetti python
  - region proposal network
  - rilevatore two-stage
  - torchvision detection
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFasterRCNNl.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFasterRCNNl.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFasterRCNNl.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFasterRCNNl.pt format=onnx imgsz=800
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory instrada in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreFasterRCNNl.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 3fd82eb835399560
---

## Installazione

Faster R-CNN non richiede nessun extra opzionale. Tutto quello che importa è
nell'installazione base.

```bash
pip install libreyolo
```

## Predizione

I pesi si scaricano da Hugging Face al primo utilizzo e restano in cache in
locale.

<code-tabs name="predict" />

L'oggetto `Results` restituito è lo stesso che restituisce ogni famiglia,
quindi passare a un rilevatore diverso è una modifica di una riga. `conf` e
`iou` impostano le soglie di confidenza e di NMS; Faster R-CNN mantiene il suo
passaggio di NMS upstream, a differenza di un rilevatore basato su query. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Varianti

Quattro dimensioni, ognuna una configurazione di torchvision diversa invece di
una versione scalata della stessa: `n` è MobileNetV3-Large con input a 320 px,
`s` è lo stesso backbone a 800 px, `m` è ResNet-50 con una feature pyramid e
`l` è la revisione v2, con una testa di region proposal più profonda e una
testa di box a quattro convoluzioni al posto di quella di `m`. `n` e `s`
scambiano accuratezza per un backbone più leggero.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurate su qualsiasi dataset nel formato con cui
hai addestrato.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Faster R-CNN esporta solo in ONNX, con dimensione del batch pari a 1. Il grafo
esportato mantiene al suo interno il passaggio di resize upstream, quindi
LibreYOLO forza `dynamic=True` indipendentemente da quello che viene passato,
per mantenere il grafo valido con sorgenti non quadrate. Un file `.onnx`
esportato si ricarica tramite `LibreYOLO()` in base al suffisso del file e
restituisce lo stesso `Results`.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenza

<provenance-box></provenance-box>

## Citazione

<citation-block />
