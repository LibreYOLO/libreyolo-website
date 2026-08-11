---
title: RetinaNet
families:
  - retinanet
seo_title: 'RetinaNet in LibreYOLO: predizione, validazione ed esportazione'
description: >-
  Esegui RetinaNet in LibreYOLO per il rilevamento di oggetti a stadio singolo
  con la focal loss. Installa, fai predizioni, valida ed esporta il porting da
  torchvision con licenza BSD-3-Clause.
lead: >-
  RetinaNet è un rilevatore a stadio singolo addestrato con la focal loss, che
  riduce il peso dei negativi facili: così una griglia densa di anchor non ha
  più bisogno di uno stadio separato di proposta per restare accurata. LibreYOLO
  porta l'implementazione di torchvision per il rilevamento.
keywords:
  - RetinaNet
  - focal loss
  - object detection python
  - rilevamento oggetti python
  - one-stage detector
  - torchvision RetinaNet
  - esportare RetinaNet onnx
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRetinaNetr50v2.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRetinaNetr50v2.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRetinaNetr50v2.pt format=onnx imgsz=800
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory instrada in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreRetinaNetr50v2.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 1cc7ceb6de290bdb
---

## Installazione

RetinaNet non richiede nessun extra opzionale. Tutto quello che importa è
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
`iou` impostano le soglie di confidenza e di NMS; RetinaNet mantiene il
passaggio di NMS upstream sulla griglia densa di anchor. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Varianti

Due dimensioni, entrambe ResNet-50 con una piramide di feature: `r50` è la
testa originale, mentre `r50v2` la sostituisce con una testa GroupNorm e un
blocco P6 più ampio, alimentato dall'ultimo stadio del backbone anziché
dall'output della FPN.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurate su qualsiasi dataset nel formato con cui
hai addestrato.

<code-tabs name="val" />

## Esportazione

<export-matrix />

RetinaNet esporta solo in ONNX, con dimensione del batch 1. RetinaNet
ridimensiona a un input variabile che preserva le proporzioni, quindi LibreYOLO
forza `dynamic=True` a prescindere da quello che passi, per mantenere il grafo
valido con sorgenti di forme diverse. Un file `.onnx` esportato si ricarica
tramite `LibreYOLO()` in base al suffisso del file e restituisce gli stessi
`Results`.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenza

<provenance-box></provenance-box>
