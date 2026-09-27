---
title: LW-DETR
families:
  - lwdetr
seo_title: 'LW-DETR: fai predizioni ed esporta con licenza Apache-2.0'
description: >-
  Usa LW-DETR in LibreYOLO per il rilevamento di oggetti in tempo reale.
  Installa, fai predizioni, valida ed esporta cinque dimensioni basate su ViT,
  tutte con licenza Apache-2.0.
lead: >-
  Un detection transformer a ViT puro che Baidu ha posizionato come alternativa
  in tempo reale ai rilevatori YOLO. LibreYOLO ne include cinque dimensioni per
  il rilevamento, solo in inferenza.
keywords:
  - LW-DETR
  - detection transformer
  - real-time object detection
  - rilevamento oggetti in tempo reale
  - plain ViT
  - DETR
  - rilevamento oggetti python
  - Baidu
  - Atten4Vis
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLWDETRt.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLWDETRt.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")

        # val() restituisce un dict semplice, non un oggetto
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLWDETRt.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreLWDETRt.pt format=onnx imgsz=640

        libreyolo export model=LibreLWDETRt.pt format=tensorrt imgsz=640
        half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreLWDETRt.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: badd1d8255df5bbd
---

## Installazione

LW-DETR non richiede nessun extra opzionale. Tutto ciò che importa è già
nell'installazione di base.

```bash
pip install libreyolo
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella cache
locale.

<code-tabs name="predict" />

L'oggetto `Results` restituito è quello che restituisce ogni famiglia, quindi
passare a un rilevatore diverso è una modifica di una riga. `conf` e `max_det`
filtrano la selezione delle query; `iou` è accettato per parità di API ma non ha
effetto, perché il decoder è un predittore di insiemi senza passaggio di NMS.
Vedi [predizione](/docs/predict) per sorgenti, streaming e gestione dei
risultati.

In LibreYOLO, LW-DETR è solo per inferenza. L'upstream lo addestra con la
supervisione uno-a-molti di Group-DETR su più gruppi di query e una loss di
classificazione consapevole dell'IoU; quella ricetta non è collegata qui, quindi
`train()` solleva `NotImplementedError`.

## Varianti

Cinque dimensioni, che condividono tutte l'encoder a ViT puro, il proiettore
multi-scala e il decoder di deformable DETR, e girano tutte alla stessa
risoluzione di input. Le due più piccole condividono la larghezza dell'encoder e
si distinguono per la profondità in blocchi; le due successive condividono un
encoder più largo e si distinguono per quanti livelli del proiettore alimentano
il decoder; la più grande sale all'encoder più largo di tutti.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurate su qualsiasi dataset nel formato in cui hai
addestrato.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica con `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. [Esportazione](/docs/export) elenca gli
argomenti che ogni formato accetta.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
