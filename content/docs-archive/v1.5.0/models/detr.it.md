---
title: DETR
families:
  - detr
seo_title: 'DETR: predizione ed esportazione con licenza Apache-2.0'
description: >-
  Esegui DETR, il transformer di rilevamento originale, in LibreYOLO. Installa,
  fai predizioni, valida ed esporta quattro dimensioni basate su ResNet, tutte
  con licenza Apache-2.0.
lead: >-
  DETR è il transformer di rilevamento originale: predice un insieme fisso di
  oggetti con un decoder transformer accoppiato dall'algoritmo ungherese, invece
  di usare anchor o una griglia densa. LibreYOLO include quattro dimensioni per
  il rilevamento, solo per inferenza.
keywords:
  - DETR
  - detection transformer
  - object detection python
  - rilevamento oggetti python
  - Hungarian matching
  - decoder transformer
  - DETR pytorch
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")

        # val() restituisce un dict semplice, non un oggetto
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory instrada in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreDETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: c5549a596742d2a5
---

## Installazione

DETR non richiede nessun extra opzionale. Tutto quello che importa è
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
`max_det` filtrano la selezione delle query; `iou` è accettato per parità di
API ma non ha effetto, perché il decoder predice un insieme completo e non ha
nessun passaggio di NMS. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

In LibreYOLO, DETR è solo per inferenza. Il progetto upstream addestra per 500
epoche con matching ungherese; quella ricetta non è implementata qui, quindi
`train()` solleva `NotImplementedError`.

## Varianti

Quattro checkpoint combinano due profondità di backbone, ResNet-50 o
ResNet-101, con uno stadio C5 dilatato opzionale: le varianti DC5 mantengono
l'ultimo stadio del backbone a piena risoluzione invece di ridurla
ulteriormente, quindi il decoder legge una feature map più fine a partire dalla
stessa dimensione di input. Tutte e quattro condividono 100 object query
apprese e un encoder-decoder transformer a sei layer, e girano tutte alla
stessa risoluzione di input.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurate su qualsiasi dataset nel formato con cui hai addestrato.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica tramite `LibreYOLO()` in base al suffisso
del file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. [Esportazione](/docs/export) elenca gli
argomenti che ogni formato accetta.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenza

<provenance-box></provenance-box>
