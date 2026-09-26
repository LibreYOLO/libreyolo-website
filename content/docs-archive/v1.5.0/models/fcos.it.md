---
title: FCOS
families:
  - fcos
seo_title: 'FCOS in LibreYOLO: fai predizioni, valida ed esporta'
description: >-
  Esegui FCOS in LibreYOLO per il rilevamento di oggetti anchor-free. Installa,
  fai predizioni, valida ed esporta il port torchvision con licenza
  BSD-3-Clause, ResNet-50/FPN.
lead: >-
  FCOS rileva gli oggetti pixel per pixel invece di affidarsi a un insieme di
  anchor box predefiniti, predicendo un box e un punteggio di centerness in ogni
  posizione della feature map. LibreYOLO porta l'implementazione di torchvision
  per il rilevamento.
keywords:
  - FCOS
  - anchor-free detection
  - object detection python
  - rilevamento oggetti python
  - detector anchor-free
  - one-stage detector
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFCOSr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCOSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCOSr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="torchscript", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCOSr50.pt format=onnx imgsz=800
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory instrada in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreFCOSr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 60bd7b8dfd903a8c
---

## Installazione

FCOS non richiede nessun extra opzionale. Tutto quello che importa è
nell'installazione base.

```bash
pip install libreyolo
```

## Predizione

I pesi si scaricano da Hugging Face al primo utilizzo e restano in cache in
locale.

<code-tabs name="predict" />

L'oggetto `Results` restituito è lo stesso che restituisce ogni famiglia,
quindi passare a un rilevatore diverso è una modifica di una riga. Chiamare il
modello senza argomenti di soglia applica i valori predefiniti pubblicati da
FCOS stesso, `conf=0.2`, `iou=0.6` e `max_det=100`; passa uno qualsiasi dei tre
per sovrascriverli. FCOS mantiene un passaggio finale di NMS sulle sue
predizioni pixel per pixel. Vedi [predizione](/docs/predict) per sorgenti,
streaming e gestione dei risultati.

## Varianti

Una sola taglia: ResNet-50 con una feature pyramid, l'unica variante che questa
famiglia riconosce.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurate su qualsiasi dataset nel formato con cui
hai addestrato.

<code-tabs name="val" />

## Esportazione

<export-matrix />

FCOS esporta in ONNX, TorchScript e OpenVINO. FCOS preserva le proporzioni
della sorgente prima che il grafo venga eseguito, quindi LibreYOLO forza
`dynamic=True` per i percorsi ONNX e OpenVINO indipendentemente da quello che
viene passato, per mantenere il grafo valido con forme di input con padding. Un
file `.onnx` esportato si ricarica tramite `LibreYOLO()` in base al suffisso del
file e restituisce lo stesso `Results`.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenza

<provenance-box></provenance-box>

## Citazione

<citation-block />
