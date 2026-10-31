---
title: Mask R-CNN
families:
  - mask_rcnn
seo_title: 'Mask R-CNN in LibreYOLO: predizione, validazione ed esportazione'
description: >-
  Esegui Mask R-CNN in LibreYOLO per il rilevamento di oggetti e la
  segmentazione di istanze. Installa, fai predizioni, valida ed esporta il port
  di torchvision con licenza BSD-3-Clause.
lead: >-
  Mask R-CNN aggiunge a Faster R-CNN un ramo di maschere per regione, che
  predice una maschera di segmentazione insieme a ogni box che rileva. LibreYOLO
  include il port dell'implementazione di torchvision per il rilevamento e la
  segmentazione di istanze.
keywords:
  - Mask R-CNN
  - Mask R-CNN pytorch
  - instance segmentation
  - segmentazione di istanze
  - object detection python
  - Faster R-CNN
  - torchvision
  - rilevatore two-stage
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMaskRCNNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Solo box
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # task="detect" salta la testa delle maschere e restituisce i box dallo
        # stesso checkpoint, senza maschere nel risultato.
        model = LibreYOLO("LibreMaskRCNNr50.pt", task="detect")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])      # maschere
        print(metrics["metrics/mAP50-95(B)"])   # box
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMaskRCNNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMaskRCNNr50.pt format=onnx imgsz=800
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory instrada in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreMaskRCNNr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
source_hash: 9608459b801aa6d5
---

## Installazione

Mask R-CNN non richiede nessun extra opzionale. Tutto quello che importa è
nell'installazione base.

```bash
pip install libreyolo
```

## Predizione

I pesi si scaricano da Hugging Face al primo utilizzo e restano in cache in
locale.

<code-tabs name="predict" />

L'oggetto `Results` restituito è lo stesso che restituisce ogni famiglia,
quindi passare a un rilevatore diverso è una modifica di una riga. Caricare il
checkpoint senza l'argomento `task` restituisce le maschere di istanza, dato
che la segmentazione è il task predefinito di questa famiglia; `result.masks`
le porta quindi insieme ai box. Passare `task="detect"` carica gli stessi pesi
senza la testa delle maschere e restituisce solo i box. `conf` e `iou`
impostano le soglie di confidenza e di NMS; Mask R-CNN mantiene il suo
passaggio di NMS upstream, a differenza di un rilevatore basato su query. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Varianti

Un solo backbone: ResNet-50 con una feature pyramid, usando il builder v2 di
Mask R-CNN di torchvision. Il checkpoint pubblicato ha una licenza
BSD-3-Clause e serve entrambi i task di questa famiglia, quindi non c'è nessuna
dimensione tra cui scegliere.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/`. Con il task di
segmentazione predefinito di questo checkpoint, la chiave `metrics/mAP50-95`
semplice contiene il punteggio delle maschere, e la stessa esecuzione riporta i
box sotto il suffisso `(B)`, quindi entrambi sono disponibili con un solo
passaggio.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Mask R-CNN esporta solo in ONNX, con dimensione del batch pari a 1. Il grafo
esportato mantiene al suo interno i passaggi di resize e di incollaggio delle
maschere upstream, quindi LibreYOLO forza `dynamic=True` indipendentemente da
quello che viene passato, per mantenere il grafo valido con sorgenti non
quadrate. Un file `.onnx` esportato si ricarica tramite `LibreYOLO()` in base al
suffisso del file e restituisce lo stesso `Results`.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia. L'unico checkpoint qui
sotto è elencato sotto detect, ma lo stesso file si carica anche per la
segmentazione: non passare l'argomento `task` e restituirà le maschere per
impostazione predefinita.

<checkpoint-table />

## Licenza

<provenance-box>

Mask R-CNN è costruito come sottoclasse del wrapper di Faster R-CNN di
LibreYOLO: condivide la stessa sorgente di torchvision e la stessa licenza
BSD-3-Clause, e aggiunge il predittore di maschere e la testa RoI delle
maschere dello stesso commit portato.

</provenance-box>
