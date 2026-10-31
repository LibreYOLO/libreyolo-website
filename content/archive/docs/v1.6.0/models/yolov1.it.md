---
title: YOLOv1
families:
  - yolo1
seo_title: 'YOLOv1 in LibreYOLO: predizione, validazione, esportazione'
description: >-
  Esegui il rilevatore YOLOv1 originale in LibreYOLO: una famiglia da museo,
  congelata e solo per inferenza. Fai predizioni, valida ed esporta, con licenza
  di pubblico dominio.
lead: >-
  YOLOv1 è il rilevatore originale del 2016 che ha dato il nome alla famiglia
  YOLO: una sola rete convoluzionale con una testa fully connected predice ogni
  box e ogni punteggio di classe in un unico passaggio, senza anchor box.
  LibreYOLO lo conserva come pezzo da museo, congelato e solo per inferenza.
keywords:
  - YOLOv1
  - YOLO v1
  - Darknet
  - object detection
  - rilevamento oggetti python
  - Pascal VOC
  - modelli YOLO storici
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO1b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO1b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO1b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO1b.pt format=onnx
        libreyolo export model=LibreYOLO1b.pt format=tensorrt half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory instrada in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreYOLO1b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: a786372dba86f2f8
---

## Installazione

YOLOv1 non richiede nessun extra oltre al pacchetto base.

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
quindi passare a un rilevatore diverso è una modifica di una riga. Due cose
sono specifiche di questa famiglia. Il checkpoint pubblicato è addestrato su
Pascal VOC (2007+2012), non su COCO, quindi `box.cls` indicizza le 20 categorie
VOC (aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train,
tvmonitor) invece delle 80 di COCO. E la testa di rilevamento fully connected
accetta una sola immagine per volta, quindi una lista di sorgenti viene
percorsa in ciclo anziché eseguita come un vero batch. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurate su un dataset nello stesso spazio di
etichette in stile VOC con cui è stato addestrato il checkpoint.

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
