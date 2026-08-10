---
title: Deformable DETR
families: [deformable_detr]
seo_title: "Deformable DETR: predizione ed esportazione, Apache-2.0"
description: "Esegui Deformable DETR in LibreYOLO per il rilevamento di oggetti. Installa, fai predizioni, valida ed esporta cinque dimensioni ad attenzione sparsa, tutte con licenza Apache-2.0."
lead: "Deformable DETR sostituisce la cross-attention densa di DETR con un campionamento sparso e multiscala attorno a ogni punto di riferimento, ed è questo che ha reso i detector transformer addestrabili nella pratica. LibreYOLO distribuisce cinque dimensioni per il rilevamento di oggetti, solo in inferenza."
keywords: [Deformable DETR, detection transformer, sparse attention, object detection python, rilevamento oggetti python, SenseTime]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDeformableDETRr50.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")

        # val() restituisce un semplice dict, non un oggetto
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeformableDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDeformableDETRr50.pt format=onnx imgsz=800
        libreyolo export model=LibreDeformableDETRr50.pt format=tensorrt imgsz=800 half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreDeformableDETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Installazione

Deformable DETR non richiede nessun extra opzionale. Tutto ciò che importa è
già nell'installazione di base, con un core di attenzione deformabile
multiscala in PyTorch puro.

```bash
pip install libreyolo
```

Installare `libreyolo[hub-kernels]` è opzionale. Quando il pacchetto `kernels`
è presente, LibreYOLO scarica a runtime un kernel compilato di attenzione
deformabile multiscala dall'Hugging Face Hub e lo usa al posto del core in
PyTorch puro; `LIBREYOLO_HUB_KERNELS=0` lo disattiva di nuovo.

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella
cache locale.

<code-tabs name="predict" />

L'oggetto `Results` restituito è lo stesso che restituiscono tutte le famiglie,
quindi passare a un altro detector è una modifica di una riga. `conf` e
`max_det` filtrano la selezione delle query; `iou` è accettato per parità di
API ma non ha effetto, perché il decoder è un predittore di insiemi senza
passaggio di NMS. Vedi [predizione](/docs/predict) per sorgenti, streaming e
gestione dei risultati.

In LibreYOLO Deformable DETR è solo per l'inferenza. Upstream lo addestra con
matching ungherese e una focal loss di classificazione; quella ricetta non è
implementata qui, quindi `train()` solleva `NotImplementedError`.

## Varianti

Cinque checkpoint coprono le configurazioni rilasciate, tutte alla stessa
risoluzione di input. `r50ss` limita l'attenzione a una sola scala di feature;
`r50ssdc5` aggiunge sopra uno stadio C5 dilatato nel backbone. `r50` è la
configurazione multiscala predefinita, che campiona su quattro livelli di
feature map. `r50refine` aggiunge il raffinamento iterativo dei bounding box
lungo i layer del decoder, e `r50twostage` genera le proposte di regione
iniziali dall'output dell'encoder invece che da query apprese.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurate su qualsiasi dataset nel formato con cui
hai addestrato.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica con `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. [Esportazione](/docs/export) elenca gli
argomenti accettati da ogni formato.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
