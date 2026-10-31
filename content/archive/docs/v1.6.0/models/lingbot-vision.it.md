---
title: LingBot-Vision
families:
  - lingbotvision
seo_title: 'LingBot-Vision: la segmentazione semantica in LibreYOLO'
description: >-
  Usa LingBot-Vision in LibreYOLO per la segmentazione semantica su un backbone
  ViT Apache-2.0. Installa, fai predizioni, addestra, valida ed esporta, nelle
  taglie s/b/l.
lead: >-
  LingBot-Vision è una famiglia di backbone vision transformer
  autosupervisionati, addestrati con un masked modeling centrato sui contorni
  per la percezione spaziale densa, rilasciata da Robbyant. LibreYOLO abbina il
  backbone a una testa densa e lo supporta per un solo task: la segmentazione
  semantica.
keywords:
  - LingBot-Vision
  - semantic segmentation
  - segmentazione semantica python
  - vision transformer
  - self-supervised pretraining
  - preaddestramento autosupervisionato
  - Robbyant
  - dense prediction
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLingBotVisions-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python (linear probe)
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Backbone congelato per impostazione predefinita, come nel protocollo
        # di valutazione upstream: si addestra solo la testa densa 1x1.
        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(data="my-dataset.yaml", epochs=20, imgsz=512, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 imgsz=512 batch=16
    - label: Fine-tuning completo
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(
            data="my-dataset.yaml", epochs=20, imgsz=512, batch=16,
            freeze_backbone=False,
        )
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLingBotVisions-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="coreai", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreLingBotVisions-sem.pt format=onnx imgsz=512
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreLingBotVisions-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: c47b33fdc6fa1139
---

## Installazione

LingBot-Vision non richiede nessun extra opzionale. Tutto ciò che importa è
già nell'installazione di base.

```bash
pip install libreyolo
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella
cache locale.

<code-tabs name="predict" />

`result.semantic_mask` porta la mappa densa delle classi: `.data` è un tensore
`(H, W)` di id di classe alla dimensione originale dell'immagine, e `.classes`
elenca gli id di classe effettivamente presenti. `result.boxes` è `None`,
perché non ci sono rilevamenti per singola istanza. `conf` e `iou` sono
accettati per parità di API ma non cambiano l'output, dato che il modello
restituisce una classe per pixel invece di rilevamenti da filtrare. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Varianti

Tre taglie pubblicate, s, b e l, distillate da un teacher ViT-g/16 da 1,1
miliardi di parametri. Il teacher stesso, taglia `g`, si carica e si affina in
LibreYOLO, ma LibreYOLO non ospita un checkpoint `g` proprio.

<checkpoint-table />

## Addestramento

`train()` fa fine-tuning di un checkpoint pubblicato. La ricetta predefinita è
il linear probe del report upstream: il backbone ViT è congelato e si addestra
solo la testa densa 1x1, esattamente come sono stati prodotti i pesi ospitati
da LibreYOLO qui sopra. Passa `freeze_backbone=False` per fare invece
fine-tuning dell'intera rete, e aspettati di dover abbassare `lr0` di
conseguenza.

<code-tabs name="train" />

Vedi [addestramento](/docs/train) per dataset, data augmentation, multi-GPU e
logger.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/`: mIoU e accuratezza per
pixel, misurate su qualsiasi dataset nel formato con cui hai addestrato.

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

<provenance-box>

La release upstream documenta il proprio ViT come costruito sull'architettura
DINOv2/DINOv3 pubblicata da Meta AI. Robbyant distribuisce la propria
implementazione con licenza Apache-2.0, e questo port per LibreYOLO è stato
realizzato solo a partire dal repository di Robbyant, mai dal codice DINOv2 o
DINOv3 di Meta.

</provenance-box>

## Citazione

<citation-block />
