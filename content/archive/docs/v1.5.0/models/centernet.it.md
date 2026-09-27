---
title: CenterNet
families:
  - centernet
seo_title: 'CenterNet: rilevamento di oggetti in LibreYOLO'
description: >-
  Esegui CenterNet (Objects as Points) in LibreYOLO con i backbone ResDCN-18 e
  DLA-34. Fai predizioni, valida ed esporta in ONNX con licenza MIT. Nessun
  percorso di addestramento.
lead: >-
  CenterNet modella un oggetto come il punto centrale del suo bounding box e
  ricava per regressione ogni altra proprietà da un picco della heatmap, quindi
  non ha bisogno di anchor né di un passaggio di non-maximum-suppression.
  LibreYOLO lo include come rilevatore solo per inferenza.
keywords:
  - CenterNet
  - Objects as Points
  - object detection python
  - rilevamento oggetti python
  - detector anchor-free
  - keypoint detection
  - ResDCN-18
  - DLA-34
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreCenterNetresdcn18.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: DLA-34
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetdla34.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCenterNetresdcn18.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")

        # L'esportazione in ONNX richiede opset 16 o superiore: la fase di
        # upsampling a convoluzione deformabile viene tradotta in GridSample,
        # introdotto dall'opset 16.
        model.export(format="onnx", opset=18)
        model.export(format="tensorrt")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreCenterNetresdcn18.pt format=onnx opset=18
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO

        # La factory instrada in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreCenterNetresdcn18.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 20aaef83cc95590d
---

## Installazione

CenterNet non richiede nessun extra opzionale. Tutto quello che importa è
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
`max_det` filtrano i picchi della heatmap già ordinati; `iou` è accettato per
parità di API ma non ha effetto, perché il decode top-k dei picchi di CenterNet
non richiede nessun passaggio di soppressione per IoU dei box. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Varianti

Due backbone. `resdcn18` abbina un tronco ResNet-18 a un upsampling a
convoluzione deformabile; `dla34` abbina un tronco DLA-34 a un upsampling ad
aggregazione profonda iterativa. Entrambi alimentano le stesse tre teste dense
(heatmap, larghezza/altezza, offset) e lo stesso canvas di input.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurate su qualsiasi dataset nel formato con cui
hai addestrato.

<code-tabs name="val" />

## Esportazione

<export-matrix />

L'esportazione in ONNX richiede opset 16 o superiore: la fase di upsampling a
convoluzione deformabile presente in entrambi i backbone viene tradotta
nell'operatore ONNX `GridSample`, introdotto dall'opset 16. Se chiedi un opset
precedente viene sollevato un errore prima che inizi il tracing.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenza

<provenance-box>

Il grafo di ResDCN-18 accredita anche human-pose-estimation.pytorch di
Microsoft, distribuito con licenza MIT, e il grafo di DLA-34 accredita
l'implementazione DLA di Fisher Yu, con licenza BSD-3-Clause. LibreYOLO non
incorpora l'estensione DCNv2 originale usata dal progetto upstream;
l'esecuzione nativa usa invece `deform_conv2d` di torchvision, con licenza
BSD-3-Clause, e l'implementazione portabile, solo per l'esportazione, è stata
scritta a parte per LibreYOLO.

</provenance-box>

## Citazione

<citation-block />
