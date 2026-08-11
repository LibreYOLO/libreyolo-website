---
title: EfficientDet
families:
  - efficientdet
seo_title: 'EfficientDet: rilevamento di oggetti in LibreYOLO'
description: >-
  Esegui EfficientDet D0-D4 in LibreYOLO: rilevatori BiFPN per predizione,
  validazione ed esportazione in ONNX, TensorRT e OpenVINO con licenza
  Apache-2.0.
lead: >-
  EfficientDet abbina un backbone EfficientNet a una rete piramidale di feature
  bidirezionale ripetuta (BiFPN) e scala insieme profondità, larghezza e
  risoluzione su cinque dimensioni. LibreYOLO lo include come rilevatore solo
  per inferenza.
keywords:
  - EfficientDet
  - BiFPN
  - EfficientNet
  - object detection python
  - rilevamento oggetti python
  - compound scaling
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEfficientDetd0.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEfficientDetd0.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEfficientDetd0.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEfficientDetd0.pt format=onnx
        libreyolo export model=LibreEfficientDetd0.pt format=tensorrt half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO

        # La factory instrada in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreEfficientDetd0.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 12c61fb0035437ce
---

## Installazione

EfficientDet non richiede nessun extra opzionale. Tutto quello che importa è
nell'installazione base.

```bash
pip install libreyolo
```

## Predizione

I pesi si scaricano da Hugging Face al primo utilizzo e restano in cache in
locale.

<code-tabs name="predict" />

L'oggetto `Results` restituito è lo stesso che restituisce ogni famiglia,
quindi passare a un rilevatore diverso è una modifica di una riga. EfficientDet
decodifica candidati basati su anchor e poi esegue una non-maximum suppression
per classe, quindi qui `conf`, `iou` e `max_det` hanno tutti un effetto reale.
Vedi [predizione](/docs/predict) per sorgenti, streaming e gestione dei
risultati.

## Varianti

Cinque dimensioni, da D0 a D4. Ogni passo avanti abbina un backbone
EfficientNet più grande a una BiFPN più profonda e più larga e a una testa di
predizione più profonda, quindi il numero di parametri e il calcolo crescono
insieme, seguendo la regola di compound scaling del paper.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurate su qualsiasi dataset nel formato con cui
hai addestrato.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica con `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenza

<provenance-box>

I checkpoint D0-D4 di LibreYOLO sono convertiti attraverso il progetto
Apache-2.0 rwightman/efficientdet-pytorch, che a sua volta rispecchia i pesi
ufficiali addestrati in TensorFlow di google/automl senza modificare i tensori
appresi. Non è stato consultato né usato nessun sorgente del progetto
zylo117/Yet-Another-EfficientDet-Pytorch, distribuito con licenza LGPL.

</provenance-box>
