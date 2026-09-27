---
title: EfficientNetV2
families:
  - efficientnetv2
seo_title: 'EfficientNetV2: addestra, valida ed esporta con licenza Apache-2.0'
description: >-
  Usa EfficientNetV2 in LibreYOLO per la classificazione di immagini. Installa,
  fai predizioni, fine-tuning, validazione ed esportazione di
  LibreEfficientNetV2 da b0 a b3.
lead: >-
  EfficientNetV2 è un classificatore di immagini la cui profondità, larghezza e
  scelta dei blocchi stage per stage sono state trovate tramite neural
  architecture search, ottimizzando insieme accuratezza e velocità di
  addestramento invece della sola accuratezza. LibreYOLO lo supporta per un solo
  task: la classificazione.
keywords:
  - EfficientNetV2
  - EfficientNetV2-b0
  - image classification python
  - classificazione immagini python
  - neural architecture search
  - MBConv
  - classificatore ImageNet
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEfficientNetV2b0-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEfficientNetV2b0-cls.pt source=cat.jpg
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientNetV2b0-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreEfficientNetV2b0-cls.pt data=imagenette160
        epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreEfficientNetV2b0-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientNetV2b0-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEfficientNetV2b0-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientNetV2b0-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreEfficientNetV2b0-cls.pt format=onnx

        libreyolo export model=LibreEfficientNetV2b0-cls.pt format=tensorrt
        half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreEfficientNetV2b0-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: ad3ff140aad824bd
---

## Installazione

EfficientNetV2 non richiede nessun extra opzionale. Tutto ciò che importa è già
nell'installazione di base.

```bash
pip install libreyolo
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella cache
locale.

<code-tabs name="predict" />

L'oggetto `Results` restituito è lo stesso che restituisce ogni famiglia, quindi
passare a un altro modello è una modifica di una riga. Un classificatore non
porta con sé box né maschere: `result.probs` contiene la predizione sull'intera
immagine, con `top1`, `top5`, `top1conf` e `top5conf`. `conf`, `iou` e `max_det`
sono accettati per parità di API ma non hanno effetto, dato che su un singolo
vettore di probabilità non c'è nulla da sogliare o sopprimere. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Varianti

Quattro dimensioni, da b0 a b3, ognuna valutata alla propria risoluzione e con il
proprio rapporto di crop invece di condividere un'unica dimensione di input in
tutta la famiglia. Scegliere una dimensione è un semplice scambio tra numero di
parametri e accuratezza. Il task è fisso: ogni dimensione copre solo la
classificazione. Il nome del file dei pesi termina in `-cls.pt` per tutte le
dimensioni, ed è quel suffisso che la factory legge per smistare verso questa
famiglia; non serve nessun argomento `task=`.

## Addestramento

Il fine-tuning parte dal backbone ImageNet pubblicato e ricostruisce
automaticamente il layer finale del classificatore sul numero di classi del
dataset di destinazione. Se non lo imposti esplicitamente, `imgsz` assume come
valore predefinito la risoluzione di valutazione propria di quella dimensione.

<code-tabs name="train" />

Se non tocchi nulla, il trainer esegue 100 epoche a `lr0=1e-3` con AdamW, un
batch di 64 ed early stopping dopo 50 epoche senza miglioramenti. `data` accetta
la radice di un dataset (`train/` e `val/`, una cartella per classe), un nome
breve noto come `imagenette160`, oppure l'URL di un `.zip`. Qui `lora=True` non è
supportato; passarlo solleva un errore, dato che in LibreYOLO LoRA agisce sui
componenti transformer con layer `nn.Linear` e i blocchi MBConv di questa
famiglia non ne hanno.

Vedi [addestramento](/docs/train) per dataset, data augmentation, multi-GPU e
logger.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/`. Per la classificazione
sono l'accuratezza top-1 e top-5 sullo split di validazione.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica con `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. [Esportazione](/docs/export) elenca gli
argomenti accettati da ogni formato e gli extra che alcuni di essi aggiungono.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box></provenance-box>
