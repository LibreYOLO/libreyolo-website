---
title: EdgeCrafter
families:
  - ec
seo_title: 'EdgeCrafter: rilevamento, posa e segmentazione in LibreYOLO'
description: >-
  Usa EdgeCrafter in LibreYOLO per il rilevamento, la stima della posa e la
  segmentazione di istanze. Installa, fai predizioni, valida ed esporta, con
  codice sotto licenza MIT.
lead: >-
  Un vision transformer compatto per la predizione densa su hardware edge,
  pubblicato dall'upstream come tre modelli fratelli: ECDet, ECPose ed ECSeg.
  LibreYOLO li carica tutti e tre come un'unica famiglia, con il task portato
  dal checkpoint.
keywords:
  - EdgeCrafter
  - ECDet
  - ECPose
  - ECSeg
  - vision transformer compatto
  - object detection
  - pose estimation python
  - segmentazione di istanze python
  - inferenza edge
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreECs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Posa
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Il suffisso -pose nel nome del file seleziona la testa dei keypoint,
        # quindi qui non serve l'argomento task.
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.conf)
    - label: Segmentazione di istanze
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
            batch=8,
            lr0=5e-4,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreECs.pt data=my-dataset.yaml epochs=50
        imgsz=640 batch=8 lr0=5e-4
    - label: Posa
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Richiede un dataset di keypoint a classe singola il cui data.yaml
        # dichiari kpt_shape, e imgsz alla dimensione nativa del checkpoint.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="my-pose-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: Segmentazione di istanze
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Richiede etichette poligonali, e imgsz alla dimensione nativa del
        checkpoint.

        model = LibreYOLO("LibreECs-seg.pt")

        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            lora=True,
        )
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs.pt data=my-dataset.yaml
    - label: Posa
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        metrics = model.val(data="my-pose-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: Segmentazione di istanze
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # maschere
        print(metrics["metrics/mAP50-95(B)"])   # box
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-seg.pt format=onnx imgsz=640
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreECs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 39c6975fc16b3ff1
---

## Installazione

EdgeCrafter non richiede nessun extra opzionale. Tutto ciò che importa è già
nell'installazione di base.

```bash
pip install libreyolo
```

Il fine-tuning con adattatori tramite `lora=True` è l'eccezione, e richiede
l'extra `lora`.

```bash
pip install "libreyolo[lora]"
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella cache
locale.

<code-tabs name="predict" />

Il task viene dal nome del file, quindi un checkpoint `-pose` o `-seg` seleziona
da solo la propria testa e non richiede nessun argomento task. Tutti e tre
restituiscono l'oggetto `Results` che restituisce ogni famiglia, con
`result.keypoints` in più per la posa e `result.masks` per la segmentazione. La
posa copre una sola classe, person, con i 17 keypoint COCO, e il conteggio è
fissato quando il modello viene costruito. Non ha una testa per i box, quindi
ogni box della posa è l'estensione che racchiude i suoi stessi keypoint, e il
terzo canale del keypoint è una costante invece che un punteggio per punto.

`conf` e `max_det` filtrano la selezione delle query; `iou` è accettato per
parità di API ma non ha effetto, perché tutte e tre le teste decodificano un
insieme di query senza passaggio di NMS. Vedi [predizione](/docs/predict) per
sorgenti, streaming e gestione dei risultati.

## Varianti

Quattro dimensioni. Girano tutte alla stessa risoluzione di input, quindi la
tabella le separa per numero di parametri e accuratezza.

<benchmark-table task="detect" />

<va-embed />

L'upstream pubblica ECDet, ECPose ed ECSeg come tre modelli separati invece che
un solo modello con tre teste. Condividono il backbone ECViT e l'encoder ibrido
e differiscono solo nella testa, quindi LibreYOLO li riunisce in un'unica
famiglia e lascia che sia il nome del file del checkpoint a portare il task. Una
lettera di dimensione significa quindi lo stesso backbone e lo stesso encoder
per tutti e tre, e predizione, validazione ed esportazione accettano gli stessi
argomenti qualunque sia quello che carichi.

## Addestramento

Tutti e tre i task si addestrano tramite `train()`, che legge il task dal
checkpoint caricato e sceglie il trainer corrispondente.

<code-tabs name="train" />

Cosa è stato verificato per il rilevamento e la segmentazione: la parità di
inferenza con l'upstream a 1e-5, layer per layer e per ogni dimensione, e che la
loss e un singolo passo di addestramento girino su input sintetici. Cosa no,
stando alla docstring di `train()` stessa: la convergenza di un fine-tuning
completo, l'addestramento multi-GPU, il passaggio di ricarica del migliore allo
stop della data augmentation, e il rimappaggio delle classi da Objects365 a
COCO. La via della posa segue la ricetta pubblicata di DETRPose, un matcher
ungherese su costi di classe, L1 sui keypoint e OKS con denoising contrastivo
dei keypoint, e nemmeno la sua convergenza è stata verificata end to end.

Se non tocchi niente, il trainer esegue 74 epoche con `lr0=5e-4` e la precisione
mista attiva, seguendo la ricetta dell'upstream: AdamW, una schedule cosine
piatta, EMA a 0.9999 e input normalizzati su ImageNet. Posa e segmentazione
richiedono entrambe `imgsz` alla dimensione nativa del checkpoint, perché la
loro griglia di anchor per la valutazione viene costruita quando il modello
viene costruito; un valore diverso solleva un errore prima che l'esecuzione
parta. La posa richiede inoltre un dataset a classe singola il cui `data.yaml`
dichiari `kpt_shape`, con un numero di keypoint che corrisponde alla testa.

`lora=True` si applica solo al rilevamento; posa e segmentazione sollevano un
`ValueError` se lo ricevono. Su Apple silicon il trainer tiene l'esecuzione
sulla GPU e manda una sola operazione su CPU, il backward del grid-sample dentro
l'attenzione deformabile, che PyTorch non implementa in Metal.

Vedi [addestramento](/docs/train) per dataset, data augmentation, multi-GPU e
logger.

## Validazione

`val()` restituisce un dizionario indicizzato per nome della metrica, e stampa i
risultati per classe se `verbose` resta attivo.

<code-tabs name="val" />

La posa riporta le metriche OKS sui keypoint sotto `metrics/keypoints_*`. La
segmentazione riporta le maschere sotto la chiave semplice `metrics/mAP50-95` e
ripete entrambe le viste in una sola passata, i box sotto `(B)` e le maschere
sotto `(M)`.

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica con `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. Posa e segmentazione esportano con un input
fisso di 640 per 640 invece che con forme dinamiche, e anche diversi target di
rilevamento sono a canvas fisso, tra cui OpenVINO, Paddle, MNN, ExecuTorch e
Core AI. [Esportazione](/docs/export) elenca gli argomenti che ogni formato
accetta e gli extra che qualcuno di essi aggiunge.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
