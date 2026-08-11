---
title: Rilevamento di oggetti
seo_title: Rilevamento di oggetti in LibreYOLO
description: >-
  Rileva gli oggetti come box allineati agli assi in LibreYOLO: le famiglie che
  coprono il task, il formato delle etichette e le chiamate di predizione,
  addestramento, validazione ed esportazione.
lead: >-
  Il rilevamento di oggetti individua ogni istanza di oggetto in un'immagine e
  restituisce per ciascuna un rettangolo allineato agli assi, un'etichetta di
  classe e un punteggio. La chiave del task è detect.
keywords:
  - object detection python
  - rilevare oggetti in un'immagine
  - bounding box detection
  - libreria object detection MIT
  - alternativa a YOLO
  - addestrare un object detector
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9t.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 'Un''altra famiglia, stessa chiamata'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al checkpoint, e ogni rilevatore restituisce
        # lo stesso oggetto Results, quindi cambiare famiglia è una modifica di
        # una riga.
        model = LibreYOLO("LibreDFINEn.pt")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy.shape)
    - label: Video e stream
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Qualsiasi sorgente accettata dalla libreria: file, cartella, URL,
        # indice della webcam, stream RTSP o una lista .streams.
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # coco128.yaml scarica un campione di 128 immagini al primo utilizzo.
        # Punta data allo YAML del tuo dataset per un'esecuzione reale.
        model.train(data="coco128.yaml", epochs=50, imgsz=640, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 imgsz=640 batch=8
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() restituisce un semplice dict, non un oggetto.
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/AR100"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9t.pt data=coco128.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9t.pt format=onnx imgsz=640
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come un checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreYOLO9t.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: c735b6e3de78dd2b
---

## Definizione

Il rilevamento di oggetti dice dove si trova ogni oggetto e che cos'è.
Un'immagine in ingresso, una riga per istanza in uscita: quattro numeri per il
rettangolo, un indice di classe e un punteggio. Non c'è nulla sulla forma a
livello di pixel, sull'orientamento o sulle parti, ed è questo che lo separa
dalla [segmentazione di istanze](/docs/tasks/instance-segmentation), dai
[box orientati](/docs/tasks/oriented-detection) e dalla
[posa](/docs/tasks/pose-estimation).

`detect` è la chiave canonica del task ed è quella predefinita: un checkpoint il
cui nome di file non porta nessun suffisso di task si carica come rilevatore.

`predict()` riempie `result.boxes`. `.xyxy` dà gli angoli in pixel sul canvas
dell'immagine originale, `.conf` il punteggio e `.cls` l'indice di classe dentro
`result.names`. `.xywh`, `.xyxyn` e `.xywhn` sono viste derivate dalle stesse
righe, e `.id` porta un id di traccia una volta agganciato un tracker. Iterare
un oggetto `Boxes` produce slice di una sola riga, quindi `box.cls`, `box.conf` e
`box.xyxy` funzionano tutti per singolo rilevamento.

## Modelli

Dodici famiglie addestrano e predicono: [YOLOv9](/docs/models/yolov9),
[RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter),
[RT-DETR](/docs/models/rt-detr), [D-FINE](/docs/models/d-fine),
[DEIM](/docs/models/deim), [Dome-DETR](/docs/models/dome-detr),
[YOLO-NAS](/docs/models/yolo-nas),
[YOLOX](/docs/models/yolox), [YOLOv7](/docs/models/yolov7),
[RTMDet](/docs/models/rtmdet) e [PicoDet](/docs/models/picodet). YOLOv9 e
RF-DETR sono le due famiglie di punta, e le nuove funzionalità arrivano prima su
di loro. RF-DETR richiede il suo extra, `pip install "libreyolo[rfdetr]"`; le
altre funzionano con il pacchetto base.

Altre undici predicono, validano ed esportano, ma il loro `train()` solleva
`NotImplementedError`: [LW-DETR](/docs/models/lw-detr),
[DETR](/docs/models/detr), [Deformable DETR](/docs/models/deformable-detr),
[DINO-DETR](/docs/models/dino-detr), [Faster R-CNN](/docs/models/faster-rcnn),
[Mask R-CNN](/docs/models/mask-rcnn), [FCOS](/docs/models/fcos),
[RetinaNet](/docs/models/retinanet), [SSD](/docs/models/ssd),
[CenterNet](/docs/models/centernet) e
[EfficientDet](/docs/models/efficientdet).

La linea Darknet, [YOLOv1](/docs/models/yolov1),
[YOLOv2](/docs/models/yolov2), [YOLOv3](/docs/models/yolov3) e
[YOLOv4](/docs/models/yolov4), è conservata come un pezzo da museo congelato:
predizione, validazione ed esportazione funzionano, l'addestramento no.

Un gruppo a parte prende la sua lista di classi a runtime invece che dal
checkpoint, quindi rileva nomi mai visti durante l'addestramento:
[Grounding DINO](/docs/models/grounding-dino), [OWLv2](/docs/models/owlv2),
[OMDet-Turbo](/docs/models/omdet-turbo) e [OV-DEIM](/docs/models/ov-deim),
più le famiglie vision-language
[Florence-2](/docs/models/florence-2), [Kosmos-2](/docs/models/kosmos-2),
[Qwen3-VL](/docs/models/qwen3-vl), [SmolVLM2](/docs/models/smolvlm2),
[InternVL3](/docs/models/internvl3), [LFM2-VL](/docs/models/lfm2-vl),
[LocateAnything](/docs/models/locate-anything),
[SenseNova-Vision](/docs/models/sensenova-vision) e
[LibreMODUS](/docs/models/libremodus). Queste si caricano attraverso una factory
e degli extra propri; ogni pagina di modello riporta la chiamata esatta.

## Predizione

I pesi si scaricano da Hugging Face al primo utilizzo e restano in cache in
locale.

<code-tabs name="predict" />

`conf` imposta la soglia di confidenza e `max_det` limita il numero di righe.
`iou` è la soglia di NMS, quindi ha effetto solo su una famiglia che esegue NMS;
RF-DETR e la testa end-to-end di YOLOv9 decodificano un insieme fisso di
predizioni e la ignorano. Vedi [predizione](/docs/predict) per sorgenti,
streaming e gestione dei risultati.

## Formato del dataset

Un file di etichette `.txt` per immagine, individuato sostituendo `images` con
`labels` nel percorso dell'immagine e cambiando l'estensione.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

Ogni riga ha esattamente cinque campi, un indice di classe seguito da un box
normalizzato espresso come centro e dimensioni:

```text
<class_id> <cx> <cy> <w> <h>
```

Le coordinate sono float in `[0, 1]`, relative a larghezza e altezza
dell'immagine originale. `w` e `h` devono essere positivi. Un file di etichette
mancante o vuoto significa che l'immagine non ha oggetti. Le righe non portano
né la confidenza né un id di traccia.

Lo YAML indica gli split e le classi:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

`train` e `val` possono essere directory di immagini, file `.txt` con la lista
delle immagini, oppure liste dell'uno o dell'altro tipo. `nc` è facoltativo e,
quando c'è, deve corrispondere a `names`. Funziona anche il JSON COCO nativo:
aggiungi una mappatura `annotations` dal nome dello split al file JSON, e a quel
punto il percorso dello split indica la radice delle immagini. Quando `names` è
presente definisce gli id delle etichette, quindi i nomi delle categorie nel
JSON devono corrispondergli.

## Addestramento

<code-tabs name="train" />

`epochs`, `imgsz`, `batch` e `lr0` sono gli argomenti che si toccano per primi.
`lr0` è quello che non si trasferisce da una famiglia all'altra: un valore che
un rilevatore convoluzionale tollera fa divergere un rilevatore transformer,
quindi prendilo dalla pagina del modello e non dall'esempio di un'altra
famiglia. Una famiglia può anche ignorare del tutto un argomento, e la sua
pagina elenca quali. Vedi [addestramento](/docs/train) per dataset, data augmentation,
multi-GPU e logger.

## Validazione

`val()` restituisce un semplice dizionario di chiavi `metrics/`, calcolate con
la valutazione COCO sullo split indicato da `val` nello YAML del dataset.

<code-tabs name="val" />

`metrics/mAP50-95` è la mean average precision mediata sulle soglie di IoU da
0.50 a 0.95, ed è il numero di riferimento. `metrics/mAP50` e `metrics/mAP75`
sono le versioni a soglia singola. `metrics/mAP_small`, `metrics/mAP_medium` e
`metrics/mAP_large` suddividono la stessa media per area dell'oggetto, e
`metrics/AR1`, `metrics/AR10`, `metrics/AR100`, `metrics/AR_small`,
`metrics/AR_medium` e `metrics/AR_large` sono i valori di average recall
corrispondenti. `metrics/AR_max_det` e `metrics/max_det` registrano il limite di
rilevamenti usato dall'esecuzione.

Leggi `metrics/precision` e `metrics/recall` con attenzione su questo task. Sono
mantenute per retrocompatibilità e sono alias, non un punto operativo:
`metrics/precision` contiene lo stesso valore di `metrics/mAP50-95`, e
`metrics/recall` lo stesso valore di `metrics/AR100`. Rappresentarle come una
coppia precisione-recall riporta lo stesso numero due volte. Quattro chiavi si
ripetono anche con un suffisso `(B)`, per box, così che una chiave di
rilevamento si legga allo stesso modo su un modello che predice anche le
maschere: `metrics/mAP50-95(B)`, `metrics/mAP50(B)`,
`metrics/precision(B)` e `metrics/recall(B)`.

## Esportazione

<code-tabs name="export" />

Un artefatto esportato si ricarica attraverso `LibreYOLO()` in base al suffisso
del file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. La copertura dei formati cambia da famiglia a
famiglia; la matrice su ogni pagina di modello è generata dall'insieme validato
invece che scritta a mano. Vedi
[esportazione e deployment](/docs/export) per i formati, i loro extra e i loro
vincoli.
