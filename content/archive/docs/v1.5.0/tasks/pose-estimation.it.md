---
title: Stima della posa
seo_title: Stima della posa in LibreYOLO
description: >-
  Predici i keypoint per ogni istanza in LibreYOLO: le famiglie che coprono il
  task, il formato delle etichette e le chiamate di predizione, addestramento,
  validazione ed esportazione.
lead: >-
  La stima della posa localizza ogni istanza e ne restituisce un insieme
  ordinato di keypoint con nome, così l'output porta con sé la struttura interna
  dell'oggetto e non solo la sua estensione. La chiave del task è pose.
keywords:
  - pose estimation python
  - keypoint detection
  - stima della posa umana
  - COCO keypoints
  - OKS mAP
  - addestrare modello pose
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Il suffisso -pose nel nome del file seleziona la testa dei keypoint,
        # quindi non serve l'argomento task.
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy.shape)   # (N, K, 2) coordinate in pixel
        print(result.boxes.xyxy.shape)     # (N, 4), le stesse N istanze
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreECs-pose.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Solo i keypoint visibili
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreECs-pose.pt")(SAMPLE_IMAGE)
        kpts = result.keypoints

        # .has_visible deriva dalla terza colonna dei keypoint ed è tutta True
        # quando il checkpoint predice solo (x, y).
        for person, visible in zip(kpts.xy, kpts.has_visible):
            print(person[visible])
    - label: 'In alternativa, top-down'
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # HRNet è top-down: ritaglia prima ogni persona. Se non gli dai una
        sorgente

        # di persone, si abbina a un rilevatore LibreYOLO9t e registra la
        scelta.

        model = LibreYOLO("LibreHRNetw32-pose.pt")

        result = model(SAMPLE_IMAGE)


        print(result.keypoints.xy.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # coco8-pose.yaml contiene uno script di download incorporato, quindi
        serve

        # un permesso esplicito a meno che i dati non siano già in locale.

        model = LibreYOLO("LibreECs-pose.pt")

        model.train(
            data="coco8-pose.yaml",
            epochs=50,
            imgsz=640,
            batch=4,
            allow_download_scripts=True,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreECs-pose.pt data=coco8-pose.yaml \
          epochs=50 imgsz=640 batch=4 allow_download_scripts=True
    - label: Il tuo dataset
      language: python
      code: |
        from libreyolo import LibreYOLO

        # data.yaml deve dichiarare kpt_shape, e le righe delle etichette devono
        # avere esattamente 5 + K * D campi.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(data="my-pose-dataset.yaml", epochs=50, imgsz=640, batch=8)
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreECs-pose.pt")


        # val() restituisce un semplice dict, non un oggetto.

        metrics = model.val(data="coco8-pose.yaml", allow_download_scripts=True)


        print(metrics["metrics/keypoints_mAP50-95"])

        print(metrics["metrics/keypoints_mAP50"],
        metrics["metrics/keypoints_mAP75"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs-pose.pt data=coco8-pose.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
    - label: Usare il file esportato
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory instrada in base al suffisso del file, quindi un artefatto

        # esportato si carica come un checkpoint e restituisce lo stesso
        Results.

        model = LibreYOLO("LibreECs-pose.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.keypoints.xy)
source_hash: 9de01d1f615bdf33
---

## Definizione

La stima della posa restituisce struttura, non solo estensione. Ogni istanza
riceve comunque un box, una classe e un punteggio, e riceve anche `K` keypoint
in un ordine fisso, così l'indice 5 indica la stessa parte del corpo su ogni
istanza e in ogni immagine. L'insieme delle etichette definisce quell'ordine;
niente nell'output identifica un keypoint per nome.

`pose` è la chiave canonica del task, e il suffisso `-pose` nel nome del file di
un checkpoint la seleziona, quindi `task=` non serve quando carichi pesi
pubblicati.

`predict()` riempie `result.keypoints` accanto a `result.boxes`. `.data`
è `(N, K, 2)` o `(N, K, 3)`, allineato per riga ai box, quindi l'istanza `i` in
uno è l'istanza `i` nell'altro. `.xy` estrae le coordinate in pixel e `.xyn` le
normalizza rispetto alla dimensione dell'immagine originale. `.conf` è la terza
colonna quando il checkpoint la predice ed è `None` quando non lo fa, e
`.has_visible` è la maschera booleana che ne deriva, tutta `True` quando non
c'è una terza colonna.

Due architetture arrivano a questo output. Un modello one-stage predice box e
keypoint in un solo passaggio. Un modello top-down esegue prima un rilevatore,
ritaglia ogni istanza e ne stima i keypoint per regressione dentro il ritaglio, quindi la sua
accuratezza dipende dal rilevatore che ha davanti.

## Modelli

Tre famiglie coprono sia l'addestramento sia la predizione:
[RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter) e
[YOLO-NAS](/docs/models/yolo-nas), tutte one-stage. RF-DETR richiede il suo
extra, `pip install "libreyolo[rfdetr]"`. RF-DETR ed EdgeCrafter pubblicano
checkpoint pose e fanno entrambe fine-tuning su dataset a classe singola, con
sole persone; la testa dei keypoint di EdgeCrafter è fissata alla costruzione e
rifiuta un dataset che dichiara un conteggio diverso, mentre RF-DETR
reinizializza la propria testa per adattarla. YOLO-NAS prende i pesi dal CDN di
Deci.AI con una licenza non commerciale, e LibreYOLO non ne pubblica nessuno;
anche la sua testa pose si ricostruisce per un nuovo numero di keypoint, ed è
l'unica delle tre il cui numero di classi non è fissato a uno, quindi è la
famiglia da usare per uno scheletro multi-classe o non umano, come la posa
degli animali.

[HRNet](/docs/models/hrnet) è l'opzione top-down. Predice, valida ed esporta, e
il suo `train()` solleva `NotImplementedError`. Se non gli dai una sorgente di
persone, si abbina automaticamente a un rilevatore LibreYOLO9t; `cropped=True`
tratta l'intera immagine come un'unica istanza, `person_boxes=` accetta box che
hai già e `person_detector=` indica un rilevatore diverso.

Anche [SenseNova-Vision](/docs/models/sensenova-vision) emette keypoint. È un
modello generativo guidato da prompt, con una factory propria, `LibreVLM`, e un
extra proprio; senza un vocabolario impostato, `set_task("pose")` ricade sulla
categoria persona. I suoi pesi sono non commerciali, e la latenza per immagine è
molto più alta di quella di una testa pose dedicata, perché ogni predizione è una
decodifica per diffusione.

## Predizione

I pesi si scaricano da Hugging Face al primo uso e restano in cache in locale.

<code-tabs name="predict" />

Il numero e l'ordine dei keypoint sono proprietà del checkpoint, non della
libreria, quindi un modello addestrato su uno scheletro diverso restituisce un
`K` diverso e un significato diverso per ogni indice. Anche il contenuto della
terza colonna dei keypoint è una proprietà del checkpoint: EdgeCrafter ci scrive
una costante invece di un punteggio per punto, e non ha affatto una testa per i
box, quindi ognuno dei suoi box pose è l'estensione che racchiude i keypoint di
quella stessa istanza. Vedi [predizione](/docs/predict) per sorgenti, streaming
e gestione dei risultati.

## Formato del dataset

Il layout è quello del rilevamento: un file di etichette `.txt` per immagine,
trovato scambiando `images` con `labels` nel percorso dell'immagine e cambiando
l'estensione.

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

Una riga è una riga di rilevamento con i keypoint in coda:

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

Il numero di campi è esattamente `5 + K * D`, dove `D` è il secondo valore di
`kpt_shape`. Le coordinate dei box e dei keypoint sono float normalizzati
rispetto a larghezza e altezza dell'immagine originale. La visibilità `v`,
presente solo quando `D` vale 3, è `0`, `1` o `2`.

Lo YAML aggiunge due chiavi al contratto condiviso:

```yaml
path: dataset
train: images/train
val: images/val
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
names:
  0: person
```

`kpt_shape` è obbligatorio ed è `[K, 2]` o `[K, 3]`. `flip_idx` è facoltativo ed
è una permutazione di `0..K-1` che dà, per ogni keypoint, l'indice che assume
dopo un ribaltamento orizzontale: è così che un polso sinistro resta un polso
sinistro. Se lo ometti, l'augmentation di ribaltamento orizzontale viene
disattivata per i keypoint invece di essere applicata con l'ordine degli indici
sbagliato.

## Addestramento

<code-tabs name="train" />

L'addestramento riparte da un checkpoint `-pose` pubblicato, che porta già la
testa dei keypoint; il task si legge dal checkpoint che carichi, non da un flag
passato al momento dell'addestramento, quindi un checkpoint di rilevamento non
diventa una run pose solo perché la chiedi. Con EdgeCrafter il `kpt_shape` del
tuo YAML deve corrispondere esattamente alla testa, dato che la sua testa è
fissata alla costruzione, mentre RF-DETR e YOLO-NAS ridimensionano invece la
testa per un conteggio diverso. Vedi
[addestramento](/docs/train) per dataset, augmentation, multi-GPU e logger.

## Validazione

`val()` restituisce un semplice dizionario di chiavi `metrics/`. Il punteggio si
calcola con la valutazione COCO dei keypoint basata sulla Object Keypoint Similarity, che
pesa l'errore di distanza di ogni keypoint in base alla scala dell'istanza e a
una tolleranza per keypoint, quindi svolge il ruolo che IoU svolge per i box.
Richiede `pycocotools`, che è nell'installazione base.

<code-tabs name="val" />

`metrics/keypoints_mAP50-95` è il numero di riferimento, la mean average
precision mediata sulle soglie OKS da 0.50 a 0.95, ed è quello che
l'addestramento usa per scegliere l'epoca migliore. `metrics/keypoints_mAP50` e
`metrics/keypoints_mAP75` sono le versioni a soglia singola, e
`metrics/keypoints_mAP_M` e `metrics/keypoints_mAP_L` dividono la media per area
dell'istanza, media e grande; la valutazione COCO dei keypoint non definisce un
gruppo small. I valori corrispondenti di average recall sono
`metrics/keypoints_AR50-95`, `metrics/keypoints_AR50`,
`metrics/keypoints_AR75`, `metrics/keypoints_AR_M` e
`metrics/keypoints_AR_L`. Ogni chiave di questo task ha il prefisso
`keypoints_`, quindi le chiavi `mAP` dei box che restituisce un rilevatore non
compaiono.

## Esportazione

<code-tabs name="export" />

Un artefatto esportato si ricarica tramite `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. La copertura dei formati cambia da famiglia a
famiglia; la matrice sulla pagina di ogni modello è generata dall'insieme validato
invece che scritta a mano. Vedi
[esportazione e deployment](/docs/export) per i formati, i loro extra e i loro
vincoli.
