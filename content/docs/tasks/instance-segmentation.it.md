---
title: Segmentazione di istanze
seo_title: Segmentazione di istanze in LibreYOLO
description: >-
  Segmenta i singoli oggetti in LibreYOLO: le famiglie che coprono il task, il
  formato delle etichette a poligoni e le chiamate di predizione, addestramento,
  validazione ed esportazione.
lead: >-
  La segmentazione di istanze localizza ogni istanza di oggetto e restituisce
  per ciascuna una maschera pixel per pixel, oltre al box, alla classe e al
  punteggio che restituisce un detector. La chiave del task è segment.
keywords:
  - instance segmentation python
  - segmentazione di istanze python
  - predizione maschere oggetti
  - addestrare modello di segmentazione
  - etichette a poligoni yolo
  - mask mAP
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Il suffisso -seg nel nome del file seleziona la testa delle maschere,

        # quindi non serve l'argomento task.

        model = LibreYOLO("LibreDFINEn-seg.pt")

        result = model(SAMPLE_IMAGE, save=True)


        print(result.masks.data.shape)   # (N, H, W), una maschera per
        rilevamento

        print(result.boxes.xyxy.shape)   # (N, 4), le stesse N righe
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDFINEn-seg.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Contorni delle maschere
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDFINEn-seg.pt")

        result = model(SAMPLE_IMAGE)


        # .xy è una lista di contorni (P, 2) in pixel, .xyn gli stessi
        normalizzati.

        for name, contour in zip(result.boxes.cls, result.masks.xy):
            print(result.names[int(name)], contour.shape)
    - label: 'Un''altra famiglia, stessa chiamata'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Continua dai pesi di segmentazione pubblicati, testa delle maschere
        inclusa.

        # data deve puntare a un dataset le cui etichette contengono poligoni.

        model = LibreYOLO("LibreDFINEn-seg.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: Dai pesi di rilevamento
      language: bash
      code: >
        # I pesi di rilevamento non hanno una testa delle maschere, quindi
        questo

        # è un trasferimento esplicito: la testa parte non addestrata. È la

        # richiesta di task=segment ad autorizzarlo.

        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])       # maschere
        print(metrics["metrics/mAP50-95(M)"])    # maschere, esplicito
        print(metrics["metrics/mAP50-95(B)"])    # box
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn-seg.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDFINEn-seg.pt format=onnx imgsz=640
    - label: Usare il file esportato
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory instrada in base al suffisso del file, quindi un artefatto

        # esportato si carica come un checkpoint e restituisce lo stesso oggetto
        Results.

        model = LibreYOLO("LibreDFINEn-seg.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.masks.data.shape)
source_hash: 33e331eac0f9b0af
---

## Definizione

La segmentazione di istanze è rilevamento più forma. Ogni istanza di oggetto
riceve comunque un box, una classe e un punteggio, e riceve anche una maschera
binaria che copre i pixel che le appartengono. Le maschere possono sovrapporsi,
e i pixel che non appartengono a nessun oggetto restano non assegnati: è questo
che distingue il task dalla
[segmentazione semantica](/docs/tasks/semantic-segmentation) e dalla
[segmentazione panottica](/docs/tasks/panoptic-segmentation).

`segment` è la chiave canonica del task, e il suffisso `-seg` nel nome del file
di un checkpoint la seleziona, quindi `task=` non serve quando carichi pesi
pubblicati.

`predict()` riempie `result.masks` accanto a `result.boxes`. `.data` è uno
stack `(N, H, W)` sul canvas dell'immagine originale, allineato per riga con i
box, quindi la maschera `i` appartiene al box `i`. `.xy` converte ogni maschera
nel suo contorno esterno più grande come array di pixel `(P, 2)`, e `.xyn`
restituisce lo stesso contorno normalizzato.

## Modelli

Quattro famiglie addestrano e predicono maschere: [RF-DETR](/docs/models/rf-detr),
[EdgeCrafter](/docs/models/edgecrafter), [D-FINE](/docs/models/d-fine) e
[RTMDet](/docs/models/rtmdet). RF-DETR richiede il suo extra dedicato,
`pip install "libreyolo[rfdetr]"`; le altre tre funzionano con il pacchetto base.

[Mask R-CNN](/docs/models/mask-rcnn) predice, valida ed esporta maschere, ma il
suo `train()` solleva `NotImplementedError`.

[EoMT](/docs/models/eomt) predice e valida maschere e nemmeno lui può
addestrare, e la sua esportazione è ancora più ristretta: `export()` accetta
solo il task semantico, e solleva `NotImplementedError` per `segment` e
`panoptic`, perché il contratto di runtime per le maschere da query di cui
questi due hanno bisogno non è stato definito. Usa EoMT per le maschere di
istanza in Python, non tramite un grafo esportato.

Un gruppo a parte segmenta a partire da un prompt invece che da un elenco di
classi: un clic, un box o una frase seleziona l'oggetto, e il modello ne
restituisce la maschera. [SAM](/docs/models/sam), [SAM 2](/docs/models/sam-2),
[SAM 3](/docs/models/sam-3), [MobileSAM](/docs/models/mobilesam),
[EdgeTAM](/docs/models/edgetam) e [PicoSAM3](/docs/models/picosam3) funzionano
così, come pure [SenseNova-Vision](/docs/models/sensenova-vision), la cui
segmentazione è referring: prende una frase che nomina un solo oggetto. Si
caricano tramite la propria factory e i propri extra, e la pagina di ogni
modello riporta la chiamata esatta.

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e messi in cache in
locale.

<code-tabs name="predict" />

`conf` e `max_det` modellano l'output nello stesso modo in cui lo fanno per il
rilevamento, e le maschere vengono filtrate insieme ai box a cui appartengono.
Vedi [predizione](/docs/predict) per le sorgenti, lo streaming e la gestione dei
risultati.

## Formato del dataset

La struttura è quella del rilevamento: un file di etichette `.txt` per
immagine, trovato sostituendo `images` con `labels` nel percorso dell'immagine e
cambiando l'estensione.

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

Quello che cambia è la riga. Un segmento è un indice di classe seguito da un
poligono appiattito:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

Almeno tre punti, quindi il numero di coordinate dopo l'indice di classe è pari
e almeno sei, e il poligono deve essere non degenere. Le coordinate sono float
in `[0, 1]` relativi alla larghezza e all'altezza dell'immagine originale. In un
dataset di segmentazione è accettata anche una riga di rilevamento a cinque
campi, che viene letta come un segmento rettangolare: questo rende caricabile un
dataset con soli box senza un passaggio di conversione.

Il YAML è quello del rilevamento:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

Funziona anche il COCO JSON nativo: aggiungi una mappatura `annotations` dal
nome dello split al file JSON, e il percorso dello split indica la radice delle
immagini.

## Addestramento

<code-tabs name="train" />

Per impostazione predefinita l'addestramento continua da un checkpoint `-seg`
pubblicato. Partire dai pesi di rilevamento è possibile ma è un trasferimento
deliberato: quei pesi non hanno una testa delle maschere, quindi parte non
addestrata, ed è il passaggio di `task=segment` ad autorizzare lo scambio. Vedi
[addestramento](/docs/train) per dataset, data augmentation, multi-GPU e logger.

## Validazione

`val()` restituisce un semplice dizionario di chiavi `metrics/`. Box e maschere
vengono valutati separatamente, entrambi con la valutazione COCO, e i numeri
delle maschere sono quelli principali.

<code-tabs name="val" />

Le chiavi senza suffisso contengono i risultati delle maschere:
`metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, poi `metrics/mAP_small`,
`metrics/mAP_medium` e `metrics/mAP_large` per area dell'oggetto, e
`metrics/AR1`, `metrics/AR10`, `metrics/AR100`, `metrics/AR_small`,
`metrics/AR_medium`, `metrics/AR_large` per il recall medio.
`metrics/AR_max_det` e `metrics/max_det` registrano il limite di rilevamenti
usato dall'esecuzione.

Quattro valori vengono pubblicati anche con un suffisso esplicito, `(M)` per
mask e `(B)` per box, così che un confronto non dipenda mai da quale numero la
famiglia ha deciso di considerare principale: `metrics/mAP50-95(M)` e
`metrics/mAP50-95(B)`, `metrics/mAP50(M)` e `metrics/mAP50(B)`,
`metrics/precision(M)` e `metrics/precision(B)`, `metrics/recall(M)` e
`metrics/recall(B)`. In questo task non esistono `metrics/precision` o
`metrics/recall` senza suffisso.

Leggi con attenzione le chiavi di precisione e recall. Sono mantenute per
retrocompatibilità e sono alias, non un punto di lavoro:
`metrics/precision(M)` contiene lo stesso valore di `metrics/mAP50-95(M)`, e
`metrics/recall(M)` lo stesso valore dell'AR delle maschere a 100 rilevamenti,
con `(B)` che si comporta allo stesso modo per i box. Tracciare una coppia di
queste chiavi significa riportare due volte lo stesso numero.

## Esportazione

<code-tabs name="export" />

Un artefatto esportato si ricarica tramite `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce gli stessi `Results`. La copertura della segmentazione è più
ristretta di quella del rilevamento nella stessa famiglia. La matrice su ogni
pagina di modello è generata a partire dall'insieme validato e indica il motivo
per cui un target non è disponibile. Vedi
[esportazione e deployment](/docs/export) per i formati, i loro extra e i loro
vincoli.
