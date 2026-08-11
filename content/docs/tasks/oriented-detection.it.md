---
title: Rilevamento orientato
seo_title: Rilevamento orientato in LibreYOLO
description: >-
  Rileva oggetti ruotati in LibreYOLO: le famiglie che offrono box orientati, la
  riga di etichetta a quattro vertici e le chiamate di predizione,
  addestramento, validazione ed esportazione.
lead: >-
  Il rilevamento orientato di oggetti localizza ogni istanza con un rettangolo
  ruotato invece che con uno allineato agli assi, così un oggetto inclinato
  viene racchiuso in modo aderente invece che da un box pieno di sfondo. La
  chiave del task è obb.
keywords:
  - oriented bounding box detection
  - rilevamento oggetti ruotati python
  - OBB python
  - DOTA dataset
  - rilevamento oggetti immagini aeree
  - rotated IoU
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        # Richiede l'extra rfdetr: pip install "libreyolo[rfdetr]"

        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Il suffisso -obb nel nome del file seleziona il task, quindi non

        # serve un argomento task.

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        result = model(SAMPLE_IMAGE, save=True)


        obb = result.obb

        print(obb.xywhr)   # (N, 5): centro x, centro y, larghezza, altezza,
        radianti

        print(obb.conf, obb.cls)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRFDETRs-obb.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Vertici invece di angoli
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        result = LibreYOLO("LibreRFDETRs-obb.pt")(SAMPLE_IMAGE)

        obb = result.obb


        print(obb.xyxyxyxy.shape)    # (N, 4, 2) vertici in pixel

        print(obb.xyxyxyxyn.shape)   # gli stessi, normalizzati

        print(obb.xyxy.shape)        # (N, 4) box allineato agli assi che lo
        racchiude
    - label: Un checkpoint più piccolo
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRn-obb.pt")
        result = model(SAMPLE_IMAGE)

        print(result.obb.xywhr.shape)
    - label: RT-DETRv2
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Pesi DOTA v1.0, 15 classi aeree a 1024 px. Il grafo orientato viene
        # riconosciuto dai tensori stessi del checkpoint, quindi nessun task.
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        result = model("aerial.png", save=True)

        obb = result.obb
        print(obb.xywhr)
        print(result.names)   # plane, ship, harbor, helicopter e altre 11
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Continua dai pesi orientati pubblicati. data deve puntare a un

        # dataset le cui righe di etichetta portano quattro vertici.

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        model.train(data="my-obb-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: Dai pesi di rilevamento
      language: bash
      code: |
        # I pesi di rilevamento non predicono alcun angolo, quindi questo è un
        # transfer esplicito. È la richiesta di task=obb ad autorizzarlo.
        libreyolo train model=LibreRFDETRs.pt data=my-obb-dataset.yaml \
          task=obb epochs=50 imgsz=512
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        # val() restituisce un semplice dict, non un oggetto.
        metrics = model.val(data="my-obb-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml
    - label: RT-DETRv2
      language: bash
      code: |
        libreyolo val model=LibreRTDETRv2n-obb.pt data=my-obb-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRFDETRs-obb.pt format=onnx imgsz=512
    - label: RT-DETRv2
      language: bash
      code: >
        # ONNX e TorchScript sono qui i target validati, a FP32,

        # batch 1, su un canvas fisso di 1024 per 1024.

        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024

        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript
        imgsz=1024
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file: un artefatto esportato
        # si carica come un checkpoint e restituisce lo stesso oggetto Results.
        model = LibreYOLO("LibreRFDETRs-obb.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.obb.xywhr)
source_hash: 0d605d956f3ea025
---

## Definizione

Il rilevamento orientato aggiunge un numero a un rilevamento: l'angolo. Ogni
istanza riceve un rettangolo ruotato, una classe e un punteggio. Il guadagno è
l'aderenza. Una nave a 45 gradi, il tetto di un capannone, una fila di camion
parcheggiati: un box allineato agli assi attorno a uno qualsiasi di essi è per
lo più sfondo, e due box vicini si sovrappongono anche quando gli oggetti non
lo fanno. Per questo il task è standard nelle immagini aeree e nell'analisi
del layout dei documenti, e per questo il dataset di riferimento è DOTA.

`obb` è la chiave canonica del task, e il suffisso `-obb` nel nome di un file
di checkpoint la seleziona, quindi `task=` non serve quando carichi pesi
pubblicati.

`predict()` riempie `result.obb`. `.xywhr` è la forma canonica `(N, 5)`:
centro x, centro y, larghezza, altezza e un angolo in radianti che dà la
rotazione del lato della larghezza attorno al centro. `.conf` e `.cls` portano
il punteggio e l'indice della classe in `result.names`, e `.id` un id di
traccia quando si fa tracking. `.xyxyxyxy` converte ogni riga nei suoi quattro
vertici come pixel `(N, 4, 2)`, `.xyxyxyxyn` normalizza quei vertici e `.xyxy`
dà il box allineato agli assi che lo racchiude, che è quello da usare quando
il codice a valle capisce solo rettangoli. Anche `result.boxes` viene
riempito, con la forma allineata agli assi.

## Modelli

Due famiglie coprono questo task, e quale scegliere dipende dal fatto che tu
debba addestrare o no.

[RF-DETR](/docs/models/rf-detr) è quella che si addestra. Predice, addestra,
valida ed esporta box orientati, e pubblica checkpoint orientati in quattro
taglie, n, s, m e l. Richiede il proprio extra,
`pip install "libreyolo[rfdetr]"`, e la pagina del modello riporta la licenza
dei pesi e la provenienza.

Leggi la sezione qui sotto su che cosa predicono davvero quei checkpoint prima
di basare i tuoi piani su di essi.

[RT-DETRv2](/docs/models/rt-detr) è quella con i pesi per le immagini aeree.
Pubblica da `LibreRTDETRv2n-obb.pt` a `LibreRTDETRv2x-obb.pt`, i checkpoint
ufficiali DOTA v1.0 a scala singola convertiti nel formato di LibreYOLO, che
coprono le 15 classi di DOTA a 1024 px. Non richiede nessun extra oltre al
pacchetto base, il grafo orientato viene riconosciuto dai tensori stessi del
checkpoint, e predizione, validazione ed esportazione ONNX e TorchScript sono
tutte supportate. L'addestramento no: su quella famiglia il task orientato è
di sola inferenza, `train()` solleva un errore, e non c'è transfer dai suoi
pesi di rilevamento, che usano un backbone diverso. Anche il tracking e la
test-time augmentation non sono disponibili per i box orientati.

Quindi: categorie DOTA pronte all'uso, RT-DETRv2. Le tue etichette orientate,
RF-DETR.

## Predizione

I pesi si scaricano da Hugging Face al primo utilizzo e restano in cache in
locale.

<code-tabs name="predict" />

Sappi che cosa sono i checkpoint pubblicati di RF-DETR prima di eseguirli.
Anche se DOTA è il benchmark di riferimento per questo task, quei pesi non
sono stati addestrati su di esso. Tutti e quattro sono stati inizializzati dai
pesi di rilevamento di RF-DETR e affinati su un singolo dataset di Roboflow
Universe di riprese da drone, con sei classi di veicoli: bike, bus, car,
other_vehicle, taxi e truck. Le loro model card li descrivono come pesi di
sviluppo, prodotti mentre si validava il supporto all'addestramento orientato,
e dicono che non vanno letti come pesi di produzione o ufficiali per i
benchmark.

In pratica significa che sono un punto di partenza funzionante per i box
orientati su veicoli visti dall'alto, e per verificare che la tua pipeline
giri da un capo all'altro. Qualsiasi altro dominio richiede l'addestramento
sulle tue etichette orientate, e per le categorie aeree per cui DOTA è nota, i
checkpoint RT-DETRv2 sono quelli davvero addestrati su quei dati. `conf` e
`max_det` modellano l'output come fanno per il rilevamento. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Formato del dataset

La struttura è quella del rilevamento: un file di etichette `.txt` per
immagine, trovato sostituendo `images` con `labels` nel percorso
dell'immagine e cambiando l'estensione.

```text
dataset/
  data.yaml
  images/
    train/P0001.png
    val/P0101.png
  labels/
    train/P0001.txt
    val/P0101.txt
```

Una riga è composta esattamente da nove campi, un indice di classe seguito da
quattro vertici in ordine:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

I quattro punti sono float normalizzati in `[0, 1]` e devono formare un
rettangolo orientato non degenere. Nel file di etichette non è memorizzato
alcun angolo: il loader ricava lo `xywhr` canonico dai vertici. Il parser è
rigoroso per default e rifiuta le coordinate fuori intervallo, mentre
l'acquisizione del dataset e della validazione può prima limitare le
coordinate a `[0, 1]` per etichette al bordo di un ritaglio altrimenti valide,
e poi rifiutare comunque i box degeneri.

Il parsing delle righe tiene conto del task. Nove campi significano un box
orientato solo in modalità `obb`; in modalità `segment` la stessa riga viene
letta come un poligono a quattro punti.

Lo YAML è quello del rilevamento:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: plane
  1: ship
```

Si carica anche il COCO JSON nativo, con una mappa `annotations` dal nome
dello split al file JSON. Le annotazioni sono lette in ordine di priorità: un
campo `obb` con otto vertici in spazio pixel, un campo `obb` con
`[cx, cy, w, h, angle]` con l'angolo in radianti, un poligono `segmentation` o
una RLE riadattati al loro rettangolo di area minima, oppure un semplice
`bbox` COCO, che viene trattato come un rettangolo allineato agli assi e reso
canonico in `xywhr`.

Il parser canonico delle righe è `libreyolo.data.parse_yolo_obb_label_line`.

## Addestramento

<code-tabs name="train" />

Addestrare su questo task significa RF-DETR. Per default l'addestramento
continua da un checkpoint `-obb` pubblicato. Partire dai pesi di rilevamento è
un transfer deliberato: quei pesi non predicono alcun angolo, ed è il
passaggio di `task=obb` ad autorizzare lo scambio. Tieni `lr0` a `1e-4` o al
di sotto, come per gli altri task della famiglia. I checkpoint orientati di
RT-DETRv2 non si possono affinare; usali così come sono, oppure addestra un
modello RF-DETR sulle tue etichette. Vedi [addestramento](/docs/train) per
dataset, augmentation, multi-GPU e logger.

## Validazione

`val()` restituisce un semplice dizionario di chiavi `metrics/`. Il matching
usa lo IoU ruotato, calcolato tra rettangoli orientati invece che tra i box
allineati agli assi che li racchiudono, quindi una predizione con la posizione
giusta e l'angolo sbagliato viene contata come un mancato rilevamento.

<code-tabs name="val" />

`metrics/mAP50-95` è la mean average precision mediata sulle soglie di IoU da
0.50 a 0.95 a passi di 0.05, ed è il numero di riferimento. A differenza del
percorso COCO usato dal rilevamento, questo task rispetta `iou_thresholds`
nella configurazione di validazione, quindi l'intervallo di soglie si può
cambiare. `metrics/mAP50` e `metrics/mAP75` sono le versioni a soglia singola.
`metrics/precision` e `metrics/recall` sono la precisione e il recall reali a
IoU 0.50, letti nel punto di lavoro più permissivo: viene contata ogni
predizione che ha superato la soglia di confidenza, e in validazione quella
soglia vale 0.001 per default. Alzare `conf` quindi li sposta, mentre i valori
di mAP, che usano tutta la curva precisione-recall, restano fermi. Quattro di
queste si ripetono con un suffisso `(OBB)`, `metrics/mAP50-95(OBB)`,
`metrics/mAP50(OBB)`, `metrics/precision(OBB)` e `metrics/recall(OBB)`, ed è
così che chi chiama distingue un risultato orientato da uno allineato agli
assi quando entrambi stanno nella stessa tabella. `metrics/mAP75` non ha un
gemello con suffisso.

Due opzioni non fanno nulla su questo task. `save_json` e `save_plots` sono
accettate e registrano un avviso: i dump delle predizioni orientate e i
grafici di validazione non sono implementati.

## Esportazione

<code-tabs name="export" />

Un artefatto esportato si ricarica tramite `LibreYOLO()` in base al suffisso
del file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. La copertura dei formati cambia da task a
task nella stessa famiglia, e la matrice sulla pagina del modello è generata
dall'insieme validato e indica il motivo per cui un target non è disponibile.
Vedi [esportazione e deployment](/docs/export) per i formati, i loro extra e i
loro vincoli.
