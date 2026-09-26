---
title: RT-DETR
families:
  - rtdetr
seo_title: 'RT-DETR, RT-DETRv2 e RT-DETRv4 in LibreYOLO'
description: >-
  Usa RT-DETR, RT-DETRv2 e RT-DETRv4 in LibreYOLO per il rilevamento di oggetti,
  più i box orientati su RT-DETRv2. Installa, fai predizioni, addestra, valida
  ed esporta, con pesi Apache-2.0.
lead: >-
  Un detection transformer costruito per l'inferenza in tempo reale: decodifica
  un insieme fisso di query invece di una griglia densa, quindi non esegue NMS.
  LibreYOLO ne porta tre versioni, distinte dal checkpoint che carichi, e la
  versione 2 serve anche i box orientati.
keywords:
  - RT-DETR
  - RT-DETRv2
  - RT-DETRv4
  - real-time detection transformer
  - DETR
  - object detection
  - oriented bounding box detection
  - OBB
  - DOTA
  - rilevamento oggetti python
  - box orientati
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTDETRr18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRTDETRr18.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Video
      language: python
      code: |
        from libreyolo import LibreYOLO

        # La versione fa parte del nome del file, e la factory smista in base
        # al checkpoint, quindi tutte e tre si caricano allo stesso modo.
        model = LibreYOLO("LibreRTDETRv4s.pt")

        # Qualsiasi sorgente accettata dalla libreria: file, cartella, URL,
        # indice della webcam, stream RTSP o una lista .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
    - label: Box orientati
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Solo versione 2. Il suffisso -obb seleziona il task, e il checkpoint
        # viene riconosciuto come orientato dai suoi stessi tensori, quindi non
        # serve l'argomento task. Questi pesi sono DOTA v1.0, 15 classi aeree
        # a 1024 px.
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        result = model("aerial.png", save=True)

        obb = result.obb
        print(obb.xywhr)     # (N, 5): cx, cy, w, h, radianti
        print(obb.xyxyxyxy)  # le stesse righe come quattro punti d'angolo
        print(result.boxes.xyxy)  # i box allineati agli assi che li racchiudono
    - label: 'Box orientati, CLI'
      language: bash
      code: >
        libreyolo predict model=LibreRTDETRv2n-obb.pt source=aerial.png
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")

        # coco128.yaml scarica un campione di 128 immagini al primo utilizzo.
        # Punta `data` al YAML del tuo dataset per un'esecuzione reale.
        model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 batch=4 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        # Serve l'extra lora: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")

        # val() restituisce un dict semplice, non un oggetto
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTDETRr18.pt data=coco128.yaml
    - label: Contro COCO
      language: bash
      code: |
        # coco-val-only.yaml scarica le 5000 immagini di val2017 e salta il
        # set di addestramento. Porta con sé uno script di download
        # incorporato, quindi serve un permesso esplicito a meno che il
        # dataset non sia già in locale.
        libreyolo val model=LibreRTDETRr18.pt data=coco-val-only.yaml \
          allow_download_scripts=True
    - label: Box orientati
      language: python
      code: |
        from libreyolo import LibreYOLO

        # La validazione orientata fa il matching con IoU ruotato, quindi una
        # predizione nel punto giusto ma con l'angolo sbagliato conta come un
        # mancato rilevamento.
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        metrics = model.val(data="my-obb-dataset.yaml")

        print(metrics["metrics/mAP50-95(OBB)"])
        print(metrics["metrics/mAP50(OBB)"])
  export:
    - label: Python
      language: python
      code: |
        # Serve l'extra onnx: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRTDETRr18.pt format=onnx
    - label: Box orientati
      language: bash
      code: >
        # ONNX e TorchScript sono i target validati per il task orientato,

        # a FP32, batch 1, su un canvas fisso di 1024 per 1024.

        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024

        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript
        imgsz=1024
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreRTDETRr18.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 8022a5a591922a90
---

## Installazione

RT-DETR non richiede nessun extra opzionale. Tutto ciò che importa è già
nell'installazione di base, e l'extra `rtdetr` è un nome stabile che non ci
aggiunge niente.

```bash
pip install libreyolo
```

Il fine-tuning con adattatori tramite `lora=True` è l'eccezione, e richiede
l'extra `lora`.

```bash
pip install "libreyolo[lora]"
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella
cache locale.

<code-tabs name="predict" />

L'oggetto `Results` restituito è quello che restituisce ogni famiglia, quindi
passare a un rilevatore diverso è una modifica di una riga. `conf` e `max_det`
filtrano una decodifica top-k su query e classi; non c'è nessun passaggio di NMS
da regolare, e `iou` è accettato ma non usato. Un checkpoint orientato riempie
`result.obb` in modo nativo e riempie anche `result.boxes` con i rettangoli
allineati agli assi che li racchiudono. Vedi [predizione](/docs/predict) per
sorgenti, streaming e gestione dei risultati.

## Varianti

Tre versioni, due task in tutto, e i codici di dimensione non seguono un'unica
serie. La versione 1 chiama le sue dimensioni come il backbone, ResNet o
HGNetv2. La versione 2 riutilizza solo i nomi ResNet: la versione 1 pubblica già
le due dimensioni HGNetv2, e lì i risultati della versione 2 erano abbastanza
vicini da far sì che LibreYOLO non pubblichi pesi duplicati per esse. La
versione 4 usa una semplice serie di lettere, che va in conflitto con i nomi
HGNetv2 della versione 1, quindi un codice di dimensione da solo non identifica
un modello. La versione è scritta nel nome del file del checkpoint.

<benchmark-table task="detect" />

<va-embed />

La versione 2 mantiene l'architettura e la disposizione dello state dict della
versione 1 e cambia il modo in cui campiona l'attenzione deformabile, ed è per
questo che le due si distinguono dai metadati nel checkpoint e non dalla forma.
La versione 4 è una linea diversa: riutilizza l'architettura e il trainer di
D-FINE, e i suoi pesi vengono dalla distillazione di un modello fondazionale di
visione DINOv3 come insegnante in uno studente HGNetv2. In LibreYOLO
`LibreRTDETRv4` è una sottoclasse di `LibreDFINE` con la testa delle maschere
disattivata in modo fisso, quindi resta solo rilevamento.

### Box orientati nella versione 2

La versione 2 è l'unica versione che porta un secondo task. I suoi task
supportati sono `detect` e `obb`, e i due non condividono né il grafo né la
serie di dimensioni. Il rilevamento usa le dimensioni ResNet a 640 px; il
rilevamento orientato usa una serie HGNetv2, n, s, m, l e x, a 1024 px, e la
dimensione di input si risolve per task e non per famiglia. Un checkpoint viene
riconosciuto come orientato dai suoi stessi tensori, dalle teste dei box a
cinque coordinate e dai parametri di campionamento della versione 2, quindi i
pesi `-obb` si caricano nel grafo orientato senza un argomento `task` e una
discordanza tra i due è un errore bloccante invece che una reinterpretazione
silenziosa.

I file pubblicati vanno da `LibreRTDETRv2n-obb.pt` a
`LibreRTDETRv2x-obb.pt`. Sono i checkpoint ufficiali DOTA v1.0 a scala singola
convertiti nel formato di LibreYOLO, 15 classi aeree che vanno da aereo e nave
fino a porto ed elicottero, e i loro nomi di classe sono impressi nel
checkpoint. A differenza del lato rilevamento, il task orientato è di sola
inferenza: predizione, validazione ed esportazione funzionano, e `train()` su un
modello orientato solleva un errore. Nemmeno il tracking e la data augmentation
al momento del test supportano i box orientati.
[Il rilevamento orientato](/docs/tasks/oriented-detection) copre il task, il
formato delle etichette e le metriche.

## Addestramento

L'addestramento parte da un checkpoint pubblicato. `pretrained` viene accettato
e poi ignorato su tutte e tre le versioni, quindi `pretrained=False` non ti dà
un modello inizializzato a caso. Tutto quello che sta in questa sezione riguarda
il rilevamento: il task orientato della versione 2 è di sola inferenza, e non
esiste una via di trasferimento dai pesi di rilevamento verso di esso, perché i
due usano backbone diversi.

<code-tabs name="train" />

Il learning rate è l'argomento da azzeccare, e ogni versione porta il proprio
valore predefinito invece di quello valido per tutta la libreria. La firma di
`train()` in Python lo legge dalla configurazione di addestramento di quella
versione, e la CLI risolve lo stesso valore quando `lr0` non viene passato. Le
versioni 1 e 2 accettano anche `lr_backbone` e per default lo fissano a un
ventesimo di `lr0`, seguendo la ricetta originale; la versione 4 gira attraverso
il trainer di D-FINE, che invece scala il gruppo di parametri del backbone con
`backbone_lr_mult`.

Lascia `imgsz` alla dimensione nativa del checkpoint a meno che tu non abbia un
motivo per cambiarla. La validazione e la predizione ad altre dimensioni
funzionano, con un residuo: una dimensione rettangolare il cui numero di token
coincide con quello della dimensione nativa riutilizza comunque un embedding
costruito per un rapporto d'aspetto sbagliato.

Vedi [addestramento](/docs/train) per dataset, data augmentation, multi-GPU e
logger.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurate su qualsiasi dataset nel formato con cui
hai addestrato.

<code-tabs name="val" />

Le righe della tabella di benchmark qui sopra vengono dall'harness di benchmark
di LibreYOLO; la nota sotto quella tabella indica quale dataset le ha prodotte e
rimanda ai record delle esecuzioni.

La validazione orientata passa dalla stessa chiamata e riporta le stesse chiavi,
più quattro ripetute sotto un suffisso `(OBB)`. Il matching usa l'IoU ruotato
invece dell'IoU dei rettangoli che li racchiudono, quindi un errore di angolo è
un mancato rilevamento. `augment=True` viene rifiutato su questo task.

## Esportazione

<export-matrix />

La matrice copre tutta la linea in una sola pagina: dove le tre versioni non
concordano su un formato, la cella mostra la più debole delle tre, quindi qui
niente è sopravvalutato per la versione che carichi. La riga dei box orientati
appartiene alla sola versione 2. Lì ONNX e TorchScript sono validati, a FP32,
batch 1 e con un canvas fisso di 1024 per 1024; OpenVINO, TensorRT ed
ExecuTorch convertono e ricaricano ma non hanno raggiunto la parità dell'output
grezzo sull'intero insieme di query, quindi i box in cima coincidono fino a una
frazione di pixel mentre la coda va alla deriva.

Un artefatto esportato si ricarica con `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

Il nome del file porta la versione, poi la dimensione, poi il task. I pesi di
rilevamento sono `LibreRTDETR<size>.pt`, `LibreRTDETRv2<size>.pt` e
`LibreRTDETRv4<size>.pt`, tutti a 640 px. I pesi orientati esistono solo per la
versione 2 e aggiungono il suffisso del task, da `LibreRTDETRv2n-obb.pt` a
`LibreRTDETRv2x-obb.pt`, tutti a 1024 px e addestrati su DOTA v1.0 invece che su
COCO.

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />

Il blocco qui sopra è quello che pubblicano gli autori per il rilevamento delle
versioni 1 e 2. I pesi orientati della versione 2 hanno un terzo upstream, il
repository RiO-DETR sotto Apache-2.0 su
[github.com/RicePasteM/RiO-DETR](https://github.com/RicePasteM/RiO-DETR), da cui
arrivano i checkpoint DOTA; cita quel progetto se ne hai usato uno. La versione 4
è un articolo a parte di un gruppo diverso e ha il proprio blocco di citazione su
[github.com/RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4#4-citation);
cita quello se hai usato un checkpoint della versione 4.
