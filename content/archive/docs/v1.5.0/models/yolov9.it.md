---
title: YOLOv9
families:
  - yolo9
seo_title: 'YOLOv9: predici, addestra ed esporta con licenza MIT'
description: >-
  Usa YOLOv9 in LibreYOLO, inclusa la testa end-to-end senza NMS e la testa
  stride 4 per oggetti piccoli. Installa, fai predizioni, addestra, valida ed
  esporta.
lead: >-
  Un rilevatore convoluzionale a singolo stadio: una sola passata assegna un
  punteggio a una griglia densa di box e NMS scarta i duplicati. LibreYOLO ne
  porta tre varianti, una delle quali senza passaggio di NMS.
keywords:
  - YOLOv9
  - YOLO9
  - object detection
  - rilevamento oggetti python
  - detection senza NMS
  - end-to-end detection
  - small object detection
  - GELAN
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Senza NMS
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Stessa chiamata, checkpoint diverso. La testa end-to-end restituisce

        # le sue predizioni migliori, quindi non viene eseguito NMS e iou è
        ignorato.

        model = LibreYOLO("LibreYOLO9E2Es.pt")

        result = model(SAMPLE_IMAGE, conf=0.25, max_det=300)


        print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: Oggetti piccoli
      language: python
      code: >
        from libreyolo import LibreYOLO9P2


        # La variante stride 4 non ha un checkpoint COCO proprio: indicane uno

        # di rilevamento base, il backbone e il neck si caricano invariati e la

        # torre della testa stride 4 parte da un'inizializzazione casuale.

        model = LibreYOLO9P2(None, size="s")

        model.train(data="my-dataset.yaml", epochs=100,
        pretrained="LibreYOLO9s.pt")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=my-dataset.yaml
    - label: Su COCO
      language: bash
      code: >
        # Il file yaml di COCO incluso porta uno script di download integrato,

        # quindi serve un permesso esplicito a meno che il dataset non sia già
        in locale.

        libreyolo val model=LibreYOLO9c.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: Con NMS nel grafo
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx nms=True \
          conf=0.25 iou=0.45 max_det=300
    - label: Usare il file esportato
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory smista in base al suffisso del file, quindi un artefatto

        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        Results.

        model = LibreYOLO("LibreYOLO9s.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: eaa6023a4a0b9e71
---

## Installazione

YOLOv9 non richiede nessun extra oltre al pacchetto base.

```bash
pip install libreyolo
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella cache
locale.

<code-tabs name="predict" />

L'oggetto `Results` restituito è quello che restituisce ogni famiglia, quindi
passare a un rilevatore diverso è una modifica di una riga. Sul modello base e
su quello stride 4, `conf` fissa la soglia di confidenza e `iou` la soglia di
NMS. Il modello end-to-end non esegue NMS e ignora `iou`, quindi sono `conf` e
`max_det` a dare forma al suo output. Vedi [predizione](/docs/predict) per
sorgenti, streaming e gestione dei risultati.

## Varianti

Tre varianti condividono un backbone. Tutte e tre fanno solo rilevamento, e
accettano gli stessi argomenti.

Il modello base predice su tre scale di feature ed elimina i bounding box
duplicati con NMS.

Il modello end-to-end mantiene quella testa e le affianca un ramo di matching
uno-a-uno. L'inferenza legge soltanto il ramo uno-a-uno e ne prende le
predizioni con il punteggio più alto, quindi non viene eseguito nessun NMS.
Scegli questo quando il runtime su cui metti in produzione non ha un operatore
NMS.

Il modello stride 4 fa affiorare un livello più in alto nel backbone, estende il
neck fino a lì e predice su quattro scale invece che tre. La scala in più serve
per gli oggetti che coprono pochi pixel; l'unico checkpoint pubblicato per
questo modello è addestrato su immagini aeree. I checkpoint di rilevamento base
si trasferiscono su di esso: il backbone e il neck si caricano invariati, le tre
torri della testa preaddestrate salgono di una posizione e la torre stride 4
parte da un'inizializzazione casuale.

<benchmark-table task="detect" />

<va-embed />

## Addestramento

<code-tabs name="train" />

`pretrained` decide da cosa parte l'esecuzione. Passa `True` per caricare il
checkpoint pubblicato dello stesso modello e della stessa dimensione, oppure un
nome o un percorso per qualsiasi altra cosa. I tensori la cui forma non coincide
vengono saltati invece che rifiutati, e l'esecuzione registra quanti ne sono
stati caricati, quindi un checkpoint addestrato su un numero di classi diverso
resta un punto di partenza utilizzabile.

Il modello stride 4 non ha un checkpoint COCO pubblicato proprio, quindi lì
`True` si risolve in un file che non esiste e il download fallisce. Indica
invece un checkpoint di rilevamento base.

Vedi [addestramento](/docs/train) per dataset, data augmentation, multi-GPU e
logger.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurate su qualsiasi dataset nel formato con cui
hai addestrato.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Una spunta vale per tutte e tre le varianti: dove differiscono, la matrice
riporta la più debole delle tre.

Un artefatto esportato si ricarica con `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. Eseguire il grafo in un runtime nudo, senza
LibreYOLO installato, è supportato anche questo, ma allora il preprocessing e il
postprocessing tocca scriverli a te.

Per il modello di rilevamento base, la metà di postprocessing può spostarsi
dentro il grafo. `nms=True` su un'esportazione ONNX mette la soppressione dentro
il modello, e il primo output diventa un tensore fisso `(1, max_det, 6)` le cui
righe sono `x1, y1, x2, y2, score, class`, riempite di zeri oltre il numero di
rilevamenti. Quel grafo è a batch 1 e non porta assi dinamici. I modelli
end-to-end e stride 4 non accettano il flag.

Ogni formato installa un extra diverso e accetta qualche argomento suo. Entrambe
le cose stanno nella pagina di quel formato.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box>

Un checkpoint qui non è MIT. Il modello stride 4 addestrato su VisDrone2019-DET
eredita i termini CC BY-NC-SA 3.0 di quel dataset: solo uso non commerciale,
share-alike su tutto ciò che ne deriva, e fuori dalla licenza permissiva con cui
viene distribuito il resto di questa famiglia. Predice le classi aeree di
VisDrone invece di quelle di COCO. La libreria stampa tutto questo prima di
scaricare il file.

</provenance-box>

## Citazione

<citation-block />
