---
title: Soglie e filtraggio
seo_title: 'conf, iou e max_det in LibreYOLO'
description: >-
  Cosa fanno davvero conf, iou, max_det e classes al momento della predizione,
  quali famiglie ignorano iou perché non eseguono NMS e perché agnostic_nms non
  ha alcun effetto.
lead: >-
  Quattro argomenti decidono quali predizioni sopravvivono: conf, iou, max_det e
  classes. Solo due valgono per ogni famiglia, perché un predittore di insiemi
  decodifica un insieme fisso di query e non esegue mai la NMS.
keywords:
  - soglia di confidenza yolo
  - conf yolo python
  - iou threshold nms
  - max_det yolo
  - filtrare classi detection python
  - agnostic nms
  - detr senza nms
  - filtro classi inferenza
last_verified: 1.5.0
verification: >-
  Valori predefiniti presi da InferenceRunner.__call__ in
  libreyolo/models/base/inference.py. Comportamento della NMS per famiglia letto
  da ogni modulo in libreyolo/postprocess/ e verificato rispetto a
  _is_nms_free_family in libreyolo/backends/base.py. Filtraggio per classe da
  InferenceRunner._apply_classes_filter e _wrap_results. Stato di agnostic_nms
  da NOOP_PREDICT_KWARGS in libreyolo/utils/predict_args.py. Gestione del
  vocabolario aperto da NMS_THRESHOLD in libreyolo/models/openvocab/base.py.
  Valori predefiniti della validazione da BaseModel.val.
snippets:
  basic:
    - label: I quattro argomenti
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(
            SAMPLE_IMAGE,
            conf=0.25,      # tiene le predizioni con punteggio pari o superiore
            iou=0.45,       # soglia di sovrapposizione della NMS, dove la NMS viene eseguita
            max_det=300,    # limite per immagine
            classes=None,   # oppure una lista di id di classe
        )
        print(len(result.boxes))
    - label: Variare conf
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        for conf in (0.1, 0.25, 0.5, 0.75):
            result = model(SAMPLE_IMAGE, conf=conf)
            print(conf, len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt conf=0.4 iou=0.5 max_det=100 \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  classes:
    - label: Filtrare su classi specifiche
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # Gli id di classe indicizzano model.names. Su COCO, 0 è person.
        result = model(SAMPLE_IMAGE, classes=[0])

        print({result.names[int(c)] for c in result.boxes.cls.tolist()})
    - label: Trovare l'id di un nome
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        wanted = {"person", "backpack"}
        ids = [i for i, name in result.names.items() if name in wanted]
        print(ids)

        filtered = model(SAMPLE_IMAGE, classes=ids)
        print(len(filtered.boxes))
  nmsfree:
    - label: iou su una famiglia che non esegue la NMS
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # RF-DETR decodifica un insieme fisso di query, quindi iou qui non
        cambia nulla.

        model = LibreYOLO("LibreRFDETRs.pt")


        loose = model(SAMPLE_IMAGE, iou=0.9)

        tight = model(SAMPLE_IMAGE, iou=0.1)


        # Stesso conteggio in entrambi i casi. conf e max_det sono i controlli
        che funzionano.

        print(len(loose.boxes), len(tight.boxes))
source_hash: 0b978963c356027d
---

## I quattro argomenti

| Argomento | Predefinito | Si applica a |
|---|---|---|
| `conf` | `0.25` | Ogni famiglia |
| `iou` | `0.45` | Le famiglie che eseguono la non-maximum suppression |
| `max_det` | `300` | Ogni famiglia |
| `classes` | `None` | Ogni famiglia |

<code-tabs name="basic" />

Due di questi sono universali e due no, ed è la cosa più utile da sapere prima
di mettere mano a qualsiasi valore.

La validazione usa valori predefiniti diversi di proposito: `val()` gira a
`conf=0.001` e `iou=0.6`, perché la precisione media si calcola su una curva
precisione-recall completa e un taglio a 0.25 la troncherebbe.

## conf

`conf` è il punteggio sotto al quale una predizione viene scartata. Si applica a
ogni famiglia, comprese quelle che non eseguono mai la NMS, ed è il primo
controllo a cui ricorrere quando i rilevamenti sono troppi o troppo pochi.

Il valore predefinito di `0.25` va bene per guardare delle immagini. Alimentare
un sistema a valle di solito ne richiede uno più alto; misurare l'accuratezza ne
richiede uno molto più basso.

## iou

`iou` è la sovrapposizione oltre la quale la non-maximum suppression rimuove il
box con punteggio più basso fra due box della stessa classe. Ha un significato
solo se la famiglia esegue davvero la soppressione.

Un predittore di insiemi decodifica un numero fisso di query e prende quelle con
il punteggio più alto. I duplicati vengono soppressi dentro l'architettura
durante l'addestramento, non da un passaggio di postprocessing, quindi non c'è
alcuna soglia da regolare. Queste famiglie accettano `iou` per parità di API e lo
ignorano:

CenterNet, DEIM, DETR, Deformable DETR, D-FINE, DINO-DETR, EdgeCrafter,
Faster R-CNN, LW-DETR, Mask R-CNN, RF-DETR, RT-DETR e la testa end-to-end di
YOLOv9. Le varianti costruite su quei decoder ereditano il comportamento.

<code-tabs name="nmsfree" />

La maggior parte lo dichiara nelle docstring di postprocessing, ma a runtime non
viene sollevato alcun avviso, quindi una scansione su `iou` con RF-DETR produce
una linea piatta invece di un errore. Faster R-CNN e Mask R-CNN sono un caso
leggermente diverso: entrambi hanno già eseguito la NMS dentro il modello, a una
soglia fissa a monte che `iou` non ha alcun modo supportato di cambiare.

Queste famiglie invece la usano: da YOLOv1 a YOLOv4, YOLOv7, YOLOv9, YOLOX,
YOLO-NAS, RTMDet, PicoDet, EfficientDet, FCOS, RetinaNet e SSD.

Due opzioni al momento della predizione rendono `iou` rilevante anche per un
predittore di insiemi, perché entrambe uniscono i box dopo che il modello ha
finito:

- `tiling=True` riconcilia i tasselli sovrapposti con una NMS per classe a `iou`
- `augment=True` unisce le viste ribaltate con una NMS per classe a `iou`

Entrambe sono trattate in [Prestazioni dell'inferenza](/docs/predict/performance).

I rilevatori a vocabolario aperto hanno una regola propria. Una famiglia il cui
processor esegue la NMS dichiara una propria soglia predefinita e rispetta `iou`,
ed è il caso di OMDet-Turbo. Le famiglie che non sopprimono nulla, Grounding
DINO, OWLv2 e OV-DEIM, emettono un avviso quando viene passato `iou`. Quell'avviso
è l'unico del suo genere nella libreria.

## max_det

`max_det` limita quante predizioni tornano indietro per una singola immagine. Si
applica ovunque, ma attraverso meccanismi diversi: una famiglia con NMS tronca
dopo la soppressione, un predittore di insiemi lo usa come dimensione della sua
selezione top-k.

Alcune famiglie limitano al di sotto di quello che chiedi, perché lo fa la loro
configurazione di riferimento a monte. SSD si ferma a 200, la segmentazione di
istanze di RTMDet a 100 e FCOS al proprio limite di rilevamenti per immagine.
Alzare `max_det` oltre quei valori non ha effetto.

L'unico punto in cui `max_det` viene applicato centralmente e non per famiglia è
l'inferenza a tasselli, dove la lista unita viene troncata dopo la riconciliazione
dei tasselli.

## Filtraggio per classe

<code-tabs name="classes" />

`classes` prende una lista di id di classe e tiene solo le predizioni la cui
classe è in quella lista. Gli id indicizzano `result.names`, e il modo più sicuro
per ottenerne uno è leggere `names` da un risultato invece di dare per scontato
l'ordinamento di un dataset.

Il filtraggio avviene centralmente, dopo il postprocessing di ogni famiglia, nell'unico
imbuto attraversato da ogni percorso di predizione. Questo ha due conseguenze
che vale la pena conoscere. Funziona su ogni famiglia, comprese quelle senza NMS.
E filtra anche i payload allineati ai box, così maschere, keypoint e box orientati
vengono ridotti insieme a loro invece di restare disallineati.

Da riga di comando, `classes` accetta un intero singolo, una lista o una stringa
separata da virgole:

```bash
libreyolo predict model=LibreYOLO9s.pt classes=0 source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
libreyolo predict model=LibreYOLO9s.pt classes="[0,2,5]" source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
```

Filtrare non regala accuratezza. Un modello spende comunque il suo budget per
predire classi che poi scarti, e `max_det` viene applicato dalla famiglia prima
del filtro, quindi un'immagine affollata di classi indesiderate può raggiungere
il limite prima che si arrivi alla tua classe. Se succede, abbassa `conf` o alza
`max_det`.

## agnostic_nms

`agnostic_nms` viene accettato e non fa nulla. Passarlo solleva un avviso che
spiega che non ha alcun effetto e serve solo per compatibilità con la riga di
comando, e l'argomento viene scartato.

Non esiste una modalità di soppressione agnostica rispetto alla classe. Ogni
chiamata alla NMS nella libreria tiene conto della classe, quindi due box
sovrapposti di classi diverse sopravvivono entrambi, a qualsiasi `iou`. Dove
questo è un problema, filtra prima con `classes` oppure applica tu stesso la
soppressione fra classi su `result.boxes`.

## Cosa rifiuta predict

Due argomenti sollevano un'eccezione invece di limitarsi a un avviso: `visualize`
e `embed` sollevano entrambi `NotImplementedError`. Per gli embedding, carica il
modello con `task="embed"` e chiama `predict` o `embed` normalmente.

Qualsiasi cosa non riconosciuta solleva `TypeError` elencando le opzioni
supportate, così un errore di battitura fallisce subito invece di essere ignorato
in silenzio.

Questi vengono accettati, segnalati con un avviso e scartati: `agnostic_nms`,
`boxes`, `dnn`, `half`, `line_width`, `retina_masks`, `show_conf`, `show_labels`
e `verbose`.
