---
title: Ensemble di detector
seo_title: Ensemble di detector in LibreYOLO
description: >-
  Esegui più detector sulla stessa immagine e fondi i loro box con la weighted
  boxes fusion o con NMS, anche quando i modelli hanno liste di classi diverse.
lead: >-
  LibreEnsemble esegue due o più detector sulla stessa immagine decodificata e
  fonde i loro box in un unico oggetto Results. Ogni membro conserva i propri
  pesi, le proprie soglie, il proprio device e la propria lista di classi.
keywords:
  - ensemble di modelli object detection
  - weighted boxes fusion
  - wbf python
  - combinare due detector
  - fondere bounding box
  - LibreEnsemble
  - ensemble detection python
  - min_votes
last_verified: 1.5.0
verification: >-
  Firme del costruttore e della chiamata, valori predefiniti, errori di
  validazione, unificazione dello spazio delle classi, conteggio dei voti e
  Results restituito letti da libreyolo/ensemble/model.py. Algoritmi di fusione
  e relativi argomenti da libreyolo/ops/fusion.py. Intento progettuale da
  docs/adr/0004-model-ensembling.md. Pattern d'uso verificati con
  tests/unit/test_ensemble.py e tests/unit/test_ops_fusion.py.
snippets:
  basic:
    - label: 'Due detector, fusi'
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        # I membri possono essere percorsi di checkpoint o modelli già caricati.
        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        result = ensemble(SAMPLE_IMAGE)
        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: Pesi e un requisito di voti
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ensemble = LibreEnsemble(
            ["LibreYOLO9s.pt", "LibreRFDETRs.pt"],
            weights=[1.0, 1.3],   # per convenzione, proporzionali alla mAP di validazione
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,          # tieni solo i box trovati da entrambi i membri
        )

        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes), "agreed detections")
    - label: Soglie per singolo membro
      language: python
      code: >
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE


        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])


        # Uno scalare vale per ogni membro; una lista viene letta membro per
        membro.

        result = ensemble(SAMPLE_IMAGE, conf=[0.3, 0.5], iou=0.5)

        print(len(result.boxes))
  external:
    - label: Includere un detector che LibreYOLO non ha caricato
      language: python
      code: |
        from libreyolo import ExternalDetector, LibreEnsemble, SAMPLE_IMAGE

        def my_detector(pil_image):
            # Restituisci (boxes, scores, labels): xyxy in pixel dell'immagine originale.
            return ([[100.0, 100.0, 200.0, 300.0]], [0.9], [0])

        external = ExternalDetector(my_detector, names={0: "person"})

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", external])
        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes))
  sources:
    - label: Le stesse sorgenti accettate da un singolo modello
      language: python
      code: |
        from libreyolo import LibreEnsemble

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        # Sostituisci clip.mp4 con un file video sul disco.
        for result in ensemble("clip.mp4", stream=True, vid_stride=2):
            print(result.frame_idx, len(result.boxes))
source_hash: 4f4c54c52b295795
---

## Che cos'è un ensemble

`LibreEnsemble` prende due o più detector, esegue ciascuno sulla stessa immagine
e fonde i loro box in un unico `Results`. È un costrutto che vive al momento
della predizione: non c'è nulla da addestrare e i membri restano modelli
indipendenti, validabili ed esportabili singolarmente.

Il rilevamento di oggetti è l'unico task supportato. Un membro con un task
diverso solleva `ValueError` alla costruzione, indicando l'indice del membro e
il suo task.

Entrambi i nomi sono importati in modo lazy, quindi non costano nulla finché non
li usi:

```python
from libreyolo import LibreEnsemble, ExternalDetector
```

## Costruirne uno

<code-tabs name="basic" />

```python
LibreEnsemble(
    members,
    *,
    weights=None,
    fusion="wbf",
    fusion_iou=0.55,
    min_votes=1,
)
```

`members` è una sequenza di due o più elementi. Una voce `str` o `Path` viene
caricata tramite `LibreYOLO()`; qualsiasi altra cosa deve essere callable ed
esporre un dict `names`. Meno di due solleva `ValueError`, e passare una stringa
nuda solleva `TypeError` invece di iterarne i caratteri.

`weights` vale `None` di default, cioè pesatura uniforme. I pesi forniti devono
essere uno per membro e strettamente positivi, così un peso zero solleva un
errore invece di scartare silenziosamente un membro. La convenzione documentata
è impostarli proporzionali alla mAP di validazione di ciascun membro.

`fusion_iou` vale `0.55` di default ed è l'IoU al quale i box di membri diversi
vengono raggruppati in cluster. È una soglia distinta dall'`iou` passato alla
chiamata, che è invece l'impostazione NMS di ciascun membro.

`min_votes` vale `1` di default, cioè un singolo membro basta a sostenere un
box. Alzandolo si tengono solo i cluster confermati da quel numero di membri
distinti. Deve essere un intero positivo non maggiore del numero di membri, e
viene limitato per classe al numero di membri che conoscono davvero quella
classe, così una classe su cui è stato addestrato un solo membro non viene
cancellata in silenzio.

## Metodi di fusione

Tre sono accettati per nome, e si accetta anche un callable.

| `fusion` | Comportamento |
|---|---|
| `"wbf"` | Weighted boxes fusion, sequenziale e fedele al paper. Il default |
| `"wbf_seeded"` | Weighted boxes fusion in un solo passaggio; una NMS per classe sceglie i seed dei cluster |
| `"nms"` | Concatena i box di tutti i membri, poi applica una NMS per classe |

La weighted boxes fusion media le coordinate di un cluster pesandole per la
confidenza, producendo un box che nessun singolo membro aveva proposto. Le due
varianti pesate coincidono quando i cluster non sono ambigui e possono differire
leggermente su catene di cluster sovrapposti. `"nms"` sceglie un sopravvissuto
invece di mediare, quindi i sopravvissuti mantengono i punteggi originali e i
pesi influenzano solo quale box vince. Poiché seleziona invece di raggruppare,
non può contare i voti: combinare `fusion="nms"` con `min_votes` maggiore di `1`
solleva `ValueError`.

La weighted boxes fusion riscala il punteggio di un cluster in base alla quota
di peso dei membri che lo ha sostenuto. Con due membri pesati allo stesso modo,
un box trovato da uno solo dei due mantiene metà del punteggio: `0.9` diventa
`0.45`. Una confidenza fusa può quindi scendere sotto il `conf` con cui ogni
membro è stato eseguito, perciò filtra sul punteggio fuso invece di dare per
scontato che la soglia del membro valga ancora.

## Membri con liste di classi diverse

I membri non devono per forza condividere la lista delle classi. I loro spazi di
etichette vengono unificati per nome, e ogni membro riceve una tabella di lookup
che rimappa i propri id di classe nell'unione. `ensemble.names` è quell'unione,
ed è ciò che porta con sé il `Results` restituito.

I box si fondono soltanto all'interno dello stesso nome di classe. Una classe
conosciuta da un solo membro passa senza fusione, e non viene penalizzata per
questo: il riscalamento del punteggio usa un denominatore per classe, quindi una
classe nota a un solo membro mantiene il proprio punteggio.

Una sovrapposizione parziale registra un warning che elenca le classi non
condivise da tutti i membri. È il warning da leggere con attenzione, perché un
checkpoint i cui nomi di classe sono segnaposto come `class_0` costruisce
un'unione disgiunta da quella di ogni altro membro, e non avviene alcuna fusione
tra membri.

Un membro che restituisce un id di classe fuori dai propri `names` solleva
`RuntimeError`.

## Detector esterni

<code-tabs name="external" />

`ExternalDetector(fn, names)` avvolge qualsiasi callable che prende
un'immagine PIL e restituisce `(boxes, scores, labels)`, con i box in formato
xyxy in pixel dell'immagine originale. Verifica l'arità, la forma dei box, la
corrispondenza delle lunghezze e che ogni id di classe compaia in `names`, e
applica da sé la soglia `conf`.

È così che un detector non caricato da LibreYOLO prende parte a una fusione.

## Come si chiama

<code-tabs name="sources" />

La firma della chiamata rispecchia quella di un singolo modello e accetta le
stesse sorgenti: immagini, cartelle, liste, video, cattura schermo, webcam e
stream di rete. Le sorgenti live richiedono `stream=True` per lo stesso motivo
per cui lo richiedono altrove.

| Argomento | Default | Note |
|---|---|---|
| `conf` | `0.25` | Per membro; uno scalare viene propagato, oppure uno per membro |
| `iou` | `0.45` | La soglia NMS di ciascun membro, non la soglia di fusione |
| `imgsz` | `None` | Una `list` viene letta per membro; un `int` o una tupla vengono propagati |
| `device` | `None` | Scalare o uno per membro, così i membri possono stare su device diversi |
| `classes` | `None` | Filtra il risultato fuso, sugli id di classe dell'unione |
| `max_det` | `300` | Si applica al risultato fuso |

Poiché per `imgsz` una `list` significa "per membro", `imgsz=[480, 640]` vuol
dire 480 per il primo membro e 640 per il secondo, mentre `imgsz=(480, 640)` è
un'unica dimensione rettangolare per tutti. È una distinzione su cui è facile
inciampare.

I membri vengono chiamati con un `max_det` di almeno 300 a prescindere da quello
che chiedi, così ciascuno lavora con margine e l'ensemble taglia una sola volta
alla fine.

L'immagine viene decodificata una volta sola e lo stesso oggetto viene passato a
ogni membro. `batch` è accettato per uniformità e ignorato; le immagini sono
elaborate in sequenza.

## Che cosa torna indietro

Un normale `Results`, lo stesso tipo che restituisce un singolo modello, con
`names` impostato sullo spazio delle classi unificato. Tutto quello che c'è in
[Lavorare con i risultati](/docs/predict/results) vale senza modifiche.

L'unica differenza è `result.speed`, che un ensemble popola davvero. Le sue
chiavi sono `member_0`, `member_1` e così via, più `fusion`, in millisecondi. È
l'unico punto della libreria in cui `speed` viene riempito.

Le righe con box o punteggi non finiti vengono scartate prima della fusione.
Quando i membri stanno su device diversi, la fusione gira sul device del primo
membro che ha restituito qualcosa.

## Che cosa un ensemble non può fare

`val()` ed `export()` sollevano entrambi `NotImplementedError` e ti rimandano ai
membri: valida ed esporta ognuno singolarmente. Non esiste proprio un metodo
`train`, quindi chiamarlo solleva `AttributeError`.

La mezza precisione non è gestita a livello di ensemble. `half=True` finisce
sullo stesso percorso no-op con warning di sempre; configura la precisione su
ogni membro.

Non c'è un'interfaccia a riga di comando per l'ensembling. È un'API Python.
