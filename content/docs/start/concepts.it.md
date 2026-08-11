---
title: Concetti fondamentali
seo_title: Concetti fondamentali di LibreYOLO
description: >-
  Come si incastrano task, famiglie di modelli, dimensioni e nomi dei file dei
  checkpoint in LibreYOLO, e cosa promette ogni livello di supporto.
lead: >-
  Quattro idee descrivono ogni modello di LibreYOLO: il task che esegue, la
  famiglia a cui appartiene, la dimensione all'interno di quella famiglia e il
  livello di supporto in cui si trova la famiglia. Il nome del file del
  checkpoint codifica le prime tre.
keywords:
  - concetti libreyolo
  - task libreyolo
  - famiglie di modelli libreyolo
  - nomi dei checkpoint libreyolo
  - livelli di supporto libreyolo
  - tipi di task computer vision
last_verified: 1.5.0
meta:
  - label: Schema del nome file
    value: 'Libre<FAMILY><size>[-<task>].pt'
    mono: true
  - label: Task canonici
    value: 17
  - label: Livelli di supporto
    value: 'Flagship, Core, Supported, Inference only, Museum, Sibling tier'
snippets:
  inspect:
    - label: Elencare le famiglie
      language: bash
      code: |
        # Task, dimensioni e risoluzioni di input di ogni famiglia registrata.
        libreyolo models
    - label: Un modello
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        print(model.family, model.size, model.task)
        print(model.input_size)
        print(model.nb_classes, model.names[0])
    - label: Scegliere un task
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Gli alias si normalizzano al confine dell'API: "keypoints" diventa

        # "pose", "det" diventa "detect", "semantic-segmentation" diventa
        "semantic".

        model = LibreYOLO("LibreYOLO9t.pt", task="det")

        print(model.task)
source_hash: 23d045463a6a8411
---

## Task

Un task è ciò che un modello restituisce. LibreYOLO ha diciassette nomi di task
canonici, e ognuno dà il nome al campo dell'oggetto `Results` che ne trasporta
l'output.

| Task | Restituisce |
|---|---|
| `detect` | Box allineati agli assi, con una classe e una confidenza |
| `segment` | Maschere per istanza, una maschera per ogni oggetto rilevato |
| `semantic` | Un'etichetta di classe per pixel, senza separazione tra le istanze |
| `panoptic` | Un'etichetta non sovrapposta per pixel, che unisce le cose numerabili con la materia amorfa |
| `pose` | Keypoint per istanza, con le righe allineate ai box |
| `classify` | Una probabilità su un insieme di etichette per l'intera immagine |
| `obb` | Box orientati, con un angolo di rotazione |
| `point` | Una coordinata dell'immagine per ogni rilevamento, invece di un box |
| `depth` | Una mappa densa di profondità inversa relativa |
| `normal` | Un campo denso di normali alla superficie a vettori unitari |
| `edge` | Una mappa densa di probabilità dei bordi |
| `restore` | Un'immagine RGB ripristinata, per deblurring, denoising o super-risoluzione |
| `matte` | Una mappa morbida del primo piano da 0 a 1, per rimuovere lo sfondo |
| `ocr` | Quadrilateri di testo con le trascrizioni, in ordine di lettura |
| `embed` | Un vettore normalizzato L2 il cui prodotto scalare misura l'accordo |
| `gaze` | Una direzione dello sguardo per ogni volto rilevato |
| `mesh` | Un corpo 3D in posa per ogni persona rilevata |

Sono questi i nomi che compaiono nei metadati dei checkpoint e nei nomi dei
file. Gli alias più familiari sono accettati ovunque si passi un task e vengono
normalizzati prima di qualsiasi altra cosa: `detection` e `det` diventano
`detect`, `keypoints` diventa `pose`, `cls` diventa `classify`, `deblur`,
`denoise` e `super-resolution` diventano tutti `restore`, `face-recognition` e
`reid` diventano `embed`. Un nome non riconosciuto solleva un errore invece di
ricadere silenziosamente su un valore predefinito.

`segment`, `semantic` e `panoptic` sono tre task diversi, non tre parole per uno
solo. Le maschere di istanza, le etichette per pixel e la mappa unita di cose e
materia hanno ground truth diverso, metriche diverse e campi di risultato
diversi.

## Famiglie di modelli

Una famiglia è una linea architetturale con il proprio codice di caricamento,
preprocessing e postprocessing. Ogni famiglia dichiara un identificatore
`FAMILY` come `yolo9`, `rfdetr` o `dfine`, i task che supporta e la risoluzione
di input per ogni dimensione che pubblica.

`LibreYOLO()` è una factory, non una classe. Dato un percorso carica il file,
identifica la famiglia dai metadati del checkpoint o, in mancanza di quelli,
dalle chiavi dei tensori stesse, e restituisce un'istanza del modello di quella
famiglia. Per questo cambiare detector è una modifica di una riga: l'oggetto che
torna indietro espone la stessa superficie `predict`, `train`, `val` ed `export`
e restituisce lo stesso tipo `Results`.

<code-tabs name="inspect" />

Una famiglia che serve più di un task di solito pubblica un checkpoint separato
per task, spesso con un insieme di dimensioni diverso per ciascuno; alcune
invece condividono un solo artefatto tra due task a runtime. In ogni caso i task
supportati sono una lista fissa, e chiederne uno fuori da quella lista solleva
un errore con la lista dei task supportati nel messaggio, invece di caricare
qualcosa di approssimativo.

L'elenco completo, con i benchmark per famiglia e i pesi pubblicati, è in
[tutti i modelli](/docs/models).

## Dimensioni

Una dimensione è una variante all'interno di una famiglia, scritta come un
codice minuscolo attaccato direttamente al prefisso della famiglia. Le lettere
comuni sono `n` per nano, `t` per tiny, `s` per small, `m` per medium, `l` per
large e `x` per xlarge, ma i codici sono specifici di ogni famiglia e diverse
famiglie usano qualcosa di completamente diverso: codici che prendono il nome
dal backbone come `r50` o `r101`, dove la dimensione è una profondità ResNet,
codici di scaling composto come da `b0` a `b3`, oppure un nome che identifica
l'unico checkpoint rilasciato. YOLOv9 usa `c` per compact dove altre famiglie
usano `l`.

La dimensione fissa anche la risoluzione di input, e per le famiglie con più
task la risoluzione può variare da task a task. Entrambe si leggono dalla
famiglia, mai per supposizione; `libreyolo models` le stampa.

## Nomi dei file dei checkpoint

Ogni file di pesi pubblicato segue un unico schema:

```text
Libre<FAMILY><size>[-<task>].pt
```

Il prefisso della famiglia è una stringa fissa per ogni famiglia, la dimensione
è in minuscolo e attaccata senza separatore, e il suffisso del task è preceduto
da un trattino. Il rilevamento non porta suffisso, seguendo la convenzione che i
checkpoint YOLO hanno sempre usato, quindi `LibreYOLO9t.pt` è un detector e
`LibreRFDETRn-seg.pt` è un modello di segmentazione della stessa famiglia.

| Task | Suffisso |
|---|---|
| `detect` | |
| `segment` | `-seg` |
| `semantic` | `-sem` |
| `panoptic` | `-panoptic` |
| `pose` | `-pose` |
| `classify` | `-cls` |
| `gaze` | `-gaze` |
| `obb` | `-obb` |
| `point` | `-point` |
| `depth` | `-depth` |
| `edge` | `-edge` |
| `normal` | `-normal` |
| `restore` | `-restore` |
| `matte` | `-matte` |
| `ocr` | `-ocr` |
| `embed` | `-embed` |
| `mesh` | `-mesh` |

Una famiglia che non ha nessun task privo di suffisso può esigere il suffisso,
così che un nome che ne è sprovvisto non venga accettato come checkpoint valido
per essa. Una famiglia che pubblica pesi addestrati su un dataset diverso da
quello predefinito aggiunge il nome del dataset come ulteriore suffisso, e
quella variante resta parte del nome del repository da cui il file viene
scaricato.

Tre livelli restano fuori da questo schema. Le famiglie di segmentazione
promptable, le famiglie vision-language e i detector a vocabolario aperto non
sono registrati nella factory dei checkpoint e non emettono alcun file
`Libre<FAMILY><size>.pt`. Il loro prefisso nomina invece uno snapshot scaricato
da Hugging Face o un checkpoint promptable, e lì le maiuscole del marchio
originale sono conservate di proposito.

## Come viene deciso il task

Quando più segnali potrebbero dare un nome al task, vengono consultati in un
ordine fisso e vince il primo presente: l'argomento `task` che hai passato, poi
il task registrato nei metadati del checkpoint, poi il suffisso di task nel nome
del file, poi il task predefinito della famiglia. Il risultato viene verificato
rispetto ai task supportati dalla famiglia prima che il modello venga costruito,
così una discrepanza fallisce al momento del caricamento invece di produrre
output sbagliati più avanti.

## Livelli di supporto

Le famiglie sono iscritte a esattamente un livello. Un livello è
un'affermazione sull'attenzione ingegneristica, non sull'accuratezza: ti dice
dove atterra per prima una nuova funzionalità e cosa viene tenuto verde.

| Livello | Cosa significa |
|---|---|
| Flagship | Le funzionalità vengono progettate e validate a fondo su GPU qui per prime |
| Core | Detector addestrabili principali. Le funzionalità seguono i flagship nella stessa ondata di release |
| Supported | Famiglie addestrabili di supporto. Tenute verdi in CI, le funzionalità arrivano quando capita |
| Inference only | Predizione, validazione ed esportazione. Le funzionalità di addestramento non si applicano |
| Museum | Un reperto congelato. Solo correzioni di bug |
| Sibling tier | Una superficie di prodotto separata, con la propria factory e il proprio contratto |

Ogni pagina di modello riporta nell'intestazione il livello della sua famiglia.
Le due famiglie flagship sono [YOLOv9](/docs/models/yolov9) per i detector CNN e
[RF-DETR](/docs/models/rf-detr) per i detector transformer; parti da lì a meno
che tu non abbia un motivo per non farlo.

Inference only dice cosa manca, cioè un ciclo di addestramento in LibreYOLO. La
predizione, la validazione e, dove la famiglia lo supporta, l'esportazione
funzionano tutte. Chiamare `train()` su una famiglia di questo tipo solleva
`NotImplementedError` indicandone il motivo.
