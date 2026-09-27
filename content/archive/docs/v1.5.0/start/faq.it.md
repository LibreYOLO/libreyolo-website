---
title: FAQ
seo_title: Domande frequenti su LibreYOLO
description: >-
  Risposte brevi alle domande che riguardano tutti i modelli LibreYOLO:
  hardware, licenze, pesi, dispositivi, addestramento, copertura
  dell'esportazione e la CLI.
lead: >-
  Risposte a domande che non sono specifiche di una famiglia di modelli. Tutto
  ciò che è specifico di una famiglia sta nella pagina di quella famiglia.
keywords:
  - libreyolo faq
  - libreyolo domande frequenti
  - libreyolo serve gpu
  - libreyolo licenza commerciale
  - dove si salvano i pesi libreyolo
  - libreyolo cli
  - libreyolo senza internet
last_verified: 1.5.0
source_hash: a729b43a6642f2a0
---

## Con quale modello conviene iniziare?

YOLOv9 per un rilevatore basato su CNN e RF-DETR per uno basato su transformer.
Entrambi stanno nel livello flagship, il che significa che le funzionalità
vengono progettate e validate su GPU con questi modelli prima che con qualsiasi
altro. Vedi [YOLOv9](/docs/models/yolov9) e [RF-DETR](/docs/models/rf-detr), oppure
[tutti i modelli](/docs/models) per il resto.

## Mi serve una GPU?

No. Ogni modello gira su CPU, e tutto quello che sta nel
[quickstart](/docs/quickstart) è scritto per funzionare lì. Una GPU cambia
quanto tempo richiedono l'addestramento e l'inferenza su video, non se
funzionano.

## Come fa LibreYOLO a scegliere il dispositivo?

Il valore predefinito è `device="auto"`, che usa CUDA quando PyTorch la segnala
come disponibile, poi Metal Performance Shaders quando sono disponibili, e la
CPU altrimenti. Per fissarlo, passa `device` al modello oppure a `predict`, `train`,
`val` ed `export`. Accetta `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`, un semplice
intero come `0`, o una stringa di cifre; gli ultimi due si espandono in
`cuda:<n>`.

`libreyolo checks` stampa la build di Torch, le sue versioni di CUDA e cuDNN, e
ogni GPU che riesce a vedere. Se quel comando non mostra CUDA, il wheel di
PyTorch è una build per CPU; [l'installazione](/docs/install) spiega come
sostituirlo.

## Dove finiscono i pesi scaricati?

In `weights/`, con percorso relativo alla directory di lavoro. Un riferimento a un modello
senza componente di directory si risolve lì e viene scaricato al primo uso; un
riferimento che include una directory viene usato esattamente come è scritto e
non viene mai scaricato. Vedi [checkpoint e pesi](/docs/weights).

## Posso eseguirlo senza accesso alla rete?

Sì. Scarica i checkpoint una volta su una macchina connessa, copia la directory
`weights/`, e niente toccherà più la rete. Funziona anche un percorso condiviso
in sola lettura, dato che un riferimento che contiene una directory viene preso
alla lettera. I dataset si risolvono sotto `~/datasets`, oppure sotto
`LIBREYOLO_DATASETS_DIR`.

## Posso usare LibreYOLO in ambito commerciale?

Il codice ha licenza MIT. I pesi preaddestrati sono una questione a parte:
possono ereditare le condizioni del progetto o del dataset da cui provengono, e
quelle condizioni non sono uniformi nemmeno all'interno di una stessa famiglia.
La licenza del repository Hugging Face specifico fa fede, e ogni pagina di
modello ha una sezione sulle licenze che la riporta. Dove i pesi sono soggetti a
restrizioni, LibreYOLO stampa la restrizione prima che inizi il download.

## Posso caricare un checkpoint di un altro progetto?

Di solito sì, passando il suo percorso a `LibreYOLO()`. I layout upstream
riconosciuti vengono convertiti al momento del caricamento, mantenendo numero e
nomi delle classi, e un checkpoint LibreYOLO viene scritto accanto al file di
origine. [Importare pesi esistenti](/docs/migrate) spiega cosa viene
riconosciuto e cosa richiede uno script di conversione.

## Perché train solleva NotImplementedError?

Perché quella famiglia offre solo l'inferenza, e l'eccezione ne indica il
motivo. La predizione, la validazione e, dove è supportata, l'esportazione
funzionano tutte; per quell'architettura non c'è un ciclo di addestramento in
LibreYOLO. Il livello di supporto nell'intestazione della pagina di un modello
te lo dice prima ancora che tu ci provi. Vedi [concetti fondamentali](/docs/concepts).

## Cosa restituisce val?

Un semplice dizionario, non un oggetto. Per il rilevamento le chiavi includono
`metrics/precision`, `metrics/recall`, `metrics/mAP50` e `metrics/mAP50-95`. Gli
altri task restituiscono le chiavi che hanno senso per loro, come
`metrics/accuracy_top1` per la classificazione o `metrics/PQ`, `metrics/SQ` e
`metrics/RQ` per la segmentazione panottica.

## Come eseguo su una cartella, un video o una webcam?

Passalo come sorgente. Il percorso di un file è una singola immagine, una
directory è ogni immagine al suo interno, il percorso di un video è un video, un
intero è l'indice di una webcam, e un URL RTSP, RTMP, TCP, UDP o HLS è uno
stream dal vivo. Un file `.streams` elenca più sorgenti in una volta sola. Le
sorgenti dal vivo richiedono `stream=True`, che restituisce un `Results` per
frame invece di costruire una lista; vale la pena usare lo stesso flag anche per
i video lunghi e le directory di grandi dimensioni. Solo gli URL delle pagine YouTube
richiedono un extra, `libreyolo[stream]`.

## Come tengo solo alcune classi?

Passa `classes` a `predict` con gli indici delle classi che vuoi, per esempio
`classes=[0, 2]`. `conf` imposta la soglia di confidenza, predefinita `0.25`, e
`max_det` limita i rilevamenti per immagine, predefinito `300`.

## La CLI usa flag o coppie chiave=valore?

Chiave e valore uniti da un segno di uguale, per ogni comando:

```bash
libreyolo predict model=yolo9-t source=my-image.jpg save=True
libreyolo train model=yolo9-t data=coco8.yaml epochs=50 imgsz=640
```

`model` accetta un percorso o un nome breve nella forma `family-size`,
opzionalmente con un suffisso di task, e `libreyolo models` elenca tutti quelli
validi. I comandi di diagnostica e di inventario accettano anche `--json`, che
stampa gli stessi dati come oggetto leggibile da una macchina su stdout.

## Ogni modello può essere esportato in ogni formato?

No. La copertura è per famiglia e per task, non è uniforme, e ogni formato ha il
proprio extra da installare. Ogni pagina di modello riporta la matrice di
esportazione della sua famiglia; la [sezione sull'esportazione](/docs/export)
tratta i formati veri e propri.

## Qual è la differenza tra segment, semantic e panoptic?

Tre task distinti. `segment` produce una maschera per ogni oggetto rilevato.
`semantic` etichetta ogni pixel con una classe e non separa nulla in istanze.
`panoptic` assegna a ogni pixel esattamente un'etichetta, unendo le cose
numerabili con la materia amorfa. Hanno ground truth diverso, campi di risultato
diversi e metriche diverse, e ogni famiglia supporta quelli che compaiono nella
sua lista di task.

## Come addestro sulle mie classi?

Scrivi un YAML del dataset con `train`, `val` e `names`. Le etichette stanno
accanto alle immagini in un albero `labels/` parallelo, un `.txt` per immagine,
con coordinate normalizzate. `nc` è facoltativo e, quando c'è, deve essere
coerente con `names`. Esegui prima `libreyolo doctor <data.yaml>`: controlla il
dataset alla ricerca di problemi ed esce con codice diverso da zero quando trova
errori, il che lo rende utilizzabile come gate di CI.

## Perché il caricamento stampa un avviso sui metadati?

Perché il checkpoint non porta con sé metadati v1.0 completi. Il caricamento
prosegue lungo un percorso di compatibilità, e l'avviso indica esattamente quali
chiavi mancano. Esegui `libreyolo metadata path=<file>` per vedere cosa c'è, e
consulta [checkpoint e pesi](/docs/weights) per sapere cosa richiede lo schema.

## Un import ha smesso di funzionare dopo un aggiornamento. Cosa è cambiato?

Due nomi di classe sono stati rinominati per coerenza: `LibreYOLORTDETR` è
diventato `LibreRTDETR` e `LibreYOLORFDETR` è diventato `LibreRFDETR`. I vecchi
nomi si risolvono ancora ed emettono un `DeprecationWarning` che punta a quello
nuovo, così il codice esistente continua a funzionare mentre lo aggiorni.
