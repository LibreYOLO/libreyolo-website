---
title: Dataset
seo_title: I dataset di addestramento in LibreYOLO
description: >-
  Lo YAML del dataset che LibreYOLO legge, la struttura di cartelle che si
  aspetta, come funziona il download automatico e il comando doctor che
  controlla un dataset prima dell'addestramento.
lead: >-
  Un dataset LibreYOLO è un file YAML che indica una radice, i suoi split e i
  nomi delle classi. Tutto il resto, incluso dove si trovano i file delle
  etichette, è derivato da quel file per convenzione.
keywords:
  - formato dataset yolo
  - data.yaml
  - addestrare yolo dataset personalizzato
  - formato etichette yolo
  - dataset coco json
  - download automatico dataset
  - libreyolo doctor
  - controllo sbilanciamento classi
  - data leakage train val
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Funzionano un nome incluso, un percorso relativo o un percorso
        assoluto.

        model.train(data="coco8.yaml", epochs=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10
  doctor:
    - label: Controllare un dataset
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml
    - label: Far fallire un job di CI anche sugli avvisi
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml strict=true json=true
    - label: Saltare la passata di decodifica delle immagini
      language: bash
      code: >
        # Legge solo le etichette e lo YAML. I controlli su corruzione,
        duplicati

        # e leakage tra split hanno tutti bisogno dei pixel, quindi vengono
        saltati.

        libreyolo doctor my-dataset.yaml fast=true
    - label: Python
      language: python
      code: |
        from libreyolo import doctor

        report = doctor.diagnose("my-dataset.yaml", imgsz=640)

        for finding in report.findings:
            print(finding.severity.value, finding.check_id, finding.message)

        raise SystemExit(report.exit_code(strict=False))
source_hash: 9a12a0551c8b56e9
---

## Indicare a train quale dataset usare

`data=` accetta un percorso YAML o il nome di una config inclusa nel pacchetto.

<code-tabs name="train" />

Il nome viene risolto in un ordine fisso: un percorso assoluto che esiste, poi il
nome così com'è, relativo alla directory di lavoro, poi lo stesso nome con
`.yaml` in coda, poi la directory delle config incluse. Quando non corrisponde
niente, l'errore indica ogni directory in cui ha cercato ed elenca le config
incluse.

## Config incluse

Nel pacchetto sono incluse tredici config di dataset, sotto
`libreyolo/config/datasets/`.

| Config | Task | Note |
|---|---|---|
| `coco8.yaml` | detect | 8 immagini, si scarica da un semplice URL |
| `coco128.yaml` | detect | 128 immagini |
| `coco1000.yaml` | detect | 800 train, 200 val |
| `coco5000.yaml` | detect | 4000 train, 1000 val |
| `coco.yaml` | detect | COCO 2017 completo |
| `coco-val-only.yaml` | detect | solo val2017 |
| `coco8-pose.yaml` | pose | 8 immagini, keypoint COCO-17 |
| `coco-pose.yaml` | pose | keypoint di COCO 2017 |
| `ade20k.yaml` | semantic | 150 classi |
| `cityscapes.yaml` | semantic | 19 classi, download manuale |
| `cocostuff.yaml` | semantic | 182 classi, download manuale |
| `gopro.yaml` | restore | coppie per il deblurring |
| `sr8.yaml` | restore | coppie per la super-risoluzione |

Solo `coco8.yaml` e `coco128.yaml` contengono un semplice URL di download. Le
altre contengono un blocco di download Python, che richiede il consenso
esplicito descritto sotto, oppure si aspettano che i dati siano già su disco.

## Dove si trova un dataset su disco

La chiave YAML `path` indica la radice del dataset. Un `path` assoluto viene usato
così com'è scritto. Uno relativo viene cercato prima sotto la directory dei
dataset, poi accanto al file YAML stesso, e un dataset che sta per essere
scaricato finisce sotto la directory dei dataset.

Quella directory è `~/datasets`, sovrascritta dalla variabile d'ambiente
`LIBREYOLO_DATASETS_DIR`. Non esiste un file di impostazioni per questo.

## Le chiavi dello YAML

```yaml
path: my-dataset        # radice del dataset
train: images/train     # necessario per addestrare
val: images/val         # necessario per validare
test: images/test       # opzionale
nc: 3                   # opzionale; deve concordare con names
names:
  0: person
  1: helmet
  2: vest
download: https://example.com/my-dataset.zip   # opzionale
```

`train`, `val` e `test` accettano ciascuno una directory di immagini, un file
`.txt` che elenca un percorso immagine per riga, o una lista che mescola le due
cose. Le righe di una lista `.txt` possono essere relative, nel qual caso si
risolvono rispetto alla directory del file di lista stesso, e le righe che
iniziano con `#` vengono saltate.

`names` può essere una lista o una mappa con chiavi intere. `nc` è opzionale;
quando sono presenti entrambi e non concordano, il doctor lo segnala come errore.

## Struttura delle directory e file delle etichette

Rilevamento, segmentazione, posa e box orientati condividono tutti la stessa
struttura. Il percorso dell'etichetta è derivato dal percorso dell'immagine
riscrivendo un componente di directory `images` in `labels` e cambiando
l'estensione in `.txt`:

```text
my-dataset/
  images/train/0001.jpg   ->   labels/train/0001.txt
  images/val/0002.jpg     ->   labels/val/0002.txt
```

Viene riscritto solo un componente di percorso `images` intero, quindi una
directory chiamata `images_old` non viene toccata.

Una riga di rilevamento è fatta di cinque campi, tutti normalizzati a `[0, 1]`
rispetto a larghezza e altezza originali dell'immagine:

```text
<class_id> <cx> <cy> <w> <h>
```

Un file di etichette mancante o vuoto significa che l'immagine non ha oggetti, e
viene addestrata come sfondo invece di sollevare un errore. Una riga con più di
cinque campi viene letta come un poligono e il suo box diventa l'estensione del
poligono, così un'esportazione di segmentazione usata per addestrare al
rilevamento si carica senza errori. Il doctor segnala quante righe hanno seguito
questo percorso.

## Altri task

La segmentazione mantiene la stessa struttura con righe di poligono,
`<class_id> <x1> <y1> ... <xN> <yN>`, almeno tre punti. Una riga di rilevamento a
cinque campi è accettata e indica un'istanza rettangolare.

La posa aggiunge allo YAML `kpt_shape: [K, D]` e una permutazione `flip_idx`
opzionale. Ogni riga ha esattamente `5 + K * D` campi: il box, poi `K` keypoint
nella forma `x y` o `x y v`, con visibilità `0`, `1` o `2`.

I box orientati usano esattamente nove campi, la classe seguita da quattro punti
d'angolo in coordinate normalizzate. Nel file non viene salvato nessun angolo.

La segmentazione semantica abbina a ogni immagine una maschera a canale singolo
della stessa risoluzione, risolta sostituendo `masks_dir` (di default `masks`) a
`images`. Il valore di pixel `255` indica di ignorare il pixel. `label_mapping`
rimappa gli id di origine agli id di addestramento al momento del caricamento.

La classificazione usa un albero ImageFolder invece dei file di etichette, con
`train/` e `val/` che contengono ciascuno una directory per classe. La mappatura
da classe a indice segue i nomi delle cartelle in ordine alfabetico.

Il restauro abbina un input degradato a un target pulito di risoluzione identica
attraverso `input_dir` e `target_dir`. Profondità, normali alla superficie e bordi
abbinano ciascuno un'immagine a una mappa densa attraverso la propria chiave di
directory.

Il contratto completo per ogni task, incluse le convenzioni sulla scala della
profondità e la codifica PNG degli id di segmento panottici, è
`docs/dataset_schema.md` nel repository della libreria.

## COCO JSON nativo

Un file di annotazioni COCO JSON può essere usato direttamente. Aggiungi una mappa
`annotations`, e il percorso dello split diventa la radice delle immagini:

```yaml
path: my-dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

Quando `names` è presente, i nomi delle categorie del JSON devono corrispondergli,
e `names` definisce gli id delle etichette che il modello predice. Senza `names`,
gli id delle categorie COCO vengono ordinati e mappati in modo denso su `0..N-1`.

Questa modalità si aspetta una sola directory di immagini per split. Una lista di
percorsi o una lista di immagini `.txt` solleva un errore invece di caricare in
silenzio un insieme diverso.

## Download automatico

Un dataset è considerato presente quando il suo percorso `train` o `val` si
risolve in una directory non vuota o in un file esistente. Quando non è così, e lo YAML ha
una chiave `download`, è il valore a decidere cosa succede.

Un URL `http` o `https` viene scaricato e, se è uno zip, estratto nella radice del
dataset. Qualsiasi altra cosa viene trattata come uno script Python incorporato e
viene eseguita solo con `allow_download_scripts=True`. Senza quello, lo script
viene saltato con un avviso e l'addestramento continua con quello che c'è su
disco.

```bash
libreyolo train model=LibreYOLO9s.pt data=coco.yaml allow_download_scripts=true
```

Il flag è un gate sull'esecuzione di codice, non sulla rete. I download da URL
avvengono in ogni caso; sono i blocchi `download: |` ad averne bisogno. La CLI
stampa un avviso quando il flag è attivo, e il doctor non lo abilita mai.

## Controlla il dataset prima di addestrare

`libreyolo doctor` legge un dataset di rilevamento e segnala cosa andrebbe storto
prima che entri in gioco una GPU. Esce con codice 1 quando trova errori, quindi
funziona come gate di CI.

<code-tabs name="doctor" />

I controlli si dividono in sei famiglie:

| Famiglia | Cerca |
|---|---|
| `config` | `names` mancante, `nc` che non concorda con `names`, split mancanti o vuoti, nomi di classe duplicati |
| `files` | immagini senza file di etichette, etichette senza immagine, immagini mancanti elencate in uno split, collisioni tra nomi base |
| `labels` | righe malformate, id di classe fuori da `[0, nc)`, coordinate fuori da `[0, 1]`, box ad area zero, box minuscoli o enormi, bounding box duplicati, file di etichette identici byte per byte |
| `balance` | classi con zero o poche istanze, rapporto di sbilanciamento tra classi, classi presenti in un solo split, proporzione di immagini di sfondo |
| `images` | file non decodificabili, rotazione EXIF, disposizioni di canali insolite, immagini uniformi, duplicati esatti e approssimati |
| `splits` | la stessa immagine che compare in due split, in modo esatto o quasi identico |

`--only` e `--skip` accettano un id di controllo o un prefisso di famiglia, quindi
`skip=images,labels.tiny_object` è valido. `--fast` salta ogni controllo che ha
bisogno di decodificare i pixel, cioè le famiglie `images` e `splits`.

Ci sono due comportamenti da conoscere. `--strict` fa sì che anche gli avvisi,
oltre agli errori, facciano fallire il comando. E il doctor copre solo i dataset
di rilevamento: un dataset di posa, di segmentazione o di box orientati viene
rifiutato con un messaggio che indica cosa ha riconosciuto, invece di essere
controllato rispetto al contratto sbagliato.

## Correlati

- [Iperparametri](/docs/train/hyperparameters) per gli argomenti che `train()`
  accetta una volta che i dati sono a posto.
- [Validazione e metriche](/docs/train/validation) per valutare sullo split `val`
  o `test`.
