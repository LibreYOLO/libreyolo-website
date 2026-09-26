---
title: Formati dei dataset
seo_title: Formati dei dataset LibreYOLO per ogni task
description: >-
  Il contratto dei file di dataset per ogni task canonico: chiavi YAML,
  struttura delle cartelle, righe di etichetta, convenzioni per maschere e
  mappe, e il loader che legge ciascun formato.
lead: >-
  Questa pagina rispecchia il contratto dei file di dataset descritto nella
  documentazione della libreria stessa, docs/dataset_schema.md. Copre le chiavi
  YAML e il layout su disco che ogni task canonico si aspetta.
keywords:
  - formato dataset libreyolo
  - formato label yolo
  - data.yaml
  - dataset maschere segmentation
  - coco panoptic format
  - dataset depth estimation
  - pose kpt_shape
last_verified: 1.5.0
verification: >-
  Rispecchia docs/dataset_schema.md nel repository libreyolo alla v1.5.0, con i
  nomi dei loader verificati su libreyolo/data/.
snippets:
  usage:
    - label: Leggere una riga di etichetta detect
      language: python
      code: >
        from libreyolo.data import parse_yolo_label_line


        # class_id cx cy w h, normalizzati in [0, 1]

        row = parse_yolo_label_line("0 0.5 0.5 0.25 0.5", 640, 480,
        num_classes=80)


        # (class_id, x1, y1, x2, y2, area) in pixel

        print(row)
source_hash: a8282c079624044d
---

## YAML comune

Si applica a `detect`, `segment`, `pose` e `obb`.

| Chiave | Obbligatoria | Significato |
|---|---|---|
| `path` | | Radice del dataset |
| `train` | Per l'addestramento | Immagini di addestramento |
| `val` | Per la validazione | Immagini di validazione |
| `test` | | Immagini di test |
| `names` | Sì | Elenco delle classi, o una mappa con chiavi intere |
| `nc` | | Numero di classi; deve corrispondere a `names` quando è presente |
| `download` | | Istruzioni di download; gli script Python richiedono un consenso esplicito |
| `annotations` | | Split verso il file JSON COCO nativo, per detect, segment e obb |

`train`, `val` e `test` possono essere directory di immagini, file `.txt` con
elenchi di immagini, o liste di questi. I percorsi delle etichette seguono una
sola sostituzione:

```text
images/.../image.jpg -> labels/.../image.txt
```

Per un dataset in JSON COCO nativo, `annotations` associa uno split al suo file
JSON e il percorso dello split fornisce la radice delle immagini:

```yaml
path: dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

Quando `names` è presente, i nomi delle categorie del JSON COCO nativo devono
corrispondere ai nomi delle classi dello YAML, e quei nomi definiscono gli ID
delle etichette del modello. Senza `names`, gli ID delle categorie COCO vengono
ordinati e mappati in modo denso su `0..N-1`.

Uno YAML di dataset non contiene una chiave `task`. La selezione esplicita di
modello e task ha la precedenza.

Regole comuni a ogni file di etichette testuale:

- un file di etichette `.txt` per immagine;
- un file di etichette mancante o vuoto significa nessun oggetto;
- `class_id` è un intero in `0..nc-1`;
- le coordinate sono float normalizzati finiti in `[0, 1]`;
- le coordinate sono relative alla larghezza e all'altezza originali dell'immagine;
- le righe non contengono confidenza né ID di tracking.

<code-tabs name="usage" />

## detect

Esattamente cinque campi per riga:

```text
<class_id> <cx> <cy> <w> <h>
```

`cx cy w h` è un box normalizzato allineato agli assi, e `w` e `h` devono
essere positivi.

## segment

Una riga con un poligono:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

`N` è almeno 3, il numero di coordinate dopo `class_id` deve essere pari e il
poligono non deve essere degenere. Viene accettata anche una riga di detection
a cinque campi, che rappresenta un segmento rettangolare.

## pose

Lo YAML aggiunge `kpt_shape`, che è obbligatoria ed è `[K, 2]` o `[K, 3]`, e
l'opzionale `flip_idx`, una permutazione intera di `0..K-1`.

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

Il numero di campi è esattamente `5 + K * D`, dove `D` è il secondo valore di
`kpt_shape`. Le coordinate dei keypoint sono normalizzate. La visibilità `v`,
quando è presente, vale `0`, `1` o `2`.

## obb

Esattamente nove campi:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

I quattro punti sono coordinate immagine normalizzate in `[0, 1]` e formano un
rettangolo orientato non degenere. Nel file di etichette non viene memorizzato
alcun angolo.

Il parser canonico è rigoroso di default e rifiuta le coordinate fuori
intervallo. L'ingestione da parte del dataset e della validazione può troncare
le coordinate a `[0, 1]` per etichette altrimenti valide sul bordo del crop, e
rifiuta comunque i box degeneri. Il parsing tiene conto del task: nove campi
significano `obb` solo in modalità `obb`, mentre in modalità `segment` possono
essere un poligono a quattro punti.

Internamente, i vertici normalizzati vengono convertiti nella forma canonica
`xywhr`, con l'angolo in radianti che rappresenta la rotazione del lato della
larghezza attorno al centro del box. I risultati pubblici espongono le
detection OBB come righe `xywhr, conf, cls`.

Il caricamento OBB da JSON COCO nativo accetta le annotazioni in questo ordine
di priorità: `obb` come otto vertici nello spazio dei pixel; `obb` come
`[cx, cy, w, h, angle]` con l'angolo in radianti; un poligono o una RLE
`segmentation` COCO, riadattati al rettangolo di area minima; e un `bbox` COCO,
letto come allineato agli assi e canonicalizzato.

Mosaic e mixup sono disattivati per l'addestramento OBB finché non esisterà una
data augmentation OBB consapevole dei vertici.

Il parser canonico delle righe è `libreyolo.data.parse_yolo_obb_label_line`.

## semantic

Ogni immagine è accompagnata da una maschera densa a canale singolo in un
formato lossless, tipicamente PNG, invece che da un file `.txt`:

```text
images/.../image.jpg -> <masks_dir>/.../image.png
```

La maschera è a canale singolo, e i PNG in modalità palette vengono letti come
indici della palette. Ogni valore di pixel è un ID di classe in `0..nc-1`, il
valore di pixel `255` significa ignora ed è escluso da loss e metriche, e la
risoluzione della maschera deve essere uguale a quella dell'immagine.

Due chiavi YAML opzionali si aggiungono al contratto comune. `masks_dir` è il
nome della directory delle maschere sostituito a `images` in ogni percorso
immagine, con valore predefinito `masks`. `label_mapping` è un rimappaggio
`{source_id: train_id}` applicato ai valori dei pixel della maschera al momento
del caricamento, dove i valori sorgente non mappati diventano ignora e i train
ID devono ricadere in `0..nc-1`.

Quando `masks_dir` è omesso, le maschere vengono rasterizzate al caricamento a
partire dalle etichette poligonali `segment` risolte tramite la convenzione da
`images` a `labels`, e una classe `background` viene aggiunta dopo le classi
degli oggetti, quindi `nc` cresce di uno.

Loader canonico: `libreyolo.data.SemanticDataset`.

## panoptic

LibreYOLO adotta il formato COCO-panoptic alla lettera (Kirillov et al., CVPR
2019). Non esiste un formato panoptic specifico di LibreYOLO.

Un PNG RGB per immagine, alla risoluzione dell'immagine, codifica nel proprio
colore l'ID del segmento di ogni pixel:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

Ogni pixel appartiene esattamente a un segmento e i segmenti non si
sovrappongono mai. L'ID di segmento `0`, il nero RGB, è void: pixel non
etichettati esclusi dalla metrica.

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1, "supercategory": "person"}]
}
```

`annotations[].file_name` indica il PNG con gli ID di segmento dentro
`panoptic_dir`, e `segments_info[].id` corrisponde a un valore in quel PNG.
`iscrowd` marca le regioni di gruppo: non sono mai falsi negativi, e una
predizione che ne copre gran parte non è un falso positivo.

La distinzione tra thing e stuff è una proprietà per categoria. `isthing` sta su
`categories`, mai su `segments_info`.

I valori `category_id` di COCO-panoptic sono gli ID grezzi del dataset e sono
tipicamente non contigui. I modelli predicono valori contigui `0..nc-1`, quindi
gli ID grezzi vengono rimappati attraverso i `names` dello YAML per nome di
categoria, la stessa regola che segue il loader detect per il JSON COCO nativo.
Una categoria del JSON assente da `names` è un errore invece che una rimozione
silenziosa, perché altrimenti conterebbe come un falso negativo permanente.

```yaml
path: coco
val: images/val2017
annotations:
  val: annotations/panoptic_val2017.json
panoptic_dir:
  val: annotations/panoptic_val2017
names: {0: person, 1: bicycle, 132: rug-merged}
```

`annotations` e `panoptic_dir` accettano un singolo percorso oppure una mappa
per split.

La validazione riporta la Panoptic Quality, calcolata alla risoluzione del
ground truth e mediata sulle categorie che compaiono, poi divisa in `PQ_things`
e `PQ_stuff`. Il matching è univoco: un segmento predetto e uno di ground truth
della stessa categoria corrispondono quando la IoU è superiore a 0.5.

Loader canonico: `libreyolo.data.PanopticDataset`.

## depth

Ogni immagine è accompagnata da una mappa di profondità densa a canale singolo:

```text
images/.../image.jpg -> <depths_dir>/.../image.png
```

La mappa è un PNG o un TIF a canale singolo, oppure un file `.npy`, alla
risoluzione dell'immagine. I valori sono profondità semplice in un'unità
coerente per il dataset. Valori zero, negativi, NaN e infiniti marcano i pixel
non validi e sono esclusi da loss e metriche.

| Chiave | Predefinito | Significato |
|---|---|---|
| `depths_dir` | `depths` | Directory della profondità sostituita a `images` |
| `depth_stem_suffix` | | Suffisso aggiunto allo stem dell'immagine; quando è omesso vengono provati sia lo stesso stem sia un suffisso `_depth` |
| `depth_mask_suffix` | `_mask` | Suffisso per una maschera di validità; valori della maschera minori o uguali a zero, NaN e infiniti invalidano il pixel di profondità |
| `depth_scale` | `256.0` | Divisore per le mappe di profondità di tipo intero, la comune convenzione PNG a 16 bit |

Le mappe `.npy` in float sono usate così come sono e non applicano
`depth_scale`.

Loader canonico: `libreyolo.data.DepthDataset`.

## edge

Ogni immagine RGB è accompagnata da una mappa lossless a canale singolo con lo
stesso stem e da una maschera di validità opzionale:

```text
images/val/scene.jpg -> edges/val/scene.png
                     -> masks/val/scene.png
```

La mappa è un PNG o un TIF a canale singolo, non una visualizzazione RGB, alla
risoluzione dell'immagine. Le mappe intere vengono divise per il massimo del
loro dtype; le mappe float devono già essere finite e in `[0, 1]`. `0` significa
non-edge e `1` significa edge. I pixel della maschera opzionale sono validi
quando sono diversi da zero. Il ridimensionamento usa l'interpolazione
nearest-neighbor per target e maschere, e i pixel di padding non sono validi e
non contribuiscono alla validazione.

| Chiave | Predefinito | Significato |
|---|---|---|
| `edges_dir` | `edges` | Directory delle mappe di edge sostituita a `images` |
| `edge_stem_suffix` | | Suffisso aggiunto agli stem delle immagini |
| `edge_extension` | `.png` | Estensione lossless del target |
| `edge_invert` | | Imposta true quando le mappe sorgente hanno edge neri su bianco |
| `masks_dir` | `masks` | Directory opzionale delle maschere di validità |

```yaml
path: edge-dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

La validazione assottiglia le predizioni continue con una non-maximum
suppression del gradiente a quattro direzioni e riporta le F-measure ODS e OIS
su una scansione configurabile di soglie. I pixel predetti e quelli di ground
truth vengono associati uno a uno entro `edge_max_dist * image_diagonal`, con
una tolleranza normalizzata predefinita di `0.0075`.

Loader canonico: `libreyolo.data.EdgeDataset`. Il loader riguarda solo il
formato: non scarica né ridistribuisce i dati dei benchmark.

## normal

Ogni immagine è accompagnata da un PNG a tre canali a 16 bit con lo stesso
stem, più una maschera di validità opzionale con lo stesso stem:

```text
images/val/room.jpg -> normals/val/room.png
                    -> masks/val/room.png
```

Il PNG è esattamente `uint16` a tre canali, con i canali memorizzati come RGB,
alla risoluzione dell'immagine. Decodifica con `n = png / 65535 * 2 - 1`, poi
rinormalizza ogni vettore. I vettori decodificati usano il sistema di
riferimento della camera di OpenCV, `+x` a destra, `+y` in basso, `+z` verso la
scena, e sono rivolti verso la camera. La maschera opzionale è un PNG a canale
singolo dove un valore diverso da zero significa valido; senza maschera, ogni
vettore decodificato finito e non nullo è valido. I pixel target non validi e
quelli di padding sono rappresentati internamente da `(0, 0, 0)`. Il
ridimensionamento interpola le tre componenti in modo bilineare e poi
rinormalizza, le maschere di validità usano l'interpolazione nearest-neighbor,
e un flip orizzontale nega anche la componente x.

| Chiave | Predefinito | Significato |
|---|---|---|
| `normals_dir` | `normals` | Directory delle mappe di normali sostituita a `images` |
| `masks_dir` | `masks` | Directory opzionale delle maschere di validità |

La validazione riporta l'errore angolare medio e mediano in gradi e la
percentuale di pixel validi entro 11.25, 22.5 e 30 gradi.

Loader canonico: `libreyolo.data.NormalDataset`.

## restore

Ogni immagine di input degradata è accompagnata da un target RGB pulito:

```text
inputs/.../image.jpg -> targets/.../image.jpg
```

Input e target sono file immagine compatibili con RGB e le loro risoluzioni
devono corrispondere esattamente. La validazione mantiene la risoluzione nativa
e applica solo il padding necessario a impilare un batch, e le metriche sono
calcolate sul canvas dell'immagine originale. L'addestramento applica un crop e
un flip orizzontale accoppiati alla coppia input-target.

| Chiave | Predefinito | Significato |
|---|---|---|
| `input_dir` | `inputs` | Directory degli input degradati usata nei percorsi degli split |
| `target_dir` | `targets` | Directory dei target puliti sostituita a `input_dir` |
| `target_stem_suffix` | | Suffisso aggiunto allo stem dell'input prima della ricerca del target |
| `target_stem_suffixes` | | Forma a lista di `target_stem_suffix` |
| `degradation` | | Etichetta di metadati come `deblur` o `denoise` |
| `dataset` | | Etichetta di dataset o di provenienza |

I campi YAML relativi alle classi sono segnaposto dello schema: usa `nc: 1` e
`names: {0: image}`. I modelli restore espongono `Results.restored`, non delle
detection.

Loader canonico: `libreyolo.data.RestoreDataset`.

## matte

Ogni immagine RGB è accompagnata da un matte di ground truth a canale singolo
con lo stesso stem, dove 0 è sfondo e 255 è primo piano:

```text
images/subject.jpg -> mattes/subject.png
```

Sono accettati due layout. Una directory radice che contiene `images/` e una
directory dei matte, rilevata automaticamente tra `mattes/`, `matte/`, `gt/`,
`masks/`, `mask/` e `alpha/`, passata come `data=`. Oppure uno YAML con `path`
più `val_images` e `val_mattes` per split, e facoltativamente `train_images` e
`train_mattes`, ciascuno relativo a `path` oppure assoluto.

Il matte è in scala di grigi e viene letto come opacità in `[0, 1]`, e viene
ridimensionato al canvas della predizione con interpolazione bilineare quando le
forme differiscono. Le metriche sono MAE e S-measure (Fan et al., ICCV 2017) sul
canvas dell'immagine originale, con la S-measure come fitness per il miglior
checkpoint.

I campi YAML relativi alle classi sono segnaposto dello schema: usa `nc: 1` e
`names: {0: matte}`. I modelli matte espongono `Results.matte`.

In questa versione la validazione è solo di inferenza. Risolutore canonico delle
coppie: `libreyolo.data.matte_dataset.resolve_matte_pairs`.

## ocr

Le etichette sono un file JSONL per split, un oggetto JSON per immagine:

```text
images/val/receipt.jpg -> labels/val.jsonl
```

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon` è un quadrilatero a quattro punti in coordinate pixel assolute,
ordinate in alto a sinistra, in alto a destra, in basso a destra, in basso a
sinistra. Le regioni con testo illeggibile usano `"text": "###"`, la convenzione
do-not-care di ICDAR: sono escluse dallo scoring del riconoscimento, e le
predizioni che le sovrappongono vengono ignorate invece che penalizzate nel
matching del rilevamento.

Le metriche sono l'hmean di rilevamento con matching uno a uno dei poligoni
sopra IoU 0.5, l'F1 end-to-end che richiede sia IoU sopra 0.5 sia una
trascrizione esatta dopo normalizzazione NFKC e rimozione degli spazi, con
distinzione tra maiuscole e minuscole, e 1-NED sulle coppie associate. La
fitness per il miglior checkpoint è l'F1 end-to-end.

Sono accettati due layout: una directory radice che contiene `images/<split>/` e
`labels/<split>.jsonl`, passata come `data=`, oppure uno YAML con `path` più i
nomi opzionali delle directory `images` e `labels`.

I campi YAML relativi alle classi sono segnaposto dello schema: usa `nc: 1` e
`names: {0: text}`. I modelli OCR espongono `Results.ocr`.

In questa versione la validazione è solo di inferenza. Risolutore canonico dei
campioni: `libreyolo.data.ocr_dataset.resolve_ocr_samples`.

## classify

Un albero di directory in stile ImageFolder, non file di etichette:

```text
dataset_root/
  train/
    class_a/*.jpg
    class_b/*.jpg
  val/
    class_a/*.jpg
    class_b/*.jpg
```

`train/` è obbligatoria per l'addestramento e definisce la mappatura tra classe
e indice in base al nome della cartella, in ordine alfabetico. `val/` è
obbligatoria per la validazione. `test/` può essere presente, ma i comandi train
e val predefiniti non la usano. Gli split diversi da quello di addestramento
devono contenere gli stessi nomi di cartelle di classe dell'insieme di classi
atteso da train o dal checkpoint. Le estensioni immagine supportate sono
definite in `libreyolo.data.classify_dataset.IMAGE_EXTENSIONS`.

## gaze e point

Per `gaze` non è implementato alcun contratto di file di dataset per
addestramento o validazione.

`point` è un task di output del modello più che uno schema di etichette del
dataset. Le famiglie point possono adattare internamente etichette esistenti,
per esempio derivando i centri degli oggetti dalle righe dei box, ma non è
definito un formato di etichette testuali solo per i punti.
