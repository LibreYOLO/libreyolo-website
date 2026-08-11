---
title: Tipi di Results
seo_title: Riferimento dell'oggetto Results di LibreYOLO
description: >-
  Ogni payload che un oggetto Results di LibreYOLO può portare, uno slot per
  forma di task: boxes, masks, keypoints, probs, obb, depth, ocr, embeddings e
  altri dieci.
lead: >-
  Results è l'unico tipo di ritorno per immagine di ogni modello LibreYOLO.
  Porta diciotto slot di payload opzionali, uno per forma di task, e popola solo
  quelli che il modello ha prodotto.
keywords:
  - oggetto Results libreyolo
  - Results.boxes
  - Results.masks
  - Results.probs
  - Results.depth_map
  - risultati detection in json python
  - ottenere coordinate bounding box python
last_verified: 1.5.0
verification: >-
  Nomi degli slot, shape, proprietà e valori predefiniti letti da
  libreyolo/utils/results.py alla v1.5.0. Semantica citata dalle docstring delle
  classi di payload.
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        print(result.orig_shape, result.path)
        print(result.boxes.xyxy)
        print(result.boxes.conf)
        print(result.names[int(result.boxes.cls[0])])
  convert:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        # Ogni payload si sposta insieme agli altri.
        result = result.cpu().numpy()

        # Le righe, come dict semplici, poi come JSON.
        print(result.summary()[:1])
        print(result.to_json())
source_hash: 16f654364ae6448a
---

## L'oggetto Results

Un `Results` descrive una singola immagine. Una sorgente con una sola immagine
ne restituisce uno, una sorgente a lista o una directory restituisce una lista,
e `stream=True` restituisce un generatore che li produce uno alla volta.

| Attributo | Tipo | Significato |
|---|---|---|
| `orig_shape` | `(int, int)` | Altezza e larghezza dell'immagine originale |
| `path` | `str` | Percorso di origine quando l'input proviene dal disco |
| `names` | `dict[int, str]` | Da indice di classe a nome della classe |
| `speed` | `dict[str, float]` | Millisecondi per fase |
| `track_id` | tensore | ID di tracking quando il risultato proviene da `track()` |
| `frame_idx` | `int` | Indice del frame per sorgenti video e stream |
| `restore_scale` | `int` | Fattore di upscaling tra output e input di un risultato di restore; `1` in tutti gli altri casi |

<code-tabs name="usage" />

## Gli slot di payload

Ogni slot è `None` a meno che il modello non lo abbia prodotto. Lo slot che una
famiglia riempie è deciso dal suo task.

| Slot | Classe | Task |
|---|---|---|
| `boxes` | `Boxes` | detect |
| `masks` | `Masks` | segment |
| `keypoints` | `Keypoints` | pose |
| `probs` | `Probs` | classify |
| `obb` | `OBB` | obb |
| `gaze` | `Gaze` | gaze |
| `points` | `Points` | point |
| `semantic_mask` | `SemanticMask` | semantic |
| `panoptic` | `PanopticSegmentation` | panoptic |
| `depth_map` | `DepthMap` | depth |
| `normal_map` | `NormalMap` | normal |
| `edges` | `EdgeMap` | edge |
| `restored` | `RestoredImage` | restore |
| `matte` | `Matte` | matte |
| `ocr` | `OCRRegions` | ocr |
| `embeddings` | `Embeddings` | embed |
| `identities` | `Identities` | embed, con una gallery |
| `meshes` | `Meshes` | mesh |

`result.normals` è un alias in lettura e scrittura per `result.normal_map`.

Più slot possono essere valorizzati contemporaneamente. Un modello di
segmentazione riempie sia `boxes` sia `masks`; un modello di gaze riempie
`boxes` con i box dei volti e `gaze` con gli angoli; un modello di mesh riempie
`boxes` con i box delle persone e `meshes` allineato per riga a essi.

## Boxes

I box di rilevamento per una singola immagine.

| Membro | Restituisce |
|---|---|
| `xyxy` | Coordinate degli angoli in pixel dell'immagine originale |
| `xywh` | Centro e dimensioni in pixel |
| `xyxyn` | Angoli normalizzati in `[0, 1]` |
| `xywhn` | Centro e dimensioni normalizzati in `[0, 1]` |
| `conf` | Confidenza per ogni box |
| `cls` | Indice di classe per ogni box |
| `id` | ID di tracking per ogni box, oppure `None` |
| `is_track` | `True` quando sono presenti gli ID di tracking |
| `data` | Il tensore compattato |

`with_id(id)` e `with_orig_shape(orig_shape)` restituiscono un nuovo `Boxes`
con quel campo sostituito.

## Masks

Le maschere di istanza per una singola immagine. `data` è il tensore delle
maschere; `xy` restituisce i contorni per istanza in pixel e `xyn` li
restituisce normalizzati.

## Keypoints

I keypoint della posa, allineati per riga con `boxes`. `xy` è la coppia di
coordinate per ogni keypoint e `xyn` la coppia normalizzata. `conf` è il terzo
canale quando i dati ne portano uno, altrimenti `None`. `has_visible` è un
array booleano, vero dove `conf > 0`, e tutto vero quando non c'è un canale di
confidenza.

## Points

La localizzazione a punti per una singola immagine. `data` ha shape `(N, 4)`
con righe `x, y, class, confidence`. Le coordinate sono pixel assoluti; `xy`,
`cls` e `conf` suddividono le colonne, e `xyn` normalizza le coordinate.

## Probs

I punteggi di classificazione. `top1` è l'indice vincente, `top5` i cinque
indici migliori, e `top1conf` e `top5conf` i rispettivi punteggi.

## OBB

I box orientati. `data` contiene 7 o 8 valori per riga: `xywhr`, un ID di
tracking opzionale, poi confidenza e classe.

| Membro | Restituisce |
|---|---|
| `xywhr` | Centro, dimensioni e rotazione in radianti |
| `xyxyxyxy` | I quattro angoli in pixel |
| `xyxyxyxyn` | I quattro angoli normalizzati |
| `xyxy` | Involucro allineato agli assi in pixel |
| `conf`, `cls`, `id`, `is_track` | Come in `Boxes` |

## Gaze

Gli angoli dello sguardo per ogni volto in radianti, shape `(N, 2)`, allineati
per riga con i box dei volti in `boxes`. La colonna 0 è il pitch e la colonna 1
è lo yaw, secondo la convenzione L2CS: uno yaw positivo ruota lo sguardo verso
la sinistra del soggetto e un pitch positivo lo ruota verso il basso.
`pitch_deg` e `yaw_deg` convertono in gradi, e `direction_3d` restituisce il
vettore direzione unitario.

## SemanticMask

Mappa semantica densa, shape `(H, W)` di ID di classe interi sul canvas
dell'immagine originale. `255` è il valore di ignore e non conta mai come
classe (`SemanticMask.IGNORE_INDEX`). `classes` elenca gli ID di classe
presenti, e `class_mask(class_id)` restituisce la maschera booleana di una
singola classe.

## PanopticSegmentation

Ogni pixel riceve esattamente un segmento non sovrapposto, unificando le
regioni stuff e le istanze thing. `data` è una mappa `(H, W)` di ID di segmento
interi; l'ID di segmento `0` è senza etichetta
(`PanopticSegmentation.IGNORE_INDEX`). `segments_info` è una lista di dict, uno
per segmento, ciascuno con almeno `{"id": int, "category_id": int}`, dove `id`
corrisponde a un valore nella mappa e `category_id` indicizza `names`.
`segment_ids` elenca gli ID presenti e `segment_mask(segment_id)` restituisce
la maschera booleana di un singolo segmento.

La distinzione thing/stuff è una proprietà della categoria, non del segmento.
Un payload può denormalizzarla su ogni segmento come `"isthing": bool`, e
quando lo fa il valore deve concordare con la mappa a livello di categoria.

## DepthMap

Mappa densa di profondità inversa relativa, shape `(H, W)` di float sul canvas
dell'immagine originale. Valori più alti significano più vicino alla camera. I
valori sono relativi, non metri in scala metrica. `min`, `max` e `mean` sono
calcolati sui valori finiti, e `normalized()` riscala la mappa in `[0, 1]`.

## NormalMap

Campo denso di normali alla superficie, float32 `(H, W, 3)` sul canvas
dell'immagine originale, nel sistema di riferimento della camera OpenCV: `+x` a
destra, `+y` in basso, `+z` verso l'interno della scena. Le normali sono
rivolte verso la camera, quindi una superficie fronto-parallela è `(0, 0, -1)`.
Ogni pixel è un vettore unitario. `assert_normalized(atol=1e-4)` verifica
questa invariante.

## EdgeMap

Mappa densa di probabilità dei bordi, float32 `(H, W)` sul canvas
dell'immagine originale, dove `0` è non-bordo e `1` è bordo. La mappa continua
viene mantenuta così che la soglia resti una scelta di chi chiama:
`binary(threshold=0.5)` ne applica una, e `array` restituisce la vista numpy.

## RestoredImage

L'immagine RGB ripristinata, `(H, W, 3)` uint8. Per la super-risoluzione il
canvas è `Results.restore_scale` volte l'input. `array` restituisce la vista
numpy e `save(path)` scrive l'immagine.

## Matte

Matte di opacità morbido, float32 `(H, W)` in `[0, 1]` sul canvas
dell'immagine originale. `1` è completamente primo piano e `0` completamente
sfondo. Un matte morbido comprende in sé una maschera netta di rimozione dello
sfondo, con soglia a 0.5, e conserva i bordi con anti-aliasing che una
maschera binaria scarta. `array` restituisce la vista numpy.

Su un risultato di tipo matte, `Results.cutout(image=None)` restituisce un
array RGBA `(H, W, 4)` uint8 il cui quarto canale è il matte, e
`Results.save(path, image=None)` scrive quel ritaglio come PNG con sfondo
trasparente. Entrambi prendono l'RGB da `image` quando viene fornito,
altrimenti lo ricaricano da `Results.path`.

## OCRRegions

Testo localizzato con le relative trascrizioni. `data` contiene poligoni float
`(N, 4, 2)` in pixel dell'immagine originale, ordinati in alto a sinistra, in
alto a destra, in basso a destra, in basso a sinistra, e le regioni arrivano in
ordine di lettura, dall'alto verso il basso e poi da sinistra a destra.
`texts` è la lista delle N trascrizioni. `conf` è il punteggio di
riconoscimento per regione e `det_conf` il punteggio di rilevamento, entrambi
`(N,)`.

I quadrilateri di rilevamento sono veri poligoni, quindi non popolano
`Results.boxes`. `xyxy` fornisce gli involucri allineati agli assi.

## Embeddings

Vettori normalizzati in L2 prodotti dal task `embed`, sempre di shape
`(N, D)`. Un risultato sull'intera immagine porta una riga e nessun box; gli
embedding di regione sono allineati per riga con `boxes`. Poiché ogni riga è
normalizzata, la similarità coseno è un prodotto scalare.

| Membro | Restituisce |
|---|---|
| `dim` | `D` |
| `normalized` | Le righe, rinormalizzate |
| `similarity(other)` | Similarità coseno a coppie rispetto a un altro `Embeddings` o tensore |
| `verify(i, j, threshold=0.4)` | `True` quando le righe `i` e `j` corrispondono |

## Identities

Corrispondenze con nome dalla gallery, allineate per riga con `embeddings`.
Vengono prodotte quando si passa una `Gallery` a una predizione `embed`.
`name` è una lista in cui una voce è `None` sotto la soglia di corrispondenza,
e il nome più vicino sotto soglia non viene mai indovinato. `score` è l'array
dei punteggi di corrispondenza e `data` li accoppia.

## Meshes

Mesh parametriche di corpi umani, allineate per riga con i box delle persone in
`boxes`. Tutto è espresso nel sistema di riferimento della camera
dell'immagine originale. `transl` è metrico, in metri, con `+z` che punta
lontano dalla camera; `vertices` e `joints3d` sono metrici e includono già
`transl`; `joints2d` è in pixel sul canvas dell'immagine originale, non sul
ritaglio visto dalla rete. Nessun campo porta un sistema di riferimento del
mondo o della gravità.

La disposizione dei parametri cambia tra i body model, quindi nulla delle shape
è codificato staticamente. `body_model` indica il nome della parametrizzazione
e i conteggi si rileggono dai tensori: `num_vertices`, `num_joints`,
`num_betas` e `has_vertices`. `params` restituisce il dict dei parametri, e
`save_obj(path, index=0)` scrive una singola mesh. I campi sono
`global_orient`, `body_pose`, `betas`, `transl`, `vertices`, `faces`,
`joints3d`, `joints2d`, `conf`, `focal_length` ed `extras`.

Per `body_model="mhr"` le rotazioni sono angoli di Eulero in radianti anziché
axis-angle, `body_pose` è un vettore piatto di parametri per giunto anziché una
tripletta per giunto, e i `betas` sono coefficienti di blendshape
dell'identità. La scala dello scheletro, la posa delle mani e l'espressione
facciale stanno in `extras`.

## Conversione e selezione

Ogni payload espone `to(*args, **kwargs)`, `cpu()`, `cuda()` e `numpy()`, e
chiamarne uno sul `Results` lo applica in un colpo solo a tutti gli slot
valorizzati.

<code-tabs name="convert" />

`result[idx]` seleziona le righe su tutti i payload allineati per riga.
`len(result)` è il numero di rilevamenti, oppure di punti quando non ci sono
box. `result.update(...)` restituisce una copia con gli slot indicati
sostituiti; accetta ogni slot più `track_id` e `restore_scale`.

## summary e to_json

`summary(normalize=False, decimals=5, embeddings=False)` restituisce una lista
di dict semplici, una riga per ogni rilevamento, segmento, punto o regione a
seconda di quali slot sono valorizzati. `to_json(**kwargs)` passa i suoi
argomenti a `summary` e restituisce la stringa JSON.

`plot()` rende un risultato denso di normali o di bordi nella sua
visualizzazione canonica; per gli altri tipi di risultato solleva un'eccezione.
Le immagini annotate per gli altri task si ottengono con `predict(save=True)`.
