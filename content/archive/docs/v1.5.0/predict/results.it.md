---
title: Lavorare con i risultati
seo_title: L'oggetto Results di LibreYOLO
description: >-
  Un oggetto Results per immagine, con uno slot per ogni tipo di payload: box,
  maschere, keypoint, probs, profondità, panottico, OCR e altro. Disegno,
  salvataggio e JSON.
lead: >-
  Ogni predizione restituisce un oggetto Results per immagine. Ha uno slot con
  nome per ogni tipo di payload, tutti vuoti tranne quelli che il modello
  produce, più gli stessi slot su un artefatto esportato.
keywords:
  - oggetto results yolo python
  - results.boxes xyxy
  - results to json yolo
  - salvare immagine annotata yolo
  - maschere segmentazione python
  - keypoints results python
  - mappa di profondità results
  - results summary yolo
  - onnx stessi risultati yolo
last_verified: 1.5.0
verification: >-
  Classi dei payload, slot, semantica di spostamento, summary(), to_json(),
  plot(), save() e cutout() letti da libreyolo/utils/results.py. Comportamento
  di annotazione e di scrittura su disco da
  InferenceRunner._save_annotated_image in libreyolo/models/base/inference.py e
  da resolve_save_path in libreyolo/utils/general.py. Dispatch sul suffisso da
  LibreYOLO() in libreyolo/models/__init__.py.
snippets:
  basic:
    - label: Box
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(SAMPLE_IMAGE)


        print(result.orig_shape)   # (altezza, larghezza) dell'immagine di
        origine

        print(result.path)         # percorso di origine, None per input in
        memoria


        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: Coordinate normalizzate
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy[:1])    # pixel, x1 y1 x2 y2

        print(result.boxes.xywh[:1])    # pixel, centro x, centro y, w, h

        print(result.boxes.xyxyn[:1])   # lo stesso box diviso per larghezza e
        altezza

        print(result.boxes.xywhn[:1])
    - label: NumPy e device
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        # Ognuno di questi restituisce un nuovo Results; l'originale non cambia.
        as_numpy = result.numpy()
        on_cpu = result.cpu()

        print(type(as_numpy.boxes.xyxy).__name__)
        print(type(on_cpu.boxes.xyxy).__name__)
  json:
    - label: summary e to_json
      language: python
      code: |
        import json

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        rows = result.summary()
        print(json.dumps(rows[:2], indent=2))

        # Lo stesso contenuto come stringa, con gli stessi argomenti con nome.
        print(result.to_json(normalize=True, decimals=3)[:200])
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt --json \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  saving:
    - label: Immagini annotate
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # save=True disegna il payload e lo scrive sotto runs/detect/predict*.
        result = model(SAMPLE_IMAGE, save=True)
        print(result.saved_path)
  exported:
    - label: Installare l'extra di esportazione
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: Lo stesso Results da un artefatto esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")   # restituisce il percorso scritto

        # LibreYOLO() fa dispatch sul suffisso del file.
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)

        print(type(result).__name__, len(result.boxes))
source_hash: 548dbc9c7f5552ec
---

## Un oggetto, uno slot per payload

Una predizione su una singola immagine restituisce un `Results`. Porta con sé
diciotto slot di payload, e un modello riempie solo quelli che il suo task
produce. Ogni altro slot è `None`, quindi leggere `result.masks` su un detector
dà `None` invece di un errore.

| Slot | Classe | Forma | Prodotto da |
|---|---|---|---|
| `boxes` | `Boxes` | `(N, 4)` più punteggi e classi | Rilevamento, e qualsiasi task che prima localizza |
| `masks` | `Masks` | `(N, H, W)` | Segmentazione di istanze |
| `keypoints` | `Keypoints` | `(N, K, 2)` o `(N, K, 3)` | Posa |
| `probs` | `Probs` | `(C,)` | Classificazione |
| `obb` | `OBB` | `(N, 7)` o `(N, 8)` | Box orientati |
| `gaze` | `Gaze` | `(N, 2)` pitch e yaw in radianti | Stima dello sguardo |
| `points` | `Points` | `(N, 4)` come x, y, classe, confidenza | Localizzazione di punti |
| `semantic_mask` | `SemanticMask` | `(H, W)` id di classe | Segmentazione semantica |
| `panoptic` | `PanopticSegmentation` | `(H, W)` id di segmento più `segments_info` | Segmentazione panottica |
| `depth_map` | `DepthMap` | `(H, W)` float | Stima della profondità |
| `normal_map` | `NormalMap` | `(H, W, 3)` vettori unitari | Normali di superficie |
| `edges` | `EdgeMap` | `(H, W)` float in `[0, 1]` | Rilevamento dei bordi |
| `restored` | `RestoredImage` | `(H, W, 3)` RGB uint8 | Restauro e super-risoluzione |
| `matte` | `Matte` | `(H, W)` float in `[0, 1]` | Alpha matting e rimozione dello sfondo |
| `ocr` | `OCRRegions` | `(N, 4, 2)` poligoni più trascrizioni | Rilevamento e riconoscimento del testo |
| `embeddings` | `Embeddings` | `(N, D)` righe normalizzate L2 | Il task `embed` |
| `identities` | `Identities` | N nomi e punteggi | Il task `embed` con una galleria |
| `meshes` | `Meshes` | Parametri del corpo e vertici opzionali | Ricostruzione della mesh corporea |

Accanto a loro stanno i campi che ogni risultato ha: `orig_shape` come
`(altezza, larghezza)`, `path` (il percorso di origine, o `None` per input in
memoria), `names` che mappa l'id di classe al nome della classe, `frame_idx` per
i frame di video e live, `track_id` quando c'è il tracking, e `restore_scale`,
il fattore di ingrandimento intero di un risultato di restauro.

`result.normals` è un alias di `result.normal_map`.

`result.speed` esiste su ogni risultato ma viene popolato solo dagli
[ensemble](/docs/predict/ensembling), dove le sue chiavi sono `member_0`,
`member_1` e `fusion` in millisecondi. Per un singolo modello resta un dict
vuoto.

## Box

<code-tabs name="basic" />

`Boxes` tiene le coordinate e i punteggi come array separati invece che in un
unico tensore compatto.

| Attributo | Contenuto |
|---|---|
| `xyxy` | `(N, 4)` pixel assoluti, x1 y1 x2 y2 |
| `xywh` | `(N, 4)` pixel assoluti, centro x, centro y, larghezza, altezza |
| `xyxyn`, `xywhn` | Gli stessi divisi per larghezza e altezza dell'immagine |
| `conf` | `(N,)` confidenza |
| `cls` | `(N,)` id di classe, come float |
| `id` | `(N,)` id di tracking, o `None` |
| `is_track` | Se `id` è impostato |
| `data` | Tutto concatenato: box, id opzionale, conf, cls |

`cls` è un array di float, quindi usalo come `result.names[int(cls)]`.

`xyxyn` e `xywhn` hanno bisogno di `orig_shape`, che `Results` compila per te.

## Payload densi

I payload che coprono l'intera immagine si comportano diversamente da quelli
per istanza, e la cosa conta quando fai slicing.

`SemanticMask` contiene `(H, W)` id di classe sul canvas originale, con `255`
riservato come valore di ignore che non conta mai come classe. `classes` elenca
gli id presenti e lo esclude; `class_mask(id)` restituisce un `(H, W)` booleano.

`PanopticSegmentation` contiene `(H, W)` id di segmento, con `0` come id di
void, e una lista `segments_info` di dict che portano almeno `id` e
`category_id`. `segment_ids` elenca gli id presenti, `segment_mask(id)` ne
seleziona uno.

`DepthMap` contiene `(H, W)` di profondità inversa relativa: più alto significa
più vicino, e i valori non sono metri in scala metrica. Espone `min`, `max`,
`mean` sui valori finiti, e `normalized()` che riscala a `[0, 1]`.

`NormalMap` contiene `(H, W, 3)` vettori unitari nel sistema di riferimento
della camera OpenCV, con `+x` a destra, `+y` in basso e `+z` verso la scena,
quindi una superficie rivolta verso la camera è `(0, 0, -1)`.
`assert_normalized()` verifica che ogni pixel sia finito e di lunghezza
unitaria.

`EdgeMap` contiene `(H, W)` float32 in `[0, 1]`. La mappa continua viene
conservata invece di essere ridotta con una soglia, quindi
`binary(threshold=0.5)` è il punto in cui scegli il taglio.

`Matte` contiene `(H, W)` float32 in `[0, 1]`, dove `1` è primo piano pieno.
`array` lo restituisce limitato all'intervallo, come float32.

`RestoredImage` contiene `(H, W, 3)` RGB uint8, con `array` per l'ndarray
grezzo e `save(path)` per scriverlo.

`Probs` contiene un solo vettore di probabilità per l'immagine. `top1` e `top5`
sono indici di classe, `top1conf` e `top5conf` i punteggi corrispondenti.

`Embeddings` contiene `(N, D)` righe già normalizzate L2, quindi la similarità
coseno è un prodotto scalare. `similarity(other)` restituisce `(N, M)` contro
una galleria oppure `(N,)` contro un singolo vettore, e
`verify(i, j, threshold=0.4)` confronta due righe.

`OCRRegions` contiene `(N, 4, 2)` poligoni in ordine di lettura, con gli angoli
ordinati in alto a sinistra, in alto a destra, in basso a destra, in basso a
sinistra. Le trascrizioni sono in `texts`, i punteggi di riconoscimento in
`conf`, quelli di rilevamento in `det_conf`. Poiché si tratta di veri poligoni
ruotati, non popolano `boxes`; `ocr.xyxy` dà gli inviluppi allineati agli assi
quando ti servono dei rettangoli.

## Slicing e spostamento

`result[i]` restituisce un nuovo `Results` che contiene una sola istanza. I
payload per istanza vengono affettati; quelli sull'intera immagine vengono
riportati invariati, così affettare un risultato di classificazione non può
troncare il suo vettore di probabilità a una sola classe, e affettare un
risultato di profondità non può corrompere il layout `(H, W)`.

`len(result)` conta le istanze: box, punti, embedding, regioni OCR o mesh.
Qualsiasi payload denso sull'intera immagine conta come `1`. Un risultato che
non ha niente dentro è `0`.

`to()`, `cpu()`, `cuda()` e `numpy()` restituiscono ciascuno un nuovo `Results`
con ogni slot popolato convertito. Non modificano l'originale.

`update()` è l'unico metodo che muta sul posto, sostituendo gli slot indicati e
restituendo lo stesso oggetto.

## JSON

<code-tabs name="json" />

`summary()` restituisce una lista di dict semplici, e `to_json()` è quella
lista passata attraverso `json.dumps`. Entrambi accettano gli stessi tre
argomenti: `normalize=False` porta le coordinate in `[0, 1]`, `decimals=5`
imposta l'arrotondamento, e `embeddings=False` controlla se i vettori di
embedding vengono inclusi.

La forma della riga segue il payload. Le righe di rilevamento portano `name`,
`class`, `confidence` e un dict `box`, e prendono anche `segments` quando ci
sono le maschere, `obb` e `corners` per i box orientati, gli angoli `gaze` sia
in radianti sia in gradi, `track_id` quando c'è il tracking, e i parametri
`mesh` quando ci sono le mesh.

Dove non ci sono box, è un solo payload a decidere le righe: l'OCR emette una
riga per regione con il suo `text`, i punti una riga per punto, il panottico
una riga per segmento con `pixel_count` e `pixel_fraction`, il semantico una
riga per classe presente, la classificazione le prime cinque classi.
Profondità, normali, bordi, restauro e matting emettono ciascuno una singola
riga di riepilogo che descrive la mappa invece dei suoi pixel.

Due payload sono volutamente abbreviati. Un vettore di embedding viene
riportato solo come `embedding_dim`, perché una riga di 512 float è circa 2 KB
per volto; passa `embeddings=True` per includere i valori. I vertici delle mesh
non vengono inclusi mai, dato che sono decine di migliaia di coordinate per
persona. Leggi `result.meshes.vertices` o chiama
`result.meshes.save_obj(path)` per la geometria.

## Disegno e salvataggio

<code-tabs name="saving" />

`predict(save=True)` è il percorso che annota e scrive. Sceglie la routine di
disegno in base a quale slot è riempito, così un risultato semantico viene
scritto come maschera colorata, un risultato di profondità come visualizzazione
della profondità, un risultato panottico con i suoi segmenti, un matte come PNG
RGBA con sfondo trasparente, e un detector come box con le maschere sotto. Il
percorso scritto viene attaccato al risultato come `result.saved_path`.

`Results.plot()` è più ristretto di quanto suggerisca il nome. È definito solo
per le mappe di normali e le mappe di bordi, e solleva `NotImplementedError`
per tutto il resto. Per gli altri task usa `save=True`.

`Results.save(path)` è altrettanto ristretto: scrive un risultato di matting
come ritaglio PNG RGBA con sfondo trasparente e altrimenti solleva
`NotImplementedError`. `Results.cutout()` restituisce lo stesso array RGBA
senza scriverlo. Entrambi hanno bisogno dell'immagine di origine, presa da
`result.path` o passata con `image=`.

Due payload portano i propri writer: `result.restored.save(path)` per
un'immagine restaurata, e `result.meshes.save_obj(path, index=0)` per una mesh.

Per sapere dove finiscono i file e come si comportano `output_path` e
`output_file_format`, vedi [Sorgenti di predizione](/docs/predict/sources).

## Gli artefatti esportati restituiscono lo stesso oggetto

<code-tabs name="exported" />

`LibreYOLO()` fa dispatch sul suffisso del file, quindi un artefatto esportato
si carica con la stessa chiamata di un checkpoint `.pt` e restituisce lo stesso
`Results`. I file `.onnx`, `.engine`, `.pte` e `.mnn` vengono riconosciuti dal
suffisso, così come le directory OpenVINO, Paddle e ncnn e un URL di modello
Triton. Il codice che legge `result.boxes.xyxy` non cambia quando un modello
viene sostituito con la sua build esportata. Vedi
[Esportazione](/docs/export) per l'insieme completo dei formati.

Ricorrere invece all'API del runtime significa doverti occupare da solo di
preprocessing, postprocessing e nomi delle classi.
