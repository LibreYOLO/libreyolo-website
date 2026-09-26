---
title: Riconoscimento facciale
seo_title: Riconoscimento facciale in LibreYOLO
description: >-
  Rileva, genera embedding e identifica volti in LibreYOLO. Registra una
  galleria, confronta due immagini e abbina per similarità del coseno, da Python
  o dalla CLI.
lead: >-
  Il riconoscimento facciale è il task embed applicato ai volti. Un rilevatore
  localizza e allinea ogni volto, una testa di riconoscimento restituisce un
  vettore normalizzato L2 per volto, e l'identità viene decisa dalla similarità
  del coseno rispetto a riferimenti registrati invece che da una lista fissa di
  classi.
keywords:
  - riconoscimento facciale python
  - face recognition python
  - face embedding
  - verifica facciale python
  - identificare volti nelle foto
  - arcface onnx
  - similarità del coseno volti
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # I nomi librefacerec-* puntano alla famiglia di embedding facciali

        # indipendentemente dal suffisso del file, e al primo uso si scaricano

        # dall'org LibreYOLO su Hugging Face insieme al rilevatore di volti
        predefinito.

        model = LibreYOLO("librefacerec-l.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)             # (N, 4) box dei volti

        print(result.embeddings.data.shape)  # (N, D), una riga per volto

        print(result.embeddings.dim)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=photo.jpg
    - label: Confrontare due immagini
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("librefacerec-l.onnx")


        # Esegue rilevamento ed embedding su entrambe le immagini e confronta il

        # volto con la confidenza più alta. La similarità del coseno sta in [-1,
        1].

        outcome = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)

        print(outcome["similarity"], outcome["same_person"])
    - label: Registrare una galleria e identificare
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("faces.npz")

        result = model("group_photo.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # il nome è None sotto la soglia
    - label: Registrare e identificare dalla CLI
      language: bash
      code: >
        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=faces.npz

        libreyolo predict model=librefacerec-l.onnx source=group_photo.jpg
        gallery=faces.npz
    - label: Usare i propri box dei volti
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("librefacerec-l.onnx")


        # face_boxes salta del tutto il rilevamento; face_detector accetta un

        # callable, un modello di rilevamento LibreYOLO o un'istanza di
        FaceDetector.

        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        print(result.embeddings.data.shape)
source_hash: d7dfcb6f812ebb2d
---

## Definizione

Il riconoscimento facciale restituisce un vettore per volto, non un'etichetta. La
predizione avviene in due fasi: un rilevatore di volti localizza ogni volto e i suoi
cinque landmark, il ritaglio viene trasformato nell'allineamento canonico 112x112, e
una testa di riconoscimento emette un embedding normalizzato L2.

`result.embeddings` è un payload `Embeddings` di forma `(N, D)`, allineato per righe
con `result.boxes`, quindi la riga `i` descrive il volto nel box `i`. Poiché le righe
sono vettori unitari, la similarità del coseno è un prodotto scalare, e
`embeddings.similarity()` la calcola rispetto a un altro `Embeddings` o a un'intera
matrice in una sola chiamata.

Dare un nome a un volto è un passo separato. Una `Gallery` contiene vettori di
riferimento con un nome; passare `gallery=` a `predict()` aggiunge
`result.identities`, allineato per righe con gli embedding, che contiene per ogni
volto un nome e il relativo miglior punteggio coseno. Un volto sotto la soglia di
match mantiene `None` come nome, e il nome più vicino ma rimasto sotto soglia non
viene mai messo al suo posto.

La chiave di task canonica della libreria è `embed`. `face-recognition`,
`facial-recognition`, `reid` e `face` si normalizzano tutte a quella, quindi
`task="face-recognition"` e `task="embed"` selezionano la stessa cosa. I volti sono
la forma a regioni di quel task più ampio; [embeddings](/docs/tasks/embeddings) copre
la forma a immagine intera e quella testuale, l'API condivisa `Embeddings`,
`Identities` e `Gallery`, e i modelli che producono vettori senza rilevare nulla.

## Modelli

[LibreFaceRec](/docs/models/librefacerec) è la famiglia per questo task. Sono due
artefatti ONNX dietro un'unica chiamata: `librefacerec-l.onnx`, una testa di
riconoscimento iResNet100 che produce embedding a 512 dimensioni, e
`librefacerec-det.onnx`, il rilevatore di volti predefinito con cinque landmark,
preso dallo zoo di OpenCV. Entrambi si scaricano dall'org LibreYOLO su Hugging Face
al primo uso. Qualsiasi altro file ONNX con la convenzione ArcFace (in ingresso
112x112 allineato, in uscita `(N, D)`) può sostituire la testa di riconoscimento
passandone il percorso al posto di un nome `librefacerec-*`.

La chiave di task `embed` è più ampia dei volti. Anche [CLIP](/docs/models/clip),
[SigLIP2](/docs/models/siglip2) e [DINOv2](/docs/models/dinov2) supportano
`task="embed"` e restituiscono un solo vettore per l'immagine intera, il che è
recupero di immagini piuttosto che identità facciale. Condividono l'API `Gallery` ed
`Embeddings`, quindi il flusso di registrazione e match descritto qui sotto vale
anche per loro, ma non rilevano né allineano volti.

La testa di riconoscimento gira su `onnxruntime`, che l'installazione base non
include:

```bash
pip install "libreyolo[onnx]"
```

## Predizione

<code-tabs name="predict" />

Se non intervieni, `predict()` scarica e abbina il rilevatore predefinito.
`face_detector` lo sostituisce con un callable, un modello di rilevamento LibreYOLO o
un'istanza di `FaceDetector`, e si può impostare sul costruttore o per singola
chiamata. `face_boxes` aggira il rilevamento con box che hai già. Dalla CLI,
`face_detector=` accetta il percorso di un `.onnx` di rilevamento volti o il nome di
un rilevatore LibreYOLO.

`model.verify(image_a, image_b)` è la scorciatoia a due immagini: genera l'embedding
del volto con la confidenza più alta in ciascuna e restituisce `{"similarity",
"same_person", "threshold"}`. `model.embed(sources)` restituisce tutte le righe dei
volti presenti in una o più immagini, impilate in un unico tensore `(N_total, D)`.
Vedi [predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Formato del dataset

La registrazione legge una cartella per identità. Il nome della cartella diventa
l'identità, e ogni immagine al suo interno fornisce riferimenti per quel nome:

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

`libreyolo enroll` percorre quell'albero e scrive una galleria `.npz`. Un file di
galleria esistente viene esteso sul posto invece che sostituito, così le identità si
possono aggiungere nel tempo. Le gallerie sono legate ai pesi che le hanno prodotte
tramite la dimensione dell'embedding e un'impronta del file; fare match con un modello
diverso solleva un errore invece di confrontare spazi vettoriali incompatibili.

Di default ogni immagine sorgente fornisce una riga di riferimento, il volto con
la confidenza più alta, quindi un ritratto che contiene passanti registra solo il suo
soggetto. Passa `select="all"` a `Gallery.enroll` per memorizzare ogni riga
restituita.

## Addestramento

Nessuna famiglia di questo task si addestra all'interno di LibreYOLO.
`LibreFaceEmbedder.train()` solleva un errore: addestra una testa di riconoscimento a
monte, esportala in ONNX con la convenzione ArcFace e carica il file indicandone il
percorso.

## Validazione

Non esiste un validatore di dataset per questo task, e `val()` solleva un errore
invece di fingere il contrario. L'accuratezza di verifica si misura con
`model.verify()` su coppie di immagini etichettate, facendo variare `threshold` per
scegliere il punto di lavoro che vuoi. L'accuratezza di identificazione si misura registrando
una galleria e leggendo `result.identities.name` e `result.identities.score` su
immagini tenute da parte, contando un nome `None` come un rifiuto.

## Esportazione

La testa di riconoscimento è già un grafo ONNX, quindi non c'è nulla da convertire:
`LibreFaceEmbedder.export()` solleva un errore. Metti in produzione direttamente il
file `.onnx`, oppure fai puntare LibreYOLO a quel file e lascia che sia la famiglia a
occuparsi di rilevamento, allineamento e normalizzazione.
