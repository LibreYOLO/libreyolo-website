---
title: Embedding
seo_title: Embedding di immagini e regioni in LibreYOLO
description: >-
  Il task embed restituisce vettori float32 normalizzati L2 per un'immagine
  intera, per ogni regione rilevata o per il testo. Registra una gallery,
  confronta per similarità coseno e cerca da Python o dalla CLI.
lead: >-
  Un solo task copre ogni vettore che LibreYOLO produce. embed restituisce righe
  float32 di lunghezza unitaria il cui prodotto scalare è un punteggio di
  similarità, che la riga descriva un'immagine intera, un singolo volto rilevato
  o una riga di testo, e la stessa Gallery le confronta tutte.
keywords:
  - image embeddings python
  - embedding normalizzato l2
  - ricerca per similarità coseno
  - libreyolo embed task
  - image retrieval python
  - gallery enroll
  - clip embeddings
  - dinov2 embeddings
  - reid embeddings
last_verified: 1.5.0
verification: >-
  Chiave del task e alias letti da libreyolo/tasks.py. Payload dei risultati
  dalle classi Embeddings e Identities in libreyolo/utils/results.py. API
  Gallery da libreyolo/utils/gallery.py. embed e _postprocess_embeddings da
  libreyolo/models/base/model.py. Famiglie supportate individuate cercando embed
  in SUPPORTED_TASKS dentro libreyolo/models/**/model.py. Superficie CLI da
  libreyolo/cli/__init__.py, libreyolo/cli/commands/special.py e
  libreyolo/cli/commands/predict.py. Intento di progettazione da
  docs/adr/0015-embed-generalization.md.
meta:
  - label: Chiave del task
    value: embed
    mono: true
  - label: Alias
    value: 'face-recognition, reid, face'
    mono: true
  - label: Payload dei risultati
    value: 'Embeddings, Identities'
    mono: true
  - label: dtype delle righe
    value: 'float32, lunghezza unitaria'
snippets:
  predict:
    - label: Immagine intera
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # CLIP usa classify come default, quindi chiedi il vettore
        esplicitamente.

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        result = model(SAMPLE_IMAGE)


        print(result.embeddings.data.shape)  # (1, 512), una riga per immagine

        print(result.boxes)                  # None: non è stato localizzato
        nulla
    - label: Per regione
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        # La riga i descrive la regione nel box i.
        print(result.boxes.xyxy.shape)       # (N, 4)
        print(result.embeddings.data.shape)  # (N, 512)
    - label: Più immagini in una volta sola
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # Ogni riga di ogni risultato, concatenata in un unico tensore.
        vectors = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(vectors.shape)  # (3, 384)
    - label: Testo
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        # Il testo è un metodo, mai una sorgente di predizione. Una stringa
        # passata a model(...) resta un percorso o un URL.
        text = model.embed_text(["a photo of a cat", "a photo of a dog"])
        print(text.shape)  # (2, 512)
  similarity:
    - label: Confrontare due insiemi di righe
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")


        query = model.embed("query.jpg")          # (1, 512)

        pool = model.embed(["a.jpg", "b.jpg"])    # (2, 512)


        # Le righe hanno lunghezza unitaria: la similarità coseno è un prodotto
        scalare.

        scores = model("query.jpg").embeddings.similarity(pool)

        print(scores.shape)  # (1, 2)
    - label: Immagine contro testo
      language: python
      code: |
        import torch

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        image = model.embed("photo.jpg")                       # (1, 512)
        text = model.embed_text(["a cat", "a dog", "a car"])   # (3, 512)

        print(torch.matmul(image, text.T))
  gallery:
    - label: Registrare e identificare
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("refs.npz")

        result = model("group.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # name è None sotto la soglia
    - label: Ricerca top-k
      language: python
      code: |
        from libreyolo import Gallery
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        gallery = Gallery.load("refs.npz", model=model)

        result = model("query.jpg")
        matches = gallery.match(result.embeddings, top_k=5, threshold=0.4)
        print(matches[0])   # [(name, score), ...] per la prima riga
    - label: Registrare un vettore che hai già
      language: python
      code: |
        from libreyolo import Gallery

        gallery = Gallery()
        gallery.enroll_embedding("ada", vector)  # normalizzato in ingresso
        print(gallery.identities, gallery.dim, len(gallery))
  cli:
    - label: Registrare un albero di cartelle
      language: bash
      code: >
        # source/<identity>/*.jpg. Una gallery esistente viene estesa sul posto.

        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=refs.npz
    - label: Identificare durante la predizione
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=group.jpg \
          gallery=refs.npz gallery_threshold=0.45
    - label: Confrontare due immagini
      language: bash
      code: >
        libreyolo compare model=librefacerec-l.onnx \
          source=a.jpg source2=b.jpg threshold=0.4

        # verify è lo stesso comando con un secondo nome.

        libreyolo verify model=librefacerec-l.onnx source=a.jpg source2=b.jpg
        --json
source_hash: ffbaad5599035bc7
---

## Definizione

`embed` trasforma un'immagine, una regione di un'immagine o una stringa in una
riga float32 di larghezza fissa la cui lunghezza è uno. Poiché ogni riga è un
vettore unitario, confrontarne due è un prodotto scalare, e confrontare due
insiemi di righe è una sola moltiplicazione tra matrici. Nient'altro nel task
dipende dal modello: retrieval, rilevamento di duplicati, re-identificazione e
riconoscimento facciale sono tutti la stessa aritmetica su righe diverse.

Il vettore è l'output. Non c'è un elenco di classi, quindi il nome viene
assegnato in un secondo momento, confrontando con i riferimenti che fornisci tu
e non con qualcosa che la rete è stata addestrata a predire.

### Tre forme

| Forma | `Results.embeddings` | `Results.boxes` | Prodotta da |
|---|---|---|---|
| Immagine intera | `(1, D)` | `None` | Passare un'immagine a una famiglia che lavora sull'immagine intera |
| Regione | `(N, D)` | `(N, 4)`, allineati per riga | Famiglie che localizzano prima, come il riconoscimento facciale |
| Testo | non è affatto un `Results` | | `model.embed_text(texts)`, che restituisce `(M, D)` |

Un risultato sull'immagine intera resta bidimensionale anche per una sola
immagine. `(D,)` non è una forma di ritorno ammessa, così chi consuma il dato
non deve mai trattare il caso a riga singola come un'eccezione. Il testo
restituisce un tensore semplice invece di un `Results`, perché una stringa non è
una sorgente di immagini: passarne una a `model(...)` significa comunque un
percorso o un URL, e la libreria non indovina mai che una stringa sia prosa.

La chiave canonica del task è `embed`. `embedding`, `embeddings`,
`face-recognition`, `facial-recognition`, `recognition`, `face`, `faceid` e
`reid` si normalizzano tutti su di essa, quindi `task="reid"` e `task="embed"`
selezionano esattamente la stessa cosa.

## Modelli

Quattro famiglie coprono il task, e si dividono nettamente in base al fatto che
localizzino o meno qualcosa prima.

| Famiglia | Forma | Dimensione | Supporta anche |
|---|---|---|---|
| [LibreFaceRec](/docs/models/librefacerec) | Regione, una riga per volto rilevato | 512 | Nulla; `embed` è il suo unico task |
| [CLIP](/docs/models/clip) | Immagine intera, con una torre testuale abbinata | 512 per `b32` e `b16`, 768 per `l14` | `classify`, che resta il suo default |
| [SigLIP 2](/docs/models/siglip2) | Immagine intera, con una torre testuale abbinata | 768 per `b16`, 1152 per `so400m` | `classify`, che resta il suo default |
| [DINOv2](/docs/models/dinov2) | Immagine intera, solo immagini | 384 | `semantic`, `classify` |

CLIP e SigLIP 2 mantengono `classify` come task predefinito, quindi `task="embed"`
va chiesto esplicitamente. Il loro checkpoint `-cls` esistente è l'artefatto
condiviso a due torri; per pesi identici non viene pubblicato un checkpoint
`-embed` duplicato.

`embed_text` esiste solo su CLIP e SigLIP 2, le due famiglie con una torre
testuale. DINOv2 non ne ha. L'embedding di DINOv2 aggira le teste semantica e di
classificazione e legge il token CLS finale normalizzato a 224 pixel; le varianti
`n`, `s`, `m` e `l` condividono tutte l'encoder DINOv2-S, quindi tutte e quattro
restituiscono `D = 384`.

I backbone di sola classificazione aggiunti in questa release, [ViT](/docs/models/vit),
[Swin](/docs/models/swin) e [DeiT](/docs/models/deit), dichiarano solo `classify`
e non coprono questo task.

<code-tabs name="predict" />

`model.embed(source, **kwargs)` è la scorciatoia per il batch: esegue `predict` e
concatena ogni riga di ogni risultato in un unico tensore float32 su CPU
`(N_total, D)`, sollevando un errore se le righe hanno dimensioni diverse. Una
famiglia che non ha `embed` tra i task supportati solleva `NotImplementedError`.

## Payload dei risultati

`result.embeddings` è un payload `Embeddings`. Il suo `data` è sempre `(N, D)`
float32, già normalizzato L2 dal percorso di inferenza, e un input non
bidimensionale solleva un errore invece di essere rimodellato in silenzio.

| Membro | Significato |
|---|---|
| `.data` | La matrice `(N, D)` |
| `.dim` | `D` |
| `.normalized` | Le stesse righe, rinormalizzate per sicurezza |
| `.similarity(other)` | `(N, M)` contro un altro insieme, oppure `(N,)` contro un singolo vettore `(D,)` |
| `.verify(i, j, threshold=0.4)` | Se le righe `i` e `j` sono lo stesso soggetto |

`result.identities` è un payload `Identities`, presente solo quando è stata
passata una gallery. È un contenitore semplice, non un tensore, quindi spostare
un `Results` tra dispositivi lo lascia intatto.

| Membro | Significato |
|---|---|
| `.name` | Elenco di nomi, `None` dove nulla ha superato la soglia |
| `.score` | Miglior punteggio coseno float32 `(N,)`, mantenuto anche quando il nome è `None` |
| `.data` | Elenco di tuple `(name, score)` |

<code-tabs name="similarity" />

I vettori sono esclusi da `summary()` e `to_json()` per impostazione predefinita,
dato che una riga di 512 float pesa circa due kilobyte per soggetto. Ogni riga
riporta invece `embedding_dim`, più `identity` e `identity_score` quando è stata
usata una gallery. Passa `summary(embeddings=True)` per includere i numeri.

## Gallery

Una `Gallery` è un insieme di righe di riferimento con un nome. Conserva ogni
riferimento separatamente invece di farne la media, così un nome viene valutato
in base al suo singolo riferimento più simile, e aggiungere una foto scadente non
può spostare il centroide di un'identità.

<code-tabs name="gallery" />

`Gallery(model)` si lega ai pesi che produrranno i suoi vettori.
`enroll(name, sources, select="best")` esegue la predizione su ogni sorgente e
tiene la riga con la confidenza più alta per risultato; `select="all"` tiene
invece tutte le righe, ed è quello che vuoi quando un'immagine di riferimento
contiene legittimamente più soggetti. `enroll_embedding(name, vector)` salta
l'inferenza e prende un vettore direttamente, normalizzandolo e rifiutando una
riga tutta a zeri.

`FaceGallery` è un alias permanente della stessa classe, e gli archivi scritti
dalle release precedenti dedicate ai soli volti si caricano ancora.

### Confronto e soglie

Il confronto è una moltiplicazione densa tra matrici contro ogni riferimento
memorizzato, ridotta a un punteggio per nome prendendo il massimo. Non c'è un
indice approssimato, il che mantiene i numeri esatti e pone un tetto pratico alla
dimensione della gallery.

Due punti di ingresso si comportano diversamente sotto la soglia. `match()`
restituisce `[(name, score), ...]` per riga, con tutto ciò che sta sotto la
soglia scartato, quindi una riga senza corrispondenze è una lista vuota.
`identify()` restituisce un payload `Identities` che conserva sempre il punteggio
migliore e mette il nome a `None` quando è sotto la soglia. Nessuno dei due
sostituisce mai il nome più vicino ma sotto soglia.

La soglia predefinita è `0.4` ovunque. È un valore coseno, non una probabilità, e
il punto di lavoro giusto è una proprietà dei tuoi dati e della tua tolleranza
alle false corrispondenze, quindi tarala su coppie etichettate invece di
accettare il valore predefinito. `libreyolo enroll` e l'argomento di predizione `gallery=`
usano lo stesso numero.

### Persistenza

`save(path)` scrive un `.npz` compresso che contiene i vettori, i nomi e un
blocco di metadati con la versione del formato, la dimensione dell'embedding e
un'impronta dei pesi che hanno prodotto le righe. `Gallery.load(path,
model=...)` controlla entrambi prima di confrontare qualsiasi cosa, così puntare
una gallery a un modello diverso solleva un errore invece di valutare in silenzio
vettori provenienti da due spazi che non hanno nulla a che fare tra loro. Salvare
una gallery vuota viene rifiutato.

## Riga di comando

| Comando | Scopo |
|---|---|
| `libreyolo enroll` | Percorre un albero con una cartella per identità e scrive o estende una gallery `.npz` |
| `libreyolo compare` | Calcola l'embedding del soggetto principale in due immagini e riporta la similarità coseno |
| `libreyolo verify` | Lo stesso comando con un secondo nome |
| `libreyolo predict gallery=...` | Allega le identità a una normale esecuzione di predizione |

<code-tabs name="cli" />

Ogni comando LibreYOLO accetta sia `key=value` sia `--key value`, quindi
`gallery=refs.npz` e `--gallery refs.npz` sono lo stesso argomento.

`enroll` prende `model`, `source` e `gallery`, più gli opzionali `face-detector`,
`device`, `--json` e `--quiet`. Legge una cartella per identità, dove il nome
della cartella è l'identità e ogni immagine al suo interno fornisce dei
riferimenti:

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

Un'immagine che non produce nulla viene saltata con una riga su stderr invece di
interrompere l'esecuzione, e il riepilogo riporta quanti riferimenti sono stati
memorizzati per ciascun nome. Un file di gallery esistente viene esteso sul
posto, così le identità possono essere aggiunte nel tempo.

`compare` e `verify` sono una sola funzione registrata due volte. Prendono
`model`, `source`, `source2` e un `threshold` opzionale, e stampano la similarità
coseno, il verdetto stesso-o-diverso e la soglia che l'ha prodotto. `--json`
stampa gli stessi tre campi come oggetto.

Su `predict`, `gallery` punta a un `.npz` salvato e `gallery_threshold`
sovrascrive il valore predefinito `0.4`. Passare una gallery a un modello il cui
task non è `embed` è un errore e non un'operazione a vuoto silenziosa, e un file
di gallery mancante suggerisce il comando `libreyolo enroll` che lo creerebbe.

## Volti

Il riconoscimento facciale è la forma a regione di questo task, ed è l'unica
implementazione di quella forma inclusa nella libreria. Aggiunge una fase di
rilevamento e allineamento davanti alla testa di embedding, più un metodo
`verify()`, un argomento per fornire box propri, numeri di accuratezza pubblicati
e indicazioni per calibrare la soglia. Tutto questo sta su
[riconoscimento facciale](/docs/tasks/face-recognition), che è la guida da
seguire quando il soggetto sono i volti. Tutto quello che c'è in questa pagina
vale anche in quel caso, senza modifiche.

## Addestramento, validazione ed esportazione

In questo task non si addestra nulla dentro LibreYOLO. La testa di embedding
facciale è un artefatto ONNX i cui `train()`, `val()` ed `export()` sollevano
tutti un errore; addestra una testa a monte e carica il file indicandone il
percorso. CLIP,
SigLIP 2 e DINOv2 si addestrano ed esportano attraverso i loro task di
classificazione e segmentazione, non attraverso `embed`.

Non esiste un validatore per il retrieval. Misura l'accuratezza di verifica su
coppie etichettate esplorando `threshold`, e l'accuratezza di identificazione
registrando una gallery e leggendo `identities.name` e `identities.score` su
immagini tenute da parte, contando un nome `None` come un rifiuto.
