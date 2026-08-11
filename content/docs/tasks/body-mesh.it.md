---
title: Mesh corporea
seo_title: Recupero della mesh corporea in LibreYOLO
description: >-
  Recupera una mesh corporea 3D parametrica per persona in LibreYOLO. Fai
  predizioni a partire dai box delle persone o da un rilevatore, e leggi
  vertici, giunti e traslazione della camera.
lead: >-
  Il recupero della mesh corporea trasforma una singola immagine e un insieme di
  box di persone in un corpo 3D parametrico per ciascuna persona: parametri di
  forma e di posa, vertici in posa, giunti 3D e la traslazione della camera che
  li colloca davanti all'obiettivo.
keywords:
  - human mesh recovery python
  - mesh corporea 3d
  - posa 3d corpo umano
  - SAM 3D Body
  - MHR
  - modello corporeo parametrico
  - task mesh libreyolo
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import SAMPLE_IMAGE

        from libreyolo.models.sam3dbody import LibreSAM3DBody


        # Questa famiglia non è registrata nella factory LibreYOLO(), quindi si

        # costruisce direttamente. model_path=None avvia il download da Hugging Face,

        # ad accesso condizionato; una stringa è trattata come un checkpoint locale

        # esistente e non viene mai scaricata. L'inferenza richiede CUDA.

        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])


        meshes = result.meshes

        print(meshes.body_model)      # la parametrizzazione usata da questi
        tensori

        print(meshes.vertices.shape)  # (N, V, 3), riferimento della camera,
        metri

        print(meshes.joints3d.shape)  # (N, J, 3)

        print(meshes.joints2d.shape)  # (N, J, 2), pixel sull'immagine sorgente
    - label: Con un rilevatore di persone
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        from libreyolo.models.sam3dbody import LibreSAM3DBody


        # person_detector accetta un rilevatore LibreYOLO costruito, un semplice

        # callable o un'istanza di PersonDetector. Non esiste scorciatoia per
        nome.

        detector = LibreYOLO("LibreYOLO9s.pt")

        model = LibreSAM3DBody(None, size="d3", device="cuda")


        result = model(SAMPLE_IMAGE, person_detector=detector)
source_hash: 31c5b44171cbcd0e
---

## Definizione

Il recupero della mesh corporea restituisce un payload `Meshes` per immagine,
allineato per righe con `result.boxes`: la riga `i` descrive la persona nel box
`i`, lo stesso contratto che il task di stima della posa usa per i keypoint.

Tutto è espresso nel sistema di riferimento della camera dell'immagine
originale. `transl` è metrico, in metri, con +z che punta in direzione opposta
alla camera. `vertices` e `joints3d` sono metrici e includono già `transl`,
quindi non richiedono ulteriori composizioni. `joints2d` è in pixel sul canvas
dell'immagine originale, non sul ritaglio che la rete ha visto. `faces` contiene
la topologia della mesh una sola volta per l'intera immagine anziché per riga,
perché è condivisa da tutte le persone. In questa versione non esiste un
sistema di riferimento del mondo o della gravità, e nessun campo ne fa
silenziosamente le veci.

I layout dei parametri differiscono da un modello corporeo all'altro, quindi
nessuna dimensione dei tensori è fissa: `body_model` indica la parametrizzazione
e i conteggi si rileggono dai tensori. Per `"mhr"`, il Momentum Human Rig, le
rotazioni sono angoli di Eulero in radianti anziché asse-angolo, `body_pose` è
un vettore piatto di parametri per giunto anziché una tripletta per giunto, e
`betas` sono coefficienti di blendshape dell'identità. La scala dello scheletro,
la posa delle mani e l'espressione facciale si trovano in `extras`.

La chiave canonica del task è `mesh`. `body-mesh`, `hmr` e
`human-mesh-recovery` vengono normalizzati a questa chiave.

## Modelli

[SAM 3D Body](/docs/models/sam-3d-body) è l'unica famiglia che serve questo
task, ed è un wrapper anziché un porting: il pacchetto `sam-3d-body` di Meta è
pubblicato sotto la SAM License, da cui il codice di LibreYOLO non può derivare,
quindi non ne viene incluso nulla nel repository. Due backbone condividono lo
stesso modello corporeo MHR, `d3` su un encoder DINOv3 ViT-H/16+ e `h` sul
ViT-H originale.

Prima della prima predizione valgono tre requisiti, e nessuno di essi è
opzionale.

Il pacchetto upstream lo installi tu, non LibreYOLO:

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

Indica alla libreria dove si trova il clone con `sam_3d_body_path=` o con la
variabile d'ambiente `SAM_3D_BODY_PATH`. Chi non costruisce mai questa famiglia
non attiva mai l'import.

Il mirror del checkpoint è ad accesso condizionato. Accetta la licenza sulla
pagina del modello su Hugging Face e autenticati con `hf auth login`, altrimenti
il primo download fallisce. Il modello corporeo MHR è invece una release
Apache-2.0 separata, scaricata da una sua posizione pubblica e memorizzata nella
cache locale.

L'inferenza richiede un dispositivo CUDA. Lo stimatore upstream sposta il suo
batch sulla GPU senza controlli, quindi non c'è un percorso CPU di ripiego e
`device="cpu"` solleva un'eccezione.

## Predizione

<code-tabs name="predict" />

Le persone arrivano al modello in due modi. `person_boxes` passa box che
possiedi già, solo per una singola immagine: un insieme fisso di box non può
seguire le persone lungo i frame di un video, quindi passarlo con una sorgente
video solleva un'eccezione invece di riutilizzare silenziosamente i box del
primo frame. `person_detector` accetta un rilevatore LibreYOLO già costruito, un
callable o un `PersonDetector`, ed è la strada da seguire per il video.
`focal_length` fornisce un parametro intrinseco noto della camera; se non è
impostato, il modello usa la propria stima, che è quella riportata da
`meshes.focal_length`.

Questa famiglia non è collegata alla factory `LibreYOLO()` né al comando CLI
`libreyolo predict`. `LibreSAM3DBody` è l'unico punto d'ingresso. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Addestramento

Nessuna famiglia di questo task si addestra all'interno di LibreYOLO.
`LibreSAM3DBody.train()` solleva un'eccezione: addestra nel progetto upstream e
carica qui il checkpoint risultante.

## Validazione

Non esiste un validatore per le mesh, e `val()` solleva un'eccezione. I
benchmark abituali hanno licenza solo per la ricerca, quindi nessuno è incluso e
nessuno può essere scaricato al posto tuo.

Le metriche in sé sono disponibili come `libreyolo.validation.mesh_metrics`, per
valutare su un dataset che possiedi già. La funzione accetta i giunti predetti e
quelli di riferimento, opzionalmente anche i vertici predetti e quelli di
riferimento, e restituisce un dizionario con esattamente le stesse chiavi di
quello di un validatore:

`metrics/mpjpe` è l'errore medio di posizione per giunto dopo l'allineamento del
giunto radice, quindi valuta la posa ignorando dove si trova la persona nella
scena. `metrics/pa_mpjpe` è la stessa quantità dopo un allineamento di Procuste
completo, rotazione, scala uniforme e traslazione, che elimina l'errore di
orientamento globale e di dimensione del corpo e lascia la posa articolata.
`metrics/pve` è l'errore medio per vertice sulla superficie della mesh dopo
l'allineamento sul centroide dei vertici; a differenza delle metriche sui giunti
è sensibile alla forma del corpo, e compare solo quando vengono forniti entrambi
gli array di vertici. Per tutte e tre, valori più bassi sono migliori. Si assume
che gli input siano metrici, in metri, e `scale_to_mm` converte i risultati nei
millimetri riportati in letteratura.

## Esportazione

L'esportazione delle mesh non è implementata. LibreYOLO non ha definito un
contratto di metadati per il grafo esportato di questo task, compreso il modo in
cui trasportare il layout dei parametri MHR fuori da PyTorch, quindi `export()`
solleva un'eccezione anziché emettere un grafo il cui output non potrebbe essere
interpretato.
