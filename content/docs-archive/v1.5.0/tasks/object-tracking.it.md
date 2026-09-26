---
title: Tracciamento di oggetti
seo_title: Tracciamento di oggetti in LibreYOLO
description: >-
  Segui gli oggetti lungo i frame di un video in LibreYOLO con ByteTrack,
  BoT-SORT, OC-SORT o Deep OC-SORT, su qualsiasi modello di rilevamento,
  segmentazione o posa.
lead: >-
  Il tracking assegna un'identità stabile a ogni rilevamento lungo i frame di un
  video. LibreYOLO non lo modella come un task con pesi propri: è una modalità
  di predizione, model.track(), che esegue il tracker scelto sull'output per
  frame di un modello di rilevamento, segmentazione o posa.
keywords:
  - object tracking python
  - tracking multi oggetto video
  - bytetrack
  - botsort
  - ocsort
  - deep ocsort
  - track id yolo
  - tracking con reid
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # track() è un generatore: un Results per ogni frame elaborato.
        for result in model.track("video.mp4"):
            print(result.track_id)        # tensore int (N,), allineato con i box
            print(result.boxes.xyxy)
    - label: Scegliere un tracker
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # "bytetrack" (predefinito), "botsort", "ocsort" o "deepocsort".
        for result in model.track("video.mp4", tracker="botsort"):
            print(result.track_id)
    - label: Salvare un video annotato
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Senza output_path, il file finisce in runs/track/<video_stem>.mp4.
        for result in model.track("video.mp4", save=True, vid_stride=2):
            pass
    - label: Regolare un tracker
      language: python
      code: >
        from libreyolo import BoTSortConfig, LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Il tipo di config seleziona il tracker, quindi qui tracker= è
        ridondante.

        config = BoTSortConfig(track_buffer=60, frame_rate=25, enable_cmc=False)

        for result in model.track("video.mp4", tracker_config=config):
            print(result.track_id)

        # Oppure passa gli stessi campi come argomenti keyword e lascia che sia
        track() a costruirla.

        for result in model.track("video.mp4", tracker="botsort",
        track_buffer=60):
            print(result.track_id)
source_hash: f1fa7dcf60597d6b
---

## Definizione

Il tracking non è una delle chiavi task di LibreYOLO, e non c'è nessun
checkpoint di tracking da scaricare. È un metodo del modello,
`model.track(source)`, che esegue il rilevamento su ogni frame e associa i
risultati nel tempo. Il metodo è un generatore: produce un `Results` per ogni
frame elaborato, con `result.track_id` impostato a un tensore di interi `(N,)`
allineato con `result.boxes`. Gli stessi ID si trovano anche su
`result.boxes.id`.

Vengono prodotti solo gli oggetti confermati e tracciati in quel momento. Una
traccia che l'associazione perde resta viva per un numero configurato di frame
prima di essere scartata, `track_buffer` per ByteTrack e BoT-SORT e `max_age`
per le due varianti di OC-SORT, così un oggetto ritrovato dentro quella
finestra mantiene il suo ID originale.

Dato che l'associazione avviene dopo il rilevamento, gli altri contenuti del
frame le sopravvivono: il `Results` tracciato è il `Results` del rilevamento
ridotto alle righe abbinate, quindi maschere e keypoint arrivano insieme ai box.

## Modelli

In un'esecuzione di tracking entrano due scelte indipendenti: il modello che
produce i box a ogni frame, e il tracker che li collega.

Qualsiasi modello nativo di LibreYOLO il cui task sia rilevamento, segmentazione
o posa espone `track()`, quindi la scelta del rilevatore è quella di sempre.
Vedi [l'indice dei modelli](/docs/models) per l'elenco completo, oppure parti da
[YOLO9](/docs/models/yolov9), [RF-DETR](/docs/models/rf-detr),
[D-FINE](/docs/models/d-fine) o [RTMDet](/docs/models/rtmdet). I task i cui
risultati non hanno un box da associare rifiutano la chiamata invece di
restituire ID privi di significato: classificazione, box orientati, punti,
profondità, normali di superficie, bordi, segmentazione semantica e panottica,
restauro, OCR e mesh corporea sollevano tutti un errore da `track()`.

Anche due delle fasce di modelli di LibreYOLO lo rifiutano. I modelli caricati
tramite `LibreSAM` sono segmentatori di immagini, e i modelli caricati tramite
`LibreOpenVocab` sono rilevatori per singolo frame; entrambi sollevano un errore
da `track()` e si usano invece con `predict()` su ogni frame.

Il tracking gira sui modelli PyTorch nativi. Un artefatto esportato caricato con
`LibreYOLO("model.onnx")` restituisce un oggetto backend di runtime, che espone
`predict()` ma non `track()`.

La libreria include quattro tracker, selezionati dall'argomento `tracker`:

`"bytetrack"` è quello predefinito. Usa solo il movimento, con un filtro di
Kalman e un'associazione a tre fasi: prima i rilevamenti ad alta confidenza, poi
un secondo passaggio che dà ai rilevamenti a bassa confidenza la possibilità di
abbinarsi a una traccia esistente prima di essere scartati, infine le tracce non
confermate. Si configura con `TrackConfig`.

`"botsort"` mantiene il ciclo di vita a tre fasi di ByteTrack ma usa uno stato
di Kalman centro-larghezza-altezza e compensa il movimento della telecamera
nelle tracce predette prima dell'abbinamento. È la variante di BoT-SORT basata solo
sul movimento; non esegue nessun modello di aspetto. Si configura con
`BoTSortConfig`, che aggiunge `enable_cmc`, `cmc_method` e `cmc_downscale`.

Anche `"ocsort"` usa solo il movimento, e aggiunge al costo di associazione un
termine di direzione della velocità, un secondo passaggio di associazione
rispetto all'ultima osservazione reale di ogni traccia, e uno smoothing dello
stato di Kalman lungo una traiettoria virtuale quando una traccia viene
ritrovata. Si configura con `OCSortConfig`.

`"deepocsort"` estende OC-SORT con l'aspetto. Ogni traccia mantiene una media
mobile pesata per confidenza degli embedding di re-identificazione, e un termine
di similarità coseno si unisce al costo di associazione, così le identità
sopravvivono a occlusioni lunghe e a bersagli che si incrociano. Costa un
forward pass di una piccola rete di embedding per frame, e i suoi pesi OSNet
vengono scaricati al primo utilizzo. Si configura con `DeepOCSortConfig`.

## Predizione

<code-tabs name="predict" />

`track_conf` imposta la soglia per la prima fase di associazione:
`track_high_thresh` per ByteTrack e BoT-SORT, `det_thresh` per OC-SORT e Deep
OC-SORT. Non è il `conf` di `predict()`, e per ByteTrack, BoT-SORT e OC-SORT il
rilevatore gira internamente con una soglia più bassa, così i rilevamenti deboli
restano disponibili per il passaggio di recupero. Deep OC-SORT esegue il
rilevatore proprio a `det_thresh`. Per ByteTrack e BoT-SORT, `track_conf` deve
essere pari o superiore a `track_low_thresh`, che vale 0.1 per default.

Le impostazioni del tracker si possono fornire in due modi. Passa un'istanza di
config a `tracker_config=`, e il suo tipo seleziona il tracker, rendendo
`tracker=` ridondante. Oppure passa i campi come argomenti keyword e lascia che `track()`
costruisca la config per il tracker che hai indicato; le chiavi sconosciute
generano un warning invece di essere applicate in silenzio. In entrambi i casi,
`track_conf` viene ignorato quando la chiave corrispondente è impostata
esplicitamente.

Gli argomenti restanti rispecchiano quelli della predizione: `iou`, `imgsz`,
`classes`, `max_det`, `vid_stride`, `show`, e `save` con `output_path`. La
sorgente è il percorso di un file video. Vedi [predizione](/docs/predict) per la
gestione dei risultati.

## Addestramento

I tracker non si addestrano. Tre dei quattro sono modelli di puro movimento
senza alcun parametro appreso, e la rete di aspetto di Deep OC-SORT è un
checkpoint di re-identificazione pubblicato che viene scaricato al primo
utilizzo. Migliorare la qualità del tracking significa migliorare il rilevatore,
o regolare le soglie di associazione descritte sopra.
