---
title: Stima dello sguardo
seo_title: Stima dello sguardo in LibreYOLO
description: >-
  Stima pitch e yaw dello sguardo per ogni volto in LibreYOLO. Fai predizioni da
  Python o dalla CLI, leggi gli angoli in radianti ed esporta la testa dello
  sguardo in ONNX.
lead: >-
  La stima dello sguardo restituisce una direzione di sguardo per ogni volto
  presente in un'immagine. LibreYOLO la modella come un task in due fasi: prima
  gira un rilevatore di volti, poi una testa dello sguardo legge pitch e yaw da
  ogni ritaglio di volto che il rilevatore restituisce.
keywords:
  - gaze estimation python
  - eye tracking python
  - direzione dello sguardo
  - pitch yaw gaze
  - L2CS-Net
  - head pose estimation
  - stima dello sguardo libreyolo
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Senza face_detector, la predizione ricade sul rilevatore incluso
        # in OpenCV, quindi non si scarica nulla oltre al checkpoint.
        model = LibreYOLO("LibreL2CSr50.pt")
        result = model(SAMPLE_IMAGE)

        gaze = result.gaze
        print(gaze.pitch, gaze.yaw)              # radianti, una riga per volto
        print(gaze.pitch_deg, gaze.yaw_deg)      # gli stessi angoli in gradi
        print(gaze.direction_3d)                 # vettori unitari (N, 3)
    - label: CLI
      language: bash
      code: >
        # A differenza del percorso Python, la CLI non ha alcun fallback

        # automatico: i modelli gaze richiedono un rilevatore di volti

        # esplicito, e deve essere un rilevatore LibreYOLO i cui box siano
        volti.

        libreyolo predict model=LibreL2CSr50.pt source=photo.jpg
        face_detector=face-detector.pt save=True
    - label: Scegliere la sorgente dei volti
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreL2CSr50.pt")


        # Passa alla testa dello sguardo i box di un rilevatore che hai già
        eseguito.

        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])


        # Oppure indica uno dei rilevatori inclusi.

        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
source_hash: 22aa3c3d87b0c730
---

## Definizione

La stima dello sguardo restituisce due angoli per volto. `result.gaze` è un
payload `Gaze` di forma `(N, 2)`, colonna 0 il pitch e colonna 1 lo yaw, in
radianti, allineato riga per riga con `result.boxes`, i box dei volti rilevati.
La convenzione è quella usata da L2CS-Net: uno yaw positivo ruota lo sguardo
verso la sinistra del soggetto, un pitch positivo lo ruota verso il basso.

Lo stesso payload espone `pitch_deg` e `yaw_deg` per i gradi, e `direction_3d`,
un vettore unitario `(N, 3)` nel sistema di riferimento della fotocamera con
colonne `(x, y, z)`.

Poiché il task è in due fasi, una predizione dipende da due modelli. I volti che
il rilevatore non trova non hanno alcuna riga di sguardo, e i box che colloca
male producono angoli calcolati su un volto ritagliato male. La chiave canonica
del task è `gaze`; `gaze-estimation` si normalizza a questa chiave.

## Modelli

[L2CS-Net](/docs/models/l2cs) è l'unica famiglia che serve questo task. Abbina
un tronco ResNet a due teste parallele di classificazione a bin di angoli, una
per il pitch e una per lo yaw, su ritagli di volto 448x448. L'architettura
supporta cinque profondità di backbone, e una, la ResNet-50, ha un checkpoint
pubblicato.

I pesi sono soggetti a una restrizione di licenza. Sono addestrati su Gaze360, la
cui licenza consente solo l'uso a scopo di ricerca e non commerciale e vieta la
ridistribuzione, quindi per questa famiglia LibreYOLO non ospita alcun mirror.
L'unico checkpoint che la libreria può scaricare automaticamente arriva
direttamente dalla distribuzione su Google Drive degli autori stessi, tramite
`gdown`, dopo aver stampato i termini della licenza. Leggi
[L2CS-Net](/docs/models/l2cs) prima di metterlo in produzione.

Quel percorso di download richiede l'extra `gaze`:

```bash
pip install "libreyolo[gaze]"
```

Senza di esso la libreria stampa le istruzioni per il download manuale invece di
tentare il trasferimento. Fare predizioni su un checkpoint che hai già ed
esportarlo non richiede alcun extra.

## Predizione

<code-tabs name="predict" />

La sorgente dei volti si sceglie in uno di tre modi. `face_boxes` passa box che
hai già calcolato e salta il rilevamento. `face_detector` accetta `"auto"`, `"haar"`,
`"yunet"`, un modello di rilevamento LibreYOLO o un semplice callable, e può
essere impostato nel costruttore oppure per singola chiamata. Se in Python non lo
imposti, la predizione ricade sul rilevatore incluso in OpenCV, così una chiamata
essenziale funziona senza configurare nulla. Su OpenCV 4 si tratta della cascata
di Haar inclusa nel wheel, che non richiede alcun download; su OpenCV 5, dove
l'API di Haar è stata rimossa, è YuNet, che scarica una sola volta un piccolo
file di modello dallo zoo di OpenCV.

La CLI non condivide quel fallback. `libreyolo predict` rifiuta un modello gaze
senza `face_detector=`, e il valore che accetta è il nome di un rilevatore
LibreYOLO o il percorso di un checkpoint. Vedi
[predizione](/docs/predict) per le sorgenti, lo streaming e la gestione dei
risultati.

## Addestramento

Nessuna famiglia di questo task si addestra all'interno di LibreYOLO.
`LibreL2CS.train()` solleva un'eccezione: addestra nel progetto originale
L2CS-Net e carica qui lo state dict risultante.

## Validazione

La validazione su dataset con ground truth dello sguardo è fuori dallo scopo
della libreria, e `val()` solleva un'eccezione invece di restituire metriche che
non ha calcolato.
Per questo task non esiste alcun dizionario `metrics/`. Valuta a monte, sul
dataset per cui il checkpoint è stato addestrato.

## Esportazione

<code-tabs name="export" />

Il contratto di esportazione per lo sguardo copre ONNX, TorchScript, ExecuTorch,
TensorRT e OpenVINO. Dalla libreria escono soltanto il tronco ResNet e le due
teste a bin di angoli: il grafo prende un ritaglio di volto 448x448
preprocessato e restituisce i logit grezzi di yaw e pitch. Il rilevamento dei
volti, il ritaglio, la softmax, il valore atteso sui bin e la conversione in
angoli restano tutti in Python, in `libreyolo.models.l2cs.utils`. Vedi
[esportazione](/docs/export) per i formati e i loro argomenti.
