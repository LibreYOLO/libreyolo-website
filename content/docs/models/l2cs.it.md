---
title: L2CS-Net
families:
  - l2cs
seo_title: 'L2CS-Net: la stima dello sguardo in LibreYOLO'
description: >-
  Usa L2CS-Net in LibreYOLO per la stima a due stadi di pitch e yaw dello
  sguardo. Installazione, predizione ed esportazione; il checkpoint Gaze360 è
  solo per la ricerca.
lead: >-
  L2CS-Net è uno stimatore dello sguardo a due stadi: un rilevatore di volti
  individua i volti e un tronco ResNet con due teste di classificazione su bin
  angolari predice pitch e yaw per ogni volto. LibreYOLO lo integra solo per
  l'inferenza.
keywords:
  - L2CS-Net
  - gaze estimation
  - stima dello sguardo python
  - eye tracking
  - pitch yaw
  - Gaze360
  - rilevamento volti python
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Senza face_detector: si ricade sul rilevatore di volti incluso in
        # OpenCV (Haar su OpenCV 4, YuNet su OpenCV 5), quindi questo gira
        # senza download aggiuntivi oltre al checkpoint L2CS stesso.
        model = LibreYOLO("LibreL2CSr50.pt")
        result = model(SAMPLE_IMAGE)

        print(result.gaze.pitch, result.gaze.yaw)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreL2CSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Sorgente dei volti
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # Passa a L2CS i box di un rilevatore che hai già eseguito.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # Oppure indica uno specifico rilevatore di volti incluso.
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
    - label: Usare il file esportato
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        # Il grafo esportato è il solo tronco ResNet con le due teste su bin
        # angolari: prende un ritaglio di volto 448x448 già preprocessato e
        # restituisce (yaw_logits, pitch_logits) grezzi, non angoli
        # decodificati. La softmax, il valore atteso sui bin e la conversione
        # in gradi restano in Python; vedi
        # libreyolo.models.l2cs.utils.bin_logits_to_angles.
        session = ort.InferenceSession("LibreL2CSr50.onnx")
        name = session.get_inputs()[0].name
        yaw_logits, pitch_logits = session.run(
            None, {name: np.zeros((1, 3, 448, 448), dtype=np.float32)}
        )
source_hash: 4ec43f4673b4be3e
---

## Installazione

L2CS-Net non richiede nessun extra per costruire un modello, farci una
predizione o esportarlo, se ne hai già il checkpoint.

```bash
pip install libreyolo
```

L'unico checkpoint che LibreYOLO può scaricare automaticamente, una ResNet-50
addestrata su Gaze360, arriva tramite `gdown` e non da un semplice mirror HTTP,
perché si trova sul Google Drive dell'autore e non nell'organizzazione
LibreYOLO. Quel percorso richiede l'extra `gaze`:

```bash
pip install "libreyolo[gaze]"
```

Senza di esso, LibreYOLO stampa le istruzioni per il download manuale invece di
fallire in silenzio.

## Predizione

<code-tabs name="predict" />

L2CS-Net è uno stimatore a due stadi: prima viene eseguito un rilevatore di
volti, poi la testa dello sguardo legge pitch e yaw da ogni ritaglio di volto
che quello restituisce. Se non intervieni, la predizione ricade sul rilevatore
incluso in OpenCV, quindi una chiamata nuda funziona senza download aggiuntivi
una volta che hai in mano il checkpoint L2CS. `face_boxes` accetta i box di un
rilevatore che hai già eseguito; `face_detector` accetta `"auto"`, `"haar"`,
`"yunet"`, un modello di rilevamento LibreYOLO o un semplice callable.
`result.gaze` contiene pitch e yaw in radianti, allineati riga per riga con
`result.boxes`, i box dei volti rilevati. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Varianti

Cinque profondità di backbone condividono un'unica risoluzione di ingresso e
accettano gli stessi argomenti. Gaze360, il dataset dietro l'unico checkpoint
pubblicato, ha addestrato una ResNet-50; le altre quattro profondità sono
supportate a livello architetturale ma non hanno pesi pubblicati da caricare.

## Esportazione

<export-matrix />

<code-tabs name="export" />

## Licenze

<provenance-box>

LibreYOLO non ospita né fa da mirror a nessun checkpoint L2CS: per questa
famiglia non esiste nulla nell'organizzazione Hugging Face di LibreYOLO, a
differenza della maggior parte delle altre famiglie su questo sito. L'unico
checkpoint che la libreria può scaricare automaticamente arriva direttamente
dalla distribuzione su Google Drive dell'autore, protetta dall'avviso di licenza
Gaze360 stampato prima che il trasferimento inizi, e non è la copia
"ripubblicata su huggingface.co/LibreYOLO" che il riepilogo qui sopra lascia
intendere.

</provenance-box>
