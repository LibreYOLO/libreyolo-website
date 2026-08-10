---
title: Triton Inference Server
seo_title: "Servire un modello LibreYOLO su NVIDIA Triton"
description: "Servi un'esportazione ONNX di LibreYOLO tramite NVIDIA Triton: il layout del model repository, il config.pbtxt generato e le predizioni verso l'URL HTTP di un modello."
lead: "Triton Inference Server ospita un model repository e risponde alle richieste di inferenza via HTTP. LibreYOLO esporta il grafo ONNX, genera un config.pbtxt che porta con sé i metadati dell'esportazione come un unico parametro Triton, e tratta l'URL di un modello come un percorso di modello caricabile."
keywords:
  - libreyolo triton
  - triton inference server
  - config.pbtxt
  - tritonclient http
  - model repository triton
  - inferenza yolo remota
last_verified: "1.5.0"
meta:
  - label: Chiamata
    value: 'LibreYOLO("http://127.0.0.1:8000/yolo9")'
    mono: true
  - label: Helper
    value: "create_triton_config(onnx_path, config_path, model_name=..., max_batch_size=8)"
    mono: true
  - label: Extra
    value: 'pip install "libreyolo[onnx,triton]"'
    mono: true
  - label: Protocollo
    value: "Solo inferenza HTTP e HTTPS V2. Niente gRPC, autenticazione, shared memory o caricamento e scaricamento dei modelli."
  - label: Timeout
    value: "I timeout di connessione e di rete sono di 30 secondi per default"
verification: "Letto da libreyolo/backends/triton.py, libreyolo/models/__init__.py, docs/triton.md e pyproject.toml sul branch dev. I comandi del container sono quelli fissati in docs/triton.md."
snippets:
  install:
    - label: Installazione
      language: bash
      code: |
        pip install "libreyolo[onnx,triton]"
  repo:
    - label: Esportare nel layout del repository
      language: python
      code: |
        from pathlib import Path

        from libreyolo import LibreYOLO

        model_dir = Path("triton_repo/yolo9/1")
        model_dir.mkdir(parents=True, exist_ok=True)

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            output_path=str(model_dir / "model.onnx"),
            dynamic=True,
            simplify=False,
        )
    - label: Generare config.pbtxt
      language: python
      code: |
        from libreyolo import create_triton_config

        create_triton_config(
            "triton_repo/yolo9/1/model.onnx",
            "triton_repo/yolo9/config.pbtxt",
            model_name="yolo9",
            max_batch_size=8,
        )
    - label: Layout risultante
      language: text
      code: |
        triton_repo/
          yolo9/
            config.pbtxt
            1/
              model.onnx
  serve:
    - label: Avviare il server
      language: bash
      code: |
        docker run --rm --name libreyolo-triton \
          -p 8000:8000 -p 8002:8002 \
          -v "$(pwd)/triton_repo:/models:ro" \
          nvcr.io/nvidia/tritonserver:26.04-py3 \
          tritonserver --model-repository=/models --exit-on-error=true
    - label: Attendere che sia pronto
      language: bash
      code: |
        until curl --fail --silent http://127.0.0.1:8000/v2/health/ready; do sleep 1; done
    - label: Fermarlo
      language: bash
      code: |
        docker stop libreyolo-triton
  run:
    - label: Fare una predizione sul modello servito
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9")
        result = remote.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Confrontare con il modello locale
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9").predict(SAMPLE_IMAGE)
        native = LibreYOLO("LibreYOLO9t.pt").predict(SAMPLE_IMAGE)

        print(len(remote.boxes), len(native.boxes))
        print(remote.boxes.xyxy[:3])
        print(native.boxes.xyxy[:3])
    - label: Fissare una versione, o cambiare il timeout
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.backends.triton import TritonBackend

        # Un secondo segmento di percorso seleziona la versione del modello.
        # Senza, decide la version policy configurata in Triton.
        pinned = LibreYOLO("http://127.0.0.1:8000/yolo9/1")

        # I timeout di connessione e di rete sono di 30 secondi per default.
        patient = TritonBackend("http://127.0.0.1:8000/yolo9", timeout=120)
---

## Installazione

<code-tabs name="install" />

L'extra `triton` installa `tritonclient[http]`. Gli extra per gRPC e per la
shared memory sono esclusi di proposito: questa integrazione fa solo inferenza
HTTP e HTTPS V2. `onnx` serve perché sia l'artefatto servito sia il generatore
di configurazione partono da un grafo ONNX.

## Costruire il model repository

Esporta con un asse di batch dinamico, nel layout di directory che Triton si
aspetta.

<code-tabs name="repo" />

Triton non conserva i metadati personalizzati dell'ONNX nella sua risposta di
model-config, quindi i metadati completi dell'esportazione devono viaggiare in
un altro modo. `create_triton_config` li codifica come un unico parametro
stringa JSON chiamato `libreyolo_metadata` dentro `config.pbtxt`, emette le
dichiarazioni di input e output nell'ordine del grafo, gestisce l'escaping JSON
e fissa il modello a `KIND_CPU`.

L'helper valida prima di scrivere. Richiede esattamente un input del grafo ONNX,
almeno un output, forme dei tensori risolvibili e metadati la cui mappa `names`
definisce ogni indice di classe da 0 a `nc - 1`. Un modello che non supera uno
di questi controlli viene rifiutato al momento della configurazione, non alla
prima richiesta.

`max_batch_size: 8` corrisponde a un'esportazione dinamica e permette al server
di raggruppare fino a otto immagini per richiesta. Per un grafo ONNX a batch 1
fisso usa `max_batch_size=0`; LibreYOLO invia allora le immagini in sequenza.

## Avviare il server

<code-tabs name="serve" />

I comandi fissano Triton Server 26.04 e omettono di proposito i flag GPU di
Docker, dato che `KIND_CPU` nella configurazione generata impedisce comunque il
posizionamento su GPU.

## Eseguire l'artefatto

L'URL di un modello Triton è un percorso di modello. `LibreYOLO()` controlla la
presenza di uno schema `http` o `https` prima di qualsiasi gestione dei percorsi
locali e restituisce un backend che dialoga con il server, quindi il punto di
chiamata è identico a quello di un checkpoint locale, e così è anche l'oggetto
`Results` che torna indietro.

<code-tabs name="run" />

La forma dell'URL è `http(s)://host:port/model` con un segmento di versione
opzionale. La porta deve essere esplicita. Credenziali incorporate, query string
e fragment vengono tutti rifiutati, così come un percorso con più di due
segmenti.

`device` viene accettato e ignorato con una riga di log, perché il posizionamento
è una decisione del server.

## Vincoli

Il backend fallisce con un errore diretto invece di restituire un risultato
degradato quando il contratto non è rispettato: metadati LibreYOLO mancanti
nella configurazione del modello, più di un input del modello, una discrepanza
tra gli output configurati e i metadati del modello, un datatype di input non
supportato, oppure un server o un modello non pronti.

Fuori dal contratto in questa versione: gRPC, autenticazione, shared memory e
caricamento o scaricamento dei modelli tramite API.

Si può servire qualsiasi formato che Triton stesso supporti, ma qui il parametro
dei metadati e la configurazione generata hanno la forma di ONNX, quindi il
percorso di LibreYOLO è [ONNX](/docs/export/onnx) dentro il repository. Per una
pipeline video completa invece di un server request-response, vedi
[DeepStream](/docs/export/deepstream).
