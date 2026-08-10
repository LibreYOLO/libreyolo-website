---
title: Core ML
seo_title: Esportare in Core ML da LibreYOLO
description: >-
  Esporta un rilevatore LibreYOLO in un .mlpackage di Core ML: il contratto di
  input ImageType, FP16, le compute unit, l'NMS integrato e le quattro famiglie
  supportate.
lead: >-
  Core ML è il formato di modelli on-device di Apple. LibreYOLO traccia il
  rilevatore dietro un wrapper di preprocessing specifico per ogni famiglia,
  così il grafo convertito riceve sempre un input immagine RGB canonico, e poi
  scrive un .mlpackage in formato ML Program con i metadati del modello
  allegati.
keywords:
  - esportare yolo coreml
  - mlpackage
  - coremltools
  - ct.ImageType
  - apple neural engine
  - compute_units
  - coreml nms pipeline
  - yolo su ios
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="coreml")
    mono: true
  - label: Scrive
    value: Un bundle .mlpackage (una directory) in formato ML Program
  - label: Extra
    value: 'pip install "libreyolo[coreml]"'
    mono: true
  - label: Si ricarica con
    value: LibreYOLO("weights/LibreYOLO9t.mlpackage") su macOS
    mono: true
  - label: Forme
    value: Fisse. L'input è un ct.ImageType a forma rigida.
  - label: Precisione
    value: 'FP32, FP16 (half=True). Nessun INT8.'
  - label: Famiglie
    value: 'Solo rilevamento, per yolox, yolo9, rtdetr e rfdetr'
verification: >-
  Letto da libreyolo/export/coreml.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/coreml.py e pyproject.toml sul
  branch dev.
snippets:
  install:
    - label: Installazione
      language: bash
      code: |
        pip install "libreyolo[coreml]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Scrive il bundle weights/LibreYOLO9t.mlpackage
        path = model.export(format="coreml")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml
    - label: Argomenti
      language: python
      code: |
        model.export(
            format="coreml",
            imgsz=640,
            batch=1,
            half=False,           # True converte con precisione di calcolo FLOAT16
            compute_units="all",  # all | cpu_and_gpu | cpu_and_ne | cpu_only
            output_path=None,     # None scrive weights/<stem>.mlpackage
        )

        # dynamic è accettato, ma l'input è un ct.ImageType a forma fissa,
        # e i metadati integrati registrano dynamic=False in ogni caso.
  nms:
    - label: Integrare il layer NMS di Apple
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Solo rilevamento con YOLOX e YOLO9, batch 1.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="coreml",
            nms=True,
            conf=0.25,
            iou=0.45,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml --nms \
          --conf 0.25 --iou 0.45
  run:
    - label: 'Tramite LibreYOLO, su macOS'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO(
            "weights/LibreYOLO9t.mlpackage",
            compute_units="all",   # oppure cpu_and_ne per fissare il Neural Engine
        )
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: coremltools da solo
      language: python
      code: >
        import coremltools as ct

        from PIL import Image


        mlmodel = ct.models.MLModel("weights/LibreYOLO9t.mlpackage")

        print(mlmodel.user_defined_metadata["model_family"])

        print(mlmodel.user_defined_metadata["names"])


        # L'input è un'immagine chiamata "image" alla dimensione fissa di
        esportazione.

        image = Image.open(SAMPLE_IMAGE).convert("RGB").resize((640, 640))

        out = mlmodel.predict({"image": image})

        print({name: value.shape for name, value in out.items()})


        # Su questa strada letterboxing e postprocessing sono a tuo carico.
  support:
    - label: Controllare una famiglia e un task prima di esportare
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 09c5394e3837eca2
---

## Installazione

<code-tabs name="install" />

La predizione richiede macOS. `LibreYOLO()` rifiuta un `.mlpackage` su qualsiasi altra
piattaforma con un messaggio che nomina quella corrente, e la matrice di supporto registra
queste combinazioni come disponibili perché la parità a runtime richiede un runner macOS.

## Esportazione

<code-tabs name="export" />

Il bundle viene scritto in `weights/` con lo stem del checkpoint, con `_fp16`
aggiunto in coda quando `half=True`. Un `.mlpackage` è una directory, quindi copia l'albero intero.

Ogni famiglia viene tracciata dietro un wrapper di preprocessing, così il grafo convertito
riceve un unico input canonico: RGB, `scale=1/255`, senza bias, dichiarato come
`ct.ImageType`. Il wrapper assorbe la convenzione propria della famiglia, che è BGR
nell'intervallo da 0 a 255 per YOLOX, media e deviazione standard di ImageNet per RF-DETR,
e identità per YOLO9 e RT-DETR. Ecco perché un consumatore Core ML passa
un'immagine normale invece di un tensore specifico della famiglia.

La conversione punta a ML Program con un deployment target minimo di iOS 15.
`compute_units` viene salvato sul modello convertito e può essere sovrascritto di nuovo
quando l'artefatto viene caricato.

I metadati del modello finiscono in `user_defined_metadata` come stringhe, ed è da lì che il
backend legge la famiglia, il task, i nomi delle classi, la dimensione dell'input e lo schema della posa.

### NMS integrato

<code-tabs name="nms" />

`nms=True` avvolge il modello in una pipeline Core ML che termina con il layer
`NonMaximumSuppression` di Apple. Il risultato ha due output: `confidence`, di forma
`N` per il numero di classi, e `coordinates`, di forma `N` per 4 come `xywh` normalizzato.

Vale solo per il rilevamento con YOLOX e YOLO9, e richiede batch 1. Le famiglie in
stile DETR vengono rifiutate per nome, perché la set prediction fa un top-k su
query e classi senza passaggio di IoU e non può usare quel layer. Nemmeno `max_det` è
esposto qui; quando il numero massimo di rilevamenti conta, usa invece
l'[NMS integrato di ONNX](/docs/export/onnx).

## Eseguire l'artefatto

<code-tabs name="run" />

`LibreYOLO()` riconosce una directory con suffisso `.mlpackage` e restituisce lo
stesso oggetto `Results` del checkpoint. `compute_units` è l'unico argomento che la
factory inoltra per questo formato, e accetta `all`, `cpu_and_gpu`,
`cpu_and_ne` e `cpu_only`. L'argomento `device` viene ignorato, perché Core ML
instrada attraverso le compute unit.

Il secondo snippet è la strada del runtime nudo e crudo. Lì letterboxing, decodifica, NMS e
riscalamento delle coordinate diventano affare tuo, e i nomi delle classi stanno in
`user_defined_metadata`.

## Vincoli

Quattro famiglie, solo rilevamento: `yolox`, `yolo9`, `rtdetr` e `rfdetr`. Tutto il
resto viene rifiutato in preflight, perché è il wrapper di preprocessing consapevole della
famiglia a rendere corretto il contratto di input a immagine fissa, e una famiglia fuori da
questo elenco verrebbe convertita con la normalizzazione sbagliata. L'errore indica ONNX e
TorchScript come alternative.

La forma dell'input è fissata rigidamente da `ct.ImageType`, quindi `dynamic=True` non cambia nulla
e i metadati registrano `dynamic=False`. Esporta un secondo bundle per una seconda
risoluzione.

`half=True` converte con precisione di calcolo FP16. Da questo esportatore non esiste
alcuna strada verso INT8.

Per la griglia completa di famiglie e task, vedi
[la matrice di esportazione](/docs/reference/export-matrix). Per il formato on-device più
recente di Apple, vedi [Core AI](/docs/export/coreai). Per una singola combinazione:

<code-tabs name="support" />
