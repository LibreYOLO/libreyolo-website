---
title: TFLite
seo_title: Esportare in TFLite (LiteRT) da LibreYOLO
description: 'Esporta un modello LibreYOLO in un FlatBuffer .tflite con onnx2tf: dimensioni statiche, FP32 e percorsi INT8 supportati, input NHWC e metadati di runtime.'
lead: >-
  TFLite è il formato FlatBuffer che LiteRT esegue su target mobile ed embedded.
  LibreYOLO esporta un grafo ONNX statico, lo converte con onnx2tf in modalità
  flatbuffer-direct e scrive i metadati del modello accanto all'artefatto come
  sidecar JSON.
keywords:
  - esportare yolo tflite
  - litert
  - onnx2tf
  - ai-edge-litert
  - tflite flatbuffer
  - input nhwc tflite
  - inferenza su edge
last_verified: 1.6.0
meta:
- label: Flag
  value: export(format="tflite")
  mono: true
- label: Scrive
  value: Un file .tflite più un sidecar di metadati .tflite.json
- label: Extra
  value: pip install "libreyolo[tflite]"
  mono: true
- label: Si ricarica con
  value: LibreYOLO("weights/LibreYOLO9t.tflite")
  mono: true
- label: Forme
  value: Solo statiche. dynamic=True viene rifiutato.
- label: Precisione
  value: FP32; INT8 per rilevamento YOLO9 e YOLOX. FP16 viene rifiutato.
- label: Richiede
  value: Python 3.12 o superiore, perché onnx2tf 2.4.x non pubblica wheel per versioni precedenti
verification: >-
  Letto da libreyolo/export/tflite.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/tflite.py e pyproject.toml sul
  branch dev.
snippets:
  install:
    - label: Installazione
      language: bash
      code: >
        # LiteRT è il nome attuale che Google usa per TensorFlow Lite. Entrambi

        # gli extra installano lo stesso toolchain e producono lo stesso
        .tflite.

        pip install "libreyolo[tflite]"
    - label: Controllare prima la versione di Python
      language: bash
      code: |
        python -c "import sys; print(sys.version_info >= (3, 12))"

  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Scrive weights/LibreYOLO9t.tflite e weights/LibreYOLO9t.tflite.json
        path = model.export(format="tflite", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tflite --imgsz 640

        # "litert" è accettato come alias e risolve allo stesso exporter.
        libreyolo export --model LibreYOLO9t.pt --format litert --imgsz 640
    - label: Argomenti
      language: python
      code: |
        model.export(
            format="tflite",
            imgsz=640,        # int, oppure (height, width)
            batch=1,
            simplify=True,    # onnxsim sul grafo ONNX intermedio
            output_path=None, # None scrive weights/<stem>.tflite
            verbose=False,    # True mostra il log di onnx2tf in streaming
        )

        # dynamic=True genera ValueError: il convertitore richiede dimensioni statiche.
        # FP16 viene rifiutato. INT8 richiede un rilevatore supportato e dati di calibrazione.

  run:
    - label: Tramite LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.tflite")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: LiteRT puro
      language: python
      code: >
        import json


        import numpy as np

        from ai_edge_litert.interpreter import Interpreter


        interpreter = Interpreter(model_path="weights/LibreYOLO9t.tflite")

        interpreter.allocate_tensors()

        detail = interpreter.get_input_details()[0]

        print(detail["shape"], detail["dtype"])   # NHWC, non NCHW


        interpreter.set_tensor(detail["index"], np.zeros(detail["shape"],
        np.float32))

        interpreter.invoke()

        for output in interpreter.get_output_details():
            print(output["name"], interpreter.get_tensor(output["index"]).shape)

        # I nomi delle classi, il task e la dimensione di input stanno nel
        sidecar.

        meta = json.load(open("weights/LibreYOLO9t.tflite.json"))

        print(meta["model_family"], meta["task"], meta["names"])


        # Il preprocessing, la trasposizione da NCHW a NHWC e il postprocessing
        sono a tuo carico.

  support:
    - label: Controllare una famiglia e un task prima di esportare
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 3548d74e992bb76d
---

## Installazione

<code-tabs name="install" />

L'extra installa `onnx2tf` per la conversione e `ai-edge-litert` per eseguire il
risultato, entrambi dietro un marker Python 3.12. Su un interprete più vecchio
l'esportazione solleva un `ImportError` che indica il requisito di versione,
invece di fallire dentro il convertitore.

`libreyolo[litert]` installa esattamente la stessa cosa. La stringa di formato
`litert` è un alias di `tflite`, e in entrambi i casi il file prodotto è un `.tflite`.

## Esportazione

<code-tabs name="export" />

La famiglia e il task vengono controllati prima di qualsiasi altra cosa, quindi una
combinazione non supportata fallisce subito con l'errore specifico del convertitore
o del runtime che l'ha esclusa, non con un messaggio generico. La conversione vera e
propria è una chiamata in subprocess a `onnx2tf` in modalità `flatbuffer_direct` su un
intermedio ONNX statico.

I metadati stanno in un sidecar. `weights/LibreYOLO9t.tflite.json` contiene la famiglia,
il task, i nomi delle classi, la dimensione di input e lo schema della posa; il FlatBuffer
in sé non ha un campo di metadati LibreYOLO, quindi i due file viaggiano insieme.

Il rilevamento YOLO9 e YOLOX supporta `int8=True` con `data=...`, `fraction=1.0`, `batch=1` e `dynamic=False`. Installa `onnx2tf[tensorflow]`. Se mancano i dati di calibrazione, viene usato `coco8.yaml` con un avviso. Gli output separati di box normalizzati e punteggi usano scale di quantizzazione indipendenti; conserva i metadati `output_layout` del file associato durante il deployment. Alcuni operatori interni possono restare in virgola mobile.

## Eseguire l'artefatto

<code-tabs name="run" />

`LibreYOLO()` fa il dispatch sul suffisso `.tflite` e restituisce lo stesso oggetto
`Results` del checkpoint. Il backend legge il sidecar, traspone il blob NCHW in NHWC
quando l'interprete chiede un input channels-last, applica la scala e lo zero point di
quantizzazione dell'interprete dove presenti, e traspone gli output di nuovo nel layout
che il postprocessing di LibreYOLO si aspetta.

Il secondo snippet è il percorso con runtime puro. Lì il preprocessing, la trasposizione
del layout, il decoding, l'NMS e il riscalamento delle coordinate diventano a tuo carico,
e il dettaglio del layout è quello che sfugge più facilmente: onnx2tf produce input
channels-last, quindi un blob di forma `(1, 3, 640, 640)` non verrà accettato.

## Vincoli

Solo forme statiche. `dynamic=True` solleva `ValueError` prima del tracing, e il canvas
di esportazione resta fissato al valore a cui si è risolto `imgsz`.

`half=True` viene rifiutato. INT8 è limitato al rilevamento YOLO9 e YOLOX con batch 1; altre famiglie e task INT8 generano un errore.

Qui la copertura è più ristretta che per i formati a grafo, ed è decisa dalla misurazione
più che dalla famiglia. Tra le combinazioni validate ci sono il rilevamento con YOLO9,
YOLOX e YOLO-NAS, la segmentazione semantica con PIDNet, le quattro famiglie di
classificazione CNN, l'embedding con DINOv2 e SigLIP2, la classificazione con SigLIP2,
l'edge con TEED e DexiNed, e il restauro con Real-ESRGAN e SwinIR. SwinIR porta con sé un
avvertimento in più: la parità regge quando le dimensioni della sorgente coincidono
esattamente con il canvas di esportazione, e le sorgenti più piccole vengono riempite con
padding fino al canvas prima che il transformer venga eseguito, il che può divergere
dall'inferenza nativa a dimensione variabile.

Le voci bloccate indicano l'esatto punto di fallimento, che vale la pena leggere prima di
tentare una soluzione alternativa. Qualche esempio: il rilevamento con RF-DETR si converte
sul suo canvas nativo da 384 ma LiteRT non riesce ad allocarlo perché `STRIDED_SLICE` riceve
un input di rango superiore al 5-D supportato; PicoDet viene rifiutato perché un `RESHAPE`
mappa 19.200 elementi in input su 9.600 elementi in output; D-FINE fa crashare il convertitore
nella gestione delle forme di `GatherElements`; RTMDet esporta e si ricarica con la parità raw
intatta, ma i box pubblici scendono a 0.911 IoU con 29.9 px di deriva sulle coordinate.

Per la griglia completa di famiglie e task, vedi
[la matrice di esportazione](/docs/reference/export-matrix). Per una singola combinazione,
inclusa la stringa con il motivo del blocco:

<code-tabs name="support" />
