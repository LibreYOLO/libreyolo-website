---
title: TensorRT
seo_title: "Esportare in TensorRT da LibreYOLO"
description: "Costruisci un engine TensorRT da un modello LibreYOLO: l'ONNX intermedio, le build FP16 e INT8, i profili a batch dinamico e i limiti di portabilità dell'engine."
lead: "TensorRT compila un grafo in un engine ottimizzato per una sola GPU. LibreYOLO esporta prima un ONNX intermedio, lo analizza con il parser ONNX di TensorRT, costruisce l'engine e scrive accanto a esso i metadati del modello come sidecar JSON."
keywords:
  - esportare yolo tensorrt
  - tensorrt engine
  - trt fp16
  - calibrazione int8 tensorrt
  - optimization profile
  - batch dinamico tensorrt
  - hardware compatibility level
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="tensorrt")'
    mono: true
  - label: Scrive
    value: "Un file .engine più un sidecar di metadati .engine.json"
  - label: Extra
    value: 'pip install "libreyolo[onnx,tensorrt]"'
    mono: true
  - label: Si ricarica con
    value: 'LibreYOLO("weights/LibreYOLO9t.engine")'
    mono: true
  - label: Forme
    value: "Statiche per default; dynamic=True aggiunge un profilo di ottimizzazione sull'asse del batch"
  - label: Precisione
    value: "FP32, FP16 (half=True), INT8 (int8=True con data=)"
  - label: Richiede
    value: "Una GPU NVIDIA sia quando costruisci sia quando esegui. Gli engine non si spostano tra architetture di GPU."
verification: "Letto da libreyolo/export/tensorrt.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/tensorrt.py e pyproject.toml sul branch dev."
snippets:
  install:
    - label: Installazione
      language: bash
      code: |
        # L'engine viene costruito da un ONNX intermedio, quindi servono entrambi gli extra.
        pip install "libreyolo[onnx,tensorrt]"
    - label: Verificare la toolchain prima di costruire
      language: bash
      code: |
        python -c "import tensorrt, torch; print(tensorrt.__version__, torch.cuda.is_available())"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Scrive weights/LibreYOLO9t_fp16.engine e weights/LibreYOLO9t_fp16.engine.json
        path = model.export(format="tensorrt", half=True)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tensorrt --half
    - label: Argomenti
      language: python
      code: |
        model.export(
            format="tensorrt",
            imgsz=640,
            batch=1,
            half=False,
            int8=False,
            data=None,                      # obbligatorio quando int8=True
            dynamic=False,
            workspace=4.0,                  # GiB di memoria di lavoro per la build
            min_batch=1,                    # limiti del profilo dinamico
            opt_batch=1,
            max_batch=8,
            hardware_compatibility="none",  # oppure "ampere_plus"
            gpu_device=0,                   # dispositivo di build su un host multi-GPU
            verbose=False,
        )
  dynamic:
    - label: Engine a batch dinamico
      language: python
      code: |
        from libreyolo import LibreYOLO

        # L'ONNX intermedio deve avere l'asse del batch dinamico, altrimenti il
        # profilo non ha nulla a cui agganciarsi.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            dynamic=True,
            min_batch=1,
            opt_batch=4,
            max_batch=8,
            half=True,
        )
  int8:
    - label: INT8 con dati di calibrazione
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            int8=True,
            data="coco128.yaml",   # obbligatorio: per questo formato non c'è un valore predefinito
            fraction=1.0,
        )
  run:
    - label: Tramite LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_fp16.engine")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: TensorRT puro
      language: python
      code: |
        import json

        import tensorrt as trt

        path = "weights/LibreYOLO9t_fp16.engine"
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(path, "rb") as handle:
            engine = runtime.deserialize_cuda_engine(handle.read())

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            print(engine.get_tensor_mode(name), name, engine.get_tensor_shape(name))

        # I nomi delle classi, il task e la dimensione di input stanno nel sidecar, non nell'engine.
        # Qui l'allocazione dei buffer, il preprocessing e il postprocessing sono a tuo carico.
        print(json.load(open(path + ".json"))["names"])
  support:
    - label: Controllare una famiglia e un task prima di costruire
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Installazione

Sia la build sia l'esecuzione richiedono una GPU NVIDIA con uno stack CUDA
funzionante. Per questo formato non esiste un ripiego su CPU.

<code-tabs name="install" />

L'extra `tensorrt` fissa `tensorrt-cu12` e `pycuda`, e il marker esclude entrambi
su macOS. Su una Jetson non usare quell'extra: fissa una build per CUDA 12 su una
piattaforma CUDA 13. Usa invece il TensorRT che installa JetPack, come descritto in
[NVIDIA Jetson](/docs/export/jetson).

## Esportazione

<code-tabs name="export" />

L'esportazione avviene in due passi. Il primo scrive un ONNX intermedio in un
percorso temporaneo, il secondo lo analizza e costruisce l'engine, e alla fine
l'intermedio viene rimosso. `workspace` è la memoria di lavoro in GiB usata
durante la build; un valore più alto permette al builder di provare più kernel e
non incide sulla memoria di inferenza.

Il sidecar dei metadati viene scritto accanto all'engine come `<engine>.json` e
registra la precisione che la build ha realizzato davvero. Quando la GPU non ha
FP16 veloce o INT8 veloce il builder avvisa e ripiega, e il sidecar riporta la
precisione che è uscita, non quella che era stata chiesta.

In FP16 un backbone ViT presente nel grafo viene riconosciuto e i suoi layer float
vengono fissati a FP32. I backbone in stile DINOv2 vanno in overflow in FP16 e
producono NaN, quindi la build imposta `OBEY_PRECISION_CONSTRAINTS` e riporta
`FP16 (FP32 ViT backbone)`. Sui backbone CNN questa passata non fa nulla.

### Batch dinamico

<code-tabs name="dynamic" />

`dynamic=True` aggiunge un profilo di ottimizzazione che va da `min_batch` a
`max_batch`, ottimizzato a `opt_batch`, e registra quei tre valori nel sidecar. Il
profilo viene aggiunto solo quando l'ONNX intermedio porta davvero una dimensione
di batch dinamica; altrimenti la build scrive nel log che sta usando
l'ottimizzazione statica e prosegue.

### INT8

<code-tabs name="int8" />

INT8 usa il calibratore a entropia di TensorRT su un loader di calibrazione
LibreYOLO, e `data` è obbligatorio: questo formato non ha un ripiego a otto
immagini. La calibrazione richiede `cuda-python` o `pycuda` per il buffer sul
dispositivo. La cache di calibrazione è indicizzata su un hash dei byte ONNX, così
le scale di un modello non vengono mai riusate per un altro che per caso scrive
sullo stesso percorso di output.

`half=True` e `int8=True` insieme producono un avviso e costruiscono in INT8, che
mantiene un ripiego FP16 per i layer che TensorRT non riesce a quantizzare.

## Eseguire l'artefatto

<code-tabs name="run" />

`LibreYOLO()` si regola sul suffisso `.engine`, legge dal sidecar i nomi delle
classi, il task e lo schema della posa, e restituisce lo stesso oggetto `Results`
del checkpoint. Solleva subito un errore quando non è presente nessun dispositivo
CUDA.

Il secondo snippet è il percorso a runtime puro. L'allocazione dei buffer su host e
su dispositivo, il preprocessing, il decoding, l'NMS e il riscalamento delle
coordinate diventano tutti a tuo carico, e l'engine di per sé non porta i nomi
delle classi, quindi il sidecar deve viaggiare insieme a esso.

## Vincoli

Un engine serializzato è legato all'architettura della GPU, allo stack dei driver e
alla versione di TensorRT che lo ha costruito. Un engine costruito su una
workstation non si carica su un'architettura diversa, ed è per questo che il passo
di build gira sulla macchina di deployment. `hardware_compatibility="ampere_plus"`
cede un po' di prestazioni in cambio della portabilità su Ampere e successive. Il
valore `"same_compute_capability"` corrisponde a `NONE` e produce un avviso:
l'engine è ottimizzato solo per la GPU corrente, e l'esportazione lo dice invece di
rivendicare una portabilità che non ha applicato.

Viene profilato solo l'asse del batch. Una build con dimensioni spaziali dinamiche
non fa parte di questo contratto, ed è per questo che FCOS è bloccato: gli servono
altezza e larghezza dinamiche con padding per preservare la sua trasformazione di
aspetto 800 per 1333.

Bloccati prima del tracing: la segmentazione YOLO9, la segmentazione RTMDet-Ins,
SSD, il rilevamento con Faster R-CNN e RetinaNet, e il matting con BiRefNet o
FeyNobg, dove TensorRT 10.16 arriva al nodo ONNX condiviso `DeformConv` e non
riesce ad analizzarlo perché `ModulatedDeformConv2d` non è presente nel registro
dei plugin.

Quando una combinazione non è né validata né bloccata, il percorso del converter è
disponibile e il progetto non ha registrato una parità di runtime TensorRT per
essa. È un'affermazione sulle prove disponibili, non sul fatto che la build
riesca.

Per la griglia completa di famiglie e task, vedi
[la matrice di esportazione](/docs/reference/export-matrix). Per una sola
combinazione:

<code-tabs name="support" />
