---
title: ncnn
seo_title: Esportare in ncnn da LibreYOLO
description: >-
  Esporta un modello LibreYOLO in ncnn tramite PNNX: la coppia param e bin, il
  canvas di esportazione fisso, la riscrittura del Focus di YOLOX e quali
  famiglie si convertono.
lead: >-
  ncnn è la libreria di inferenza su CPU di Tencent per i target mobile.
  LibreYOLO converte tramite PNNX, scrivendo un grafo model.ncnn.param accanto a
  un file di pesi model.ncnn.bin e a un metadata.yaml che porta con sé la
  famiglia, il task e i nomi delle classi.
keywords:
  - esportare yolo ncnn
  - pnnx
  - model.ncnn.param
  - inferenza cpu mobile
  - ncnn extractor
  - focus pixel_unshuffle
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="ncnn")
    mono: true
  - label: Scrive
    value: 'Una directory con model.ncnn.param, model.ncnn.bin e metadata.yaml'
  - label: Extra
    value: 'pip install "libreyolo[ncnn]"'
    mono: true
  - label: Si ricarica con
    value: LibreYOLO("weights/LibreYOLO9t_ncnn")
    mono: true
  - label: Forme
    value: Fisse. I metadati registrano dynamic=False indipendentemente dal flag.
  - label: Precisione
    value: Solo FP32. half=True e int8=True vengono rifiutati.
verification: >-
  Letto da libreyolo/export/ncnn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/ncnn.py e pyproject.toml sul
  branch dev.
snippets:
  install:
    - label: Installazione
      language: bash
      code: |
        # pnnx converte, ncnn esegue il risultato.
        pip install "libreyolo[ncnn]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Scrive la directory weights/LibreYOLO9t_ncnn
        path = model.export(format="ncnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format ncnn --imgsz 640
    - label: Argomenti
      language: python
      code: |
        model.export(
            format="ncnn",
            imgsz=640,        # int, oppure (altezza, larghezza)
            batch=1,
            simplify=True,    # si applica solo al percorso di fallback ONNX
            opset=None,       # automatico; si applica solo al percorso di fallback ONNX
            output_path=None, # None scrive weights/<stem>_ncnn
        )

        # half=True e int8=True vengono rifiutati durante la validazione.
  run:
    - label: Tramite LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_ncnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: ncnn puro
      language: python
      code: >
        import ncnn

        import numpy as np

        import yaml


        directory = "weights/LibreYOLO9t_ncnn"

        net = ncnn.Net()

        net.load_param(f"{directory}/model.ncnn.param")

        net.load_model(f"{directory}/model.ncnn.bin")


        # ncnn accetta una singola immagine CHW, non un batch.

        mat_in = ncnn.Mat(np.zeros((3, 640, 640), dtype=np.float32))

        extractor = net.create_extractor()

        extractor.input("in0", mat_in)

        ret, mat_out = extractor.extract("out0")

        print(ret, np.array(mat_out).shape)


        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))

        print(meta["model_family"], meta["task"], meta["names"])


        # Il preprocessing e il postprocessing sono a tuo carico su questo
        percorso.
  support:
    - label: Controllare una famiglia e un task prima di esportare
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 9a849a16a3b32334
---

## Installazione

<code-tabs name="install" />

L'extra installa entrambe le metà della toolchain: `pnnx` effettua la conversione e
`ncnn` esegue il risultato. Nessuna delle due passa per ONNX sul percorso principale.

## Esportazione

<code-tabs name="export" />

L'artefatto è una directory. `weights/LibreYOLO9t_ncnn` contiene
`model.ncnn.param`, `model.ncnn.bin` e `metadata.yaml`; tutti e tre sono un unico
artefatto e si spostano insieme.

La conversione prova prima PNNX direttamente da PyTorch. Se fallisce, esporta un
grafo ONNX statico in una directory temporanea e ci lancia sopra il tool da riga di
comando `pnnx`, e l'esportazione solleva un errore solo quando falliscono entrambi i
percorsi, riportando i due errori. `opset` e `simplify` quindi influiscono solo sul
fallback.

YOLOX ha bisogno di una riscrittura per potersi convertire. Il suo layer Focus usa
slicing con stride, che PNNX non riesce ad abbassare di livello, quindi
l'esportazione lo sostituisce con `pixel_unshuffle` e permuta i canali di ingresso
della convoluzione successiva per compensare il diverso ordinamento dei canali.
L'output è numericamente identico, e i pesi originali vengono ripristinati dopo
l'esportazione.

## Eseguire l'artefatto

<code-tabs name="run" />

`LibreYOLO()` riconosce qualsiasi directory che contenga `model.ncnn.param` e
`model.ncnn.bin`, legge `metadata.yaml` e restituisce lo stesso oggetto `Results` del
checkpoint.

Il secondo snippet è il percorso a runtime puro, e due dettagli differiscono da ogni
altro formato qui presente. ncnn lavora su una singola immagine CHW anziché su un
batch, quindi non c'è un asse di batch iniziale. I nomi dei blob arrivano dal file
`.param`; PNNX scrive `in0` e `out0` per convenzione, e il backend fa il parsing del
file invece di darli per scontati. Preprocessing, decodifica, NMS e riscalatura delle
coordinate sono a tuo carico su quel percorso.

## Vincoli

FP32 su un canvas fisso. `half=True` e `int8=True` vengono entrambi rifiutati durante
la validazione, e i metadati esportati registrano `dynamic=False` qualunque cosa
dicesse il flag, così nessun backend assume un asse che il grafo non ha.

Ogni famiglia in stile DETR viene rifiutata in preflight: `detr`, `deformable_detr`,
`dinodetr`, `dfine`, `lwdetr`, `deim`, `deimv2`, `rtdetr`, `rtdetrv2`, `rtdetrv4`,
`rfdetr` e `ec`. Il messaggio è lo stesso per tutte: il modello ha bisogno di
operazioni di decoder o di sampling non disponibili in ncnn, e punta invece a ONNX,
OpenVINO, TorchScript o TensorRT.

Quello che si converte è ampio sul fronte convoluzionale: YOLO9 e YOLO9-E2E, YOLOX,
PicoDet, YOLO-NAS in rilevamento e posa, i più vecchi detector YOLO1, YOLO3, YOLO4 e
YOLO7, le quattro famiglie di classificazione CNN, la segmentazione semantica PIDNet,
il rilevamento di punti FOMO a 96 per 96 fissi, ZipDepth, NAFNet e Real-ESRGAN.

Le voci bloccate nominano il fallimento concreto. I grafi transformer di solito
lasciano dietro di sé nodi `pnnx.Expression` non supportati, il che produce una rete
senza blob di input eseguibile, ed è questo che ferma DINOv2, CLIP, SigLIP2 e
SegFormer. BiRefNet richiede la convoluzione deformabile di torchvision, che PNNX non
riesce ad abbassare di livello. Il grafo convertito di YOLO2 termina il runtime di
ncnn su Windows con una divisione intera per zero nativa durante l'estrazione
dell'output.

Per la griglia completa di famiglie e task, vedi
[la matrice di esportazione](/docs/reference/export-matrix). Per una singola
combinazione:

<code-tabs name="support" />
