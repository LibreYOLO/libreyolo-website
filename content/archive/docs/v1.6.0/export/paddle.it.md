---
title: Paddle
seo_title: Esportare in PaddlePaddle da LibreYOLO
description: >-
  Converti un rilevatore LibreYOLO in un modello di inferenza PaddlePaddle
  tramite X2Paddle: la toolchain fissata, i grafi statici FP32 con batch 1 e
  l'inferenza su CPU.
lead: >-
  I modelli di inferenza PaddlePaddle sono un grafo model.pdmodel accanto a un
  file di pesi model.pdiparams. LibreYOLO esporta un grafo ONNX statico con
  opset 15, lo converte con X2Paddle e impacchetta il risultato con un
  metadata.yaml, così si carica attraverso la stessa factory di ogni altro
  runtime.
keywords:
  - esportare yolo paddle
  - inferenza paddlepaddle
  - x2paddle
  - model.pdmodel
  - model.pdiparams
  - onnx opset 15
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="paddle")
    mono: true
  - label: Scrive
    value: 'Una directory con model.pdmodel, model.pdiparams e metadata.yaml'
  - label: Extra
    value: 'pip install "libreyolo[paddle]"'
    mono: true
  - label: Si ricarica con
    value: 'LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")'
    mono: true
  - label: Backend
    value: libreyolo.backends.paddle.PaddleBackend
    mono: true
  - label: Forme
    value: 'Statiche, batch 1, opset 15. Tutti e tre i vincoli sono imposti.'
  - label: Precisione
    value: 'Solo FP32, solo CPU.'
  - label: Toolchain
    value: >-
      PaddlePaddle 2.6.2, X2Paddle 1.6.0, ONNX 1.17 o precedente, controllati in
      modo esatto
verification: >-
  Letto da libreyolo/export/paddle.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/paddle.py, docs/paddle.md e
  pyproject.toml sul branch dev.
snippets:
  install:
    - label: Installazione
      language: bash
      code: >
        # Python da 3.10 a 3.12. WSL2 con Ubuntu 22.04 è il percorso validato su
        Windows.

        pip install "libreyolo[paddle]"
    - label: Confermare le versioni fissate
      language: bash
      code: >
        python -c "from importlib.metadata import version;
        print(version('paddlepaddle'), version('x2paddle'), version('onnx'))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Scrive la directory weights/LibreYOLO9t_paddle
        path = model.export(format="paddle")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format paddle
    - label: Argomenti
      language: python
      code: |
        model.export(
            format="paddle",
            imgsz=640,        # int; il canvas quadrato di questa famiglia
            batch=1,          # qualsiasi altro valore solleva ValueError
            dynamic=False,    # True solleva ValueError
            simplify=True,    # False solleva ValueError
            opset=15,         # qualsiasi altro valore solleva ValueError
            output_path=None, # None scrive weights/<stem>_paddle
        )
  run:
    - label: Tramite LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: CLI
      language: bash
      code: |
        libreyolo predict --model weights/LibreYOLO9t_paddle \
          --source https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg --device cpu --save
    - label: Il backend direttamente
      language: python
      code: |
        from libreyolo.backends.paddle import PaddleBackend

        # Ciò che LibreYOLO() costruisce per una directory Paddle. Stesso
        # oggetto Results, senza il routing della factory in mezzo.
        backend = PaddleBackend("weights/LibreYOLO9t_paddle", device="cpu")
        result = backend.predict("parkour.jpg")
        print(result.boxes.xyxy[:3])
    - label: Paddle puro
      language: python
      code: >
        import numpy as np

        import paddle.inference as paddle_infer

        import yaml


        directory = "weights/LibreYOLO9t_paddle"

        config = paddle_infer.Config(
            f"{directory}/model.pdmodel", f"{directory}/model.pdiparams"
        )

        config.disable_gpu()

        config.disable_mkldnn()

        config.switch_ir_optim(False)


        predictor = paddle_infer.create_predictor(config)

        handle = predictor.get_input_handle(predictor.get_input_names()[0])

        handle.reshape([1, 3, 640, 640])

        handle.copy_from_cpu(np.zeros((1, 3, 640, 640), dtype=np.float32))

        predictor.run()

        for name in predictor.get_output_names():
            print(name, predictor.get_output_handle(name).copy_to_cpu().shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))

        print(meta["model_family"], meta["task"], meta["names"])


        # Su questo percorso il preprocessing e il postprocessing sono a tuo
        carico.
  support:
    - label: Controllare una famiglia e un task prima di esportare
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cdd8bf12286e2f53
---

## Installazione

<code-tabs name="install" />

L'extra fissa esattamente lo stack su cui è stata misurata la parità:
PaddlePaddle 2.6.2, X2Paddle 1.6.0 e ONNX 1.17 o precedente. Quei vincoli di
versione sono controllati al momento dell'esportazione, non solo
all'installazione, e una versione diversa solleva un `ImportError` che indica
quella attesa. Le release più recenti di Paddle rifiutano parti del codice
statico generato da X2Paddle 1.6.0, quindi fallire subito è meglio che produrre
un artefatto che nessuno ha validato.

## Esportazione

<code-tabs name="export" />

Quattro argomenti sono fissi, non semplicemente predefiniti. `dynamic` deve
essere `False`, `batch` deve essere 1, `simplify` deve essere `True` per un grafo
di conversione completamente statico e `opset` deve essere 15, che è il massimo
accettato da X2Paddle 1.6.0. Passare qualsiasi altra cosa solleva un errore prima
del tracing.

Sul grafo intermedio viene eseguita una normalizzazione. ONNX definisce come uno
la dilation di MaxPool quando è omessa, PyTorch scrive esplicitamente l'attributo
con tutti uno e X2Paddle 1.6.0 lo rifiuta, quindi l'exporter rimuove quel valore
predefinito ridondante e lascia invariata l'operazione specificata.

L'artefatto è una directory: `model.pdmodel`, `model.pdiparams` e
`metadata.yaml`. Il codice Python che X2Paddle genera durante la conversione non
ne fa parte.

## Eseguire l'artefatto

<code-tabs name="run" />

`LibreYOLO()` riconosce qualsiasi directory che contenga sia `model.pdmodel` sia
`model.pdiparams`, legge `metadata.yaml` e restituisce lo stesso oggetto
`Results` del checkpoint. Un device diverso da `auto` o `cpu` solleva un errore:
questo backend funziona solo su CPU.

Quello che la factory costruisce è `PaddleBackend`, esportato da `libreyolo` e
importabile come `libreyolo.backends.paddle.PaddleBackend`. Costruiscilo tu
quando vuoi il backend senza il routing per suffisso della factory, per esempio
per passare `task=` in modo esplicito per una directory il cui `metadata.yaml`
non hai scritto tu. Il suo `predict()` accetta le stesse sorgenti e restituisce
gli stessi risultati.

Lo snippet con il runtime puro rispecchia ciò che configura il backend, e le tre
opzioni disabilitate sono volute. La pipeline di fusione su CPU di Paddle 2.6 può
andare in crash mentre ottimizza i grandi grafi di gather e scatter emessi per la
deformable attention, quindi il grafo statico portabile e non fuso è quello su
cui è stata misurata la parità. Su quel percorso il preprocessing, il decoding,
l'NMS e il riscalamento delle coordinate diventano a tuo carico.

## Vincoli

Niente forme dinamiche, niente FP16, niente INT8, niente NMS integrato, niente
runtime su GPU.

Le combinazioni validate sono il rilevamento con YOLO9, il rilevamento con
YOLO9-E2E e YOLO9-P2, il rilevamento, la stima della posa e la segmentazione con
EC, il rilevamento con RT-DETRv4, D-FINE, DEIM e DEIMv2, e il rilevamento e la
stima della posa con YOLO-NAS. Ognuna è coperta dalla conversione, da una
ricarica del runtime su CPU, dalla parità degli output grezzi e da risultati
pubblici corrispondenti.

Bloccate, con la ragione registrata per ogni combinazione:

| Combinazione | Perché |
|---|---|
| RF-DETR, tutti i task | Richiede ONNX opset 17 e GridSample; X2Paddle 1.6.0 accetta opset 15 o inferiore e non ha un mapper per GridSample |
| Rilevamento con RT-DETR e RT-DETRv2 | I grafi addestrati richiedono GridSample con opset 16 o superiore |
| Segmentazione con D-FINE | Converte e si ricarica, ma l'errore RMS relativo sui logit delle maschere è 3.52% e la IoU minima tra maschere corrispondenti è 0.582 |
| Segmentazione con YOLO9 | In LibreYOLO YOLO9 fa solo rilevamento |
| Segmentazione con RTMDet-Ins | Il decode delle maschere a kernel dinamico non ha un contratto per il runtime esportato |

Tutto ciò che non è elencato come validato o bloccato viene rifiutato con la nota
che non è stato validato attraverso il percorso di conversione da ONNX a Paddle.

Per la griglia completa di famiglie e task, vedi
[la matrice di esportazione](/docs/reference/export-matrix). Per una singola
combinazione:

<code-tabs name="support" />
