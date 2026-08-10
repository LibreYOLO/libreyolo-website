---
title: MNN
seo_title: "Esportare in MNN da LibreYOLO"
description: "Esporta un detector LibreYOLO in MNN passando per ONNX e mnnconvert: una forma NCHW fissa, FP32 su CPU e un sidecar di metadati richiesto dal contratto di runtime."
lead: "MNN è il motore di inferenza leggero di Alibaba. LibreYOLO esporta un grafo ONNX statico, lo converte con lo strumento mnnconvert incluso nel pacchetto MNN e scrive un sidecar JSON che registra i nomi di input e output, la forma di input fissa e i nomi delle classi."
keywords:
  - esportare yolo mnn
  - mnnconvert
  - inferenza mnn
  - inferenza detector su mobile
  - forma nchw fissa
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="mnn")'
    mono: true
  - label: Scrive
    value: "Un file .mnn più un sidecar di metadati .mnn.json"
  - label: Extra
    value: 'pip install "libreyolo[mnn]"'
    mono: true
  - label: Si ricarica con
    value: 'LibreYOLO("weights/LibreYOLO9t.mnn")'
    mono: true
  - label: Forme
    value: "NCHW fissa. dynamic=True viene rifiutato."
  - label: Precisione
    value: "Solo FP32, solo CPU."
  - label: Task
    value: "Solo rilevamento in questa versione"
verification: "Letto da libreyolo/export/mnn.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/mnn.py e pyproject.toml sul branch dev."
snippets:
  install:
    - label: Installazione
      language: bash
      code: |
        # L'extra include libreyolo[onnx]: MNN converte da un intermedio ONNX.
        pip install "libreyolo[mnn]"
    - label: Verificare che il convertitore sia nel PATH
      language: bash
      code: |
        mnnconvert --version
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Scrive weights/LibreYOLO9t.mnn e weights/LibreYOLO9t.mnn.json
        path = model.export(format="mnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format mnn --imgsz 640
    - label: Argomenti
      language: python
      code: |
        model.export(
            format="mnn",
            imgsz=640,        # int, oppure (altezza, larghezza)
            batch=1,          # incorporato nell'artefatto
            simplify=True,    # onnxsim sull'intermedio ONNX
            output_path=None, # None scrive weights/<stem>.mnn
            verbose=False,    # True mostra il log di mnnconvert
        )

        # dynamic=True solleva ValueError. half=True e int8=True vengono rifiutati.
  run:
    - label: Tramite LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.mnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: MNN puro
      language: python
      code: |
        import json

        import MNN
        import numpy as np

        meta = json.load(open("weights/LibreYOLO9t.mnn.json"))
        print(meta["mnn_input_names"], meta["mnn_output_names"], meta["mnn_input_shape"])

        runtime = MNN.nn.create_runtime_manager(
            ({"backend": 0, "precision": 1, "numThread": 4},)
        )
        module = MNN.nn.load_module_from_file(
            "weights/LibreYOLO9t.mnn",
            meta["mnn_input_names"],
            meta["mnn_output_names"],
            runtime_manager=runtime,
            dynamic=False,
            shape_mutable=False,
        )

        blob = np.zeros(meta["mnn_input_shape"], dtype=np.float32)
        input_var = MNN.expr.const(
            blob, list(blob.shape), MNN.expr.NCHW, MNN.expr.float
        )
        outputs = module.forward([input_var])
        for out in outputs:
            print(np.array(MNN.expr.convert(out, MNN.expr.NCHW).read()).shape)

        # Il preprocessing e il postprocessing sono a tuo carico su questo percorso.
  support:
    - label: Controllare una famiglia e un task prima di esportare
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Installazione

<code-tabs name="install" />

L'extra include `libreyolo[onnx]`, perché la conversione passa per un intermedio
ONNX. Porta con sé anche l'eseguibile `mnnconvert`, che l'exporter cerca prima
accanto all'interprete Python attivo e poi nel `PATH`. Se il convertitore manca
viene sollevato un `ImportError` che indica il comando di installazione, invece di
fallire a metà conversione.

## Esportazione

<code-tabs name="export" />

Prima di passare il grafo, l'exporter legge il contratto di input ONNX e rifiuta
tutto ciò che non può esprimere: più di un input immagine, oppure una forma di
input con una dimensione simbolica. In questa versione MNN richiede una forma NCHW
completamente fissa, e `batch` viene incorporato nell'artefatto anziché negoziato
al caricamento.

Il sidecar non è contabilità opzionale. `weights/LibreYOLO9t.mnn.json` registra i
nomi di input e output, la forma di input fissa, il batch, i nomi delle classi, la
versione di MNN usata e il backend per cui l'artefatto è stato costruito, e il
runtime valida ognuno di questi campi al caricamento.

Su Windows, MNN 3.6.1 a volte completa la conversione e poi termina durante la
chiusura del processo con una violazione di accesso o uno stato di fail-fast.
L'exporter riconosce quegli specifici codici di uscita e considera riuscita la
conversione quando il file di output è presente.

## Eseguire l'artefatto

<code-tabs name="run" />

`LibreYOLO()` smista in base al suffisso `.mnn` e restituisce lo stesso oggetto
`Results` del checkpoint. Il caricamento è severo per scelta: il sidecar deve
dichiarare `format=mnn`, `mnn_backend=cpu`, `dynamic=false`, `precision=fp32`, una
dimensione, un task di rilevamento, una forma NCHW fissa e positiva che concordi
con la dimensione immagine registrata, e nomi di classe che coprano ogni indice da
0 a `nc - 1`. Qualsiasi discrepanza solleva un errore invece di tirare a indovinare.

Anche fare una predizione con un `imgsz` diverso da quello per cui l'artefatto è
stato costruito solleva un errore, e `device` viene ignorato con un avviso, perché
qui le esportazioni MNN girano su CPU.

Il secondo snippet è il percorso a runtime puro. Lì il preprocessing, il decoding,
l'NMS e il riscalamento delle coordinate diventano a tuo carico, e i nomi di input
e output arrivano dal sidecar perché il module loader di MNN li vuole espliciti.

## Vincoli

Solo rilevamento. Il backend rifiuta qualsiasi altro task al caricamento, e il lato
esportazione fa lo stesso: al di fuori delle combinazioni registrate, il preflight
solleva un errore con "MNN v1 has no implemented runtime contract for this family
and task."

FP32, CPU, forma fissa. `dynamic=True` solleva `ValueError`, e `half=True` e
`int8=True` vengono rifiutati durante la validazione.

Le famiglie di rilevamento validate sono YOLO9, YOLO9-E2E, YOLO9-P2, RF-DETR, EC,
RT-DETR, RT-DETRv2, RT-DETRv4, D-FINE, DEIM e YOLO-NAS, ciascuna coperta da
conversione, ricaricamento di un artefatto appena creato, esecuzione MNN su CPU,
controlli sui metadati e parità di rilevamento post-NMS con corrispondenza rispetto
al modello PyTorch. DEIMv2 converte, si ricarica, viene eseguito e preserva i
rilevamenti post-NMS, ma il suo percorso ONNX intermedio ha una parità di punteggio
a livello di query incompleta, quindi è registrato come disponibile anziché
validato.

Per la griglia completa di famiglie e task, vedi
[la matrice di esportazione](/docs/reference/export-matrix). Per una singola
combinazione:

<code-tabs name="support" />
