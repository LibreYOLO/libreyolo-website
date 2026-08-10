---
title: ONNX
seo_title: "Esportare in ONNX da LibreYOLO"
description: "Esporta un modello LibreYOLO in ONNX: l'opset che LibreYOLO sceglie per famiglia, gli assi dinamici, l'NMS integrato, INT8 e come il grafo si ricarica."
lead: "ONNX è un formato di grafo portabile. LibreYOLO traccia il modello con torch.onnx.export, se vuoi semplifica il grafo e scrive la famiglia, il task, i nomi delle classi e la dimensione di input nei metadati del file stesso, così che qualsiasi backend LibreYOLO possa ricostruire il postprocessing."
keywords:
  - esportare yolo onnx
  - onnxruntime
  - torch.onnx.export
  - onnx opset
  - assi dinamici onnx
  - nms integrato onnx
  - onnx int8 qdq
  - onnx metadata_props
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="onnx")'
    mono: true
  - label: Scrive
    value: "Un file .onnx, con i metadati integrati nel grafo"
  - label: Extra
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: Si ricarica con
    value: 'LibreYOLO("weights/LibreYOLO9t.onnx")'
    mono: true
  - label: Forme
    value: "Batch dinamico di default in Python; eccezioni per task più sotto"
  - label: Precisione
    value: "FP32, FP16 (half=True), INT8 (int8=True, rilevamento YOLO9)"
verification: "Letto da libreyolo/export/onnx.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/onnx.py e libreyolo/cli/commands/export.py sul branch dev."
snippets:
  install:
    - label: Installazione
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Scrive weights/LibreYOLO9t.onnx
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx
    - label: Argomenti
      language: python
      code: |
        model.export(
            format="onnx",
            imgsz=640,        # int, oppure (altezza, larghezza)
            batch=1,
            dynamic=True,     # default in Python; sulla CLI il default è False
            simplify=True,    # esegue onnxsim sul grafo
            opset=None,       # None sceglie 13, o 17 per le famiglie in stile DETR
            half=False,       # pesi e attivazioni FP16
            int8=False,       # INT8 QDQ, solo rilevamento YOLO9
            data=None,        # data.yaml di calibrazione, solo INT8
            device=None,      # device per il tracing; None usa quello del modello
            output_path=None, # None scrive weights/<stem>.onnx
        )
  nms:
    - label: Integrare l'NMS nel grafo
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Solo rilevamento YOLO9, batch 1. dynamic viene forzato a False.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            nms=True,
            conf=0.25,
            iou=0.45,
            max_det=300,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx --nms \
          --conf 0.25 --iou 0.45 --max-det 300
  int8:
    - label: INT8 con dati di calibrazione
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            int8=True,
            data="coco128.yaml",   # qualche centinaio di immagini rappresentative
            fraction=1.0,
        )
  run:
    - label: Con LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.onnx")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: ONNX Runtime puro
      language: python
      code: |
        import numpy as np
        import onnx
        import onnxruntime as ort

        session = ort.InferenceSession(
            "weights/LibreYOLO9t.onnx",
            providers=["CPUExecutionProvider"],
        )

        # Su questa strada il preprocessing e il postprocessing sono a tuo carico.
        batch = np.zeros((1, 3, 640, 640), dtype=np.float32)
        outputs = session.run(None, {session.get_inputs()[0].name: batch})
        print([out.shape for out in outputs])

        # Il grafo porta con sé la famiglia, il task, i nomi delle classi e la dimensione di input.
        meta = {p.key: p.value for p in onnx.load("weights/LibreYOLO9t.onnx").metadata_props}
        print(meta["model_family"], meta["task"], meta["imgsz"])
  support:
    - label: Controllare una famiglia e un task prima di esportare
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Installazione

<code-tabs name="install" />

L'extra tira dentro `onnx`, `onnxsim` e `onnxruntime`. `onnx` da solo basta per
scrivere il file; `onnxsim` esegue il passaggio di semplificazione e
`onnxruntime` esegue l'artefatto e si occupa della calibrazione INT8.

## Esportazione

<code-tabs name="export" />

Senza `output_path`, il file finisce in `weights/` con lo stem del checkpoint, con
`_fp16` o `_int8` aggiunto in coda quando è stata richiesta quella precisione.

`dynamic` vale `True` di default in Python e `False` sulla CLI. Quando è attivo,
l'asse del batch diventa simbolico e alcuni task si aprono ancora di più: la
segmentazione semantica apre anche l'altezza e la larghezza della maschera, il
restauro Real-ESRGAN apre gli assi spaziali, e i detector a due stadi tengono
dinamiche l'altezza e la larghezza sorgente perché il loro ridimensionamento
avviene dentro il grafo.

`opset` viene scelto per famiglia quando lo ometti. Le famiglie in stile DETR
(`detr`, `deformable_detr`, `dinodetr`, `dfine`, `deim`, `deimv2`, `ec`,
`lwdetr`, `rfdetr`, `rtdetr`, `rtdetrv2`, `rtdetrv4`) più `deit`, `midas` e
`moge2` ricevono l'opset 17, che è dove viene abbassato
`aten::scaled_dot_product`. Tutto il resto riceve il 13. Il matting sale a 19 in
ogni caso, perché il decoder di BiRefNet ha bisogno dell'operatore `DeformConv`,
che ONNX definisce a partire dall'opset 19.

`simplify=True` esegue `onnxsim` e tiene il grafo originale se il passaggio
fallisce, così un errore di semplificazione è un avviso e non un fallimento
dell'esportazione. Su macOS arm64 con `onnx` 1.22 o superiore e `onnxsim` 0.6.5 o
precedente il passaggio viene saltato del tutto, perché quella combinazione può
far abortire il processo Python.

### NMS integrato

<code-tabs name="nms" />

`nms=True` vale solo per il rilevamento YOLO9 e richiede batch 1; chiederlo con
`dynamic=True` scrive un avviso nel log e disattiva dynamic. Il grafo ha allora
due output: `output`, di forma `(batch, max_det, 6)`, e `raw`, il tensore del
detector non decodificato che usa il backend di LibreYOLO, così il postprocessing
resta identico a quello del percorso PyTorch.

### DeepStream

`deepstream=True` è un'opzione solo per ONNX. Esporta il grafo nel layout che si
aspetta il parser di NVIDIA DeepStream e scrive due file di appoggio accanto ad
esso, `config_infer_primary_<stem>.txt` e `<stem>_labels.txt`, così l'artefatto
entra in una pipeline senza configurazione scritta a mano.

È mutuamente esclusiva con `nms=True`, e chiederle entrambe solleva un
`ValueError`: DeepStream esegue la soppressione nel suo stadio di clustering.
Passarla a un formato diverso da ONNX solleva a sua volta un errore. Vedi
[DeepStream](/docs/export/deepstream) per la griglia di famiglie e task
supportati e per la compilazione del parser.

### INT8

<code-tabs name="int8" />

`int8=True` esegue la quantizzazione statica di ONNX Runtime e scrive un grafo QDQ
con input e output in float32. Vengono quantizzati solo i nodi `Conv` e `Gemm`.
Lasciare in float32 il decode della testa di rilevamento è una scelta voluta:
quella concatenazione mescola coordinate dei box in scala di pixel con punteggi
di classe nell'intervallo da 0 a 1, e un'unica scala di attivazione per tensore
dominata dall'ampiezza dei box porterebbe a zero ogni punteggio.

Questo flag al momento vale solo per il rilevamento YOLO9, e qualsiasi altra cosa
solleva `NotImplementedError` nei controlli preliminari. Omettere `data` fa
ripiegare su `coco8.yaml` con un avviso; otto immagini non sono un set di
calibrazione rappresentativo. Un modello già quantizzato in PyTorch segue un'altra
strada, descritta in [Quantizzazione](/docs/export/quantization).

## Eseguire l'artefatto

<code-tabs name="run" />

`LibreYOLO()` smista in base al suffisso `.onnx` e restituisce lo stesso oggetto
`Results` di un checkpoint `.pt`, perché i nomi delle classi, il task, la
dimensione di input e lo schema della posa sono stati scritti nei
`metadata_props` del grafo al momento dell'esportazione. Con `device="auto"` la
sessione prende `CUDAExecutionProvider` quando ONNX Runtime lo segnala e
altrimenti ripiega sulla CPU.

Il secondo snippet è per chi legge senza avere LibreYOLO installato. Su quella
strada il preprocessing, il decoding, l'NMS e il riscalamento delle coordinate
diventano affare tuo; il blocco dei metadati resta comunque lì da leggere.

## Vincoli

I nomi dei tensori di output sono fissi per task, ed è ciò che il consumatore
senza metadati deve rispettare:

| Task | Nomi degli output |
|---|---|
| Rilevamento, teste a griglia e ad anchor | `output` |
| Rilevamento, stile DETR | `pred_logits`, `pred_boxes` |
| Rilevamento, RF-DETR | `dets`, `labels` |
| Classificazione | `output` |
| Segmentazione semantica | `semantic_logits` |
| Profondità | `depth` |
| Normali di superficie | `normal` |
| Contorni | `edges` |
| Restauro | `restored` |
| Matting | `matte` |
| Sguardo | `yaw_logits`, `pitch_logits` |

RF-DETR è anche l'unica famiglia il cui tensore di input si chiama `input` invece
di `images`.

In questa versione diversi task portano con sé un contratto di runtime a
risoluzione fissa. Profondità, normali di superficie e contorni rifiutano
`batch != 1` e forzano `dynamic=False`. Il matting forza il quadrato nativo da
1024, perché le tabelle di posizione relativa dello Swin di BiRefNet sono legate
alla loro risoluzione. Il restauro forza una tela fissa per ogni famiglia tranne
Real-ESRGAN, il cui generatore è completamente convoluzionale.

Un `imgsz` rettangolare funziona per le famiglie YOLO9, HRNet, NAFNet e
Real-ESRGAN. Le famiglie con un contratto di quadrato fisso (`clip`,
`deformable_detr`, `detr`, `dinodetr`, `dfine`, `deim`, `deimv2`, `ec`, `lwdetr`,
`moge2`, `rtdetr`, `rtdetrv2`, `rtdetrv4`, `rfdetr`, `siglip2`, `ssd`) lo
rifiutano del tutto.

Due combinazioni vengono rifiutate prima del tracing: la segmentazione YOLO9,
perché in LibreYOLO YOLO9 è solo rilevamento, e la segmentazione RTMDet-Ins, il
cui decode delle maschere a kernel dinamici non ha un contratto di runtime
esportato.

Per la griglia completa di famiglie e task, vedi
[la matrice di esportazione](/docs/reference/export-matrix). Per una singola
combinazione, chiedi direttamente alla libreria:

<code-tabs name="support" />
