---
title: Hailo
seo_title: "Eseguire modelli LibreYOLO sugli acceleratori Hailo"
description: "Metti in produzione un modello LibreYOLO su Hailo-8 o Hailo-8L: l'esportazione ONNX statica, la fase di Dataflow Compiler che esegui tu e quali architetture si compilano."
lead: "Gli acceleratori Hailo si compilano con l'Hailo Dataflow Compiler, un SDK proprietario distribuito attraverso la Developer Zone di Hailo. La parte del flusso che spetta a LibreYOLO è una semplice esportazione ONNX statica; il parsing, la quantizzazione e la compilazione in un HEF avvengono dopo, dentro il DFC."
keywords:
  - libreyolo hailo
  - hailo-8
  - hailo-8l
  - raspberry pi ai kit
  - ai hat+
  - hailo dataflow compiler
  - compilare hef
  - hailortcli
last_verified: "1.5.0"
meta:
  - label: Passo LibreYOLO
    value: 'export(format="onnx", imgsz=640, dynamic=False)'
    mono: true
  - label: Non è un formato
    value: 'Non esiste format="hef". Il DFC non può essere una dipendenza pip.'
  - label: Extra
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: Host di compilazione
    value: "Linux x86_64, inclusa WSL2 Ubuntu 22.04. La compilazione non può girare su ARM."
  - label: Cosa si compila
    value: "Grafi puramente CNN, a forma fissa. L'attenzione, le forme dinamiche e i design dominati da LayerNorm no."
  - label: Stato
    value: "Nessuna famiglia LibreYOLO è ancora stata portata da un capo all'altro del DFC fino a un HEF funzionante."
verification: "Letto da skills/libreyolo-export-hailo/SKILL.md, libreyolo/export/onnx.py e libreyolo/cli/commands/export.py sul branch dev. I vincoli del DFC sono quelli registrati in quella skill; nessun HEF LibreYOLO è stato compilato e misurato."
snippets:
  install:
    - label: Lato LibreYOLO
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: Lato Hailo, lo installi tu
      language: text
      code: |
        Prerequisites, none of them installable from PyPI:

        - A Linux x86_64 machine. WSL2 Ubuntu 22.04 works. The Raspberry Pi is a
          runtime target, never the compile host.
        - The Dataflow Compiler wheel (hailo_sdk_client) from the Hailo Developer
          Zone, which is free to register for.
        - For Hailo-8 and Hailo-8L, the Hailo Model Zoo v2.x line, for its
          recipes and NMS configurations.
        - A GPU on the compile host is strongly recommended: the quantization
          step takes hours without one.
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Hailo richiede batch 1, una risoluzione fissa e nessun asse dinamico.
        # L'API Python usa dynamic=True di default, quindi disattivalo esplicitamente.
        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    - label: CLI
      language: bash
      code: |
        # La CLI usa già forme statiche di default.
        libreyolo export --model LibreYOLOXs.pt --format onnx --imgsz 640
    - label: Verificare che il grafo sia statico prima di compilare
      language: python
      code: |
        import onnx

        graph = onnx.load("weights/LibreYOLOXs.onnx").graph
        shape = graph.input[0].type.tensor_type.shape
        print([d.dim_value or d.dim_param for d in shape.dim])
  compile:
    - label: Parsing, quantizzazione e compilazione
      language: python
      code: |
        from pathlib import Path

        import numpy as np
        from hailo_sdk_client import ClientRunner
        from PIL import Image

        ONNX = "weights/LibreYOLOXs.onnx"
        HW_ARCH = "hailo8"     # hailo8 | hailo8l | hailo10h
        IMGSZ = 640

        runner = ClientRunner(hw_arch=HW_ARCH)

        # Per YOLOX, esegui la traduzione una volta senza end_node_names: il log
        # del DFC stampa i nodi finali che suggerisce. Rilancia con quelli.
        runner.translate_onnx_model(ONNX)

        # La normalizzazione deve corrispondere al preprocessing di LibreYOLO.
        # YOLOX e YOLO9 non richiedono media né deviazione standard, solo la
        # scala da 0-255 a 0-1.
        script = "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n"

        # Facoltativo: lascia la NMS a Hailo. La configurazione è specifica sia
        # del numero di classi sia della dimensione di input, quindi una
        # configurazione COCO-80 è sbagliata per un modello a tre classi
        # affinato con fine-tuning. Senza questa riga l'HEF emette i tensori
        # grezzi della testa e li decodifica l'applicazione.
        # script += 'nms_postprocess("yolox_nms_config.json", meta_arch=yolox, engine=cpu)\n'

        runner.load_model_script(script)

        # Le immagini di calibrazione devono essere rappresentative dei dati di
        # produzione. Immagini casuali si compilano e distruggono in silenzio
        # l'accuratezza.
        calib_paths = sorted(Path("calib_images").glob("*.jpg"))[:128]
        calib = np.stack([
            np.asarray(
                Image.open(p).convert("RGB").resize((IMGSZ, IMGSZ)),
                dtype=np.float32,
            )
            for p in calib_paths
        ])

        runner.optimize(calib)
        Path("libreyoloxs.hef").write_bytes(runner.compile())
    - label: Nodi finali di YOLO9
      language: python
      code: |
        # I grafi LibreYOLO usano un prefisso "/head/...", non il prefisso
        # "model.N" che si vede nelle configurazioni scritte per altre
        # esportazioni. Una configurazione copiata non corrisponde. Se il
        # parsing fallisce, verifica i nomi nel tuo grafo.
        END_NODES = [
            "/head/cv2.0/cv2.0.2/Conv", "/head/cv3.0/cv3.0.2/Conv",
            "/head/cv2.1/cv2.1.2/Conv", "/head/cv3.1/cv3.1.2/Conv",
            "/head/cv2.2/cv2.2.2/Conv", "/head/cv3.2/cv3.2.2/Conv",
        ]
        runner.translate_onnx_model(ONNX, end_node_names=END_NODES)
  device:
    - label: Raspberry Pi 5 con l'AI Kit o l'AI HAT+
      language: bash
      code: |
        sudo apt install dkms hailo-all
        hailortcli fw-control identify       # controllo del dispositivo, e dice qual è l'architettura
        hailortcli run libreyoloxs.hef       # smoke test e throughput
---

## Installazione

In LibreYOLO non esiste `format="hef"` e non esisterà. L'Hailo Dataflow Compiler
è un SDK proprietario distribuito come wheel privata dietro la registrazione alla
Developer Zone, quindi non può essere una dipendenza né un extra. Il deployment
ha due fasi: LibreYOLO scrive un file ONNX statico, e sei tu a passarci sopra il
DFC.

```text
Libre<Model>.pt  ->  ONNX  ->  HAR (parse)  ->  HAR (quantize INT8)  ->  HEF
                 [libreyolo]           [Hailo DFC, installed by you]
```

<code-tabs name="install" />

## Esportazione

<code-tabs name="export" />

Non passare `half=True`. Il DFC prende in ingresso ONNX FP32 e fa la propria
quantizzazione INT8. Non passare nemmeno `nms=True`: o la NMS è in carico a Hailo
tramite `nms_postprocess`, o è in carico all'applicazione, e un sottografo di NMS
è peso morto oltre i nodi finali. L'opset predefinito funziona; se il parser del
DFC protesta, riesporta con `opset=11`.

Il DFC taglia il grafo ai nodi finali che indichi, cioè le convoluzioni della
testa di rilevamento, e scarta tutto quello che sta a valle. L'ONNX decodificato
ordinario di LibreYOLO è quindi un input accettabile: la coda di decodifica viene
semplicemente ignorata dal parser.

## Compilazione

<code-tabs name="compile" />

Scegli `hw_arch` in base al target: `hailo8` per Hailo-8, l'AI HAT+ da 26 TOPS e
i moduli M.2 e PCIe; `hailo8l` per Hailo-8L, il Raspberry Pi AI Kit e l'AI HAT+
da 13 TOPS; `hailo10h` per Hailo-10H, che richiede un DFC e un Model Zoo più
recenti e corrispondenti. Se hai dubbi, `hailortcli fw-control identify` sul
dispositivo risponde alla domanda.

Due famiglie corrispondono a una meta-architettura NMS di HailoRT, quindi la
soppressione può stare in carico a Hailo dentro la pipeline compilata: YOLOX
tramite `meta_arch=yolox`, e YOLO9 tramite la meta-architettura a testa
disaccoppiata di Hailo, la cui disposizione della testa è identica. Prendi la
configurazione `nms_postprocess` corrispondente dal Model Zoo di Hailo e adattala
al tuo numero di classi e alla tua dimensione di input. Ogni altro rilevatore
convoluzionale si compila come un grafo senza meta-architettura corrispondente:
l'HEF emette i tensori grezzi della testa e l'applicazione esegue decodifica e
NMS sulla CPU.

Conserva il log di compilazione quando qualcosa fallisce. Ogni correzione dipende
dal nome esatto del layer o dell'operatore che fallisce.

## Eseguire l'artefatto

<code-tabs name="device" />

L'inferenza dell'applicazione usa l'API Python `hailo_platform`. Con
`nms_postprocess` compilato dentro, l'output è `(batch, num_classes, max_dets, 5)`
e contiene `[y1, x1, y2, x2, score]` in coordinate del modello, che riporti in
scala sull'immagine di origine tu stesso. La pipeline `Results` di LibreYOLO non
entra in gioco a runtime; l'HEF è un artefatto autonomo, e il preprocessing e il
postprocessing sono dell'applicazione.

## Vincoli

Che un modello possa avere come target Hailo-8 o Hailo-8L è una proprietà della
sua architettura, non del suo nome, quindi la regola qui sotto vale anche per le
famiglie aggiunte dopo che questa pagina è stata scritta.

Un modello non si compila se contiene una qualsiasi di queste cose:

- Attenzione di qualunque tipo: self, cross, deformabile o a finestre. Questo
  esclude ogni rilevatore in stile DETR, ogni rilevatore a vocabolario aperto o
  condizionato dal testo, ogni backbone ViT e ogni torre linguistica o
  vision-language. Lo zoo di Hailo distribuisce qualche HEF transformer messo a
  punto a mano; è lavoro su misura del fornitore e non è la prova che un grafo
  con attenzione qualsiasi si compili.
- Forme dinamiche o flusso di controllo che dipende dai dati. Il DFC compila una
  sola forma di input fissa e un grafo statico, quindi numeri di query variabili,
  prompt testuali, top-k dinamico, `NonZero`, `Gather` o `TopK` con indici
  dinamici e `grid_sample` sono tutti esclusi.
- Un design dominato da LayerNorm o da GELU. La BatchNorm si fonde con le
  convoluzioni in modo pulito; il supporto per LayerNorm è scarso e GELU non è
  un'attivazione nativa, quindi uno stack in stile ConvNeXt non è adatto anche se
  è nominalmente convoluzionale.
- Lavoro image-to-image a risoluzione nativa. I modelli di restauro girano alla
  risoluzione piena dell'input e superano i budget pratici di SRAM di Hailo.

Una famiglia è candidata quando è di sole convoluzioni, usa BatchNorm con ReLU o
SiLU e ha una dimensione di input fissa. In questa libreria vuol dire i
rilevatori CNN a stadio singolo, con YOLOX e YOLO9 come target principali; gli
altri rilevatori convoluzionali come PicoDet, YOLO-NAS e RTMDet, con la
decodifica lato applicazione; i classificatori CNN ResNet, MobileNetV4-conv ed
EfficientNetV2, di cui ResNet è il meglio supportato perché il Model Zoo di Hailo
distribuisce ricette per esso; e le piccole teste di task convoluzionali come il
rilevamento a punti FOMO e lo sguardo L2CS su un backbone ResNet, compilabili in
linea di principio ma senza una ricetta Hailo.

Un avvertimento sullo stato, che è il motivo per cui nulla in questa pagina viene
presentato come supportato: nessuna famiglia LibreYOLO è stata portata da un capo
all'altro del DFC fino a un HEF funzionante. Le regole qui sopra prevedono la
compilabilità a partire dall'architettura. Il comportamento del parser, la
quantizzazione e l'accuratezza restano non dimostrati finché un HEF non viene
compilato e misurato, quindi considera che ogni candidato richieda le proprie
prove registrate: un HEF compilato a partire dal checkpoint esatto, con le
versioni di DFC, Model Zoo e HailoRT annotate, una calibrazione documentata e un
confronto di accuratezza sul dispositivo rispetto alla baseline FP32, non un
numero di throughput.

Se il modello è escluso, le alternative sono i runtime con parità registrata:
[ONNX](/docs/export/onnx), [TensorRT](/docs/export/tensorrt) e
[OpenVINO](/docs/export/openvino).
