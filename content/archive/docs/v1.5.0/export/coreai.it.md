---
title: Core AI
seo_title: Esportare in Apple Core AI da LibreYOLO
description: >-
  Esporta un modello LibreYOLO in un asset .aimodel di Apple Core AI: solo
  macOS, canvas fisso, FP32 e il contratto sull'ordine degli output con nome che
  i consumatori devono rispettare.
lead: >-
  Core AI è lo stack di inferenza on-device di Apple. LibreYOLO cattura il
  modello con torch.export, lo abbassa attraverso il convertitore di Core AI e
  scrive un asset .aimodel che porta con sé i metadati del modello e i nomi
  degli output esportati.
keywords:
  - esportare libreyolo core ai
  - aimodel
  - coreai-torch
  - torch.export apple
  - inferenza on-device apple
  - coreai_output_names
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="coreai")
    mono: true
  - label: Scrive
    value: Un asset .aimodel con i metadati allegati
  - label: Extra
    value: 'pip install "libreyolo[coreai]"'
    mono: true
  - label: Si ricarica
    value: >-
      Non tramite LibreYOLO. I consumatori usano direttamente il runtime di Core
      AI.
  - label: Forme
    value: Canvas fisso. dynamic=True solleva NotImplementedError.
  - label: Precisione
    value: Solo FP32. half=True e int8=True vengono rifiutati.
  - label: Richiede
    value: >-
      macOS. La toolchain non converte né esegue altrove, e coreai-torch fissa
      torch a 2.11.x.
verification: >-
  Letto da libreyolo/export/coreai.py, libreyolo/export/coreai_compat.py,
  libreyolo/export/exporter.py, libreyolo/export/support.py e pyproject.toml sul
  branch dev.
snippets:
  install:
    - label: 'Installazione, su macOS'
      language: bash
      code: |
        # Tenuto fuori da ogni extra aggregato di proposito: coreai-torch fissa
        # torch a 2.11.x e trascinerebbe l'intero ambiente su quella versione.
        pip install "libreyolo[coreai]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Scrive weights/LibreYOLO9t.aimodel
        path = model.export(format="coreai", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreai --imgsz 640
    - label: Argomenti
      language: python
      code: |
        model.export(
            format="coreai",
            imgsz=640,        # int, o (altezza, larghezza); questo è il canvas di esecuzione
            batch=1,
            output_path=None, # None scrive weights/<stem>.aimodel
        )

        # dynamic=True solleva NotImplementedError.
        # half=True e int8=True vengono rifiutati durante la validazione.
  outputs:
    - label: Leggi l'ordine degli output prima di collegare un consumatore
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="coreai", imgsz=640)

        # I metadati dell'asset registrano i nomi degli output esportati, in
        # ordine di grafo, sotto "coreai_output_names". Mappa per nome il
        # dizionario restituito da Core AI usando quella lista; non accoppiarlo
        # mai per posizione con la tupla del modo eager.
  support:
    - label: Controllare una famiglia e un task prima di esportare
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: a35bfeafac6d6966
---

## Installazione

Questo formato è solo per macOS. Il requisito `coreai-torch` porta un marcatore
`sys_platform == 'darwin'`, e la toolchain non converte né esegue da nessun'altra
parte.

<code-tabs name="install" />

L'extra sta fuori da ogni extra aggregato, incluso `libreyolo[all]`, perché
`coreai-torch` fissa torch alla serie 2.11. Installalo in un ambiente che sei
disposto a vincolare a quella coppia.

## Esportazione

<code-tabs name="export" />

La cattura è `torch.export`, una vera cattura del grafo con guard, e non un
singolo trace registrato. È più severa del percorso di Core ML: le letture di
scalari sull'host e il flusso di controllo dipendente dai dati vengono rifiutati
invece di essere incorporati in silenzio, ed è per questo che qui alcune famiglie
sono bloccate con un errore di cattura registrato.

Tre passi di preparazione vengono eseguiti dentro un ambito che ripristina il
modello vivo del chiamante sia che l'esportazione riesca sia che fallisca. Le
famiglie derivate da Darknet vedono la loro batch normalization di inferenza
ripiegata esattamente nelle convoluzioni precedenti, perché Core AI 0.4.1 non
preserva la formula di Darknet con l'epsilon dopo la radice quadrata. Le famiglie
a grid e ad anchor vedono i loro anchor congelati per il canvas fisso. RF-DETR
vede il suo position embedding ricalcolato per il canvas richiesto rieseguendo il
percorso di baking del modello stesso, perché il convertitore non ha un lowering
per `aten._upsample_bicubic2d_aa`.

Il lowering incorpora nella tabella delle decomposizioni la decomposizione di
riferimento di PyTorch per `aten.grid_sampler_2d`, dato che il convertitore di
Core AI non ha un lowering per il sampler della deformable attention che usano le
famiglie DETR.

Gli asset dichiarano un SO minimo v27, che è l'unico valore offerto dalla
toolchain. Questo vincola il deployment, non la conversione: la conversione e
l'esecuzione dal lato Python funzionano su macOS precedenti grazie al runtime che
sta dentro il wheel, ma i numeri differiscono tra le versioni del SO, quindi la
parità registrata è misurata su macOS 27.

## Eseguire l'artefatto

Non c'è nessuna voce Core AI in `libreyolo/backends`, quindi `LibreYOLO()` non
carica un `.aimodel`. I consumatori usano direttamente il runtime di Core AI, e
il preprocessing, il decoding, l'NMS e il riscalamento delle coordinate sono a
loro carico. Una riga validata nella matrice di supporto afferma che il grafo
esportato calcola gli stessi numeri del riferimento, non che `predict` lo
eseguirà.

L'unica cosa che un consumatore non può ricavare da sé è l'ordine degli output:

<code-tabs name="outputs" />

Core AI restituisce un dizionario con nomi il cui ordine delle chiavi non
coincide né con l'ordine della tupla del forward in modo eager né con qualcosa di
indovinabile. I nomi esportati vengono scritti nei metadati dell'asset come
`coreai_output_names` esattamente per questo motivo. Mappa per nome.

## Vincoli

Canvas fisso, FP32, batch com'è stato esportato. `dynamic=True` solleva
`NotImplementedError`, e `half=True` e `int8=True` vengono rifiutati durante la
validazione.

La copertura è ampia dal lato della conversione. Le combinazioni validate
includono le famiglie YOLO9, YOLOX, YOLO7, i quattro rilevatori dell'era Darknet,
YOLO-NAS, PicoDet, RTMDet, RT-DETR, RT-DETRv2, RT-DETRv4, D-FINE, DEIM, DEIMv2,
EC e il rilevamento RF-DETR; le quattro famiglie di classificazione CNN più CLIP
e SigLIP2 a classi congelate; Depth Anything V2 e ZipDepth; il restauro con
NAFNet e Real-ESRGAN; la segmentazione semantica con PIDNet e LingBotVision; e il
rilevamento di punti FOMO. Ognuna porta con sé il proprio contesto registrato,
che `libreyolo formats` stampa.

Bloccate, con il motivo registrato per ogni combinazione:

| Combinazione | Motivo |
|---|---|
| Segmentazione semantica EoMT | La cattura stretta fallisce con `GuardOnDataDependentSymNode`: qualcosa nel percorso delle maschere legge un valore da un tensore e ci ramifica sopra |
| Segmentazione semantica SegFormer | Il percorso di cattura non è stato valutato, e i suoi pesi pubblicati sono non commerciali a prescindere dal formato |
| Sguardo L2CS | Il modello stesso supporta solo ONNX, TorchScript, ExecuTorch, TensorRT e OpenVINO, ed è una decisione dal lato del modello |
| Profondità Depth Anything 3 | La famiglia rifiuta l'esportazione per ogni formato |

RF-DETR porta con sé un avvertimento che vale la pena leggere prima di
confrontare gli artefatti. La sua parità è registrata contro il grafo che prepara
l'esportatore Core AI stesso, non contro ONNX, e con un canvas di 640 l'artefatto
ONNX di RF-DETR è in disaccordo con quel grafo preparato. Il ricalcolo di Core AI
preserva il ridimensionamento con antialiasing che il modello esegue in modo
eager, mentre il percorso ONNX disattiva l'antialiasing. ONNX non è quindi un
riferimento valido per quella famiglia con un canvas non nativo.

Per il formato precedente di Apple, vedi [Core ML](/docs/export/coreml). Per la
griglia completa di famiglie e task, vedi [la matrice di
esportazione](/docs/reference/export-matrix). Per una sola combinazione:

<code-tabs name="support" />
