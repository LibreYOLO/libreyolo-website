---
title: Matrice completa di esportazione
seo_title: La matrice di supporto all'esportazione di LibreYOLO e le sue regole
description: >-
  Come LibreYOLO decide se una combinazione di famiglia, task e formato è
  esportabile: i dodici formati, i tre livelli, le regole di fallback e le
  soglie di parità.
lead: >-
  Il supporto all'esportazione è una consultazione sulla terna (famiglia, task,
  formato). Questa pagina descrive la forma di quella matrice, le regole che
  riempiono le celle non coperte da una voce esplicita, e come interrogarla per
  la combinazione che ti interessa.
keywords:
  - libreyolo export formati supportati
  - matrice di esportazione modelli
  - onnx tensorrt openvino tflite
  - comando libreyolo formats
  - soglia parità esportazione
  - NotImplementedError export
last_verified: 1.5.0
verification: >-
  Formati, livelli, ordine di fallback, blocchi per task e per famiglia e
  blocchi NCNN letti da libreyolo/export/support.py; alias e argomenti condivisi
  da libreyolo/export/exporter.py; definizioni dei livelli da
  docs/adr/0011-export-support-tiers.md; soglie di parità da
  docs/export_support.md, tutto alla v1.5.0. Le celle delle singole combinazioni
  non sono trascritte qui; interrogale con lo snippet qui sotto.
snippets:
  usage:
    - label: 'Consultare la matrice, senza bisogno di un modello'
      language: python
      code: |
        from libreyolo.export.support import (
            EXPORT_FORMATS,
            get_support,
            validated_alternatives,
        )

        print(EXPORT_FORMATS)

        entry = get_support("yolo9", "detect", "onnx")
        print(entry.tier, entry.since)
        print(entry.constraint)

        print(validated_alternatives("yolo9", "detect"))
    - label: CLI
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
        libreyolo formats --family yolo9 --task detect --json
  export:
    - label: 'Esportare, e leggere un rifiuto'
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.export.support import get_support


        model = LibreYOLO("LibreYOLO9t.pt")

        print(model.export(format="onnx"))


        # Controlla prima di chiamare: una combinazione bloccata fallisce nel
        preflight

        # e il messaggio riporta questa motivazione.

        blocked = get_support("domedetr", "detect", "onnx")

        print(blocked.tier)

        print(blocked.reason)
source_hash: 83de3289634888c6
---

## Forma della matrice

La matrice è indicizzata da `(family, task, format)`. Le chiavi di famiglia sono
i nomi canonici del registro dei modelli, le chiavi di task vengono da
`libreyolo.tasks.TASKS`, e i formati sono dodici:

`onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`,
`rknn`, `ncnn`, `tflite`, `coreml`, `coreai`.

`model.export(format=...)` accetta inoltre due alias: `engine` per `tensorrt` e
`litert` per `tflite`, che è il nome attuale di TensorFlow Lite. Il formato e il
suffisso `.tflite` non cambiano.

<code-tabs name="usage" />

Poiché una cella è funzione di tre chiavi, la griglia completa è grande e cambia
a ogni release. È generata invece che scritta a mano, e si trova in
`docs/export_support.md` nel repository della libreria. Interroga la matrice da
Python o dalla CLI invece di leggerne una copia.

## I tre livelli

| Livello | Significato |
|---|---|
| `validated` | La parità numerica è coperta dalla CI o da un'esecuzione notturna documentata |
| `available` | La conversione è implementata, ma non è stata registrata alcuna prova di parità numerica a runtime |
| `blocked` | Il preflight solleva `NotImplementedError` con una motivazione prima del tracing |

Le combinazioni validated e available procedono entrambe senza una presa d'atto
esplicita né un avviso generico. Le prove registrate e i loro vincoli restano
visibili nella documentazione generata. Una combinazione blocked fallisce prima
dei controlli sulle dipendenze, del caricamento della calibrazione, del tracing
o della creazione dell'artefatto.

Aggiungere una voce validated richiede un test di parità e un campo `since`.

Un `SupportEntry` porta quattro campi: `tier`, una stringa `reason`, la release
`since` e una stringa `constraint`. Il vincolo è la parte che conta quando
integri: un segno di spunta vale solo alle condizioni che indica, che di solito
sono un canvas di input fisso, batch 1, FP32 e una versione specifica del
runtime.

## Come viene decisa una cella

`get_support(family, task, fmt)` risolve in quest'ordine. Vince la prima regola
che corrisponde.

1. Un task sconosciuto, o un formato fuori dai dodici, restituisce `blocked`.
2. Una voce esplicita `(family, task, format)` viene restituita così com'è registrata.
3. Un blocco a livello di famiglia restituisce `blocked` con la motivazione di quella famiglia.
4. Un blocco a livello di task restituisce `blocked` con la motivazione di quel task.
5. Per `ncnn`, una famiglia nella lista di blocco NCNN restituisce `blocked`.
6. `mnn` restituisce `blocked`: nessun contratto di runtime per questa famiglia e questo task.
7. `rknn` restituisce `blocked`. In questa versione RKNN è limitato esattamente alle varianti di rilevamento testate sul simulatore: YOLO9-t, YOLO9-E2E-t, YOLO-NAS-s e PicoDet-s su RK3588.
8. `tensorrt` e `openvino` restituiscono `available`: il percorso di conversione esiste ma la parità a runtime non è stata registrata per quella famiglia e quel task.
9. `tflite`, `paddle`, `coreai` e `coreml` restituiscono `blocked`, ciascuno con la propria motivazione.
10. Tutto il resto restituisce `available`: la conversione è implementata, la parità numerica a runtime non è registrata.

L'asimmetria nei passi da 8 a 10 è voluta. TensorRT e OpenVINO convertono in
modo generico a partire da ONNX, quindi vale la pena tentare una combinazione
non elencata. TFLite, Paddle, Core AI e CoreML hanno bisogno ciascuno di un
percorso specifico per famiglia, quindi una combinazione non elencata è un
rifiuto e non un invito.

## Task bloccati

Questi task sono bloccati per qualsiasi famiglia priva di una voce esplicita.

| Task | Motivo |
|---|---|
| `ocr` | Due reti con ritaglio dinamico per regione non rientrano nel contratto di esportazione a grafo singolo |
| `point` | La famiglia non è collegata alla heatmap di punti condivisa né al contratto di decodifica dei picchi nel backend |
| `semantic` | La famiglia non è collegata ai logit densi condivisi né al contratto di argmax nel backend |
| `mesh` | Gli output del grafo della mesh corporea, i metadati e il contratto di runtime non sono definiti |
| `normal` | La famiglia non è collegata alle normali unitarie dense a canvas fisso né al contratto di rinormalizzazione nel backend |
| `panoptic` | L'esportazione panottica non ha un contratto di runtime nel backend |
| `gaze` | La famiglia non è collegata ai logit condivisi a due teste né al contratto di decodifica per valore atteso nel backend |

Una voce esplicita ha la precedenza su questi blocchi, ed è così che, per
esempio, una famiglia semantic già collegata riesce comunque a esportare.

## Famiglie bloccate

| Famiglia | Bloccata per |
|---|---|
| `depth_anything3` | Tutti i formati; il suo grafo di profondità non rientra nel contratto di runtime esportato |
| `domedetr` | Tutti i formati. PAQI stabilisce il numero di query per immagine, quindi un grafo tracciato è valido solo per l'immagine su cui è stato tracciato. Usa D-FINE per un DETR esportabile |
| `eomt` | Esportazione di istanze e panottica, che non hanno un parsing a runtime |
| `l2cs` | Tutto ciò che non sia ONNX, TorchScript, ExecuTorch, TensorRT e OpenVINO |
| `hrnet` | Tutto ciò che non sia ONNX, TorchScript, OpenVINO e TensorRT |
| `sam`, `sam2`, `sam3`, `edgetam`, `mobilesam` | Tutti i formati; l'esportazione di modelli promptable è fuori dallo scopo del contratto di runtime v1 |
| `grounding_dino`, `owlv2`, `omdet_turbo`, `ov_deim` | Tutti i formati; l'esportazione a runtime a vocabolario aperto è fuori dallo scopo della v1 |
| `florence2`, `kosmos2`, `lfm2vl`, `internvl3`, `qwen3vl`, `smolvlm2`, `locateanything` | Tutti i formati; l'esportazione di VLM generativi è fuori dallo scopo della v1 |

PicoSAM3 è l'eccezione nel livello promptable: esporta in ONNX la sua rete ROI
grezza da 96 pixel.

## Bloccate per NCNN

I decoder in stile DETR hanno bisogno di operazioni di sampling che NCNN non
implementa, quindi queste famiglie sono bloccate per `ncnn` a meno che una voce
esplicita non dica altrimenti: Deformable DETR, DETR, DINO-DETR, D-FINE,
LW-DETR, DEIM, DEIMv2, RT-DETR, RT-DETRv2, RT-DETRv4, RF-DETR ed EC. Il rifiuto
indica ONNX, OpenVINO, TorchScript e TensorRT come alternative.

## Soglie di parità

Una cella validated significa che l'artefatto esportato ha riprodotto il modello
nativo entro questi limiti:

| Gruppo di task | Soglia |
|---|---|
| Rilevamento e OBB | IoU dei box accoppiati sopra 0.95, MAE del punteggio sotto 0.01 |
| Segmentazione e panottica | IoU della maschera sopra 0.95 |
| Posa | L2 dei keypoint sotto i 2 pixel alla risoluzione nativa |
| Classificazione | Coseno dei logit sopra 0.999 e stessa classe top-1 |
| Profondità e restauro | PSNR sopra 40 dB rispetto all'output nativo |
| Normali di superficie | Errore angolare medio sotto 0.1 gradi |
| Punti | Posizioni dei picchi uguali entro una cella di output |

Le righe di query dei DETR sono un insieme non ordinato, quindi la parità per la
famiglia DETR allinea le righe di query come insieme e non per posizione.

## Esportazione

<code-tabs name="export" />

Una combinazione bloccata solleva `NotImplementedError` nel preflight e il
messaggio riporta la motivazione registrata. `validated_alternatives(family, task)`
restituisce i formati validated per quella coppia, che è la cosa utile da
stampare accanto a un rifiuto.

Gli argomenti condivisi da tutti gli exporter sono elencati nella
[pagina dell'API del modello](/docs/reference/model-api). Gli argomenti
specifici di ciascun formato stanno nelle pagine dei singoli formati.

## Leggere un vincolo

Una cella validated è un'affermazione su una singola configurazione misurata,
non sul formato in generale. Una stringa di vincolo come
`FP32, batch 1, fixed 520x520 input` significa che la parità è stata registrata
a quella forma e a quella precisione. Esportare a una risoluzione o a una
dimensione del batch diversa produce comunque un artefatto; semplicemente non è
la configurazione da cui viene quel numero.
