---
title: HRNet
families:
  - hrnet
seo_title: 'HRNet: stima della posa top-down in LibreYOLO'
description: >-
  Usa HRNet in LibreYOLO per la stima della posa top-down su COCO-17. Installa,
  fai predizioni, valida ed esporta i checkpoint W32 e W48, con licenza MIT.
lead: >-
  HRNet è una rete convoluzionale che mantiene un flusso di feature ad alta
  risoluzione attraverso ripetute fusioni multi-scala, invece di recuperare la
  risoluzione dopo il downsampling. LibreYOLO include la variante ufficiale
  top-down per la posa, per inferenza e validazione.
keywords:
  - HRNet
  - human pose estimation
  - stima della posa python
  - top-down pose estimation
  - COCO-17 keypoints
  - keypoint detection python
  - high-resolution network
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Nessuna sorgente di persone indicata: HRNet si abbina da solo a un
        # rilevatore leggero LibreYOLO9t e registra quella scelta una volta.
        model = LibreYOLO("LibreHRNetw32-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreHRNetw32-pose.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Sorgente delle persone
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreHRNetw32-pose.pt")


        # Salta del tutto il rilevamento: tratta l'intera immagine come una
        persona.

        result = model(SAMPLE_IMAGE, cropped=True)


        # Oppure passa a HRNet i box di un rilevatore che hai già eseguito.

        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])


        # Oppure abbinalo a un rilevatore LibreYOLO specifico invece di quello

        # predefinito LibreYOLO9t.

        result = model(SAMPLE_IMAGE, person_detector="rfdetr")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreHRNetw32-pose.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreHRNetw32-pose.pt format=onnx
    - label: Usare il file esportato
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # Il grafo esportato è la sola testa a heatmap su canvas fisso: prende
        un

        # batch di ritagli di persona già estratti e normalizzati e restituisce

        # heatmap grezze. Rilevamento delle persone, geometria del ritaglio,

        # decode delle heatmap e soppressione OKS non fanno parte del grafo;

        # eseguirlo fuori da LibreYOLO significa reimplementare da sé quel
        decode.

        session = ort.InferenceSession("LibreHRNetw32-pose.onnx")

        name = session.get_inputs()[0].name

        heatmaps = session.run(
            None, {name: np.zeros((1, 3, 256, 192), dtype=np.float32)}
        )[0]
source_hash: 5a5540fd54ee6f23
---

## Installazione

HRNet non richiede nessun extra oltre al pacchetto base.

```bash
pip install libreyolo
```

Il suo rilevatore di persone predefinito, un checkpoint LibreYOLO9t leggero, si
scarica automaticamente la prima volta che HRNet si abbina a esso.

## Predizione

I pesi si scaricano da Hugging Face al primo utilizzo e restano in cache in
locale.

<code-tabs name="predict" />

HRNet è uno stimatore della posa top-down: prima che la testa per la posa possa
girare serve un box della persona, quindi ogni chiamata ne risolve uno. Se non
gli dici nulla, la prima volta si abbina da solo a un rilevatore LibreYOLO9t e
registra quella scelta nei log. `cropped=True` salta il rilevamento e tratta
l'intera immagine come una sola persona; `person_boxes` accetta i box di un
rilevatore che hai già eseguito; `person_detector` accetta `"auto"`,
`"rfdetr"`, qualsiasi modello di rilevamento LibreYOLO oppure un semplice
callable. `flip_test=True` esegue il modello anche sul ritaglio ribaltato
orizzontalmente e fa la media delle due heatmap: è la test-time augmentation
propria di HRNet; il generico `augment=True` qui non è definito. Le sorgenti
con più immagini vengono elaborate in sequenza: il rilevatore di HRNet e il
numero di persone variabile da immagine a immagine non consentono la predizione
in batch. Vedi [predizione](/docs/predict) per sorgenti, streaming e gestione
dei risultati.

## Varianti

Due dimensioni, `w32` e `w48`, che predicono entrambe il set standard di 17
keypoint COCO a partire da un ritaglio della persona a risoluzione fissa; `w48`
è il più largo dei due backbone.

Il model zoo upstream riporta l'accuratezza della posa per ogni dimensione con
il proprio rilevatore di persone, la propria configurazione di flip-test e il
protocollo ufficiale di valutazione COCO. L'abbinamento predefinito di
LibreYOLO usa un rilevatore diverso, quindi una validazione fatta qui misura
quella combinazione, non quella upstream; per far coincidere i numeri upstream
servono gli stessi box delle persone, gli stessi punteggi del rilevatore e la
stessa impostazione di flip usati nella valutazione originale.

## Validazione

`val()` calcola l'OKS-AP sui keypoint in stile COCO e accetta un `data.yaml`
YOLO-pose oppure un JSON di keypoint COCO più una directory di immagini. Il
backend delle metriche è faster-coco-eval per impostazione predefinita, con
`pycocotools` usato automaticamente quando faster-coco-eval non è installato;
`faster_coco_eval=False` forza il percorso `pycocotools`.

<code-tabs name="val" />

La validazione richiama internamente il `predict()` di HRNet, quindi usa il
rilevatore di persone con cui il modello è stato costruito o chiamato.
Costruisci il modello con un `person_detector=` esplicito per tenere fissa
quella sorgente tra un'esecuzione e l'altra, invece di lasciare che ogni
chiamata risolva di nuovo il valore predefinito.

## Esportazione

<export-matrix />

Il contratto di esportazione di HRNet copre solo ONNX, TorchScript, OpenVINO e
TensorRT; qualsiasi altro formato solleva un errore prima che inizi il tracing.
Ogni esportazione contiene la sola testa a heatmap su canvas fisso, batch uno in
FP32, che prende un ritaglio della persona e restituisce heatmap grezze: la
geometria affine del ritaglio che la precede e il decode delle heatmap, il
ripristino del flip e la soppressione OKS che la seguono restano in Python,
quindi una pipeline completa che entra come immagine ed esce come keypoint ha
comunque bisogno di LibreYOLO dall'altro capo.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenza

<provenance-box></provenance-box>

## Citazione

<citation-block />
