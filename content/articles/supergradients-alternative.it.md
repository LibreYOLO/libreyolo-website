---
title: "Alternativa a SuperGradients: YOLO-NAS per la computer vision in tempo reale"
description: "SuperGradients ha smesso di pubblicare release dopo l'acquisizione di Deci da parte di NVIDIA nel 2024. LibreYOLO è un'alternativa mantenuta che usa gli stessi pesi YOLO-NAS: predizione, addestramento ed esportazione con un'unica API con licenza MIT."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, supergradients-alternative, yolo-nas, object-detection]
faq:
  - q: "SuperGradients è ancora mantenuto?"
    a: "No. L'ultima release, la 3.7.1, è uscita ad aprile 2024, poco prima che NVIDIA acquisisse Deci. Da allora il repository non ha ricevuto release, le issue restano senza risposta e il sito della documentazione è offline. Non è stato archiviato formalmente, ma non c'è più nessuno a occuparsene."
  - q: "Qual è la migliore alternativa a SuperGradients?"
    a: "Per usare YOLO-NAS, LibreYOLO carica gli stessi pesi di Deci tramite un'API mantenuta con licenza MIT e aggiunge addestramento, validazione ed esportazione. Se dipendi dalle ricette di addestramento di SuperGradients per altre architetture, il vecchio repository funziona ancora se blocchi le sue dipendenze."
  - q: "Posso usare YOLO-NAS a fini commerciali?"
    a: "I pesi preaddestrati di Deci non possono essere usati a fini commerciali, indipendentemente dalla libreria che li carica. L'architettura è distribuita con una licenza permissiva, quindi addestrando da zero un modello YOLO-NAS sui tuoi dati ottieni un modello che non ha alcun checkpoint Deci nella sua storia."
  - q: "LibreYOLO sostituisce tutto quello che faceva SuperGradients?"
    a: "Non tutto. LibreYOLO non esporta YOLO-NAS in CoreML e il suo percorso TensorRT per questa famiglia non è testato; inoltre, non ha un equivalente diretto delle ricette di quantizzazione consapevole dell'addestramento di SuperGradients. Per questi casi, conserva il vecchio repository."
---

SuperGradients era la libreria open source per l'addestramento di Deci e ospitava YOLO-NAS, uno dei rilevatori in tempo reale più accurati mai pubblicati. Nel maggio 2024 NVIDIA ha acquisito Deci e SuperGradients ha smesso di essere aggiornato. L'ultima release, la 3.7.1, è uscita l'8 aprile 2024. La documentazione su supergradients.com è andata offline. Sono aperte circa 120 issue, e quella intitolata ["Questo progetto è morto?"](https://github.com/Deci-AI/super-gradients/issues/2062) non ha ricevuto risposta da un maintainer, e questo è già di per sé una risposta.

Il repository non è archiviato, quindi a prima vista sembra attivo. In pratica, l'abbandono si fa sentire non appena provi a installarlo: `super-gradients` fissa `torchmetrics==0.8`, che entra in conflitto con gli stack PyTorch attuali, e si porta dietro anche hydra, omegaconf, boto3 e tensorboard. Ogni mese che passa, queste versioni fissate entrano sempre più in conflitto con il resto dell'ambiente, e non arriverà alcuna correzione.

Questo non rende YOLO-NAS un modello peggiore. La variante large raggiunge ancora 52.2 mAP su COCO a velocità in tempo reale. Il modello va bene, è la sua casa ad essere bruciata.

<iframe
  src="https://visionanalysis.org/embed/scatter?highlight=yolonas-s%2Cyolonas-m%2Cyolonas-l"
  width="100%"
  height="420"
  style="border:0;border-radius:12px;overflow:hidden"
  loading="lazy"
  title="Accuratezza di YOLO-NAS rispetto ai parametri - visionanalysis.org">
</iframe>

## Usa gli stessi pesi in LibreYOLO

LibreYOLO carica direttamente i checkpoint YOLO-NAS dal CDN di Deci, tramite la stessa API usata per tutte le altre famiglie di modelli. Non devi scrivere configurazioni hydra né estrarre i risultati da un wrapper di predizione:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")  # auto-downloads on first run
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

Le varianti di rilevamento S, M e L funzionano tutte, così come quella per la stima della posa: sostituisci il nome con `LibreYOLONASs-pose.pt` e ottieni i keypoint COCO. Poiché ogni famiglia restituisce lo stesso oggetto `Results`, confrontare YOLO-NAS con RF-DETR o D-FINE sui tuoi dati richiede una sola modifica di riga, senza dover usare una seconda base di codice.

## Addestramento ed esportazione, non solo inferenza

SuperGradients era prima di tutto una libreria per l'addestramento, quindi una vera alternativa deve permettere di addestrare. LibreYOLO fa fine-tuning di YOLO-NAS sul tuo dataset con la stessa chiamata usata per tutti gli altri modelli:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

Puoi anche addestrare un modello con pesi inizializzati casualmente, una possibilità importante per le licenze, come vedremo più avanti. La validazione restituisce metriche mAP su qualsiasi dataset nel tuo formato, mentre l'esportazione supporta ONNX, TorchScript, OpenVINO, NCNN e TFLite. Per YOLO-NAS, i formati di esportazione disponibili sono più numerosi di quelli mai offerti dalla libreria originale. Tutti i dettagli sono nella [pagina della documentazione di YOLO-NAS](https://www.libreyolo.com/docs/models/yolo-nas); trovi anche una breve nota sul fatto che [YOLO-NAS è ancora mantenuto](/articles/yolo-nas-with-libreyolo).

## Quando il vecchio repository resta la scelta giusta

Una sezione franca. Ecco tre casi in cui dovresti continuare a usare SuperGradients:

**CoreML.** LibreYOLO non esporta YOLO-NAS in CoreML. Se distribuisci su dispositivi Apple e ti serve un `.mlpackage`, SuperGradients ha ancora un percorso funzionante.

**TensorRT.** SuperGradients documentava e testava il flusso TensorRT per YOLO-NAS, comprese tutte le particolarità della dimensione del batch. Il supporto TensorRT di LibreYOLO per questa famiglia non è testato.

**Ricette di quantizzazione consapevole dell'addestramento.** YOLO-NAS è stato progettato per INT8 e SuperGradients includeva le ricette QAT che gli permettevano di brillare in questo ambito. LibreYOLO esporta con quantizzazione INT8 e FP16, ma non ha un equivalente di quelle ricette di addestramento.

Il repository funziona ancora se blocchi le dipendenze in un ambiente risalente al 2024. Semplicemente, non dovresti costruire nulla di nuovo su fondamenta che nessuno mantiene.

## Una nota sui pesi

I pesi YOLO-NAS preaddestrati sono di Deci, sono distribuiti con una licenza non commerciale e tale licenza si applica ai pesi qualunque sia la libreria che li carica. LibreYOLO non li ospita né li replica: il download proviene dal CDN pubblico di Deci e, prima di avviarlo, vengono mostrate le condizioni di Deci. Per ricerca e uso non commerciale, entrambe le opzioni vanno bene.

Se ti serve un modello YOLO-NAS per uso commerciale, ora c'è una strada semplice: l'architettura è distribuita con una licenza permissiva, quindi addestrando da zero sui tuoi dati ottieni un modello che non deriva da alcun checkpoint Deci. LibreYOLO supporta proprio questo caso con `LibreYOLONAS(None, size="s")`.

## Provalo

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASl.pt")
results = model("image.jpg", save=True)
```

LibreYOLO è distribuito con licenza MIT, funziona su Linux, Mac e Windows e gira su GPU, Apple Silicon e CPU standard senza modifiche al codice. Un'unica API supporta YOLO-NAS, RF-DETR, D-FINE, DEIM, YOLOX, RTMDet e molti altri, per il rilevamento di oggetti, la segmentazione, la stima della posa, la classificazione, la stima della profondità e il tracking.

Aggiungi una stella su GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentazione: [libreyolo.com/docs](https://www.libreyolo.com/docs)
