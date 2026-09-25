---
title: "RTMDet senza mmdetection: come eseguirlo con LibreYOLO"
description: "RTMDet è uno dei detector più veloci e accurati. L'installazione ufficiale non funziona con nessuna versione di PyTorch rilasciata negli ultimi due anni. Ecco l'alternativa."
date: 2026-06-27
author: Xuban
tags: [LibreYOLO, rtmdet, object-detection, tutorial]
faq:
  - q: "Perché mmcv non si installa con le versioni recenti di PyTorch?"
    a: "L'ultima wheel di mmcv, la 2.2.0 di aprile 2024, include binari precompilati solo fino a torch 2.4 / CUDA 12.1. Con qualsiasi versione più recente di PyTorch, mmcv deve compilare da zero le proprie operazioni C++/CUDA: l'operazione richiede da 10 a 30 minuti e serve un toolkit CUDA corrispondente, nvcc e un compilatore C++ compatibile. È questo a causare errori come l'assenza del modulo mmcv._ext."
  - q: "Posso eseguire RTMDet senza installare mmdetection?"
    a: "Sì. LibreYOLO carica i pesi RTMDet convertiti dai checkpoint OpenMMLab originali e l'inferenza è bit per bit equivalente a mmdetection sullo stesso checkpoint. Carica LibreRTMDets.pt per nome e il download parte automaticamente; tutte e cinque le dimensioni, da Tiny a X, funzionano a 640 px."
  - q: "RTMDet è gratuito per uso commerciale?"
    a: "Sì. MMDetection e RTMDet sono distribuiti con licenza Apache 2.0, i pesi convertiti hanno gli stessi termini Apache 2.0 e il codice di LibreYOLO è MIT. Nessuna restrizione commerciale."
  - q: "Quando mmdetection resta la scelta giusta?"
    a: "Se devi addestrare RTMDet o fare fine-tuning usando l'intera pipeline di data augmentation di MMDetection, oppure se sei già ben inserito nell'ecosistema OpenMMLab e mmcv funziona. Per l'inferenza e il deployment, LibreYOLO è la strada migliore."
---

RTMDet è un detector real-time di OpenMMLab che raggiunge ottimi risultati di accuratezza su COCO mantenendo una velocità adatta al deployment. L'architettura è pulita. L'installazione no.

Per eseguire RTMDet dalla fonte ufficiale, devi assemblare uno stack OpenMMLab a quattro livelli:

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

I vincoli sulle versioni sono stretti. mmdet 3.3.0 richiede `mmcv < 2.2.0`, ma `mim install mmcv>=2.0.0` installa tranquillamente la 2.2.0, che poi fallisce un'asserzione in fase di esecuzione. Devi specificare il vincolo a mano.

Poi c'è un problema più grande. L'ultima wheel di mmcv è la 2.2.0, rilasciata ad aprile 2024 e mai aggiornata da allora. Include binari precompilati solo fino a **torch 2.4 / CUDA 12.1**. Oggi, con un nuovo `pip install torch`, ottieni PyTorch 2.12. Questo divario significa che mmcv deve compilare da zero le operazioni C++/CUDA: servono da 10 a 30 minuti, un toolkit CUDA corrispondente, `nvcc` e un compilatore C++ compatibile. È qui che si accumulano gli errori documentati:

* `ModuleNotFoundError: No module named 'mmcv._ext'` -- le operazioni compilate non sono state create o non corrispondono alla versione installata,

* `nvcc fatal: Unsupported gpu architecture 'compute_86'` su qualsiasi scheda RTX serie 30,

* `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'` -- lo stack ti costringe a tornare a Python 3.8,

* segmentation fault all'importazione dovuto a una versione incompatibile di GCC.

Le FAQ ufficiali di OpenMMLab consigliano di installare mmcv con pip invece che con mim per evitare il problema delle versioni. La loro pagina Get Started consiglia di usare mim. Entrambi i documenti sono ufficiali. Si contraddicono.

Una volta avviato lo stack, per l'inferenza devi comunque scegliere un file di configurazione il cui nome codifica la ricetta di addestramento, oltre a scaricare separatamente un checkpoint con un nome che include l'hash e a collegare tutto tramite un registro che devi prima imparare a usare.

LibreYOLO sostituisce tutto questo:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")  # auto-downloads on first run
results = model("image.jpg", save=True)
```

Niente `mim`, niente matrice di versioni con quattro pacchetti, niente compilazione dal codice sorgente, niente file di configurazione, niente caccia al checkpoint tramite hash e nessun vincolo su Python 3.8. I pesi sono convertiti dai checkpoint OpenMMLab originali e l'inferenza è bit per bit equivalente a mmdetection sullo stesso checkpoint.

Sono disponibili cinque dimensioni: Tiny, Small, Medium, Large e X. Tutte funzionano a 640 px.

```python
print(results[0].boxes.xyxy)  # xyxy coordinates
print(results[0].boxes.conf)  # confidence scores
```

## Quando la versione originale resta la scelta giusta

Se devi addestrare RTMDet o fare fine-tuning usando l'intera pipeline di data augmentation di MMDetection, oppure se sei già ben inserito nell'ecosistema OpenMMLab e mmcv funziona, resta lì. LibreYOLO è la strada migliore per l'inferenza e il deployment.

## Una nota sulla licenza

MMDetection e RTMDet sono distribuiti con licenza Apache 2.0. Il codice di LibreYOLO è MIT. I pesi hanno gli stessi termini Apache 2.0 dei progetti originali. Nessuna restrizione commerciale.

## Provalo

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDetl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLO è distribuito con licenza MIT, funziona su Linux, Mac e Windows e gira su GPU, Apple Silicon e CPU standard senza modifiche al codice. Una sola API copre RTMDet, RT-DETR, RF-DETR, D-FINE, YOLOX, YOLO-NAS, segmentazione, stima della posa, profondità e altro.

Aggiungi una stella su GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentazione: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
