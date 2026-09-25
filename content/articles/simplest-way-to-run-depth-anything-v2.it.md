---
title: "Come usare Depth Anything V2 con LibreYOLO: inferenza in due righe"
description: "Depth Anything V2 produce mappe di profondità monoculare all'avanguardia. Ecco come eseguirlo in due righe con LibreYOLO."
date: 2026-06-26
author: Xuban
tags: [LibreYOLO, depth-anything-v2, depth-estimation, tutorial]
faq:
  - q: "Qual è il modo più semplice per eseguire Depth Anything V2?"
    a: "Due righe con LibreYOLO: carica LibreDepthAnythingV2l-depth.pt usando il nome e chiama predict con save=True. I pesi vengono scaricati automaticamente al primo utilizzo, mentre normalizzazione, mappa dei colori e assegnazione del dispositivo sono gestite per te su CUDA, Apple Silicon o una normale CPU."
  - q: "Posso usare Depth Anything V2 a fini commerciali?"
    a: "Solo l'encoder Small è distribuito con licenza Apache 2.0. Base, Large e Giant hanno licenza CC-BY-NC-4.0, quindi i checkpoint più potenti sono destinati all'uso non commerciale. La licenza upstream si applica indipendentemente dalla libreria che carica i pesi."
  - q: "LibreYOLO supporta l'addestramento di Depth Anything V2?"
    a: "No. L'addestramento resta nel repository originale; LibreYOLO supporta l'inferenza, i video e la validazione per il task di profondità."
---

![Il Guggenheim di Bilbao accanto alla relativa mappa di profondità](/articles/simplest-way-to-run-depth-anything-v2/guggenheim-bilbao-input-vs-depth.jpg)

Depth Anything V2 produce alcune delle migliori mappe di profondità monoculare disponibili al momento.

Se vieni dal mondo YOLO, il repository ufficiale non è il modo più semplice per iniziare. Per ottenere una singola mappa di profondità servono diversi passaggi manuali:

* scaricare a mano il checkpoint corretto e metterlo nella cartella giusta,

* abbinare l'encoder alla sua configurazione (`encoder="vitl"`, `features=256`, `out_channels=...`),

* scrivere da sé il passaggio di normalizzazione e colorazione,

* gestire l'assegnazione del dispositivo per eseguirlo su Mac o CPU, non solo su CUDA,

* imparare un'API di inferenza che non somiglia per niente al resto del tuo stack.

Niente di tutto questo è difficile. Aggiunge solo attrito, e puoi evitarlo.

LibreYOLO carica gli stessi pesi di Depth Anything V2 e offre il task di profondità con una sola chiamata a `predict()`. Indica il nome del modello e i pesi vengono scaricati al primo utilizzo:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDepthAnythingV2l-depth.pt")  # auto-downloads on first run
model.predict("image.jpg", save=True)     # writes a colorized depth map to disk
```

Tutto qui. Se hai usato l'API YOLO standard, la struttura ti sarà familiare: carica un modello usando il nome, chiama `predict` e lascia che `save=True` scriva la visualizzazione. È esattamente la stessa chiamata che useresti per il rilevamento di oggetti, ma il risultato contiene una mappa di profondità anziché dei bounding box. La mappa dei colori, la normalizzazione e l'assegnazione del dispositivo sono gestite per te. Funziona allo stesso modo su Linux con CUDA, su Mac con Apple Silicon o su una normale CPU. Senza modifiche al codice.

Da qui, la stessa chiamata fa anche di più: puoi recuperare i valori di profondità grezzi e usarli direttamente. L'addestramento di Depth Anything V2 resta nel repository originale; LibreYOLO supporta l'inferenza, i video e la validazione.

La stessa chiamata su scene molto diverse:

![L'immagine di parkour accanto alla relativa mappa di profondità: chi salta in primo piano si distingue dalle pareti di cemento sullo sfondo.](/articles/simplest-way-to-run-depth-anything-v2/parkour-input-vs-depth.jpg)

![Vista aerea della baia di La Concha a Donostia accanto alla relativa mappa di profondità: le barche e la costa appaiono vicine, il mare aperto si allontana.](/articles/simplest-way-to-run-depth-anything-v2/donostia-bay-input-vs-depth.jpg)

![Il cortile colonnato della Casa de Juntas de Gernika accanto alla relativa mappa di profondità, che mostra l'architettura che si allontana.](/articles/simplest-way-to-run-depth-anything-v2/gernika-casa-juntas-input-vs-depth.jpg)

![La folla a un festival notturno in Plaza de la Constitucion a Donostia accanto alla relativa mappa di profondità: le persone nelle file vicine si distinguono dalla facciata illuminata sullo sfondo.](/articles/simplest-way-to-run-depth-anything-v2/donostia-plaza-crowd-input-vs-depth.jpg)

Una nota sui pesi: LibreYOLO ospita checkpoint di Depth Anything V2 convertiti e li scarica al primo utilizzo, quindi non devi scaricare nulla a mano. La licenza upstream continua ad applicarsi: l'encoder Small è distribuito con licenza Apache-2.0, mentre Base, Large e Giant hanno licenza CC-BY-NC-4.0 (non commerciale), quindi i checkpoint più potenti sono destinati all'uso non commerciale. Vuoi lavorare offline o convertire i tuoi pesi? Lo script di conversione da eseguire una sola volta è ancora disponibile:

```bash
# optional: convert an official checkpoint yourself instead of auto-downloading
python weights/convert_depth_anything_v2_weights.py \
  depth_anything_v2_vitl.pth weights/LibreDepthAnythingV2l-depth.pt
```

## Provalo

```bash
pip install libreyolo
```

LibreYOLO è la libreria di computer vision più completa che puoi installare con pip. Una sola API familiare supporta una gamma crescente di modelli all'avanguardia: rilevamento di oggetti (RF-DETR, D-FINE, DEIM), segmentazione, stima della posa, box orientati e ora anche la profondità monoculare con Depth Anything V2, oltre all'addestramento e alla validazione per i task che li supportano. Funziona su tutti i principali sistemi operativi, su GPU o CPU, ed è interamente distribuita con licenza MIT, così puoi usarla nei tuoi prodotti commerciali senza vincoli.

Aggiungi una stella su GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentazione: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
