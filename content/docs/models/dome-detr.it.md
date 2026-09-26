---
title: Dome-DETR
families:
  - domedetr
seo_title: 'Dome-DETR: rilevamento di oggetti minuscoli in LibreYOLO'
description: Usa Dome-DETR per rilevamento di oggetti minuscoli, addestramento e validazione. I checkpoint preaddestrati replicati restano riservati alla ricerca accademica.
lead: >-
  Uno specialista degli oggetti minuscoli costruito su D-FINE: una testa di
  densità decide dove sono gli oggetti, l'attenzione dell'encoder è ristretta
  alle finestre che li contengono e il numero di query è dimensionato a partire
  da quella densità invece di essere fisso. LibreYOLO lo supporta per il
  rilevamento.
keywords:
  - Dome-DETR
  - tiny object detection
  - small object detection
  - rilevamento oggetti piccoli
  - immagini da drone
  - telerilevamento
  - VisDrone
  - AI-TOD
  - DETR
  - density adaptive queries
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # I pesi preaddestrati sono riservati alla ricerca accademica.
        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt", device="cpu")
        print(model(SAMPLE_IMAGE).boxes)

  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")

        model.train(data="my-dataset.yaml", epochs=160, imgsz=800, batch=4,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml \
          epochs=160 imgsz=800 batch=4 lr0=2e-4
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=4

  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml
source_hash: 8482301790a9b8d9
---

## Installazione

Dome-DETR non richiede nessun extra opzionale. Tutto quello che importa è già
nell'installazione di base.

```bash
pip install libreyolo
```

## Predizione

I sei checkpoint convertiti vengono scaricati automaticamente dai mirror di LibreYOLO. I termini upstream ne limitano l'uso alla ricerca accademica.

<code-tabs name="predict" />

L'oggetto `Results` restituito è lo stesso che restituisce ogni famiglia,
quindi passare a un detector diverso è una modifica di una riga. `conf` e
`max_det` filtrano la selezione delle query; `iou` è accettato per parità di
API ma non ha effetto, perché il decoder è un set predictor senza passaggio di
NMS. Vedi [predizione](/docs/predict) per sorgenti, streaming e gestione dei
risultati.

Due funzionalità sono disattivate per questa famiglia. La cattura dei CUDA
graph è disabilitata, perché il numero di query di PAQI dipende dai dati e il
forward pass cambia quindi forma da un'immagine all'altra, che è esattamente
ciò che la cattura dei graph non può assorbire. La test-time augmentation gira
a una singola dimensione quadrata fissa, quindi una richiesta di TTA
multi-scala non ha alcun effetto.

## Varianti

Tre dimensioni, s, m e l, tutte a 800 per 800. La dimensione seleziona il
backbone, e il dataset da cui vengono i pesi seleziona la profondità del
decoder e il budget di query, quindi un codice di dimensione da solo non
identifica un grafo. I pesi AI-TOD-V2 scelgono tra 300 e 1500 query per
immagine, i pesi VisDrone tra 250 e 500, e il modello large esegue quattro
layer del decoder su AI-TOD-V2 contro sei su VisDrone.

Dome-DETR è D-FINE con tre aggiunte. DeFE predice una mappa di densità. MWAS
usa quella mappa per restringere l'attenzione dell'encoder alle finestre che
contengono davvero degli oggetti, invece di applicarla ovunque. PAQI dimensiona
l'insieme di query a partire dalla stessa densità invece di decodificarne 300
fisse. Il guadagno si concentra dove gli oggetti sono più piccoli e si
restringe man mano che crescono: l'ablation degli autori upstream porta l'AP
sugli oggetti molto minuscoli da 14.0 a 17.8, mentre l'AP sugli oggetti medi
passa solo da 45.4 a 46.4. Consideralo un complemento a
[D-FINE](/docs/models/d-fine) per immagini aeree, da drone e di
telerilevamento, non un suo sostituto.

Non sono registrate righe di benchmark Vision Analysis per questa famiglia.

## Addestramento

Dome-DETR è addestrabile. L'addestramento esegue l'obiettivo completo di
upstream: le loss di D-FINE più la supervisione di densità e conteggio di DeFE,
con le query di padding mascherate fuori dai termini di classificazione e con
maschere di attenzione di denoising per immagine, così che il padding di
un'immagine non possa finire in quello di un'altra.

<code-tabs name="train" />

La configurazione eredita la ricetta di D-FINE e cambia quello che MWAS
richiede. `imgsz` è 800, `lr0` è `2e-4`, il gruppo di parametri del backbone è
scalato da `backbone_lr_mult=0.1` e `multi_scale` è forzato a off, perché le
finestre di MWAS hanno bisogno che l'input resti divisibile per lo stride 8.
`batch` vale 4 di default invece dei 16 di D-FINE: PAQI riempie ogni batch fino
al suo membro più largo, quindi la memoria segue l'immagine più affollata del
batch invece di quella media.

Un avvertimento onesto sull'accuratezza. Upstream addestra per 160 epoche con
`MultiStepLR(milestones=[80, 120], gamma=0.8)`, mentre questi default eseguono
lo schedule flat-cosine di D-FINE per le stesse 160 epoche. Quello schedule non
è stato riprodotto qui e nemmeno i numeri di AP del paper sono stati riprodotti,
quindi leggili come risultati degli autori upstream e non come la promessa che
questa ricetta li raggiunga. Passa lo schedule upstream se l'obiettivo è
eguagliare il paper.

Vedi [addestramento](/docs/train) per dataset, augmentation, multi-GPU e logger.

## Validazione

`val()` restituisce un dizionario indicizzato per nome della metrica, e stampa i
risultati per classe quando `verbose` è lasciato attivo.

<code-tabs name="val" />

La validazione gira sul tuo dataset, nel formato su cui hai addestrato. Il gate
di validazione COCO della libreria qui non si applica, dato che per questa
famiglia non esiste un checkpoint COCO con cui misurarsi.

## Esportazione

L'esportazione non è supportata, per nessun formato, e chiederla solleva un
errore invece di produrre un file.

Il motivo è PAQI. Decide il numero di query per immagine, a partire da proposte
filtrate per densità e da un ciclo greedy di soppressione adattiva alla densità,
quindi la lunghezza dell'output del decoder è una proprietà dell'input e non del
grafo. Il tracing fissa qualunque numero l'immagine di tracing abbia prodotto, e
ne esce un artefatto che restituisce silenziosamente risultati sbagliati per
ogni altra immagine. Una formulazione statica dovrebbe srotolare quella
soppressione su tutti i candidati, da 250 a 1500, e ridurre a un top-k fisso
eliminerebbe esattamente il recall sugli oggetti minuscoli per cui questa
famiglia esiste. Se ti serve un detection transformer esportabile,
[D-FINE](/docs/models/d-fine) è quello a cui rivolgersi.

## Checkpoint

<checkpoint-table />

## Licenze

<provenance-box>

I sei mirror mantengono il vincolo upstream di uso esclusivo per ricerca accademica. La licenza del codice è separata. Il repository upstream è Apache-2.0, il porting LibreYOLO è MIT e i pesi che addestri sui tuoi dati sono tuoi.

</provenance-box>
