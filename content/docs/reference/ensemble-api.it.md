---
title: API di ensemble
seo_title: API di LibreEnsemble e operazioni di fusione
description: >-
  LibreEnsemble, ExternalDetector e le tre operazioni di fusione in
  libreyolo.ops: weighted boxes fusion, la sua variante con seed e la fusione
  tramite NMS per classe.
lead: >-
  LibreEnsemble esegue più detector sulla stessa immagine e fonde i loro
  rilevamenti in un unico Results. La fusione avviene dopo il postprocessing di
  ciascun membro, quindi ogni membro mantiene la propria dimensione di input, la
  propria normalizzazione e la propria soppressione.
keywords:
  - LibreEnsemble
  - ensemble di modelli object detection
  - weighted boxes fusion python
  - ExternalDetector
  - libreyolo.ops.fusion
  - consenso min_votes
last_verified: 1.5.0
verification: >-
  Firme e valori predefiniti letti da libreyolo/ensemble/model.py e
  libreyolo/ops/fusion.py alla v1.5.0. Intento di progettazione da
  docs/adr/0004-model-ensembling.md.
snippets:
  usage:
    - label: 'Due membri, fusione predefinita'
      language: python
      code: >
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE


        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])


        # Una sorgente a immagine singola restituisce un solo Results, non una
        lista.

        result = ens(SAMPLE_IMAGE, conf=0.25)


        print(result.boxes.xyxy)

        print(result.speed)
    - label: Consenso e soglie per membro
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(
            ["LibreYOLO9t.pt", "LibreYOLO9s.pt"],
            weights=[1.0, 2.0],
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,
        )
        result = ens(SAMPLE_IMAGE, conf=[0.25, 0.4])
        print(len(result))
  ops:
    - label: 'Operazione di fusione, nessun modello coinvolto'
      language: python
      code: >
        import torch

        from libreyolo.ops import weighted_boxes_fusion


        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [12.0, 11.0, 51.0,
        49.0]])

        scores = torch.tensor([0.9, 0.8])

        labels = torch.tensor([0, 0])

        model_ids = torch.tensor([0, 1])


        fused = weighted_boxes_fusion(
            boxes, scores, labels, model_ids, num_models=2, iou_thr=0.55
        )

        print(fused)
source_hash: 3834f628efb1193d
---

## LibreEnsemble

```python
LibreEnsemble(
    members,
    *,
    weights=None,
    fusion="wbf",
    fusion_iou=0.55,
    min_votes=1,
)
```

| Argomento | Default | Significato |
|---|---|---|
| `members` | | Due o più detector |
| `weights` | `None` | Fattori di fiducia per membro; tutti `1.0` se omesso |
| `fusion` | `"wbf"` | `"wbf"`, `"wbf_seeded"`, `"nms"` o un callable |
| `fusion_iou` | `0.55` | Soglia IoU per il clustering della fusione |
| `min_votes` | `1` | Mantiene solo i box confermati da almeno questo numero di membri |

Un membro è un percorso di pesi risolto tramite la factory `LibreYOLO()`, un
modello già costruito, un backend esportato o un `ExternalDetector`. Ogni
membro deve essere un modello con task detect.

<code-tabs name="usage" />

La costruzione rifiuta meno di due membri, una lista `weights` di lunghezza
sbagliata, un peso non positivo, un `min_votes` che non sia un intero positivo
e un `min_votes` maggiore del numero di membri. Anche `fusion="nms"` con
`min_votes > 1` solleva un errore, perché l'NMS scarta l'appartenenza ai
cluster e non può contare i voti.

`weights` scala la fiducia riposta in ogni membro. Un peso più alto attira le
coordinate e i punteggi fusi verso quel membro. La convenzione è renderli
proporzionali al mAP di validazione.

## Spazi di classi

I membri con `names` identici passano direttamente. Altrimenti gli spazi di
classi vengono uniti per nome, gli ID di classe dei membri vengono rimappati
tramite tabelle di lookup e il `Results.names` fuso è l'unione. La fusione
unisce i box solo all'interno della stessa classe unificata, quindi una classe
conosciuta da un solo membro passa senza essere fusa. Una discrepanza registra
un warning alla costruzione.

`min_votes` è limitato per classe dal numero di membri i cui spazi di etichette
contengono quella classe, così il consenso resta significativo su vocabolari
condivisi solo in parte.

## Chiamare l'ensemble

```python
ens(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    output_path=None,
    color_format="auto",
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    **kwargs,
)
```

`predict` è un alias di `__call__`. Il valore restituito è il solito `Results`,
il cui `speed` scompone il costo per membro e aggiunge una voce `fusion`. Una
sorgente a immagine singola ne restituisce uno, una lista o una directory
restituisce una lista e `stream=True` restituisce un generatore.

`conf`, `iou` e `device` si propagano a ogni membro e accettano anche un valore
per membro, quindi `conf=[0.25, 0.4]` dà al membro 0 una soglia di 0.25 e al
membro 1 una soglia di 0.4. `imgsz` si propaga quando è un int o una tupla ed è
per membro solo quando è una lista, quindi `imgsz=(480, 640)` è un'unica
dimensione rettangolare per tutti mentre `imgsz=[480, 640]` è 480 per il membro
0 e 640 per il membro 1. Ogni voce deve essere valida per la famiglia di quel
membro.

`augment` si propaga ai membri che supportano la test-time augmentation, e i
backend esportati lo ignorano. `classes` accetta gli ID di classe dell'unione e
`max_det` si applica al risultato fuso, quindi i membri lavorano in modo
generoso e l'ensemble taglia una volta sola. `batch` è accettato per parità di
API; le immagini vengono elaborate in sequenza.

`val()` ed `export()` sollevano `NotImplementedError`. Valida ed esporta i
membri singolarmente.

## ExternalDetector

```python
ExternalDetector(fn: Callable, names: dict[int, str])
```

Adatta qualsiasi callable di rilevamento a membro dell'ensemble. `fn` riceve
un'immagine PIL e restituisce `(boxes, scores, labels)`, dove i box sono xyxy
in pixel dell'immagine originale e le etichette sono ID di classe validi in
`names`. Funzionano tensori, array e liste annidate. LibreYOLO non importa
nulla dal codice esterno.

L'adapter valida il valore restituito: deve essere una tupla di tre elementi, i
box devono avere forma `(N, 4)`, i tre array devono avere la stessa lunghezza e
ogni ID di classe deve comparire in `names`. I rilevamenti pari o inferiori a
`conf` vengono scartati prima della fusione.

## Operazioni di fusione

Le primitive di fusione sono op torch autonome in `libreyolo.ops`. Sono
indipendenti dal modello e importabili da sole, ed è per questo che vengono
esportate separatamente dall'ensemble.

<code-tabs name="ops" />

Tutte e tre accettano gli stessi argomenti posizionali, `boxes, scores, labels,
model_ids`, e restituiscono `(boxes, scores, labels)`.

| Op | Chiave nel registro | Comportamento |
|---|---|---|
| `weighted_boxes_fusion` | `wbf` | Weighted boxes fusion sequenziale, fedele al paper |
| `wbf_seeded` | `wbf_seeded` | Variante parallela a passata singola della stessa riduzione |
| `nms_fusion` | `nms` | Concatena tutto e applica una NMS per classe |

`FUSIONS` mappa le tre chiavi del registro sui callable, e `LibreEnsemble` ci
cerca dentro il valore di `fusion=`.

```python
weighted_boxes_fusion(
    boxes, scores, labels, model_ids,
    *,
    weights=None,
    num_models=None,
    iou_thr=0.55,
    skip_box_thr=0.0,
    conf_type="avg",
    min_votes=1,
    models_per_label=None,
    label_weights=None,
)
```

`wbf_seeded` ha una firma identica. `nms_fusion` accetta gli stessi argomenti
tranne `conf_type`, e solleva `ValueError` quando `min_votes > 1`.

In `weighted_boxes_fusion` i rilevamenti vengono visitati in ordine di
confidenza decrescente scalata per il peso. Ciascuno entra nel cluster
esistente con il cui box fuso corrente si sovrappone meglio, con IoU superiore
a `iou_thr` e con la stessa etichetta, oppure avvia un nuovo cluster. Il box
fuso di un cluster è la media delle coordinate dei suoi membri pesata sulla
confidenza, e il suo punteggio è la media pesata o il massimo delle loro
confidenze, riscalato in modo che i box confermati da meno modelli ottengano un
punteggio più basso.

`wbf_seeded` sceglie i seed dei cluster con una NMS per classe a `iou_thr`,
assegna ogni rilevamento al seed con la IoU migliore e la stessa etichetta,
poi riduce ogni cluster allo stesso modo. La forma dei cluster non cambia mai a
metà passata, quindi l'intera op è matematica tensoriale a forma fissa. Le due
varianti coincidono ogni volta che i cluster non sono ambigui e possono
differire leggermente su catene di cluster sovrapposti.

`nms_fusion` mantiene invariato il box a confidenza più alta di ogni gruppo
sovrapposto. I `weights` per modello scalano le confidenze solo ai fini
dell'ordinamento per la soppressione, e i box superstiti mantengono i punteggi
originali.

## Fusione personalizzata

`fusion=` accetta anche un callable con la stessa firma delle op qui sopra. Il
suo nome viene registrato in `ens.fusion`, oppure `"custom"` quando non ne ha.
Il valore restituito viene validato: deve essere una tripla `(boxes, scores,
labels)` con forme coerenti.
