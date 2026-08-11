---
title: Data augmentation
seo_title: Data augmentation per l'addestramento in LibreYOLO
description: >-
  I parametri di augmentation su TrainConfig, le quattro forme di pipeline che
  ci stanno dietro e la tabella per famiglia che dice quali parametri sono
  usati, gated o ignorati.
lead: >-
  L'augmentation si configura con i parametri di TrainConfig, ma ogni famiglia
  di modelli esegue la propria pipeline di addestramento, e una pipeline che non
  ha un ramo mosaic ignora mosaic_prob invece di approssimarlo.
keywords:
  - data augmentation yolo
  - mosaic augmentation
  - mixup
  - hsv jitter
  - random affine
  - copy paste augmentation
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_prob=1.0,
            mixup_prob=0.15,
            hsv_prob=1.0,
            flip_prob=0.5,
            no_aug_epochs=15,
        )
    - label: CLI
      language: bash
      code: |
        # La CLI scrive mosaic_prob come mosaic e mixup_prob come mixup.
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 mosaic=1.0 mixup=0.15 hsv_prob=1.0 \
          flip_prob=0.5 no_aug_epochs=15
  support:
    - label: Leggere la tabella di supporto di una famiglia
      language: python
      code: |
        from libreyolo.data.augment.spec import AUG_KNOBS, aug_support

        for knob, description in AUG_KNOBS.items():
            support = aug_support("yolo9")[knob]
            print(f"{knob:16} {support.status:16} {support.note or description}")
    - label: Solo quelli ignorati
      language: python
      code: |
        from libreyolo.data.augment.spec import ignored_aug_params

        print(sorted(ignored_aug_params("rfdetr")))
  classify:
    - label: Pacchetto di classificazione
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(
            data="my-classification-dataset",
            epochs=50,
            auto_augment="randaugment",
            erasing=0.25,
            mixup=0.2,
            cutmix=0.2,
        )
source_hash: 47461cd13aab580c
---

## Impostare i parametri

I parametri di augmentation sono normali argomenti di `train()`.

<code-tabs name="train" />

Due di questi hanno una forma più breve sulla CLI: `mosaic` corrisponde a
`mosaic_prob` e `mixup` corrisponde a `mixup_prob`. Tutti gli altri parametri si
scrivono allo stesso modo in entrambi i casi.

## Tre stati, non due

Che un parametro abbia effetto o meno dipende dalla famiglia. La libreria ne
tiene una tabella dichiarativa, e ogni voce è uno di tre stati.

`used` significa che il parametro arriva alla pipeline e modifica i campioni.
`ignored` significa che non arriva mai alla pipeline, quindi impostarlo non fa
nulla. `gated_by_mosaic` significa che si applica solo ai campioni passati dal
ramo mosaic, quindi con `mosaic_prob=0` non scatta mai anche se è collegato.

Il terzo stato è quello che coglie tutti di sorpresa. In una pipeline in stile
YOLOX la trasformazione affine viene applicata sul canvas del mosaic e MixUp
fonde un campione mosaic, quindi `mosaic_prob=0` disattiva silenziosamente `degrees`,
`translate`, `shear`, `perspective`, `mosaic_scale`, `mixup_prob` e
`mixup_scale` tutti insieme. Il trainer registra un avviso specifico per il caso
di MixUp:

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

La CLI segnala anche i parametri ignorati, elencando solo quelli che hai
effettivamente scritto:

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

## Quattro forme di pipeline

Le famiglie si raggruppano in quattro pipeline di addestramento, e la pipeline
determina quasi tutte le risposte.

La pipeline mosaic in stile YOLOX applica il jitter HSV e i flip per campione,
poi esegue l'affine e MixUp dentro il ramo mosaic. Copre YOLOX, YOLOv7, YOLOv9 e
le sue varianti E2E e P2, RTMDet, PicoDet, RT-DETR, RT-DETRv2 e FOMO.

La pipeline pass-through in stile DETR non ha né mosaic né trasformazione
affine. La sua distorsione fotometrica, lo zoom-out e il crop IoU sono costanti
della ricetta e non parametri di configurazione, quindi solo `flip_prob` e
`no_aug_epochs` sono attivi. Copre D-FINE, Dome-DETR, DEIM, DEIMv2, RT-DETRv4,
EC e, con una variazione, RF-DETR.

La pipeline ImageFolder di classificazione ignora tutti i parametri di
rilevamento. Il suo flip orizzontale è fisso a 0.5 e `flip_prob` non lo
raggiunge. Ha invece il suo pacchetto di parametri, descritto più sotto.

YOLO-NAS è una forma a sé: nessun mosaic, una trasformazione affine per campione
sempre attiva e MixUp applicato in modo indipendente invece che gated. Il suo
valore di `mosaic_scale` è riusato come intervallo di scala dell'affine.

SegFormer e NAFNet eseguono ciascuno una pipeline specifica per il task, la cui
casualità è fissata nella famiglia invece di essere configurabile. Per SegFormer
i parametri attivi sono gli attributi di classe `semantic_scale_jitter` e
`semantic_hsv_prob`, non `mosaic_scale` e `hsv_prob`. Il crop e i flip di NAFNet
sono operazioni accoppiate su input e target con probabilità fissa a 0.5.

## Quale famiglia rispetta quale parametro

La tabella qui sotto è la spec distribuita in
`libreyolo/data/augment/spec.py`, che i test della libreria stessa confrontano
con il funzionamento reale delle pipeline. Leggila lì invece di dedurla
dall'architettura.

<code-tabs name="support" />

Riassunto per pipeline, per i parametri di base:

| Parametro | Stile YOLOX | YOLO-NAS | Stile DETR | Classificazione |
|---|---|---|---|---|
| `mosaic_prob` | used | ignored | ignored | ignored |
| `mixup_prob` | gated by mosaic | used | ignored | ignored |
| `hsv_prob` | used | used | ignored | ignored |
| `flip_prob` | used | used | used | ignored |
| `flipud` | used | used | ignored | ignored |
| `degrees` | gated by mosaic | used | ignored | ignored |
| `translate` | gated by mosaic | used | ignored | ignored |
| `shear` | gated by mosaic | used | ignored | ignored |
| `perspective` | gated by mosaic | used | ignored | ignored |
| `mosaic_scale` | gated by mosaic | used | ignored | ignored |
| `mixup_scale` | gated by mosaic | used | ignored | ignored |
| `no_aug_epochs` | used | used | used | used |

Eccezioni all'interno di quelle colonne, tutte restrittive:

- RTMDet, PicoDet, RT-DETR, RT-DETRv2 e FOMO non hanno il flip verticale, quindi
  `flipud` è ignorato. Anche il wrapper mosaic di FOMO è costruito senza
  prospettiva.
- La pipeline nativa di RF-DETR non ha il jitter HSV, quindi `hsv_prob` è
  ignorato in aggiunta alla colonna in stile DETR.
- EC rispetta `hsv_prob`, `degrees` e `translate`, ma solo per `task="pose"`, la
  cui trasformazione, che tiene conto dei keypoint, li legge. I suoi percorsi
  detect e segment usano ricette fotometriche fisse.
- DINOv2 segue la colonna in stile DETR per i suoi task detect e semantic e
  aggiunge il pacchetto di classificazione per `task="classify"`.

`no_aug_epochs` è `used` ovunque, ma non significa la stessa cosa ovunque. Sulle
pipeline mosaic disattiva mosaic e MixUp per le epoche finali. Sulle pipeline in
stile DETR ferma le augmentation fotometriche, di zoom-out e di crop e dà forma
alla coda dello schedule. Sulle pipeline di classificazione e semantiche dà
forma solo alla coda.

## Il pacchetto di classificazione

Quattro parametri guidano la pipeline di classificazione e nient'altro. Le
famiglie di rilevamento li ignorano tutti e quattro.

<code-tabs name="classify" />

`auto_augment` accetta `"randaugment"`, `"autoaugment"`, `"augmix"` o `None`.
`erasing` è la probabilità di RandomErasing. `mixup` e `cutmix` sono probabilità
per batch che producono soft label; al massimo una viene applicata per batch,
MixUp per prima, quindi le due si sommano e la somma non dovrebbe superare 1.

Tutti e quattro sono disattivati di default, quindi l'addestramento di
classificazione non cambia se non lo chiedi.

Vale la pena dire chiaramente che c'è una collisione di nomi: sulla CLI, `mixup`
è l'alias del `mixup_prob` di rilevamento. Il campo `mixup` della
classificazione non ha una sua forma sulla CLI e si raggiunge solo con
`model.train(mixup=...)` in Python.

## Parametri specifici per famiglia

Alcuni parametri risiedono nella sottoclasse di configurazione di una famiglia
invece che nella classe base, quindi esistono solo per quella famiglia e non
hanno un flag sulla CLI.

| Famiglia | Parametro | Effetto |
|---|---|---|
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste` | Probabilità di augmentation copy-paste sulle istanze, solo `task="segment"` |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste_mode` | `"flip"` riusa lo stesso campione specchiato, `"mixup"` preleva un secondo campione |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `rot90` | Probabilità di rotazione casuale di 90 gradi |
| YOLOv9 | `max_labels` | Limite di ground truth per immagine nelle trasformazioni di addestramento, default 100 |
| RF-DETR | `copy_paste`, `copy_paste_mode` | Copy-paste per `task="segment"`, solo modalità `"flip"` |
| RF-DETR, D-FINE, EC | `crop_resize_prob` | Probabilità di crop-resize casuale |
| EC, YOLO-NAS | `brightness_contrast_prob`, `affine_prob` | Probabilità di jitter sul percorso pose e di affine che tiene conto dei keypoint |

`max_labels` è quello che perde dati in silenzio. I box oltre il limite vengono
scartati senza errore, quindi con immagini dense come la fotografia aerea va
alzato.

Mosaic e MixUp sono disattivati per l'addestramento con box orientati
indipendentemente dai parametri, perché l'augmentation che tiene conto degli
angoli per i box ruotati non è implementata.

## Correlati

- [Iperparametri](/docs/train/hyperparameters) per `no_aug_epochs` come
  argomento dello schedule e per il resto di `train()`.
- [Dataset](/docs/train/datasets) per i formati di etichette che queste
  trasformazioni accettano in ingresso.
