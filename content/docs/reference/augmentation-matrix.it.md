---
title: Matrice di data augmentation
seo_title: Quale famiglia LibreYOLO rispetta quale parametro di data augmentation
description: >-
  Supporto dei parametri di data augmentation per famiglia: i sedici parametri
  di TrainConfig, i tre stati, i sei archetipi di pipeline e i parametri che una
  famiglia ignora in silenzio.
lead: >-
  Impostare un parametro di data augmentation non garantisce che arrivi alla
  pipeline. Questa pagina registra come ogni famiglia addestrabile tratta ogni
  parametro di TrainConfig, a partire dalla tabella dichiarativa che la libreria
  include come unica fonte di verità.
keywords:
  - data augmentation libreyolo
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - no_aug_epochs
  - matrice supporto augmentation
  - parametri TrainConfig
last_verified: 1.5.0
verification: >-
  Elenco dei parametri, stati, archetipi, deviazioni per famiglia e funzioni
  helper letti da libreyolo/data/augment/spec.py alla v1.5.0. Quella tabella è
  ancorata alle pipeline reali da tests/unit/test_augment_spec.py.
snippets:
  usage:
    - label: Interrogare direttamente la spec
      language: python
      code: |
        from libreyolo.data.augment.spec import (
            AUG_KNOBS,
            aug_support,
            ignored_aug_params,
            uses_mosaic_gating,
        )

        print(sorted(AUG_KNOBS))

        table = aug_support("yolo9")
        print(table["mixup_prob"].status, table["mixup_prob"].note)

        print(sorted(ignored_aug_params("dfine")))
        print(uses_mosaic_gating("yolo9"), uses_mosaic_gating("yolonas"))
source_hash: d2e1b9f5c81072e1
---

## I parametri

Questi sono i nomi dei campi di `TrainConfig`, non le grafie della CLI. La CLI
mappa i propri alias su di essi, quindi `--mosaic` imposta `mosaic_prob`.

| Parametro | Significato |
|---|---|
| `mosaic_prob` | Probabilità di costruire un campione a mosaico da 4 immagini |
| `mixup_prob` | Probabilità di fondere un secondo campione |
| `hsv_prob` | Probabilità di jitter di colore HSV |
| `flip_prob` | Probabilità di flip orizzontale |
| `degrees` | Intervallo di rotazione casuale per la trasformazione affine, in gradi |
| `translate` | Frazione di traslazione casuale per la trasformazione affine |
| `mosaic_scale` | Intervallo di scala casuale per la trasformazione affine |
| `mixup_scale` | Intervallo di scala del jitter applicato all'immagine partner del MixUp |
| `shear` | Intervallo di shear casuale per la trasformazione affine, in gradi |
| `perspective` | Entità della deformazione prospettica per la trasformazione affine |
| `flipud` | Probabilità di flip verticale |
| `no_aug_epochs` | Epoche finali addestrate con la data augmentation forte disattivata |
| `auto_augment` | Policy AutoAugment per la classificazione: randaugment, autoaugment o augmix |
| `erasing` | Probabilità di RandomErasing per la classificazione |
| `mixup` | Probabilità di MixUp a livello di batch per la classificazione, con etichette soft |
| `cutmix` | Probabilità di CutMix a livello di batch per la classificazione, con etichette soft |

Gli ultimi quattro sono il pacchetto di classificazione. Le famiglie di
rilevamento li ignorano. `mixup` è un parametro disponibile solo via API: il
`--mixup` della CLI è l'alias del `mixup_prob` del rilevamento.

<code-tabs name="usage" />

## I tre stati

| Stato | Significato |
|---|---|
| `used` | Il parametro arriva alla pipeline di addestramento della famiglia e modifica i campioni |
| `gated_by_mosaic` | Il parametro si applica solo ai campioni che hanno preso il ramo mosaico, quindi con `mosaic_prob == 0` non si attiva mai |
| `ignored` | Il parametro non arriva mai alla pipeline; impostarlo non fa nulla |

`ignored` è quello che conviene controllare prima di un addestramento, perché
non fallisce nulla. La CLI avvisa quando un parametro di addestramento impostato
esplicitamente è uno di quelli che la famiglia selezionata ignora, e il trainer
avvisa quando `mixup_prob > 0` non può attivarsi perché la famiglia condiziona
MixUp al mosaico e `mosaic_prob` è zero.

## Archetipi di pipeline

Ogni famiglia coperta segue una di sei pipeline, con una manciata di deviazioni
per famiglia elencate più sotto.

| Parametro | Stile YOLOX | YOLO-NAS | Stile DETR | Classificazione | Semantica | Restauro |
|---|---|---|---|---|---|---|
| `mosaic_prob` | used | ignored | ignored | ignored | ignored | ignored |
| `mixup_prob` | gated | used | ignored | ignored | ignored | ignored |
| `hsv_prob` | used | used | ignored | ignored | ignored | ignored |
| `flip_prob` | used | used | used | ignored | ignored | ignored |
| `degrees` | gated | used | ignored | ignored | ignored | ignored |
| `translate` | gated | used | ignored | ignored | ignored | ignored |
| `mosaic_scale` | gated | used | ignored | ignored | ignored | ignored |
| `mixup_scale` | gated | used | ignored | ignored | ignored | ignored |
| `shear` | gated | used | ignored | ignored | ignored | ignored |
| `perspective` | gated | used | ignored | ignored | ignored | ignored |
| `flipud` | used | used | ignored | ignored | ignored | ignored |
| `no_aug_epochs` | used | used | used | used | used | used |
| `auto_augment` | ignored | ignored | ignored | used | ignored | ignored |
| `erasing` | ignored | ignored | ignored | used | ignored | ignored |
| `mixup` | ignored | ignored | ignored | used | ignored | ignored |
| `cutmix` | ignored | ignored | ignored | used | ignored | ignored |

Nella pipeline in stile YOLOX il preprocessing per campione applica il jitter
HSV e i flip, mentre la trasformazione affine e MixUp girano solo dentro il ramo
mosaico. YOLO-NAS invece esegue un'affine per campione sempre attiva, ignora il
mosaico e applica MixUp in modo indipendente, riusando `mosaic_scale` come
intervallo di scala dell'affine.

La pipeline in stile DETR è una trasformazione pass-through senza mosaico. La
sua distorsione fotometrica, lo zoom-out e il ritaglio IoU sono costanti della
ricetta e non parametri configurabili, ed è per questo che `hsv_prob` e i
parametri di geometria non la raggiungono mai. La pipeline di classificazione
usa una trasformazione ImageFolder il cui flip orizzontale è fisso a 0.5 invece
di `flip_prob`. Il jitter di scala e l'HSV della semantica arrivano da attributi
di classe della famiglia e non da parametri di configurazione, e i flip del
restauro sono operazioni accoppiate su input e target con probabilità fissa 0.5.

`no_aug_epochs` è rispettato ovunque, anche se ciò che disattiva cambia: mosaico
e MixUp per lo stile YOLOX, l'affine e MixUp per YOLO-NAS, le augmentation
fotometriche forti e di ritaglio più la coda del learning rate per lo stile
DETR, e la coda dello scheduler per il resto.

## Famiglie per archetipo

| Archetipo | Famiglie |
|---|---|
| Stile YOLOX | `yolox`, `yolo7`, `yolo9`, `yolo9_e2e`, `yolo9_p2`, `rtmdet`, `picodet`, `rtdetr`, `rtdetrv2`, `fomo` |
| YOLO-NAS | `yolonas` |
| Stile DETR | `dfine`, `domedetr`, `deim`, `deimv2`, `rtdetrv4`, `rfdetr`, `ec`, `dinov2` |
| Classificazione | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` |
| Semantica | `segformer` |
| Restauro | `nafnet` |

Sono coperte venticinque famiglie. Una famiglia fuori da questo elenco
restituisce un insieme di parametri ignorati vuoto, quindi per essa non viene
emesso alcun avviso.

## Deviazioni

| Famiglia | Differenza rispetto al suo archetipo |
|---|---|
| `rtmdet` | `flipud` ignorato: la sua trasformazione non ha il flip verticale |
| `picodet` | `flipud` ignorato |
| `rtdetr` | `flipud` ignorato |
| `rtdetrv2` | `flipud` ignorato |
| `fomo` | `perspective` e `flipud` ignorati |
| `ec` | `hsv_prob`, `degrees` e `translate` usati, solo per `task="pose"`; detect e segment usano ricette fotometriche fisse |
| `dinov2` | Il pacchetto di classificazione è usato, solo per `task="classify"` |

`ec` e `dinov2` sono famiglie multi-task, quindi un parametro è marcato come
ignorato solo quando tutti i task addestrabili della famiglia lo ignorano. Così
l'avviso della CLI non può mai essere sbagliato per un task e giusto per un
altro.

Dome-DETR eredita le trasformazioni di D-FINE senza modifiche. L'unica cosa che
non regge è l'addestramento multi-scala, disabilitato dalla sua config e non
dalla spec di augmentation.

## Parametri specifici per famiglia

Alcune famiglie portano i parametri di data augmentation sulla propria
sottoclasse di `TrainConfig` invece che sulla base. La CLI non li espone;
impostali tramite l'API Python.

| Famiglia | Parametro | Significato |
|---|---|---|
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste` | Probabilità della data augmentation copy-paste sulle istanze, solo `task="segment"` |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste_mode` | Sorgente del copy-paste: `flip` specchia lo stesso campione, `mixup` usa un secondo campione |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `rot90` | Probabilità di rotazione casuale di 90 gradi |
| `rfdetr` | `copy_paste` | Probabilità di copy-paste per `task="segment"`, solo in modalità `flip` |
| `rfdetr` | `copy_paste_mode` | Modalità della sorgente copy-paste per `task="segment"` |
| `rfdetr` | `crop_resize_prob` | Probabilità di crop-resize casuale nella pipeline nativa |
| `dfine` | `crop_resize_prob` | Probabilità di crop-resize casuale, `task="segment"` |
| `ec` | `crop_resize_prob` | Probabilità di crop-resize casuale, `task="segment"` |
| `ec`, `yolonas` | `brightness_contrast_prob` | Probabilità di jitter di luminosità e contrasto, `task="pose"` |
| `ec`, `yolonas` | `affine_prob` | Probabilità dell'affine consapevole dei keypoint, `task="pose"` |

`rot90` si applica a detect e OBB su `yolo9`.

## Interrogare la spec

| Helper | Restituisce |
|---|---|
| `aug_support(family)` | La tabella parametro-`Support`, oppure `None` per una famiglia sconosciuta |
| `ignored_aug_params(family)` | L'insieme dei nomi dei parametri che la famiglia ignora; vuoto per una famiglia sconosciuta |
| `uses_mosaic_gating(family)` | Se il MixUp della famiglia si attiva solo sui campioni a mosaico |
| `display_name(family)` | Il nome della famiglia mostrato agli utenti negli avvisi |
| `mixup_gating_warning(family, mosaic_prob, mixup_prob)` | Il testo dell'avviso quando MixUp non può mai attivarsi, altrimenti `None` |

Un `Support` è una named tuple di `status` e `note`, dove la nota spiega perché
un parametro è ignorato o gated per quella famiglia.

## Il gate del mosaico

In una famiglia in stile YOLOX, `mixup_prob=0.5` con `mosaic_prob=0` disattiva
del tutto MixUp, perché MixUp si applica solo ai campioni a mosaico. È una
combinazione facile da ottenere quando si spegne il mosaico verso la fine
dell'addestramento. Il trainer registra un avviso che nomina la famiglia, e
`mixup_gating_warning` è la funzione pura che ci sta dietro.
