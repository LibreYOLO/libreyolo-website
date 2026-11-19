---
title: Matrice di data augmentation
seo_title: Quale famiglia LibreYOLO rispetta quale augmentation
description: >-
  Ogni parametro di augmentation con il suo intervallo e la sua grafia sulla
  CLI, lo stato di ogni parametro in ciascuna delle 31 famiglie addestrabili e i
  valori predefiniti con cui parte ogni famiglia.
lead: >-
  Impostare un parametro di augmentation non garantisce che arrivi alla
  pipeline. Questa pagina registra come ogni famiglia addestrabile tratta ogni
  parametro di TrainConfig e il valore da cui parte ogni famiglia, dati generati
  dalla libreria alla release che documenta.
keywords:
  - data augmentation libreyolo
  - parametri data augmentation
  - default data augmentation
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - mosaic_scale
  - no_aug_epochs
  - matrice supporto augmentation
  - parametri TrainConfig
last_verified: 1.8.0
verification: >-
  Generato da scripts/build-augmentation/collect.py a partire da LibreYOLO
  1.8.0: l'elenco dei parametri, gli stati e le note da
  libreyolo/data/augment/spec.py, che i test della libreria ancorano alle
  pipeline reali, i valori predefiniti dalla sottoclasse di TrainConfig di ogni
  famiglia e le grafie della CLI dal comando libreyolo train. Gli intervalli e
  l'unico punto in cui la spec e una pipeline non concordano sono verificati
  sulla libreria dallo stesso script.
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
    - label: I valori predefiniti di una famiglia
      language: python
      code: |
        from libreyolo.training.config import YOLO9Config

        config = YOLO9Config()
        print(config.mosaic_scale, config.degrees, config.mixup_prob)
source_hash: a88efd1537732e6c
---

## I parametri

Questi sono nomi di campi di `TrainConfig`. `train()` e la CLI accettano anche
le grafie brevi della colonna CLI, quindi `mosaic` imposta `mosaic_prob`. Sulla
CLI `mixup` viene instradato in base al task: mixing a livello di batch della
classificazione per i classificatori, `mixup_prob` per i rilevatori.
`perspective` non ha un flag sulla CLI e si imposta da Python.

<aug-knobs />

Il valore predefinito di base è il valore definito su `TrainConfig` stesso. La
maggior parte delle famiglie ne sovrascrive alcuni, e [Valori predefiniti per
famiglia](#defaults-by-family) riporta i valori da cui parte effettivamente un
addestramento. Per vedere in immagini cosa fa ogni parametro, consulta [Data
augmentation](/docs/train/augmentations).

Gli intervalli tra parentesi quadre seguono la notazione degli intervalli,
quindi `[0, 1)` include 0 ed esclude 1. I parametri di classificazione vengono
controllati quando si costruisce la loro pipeline e sollevano `ValueError` fuori
dal loro intervallo. I parametri di rilevamento non vengono controllati da
`TrainConfig`, quindi un valore fuori intervallo addestra senza segnalazioni. Le
eccezioni sono una scala estratta di 0 o meno e un `perspective` negativo, che
sollevano `ValueError` quando un campione viene trasformato.

<code-tabs name="usage" />

## I tre stati

| Stato | Significato |
|---|---|
| `used` | Il parametro arriva alla pipeline di addestramento della famiglia e modifica i campioni |
| `gated_by_mosaic` | Il parametro si applica solo ai campioni che hanno preso il ramo del mosaico, quindi con `mosaic_prob == 0` non si attiva mai |
| `ignored` | Il parametro non arriva mai alla pipeline; impostarlo non fa nulla |

`ignored` è quello che conviene controllare prima di un addestramento, perché
non fallisce nulla. La CLI avvisa quando un parametro di addestramento impostato
esplicitamente è uno di quelli che la famiglia selezionata ignora, e il trainer
avvisa quando `mixup_prob > 0` non può attivarsi perché la famiglia condiziona
MixUp al mosaico e `mosaic_prob` è zero.

## Quale famiglia rispetta quale parametro

<aug-matrix />

Ogni famiglia coperta segue una di sei pipeline. Nella pipeline in stile YOLOX
il preprocessing per campione applica il jitter HSV e i flip, mentre la
trasformazione affine e MixUp girano solo dentro il ramo del mosaico. YOLO-NAS
invece esegue una trasformazione affine per campione sempre attiva, ignora il
mosaico e applica MixUp in modo indipendente, riusando `mosaic_scale` come
intervallo di scala della trasformazione affine.

La pipeline in stile DETR è una trasformazione pass-through senza mosaico. La
sua distorsione fotometrica, lo zoom-out e il ritaglio IoU sono costanti della
ricetta e non parametri configurabili, ed è per questo che `hsv_prob` e i
parametri geometrici non la raggiungono mai. La classificazione usa `flip_prob`
per i flip orizzontali e `flipud` per quelli verticali. Il jitter di scala e
l'HSV delle famiglie semantiche vengono da attributi di classe della famiglia e
non da parametri di configurazione, e i flip del restauro sono operazioni
accoppiate su input e target con una probabilità fissa di 0.5.

`no_aug_epochs` è rispettato ovunque, anche se ciò che disattiva cambia: mosaico
e MixUp per lo stile YOLOX, la trasformazione affine e MixUp per YOLO-NAS, le
augmentation forti fotometriche e di ritaglio più la coda del learning rate per
lo stile DETR, e auto-augmentation, erasing, MixUp e CutMix per la
classificazione. Il ritaglio e i flip della classificazione restano attivi.

Dove una famiglia si discosta dal resto della sua pipeline, il motivo è uno di
questi:

| Famiglia | Differenza |
|---|---|
| `rtmdet`, `picodet`, `rtdetr`, `rtdetrv2` | `flipud` ignorato: le loro trasformazioni non hanno il flip verticale |
| `fomo` | `perspective` e `flipud` ignorati |
| `ppyoloe` | `flipud` ignorato: la ricetta originale usa invece una rotazione casuale di 90 gradi |
| `yolo9`, `yolo9_e2e`, `yolo9_p2`, `yolo7`, `rtdetr`, `rtdetrv2` | `mixup_scale` ignorato: il loro MixUp fonde immagini intere con un rapporto casuale senza riscalare nessuna delle due |
| `gtr` | `mosaic_prob` e `mixup_prob` usati, per le prime `mosaic_epochs` epoche; `degrees`, `translate` e `mosaic_scale` determinano la trasformazione affine che segue il mosaico |
| `rfdetr` | `hsv_prob` ignorato: la pipeline nativa non ha il jitter HSV |
| `ec` | `hsv_prob`, `degrees` e `translate` usati, solo per `task="pose"`; detect e segment usano ricette fotometriche fisse |
| `dinov2` | Il pacchetto di classificazione è usato, solo per `task="classify"` |

`ec` e `dinov2` sono famiglie multi-task, quindi un parametro è marcato come
ignorato solo quando tutti i task addestrabili della famiglia lo ignorano. Così
l'avviso della CLI non può mai essere sbagliato per un task e giusto per un
altro.

Dome-DETR eredita le trasformazioni di D-FINE senza modifiche. L'unica cosa che
non supporta è l'addestramento multi-scala, che viene disabilitato dalla sua
config e non dalla spec di augmentation.

Sono coperte trentuno famiglie. Per una famiglia fuori da questo elenco,
`ignored_aug_params` restituisce un insieme vuoto, quindi per essa non viene
emesso alcun avviso.

## Valori predefiniti per famiglia

Il valore da cui parte la config di ogni famiglia di rilevamento. Una cella
vuota significa che la famiglia ignora quel parametro, quindi il suo valore
predefinito non cambia nulla.

<aug-defaults />

Due schemi contano più di qualsiasi singolo valore. RT-DETR, RT-DETRv2, PicoDet
e FOMO partono da `mosaic_prob=0`, quindi in quelle famiglie i parametri marcati
come condizionati nella matrice non fanno nulla finché non si attiva il mosaico.
YOLOv9 parte con rotazione, shear e MixUp disattivati e con
`mosaic_scale=(0.5, 1.5)`, mentre il valore predefinito di base è `(0.1, 2.0)`.

EdgeCrafter legge `hsv_prob`, `degrees` e `translate` solo per `task="pose"`,
quindi la sua riga mostra i valori predefiniti del task pose per questi tre
parametri.

Le famiglie di classificazione partono dai valori predefiniti di base:
`flip_prob=0.5`, `flipud=0.0`, `scale=(0.5, 1.0)`, il `crop_pct` proprio della
famiglia, e `auto_augment`, `erasing`, `mixup` e `cutmix` disattivati.

## Parametri specifici per famiglia

Alcune famiglie portano parametri di augmentation sulla propria sottoclasse di
`TrainConfig` invece che sulla base. La CLI non li espone; impostali tramite
l'API Python.

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
| `ec`, `yolonas` | `affine_prob` | Probabilità della trasformazione affine che tiene conto dei keypoint, `task="pose"` |

`rot90` si applica a detect e OBB su `yolo9`.

## Interrogare la spec

| Helper | Restituisce |
|---|---|
| `aug_support(family)` | La tabella da parametro a `Support`, oppure `None` per una famiglia sconosciuta |
| `ignored_aug_params(family)` | L'insieme dei nomi dei parametri che la famiglia ignora; vuoto per una famiglia sconosciuta |
| `uses_mosaic_gating(family)` | Se il MixUp della famiglia si attiva solo sui campioni a mosaico |
| `display_name(family)` | Il nome della famiglia mostrato agli utenti negli avvisi |
| `mixup_gating_warning(family, mosaic_prob, mixup_prob)` | Il testo dell'avviso quando MixUp non può mai attivarsi, altrimenti `None` |

Un `Support` è una named tuple di `status` e `note`, dove la nota spiega perché
un parametro è ignorato o condizionato per quella famiglia. Nella 1.8.0 la spec
elenca ancora `mixup_scale` come condizionato per le famiglie che si addestrano
attraverso il wrapper del mosaico di YOLOv9, quindi `aug_support` lo riporta
come condizionato per loro anche se la matrice qui sopra, che è verificata sulla
pipeline, lo marca come ignorato.

## Il gate del mosaico

In una famiglia in stile YOLOX, `mixup_prob=0.5` con `mosaic_prob=0` disattiva
del tutto MixUp, perché MixUp si applica solo ai campioni a mosaico. È una
combinazione facile da ottenere quando si spegne il mosaico verso la fine
dell'addestramento. Il trainer registra un avviso che nomina la famiglia, e
`mixup_gating_warning` è la funzione pura che ci sta dietro.
