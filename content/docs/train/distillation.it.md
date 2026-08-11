---
title: Distillazione della conoscenza
seo_title: Distillazione della conoscenza in LibreYOLO
description: >-
  Addestra un detector piccolo contro un teacher più grande o contro un backbone
  DINOv2 congelato: le loss MGD, CWD e feature-MSE, i punti di prelievo e le
  famiglie supportate.
lead: >-
  La distillazione aggiunge un secondo termine di loss che avvicina le feature
  map intermedie dello student a quelle di un teacher congelato. LibreYOLO
  preleva le feature con i forward hook, quindi la testa e la loss del teacher
  stesso non intervengono mai.
keywords:
  - knowledge distillation
  - distillazione della conoscenza
  - masked generative distillation
  - distillazione channel-wise
  - distillazione delle feature
  - teacher dinov2
  - addestramento teacher student
  - mgd loss
  - cwd loss
last_verified: 1.5.0
snippets:
  detector:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Un checkpoint più grande della stessa famiglia supervisiona quello
        piccolo.

        model = LibreYOLO("LibreYOLO9s.pt")

        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="mgd",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=LibreYOLO9c.pt distill_loss_type=mgd
  foundation:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Un ViT auto-supervisionato e congelato supervisiona uno stadio del
        backbone.

        model = LibreYOLO("LibreYOLO9s.pt")

        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="dinov2",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=dinov2
  tuned:
    - label: Regolare la loss
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="cwd",
            dis=1.0,           # peso globale della distillazione
            distill_tau=1.0,   # temperatura del softmax di CWD
        )
source_hash: 7210031328f6826f
---

## Distillare da un checkpoint più grande

Impostare `distill_model` attiva la distillazione. Il valore è un checkpoint di
teacher, caricato con la stessa factory di qualsiasi altro modello.

<code-tabs name="detector" />

Il teacher esegue il forward sotto `no_grad`, e sotto autocast quando AMP è
attivo, così il modello congelato non paga calcolo a precisione piena a ogni
passo. I forward hook catturano le sue feature map nei punti di prelievo con
nome, la loss le confronta con quelle dello student, e il risultato si somma alla
loss di addestramento e viene riportato come componente di nome `distill`.

## Distillare da un backbone foundation congelato

In alternativa, un ViT auto-supervisionato può supervisionare un solo stadio del
backbone dello student. Le feature del teacher arrivano dal suo estrattore di
feature invece che dagli hook, e la loss gestisce il disallineamento tra una
griglia di patch e uno stride convoluzionale.

<code-tabs name="foundation" />

`distill_model` riconosce `dinov2`, che è DINOv2-base, più `dinov2_vits14`,
`dinov2_vitb14`, `dinov2_vitl14`, `dinov2-small`, `dinov2-base`, `dinov2-large`,
e qualsiasi id grezzo dell'hub che inizi con `facebook/dinov2`. Tutto il resto
viene trattato come il percorso di un checkpoint di teacher.

Questa strada usa `feat_mse` a prescindere da `distill_loss_type`, e richiede
`transformers` installato. Un teacher che si carica con chiavi di pesi mancanti
si interrompe invece di distillare contro un backbone in parte casuale.

## Quali famiglie

Il supporto alla distillazione è un metodo del modello student, e ce ne sono due.

`get_distill_config()` fornisce i punti di prelievo multi-scala che supervisiona
un teacher detector. YOLOv9, YOLOX e RF-DETR lo implementano.

`get_backbone_distill_config()` fornisce il singolo stadio del backbone che
supervisiona un teacher foundation. YOLOv9 lo implementa, ed è l'unica famiglia a
farlo.

Tutto il resto solleva un errore invece di addestrare senza la loss:

```text
LibreDFINE does not implement get_distill_config(). Distillation is not yet
supported for the 'dfine' family.
```

```text
Foundation-model distillation into the 'yolox' family is not supported yet
(no get_backbone_distill_config()).
```

## Punti di prelievo

I punti di prelievo sono fissi per famiglia e per ruolo, quindi teacher e student
non devono essere la stessa architettura; devono avere stride delle feature che
coincidono.

| Famiglia | Ruolo | Punti di prelievo | Stride |
|---|---|---|---|
| YOLOv9 | teacher o student | `neck.elan_up2`, `neck.elan_down1`, `neck.elan_down2` | 8, 16, 32 |
| YOLOv9 | student foundation | `backbone.elan3` | 16 |
| YOLOX | teacher o student | `backbone.C3_p3`, `backbone.C3_n3`, `backbone.C3_n4` | 8, 16, 32 |
| RF-DETR | teacher o student | `model.backbone.0.projector.stages.0` | rilevato durante il setup |

Gli stride che non coincidono sollevano un errore prima che l'addestramento
inizi:

```text
Teacher and student must have matching strides. Teacher: [8, 16, 32],
Student: [16]
```

Quel controllo viene saltato per i teacher foundation, il cui senso è proprio che
le griglie siano diverse.

## Le tre loss

`distill_loss_type` seleziona la loss sulle feature per un teacher detector. Un
teacher foundation usa sempre `feat_mse`.

`mgd`, masked generative distillation, maschera una frazione delle posizioni
spaziali dello student e addestra un piccolo generatore a due convoluzioni per
ricostruire l'intera feature map del teacher a partire da ciò che resta.
`distill_mask_ratio` imposta la frazione mascherata, di default 0.65.

`cwd`, channel-wise distillation, trasforma le attivazioni spaziali di ogni
canale in una distribuzione di probabilità e minimizza la divergenza KL canale
per canale. `distill_tau` è la temperatura del softmax, di default 1.0.

`feat_mse` allinea i canali dello student a quelli del teacher con una
convoluzione 1x1, ridimensiona la griglia del teacher a quella dello student in
modo bilineare, e prende l'errore quadratico medio. `distill_normalize=True`
normalizza prima entrambe le feature map con L2 sulla dimensione dei canali, il
che rende la corrispondenza solo di angolo e invariante alla scala. Di default
vale `False`.

`dis` è il peso globale applicato sopra a tutto. Se non lo imposti, ogni loss usa
il proprio valore predefinito pubblicato: 2e-5 per MGD, 1.0 per CWD e 1.0 per
feature MSE. Differiscono di cinque ordini di grandezza, quindi un peso regolato
per un tipo di loss non significa nulla per un altro.

<code-tabs name="tuned" />

`distill_mask_ratio`, `distill_tau` e `distill_normalize` non hanno flag da CLI.
Sono argomenti Python o chiavi YAML di `cfg=`. Anche RF-DETR è solo Python per la
distillazione nel suo complesso, perché la sua mappatura degli argomenti della
CLI non porta le chiavi della distillazione.

## Adattatori, checkpoint e multi-GPU

Ogni loss costruisce piccoli moduli addestrabili che vivono fuori dallo student:
gli adattatori di canale 1x1, e il generatore di MGD. Ricevono un proprio gruppo
di parametri dell'ottimizzatore al learning rate effettivo dell'esecuzione.

Quei moduli vengono scritti nel checkpoint sotto una chiave `distiller` e
ripristinati alla ripresa, così un'esecuzione ripresa non riparte con i proiettori
azzerati.

Sotto DDP gli adattatori stanno fuori dallo student incapsulato, il che significa
che il reducer di DDP non vede mai i loro gradienti. Il trainer esegue un
all-reduce esplicito su di essi a ogni passo, così ogni rank addestra gli stessi
adattatori.

La cattura dei CUDA graph non è disponibile in un'esecuzione con distillazione.
Passare `cuda_graph=True` registra una riga e addestra in modalità eager. Vedi
[Prestazioni dell'addestramento](/docs/train/performance).

## Correlati

- [Congelamento dei layer](/docs/train/layer-freezing) e
  [fine-tuning con LoRA](/docs/train/lora), a nessuno dei quali è impedito di
  essere combinato con la distillazione.
- [Iperparametri](/docs/train/hyperparameters) per il resto di `train()`.
