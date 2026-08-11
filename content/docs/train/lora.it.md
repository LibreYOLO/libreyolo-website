---
title: Fine-tuning con LoRA
seo_title: Fine-tuning con LoRA in LibreYOLO
description: >-
  Fai fine-tuning di un detector transformer con poca VRAM usando lora=True.
  Quali nove famiglie lo supportano, la ricetta degli adapter di ciascuna e come
  si comportano i checkpoint.
lead: >-
  LoRA congela le parti pesanti preaddestrate di un modello e addestra accanto a
  esse piccoli adapter a basso rango, più i layer che devono restare densi. In
  LibreYOLO l'intera interfaccia pubblica è un booleano.
keywords:
  - lora fine tuning
  - parameter efficient fine tuning
  - peft
  - dora
  - addestrare con poca vram
  - rf-detr lora
  - d-fine lora
  - merge adapter lora
last_verified: 1.5.0
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install "libreyolo[lora]"
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 lora=true
  merge:
    - label: L'esportazione unisce gli adapter
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        model.export(format="onnx")
    - label: Unire in memoria
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training.lora import merge_lora_adapters

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        merged = merge_lora_adapters(model.model)

        print(f"{merged} adapter layers folded into dense weights")
source_hash: 603fdddf5ec0c316
---

## Installazione

LoRA si appoggia alla dipendenza opzionale `peft`.

<code-tabs name="install" />

Senza di essa, `lora=True` solleva un `ImportError` che nomina quel comando,
invece di addestrare per sbaglio un fine-tuning completo.

## Come si usa

<code-tabs name="train" />

`lora=True` è tutta l'interfaccia. Rango, alpha, dropout e moduli target sono
fissati per famiglia per corrispondere a ciascun riferimento upstream, e non sono
parametri esposti all'utente.

Una famiglia che non supporta LoRA solleva un errore al setup invece di ignorare
il flag:

```text
LoRA fine-tuning (lora=True) is not supported for yolo9. LoRA targets
transformer components with nn.Linear layers (e.g. RF-DETR, D-FINE, DEIM).
```

La CLI lo rifiuta prima ancora, prima che il modello venga costruito, usando la
sua allowlist delle stesse nove famiglie.

## Quali famiglie

RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 e v4, EC e ConvNeXt. Il controllo è
l'attributo `supports_lora` sulla classe trainer di ogni famiglia, e la CLI porta
con sé una allowlist corrispondente.

La copertura dei task è più stretta di quella delle famiglie. D-FINE ed EC
supportano solo il rilevamento, e i loro percorsi segment e pose sollevano un
errore. Il percorso semantico di RF-DETR solleva un errore. ConvNeXt è
classificazione.

Tutto il resto solleva un errore. Non esiste una modalità parziale o silenziosa.

## Cosa fa ogni ricetta

Le ricette differiscono perché differiscono le architetture, e una ricetta che
funziona su un backbone ViT non ha nulla a cui agganciarsi su uno convoluzionale.

RF-DETR usa DoRA, LoRA con decomposizione dei pesi, a rango 16 e alpha 16 sulle
proiezioni di attenzione `query`, `key` e `value` del backbone DINOv2, in linea
con il riferimento RF-DETR. Il backbone ViT si congela; il projector, il decoder
e la testa di rilevamento continuano ad addestrarsi normalmente.

D-FINE, DEIM e RT-DETR v1, v2 e v4 abbinano un backbone convoluzionale a un
encoder ibrido transformer e a un decoder deformabile, quindi la divisione si
sposta. Il backbone convoluzionale si congela per intero, il che salta anche il
suo backward pass. I blocchi transformer congelano i loro pesi di base e
addestrano semplici adapter LoRA allo stesso rango 16 e alpha 16 sui loro layer
lineari: i feed-forward `linear1` e `linear2`, il gate e le proiezioni di
deformable attention. Tutto il resto, la fusione convoluzionale dell'encoder, le
proiezioni di input, le teste di predizione e i query embedding, continua ad
addestrarsi in modo denso.

Due dettagli di quella ricetta sono voluti. La self-attention del decoder resta
congelata senza adapter, perché `nn.MultiheadAttention` di PyTorch legge
direttamente `out_proj.weight` e aggirerebbe silenziosamente un adapter iniettato.
Ed è LoRA semplice invece di DoRA, perché diversi layer lineari del decoder sono
inizializzati a zero per scelta progettuale e la normalizzazione della magnitudine
di DoRA divide per la norma dei pesi.

DEIMv2 adotta la stessa ricetta con i suoi layer feed-forward SwiGLU `w12` e `w3`
come target. Le sue taglie S, M, L e X portano anche un backbone ViT DINOv3, dove
la base ViT si congela e i suoi layer di attenzione fusa `qkv` ricevono gli
adapter, mentre la piramide di convoluzioni dello Spatial Tuning Adapter continua
ad addestrarsi come analogo del projector. Quegli adapter su `qkv` entrano anche
quando la config ha spedito il ViT già congelato, dato che adattare un backbone
congelato è proprio il punto. Le taglie sotto la S usano un backbone
convoluzionale e prendono la ricetta semplice.

EC è un DETR il cui backbone è un ViT circondato da una piramide di projector
convoluzionali addestrabile. La base ViT si congela e i suoi layer `qkv` ricevono
gli adapter, i blocchi transformer prendono la ricetta condivisa, e il projector e
le teste restano densi.

I blocchi ConvNeXt portano MLP lineari in formato channels-last, `fc1` e `fc2`, e
quelli prendono adapter semplici. Le convoluzioni depthwise, le norm e i parametri
di layer-scale si congelano. La testa di classificazione resta densa, così i
conteggi di classi personalizzati continuano a funzionare.

Le teste di rilevamento e di classificazione restano sempre addestrabili in ogni
ricetta, perché un conteggio di classi personalizzato ha bisogno di una testa
addestrata da zero.

## Checkpoint ed esportazione

`best.pt` e `last.pt` conservano i tensori degli adapter, così un run LoRA si
riprende o si ispeziona come qualsiasi altro. Caricare uno di quei checkpoint
richiede l'extra `lora` installato, perché il loader ripete l'iniezione degli
adapter in modo che le chiavi combacino.

`export()` unisce gli adapter in pesi densi, così un artefatto esportato non porta
alcuna dipendenza da `peft`. Lo stesso merge è disponibile direttamente per un
modello in memoria.

<code-tabs name="merge" />

Dopo un merge l'albero dei moduli è completamente denso e un secondo merge non fa
nulla.

## Cosa fa risparmiare, e cosa no

LoRA taglia la memoria dell'optimizer e dei gradienti, e sulle famiglie che
congelano del tutto il backbone salta anche il backward pass di quel backbone.

La memoria delle attivazioni non cambia. Le attivazioni del forward vanno comunque
mantenute per tutto ciò che resta addestrabile, ed è di solito quello a fissare il
picco. Per il budget di VRAM più stretto, abbassa anche `batch` o `imgsz`.

## Correlati

- [Congelamento dei layer](/docs/train/layer-freezing) per l'altro modo di
  addestrare un sottoinsieme dei pesi, che funziona su ogni famiglia e non
  richiede dipendenze extra. `freeze` e `lora=True` si combinano: i parametri
  degli adapter restano addestrabili anche quando il gruppo backbone che li
  contiene è congelato.
- [Iperparametri](/docs/train/hyperparameters) per `batch`, `imgsz` e il resto di
  `train()`.
