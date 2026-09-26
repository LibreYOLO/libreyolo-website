---
title: Iperparametri
seo_title: Iperparametri di addestramento in LibreYOLO
description: >-
  Gli argomenti di train() che contano: epochs, batch, lr0, optimizer, EMA,
  autobatch, accumulo dei gradienti e resume, più perché i default cambiano da
  famiglia a famiglia.
lead: >-
  Ogni argomento di addestramento è un campo di una dataclass TrainConfig. La
  classe base definisce il campo e il suo default; ogni famiglia di modelli ne
  fa una sottoclasse e sovrascrive i default che la sua ricetta pubblicata
  cambia.
keywords:
  - iperparametri addestramento yolo
  - learning rate
  - batch size
  - autobatch
  - media mobile esponenziale ema
  - gradient accumulation
  - riprendere addestramento yolo
  - early stopping patience
  - amp bfloat16
  - train config yaml
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        results = model.train(
            data="my-dataset.yaml",
            epochs=100,
            batch=16,
            imgsz=640,
            lr0=0.01,
        )

        print(results["best_mAP50_95"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 batch=16 imgsz=640 lr0=0.01
  defaults:
    - label: Leggere i default risolti di una famiglia
      language: python
      code: |
        from dataclasses import fields

        from libreyolo import LibreYOLO9
        from libreyolo.training.config import TrainConfig

        family_cfg = LibreYOLO9.TRAIN_CONFIG()
        base_cfg = TrainConfig()

        for f in fields(family_cfg):
            family_value = getattr(family_cfg, f.name)
            base_value = getattr(base_cfg, f.name, None)
            if not hasattr(base_cfg, f.name) or family_value != base_value:
                print(f"{f.name}: {family_value}")
    - label: CLI
      language: bash
      code: >
        # Stampa i default di train, val e predict, comprese le sovrascritture
        di famiglia.

        libreyolo cfg
  autobatch:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # batch=-1 sonda la memoria GPU e si risolve in una potenza di due
        concreta.

        model.train(data="my-dataset.yaml", batch=-1, imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml batch=-1
  accumulate:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 4 micro-batch da 16 per passo dell'optimizer, batch effettivo 64.
        model.train(data="my-dataset.yaml", batch=16, nbs=64)
  resume:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Carica il checkpoint della run interrotta, poi chiedi di riprendere.
        model = LibreYOLO("runs/train/exp/weights/last.pt")
        model.train(data="my-dataset.yaml", epochs=100, resume=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=runs/train/exp/weights/last.pt \
          data=my-dataset.yaml epochs=100 resume=true
  cfg:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Le chiavi nello yaml sono nomi di campo di TrainConfig. I kwarg
        espliciti vincono.

        model = LibreYOLO("LibreYOLO9s.pt")

        model.train(data="my-dataset.yaml", cfg="my-recipe.yaml", epochs=50)
source_hash: d838d1abd45af40f
---

## Impostare gli argomenti

`train()` accetta argomenti keyword e la CLI accetta gli stessi nomi in forma
`key=value`.

<code-tabs name="train" />

Entrambe le strade finiscono nello stesso punto. I kwarg vengono passati a
`TrainConfig.from_kwargs()`, che costruisce la dataclass di configurazione della famiglia.

## Un refuso non solleva errori

`from_kwargs()` scarta qualsiasi chiave che non sia un campo della configurazione ed
emette un `UserWarning` che la nomina. A quel punto l'addestramento parte con il valore
predefinito al suo posto:

```python
# UserWarning: Unknown training config keys (ignored): ['learning_rate']
model.train(data="my-dataset.yaml", learning_rate=0.001)
```

Niente fallisce, la run arriva in fondo e il learning rate (il tasso di apprendimento)
non è mai stato quello che il chiamante aveva chiesto. Leggi i warning alla prima
epoca di una nuova ricetta. La CLI è più severa, perché valida i nomi dei flag prima
che la configurazione venga costruita, quindi un flag CLI scritto male viene rifiutato
subito.

## I default sono per famiglia

`TrainConfig` definisce il campo e un default di base. Ogni famiglia ne fa una
sottoclasse e sovrascrive ciò che la sua ricetta pubblicata cambia, quindi non esiste
una sola risposta corretta a «qual è il learning rate predefinito».

I default di base sono `optimizer="sgd"`, `lr0=0.01`, `momentum=0.937`,
`weight_decay=5e-4`, `scheduler="yoloxwarmcos"`, `epochs=300`, `batch=16`,
`imgsz=640` e `amp=True`. Tre esempi di quanto una famiglia possa allontanarsi da
quei valori:

| Campo | Base | YOLOv9 | D-FINE | YOLO-NAS |
|---|---|---|---|---|
| `optimizer` | `sgd` | `sgd` | `adamw` | `adamw` |
| `lr0` | `0.01` | `0.01` | `2e-4` | `5e-4` |
| `weight_decay` | `5e-4` | `5e-4` | `1e-4` | `1e-5` |
| `scheduler` | `yoloxwarmcos` | `linear` | `flat_cosine` | `cos` |
| `epochs` | `300` | `300` | `132` | `300` |
| `amp` | `True` | `True` | `False` | `False` |

D-FINE e DEIM arrivano con `amp=False` perché il decoder di D-FINE limita le
attivazioni a 65504, il più grande valore finito rappresentabile in float16. Anche
YOLO-NAS e FOMO lo tengono disattivato per default. Il flag `--amp` della CLI vale `True` per default in
ogni famiglia, quindi conta come fornito dall'utente e sovrascrive il default della
famiglia; lascialo stare a meno che tu non voglia davvero cambiarlo.

Per leggere i default reali di una famiglia invece di tirare a indovinare:

<code-tabs name="defaults" />

## Dimensione del batch

`batch` è il batch globale. Nell'addestramento multi-GPU ogni rank carica
`batch // world_size`, quindi il numero che passi è il numero di immagini per passo
dell'optimizer, indipendentemente da quante GPU sono coinvolte. Vedi
[Addestramento multi-GPU](/docs/train/multi-gpu).

`batch=-1` attiva l'autobatch. Il trainer sonda il modello in modalità di addestramento
con un vero passo backward su potenze di due, interpola una retta sulla curva di
memoria e sceglie la più grande potenza di due strettamente sotto il valore estrapolato
che rientra nel 60 per cento della VRAM totale.

<code-tabs name="autobatch" />

Sondare in modalità di addestramento con un passo backward è il punto chiave:
una sonda in modalità inferenza non tiene conto delle attivazioni trattenute e dei
tensori dei gradienti, che per una CNN profonda valgono diverse volte l'ingombro
dell'inferenza. RF-DETR abbassa la frazione obiettivo al 45 per cento, perché il
backward sintetico della sonda sottostima comunque quanto costano il suo criterion e i
layer del decoder ausiliario.

L'autobatch è una funzionalità CUDA. Su CPU o MPS scrive una riga di log e mantiene il
batch predefinito.

## Accumulo dei gradienti

`nbs` imposta la dimensione nominale, o effettiva, del batch. Il trainer accumula
`round(nbs / batch)` micro-batch per passo dell'optimizer.

<code-tabs name="accumulate" />

Lasciato a `None`, il valore predefinito, l'accumulo è disattivato e l'addestramento non
cambia.

## Learning rate e schedule

`lr0` è il learning rate iniziale e `optimizer` accetta `sgd`, `adam` e `adamw`.
`momentum` è il momentum di SGD o beta1 di Adam, `weight_decay` è il termine L2 e
`nesterov` vale per SGD.

La forma dello schedule è data da `scheduler`, `warmup_epochs`, `warmup_lr_start` e
`min_lr_ratio`. `no_aug_epochs` stabilisce quante epoche finali girano senza data
augmentation forte, e diversi schedule lo usano anche per dare forma alla loro coda,
quindi non è solo una leva per la data augmentation. Cosa fa ogni famiglia con la parte
di data augmentation è su [Data augmentation](/docs/train/augmentations).

Alcune famiglie aggiungono le proprie leve per il learning rate.
`backbone_lr_mult` scala il gruppo del backbone rispetto alla testa, `clip_max_norm`
imposta il gradient clipping e SegFormer usa `head_lr_mult` per far girare la sua
decode head a dieci volte il learning rate del backbone. Queste stanno sulla sottoclasse
di configurazione della famiglia, non su quella di base.

## EMA

`ema=True` mantiene una media mobile esponenziale dei pesi accanto a quelli addestrati.
È attiva per default ovunque tranne che su FOMO.

`ema_decay` è il decay obiettivo. Il decay sale gradualmente invece di partire dal suo
obiettivo: il valore effettivo all'aggiornamento `n` è `ema_decay * (1 - exp(-n / tau))`
con `tau` che vale 2000 per default, quindi gli aggiornamenti iniziali seguono il
modello più da vicino e quelli tardivi lo smussano. I default di famiglia vanno da
`0.997` su YOLO-NAS pose, passando per `0.9998` su YOLOX, fino a `0.9999` su YOLOv9 e
sulla linea DETR.

I pesi EMA sono quelli che vengono validati e quelli che `best.pt` e `last.pt` portano
con sé. Anche i pesi addestrati grezzi vengono salvati, sotto la chiave `train_model`,
così un resume riprende dalla traiettoria addestrata invece che dalla media.

## Precisione

`amp=True` esegue il passo forward sotto l'autocast CUDA. `amp_dtype` sceglie
`float16` (il default) o `bfloat16`; `fp16` e `bf16` sono grafie accettate.

Float16 ha bisogno del loss scaling dinamico e riceve un `GradScaler` attivo. Il range
di esponenti più ampio di bfloat16 non ne ha bisogno, quindi il suo scaler viene
costruito ma disabilitato, il che mantiene identico il percorso dell'optimizer. Chiedere
bfloat16 su un dispositivo CUDA senza supporto bfloat16 solleva un errore durante il
setup invece di degradare in silenzio.

## Output, checkpoint e arresto

Le run vengono scritte in `project/name`. `project` vale `runs/train` per default
ovunque, ma `name` è una delle sovrascritture per famiglia: il default di base è `exp`,
mentre YOLOv9 usa `yolo9_exp` e D-FINE usa `dfine_exp`. Con `exist_ok=False`, il valore
predefinito, una directory esistente riceve un suffisso incrementale invece di essere
sovrascritta.

`save_period` scrive un `weights/epoch_<N>.pt` extra ogni N epoche, in aggiunta a
`weights/last.pt` dopo ogni epoca e a `weights/best.pt` ogni volta che la metrica
tracciata migliora. `eval_interval` stabilisce ogni quanto gira la validazione e
`patience` ferma la run dopo quel numero di epoche senza miglioramenti, con `0` che
disattiva l'early stopping.

`cache` accelera le epoche ripetute tenendo le immagini decodificate in RAM (`True` o
`"ram"`) o come file `.npy` accanto alle sorgenti (`"disk"`). Le letture dalla cache
sono identiche byte per byte a quelle lette direttamente dalle sorgenti. Con i worker
del dataloader, `"disk"` è la più sicura delle due.

## Resume

`resume=True` continua una run interrotta. Il checkpoint va caricato prima, perché
resume lo legge dal modello, non da un argomento separato.

<code-tabs name="resume" />

Resume ripristina i pesi addestrati, lo stato dell'optimizer, i pesi EMA e il conteggio
degli aggiornamenti, il tracciamento della metrica migliore, la scala del `GradScaler` e
gli stati random di PyTorch, CUDA e NumPy. Riparte dall'epoca del checkpoint più uno e
manda avanti lo schedule fino a quella posizione.

Ci sono due cose che non fa. `resume=True` non si può combinare con `pretrained`, e
provarci solleva un errore. E quando la chiave della metrica migliore del checkpoint è
diversa da quella della run corrente, il tracciamento della metrica migliore si azzera con un
warning invece di confrontare valori che non significano la stessa cosa.

## Ricette in un file

`cfg=` carica una mappatura YAML di nomi di campo di `TrainConfig` e la fonde sotto gli
argomenti keyword espliciti, quindi un kwarg vince sempre sul file.

<code-tabs name="cfg" />

`size` e `num_classes` vengono rimossi dal file, perché l'istanza del modello li
possiede già. Non c'è un flag `--cfg` sulla CLI; il percorso del file è un argomento
Python.

## Correlati

- [Dataset](/docs/train/datasets) per cosa accetta `data=`.
- [Data augmentation](/docs/train/augmentations) per le leve della data augmentation e
  quali famiglie le rispettano.
- [Congelamento dei layer](/docs/train/layer-freezing) e [LoRA](/docs/train/lora) per
  addestrare un sottoinsieme dei pesi.
- [Validazione e metriche](/docs/train/validation) per cosa riporta la run.
