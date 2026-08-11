---
title: Addestramento multi-GPU
seo_title: Addestramento multi-GPU in LibreYOLO
description: >-
  Addestra su più GPU con device="0,1". Come la libreria avvia i worker DDP,
  perché batch è il batch globale, quando impostare sync_bn e il percorso
  torchrun.
lead: >-
  L'addestramento multi-GPU in LibreYOLO è DistributedDataParallel di PyTorch:
  un processo per GPU, ognuno con una replica completa del modello e una
  porzione di ogni batch, con i gradienti mediati tra i rank a ogni passo.
keywords:
  - addestramento ddp pytorch
  - multi gpu training
  - torchrun nproc_per_node
  - distributed data parallel
  - syncbatchnorm
  - batch size globale
  - nccl gloo backend
  - multi gpu windows
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Il guard __main__ è obbligatorio: ogni worker avviato reimporta questo

        # modulo, e senza il guard rilancerebbe l'addestramento in modo
        ricorsivo.

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="my-dataset.yaml",
                epochs=100,
                batch=32,     # batch globale: 16 immagini per GPU con due GPU
                device="0,1",
            )
  torchrun:
    - label: train.py
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(data="my-dataset.yaml", epochs=100, batch=32)
    - label: Avvio
      language: bash
      code: |
        torchrun --nproc_per_node=2 train.py
  syncbn:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreRTDETRr18.pt")
            model.train(
                data="my-dataset.yaml",
                batch=32,
                device="0,1",
                sync_bn=True,
            )
  autobatch:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            # Misurato una volta sulla GPU 0, scalato a un multiplo del world size.
            model.train(data="my-dataset.yaml", batch=-1, device="0,1")
source_hash: 83c1563d68068cd0
---

## Eseguire su due GPU

Passa una lista di dispositivi. Non cambia nient'altro.

<code-tabs name="train" />

Quando riceve più di un dispositivo e non c'è un ambiente torchrun, il `train()`
del modello salva i pesi in un file temporaneo, risolve l'autobatch se richiesto e
avvia un processo worker per GPU con `torch.multiprocessing.spawn`. Ogni worker
reimporta la classe del modello, la ricostruisce dai pesi salvati ed esegue il
normale percorso a dispositivo singolo, perché dall'interno di un worker avviato
le variabili d'ambiente di torchrun sono impostate. Al termine dell'esecuzione, il
miglior checkpoint del rank 0 viene ricaricato nell'istanza del modello del
chiamante.

`device` accetta `"0,1"`, `[0, 1]`, `0`, `"cuda:0"`, `"cpu"`, `"mps"` e
`"auto"`. Solo una lista di più di un indice CUDA innesca lo spawn.

## Il guard `__main__` è obbligatorio

I worker avviati reimportano il modulo da cui provengono. Senza un guard
`if __name__ == "__main__":`, quell'import riesegue la chiamata di addestramento e
ogni worker avvia i propri worker. La libreria rileva il caso e solleva un errore
invece di lasciare che ricorra:

```text
spawn_ddp_train() was called from inside a spawned subprocess. This usually
means your script calls model.train(device=...) at the top level without a
'if __name__ == "__main__":' guard.
```

Tutto ciò che passa dentro a un worker viene serializzato con pickle, quindi
`callbacks=` deve essere picklable. Una classe a livello di modulo funziona; una
closure o una lambda no, e l'errore lo dice e indica i logger integrati come
alternativa.

## batch è il batch globale

`batch` è il numero di immagini per passo dell'ottimizzatore su tutte le GPU. Il
dataloader di ogni rank viene costruito con `batch // world_size` e un
`DistributedSampler`, quindi `batch=32` su due GPU significa 16 immagini per GPU,
non 32.

Un batch che non è divisibile esattamente per il world size solleva un errore
invece di addestrare in silenzio con una dimensione diversa:

```text
batch=6 is the global batch and must be divisible by world_size=4: each rank
trains at batch // world_size, so this value would silently train at a
different global batch than requested. Use batch=4 or batch=8.
```

I gradienti vengono mediati da DDP stesso, quindi la loss viene passata senza
riscalarla. Moltiplicarla anche per il world size gonfierebbe il learning rate
effettivo di un fattore pari all'incirca al numero di GPU.

## Autobatch con DDP

`batch=-1` funziona e restituisce un batch globale divisibile per il world size.

<code-tabs name="autobatch" />

Nel percorso dello spawn la misurazione viene eseguita nel processo padre sul
primo dispositivo, prima che esista un qualsiasi worker, quindi ogni worker riceve
un intero concreto e non serve alcun coordinamento tra processi. Con torchrun, il
rank 0 misura e trasmette il risultato in broadcast come un singolo tensore long.

La misurazione valuta la capacità di una GPU e la moltiplica per il world size.
Quando `nbs` è impostato, il batch globale viene limitato a `nbs` e arrotondato per
difetto a un multiplo del world size, così aggiungere GPU riduce il numero di passi
di accumulo invece di ridurre il batch per GPU. Il funzionamento della misurazione
in sé è descritto in [Iperparametri](/docs/train/hyperparameters).

## SyncBatchNorm

Con DDP i layer BatchNorm di ogni rank vedono solo la propria porzione. Con
`batch // world_size` quella porzione può essere abbastanza piccola da far
degradare il modello a convergenza rispetto a un'esecuzione su una sola GPU, a
causa delle statistiche accumulate.

`sync_bn=True` converte ogni BatchNorm in SyncBatchNorm, così le statistiche
vengono calcolate sull'intero batch globale. La conversione avviene solo quando la
modalità distribuita è attiva, quindi un'esecuzione su una sola GPU non è
influenzata dal flag in nessun caso.

È già attivo per default nelle famiglie convoluzionali ricche di BatchNorm: YOLOX,
YOLOv7, YOLOv9 e le sue varianti, YOLO-NAS, PicoDet, RTMDet e FOMO. Tutte le altre
famiglie lo lasciano disattivato per default. Quando un modello contiene
BatchNorm, `sync_bn` è disattivato e il batch per rank è inferiore a 16, il
trainer avvisa.

<code-tabs name="syncbn" />

Non esiste un flag CLI per `sync_bn`. È un argomento Python.

## Avviare con torchrun

Anche torchrun funziona, ed è la scelta giusta quando l'avvio dei processi è già
gestito dallo scheduler di un cluster. Scrivi lo script per un solo dispositivo e
lascia che sia torchrun a impostare l'ambiente dei rank.

<code-tabs name="torchrun" />

Non combinare le due cose. Con l'ambiente torchrun presente, `device="0,1"` non
avvia processi; il trainer prende `cuda:LOCAL_RANK` e il numero di processi è
gestito da torchrun.

## Comportamento dei rank

Il rank 0 gestisce tutti gli effetti collaterali. Risolve la directory
dell'esecuzione e ne trasmette in broadcast il nome risolto perché tutti i rank
concordino, scrive i checkpoint e gli artefatti, e attiva le callback dell'utente
e i logger. Gli altri rank addestrano e contribuiscono con i gradienti.

Ogni rank inizializza in modo diverso il seed del proprio dataloader e dell'RNG
della data augmentation, derivandolo dal `seed` configurato, così i rank non
estraggono le stesse trasformazioni.

## Piattaforma e backend

Il backend viene scelto automaticamente: NCCL quando CUDA e NCCL sono entrambi
disponibili, Gloo altrimenti. NCCL non viene compilato su Windows, quindi le
esecuzioni su Windows usano Gloo senza alcuna configurazione. Il gruppo di
processi viene inizializzato con un timeout di tre ore.

## Cosa non funziona con DDP

- La cattura dei grafi CUDA. `cuda_graph=True` registra una riga nel log e
  addestra in modalità eager. Vedi
  [Prestazioni dell'addestramento](/docs/train/performance).
- Il profiler di addestramento. `profile=True` viene ignorato con un avviso.

Non tutte le famiglie supportano lo spawn automatico. Ventiquattro lo fanno, e
coprono le famiglie di rilevamento, classificazione, semantiche e di restauro che
si addestrano. Una famiglia che non lo supporta, se riceve un dispositivo
multi-GPU, solleva un errore che nomina l'API del modello e il comando torchrun
invece di addestrare in silenzio su una sola GPU.

## Correlati

- [Iperparametri](/docs/train/hyperparameters) per `batch`, `nbs` e la ripresa
  dell'addestramento.
- [Logger degli esperimenti](/docs/train/loggers) per il vincolo di picklability
  sulle callback.
- [GPU cloud](/docs/train/cloud-gpus) per noleggiare una macchina multi-GPU.
