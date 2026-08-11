---
title: Addestrare su una GPU noleggiata
seo_title: Addestrare LibreYOLO su una GPU cloud noleggiata
description: >-
  Esegui un addestramento LibreYOLO su una GPU noleggiata o serverless: prepara
  i dati, installa, avvia, seguilo in diretta, recupera i pesi e smetti di
  pagare.
lead: >-
  Una GPU noleggiata trasforma un addestramento in un job con un inizio, una
  fine e una fattura. Il lavoro è lo stesso di quando addestri in locale; quello
  che cambia è portare dentro i dati, guardare da fuori, tirare fuori i pesi e
  spegnere la macchina.
keywords:
  - addestrare yolo su gpu cloud
  - noleggiare una gpu
  - vast.ai training
  - modal serverless gpu
  - beam gpu
  - training remoto
  - caricare dataset su hugging face
  - costo gpu per epoca
last_verified: 1.5.0
snippets:
  install:
    - label: Sulla macchina
      language: bash
      code: >
        pip install libreyolo


        # Aggiungi solo gli extra che servono all'esecuzione. rfdetr per
        addestrare

        # RF-DETR, lora per il fine-tuning efficiente, onnx per esportare dopo.

        pip install "libreyolo[rfdetr,lora]"
    - label: Controlla la GPU prima di ogni altra cosa
      language: python
      code: |
        import torch

        print(torch.__version__, torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))

        # Una wheel compilata per un'altra architettura riporta True e poi
        # fallisce al primo kernel reale, quindi lanciane uno.
        x = torch.rand(2000, 2000, device="cuda")
        print(float((x @ x).sum()))
  stage:
    - label: 'Impacchetta e carica una volta sola, dalla tua macchina'
      language: bash
      code: >
        tar cf my-dataset.tar my-dataset/

        huggingface-cli upload my-org/my-dataset my-dataset.tar --repo-type
        dataset
    - label: Prepara i dati sulla macchina
      language: python
      code: |
        import tarfile

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            "my-org/my-dataset", "my-dataset.tar", repo_type="dataset"
        )
        with tarfile.open(path) as archive:
            archive.extractall("/root/data")
  launch:
    - label: 'In background, così il job sopravvive a una disconnessione'
      language: bash
      code: |
        nohup libreyolo train \
          model=LibreYOLO9s.pt \
          data=/root/data/my-dataset/data.yaml \
          epochs=100 batch=-1 imgsz=640 \
          project=/root/runs name=run1 \
          > /root/train.log 2>&1 &
    - label: 'Multi-GPU, da un file Python'
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="/root/data/my-dataset/data.yaml",
                epochs=100,
                batch=64,          # batch globale su tutte le GPU
                device="0,1,2,3",
                project="/root/runs",
                name="run1",
            )
  watch:
    - label: Una lettura leggera
      language: bash
      code: |
        cat /root/runs/run1/status.json
    - label: Da uno script
      language: python
      code: |
        import json

        with open("/root/runs/run1/status.json") as handle:
            status = json.load(handle)

        print(status["state"], status["current_epoch"], status["eta_seconds"])
        print(status.get("metrics"))
    - label: 'Nel browser, tramite un tunnel SSH'
      language: bash
      code: |
        # Sulla macchina (di default si lega a 127.0.0.1:8420):
        libreyolo monitor /root/runs/run1 --no-browser

        # Dalla tua macchina, poi apri http://localhost:8420 in locale:
        #   ssh -L 8420:localhost:8420 <user>@<host>
  push:
    - label: Carica i pesi in un posto permanente
      language: bash
      code: |
        huggingface-cli upload my-org/my-run \
          /root/runs/run1/weights/best.pt best.pt
source_hash: 75d314de06aca3b6
---

## Prima di noleggiare qualsiasi cosa

Due decisioni costano più dopo di quanto costino adesso.

Per prima cosa porta il dataset su una CDN. Impacchettarlo come un singolo tar in
un repository dataset di Hugging Face funziona allo stesso modo su ogni provider,
viene servito veloce a tutti, e non richiede altro che un `HF_TOKEN`
nell'ambiente del job quando il repository è privato. Caricare un dataset da una
connessione casalinga, o scaricarlo sulla macchina da un'origine lenta, è tempo
GPU fatturato passato ad aspettare.

<code-tabs name="stage" />

Poi dimensiona il disco. I provider che fatturano lo storage lo fatturano sulla
capacità allocata, non su quella usata, e un disco non si può rimpicciolire dopo
la creazione. Somma i dati preparati, i checkpoint e circa il 30 percento di
margine, e fermati lì.

## Installazione sulla macchina

<code-tabs name="install" />

Installa prima PyTorch se l'immagine non porta già una build CUDA compatibile con
la scheda, poi LibreYOLO, così pip non risolve un torch solo CPU per conto suo.
Il secondo snippet non è una cerimonia opzionale: una wheel compilata per
l'architettura GPU sbagliata riporta `torch.cuda.is_available() == True` e poi
fallisce alla prima operazione reale con `CUDA error: no kernel image is
available for execution on the device`. Una sola moltiplicazione di matrici lo
individua subito, invece di scoprirlo dopo un'ora di setup.

Fai puntare `HF_HOME` a uno storage persistente se il provider offre un volume,
così i download di checkpoint e dataset sopravvivono tra le esecuzioni.

## Avvio

Esegui il job in background. Una sessione interattiva che muore insieme alla tua
connessione di rete si porta dietro l'addestramento.

<code-tabs name="launch" />

Vale la pena usare `batch=-1` proprio qui, perché di solito sei su una scheda su
cui non hai mai addestrato prima. Sonda il modello in modalità addestramento con
un passaggio backward reale e sceglie la potenza di due più grande che ci entra,
il che è più rapido che scoprire il limite con un errore di memoria esaurita dopo
venti minuti. Vedi [Iperparametri](/docs/train/hyperparameters).

Su una macchina multi-GPU, `device="0,1,2,3"` avvia da sé un worker per GPU, e
`batch` resta il batch globale su tutte quante. Il guard `__main__` è
obbligatorio, perché ogni worker reimporta lo script. Questo, e il resto del
comportamento distribuito, sta su
[Addestramento multi-GPU](/docs/train/multi-gpu).

## Seguirlo da fuori

Ogni esecuzione scrive `status.json` nella propria directory di run, riscritto
atomicamente a ogni epoca. È la lettura leggera: poche centinaia di byte che
portano lo stato, l'epoca corrente, l'ETA e le metriche più recenti, senza
analizzare un log.

<code-tabs name="watch" />

`metrics.jsonl`, accanto a esso, ha la cronologia completa per epoca, e
`train.log` ha l'output della console. `libreyolo monitor` serve una dashboard
nel browser su tutti e tre usando solo la libreria standard, quindi non richiede
nulla di installato sulla macchina oltre a LibreYOLO stesso. Raggiungila con un
port forward SSH.

Nessuno di questi tocca il processo di addestramento, quindi si agganciano a
un'esecuzione in corso, riaprono una già finita o ispezionano una andata in
crash.

## Tira fuori i pesi prima di smettere di pagare

La macchina è usa e getta. Carica i checkpoint alle tappe intermedie, non solo
alla fine, perché altrimenti un crash, una preemption o il credito esaurito fanno
perdere tutta l'esecuzione.

<code-tabs name="push" />

`weights/best.pt` e `weights/last.pt` vengono scritti a ogni epoca e a ogni
miglioramento. `save_period=N` aggiunge sopra gli snapshot
`weights/epoch_<N>.pt`, che è ciò che rende leggero un upload a metà esecuzione.
`summary.json` e `results.csv`, dove la famiglia li scrive, sono piccoli e vale
la pena prendere anche quelli.

Una callback su `on_train_epoch_end` è il modo pulito per automatizzare
l'upload. Vedi [Logger degli esperimenti](/docs/train/loggers), dove i backend
ospitati ti danno anche le metriche senza toccare affatto la macchina.

## Smettere di pagare

Questa è la parte che costa soldi veri quando va male, e la regola cambia in base
al modello del provider.

Su un marketplace dove noleggi una macchina nuda, la fatturazione va a tempo
reale finché l'istanza non viene distrutta. Una GPU inattiva costa esattamente
come una occupata, quindi uccidere il processo di addestramento da solo non fa
risparmiare niente. Un'istanza fermata continua a far pagare il suo disco.

Su una piattaforma serverless dove il job è una funzione decorata, il container
scala a zero quando la funzione ritorna, quindi è molto meno probabile
dimenticarsi una macchina accesa. Un job bloccato senza timeout continua a
costare, quindi impostane sempre uno.

Fermare invece di distruggere è una leva reale, e una trappola reale. Misurato su
un 8x RTX 4090 noleggiato con un disco da 250 GB il 2026-07-31: in esecuzione
fatturava $3.4828 all'ora, fermata fatturava $0.0694 all'ora per il solo disco, e
distrutta non fatturava niente. È un risparmio del 98 percento tenendo al loro
posto l'ambiente, i dati preparati e i checkpoint.

La tariffa da fermo è un'aritmetica che puoi fare prima di noleggiare:

```text
stopped $/hr = allocated_GB * storage_cost_per_GB_per_month / 730
             = 250 * 0.20 / 730 = $0.0694/hr
```

Confrontala con quanto costa ricostruire tutto: noleggiare di nuovo, scaricare
l'immagine, installare e ripreparare i dati. Su quella stessa macchina una
ricostruzione era circa 15 minuti di setup più 43 GB di traffico in entrata, in
tutto circa $1.00. Contro $0.0694 all'ora, se torni entro circa 14 ore conviene
fermare, mentre con una pausa più lunga conviene distruggere e ricostruire dalla
copia preparata.

Un rischio rende insicuro fermare quando l'hardware è scarso: fermare rilascia le
GPU. Nulla le riserva, quindi il riavvio riesce solo se l'host le ha ancora
libere. Il tuo disco è al sicuro; le tue GPU no.

## Serverless, come funzione

Se preferisci non gestire una macchina, sia Modal che Beam eseguono una funzione
Python decorata su una GPU e scalano a zero quando ritorna. La suite di test
notturna di LibreYOLO gira proprio su Modal, e `tools/ci/modal_nightly.py` nel
repository della libreria è l'esempio funzionante da cui copiare.

```python
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")   # librerie di sistema di OpenCV
    .pip_install("libreyolo[rfdetr]")
)
app = modal.App("libreyolo-train")
cache = modal.Volume.from_name("libreyolo-cache", create_if_missing=True)


@app.function(gpu="A100", timeout=6 * 60 * 60, volumes={"/cache": cache})
def train():
    import os

    os.environ["HF_HOME"] = "/cache/hf"          # pesi in cache tra le esecuzioni

    from libreyolo import LibreYOLO

    model = LibreYOLO("LibreYOLO9s.pt")
    model.train(data="coco8.yaml", epochs=100, project="/cache/runs")
    cache.commit()                                # rende persistente il volume


@app.local_entrypoint()
def main():
    train.remote()
```

Eseguilo con `modal run modal_train.py`. Il filesystem del container è effimero,
quindi tutto ciò che vale la pena conservare va nel volume o viene caricato
fuori. Imposta `timeout=` esplicitamente; è l'unica cosa che sta tra
un'esecuzione bloccata e una fattura senza fine.

Beam ha la stessa forma, con un decoratore `@function`, un `Volume` e
`train.remote()` chiamato da `__main__`.

## Dimensiona in base al costo per job

I $/ora sono il numero sbagliato da ottimizzare. Un modello piccolo lascia una
scheda grande a metà inattiva, quindi una GPU più lenta ed economica spesso costa
meno per epoca. Esegui il profiler per qualche step sulla scheda noleggiata prima
di impegnarti in un'esecuzione lunga: se il verdetto è `dataloader`
oppure `host / launch`, una GPU più veloce non serve a niente e più worker o un
batch più grande servono molto. Vedi
[Prestazioni dell'addestramento](/docs/train/performance).

## Correlati

- [Dataset](/docs/train/datasets) per la struttura che dovrebbe avere l'archivio
  preparato, e il comando doctor che individua i problemi prima che una GPU
  inizi a costare.
- [Addestramento multi-GPU](/docs/train/multi-gpu) per macchine con più schede.
