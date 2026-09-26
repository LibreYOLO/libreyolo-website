---
title: DEIM
families:
  - deim
seo_title: DEIM e DEIMv2 in LibreYOLO
description: >-
  Usa DEIM e DEIMv2 in LibreYOLO per il rilevamento di oggetti. Installa, fai
  predizioni, addestra, valida ed esporta, da mezzo milione di parametri in su.
lead: >-
  Un detection transformer addestrato con matching denso uno a uno, che converge
  in molte meno epoche rispetto alle ricette DETR su cui si basa. LibreYOLO ne
  porta due versioni, distinte dal checkpoint che carichi.
keywords:
  - DEIM
  - DEIMv2
  - DINOv3
  - detection transformer
  - DETR
  - object detection
  - real-time object detection
  - rilevamento oggetti python
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDEIMn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDEIMn.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Video
      language: python
      code: |
        from libreyolo import LibreYOLO

        # La versione fa parte del nome del file, e la factory smista in base
        # al checkpoint, quindi entrambe si caricano allo stesso modo.
        model = LibreYOLO("LibreDEIMv2pico.pt")

        # Qualsiasi sorgente accettata dalla libreria: file, cartella, URL,
        # indice della webcam, stream RTSP o una lista .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # coco128.yaml scarica un campione di 128 immagini al primo utilizzo.
        # Punta `data` al YAML del tuo dataset per un'esecuzione reale.
        model.train(data="coco128.yaml", epochs=50, batch=8, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 batch=8 lr0=1e-4
    - label: DEIMv2
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Se non vengono indicati, epochs, batch, imgsz e lr0 arrivano dalla
        # ricetta pubblicata per la dimensione caricata.
        model = LibreYOLO("LibreDEIMv2pico.pt")
        model.train(data="coco128.yaml", epochs=50)
    - label: LoRA
      language: python
      code: |
        # Serve l'extra lora: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # val() restituisce un dict semplice, non un oggetto
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDEIMn.pt data=coco128.yaml
    - label: Contro COCO
      language: bash
      code: |
        # coco-val-only.yaml scarica le 5000 immagini di val2017 e salta il
        # set di addestramento. Porta con sé uno script di download
        # incorporato, quindi serve un permesso esplicito a meno che il
        # dataset non sia già in locale.
        libreyolo val model=LibreDEIMn.pt data=coco-val-only.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        # Serve l'extra onnx: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDEIMn.pt format=onnx
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreDEIMn.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 6edaac5f05abaabe
---

## Installazione

Nessuna delle due versioni richiede un extra opzionale. Tutto ciò che importano
è già nell'installazione di base.

```bash
pip install libreyolo
```

Il fine-tuning con adattatori tramite `lora=True` è l'eccezione, e richiede
l'extra `lora`.

```bash
pip install "libreyolo[lora]"
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano nella cache
locale.

<code-tabs name="predict" />

L'oggetto `Results` restituito è quello che restituisce ogni famiglia, quindi
passare a un rilevatore diverso è una modifica di una riga. `conf` e `max_det`
filtrano un decode top-k su query e classi; non c'è nessun passaggio di NMS da
regolare, e `iou` è accettato ma non usato. Vedi [predizione](/docs/predict) per
sorgenti, streaming e gestione dei risultati.

## Varianti

La versione 1 porta cinque dimensioni, tutte alla stessa dimensione di input. La
versione 2 conserva quei cinque nomi e ne aggiunge tre più piccole, `atto`,
`femto` e `pico`, le prime due native a una dimensione di input più bassa
rispetto alle altre. Cinque codici di dimensione esistono quindi in entrambe le
versioni e indicano modelli diversi; la versione è scritta nel nome del file del
checkpoint.

<benchmark-table task="detect" />

<va-embed />

La versione 1 mantiene l'architettura di D-FINE e sostituisce il suo obiettivo
di classificazione con la loss sensibile alla matchability della ricetta densa
uno a uno, quindi le due famiglie condividono quasi tutte le chiavi dello state
dict e si distinguono per i metadati nel checkpoint. La versione 2 mantiene quel
contratto di addestramento e mescola i backbone: HGNetv2 sotto `s`, e un vision
transformer DINOv3 con un adattatore di tuning spaziale da `s` in su. È quel
backbone a mettere una seconda licenza su quei quattro checkpoint, quindi leggi
[licenze](#licensing) prima di metterne uno in produzione.

## Addestramento

L'addestramento parte da un checkpoint pubblicato. `pretrained` non arriva mai
al trainer: la versione 1 avvisa che la chiave è sconosciuta e la ignora, la
versione 2 la rimuove. Nessuna delle due ti dà un modello inizializzato a caso.

<code-tabs name="train" />

Sulla versione 1 passa `lr0` tu. La firma Python di `train()` usa `4e-4` come
valore predefinito, il learning rate della ricetta COCO pubblicata, mentre la
configurazione di addestramento della famiglia porta `1e-4` come valore
predefinito per il fine-tuning, ed è quel valore più basso che la CLI risolve
quando l'argomento manca. La configurazione registra la misurazione che c'è
dietro: alle dimensioni di batch che un fine-tuning usa davvero, su dataset
piccoli, il learning rate di COCO peggiorava il trasferimento in modo
misurabile.

La versione 2 risolve quei valori predefiniti da sola. Se lasci `epochs`,
`batch`, `imgsz` e `lr0` non impostati, legge ciascuno di essi dalla ricetta
pubblicata per la dimensione caricata, così le dimensioni piccole si addestrano
alla loro risoluzione di input senza doverglielo dire, e un valore che passi tu
ha la precedenza sulla ricetta. `imgsz` è l'argomento che vincola: deve essere
un multiplo positivo di 32, altrimenti la versione 2 solleva un errore prima che
l'esecuzione parta.

Vedi [addestramento](/docs/train) per dataset, data augmentation, multi-GPU e
logger.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurate su qualsiasi dataset nel formato con cui
hai addestrato.

<code-tabs name="val" />

Le righe della tabella di benchmark qui sopra vengono dall'harness di benchmark
di LibreYOLO; la nota sotto quella tabella indica quale dataset le ha prodotte e
rimanda ai record delle esecuzioni.

## Esportazione

<export-matrix />

La matrice copre entrambe le versioni in una sola pagina: dove non concordano su
un formato, la cella mostra la più debole delle due, così niente qui è
sopravvalutato per la versione che carichi.

Un artefatto esportato si ricarica con `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box>
Le quattro dimensioni di DEIMv2 da S in su prendono il backbone da DINOv3,
quindi i loro repository di pesi portano sia Apache-2.0 sia la DINOv3 License di
Meta, e LibreYOLO distribuisce il codice sorgente del backbone DINOv3 sotto lo
stesso accordo. Il resto di questa famiglia, comprese tutte le dimensioni di
DEIMv2 sotto S, è solo Apache-2.0.
</provenance-box>

## Citazione

<citation-block />

DEIMv2 è un articolo a parte e ha il suo blocco di citazione su
[github.com/Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2#5-citation);
cita quello se hai usato un checkpoint della versione 2.
