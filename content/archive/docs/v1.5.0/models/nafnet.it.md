---
title: NAFNet
families:
  - nafnet
seo_title: 'NAFNet: eliminare il rumore, addestrare ed esportare con licenza MIT'
description: >-
  Usa NAFNet in LibreYOLO per il denoising e il restauro delle immagini.
  Installa, fai predizioni, addestra, valida ed esporta il checkpoint SIDD, con
  licenza MIT.
lead: >-
  NAFNet è una rete convoluzionale per il restauro di immagini che elimina le
  funzioni di attivazione non lineari dal tipico blocco UNet, sostituendole con
  una moltiplicazione elemento per elemento. LibreYOLO lo supporta per un solo
  task, il restauro, con un checkpoint pubblicato di denoising su immagini reali
  addestrato su SIDD.
keywords:
  - NAFNet
  - image restoration
  - image denoising
  - denoising immagini python
  - rimuovere il rumore da una foto
  - deblurring immagini
  - SIDD
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model("noisy.jpg", save=True)

        restored = result.restored
        print(restored.array.shape)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreNAFNetl-restore-sidd.pt source=noisy.jpg
        save=True
    - label: Salvare l'immagine restaurata
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model.predict("noisy.jpg")

        result.restored.save("denoised.png")
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 imgsz=256 batch=16 lr0=1e-3
    - label: Provenienza del checkpoint
      language: python
      code: |
        from libreyolo import LibreYOLO

        # degradation e dataset vengono registrati sul checkpoint salvato; non
        # cambiano ciò che viene addestrato.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
    - label: Multi-GPU
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val() restituisce un semplice dict, non un oggetto
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.export(format="onnx", imgsz=256)
        model.export(format="tensorrt", imgsz=256, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=onnx
        imgsz=256

        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=tensorrt
        imgsz=256 half=True
    - label: Usare il file esportato
      language: python
      code: >
        from libreyolo import LibreYOLO


        # La factory sceglie in base al suffisso del file, quindi un artefatto

        # esportato si carica come un qualsiasi checkpoint e restituisce lo
        stesso Results.

        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")

        result = model("noisy.jpg")


        result.restored.save("denoised.png")
source_hash: 9bae9f82bee741bf
---

## Installazione

NAFNet non richiede nessun extra opzionale. Tutto ciò che importa è già
nell'installazione base.

```bash
pip install libreyolo
```

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano in cache
in locale.

<code-tabs name="predict" />

L'oggetto `Results` restituito porta un solo campo per questa famiglia,
`restored`, un'immagine RGB uint8 densa in formato HWC sul canvas originale;
non ci sono box da scorrere. `save=True` scrive quell'immagine restaurata
direttamente su disco invece di disegnare un'annotazione sopra l'input.
`conf`, `iou` e `max_det` sono accettati per parità di firma con tutte le altre
famiglie ma non hanno effetto, dato che il restauro non produce rilevamenti da
filtrare. Vedi [predizione](/docs/predict) per sorgenti, streaming e gestione
dei risultati.

## Varianti

Due larghezze condividono questa architettura: `s` (larghezza 32) e `l`
(larghezza 64), entrambe costruite attorno a una patch di addestramento da
256 px. La predizione e la validazione girano alla risoluzione nativa
dell'immagine qualunque sia la dimensione, con padding solo fino al fattore di
downsample della rete. Al momento è pubblicata solo la larghezza `l`, come
checkpoint di denoising su immagini reali addestrato su SIDD.

## Addestramento

NAFNet fa fine-tuning sulle tue coppie di immagini degradata/pulita: un YAML di
dataset che punta a una cartella `inputs/<split>/` di immagini degradate e a
una cartella `targets/<split>/` di target puliti, accoppiate per nome del file
senza estensione. `degradation` e `dataset` sono stringhe opzionali registrate
sul checkpoint salvato come provenienza; non intervengono nell'addestramento.

<code-tabs name="train" />

Se non tocchi niente, il trainer esegue 100 epoche con AdamW a `lr0=1e-3`, un
batch di 16, ritagli da 256 px ed early stopping dopo 50 epoche senza
miglioramenti del PSNR. Per questa famiglia non esiste un percorso LoRA:
`lora=True` solleva un errore invece di eseguire, perché `NAFNetTrainer` non
aderisce mai al fine-tuning con adattatori.

Durante l'addestramento la rete gira con un normale global-average pooling. Il
pooling locale a finestre di NAFNet, riservato all'inferenza (Test-time Local
Converter), viene staccato prima della prima epoca e riattaccato quando
l'addestramento finisce, perché retropropagare attraverso un pooling locale a
finestra fissa non corrisponderebbe al modo in cui il checkpoint viene usato in
inferenza.

Vedi [addestramento](/docs/train) per dataset, data augmentation, multi-GPU e
logger.

## Validazione

`val()` restituisce un dizionario con `metrics/PSNR` e `metrics/SSIM`,
calcolati in RGB su tutto il canvas valido: SSIM usa una finestra gaussiana
11x11 con sigma 1.5, e il `fitness` con cui si sceglie il miglior checkpoint è
il valore di PSNR. `data` punta allo stesso formato di dataset a immagini
accoppiate usato per l'addestramento.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica con `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`, con `restored` che porta l'immagine di output.
NAFNet esporta a una risoluzione spaziale fissa: `imgsz` deve essere divisibile
per il fattore di downsample della rete (16 per entrambe le larghezze
dell'architettura), e solo la dimensione di batch è dinamica quando
`dynamic=True`; altezza e larghezza vengono fissate al momento
dell'esportazione.

<code-tabs name="export" />

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box></provenance-box>

## Citazione

<citation-block />
