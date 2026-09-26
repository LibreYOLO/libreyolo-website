---
title: Restauro delle immagini
seo_title: Restauro e upscaling delle immagini in LibreYOLO
description: >-
  Riduci il rumore, correggi la sfocatura e ingrandisci le immagini in
  LibreYOLO. Predici un'immagine RGB restaurata, addestra NAFNet su dati
  appaiati e leggi le chiavi PSNR e SSIM.
lead: >-
  Il restauro delle immagini prende un'immagine degradata e ne restituisce una
  pulita. LibreYOLO lo espone come il task restore, che copre la riduzione del
  rumore, la correzione della sfocatura e la super-risoluzione dietro un unico
  contratto di output: entra un'immagine RGB, esce un'immagine RGB.
keywords:
  - restauro immagini python
  - image denoising python
  - image super resolution python
  - ingrandire immagine senza perdere qualità
  - deblurring immagine
  - validazione PSNR SSIM
last_verified: 1.5.0
snippets:
  predict:
    - label: Ingrandire un'immagine
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Il generatore 4x compatto; tile limita il picco di memoria su una
        # sorgente grande.
        model = LibreYOLO("LibreRealESRGANx4t-restore.pt")
        result = model(SAMPLE_IMAGE, tile=512, tile_pad=10)

        result.restored.save("upscaled.png")
        print(result.restored.array.shape)   # 4x l'input su ogni asse
    - label: Ridurre il rumore di un'immagine
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Addestrato sul rumore reale di SIDD; l'output resta alla dimensione
        dell'input.

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        result = model(SAMPLE_IMAGE)


        result.restored.save("denoised.png")

        print(result.restore_scale)   # 1: nessun ingrandimento per questo
        checkpoint
  train:
    - label: Fare fine-tuning di NAFNet su immagini appaiate
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: Registrare la provenienza nel checkpoint
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # degradation e dataset vengono scritti nel checkpoint salvato per
        # tracciare la provenienza; non partecipano all'addestramento.
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
  val:
    - label: Validare e leggere le chiavi delle metriche
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val() restituisce un semplice dict, non un oggetto.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])   # fitness
        print(metrics["metrics/SSIM"])
  export:
    - label: Esportare
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # imgsz viene fissato nel grafo, quindi passa la dimensione che il tuo
        # deployment dà davvero in pasto al modello.
        model.export(format="onnx", imgsz=256)
    - label: Eseguire il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory smista in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")
        result = model(SAMPLE_IMAGE)

        result.restored.save("denoised.png")
source_hash: 9dc81cadb3ebf18b
---

## Definizione

Il task `restore` mappa un'immagine su un'altra immagine. Riduzione del rumore,
correzione della sfocatura e super-risoluzione qui sono tutte lo stesso task,
perché condividono un unico contratto: il modello prende in ingresso un'immagine
RGB e restituisce un'immagine RGB, e la degradazione che è stato addestrato ad
annullare è una proprietà del checkpoint, non dell'API.

Una predizione riempie `result.restored`, un payload `RestoredImage` che
contiene un array RGB uint8 `(H, W, 3)`. `.array` lo restituisce come NumPy e
`.save(path)` lo scrive su disco. `result.restore_scale` registra il fattore di
ingrandimento che porta con sé la tela di output, ed è `1` per un checkpoint che
preserva la risoluzione. `result.boxes` resta vuoto, quindi `conf`, `iou` e
`max_det` sono accettati per parità di firma ma non hanno effetto, e `save=True`
scrive direttamente l'immagine restaurata invece di una foto annotata.

## Modelli

Tre famiglie coprono il task `restore`, divise per la degradazione che annullano.

[NAFNet](/docs/models/nafnet) è il denoiser, e l'unica famiglia di restore che
LibreYOLO può addestrare. La sua architettura sostituisce le attivazioni non
lineari di un blocco UNet con una moltiplicazione elemento per elemento, e il
checkpoint pubblicato è addestrato sul rumore reale di SIDD. L'output resta alla
risoluzione dell'input.

[Real-ESRGAN](/docs/models/real-esrgan) è l'upscaler pratico: tre checkpoint
addestrati su degradazioni sintetiche e non solo sul downscaling
bicubico, a 4x, 2x e un generatore 4x più piccolo e veloce, pensato per una
latenza più bassa.

[SwinIR](/docs/models/swinir) ingrandisce di 4x con un backbone Swin
Transformer, in tre dimensioni che coprono il generatore lightweight ufficiale e
due generatori per immagini reali.

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano in cache in
locale.

<code-tabs name="predict" />

Il restauro gira alla risoluzione propria dell'immagine di partenza invece che
su una tela di rete fissa, aggiungendo padding solo fino al fattore di
downsample della rete, quindi sia il tempo sia la memoria crescono con il numero
di pixel del tuo input. `tile` divide il forward pass in tasselli sovrapposti e
ne sfuma le giunzioni, e `tile_pad` è l'alone aggiunto attorno a ogni tassello
prima che venga ritagliato via; entrambi sono argomenti keyword di Python. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Formato del dataset

Il restauro accoppia ogni immagine di input degradata con un'immagine target
pulita esattamente alla stessa risoluzione, abbinata tramite lo stem del nome
del file.

```text
dataset/
  data.yaml
  inputs/
    train/photo.jpg
    val/photo.jpg
  targets/
    train/photo.jpg
    val/photo.jpg
```

```yaml
path: dataset
train: inputs/train
val: inputs/val
input_dir: inputs
target_dir: targets
degradation: denoise
dataset: MyDataset
nc: 1
names: {0: image}
```

`nc` e `names` sono segnaposto dello schema; un modello restore restituisce
`Results.restored`, non rilevamenti. `degradation` e `dataset` sono etichette di
provenienza opzionali. `target_stem_suffix` copre i dataset che chiamano
l'immagine pulita in modo diverso dalla sua coppia degradata. La validazione
mantiene la risoluzione nativa e aggiunge padding solo quanto basta per impilare
un batch, quindi le metriche sono calcolate sulla tela originale. Vedi
[formati dei dataset](/docs/reference/dataset-formats) per il contratto
completo.

## Addestramento

NAFNet è l'unica famiglia di restore con un'implementazione dell'addestramento.
`Real-ESRGAN.train()` e `SwinIR.train()` sollevano entrambi
`NotImplementedError`: quei checkpoint nascono da un addestramento GAN su
pipeline di degradazione sintetiche, e il trainer di restore su dati appaiati
girerebbe senza riprodurre quella ricetta.

<code-tabs name="train" />

Il trainer prende ritagli accoppiati della coppia input e target, così i due
lati restano allineati. Vedi [addestramento](/docs/train) per dataset, multi-GPU
e logger, e la [pagina di NAFNet](/docs/models/nafnet) per i valori predefiniti
di questa famiglia e il pooling da inferenza che disattiva durante
l'addestramento.

## Validazione

`val()` confronta l'output restaurato con il target pulito, in RGB, sulla tela
originale, senza ritaglio dei bordi e senza ridimensionamento.

<code-tabs name="val" />

`metrics/PSNR` è il rapporto segnale-rumore di picco in decibel, ed è anche
`fitness`, il valore letto dalla selezione del checkpoint migliore.
`metrics/SSIM` è la similarità strutturale in `[0, 1]`, calcolata con una
finestra gaussiana 11x11 a sigma 1.5 e mediata sui tre canali di colore. In
entrambi i casi, più alto è meglio.

## Esportazione

Un modello restore esportato si ricarica tramite `LibreYOLO()` in base al
suffisso del file, quindi un file `.onnx` o `.engine` si comporta come un
checkpoint e restituisce gli stessi `Results`, con `restored` che porta
l'immagine di output.

<code-tabs name="export" />

L'esportazione di restore fissa la risoluzione spaziale nel grafo, quindi passa
l'`imgsz` che il tuo deployment darà davvero in pasto al modello. Per NAFNet
quella dimensione deve essere divisibile per il fattore di downsample della
rete, e sotto `dynamic=True` resta dinamica solo la dimensione del batch. Per
Real-ESRGAN e SwinIR, omettere `imgsz` fa ricadere su una piccola dimensione di
patch interna invece che sulla tua risoluzione di lavoro. La copertura per
formato è su ogni pagina del modello e nella
[matrice completa delle esportazioni](/docs/reference/export-matrix).
[Esportazione](/docs/export) elenca gli argomenti che ogni formato accetta.
