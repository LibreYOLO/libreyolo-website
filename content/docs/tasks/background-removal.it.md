---
title: Rimozione dello sfondo
seo_title: Rimozione dello sfondo in LibreYOLO
description: >-
  Ritaglia un soggetto separandolo dal suo sfondo in LibreYOLO. Predici un matte
  alfa morbido, scrivi un PNG trasparente e valida con MAE e S-measure.
lead: >-
  La rimozione dello sfondo separa un soggetto da tutto quello che gli sta
  dietro. LibreYOLO la espone come il task matte, che restituisce un valore alfa
  morbido per pixel invece di una maschera binaria di primo piano.
keywords:
  - rimuovere sfondo immagine python
  - alpha matting python
  - background removal python
  - ritaglio png trasparente
  - segmentazione dicotomica immagini
last_verified: 1.5.0
snippets:
  predict:
    - label: Predire un matte
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)   # (H, W) float32 in [0, 1]
    - label: Scrivere un PNG trasparente
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # save() compone l'immagine di partenza con il matte come canale alfa.
        result.save("subject.png")

        rgba = result.cutout()   # lo stesso array (H, W, 4) uint8 in memoria
        print(rgba.shape)
    - label: Comporre su un nuovo sfondo
      language: python
      code: >
        import numpy as np

        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        result = model(SAMPLE_IMAGE)


        rgba = result.cutout()

        alpha = rgba[..., 3:4].astype(np.float32) / 255.0

        backdrop = np.full_like(rgba[..., :3], 255)          # bianco

        composited = (rgba[..., :3] * alpha + backdrop * (1 -
        alpha)).astype(np.uint8)

        print(composited.shape)
  val:
    - label: Validare e leggere le chiavi delle metriche
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # Una directory con images/ e una directory di matte funziona al posto
        # di un YAML di dataset.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])        # più basso è meglio
        print(metrics["metrics/Smeasure"])   # fitness, più alto è meglio
  export:
    - label: Esportare
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="torchscript")
    - label: Eseguire il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory usa il suffisso del file: un artefatto esportato si carica
        # come qualsiasi checkpoint e restituisce lo stesso oggetto Results.
        model = LibreYOLO("LibreBiRefNetl-matte.torchscript")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
source_hash: f7d88c74d9729268
---

## Definizione

Il task `matte` predice un valore alfa per pixel a partire da una singola
immagine RGB: `1` è primo piano pieno e `0` è sfondo pieno. Il valore è continuo
anziché binario, ed è questo il punto del task. Una maschera netta è a una sola
soglia di distanza, a 0.5, mentre il matte morbido porta con sé anche la
copertura parziale su capelli, pelo e bordi sfocati dal movimento che una
maschera binaria butta via.

Una predizione riempie `result.matte`, un payload `Matte` che contiene un array
float32 `(H, W)` in `[0, 1]` sul canvas dell'immagine originale, raggiungibile
come NumPy tramite `.array`. `result.cutout()` compone l'immagine di partenza
con quell'alfa in un array RGBA uint8 `(H, W, 4)`, e `result.save(path)` scrive
la stessa cosa in un PNG con sfondo trasparente. `result.boxes` resta vuoto,
quindi `conf`, `iou` e `max_det` non hanno effetto.

## Modelli

Due famiglie coprono `matte`, e condividono lo stesso forward path.

[BiRefNet](/docs/models/birefnet) è la rete a riferimento bilaterale attorno a
cui è costruito il task, pubblicata qui come un unico checkpoint di livello
Swin-L.

[FeyNobg](/docs/models/feynobg) è la variante più profonda di Feyn Inc.:
l'architettura di BiRefNet con il terzo stage Swin cresciuto da 18 a 24 blocchi,
poi riaddestrata. Per essa LibreYOLO riusa il forward path, il preprocessing e
l'output a logit singolo di BiRefNet, quindi predizione, validazione e gestione
dei checkpoint si comportano in modo identico; i pesi e l'identità della
famiglia sono di FeyNobg.

Le due hanno licenze dei pesi diverse. Entrambe sono indicate nelle pagine dei
modelli, e la licenza sul repository Hugging Face dello specifico checkpoint è
quella che fa fede.

## Predizione

I pesi vengono scaricati da Hugging Face al primo utilizzo e restano in cache in
locale.

<code-tabs name="predict" />

Entrambe le famiglie lavorano su un canvas nativo fisso di 1024x1024 e riportano
il matte alla dimensione dell'immagine originale. Una risoluzione diversa non è
supportata, perché le tabelle di posizione relativa del backbone Swin sono
legate a quella dimensione, e una discrepanza le interpola male invece di
sollevare un errore. `Results.save()` è definito solo per i risultati matte e ha
bisogno dell'immagine di partenza, che ricarica da `Results.path` a meno che tu
non ne passi una. Vedi [predizione](/docs/predict) per sorgenti, streaming e
gestione dei risultati.

## Formato del dataset

La validazione matte associa a ogni immagine RGB un matte alfa di ground truth a
canale singolo con lo stesso stem, dove 0 è sfondo e 255 è primo piano.

```text
my-matte-dataset/
  images/
    subject.jpg
  mattes/
    subject.png
```

Basta passare quella radice come `data=`: la directory dei matte viene rilevata
automaticamente tra `mattes/`, `matte/`, `gt/`, `masks/`, `mask/` e `alpha/`.
L'alternativa è un YAML di dataset, con `path` più `val_images` e `val_mattes`
che nominano directory relative a esso:

```yaml
path: my-matte-dataset
val_images: images
val_mattes: mattes
nc: 1
names: {0: matte}
```

`nc` e `names` sono segnaposto dello schema; un modello matte restituisce
`Results.matte`, non rilevamenti. I valori del matte vengono letti come alfa in
`[0, 1]` dividendo per 255, e un matte la cui forma differisce dal canvas di
predizione viene ridimensionato bilinearmente per farlo combaciare. Vedi
[formati dei dataset](/docs/reference/dataset-formats) per il contratto completo.

## Addestramento

Nessuna delle due famiglie matte ha un'implementazione dell'addestramento:
`train()` solleva `NotImplementedError` su entrambe, e il supporto matte copre
solo predizione, validazione ed esportazione. Ogni pagina di modello indica il
progetto upstream che distribuisce il codice di addestramento e lo script di
conversione che riporta indietro un checkpoint.

## Validazione

`val()` pilota il `predict` del modello stesso, quindi la validazione usa
esattamente il preprocessing della famiglia, ed entrambe le metriche sono
calcolate sul canvas dell'immagine originale.

<code-tabs name="val" />

`metrics/MAE` è l'errore assoluto medio rispetto all'alfa di ground truth, in
`[0, 1]`, e più basso è meglio. `metrics/Smeasure` è la S-measure di Fan et al.
(ICCV 2017), una similarità strutturale che premia l'aver azzeccato la forma del
soggetto e i suoi buchi, cosa che una media per pixel da sola non coglie; più
alto è meglio. La S-measure è anche `fitness`, il numero che legge la selezione
del checkpoint migliore. Nessuna delle due metriche dipende dalla risoluzione.

## Esportazione

Un modello matte esportato si ricarica tramite `LibreYOLO()` in base al suffisso
del file, quindi l'artefatto si comporta come un checkpoint e restituisce lo
stesso `Results`.

<code-tabs name="export" />

TorchScript è il percorso validato per questo task. La conversione ONNX funziona
ma non ha superato la stessa asticella di parità, e i formati restanti non sono
disponibili. La copertura per formato è nelle pagine
[BiRefNet](/docs/models/birefnet) e [FeyNobg](/docs/models/feynobg) e nella
[matrice completa delle esportazioni](/docs/reference/export-matrix).
