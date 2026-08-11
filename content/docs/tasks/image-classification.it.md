---
title: Classificazione di immagini
seo_title: Classificazione di immagini in LibreYOLO
description: >-
  Etichetta un'immagine intera in LibreYOLO: le famiglie che servono il task, il
  layout del dataset ImageFolder e le chiamate predict, train, validate ed
  export.
lead: >-
  La classificazione di immagini assegna una distribuzione di etichette
  all'immagine intera e non localizza nulla al suo interno. La chiave del task è
  classify.
keywords:
  - image classification python
  - classificazione di immagini python
  - addestrare un classificatore di immagini
  - ImageFolder dataset
  - accuratezza top-1
  - zero-shot classification
  - libreria di classificazione MIT
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Il suffisso -cls nel nome del file seleziona il task, quindi non
        # serve alcun argomento task.
        model = LibreYOLO("LibreResNet50-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.names[result.probs.top1], float(result.probs.top1conf))
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreResNet50-cls.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: L'intera distribuzione
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreResNet50-cls.pt")(SAMPLE_IMAGE)
        probs = result.probs

        # .data è il vettore completo (C,); top5/top5conf sono viste ordinate.
        print(probs.data.shape)
        for index, score in zip(probs.top5, probs.top5conf):
            print(result.names[index], float(score))
    - label: 'Zero-shot, senza addestramento'
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # CLIP confronta l'immagine con dei prompt testuali, quindi l'insieme

        # di etichette si definisce alla chiamata invece di essere fissato nel
        checkpoint.

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        model.set_classes(["a person jumping", "an empty street", "a parked
        car"])

        result = model(SAMPLE_IMAGE)


        print(model.names[result.probs.top1], float(result.probs.top1conf))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # imagenette160 è un nome di dataset noto e si scarica al primo uso.
        # Per i tuoi dati passa una directory con uno split train/.
        model = LibreYOLO("LibreResNet50-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")

        # val() restituisce un semplice dict, non un oggetto.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreResNet50-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreResNet50-cls.pt format=onnx
    - label: Usare il file esportato
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory instrada in base al suffisso del file, quindi un artefatto

        # esportato si carica come un checkpoint e restituisce lo stesso oggetto
        Results.

        model = LibreYOLO("LibreResNet50-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1, result.probs.top1conf)
source_hash: 836bea76cd2cdf92
---

## Definizione

La classificazione di immagini produce un punteggio per classe sull'immagine
intera e nessuna coordinata. Risponde a che cosa c'è nella foto, mai a dove, ed è
questo che la separa dal [rilevamento di oggetti](/docs/tasks/object-detection).

`classify` è la chiave canonica del task, e il suffisso `-cls` nel nome del file
di un checkpoint la seleziona. Sulle famiglie di classificazione quel suffisso è
obbligatorio anziché opzionale, quindi `LibreResNet50.pt` non viene letto come un
classificatore e solo `LibreResNet50-cls.pt` lo è.

`predict()` riempie `result.probs` e lascia `boxes` vuoto. `.data` è il vettore
completo dei punteggi, `.top1` l'indice del punteggio più alto e `.top1conf` il
suo valore, `.top5` i cinque indici più alti in ordine decrescente e `.top5conf`
i loro punteggi. Gli indici puntano dentro `result.names`. Fare lo slicing di un
oggetto `Results` non tronca mai `probs`, perché il vettore appartiene
all'immagine e non a una singola riga.

## Modelli

Cinque famiglie sia addestrano sia predicono: [ResNet](/docs/models/resnet),
[ConvNeXt](/docs/models/convnext), [MobileNetV4](/docs/models/mobilenetv4),
[EfficientNetV2](/docs/models/efficientnetv2) e
[DINOv2](/docs/models/dinov2). Le prime quattro girano con il pacchetto base e
hanno pesi pubblicati. DINOv2 richiede `pip install "libreyolo[rfdetr]"` e non ha
un checkpoint ospitato da LibreYOLO: carica il backbone originale con una testa
lineare inizializzata a caso, quindi è un punto di partenza per il fine-tuning
più che un predittore pronto all'uso.

Altre cinque predicono, validano ed esportano, ma il loro `train()` solleva
`NotImplementedError`: [ViT](/docs/models/vit), [Swin](/docs/models/swin),
[VGG](/docs/models/vgg), [AlexNet](/docs/models/alexnet) e
[DeiT](/docs/models/deit).

[CLIP](/docs/models/clip) e [SigLIP2](/docs/models/siglip2) classificano senza un
insieme fisso di etichette. Confrontano l'immagine con dei prompt testuali,
quindi `set_classes()` definisce le classi al momento della chiamata e per un
nuovo insieme di etichette non c'è alcun passaggio di addestramento. Entrambe
servono anche il task `embed`.

## Predizione

I pesi si scaricano da Hugging Face al primo uso e restano in cache in locale.

<code-tabs name="predict" />

`conf`, `iou` e `max_det` qui non hanno effetto: non ci sono candidati da
sogliare o sopprimere, solo una distribuzione. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Formato del dataset

La classificazione usa un albero di directory, non file di etichette e non uno
YAML. `data` è la radice del dataset.

```text
dataset/
  train/
    tench/000001.jpg
    parachute/000002.jpg
  val/
    tench/000101.jpg
    parachute/000102.jpg
```

`train/` è obbligatorio per l'addestramento e definisce la corrispondenza tra
classe e indice in base al nome ordinato delle cartelle, quindi la prima cartella
in ordine alfabetico diventa la classe 0. `val/` è obbligatorio per la
validazione. Può essere presente uno split `test/`, e i comandi predefiniti di
addestramento e validazione non lo usano. Ogni split diverso da `train` deve
contenere gli stessi nomi di cartelle di classe dell'insieme di classi atteso, ed
è questo che fa fallire rumorosamente una discrepanza invece di conteggiarla come
una predizione sbagliata. Le estensioni di immagine accettate sono `.jpg`,
`.jpeg`, `.png`, `.bmp`, `.webp`, `.tif` e `.tiff`.

`data` accetta tre cose: un percorso a una directory che contiene uno split
`train/`, un URL `.zip`, oppure uno dei nomi di dataset noti, `imagenette160` e
`smoke10`, che si scaricano e restano in cache al primo uso.

Il loader canonico è `libreyolo.data.classify_dataset`.

## Addestramento

<code-tabs name="train" />

Non c'è nessun `nc` da dichiarare: il numero di classi viene dai nomi delle
cartelle sotto `train/`, e il layer lineare finale viene ricostruito per
corrispondere mentre il backbone si trasferisce invariato. Vedi
[addestramento](/docs/train) per dataset, augmentation, multi-GPU e logger.

## Validazione

`val()` restituisce un semplice dizionario di chiavi `metrics/`, calcolate sullo
split `val/` della radice del dataset.

<code-tabs name="val" />

`metrics/accuracy_top1` è la quota di immagini la cui classe con il punteggio più
alto è quella vera, ed è il numero di riferimento, quello che l'addestramento usa
per scegliere l'epoca migliore. `metrics/accuracy_top5` è la quota in cui la
classe vera compare in un punto qualsiasi delle cinque classi con il punteggio
più alto, e dice tanto meno quante meno classi ha il dataset. Il dizionario porta
anche `fitness`, una copia del valore top-1.

## Esportazione

<code-tabs name="export" />

Un artefatto esportato si ricarica tramite `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce lo stesso `Results`. La copertura dei formati cambia da famiglia a
famiglia; la matrice su ogni pagina di modello è generata dall'insieme validato
invece di essere scritta a mano. Vedi
[esportazione e deployment](/docs/export) per i formati, i loro extra e i loro
vincoli.
