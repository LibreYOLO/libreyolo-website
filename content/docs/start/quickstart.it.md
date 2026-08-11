---
title: Quickstart
seo_title: Quickstart LibreYOLO
description: >-
  Esegui un rilevatore su un'immagine, fai fine-tuning su un piccolo dataset ed
  esportalo in TorchScript o ONNX, tutto su CPU, in circa dieci righe di Python.
lead: >-
  Il percorso più breve attraverso LibreYOLO: fai una predizione su un'immagine,
  addestra su un piccolo dataset, poi esporta il risultato. Ogni comando di
  questa pagina gira su CPU.
keywords:
  - libreyolo quickstart
  - tutorial libreyolo
  - libreyolo predict
  - addestrare yolo python
  - esportare yolo onnx
  - esempio yolo python
last_verified: 1.5.0
meta:
  - label: Installazione
    value: pip install libreyolo
    mono: true
  - label: Checkpoint
    value: LibreYOLO9t.pt
    mono: true
  - label: Hardware
    value: la CPU basta per tutto quello che c'è in questa pagina
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Scarica il checkpoint al primo utilizzo, poi lo mette in cache in
        weights/.

        model = LibreYOLO("LibreYOLO9t.pt")


        # Una singola immagine restituisce un solo oggetto Results.

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy.tolist())
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=yolo9-t save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Video e stream
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # stream=True produce un Results per frame invece di costruire una
        lista.

        # Sostituisci il percorso con un indice di webcam, un URL RTSP o una
        cartella.

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # coco8 è un dataset di 8 immagini incluso nella libreria. Al primo

        # utilizzo viene scaricato da un URL, quindi non c'è nessuno script da
        eseguire.

        results = model.train(
            data="coco8.yaml",
            epochs=1,
            imgsz=640,
            batch=4,
            device="cpu",
        )


        print(results["save_dir"])

        print(results["best_checkpoint"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=yolo9-t data=coco8.yaml \
          epochs=1 imgsz=640 batch=4 device=cpu
    - label: Validare
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() restituisce un semplice dict, non un oggetto.
        metrics = model.val(data="coco8.yaml", device="cpu")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
  export:
    - label: TorchScript
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # export() restituisce il percorso che ha scritto.
        path = model.export(format="torchscript")
        print(path)

        # La factory instrada in base al suffisso del file, così l'artefatto si
        # ricarica come un checkpoint e restituisce lo stesso oggetto Results.
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: ONNX
      language: bash
      code: |
        pip install "libreyolo[onnx]"
        libreyolo export model=yolo9-t format=onnx imgsz=640
source_hash: c11b6bdbf0b6fdf1
---

## Installazione

```bash
pip install libreyolo
```

È tutto ciò che serve alle sezioni di predizione e addestramento qui sotto.
L'esportazione in ONNX aggiunge un extra; vedi [installazione](/docs/install)
per l'elenco completo.

## Predizione

<code-tabs name="predict" />

`LibreYOLO()` è una factory. Legge il file, capisce a quale famiglia appartengono
i pesi e restituisce il modello di quella famiglia, così passare a un rilevatore
diverso è una modifica di una riga. Passare `LibreYOLO9t.pt` senza directory
cerca `weights/LibreYOLO9t.pt` rispetto alla directory di lavoro e lo scarica lì
quando manca. Vedi [checkpoint e pesi](/docs/weights) per le regole di download
e per come lavorare offline.

`save=True` scrive una copia annotata sotto `runs/detect/`, in una directory
`predict` che si incrementa a ogni esecuzione. Il `Results` restituito porta con
sé `boxes`, e `names` mappa l'indice di una classe sulla sua etichetta. Il
percorso di una singola immagine restituisce un solo `Results`; una directory,
una lista di immagini o `stream=True` restituiscono una lista o un generatore
di `Results`.

## Addestramento

<code-tabs name="train" />

`data` è uno YAML di dataset. `coco8.yaml` è incluso nella libreria, ed è per
questo che lo snippet gira così com'è; un nome non incluso viene letto come un
percorso. I dataset vengono risolti sotto `~/datasets`, oppure sotto
`LIBREYOLO_DATASETS_DIR` quando quella variabile è impostata.

Un'esecuzione scrive in `project/name`, che per default è una directory sotto
`runs/train`, con dentro `weights/best.pt` e `weights/last.pt`. `train()`
restituisce un dizionario che include `save_dir`, `best_checkpoint`,
`last_checkpoint`, le loss per epoca e le metriche di validazione per epoca. Il
checkpoint addestrato si carica con `LibreYOLO()` esattamente come quello
preaddestrato.

Non tutte le famiglie sono addestrabili. Dove una famiglia offre solo
l'inferenza, `train()` solleva `NotImplementedError` e lo dice. [Concetti di
base](/docs/concepts) spiega che cosa significa ciascun livello di supporto.

## Esportazione

<code-tabs name="export" />

TorchScript non richiede nulla oltre all'installazione di base. Gli altri target
hanno ciascuno il proprio extra, e la copertura è per famiglia e per task invece
che uniforme: vedi [esportazione e deploy](/docs/export).

Gli argomenti accettati da ogni formato includono `imgsz` (un int, oppure una
coppia altezza e larghezza), `batch` (default 1), `half`, `int8` con uno YAML
`data` per la calibrazione, `dynamic` (default True), `simplify` (default True),
`opset`, `device` e `output_path`. Quando `output_path` è omesso il file viene
scritto sotto `weights/` con un nome derivato dal checkpoint.

## Dove proseguire

- [Concetti di base](/docs/concepts) per task, famiglie, taglie e nomi dei checkpoint.
- [Checkpoint e pesi](/docs/weights) per il download automatico, l'uso offline e la sicurezza del caricamento.
- [Importare pesi esistenti](/docs/migrate) se hai già un checkpoint da un progetto upstream.
- [Tutti i modelli](/docs/models) per la famiglia adatta al tuo problema.
- [Addestramento](/docs/train), [Predizione](/docs/predict) ed [Esportazione](/docs/export) per i workflow completi.
