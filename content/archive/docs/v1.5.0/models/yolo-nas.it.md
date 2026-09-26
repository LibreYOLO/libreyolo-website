---
title: YOLO-NAS
families:
  - yolonas
seo_title: 'YOLO-NAS: predizioni, addestramento ed esportazione in LibreYOLO'
description: >-
  Usa YOLO-NAS in LibreYOLO per il rilevamento e la stima della posa. I pesi di
  Deci.AI sono proprietari e non commerciali, e LibreYOLO non ne pubblica
  nessuno.
lead: >-
  Un detector convoluzionale il cui backbone e il cui neck sono usciti dalla
  ricerca architetturale di Deci.AI, costruito con blocchi RepVGG pensati per la
  quantizzazione. I suoi pesi sono di Deci.AI, con licenza solo per uso non
  commerciale, e LibreYOLO non ne pubblica nessuno.
keywords:
  - YOLO-NAS
  - YOLONAS
  - Deci AI
  - SuperGradients
  - object detection
  - pose estimation
  - rilevamento oggetti python
  - stima della posa python
  - licenza YOLO-NAS uso commerciale
  - AutoNAC
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Un nome non ancora presente su disco viene scaricato dalla CDN di
        Deci.

        # Il download stampa prima i termini di licenza di Deci; prendere il
        file

        # significa accettarli.

        model = LibreYOLO("LibreYOLONASs.pt")

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLONASs.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Posa
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Il suffisso -pose seleziona la testa per la posa e il suo set di pesi.
        model = LibreYOLO("LibreYOLONASs-pose.pt")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLONASs.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: Da zero
      language: python
      code: |
        from libreyolo import LibreYOLONAS

        # Nessun checkpoint di Deci viene toccato: il modello parte da pesi
        # casuali, quindi ciò che esce dalla run deriva solo dai tuoi dati.
        model = LibreYOLONAS(None, size="s")
        model.train(data="my-dataset.yaml", imgsz=640, batch=16)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLONASs.pt data=my-dataset.yaml
    - label: Su COCO
      language: bash
      code: >
        # Il file yaml di COCO incluso porta con sé uno script di download,
        quindi

        # serve un permesso esplicito a meno che il dataset non sia già in
        locale.

        libreyolo val model=LibreYOLONASl.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLONASs.pt format=onnx imgsz=640
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory sceglie in base al suffisso del file, quindi un artefatto
        # esportato si carica come qualsiasi checkpoint e restituisce lo stesso
        # oggetto Results.
        model = LibreYOLO("LibreYOLONASs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 47c30d6e44024ce7
---

## Installazione

YOLO-NAS non richiede nulla oltre al pacchetto base.

```bash
pip install libreyolo
```

## Predizione

Il nome di un checkpoint non ancora presente su disco viene scaricato dalla CDN
pubblica di Deci, non dall'org LibreYOLO, che non ospita nessuno di questi pesi.
Prima che il trasferimento inizi, la libreria stampa i termini di licenza di
Deci una volta per processo, e prima che il file scaricato venga aperto il suo
SHA-256 viene confrontato con un valore fissato. Cosa permettono quei termini è
spiegato in [licenza](#licensing).

<code-tabs name="predict" />

L'oggetto `Results` restituito è lo stesso che restituisce ogni famiglia, quindi
passare a un detector diverso è una modifica di una riga. `conf` imposta la
soglia di confidenza e `iou` la soglia della NMS. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Varianti

Rilevamento e posa sono la stessa architettura sotto teste diverse, e prendono
gli stessi argomenti. Le dimensioni nella tabella qui sotto sono quelle del
rilevamento; la posa è pubblicata in quelle e in una dimensione più piccola. La
testa per la posa predice il set di keypoint di COCO.

<benchmark-table task="detect" />

<va-embed />

## Addestramento

<code-tabs name="train" />

`epochs`, `lr0` e `amp` vengono risolti in base al task quando li ometti, quindi
una run di posa parte da valori predefiniti diversi rispetto a una run di
rilevamento. L'optimizer predefinito è AdamW. Il numero di classi viene dallo
YAML del dataset e la testa viene ricostruita per adattarvisi prima della prima
epoca; sulla testa per la posa il numero di keypoint è gestito allo stesso modo,
quindi su un checkpoint di posa COCO si può fare fine-tuning verso uno scheletro
di dimensione diversa.

Il fine-tuning parte dai pesi di Deci, ed è questo che la licenza di Deci copre.
Addestrare da un modello inizializzato a caso non coinvolge nessun checkpoint di
Deci, ed è il terzo snippet qui sopra.

Vedi [addestramento](/docs/train) per dataset, augmentation, multi-GPU e logger.

## Validazione

`val()` restituisce un dizionario di chiavi `metrics/` che coprono precisione,
recall, mAP 50 e mAP 50-95, misurate su qualsiasi dataset nel formato su cui hai
addestrato.

<code-tabs name="val" />

## Esportazione

<export-matrix />

Un artefatto esportato si ricarica con `LibreYOLO()` in base al suffisso del
file, quindi un file `.onnx` o `.engine` si comporta come un checkpoint e
restituisce gli stessi `Results`. Eseguire il grafo in un runtime nudo, senza
LibreYOLO installato, è altrettanto supportato, ma in quel caso preprocessing e
postprocessing li scrivi tu. Ogni formato installa un extra diverso e prende
qualche argomento suo. Entrambe le cose sono sulla pagina di quel formato.

Un'esportazione è un'altra copia degli stessi pesi in un contenitore diverso.
Esportare un checkpoint di Deci non cambia né da dove vengono i pesi né la
licenza che li copre.

<code-tabs name="export" />

## Checkpoint

Non ce ne sono da elencare. La licenza di Deci vieta la ridistribuzione, quindi
l'org LibreYOLO non pubblica nessun peso di YOLO-NAS e il download si risolve
altrove: un nome della forma `LibreYOLONAS<size>.pt`, o
`LibreYOLONAS<size>-pose.pt` per la posa, corrisponde all'oggetto omologo sulla
CDN pubblica di Deci.

Solo i checkpoint di cui la libreria fissa lo SHA-256 possono essere scaricati
così. Tutto il resto fallisce invece di aprire un pickle di terze parti non
verificato, e va scaricato a mano e passato come percorso. Un file già su disco
si carica dal suo percorso, senza download e senza controllo del checksum.
Questo vale anche per un `.pth` di Deci con il suo nome originale, che il loader
riconosce.

## Licenza

<provenance-box>

LibreYOLO non ospita né fa da mirror per questi pesi: per questa famiglia non
esiste nulla nell'org Hugging Face di LibreYOLO. Ogni download automatico va
invece alla CDN pubblica di Deci, stampa i termini di Deci una volta per
processo prima di iniziare, e viene confrontato con uno SHA-256 fissato prima
che il file venga aperto.

L'alternativa è addestrare da un modello inizializzato a caso. L'architettura è
Apache-2.0 a monte e MIT qui, quindi un modello addestrato così sui tuoi dati non
deriva da nessun checkpoint di Deci.

</provenance-box>

## Citazione

YOLO-NAS è stato rilasciato senza un paper. La voce qui sotto è quella che
chiedono i suoi autori, e copre SuperGradients, la libreria in cui è arrivato.

<citation-block />
