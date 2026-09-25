---
title: "RT-DETR senza Ultralytics: v1, v2 e v4 con licenza Apache-2.0"
description: "Usa e fai fine-tuning di RT-DETR, RT-DETRv2 e RT-DETRv4 con pesi Apache-2.0 in una libreria MIT: un'unica API per predict, train, val ed export."
date: 2026-09-24
author: Xuban
tags: [LibreYOLO, rt-detr, rtdetr, rt-detrv4, object-detection, license, tutorial]
faq:
  - q: "RT-DETR è gratuito per uso commerciale?"
    a: "Sì, se usi un'implementazione permissiva. Il codice originale di RT-DETR e RT-DETRv2 (lyuwenyu/RT-DETR) e quello di RT-DETRv4 (RT-DETRs/RT-DETRv4) sono distribuiti con licenza Apache-2.0, e LibreYOLO esegue tutte e tre le versioni con pesi Apache-2.0 in una libreria MIT. L'implementazione di RT-DETR nel pacchetto ultralytics è distribuita con licenza AGPL-3.0, con una Enterprise License a pagamento come alternativa per l'uso con codice chiuso. Queste sono informazioni generali, non consulenza legale."
  - q: "Qual è la licenza di RT-DETRv4?"
    a: "Apache-2.0. Il file LICENSE su github.com/RT-DETRs/RT-DETRv4 indica Apache-2.0. RT-DETRv4 usa un teacher DINOv3 solo durante l'addestramento; i pesi student distribuiti non contengono parametri DINOv3, quindi la licenza DINOv3 di Meta non si applica a questi pesi."
  - q: "Posso fare fine-tuning di RT-DETR senza Ultralytics?"
    a: "Sì. LibreYOLO fa fine-tuning dei checkpoint di rilevamento di RT-DETR, RT-DETRv2 e RT-DETRv4 con model.train(), oppure con libreyolo train da riga di comando, partendo dai pesi COCO pubblicati. I modelli oriented box di RT-DETRv2 supportano solo l'inferenza."
  - q: "Quali pesi RT-DETR include LibreYOLO?"
    a: "RT-DETR in r18, r34, r50, r50m, r101, l e x; RT-DETRv2 in r18, r34, r50, r50m e r101; RT-DETRv4 in s, m, l e x, tutti per il rilevamento COCO a 640 px. RT-DETRv2 include anche i modelli oriented box n, s, m, l e x, addestrati su DOTA v1.0 a 1024 px. Tutti i file sono Apache-2.0."
---

RT-DETR è un transformer per il rilevamento in tempo reale di Baidu. Decodifica un insieme fisso di query invece di una griglia densa, quindi non c'è un passaggio NMS da configurare. Cercandolo, una delle prime implementazioni che trovi è quella inclusa nel pacchetto Python `ultralytics`. Questo solleva una questione di licenza che il modello in sé non ha mai avuto.

## La licenza di RT-DETR dipende dal repository

La licenza riguarda un repository, non un modello. RT-DETR ha diversi repository e le licenze non coincidono:

| Repository | Contenuto | Licenza |
|:---|:---|:---|
| [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | RT-DETR e RT-DETRv2, PyTorch e Paddle | Apache-2.0 |
| [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | RT-DETRv4 | Apache-2.0 |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | RT-DETR nel pacchetto `ultralytics` | AGPL-3.0 o Enterprise License |
| [LibreYOLO](https://github.com/LibreYOLO/libreyolo) | RT-DETR, RT-DETRv2 e RT-DETRv4 | Codice MIT, pesi Apache-2.0 |

L'implementazione di Ultralytics è solida. La [pagina RT-DETR](https://docs.ultralytics.com/models/rtdetr/) elenca i modelli preaddestrati `rtdetr-l.pt` e `rtdetr-x.pt`, con addestramento, validazione, inferenza ed esportazione. È inclusa in un pacchetto con licenza AGPL-3.0, e Ultralytics vende una Enterprise License per i team che non vogliono rendere open source l'intero progetto. Se nessuna delle due opzioni fa al caso tuo, l'architettura è disponibile con licenza Apache-2.0 dagli autori stessi. La [guida alle licenze YOLO](/articles/yolo-licenses-explained) descrive lo stesso schema per la famiglia YOLO, mentre [la guida alle licenze commerciali](/articles/yolo-commercial-license) spiega cosa comporta AGPL-3.0 per un prodotto con codice chiuso.

## Usa RT-DETR con LibreYOLO

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreRTDETRr18.pt")  # downloads from Hugging Face on first use
result = model(SAMPLE_IMAGE, save=True)

for box in result.boxes:
    print(box.cls, box.conf, box.xyxy)
```

La stessa chiamata funziona da riga di comando:

```bash
libreyolo predict model=LibreRTDETRr18.pt source=image.jpg save=True
```

`conf` e `max_det` filtrano una decodifica top-k su query e classi. `iou` viene accettato ma non usato, perché non c'è NMS. L'oggetto `Results` restituito è quello prodotto da ogni modello LibreYOLO, quindi passare a un altro detector richiede una sola riga.

## RT-DETR, RT-DETRv2 e RT-DETRv4 con un unico loader

La versione è inclusa nel nome del file e `LibreYOLO()` sceglie il modello in base al checkpoint, quindi tutte e tre le versioni si caricano allo stesso modo:

* **RT-DETR:** `LibreRTDETR` in r18, r34, r50, r50m, r101 (backbone ResNet) e l, x (HGNetv2).
* **RT-DETRv2:** `LibreRTDETRv2` in r18, r34, r50, r50m e r101. Mantiene l'architettura della versione 1 e cambia il modo in cui il deformable attention campiona.
* **RT-DETRv4:** `LibreRTDETRv4` in s, m, l e x. Riutilizza l'architettura di D-FINE e i suoi pesi derivano dalla distillazione di un teacher DINOv3 in uno student HGNetv2.

Tutti i pesi di rilevamento sono addestrati su COCO e funzionano a 640 px. Come riferimento, i README upstream riportano 46.5 AP per RT-DETR-R18 e 53.0 AP per RT-DETR-L su COCO, e da 49.8 AP per RT-DETRv4-S fino a 57.0 AP per RT-DETRv4-X. La [pagina della documentazione RT-DETR](/docs/models/rt-detr) contiene la tabella dei benchmark LibreYOLO per ogni checkpoint.

RT-DETRv2 supporta anche i box orientati. `LibreRTDETRv2n-obb.pt` fino a `LibreRTDETRv2x-obb.pt` sono i checkpoint ufficiali DOTA v1.0, con 15 classi aeree a 1024 px, convertiti nel formato LibreYOLO:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv2n-obb.pt")
result = model("aerial.png", save=True)
print(result.obb.xywhr)  # (N, 5): cx, cy, w, h, radians
```

La [rilevazione con box orientati](/docs/tasks/oriented-detection) descrive il formato delle etichette e le metriche.

## Fai fine-tuning di RT-DETR sui tuoi dati

L'addestramento parte da un checkpoint pubblicato:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRr18.pt")
model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
```

Per un addestramento reale, imposta `data` sul file YAML del tuo dataset. Ogni versione ha un proprio learning rate predefinito, e le versioni 1 e 2 impostano il learning rate del backbone a un ventesimo di `lr0`, seguendo la ricetta originale. Per usare più GPU, imposta `device=0,1` nella CLI; per il fine-tuning con adapter [LoRA](/docs/train/lora), imposta `lora=True` e installa l'extra `lora`. I modelli oriented box di RT-DETRv2 supportano solo l'inferenza. Per dataset, augmentation e logger, consulta la [guida all'addestramento](/docs/train).

## Valida ed esporta

```python
metrics = model.val(data="coco128.yaml")
print(metrics["metrics/mAP50-95"])

path = model.export(format="onnx")  # needs: pip install "libreyolo[onnx]"
```

`val()` restituisce un semplice dizionario. I formati di esportazione includono ONNX, TorchScript, ExecuTorch, TensorRT e OpenVINO; la matrice di esportazione nella [pagina del modello](/docs/models/rt-detr) mostra quali sono convalidati per ogni task. Un file `.onnx` o `.engine` esportato si carica di nuovo con `LibreYOLO()` e restituisce lo stesso `Results`. Consulta la [guida all'esportazione](/docs/export) per le istruzioni specifiche di ciascun formato.

## Quando scegliere i repository originali

Se devi riprodurre i risultati di un paper usando esattamente le configurazioni degli autori, vuoi la versione Paddle oppure vuoi eseguire tu la distillazione del teacher DINOv3 di RT-DETRv4, usa i repository upstream. LibreYOLO fa fine-tuning dello student v4 distribuito, ma non esegue il teacher. Inoltre, se il tuo progetto è già con licenza AGPL-3.0 o coperto da una Ultralytics Enterprise License, non c'è motivo legato alla licenza per passare a un altro framework.

## Una nota sulla licenza

Il codice di LibreYOLO è MIT. Ogni file dei pesi RT-DETR è Apache-2.0 e ogni repository dei pesi su Hugging Face include i file upstream LICENSE e NOTICE, che Apache-2.0 richiede di conservare quando ridistribuisci i pesi. I pesi che addestri sui tuoi dati sono tuoi. RT-DETRv4 usa DINOv3 solo come teacher durante l'addestramento, e gli student distribuiti non contengono parametri DINOv3. Queste sono informazioni generali, non consulenza legale. Prima della distribuzione, controlla i file LICENSE aggiornati.

## Provalo

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv4s.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO è distribuito con licenza MIT, funziona su Linux, Mac e Windows e gira su GPU, Apple Silicon e CPU standard senza modifiche al codice.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentazione RT-DETR](/docs/models/rt-detr)
