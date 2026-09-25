---
title: "Migliori alternative a Ultralytics nel 2026: librerie YOLO open source"
description: "Una guida pratica alle migliori alternative open source a Ultralytics YOLO nel 2026: licenze, task supportati e limiti di deployment, e il ruolo di LibreYOLO, la libreria YOLO con licenza MIT più completa."
date: 2026-07-01
author: Xuban
tags: [LibreYOLO, ultralytics-alternative, object-detection, yolo, mit-license]
faq:
  - q: "Ultralytics YOLO è gratuito per uso commerciale?"
    a: "Solo con AGPL-3.0, che richiede di rilasciare il codice sorgente dell'applicazione che ci costruisci sopra, compresi i servizi di rete. Per l'uso commerciale a codice chiuso serve la licenza commerciale a pagamento di Ultralytics. Le alternative permissive (MIT, Apache-2.0) evitano questo obbligo."
  - q: "Qual è l'alternativa a Ultralytics con la licenza più semplice?"
    a: "Per uno stack permissivo distribuibile ovunque: LibreYOLO (codice MIT), RF-DETR (Apache-2.0) e YOLOX (Apache-2.0). Nessuno impone il copyleft di AGPL."
  - q: "Cosa sta sostituendo YOLO nel 2026?"
    a: "Sempre più spesso, i detector transformer real-time: RT-DETR, RF-DETR e la linea D-FINE e DEIM. Su hardware adeguato e con una latenza simile, eguagliano o superano l'accuratezza di YOLO. Le CNN in stile YOLO restano vincenti sui piccoli dispositivi edge, dove quei transformer funzionano male e non hanno un percorso maturo per NCNN o CPU."
  - q: "Questi modelli funzionano su Raspberry Pi o su un NPU?"
    a: "Dipende dall'esportazione. I detector CNN come YOLOX e RTMDet si esportano in NCNN e funzionano su un Pi; alcuni, come YOLOX, funzionano anche sul NPU Hailo. I detector transformer come RF-DETR no: richiedono una GPU o una CPU potente."
---

Nota: LibreYOLO è il nostro progetto ed è al primo posto nell'elenco. Gli altri strumenti qui sono tutti usati nel nostro lavoro, e segnaliamo i casi in cui superano LibreYOLO.

La maggior parte dei team abbandona Ultralytics per uno di tre motivi:

- **Licenza.** I modelli YOLO di Ultralytics sono soggetti ad AGPL-3.0 (YOLOv5 dal 2023, e v8 in poi). Lo schema è noto: sviluppi un prodotto, lo distribuisci, poi una revisione legale o un cliente solleva un dubbio sul modello e ti trovi a dover scegliere se rendere open source l'intera applicazione o acquistare una licenza commerciale. Il copyleft di AGPL si estende al tuo codice non appena distribuisci il prodotto o fornisci un servizio. È un motivo comune per cercare un'alternativa.
- **Apertura.** Un unico fornitore controlla i modelli, il codice di addestramento e le condizioni. Alcuni team vogliono pesi e codice con licenze permissive su cui poter costruire senza chiedere il permesso.
- **Il modello.** YOLO non è sempre l'architettura migliore per il task. I detector transformer real-time come RT-DETR, RF-DETR e D-FINE possono essere più accurati a parità di latenza. Il problema è che sono sparsi in repository di ricerca e, come vedrai, LibreYOLO li esegue tutti e tre con un'unica API. Questo è un motivo per cambiare libreria, non per abbandonarla.

Valutiamo ogni strumento qui sotto in base a tre criteri: una licenza che consenta di distribuirlo, la possibilità di installarlo e usarlo con una versione attuale di PyTorch e l'esportazione sull'hardware su cui esegui davvero il deployment.

Leggendo, noterai uno schema ricorrente: ogni alternativa è valida, ma usarla da sola è spesso frustrante. YOLOX si installa a fatica, MMDetection ti intrappola tra le versioni dei wheel, la famiglia RT-DETR è composta da codice di ricerca senza un canale di supporto e Detectron2 è fermo dal 2021. LibreYOLO è al primo posto perché integra già la maggior parte di questi modelli, è mantenuto, usa MIT e uno stack aggiornato, e offre un'unica API familiare. Più che una voce dell'elenco, è il livello che le riunisce.

## In breve

| Strumento | Licenza | Task supportati | Ideale per | Esportazione edge |
| --- | --- | --- | --- | --- |
| **LibreYOLO** | MIT (codice) | Rilevamento, segmentazione, posa, classificazione, profondità, gaze, tracking | L'hub MIT con oltre 20 famiglie di modelli accessibili da un'unica API | ONNX, TensorRT, OpenVINO, NCNN, CoreML, TFLite |
| **RF-DETR** | Apache-2.0 (N/S/M/L) | Rilevamento, segmentazione, posa (anteprima) | Rilevamento e segmentazione in produzione | ONNX, TFLite, TensorRT; niente NCNN o Hailo |
| **Lightly / LightlyTrain** | MIT / AGPL-3.0 | Pretraining, distillazione | Dati senza etichette, distillazione di DINOv2/v3 | n/a (non è un detector) |
| **YOLOX** | Apache-2.0 | Rilevamento | Rilevamento real-time anchor-free | Il progetto originale è difficile da installare, funziona tramite LibreYOLO |
| **MMDetection / fork OneDL** | Apache-2.0 | Tutto (ricerca) | Ampia scelta di architetture, riproduzione di paper | Tramite esportazione |
| **Detectron2** | Apache-2.0 | Rilevamento, segmentazione, keypoint | Mask R-CNN e la famiglia R-CNN | Manuale |
| **D-FINE / DEIM / DEIMv2** | Apache-2.0 | Rilevamento | DETR real-time all'avanguardia | ONNX, TensorRT |
| **EdgeCrafter** | Apache-2.0 | Rilevamento, segmentazione, posa | ViT compatti per edge, distillati da DINOv3 | Livello ricerca |
| **RT-DETR (v1-v4)** | Apache-2.0 | Rilevamento | Rilevamento real-time all'avanguardia | ONNX, TensorRT |

## 1. LibreYOLO

Licenza: MIT (codice). Ideale per la libreria YOLO con licenza MIT più completa e come alternativa drop-in a Ultralytics.

**LibreYOLO è la libreria YOLO con licenza MIT che esegue ogni modello tramite un'unica API.** È il nostro progetto e, dopo aver visto il resto dell'elenco, il suo ruolo dovrebbe essere chiaro: è l'hub che esegue le alternative. LibreYOLO integra già quasi tutti i repository difficili da usare citati sopra, YOLOX, RTMDet, RF-DETR, D-FINE, DEIM e la linea RT-DETR, in un'unica API familiare e mantenuta. Così hai a disposizione i modelli senza dover ricostruire come installarli o affrontare i problemi di licenza.

C'è anche un motivo che va oltre la praticità. YOLO è nato come ricerca aperta: Darknet di Joseph Redmon, dalla v1 alla v3, era gratuito e chiunque poteva usarlo e svilupparci sopra. L'era AGPL lo ha chiuso. LibreYOLO esiste per rendere di nuovo accessibile quel lavoro, come volevano i suoi creatori: con una licenza permissiva, gestito dalla community e senza inviare dati a casa. Ci sono quindi due aspetti.

**Primo: è l'alternativa a Ultralytics YOLO con licenza MIT.** Mantiene l'API che già conosci, così puoi trasferire i tuoi dataset e script in formato YOLO con poche modifiche, ma usa MIT invece di AGPL: nessun copyleft che si estende al tuo codice, nessuna licenza commerciale da acquistare e nessuna telemetria che invia dati a casa, come fa Ultralytics per impostazione predefinita. Se sei qui per la licenza, il punto è tutto qui. YOLO9 e RF-DETR sono le opzioni predefinite, testate a fondo e pronte per la produzione; lo zoo più ampio è più recente ed è chiaramente indicato come sperimentale. Preferiamo dirlo apertamente invece di fingere che ogni modello sia stato collaudato a fondo.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")   # or LibreRFDETRl.pt, LibreDFINEl.pt, ...
results = model("image.jpg", save=True)
```

**Secondo: copre molto più di Ultralytics.** Una singola API include oltre 20 famiglie di modelli: YOLO basati su CNN, DETR transformer, backbone ViT, SAM e altro ancora, non solo la gamma di un singolo fornitore. E non offre solo detector:

- **Task:** segmentazione di istanze e semantica, posa, classificazione e box orientati, oltre a due task che Ultralytics non supporta affatto: la stima della profondità monoculare (Depth Anything V2) e la stima dello sguardo (L2CS).
- **Strumenti pratici:** tracking multi-oggetto con ID persistenti (ByteTrack e OC-SORT), inferenza video con sottocampionamento dei frame e anteprima dal vivo, inferenza a tasselli per immagini da droni e satelliti, un'interfaccia browser drag-and-drop (`libreyolo ui`) e un comando `doctor` che controlla il dataset prima di avviare un addestramento.
- **Addestramento avanzato:** accumulo del gradiente, congelamento dei layer, ripresa dell'addestramento, test-time augmentation, fine-tuning LoRA/DoRA, multi-GPU e logging con TensorBoard/MLflow/Weights & Biases.
- **Esportazione:** sette formati (ONNX, TorchScript, TensorRT, OpenVINO, NCNN, CoreML, TFLite), con quantizzazione INT8/FP16, NMS integrato e metadati del modello incorporati.
- **Funziona ovunque:** accetta percorsi, URL, PIL, NumPy, tensori o byte grezzi, ed esegue su CUDA, Apple Silicon (MPS) o una normale CPU senza modifiche al codice.

Quando il repository di un detector valido è abbandonato o difficile da installare, LibreYOLO ne esegue i pesi su uno stack aggiornato e mantenuto.

## 2. RF-DETR

Licenza: Apache-2.0 (Nano, Small, Medium, Large). Ideale per rilevamento e segmentazione in produzione.

RF-DETR, di Roboflow e presentato a ICLR 2026, è il modello più pronto per la produzione di questo elenco. È progettato bene, viene da un team che porta i prodotti sul mercato e, quando ti serve un modello affidabile per il rilevamento o la segmentazione in un sistema reale, è un'ottima prima scelta. Il pacchetto `rfdetr` e i pesi da Nano a Large usano Apache-2.0. I pesi più grandi, XL e 2XL, usano la licenza PML di Roboflow.

## 3. Lightly

Ideale per dati senza etichette, pretraining self-supervised e distillazione di un backbone DINO.

Lightly non è un detector. Serve a sfruttare i dati che non hai etichettato e si compone di due parti con licenze diverse.

- **[`lightly`](https://github.com/lightly-ai/lightly) (LightlySSL), MIT.** Un framework di componenti per il self-supervised learning: loss, head, data augmentation, memory bank e implementazioni di riferimento di oltre venti metodi (SimCLR, MoCo, BYOL, DINO, DINOv2, MAE e altri). Scrivi tu il loop di addestramento; il framework ti fornisce tutto il necessario per preaddestrare da zero una rappresentazione usando immagini senza etichette.
- **[LightlyTrain](https://github.com/lightly-ai/lightly-train), AGPL-3.0 (con opzioni commerciali e gratuite per la community).** La soluzione pronta all'uso, ovvero il workflow a cui si riferisce la maggior parte delle persone. Indica un foundation model come DINOv2 o DINOv3 e i tuoi dati senza etichette, poi distilla quel backbone in un modello più piccolo che puoi mettere in produzione (YOLO, RT-DETR, ViT o una rete personalizzata) con poche righe e senza etichette. Tieni presente la licenza: AGPL-3.0 impone lo stesso copyleft che spinge le persone a cercare alternative a Ultralytics. Leggi le condizioni se vuoi distribuire software a codice chiuso, oppure scegli la licenza commerciale.

Scegli Lightly quando il collo di bottiglia è avere un buon backbone ma non le etichette. Si abbina ai detector di questo elenco, non compete con loro. Anche le novità lo dimostrano: RT-DETRv4 integra la distillazione di un vision foundation model nell'addestramento e DEIMv2 incorpora direttamente i backbone DINOv3 nel detector.

## 4. YOLOX

Licenza: Apache-2.0. Ideale per il rilevamento real-time anchor-free, una volta installato.

Il repository originale è fermo dall'ultima release, la v0.3.0 di aprile 2022, e ha oltre 700 issue aperte. Inoltre, è difficile da installare con uno stack del 2026. Non c'è un `pyproject.toml` e `requirements.txt` fissa `onnx-simplifier==0.4.10`, che non offre wheel precompilati per versioni di Python successive alla 3.10. Con un interprete moderno il pacchetto viene quindi compilato dai sorgenti e richiede cmake e una toolchain C++. Anche NumPy non è vincolato a una versione, con il rischio di incompatibilità ABI tra NumPy 2.0 e le versioni più vecchie di pycocotools e torch. Una volta risolte le dipendenze, il rilevamento funziona, ma arrivarci ha un costo.

Il modello in sé è ancora valido: un detector real-time anchor-free di Megvii, con licenza Apache-2.0 sia per il codice sia per i pesi, e una baseline ragionevole anche se nel frattempo detector più recenti lo hanno superato. Rimettere in sesto un repository abbandonato ma valido è il tipo di lavoro che oggi un agente di coding può fare in un pomeriggio, quindi l'installazione è un ostacolo meno insormontabile di quanto sembri. Anche LibreYOLO esegue direttamente i pesi YOLOX su uno stack aggiornato. In entrambi i casi, vale la pena usare l'architettura. Abbiamo scritto una guida più completa [qui](/articles/yolox-with-libreyolo).

## 5. L'universo MMDetection

Licenza: per lo più Apache-2.0. Ideale per l'ampia scelta di architetture e per riprodurre i paper.

Per anni MMDetection ha ospitato l'implementazione ufficiale di quasi tutti i paper sul rilevamento: Faster R-CNN, DINO, Grounding-DINO, RTMDet, Mask R-CNN e centinaia di configurazioni. Nel 2026, OpenMMLab ha dismesso la serie mm. L'ultima release di [MMDetection](https://github.com/open-mmlab/mmdetection) è la v3.3.0, di gennaio 2024; le issue restano senza risposta e lo stack è bloccato al vincolo di versione di `mmcv`, i cui wheel precompilati non sono al passo con le versioni attuali di PyTorch e CUDA. Una nuova installazione tende quindi a rompersi, finché non si fa il downgrade di tutto. Abbiamo parlato di questi problemi [qui](/articles/rtmdet-without-mmdetection).

Un'azienda olandese, VBTI, lo tiene in vita. Il suo [fork OneDL](https://github.com/VBTI-development/onedl-mmdetection) ripubblica l'intero stack con il prefisso `onedl-` (`onedl-mmdetection`, `onedl-mmcv`, `onedl-mmengine` e gli altri), ricostruito per PyTorch 2.x, CUDA e Python 3.10+ attuali, risolvendo il vincolo di versione. Usa Apache-2.0 ed è mantenuto attivamente; la versione v3.5.1 è uscita a maggio 2026. Se vuoi tutta la varietà di MMDetection senza il lavoro di installazione, usa questo fork. È mantenuto da un piccolo team, quindi contribuisci se fai affidamento su di esso.

Due strumenti correlati:

- **[Detectron2](https://github.com/facebookresearch/detectron2)** (Meta, Apache-2.0) è in modalità manutenzione e non ha una release taggata dalla v0.6 del 2021, ma è affidabile e resta la fonte più pulita per Mask R-CNN e la famiglia R-CNN, per la segmentazione di istanze e panottica.
- **[TorchVision](https://github.com/pytorch/vision)** (BSD-3-Clause, mantenuto attivamente) include Faster R-CNN, RetinaNet, FCOS, SSD, Mask R-CNN e Keypoint R-CNN. Non include YOLO o DETR moderni e devi scrivere il tuo loop di addestramento, ma è la baseline senza dipendenze aggiuntive.

Una nota sull'uso commerciale: **MMYOLO**, l'hub di OpenMMLab che reimplementa YOLO, usa GPL-3.0 invece di Apache-2.0 ed è fermo da agosto 2023.

## 6. La famiglia RT-DETR

Licenza: Apache-2.0 per tutti. Ideale per il rilevamento real-time all'avanguardia con codice di livello ricerca.

L'attuale frontiera del rilevamento real-time è un gruppo di detector transformer correlati, non un modello YOLO. La maggior parte deriva da RT-DETR, condivide gran parte del codice del backbone e dell'encoder, e quasi tutti usano Apache-2.0, quindi passare da un modello all'altro è semplice. Quasi tutti supportano solo il rilevamento, con un'eccezione: EdgeCrafter applica le stesse idee di distillazione anche alla segmentazione e alla posa. Il limite comune è che questi sono repository di ricerca: addestramento tramite file di configurazione, ergonomia diversa da Ultralytics e nessun canale di supporto. I modelli sono validi e in genere l'esportazione ONNX e TensorRT è supportata in modo completo.

- **[D-FINE](https://github.com/Peterande/D-FINE)** (USTC, ICLR 2025 Spotlight). Riformula la regressione dei box come raffinamento della distribuzione con self-distillation. Offre un buon rapporto tra accuratezza e latenza (D-FINE-X raggiunge circa 55.8 AP in circa 13 ms su una T4), è disponibile nelle dimensioni da N a X ed è integrato in Hugging Face Transformers. Questo lo rende il più facile da caricare e sottoporre a fine-tuning al di fuori del suo repository.
- **[DEIM e DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)** (Intellindust AI Lab). DEIM (CVPR 2025) è una ricetta di addestramento (matching Dense O2O) che si aggiunge a D-FINE o RT-DETRv2 e dimezza circa i tempi di addestramento. DEIMv2 aggiunge le feature DINOv3 e arriva a una variante Atto con meno di 1M di parametri per dispositivi mobili; DEIMv2-S è il primo modello con meno di 10M di parametri a superare 50 AP. È mantenuto attivamente e aggiornato nel 2026.
- **[EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter)** (Intellindust AI Lab, 2026). Il lavoro più recente dello stesso gruppo e l'unica voce dell'elenco che va oltre il rilevamento: copre il rilevamento (ECDet), la segmentazione di istanze (ECSeg) e la posa (ECPose), ognuno nelle dimensioni S/M/L/X e tutti con licenza Apache-2.0. Adatta un ViT di grandi dimensioni preaddestrato con DINOv3 come teacher specializzato per task e lo distilla in backbone student compatti pensati per l'edge. I risultati sono notevoli per modelli di queste dimensioni: ECDet-S raggiunge 51.7 AP con meno di 10M di parametri sul solo COCO, ECPose-X arriva a 74.8 AP (superando YOLO26Pose-X, a 71.6) e la segmentazione di istanze è competitiva con RF-DETR. È appena uscito, quindi consideralo una release di ricerca, ma è raro trovare un modello compatto che copra tutti e tre i task.
- **[RT-DETR e RT-DETRv2](https://github.com/lyuwenyu/RT-DETR)** (Baidu). Il lavoro originale che sostiene che i "DETR superano YOLO nel rilevamento real-time" (CVPR 2024) e il capostipite della famiglia: senza NMS, end-to-end e facile da esportare. v2 è un aggiornamento del 2024 con una raccolta di miglioramenti gratuiti, un upgrade drop-in con la stessa latenza. Lo stesso repository contiene l'originale Paddle e il port PyTorch canonico, ed è integrato in HF Transformers.
- **[RT-DETRv3](https://github.com/clxia12/RT-DETRv3)** (Baidu, WACV 2025 Oral) e **[RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4)** (Peking University e Tsinghua, fine 2025). v3 è disponibile solo per PaddlePaddle. v4 è il migliore della linea attuale (X si avvicina a 57 AP), usa PyTorch e distilla un vision foundation model in un detector leggero senza costi aggiuntivi in inferenza. La sua codebase riproduce anche D-FINE, DEIM e RT-DETRv2 modificando le configurazioni.
- **[LW-DETR](https://github.com/Atten4Vis/LW-DETR)** (Baidu). Un DETR basato su ViT puro, presentato come alternativa transformer a YOLO, disponibile da tiny a xlarge e con esportazione ONNX e TensorRT. È anche alla base di OVLW-DETR a vocabolario aperto. L'attività recente è diminuita; consideralo una release di ricerca stabile.

LibreYOLO integra già molti di questi modelli (D-FINE, DEIM, DEIMv2, RT-DETRv2, RT-DETRv4, RTMDet, EdgeCrafter) con un'unica API; YOLO9 e RF-DETR sono quelli testati più a fondo. Per le implementazioni di riferimento e i checkpoint di ricerca più recenti, i repository originali sopra sono la fonte canonica.

## FAQ

**Ultralytics YOLO è gratuito per uso commerciale?**
Solo con AGPL-3.0, che richiede di rilasciare il codice sorgente dell'applicazione che ci costruisci sopra, compresi i servizi di rete. Per l'uso commerciale a codice chiuso serve la licenza commerciale a pagamento di Ultralytics. Le alternative permissive (MIT, Apache-2.0) evitano questo obbligo.

**Qual è l'alternativa a Ultralytics con la licenza più semplice?**
Per uno stack permissivo distribuibile ovunque: LibreYOLO (codice MIT), RF-DETR (Apache-2.0) e YOLOX (Apache-2.0). Nessuno impone il copyleft di AGPL.

**Cosa sta sostituendo YOLO nel 2026?**
Sempre più spesso, i detector transformer real-time: RT-DETR, RF-DETR e la linea D-FINE e DEIM. Su hardware adeguato e con una latenza simile, eguagliano o superano l'accuratezza di YOLO. Le CNN in stile YOLO restano vincenti sui piccoli dispositivi edge, dove quei transformer funzionano male e non hanno un percorso maturo per NCNN o CPU.

**Questi modelli funzionano su Raspberry Pi o su un NPU?**
Dipende dall'esportazione. I detector CNN come YOLOX e RTMDet si esportano in NCNN e funzionano su un Pi; alcuni, come YOLOX, funzionano anche sul NPU Hailo. I detector transformer come RF-DETR no: richiedono una GPU o una CPU potente.

## Provalo

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLO usa la licenza MIT, funziona su Linux, Mac e Windows, e gira su GPU, Apple Silicon e una normale CPU senza modifiche al codice. Un'unica API copre RF-DETR, D-FINE, DEIM, YOLOX, YOLO-NAS, RTMDet, RT-DETR, EdgeCrafter e altri, per rilevamento, segmentazione, posa, classificazione, profondità, gaze e tracking.

Aggiungilo ai preferiti su GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentazione: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
