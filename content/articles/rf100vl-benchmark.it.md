---
title: "Benchmark RF100-VL: quanto generalizzano i modelli LibreYOLO?"
description: Risultati RF100-VL che misurano la capacità di YOLOv9, YOLOX, YOLO-NAS, EdgeCrafter e RF-DETR di generalizzare su 100 dataset reali dopo il fine-tuning.
date: 2026-09-05
author: Xuban
layout: paper
tags: [LibreYOLO, RF100-VL, benchmark, object-detection, roboflow]
---

RF100-VL misura quanto bene un detector si adatta a contesti diversi da COCO. Ogni modello viene sottoposto separatamente a fine-tuning su 100 dataset molto diversi, che includono infrarossi, raggi X, microscopia, sport, immagini dello spettro radio, oggetti minuscoli e videogiochi. Abbiamo eseguito il benchmark su 17 configurazioni LibreYOLO: **1700 fine-tuning in totale.**

<div>
<rf100vl-explorer></rf100vl-explorer>
</div>

## Risultati

<div>
<rf100vl-results-chart></rf100vl-results-chart>
</div>

RF-DETR-L guida questo confronto con **61.76**. Segue M con 61.13, poi S con 60.41. I quattro risultati di RF-DETR sono vicini ai [numeri pubblicati da Roboflow](https://rfdetr.roboflow.com/latest/learn/benchmarks/), una conferma utile dell'addestramento di RF-DETR in LibreYOLO.

La dimensione non ha previsto tutti i risultati. YOLO-NAS-S e M sono sostanzialmente a pari merito, con 58.00 e 57.99. EdgeCrafter-M ottiene 57.93, davanti a S con 55.99 e L con 56.11. Contiamo di approfondire queste regressioni. Questa campagna non è un esperimento controllato sulla scalabilità: risoluzione, pesi preaddestrati, precisione e ricette variano.

RF-DETR-S con LoRA sul backbone ottiene 57.19, rispetto a 60.41 con il fine-tuning completo. Il fine-tuning completo vince su 95 dataset. La mediana registrata è più lenta di soli sette minuti, mentre il tempo totale dei job è pressoché invariato.

Le righe di YOLOv9 risalgono a prima di importanti correzioni all'addestramento. Il risultato anomalo di M ha contribuito a individuare l'assenza di PGI, una convenzione letterbox diversa e un limite di 100 etichette per l'addestramento. I numeri archiviati descrivono il codice eseguito. Non sono un giudizio sull'architettura YOLOv9.

Scegli un modello qui sotto per esaminare i suoi punteggi su tutti i 100 dataset.

<div>
<rf100vl-results-detail></rf100vl-results-detail>
</div>

## Metodologia

Per ogni dataset, abbiamo addestrato il modello su `train`, selezionato il checkpoint con il miglior AP50:95 di validazione e valutato una sola volta su `test`. Il punteggio principale è la media non ponderata dei 100 risultati sul test.

| Impostazione | Regola della campagna |
|---|---|
| Addestramento | 100 epoche; nessun early stopping |
| Batch effettivo | 16 |
| Seed | 0; una sola esecuzione completata per dataset |
| Checkpoint | Miglior AP50:95 di validazione con pesi EMA |
| Valutazione sul test | pycocotools con `maxDets=500` |
| Tuning | Una ricetta fissata per famiglia; nessun tuning per dataset |

Ogni famiglia ha mantenuto la propria ricetta di addestramento:

| Famiglia | Risoluzione | Precisione | Descrizione delle augmentation |
|---|---:|---:|---|
| YOLOv9 T/S/M | 640 | FP32 | Mosaic, HSV, flip; augmentation avanzate disattivate nelle ultime 15 epoche |
| YOLOX Nano/Tiny | 416 | FP32 | Mosaic, mixup, HSV, affine, flip; coda di 15 epoche |
| YOLOX S/M | 640 | FP32 | Stessa ricetta della famiglia, con learning rate specifico per dimensione |
| YOLO-NAS S/M | 640 | FP32 | Mixup, HSV e flip; mosaic disattivato |
| EdgeCrafter S/M/L | 640 | FP16 | Flip; valori predefiniti della famiglia LibreYOLO |
| RF-DETR N/S/M/L e S LoRA | 384/512/576/704; LoRA a 512 | BF16 | Multi-scala, crop/resize e flip |

Sono risultati ottenuti con un solo seed. Le piccole differenze non dimostrano miglioramenti. RF-DETR M e L hanno usato l'accumulo dei gradienti per rientrare nella memoria disponibile, e alcuni dataset densi hanno richiesto batch fisici più piccoli. Anche le ricette pubbliche di RF-DETR partono da checkpoint COCO, mentre Roboflow ha usato checkpoint privati Objects365 per la propria tabella storica.

I tempi di addestramento sono dati registrati durante la campagna, non benchmark di velocità controllati. A volte i job condividevano le GPU, venivano ritentati o ripresi. Non abbiamo eseguito il protocollo facoltativo di latenza T4 su singolo artifact. YOLO-NAS usa i [pesi preaddestrati non commerciali](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md) di Deci, distribuiti con licenza separata.

## Cosa è migliorato grazie al carico di lavoro

I test brevi possono dimostrare che l'addestramento parte. Non riproducono settimane di fine-tuning continuo su molte architetture e dataset. A questa scala, è diventato difficile non notare i percorsi lenti, la pressione sulla memoria e i problemi di correttezza.

- **Caricamento delle immagini e validazione.** Abbiamo memorizzato in cache il ridimensionamento deterministico prima dell'augmentation, riutilizzato i worker e i buffer del validator e tracciato i forward di validazione dove supportato. [PR #677](https://github.com/LibreYOLO/libreyolo/pull/677), [PR #682](https://github.com/LibreYOLO/libreyolo/pull/682)
- **CUDA graph per l'addestramento.** Il supporto alla cattura è passato da YOLOv9 e RF-DETR a 24 famiglie. Abbiamo anche corretto una race condition sul pin-memory del DataLoader e l'invalidazione dei graph durante le transizioni dell'addestramento. Un test di YOLOv9-T è passato da 428.4 a 367.7 secondi con lo stesso AP riportato. [PR #671](https://github.com/LibreYOLO/libreyolo/pull/671), [PR #681](https://github.com/LibreYOLO/libreyolo/pull/681), [PR #716](https://github.com/LibreYOLO/libreyolo/pull/716)
- **Valutazione COCO.** Sui dataset densi, la valutazione a volte richiedeva più tempo dell'addestramento. L'integrazione faster-coco-eval ha calcolato i punteggi delle predizioni salvate per tutti i 100 split di test in 8.4 secondi anziché 131.4. Su 1,400 valori di metrica, 1,381 erano identici a livello di bit; la differenza massima era 2.22e-16 e l'AP principale non è cambiato. [PR #708](https://github.com/LibreYOLO/libreyolo/pull/708)
- **RF-DETR e percorsi di addestramento condivisi.** Un matching L1 più veloce, meno sincronizzazioni del device, AdamW fuso e memoria limitata per il matcher hanno ridotto uno step di RF-DETR-S da 266 a 234 ms nel test registrato. La stessa indagine ha migliorato altri cinque matcher, il logging di YOLOv9 e il riuso dei tensori di EdgeCrafter. [PR #761](https://github.com/LibreYOLO/libreyolo/pull/761), [PR #762](https://github.com/LibreYOLO/libreyolo/pull/762), [PR #765](https://github.com/LibreYOLO/libreyolo/pull/765)
- **Attention.** Abbiamo instradato l'attention idonea attraverso PyTorch SDPA, integrato un'implementazione CUDA con licenza Apache-2.0 e corretto l'esecuzione a precisione mista. Abbiamo anche scritto un kernel Triton in-tree per la deformable attention in inferenza. Nei test documentati è risultato circa quattro volte più veloce da solo e circa il 7% più veloce end-to-end per RF-DETR-N. [PR #712](https://github.com/LibreYOLO/libreyolo/pull/712), [PR #713](https://github.com/LibreYOLO/libreyolo/pull/713), [PR #760](https://github.com/LibreYOLO/libreyolo/pull/760), [PR #784](https://github.com/LibreYOLO/libreyolo/pull/784), [PR #790](https://github.com/LibreYOLO/libreyolo/pull/790)
- **Correttezza dell'addestramento.** Abbiamo corretto la ricostruzione di BatchNorm in YOLOX e ripristinato PGI, i metadati del letterboxing, la capacità delle etichette e il warmup del momentum in YOLOv9. Il benchmark ha fornito i checkpoint e gli schemi di errore che hanno fatto emergere entrambi i problemi. [PR #700](https://github.com/LibreYOLO/libreyolo/pull/700), [PR #796](https://github.com/LibreYOLO/libreyolo/pull/796)

Questi numeri provengono da test separati su carichi di lavoro diversi. Non vanno moltiplicati per ottenere un unico miglioramento della velocità. Ora LibreYOLO è più veloce e affidabile sui carichi di lavoro reali di addestramento rispetto a prima di questa campagna.

## Harness e artifact

L'[harness di addestramento pubblico](https://github.com/LibreYOLO/vision-analysis-benchmark/tree/rf100vl-harness) distribuisce i dataset sulle GPU, riprende i job interrotti, gestisce il recupero da OOM e registra ricette, versioni dei dataset, log, checkpoint e predizioni.

La [skill run-rf100vl-benchmark](https://github.com/LibreYOLO/libreyolo/blob/release/skills/run-rf100vl-benchmark/SKILL.md) fornisce agli agenti di coding AI il protocollo e le istruzioni operative per Vast.ai. L'[archivio dei risultati](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main) contiene ogni esecuzione pubblicata.

## Grazie, Roboflow

Joseph Nelson ha offerto supporto GPU dopo che avevo scritto di voler eseguire un benchmark oltre COCO, ma di non potermi permettere le esecuzioni. Matvei Popov ha condiviso le configurazioni di riferimento, spiegato le impostazioni di valutazione e seguito i risultati man mano che arrivavano.

Il loro supporto ci ha permesso di convalidare l'addestramento di LibreYOLO su 17 configurazioni, pubblicare dati utili per scegliere un modello, rilasciare l'harness e mettere sotto pressione la libreria finché i suoi punti deboli non sono diventati evidenti.

Grazie a Joseph, Matvei e al team Roboflow per la potenza di calcolo, il tempo e la guida.
