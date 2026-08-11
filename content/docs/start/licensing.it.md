---
title: Licenze
seo_title: 'Licenze di LibreYOLO: codice e pesi'
description: >-
  Il codice di LibreYOLO è MIT. Il codice upstream incluso e i checkpoint
  pubblicati hanno licenze proprie, e diverse di queste sono non commerciali.
lead: >-
  LibreYOLO comprende tre cose con licenze separate: il proprio codice, il
  codice upstream incluso in una famiglia di modelli e i checkpoint
  preaddestrati. Spesso non hanno la stessa licenza.
keywords:
  - licenza libreyolo
  - libreria computer vision mit
  - pesi modello uso non commerciale
  - licenza checkpoint modello
  - object detection apache-2.0
last_verified: 1.5.0
source_hash: 83536fea4dc4eaec
---

## Il codice di LibreYOLO

La libreria è MIT. Questo copre l'API Python, la CLI, i trainer, i validator e
gli exporter, i loader dei dataset e gli script di conversione sotto
`weights/`. Usala in un prodotto commerciale o closed-source, conserva la riga
di copyright e il testo della licenza con ogni copia che ridistribuisci, e
l'obbligo finisce lì.

La concessione si ferma al codice. Il file
[`LICENSE`](https://github.com/LibreYOLO/libreyolo/blob/release/LICENSE) lo
dice chiaramente:

> Quelle licenze variano e non sono tutte permissive: alcuni pesi pubblicati
> sono non commerciali o comunque soggetti a restrizioni, e questa MIT License
> non si estende a loro. Scegliere un modello significa scegliere la sua
> licenza.

## Codice upstream, per famiglia

La maggior parte delle famiglie sono port di ricerca pubblicata, e diverse
includono direttamente il sorgente upstream. Un file incluso da upstream
mantiene l'header di copyright originale e la licenza originale. MIT non lo
sovrascrive, e LibreYOLO non cambia la licenza del lavoro di nessuno.
Apache-2.0 e BSD-3-Clause sono le due che ricorrono più spesso.

Apache-2.0 copre la linea DETR e gran parte del lavoro sui transformer: DETR di
Meta AI (FAIR), Deformable DETR di SenseTime, LW-DETR di Baidu, OV-DEIM di
Leilei Wang e coautori, l'implementazione di SegFormer che LibreYOLO porta da
Hugging Face Transformers, PP-OCRv5 dei PaddlePaddle Authors, SwinIR del
Computer Vision Lab dell'ETH di Zurigo, e Depth Anything 3 di ByteDance Seed.
Copre anche i classificatori derivati da timm di Ross Wightman e dei
contributori di timm, tra cui ResNet, DeiT, EfficientNetV2, MobileNetV4 e Swin,
i cui nomi dei moduli rispecchiano quelli di timm in modo che i suoi tensori
ImageNet si carichino senza modifiche.

BSD-3-Clause copre tutto ciò che deriva da torchvision: Faster R-CNN,
Mask R-CNN, FCOS, RetinaNet, SSD300, AlexNet, VGG, FCN e DeepLabv3.

MIT copre un gruppo più ristretto, che comprende NAFNet di Megvii, CenterNet di
Xingyi Zhou, e YOLOv7 nella riedizione dei suoi stessi autori, Kin-Yiu Wong e
Hao-Tang Tsui, presso MultimediaTechLab. Le famiglie da YOLOv1 a YOLOv4
riproducono architetture del progetto Darknet, di Joseph Redmon e, per YOLOv4,
di Alexey Bochkovskiy. Darknet è di pubblico dominio, quindi quelle non
comportano alcun obbligo.

Un sottoalbero incluso non ha una licenza open source. La famiglia DEIMv2
distribuisce il codice del backbone DINOv3 di Meta Platforms sotto il DINOv3
License Agreement, una licenza personalizzata non OSI. Ridistribuire quel
codice significa distribuire con esso una copia dell'accordo, e l'accordo vieta
l'uso per attività soggette a ITAR, scopi militari o bellici, industrie
nucleari, spionaggio e sviluppo di armi. Quei termini vincolano solo quel
sottoalbero.

Due file nel repository contengono il quadro completo.
[`NOTICE`](https://github.com/LibreYOLO/libreyolo/blob/release/NOTICE) elenca
ogni sottoalbero di terze parti incluso, con il suo percorso, il suo file di
licenza e la sua sorgente upstream.
[`THIRD_PARTY_NOTICES.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/THIRD_PARTY_NOTICES.txt)
elenca i progetti upstream da cui LibreYOLO deriva e riproduce per intero il
testo di ogni licenza.

## Pesi, per checkpoint

Nessun file di pesi preaddestrati è incluso nel pacchetto. I checkpoint
pubblicati si trovano su Hugging Face sotto l'[organizzazione
LibreYOLO](https://huggingface.co/LibreYOLO), e ogni repository ha il proprio
`LICENSE` e la propria attribuzione, che riflettono il progetto da cui
provengono i pesi.

Quel repository è la fonte autorevole per i termini. Non questa pagina, non la
pagina del modello e non il riepilogo nell'albero dei sorgenti. Vedi
[checkpoint e pesi](/docs/weights) per come vengono nominati i file e da dove
vengono scaricati.

Le licenze differiscono tra le famiglie, e differiscono tra i file all'interno
di una stessa famiglia. Due esempi del secondo caso:

- I checkpoint COCO di YOLO9 sono MIT. `LibreYOLO9P2s-visdrone.pt`, addestrato
  su VisDrone2019-DET, è CC BY-NC-SA 3.0, quindi non commerciale.
- I checkpoint di rilevamento di RF-DETR sono Apache-2.0. I checkpoint per box
  orientati sono CC BY 4.0, perché hanno fatto fine-tuning su un dataset di
  Roboflow Universe pubblicato sotto CC BY 4.0 e i pesi portano avanti
  l'obbligo di attribuzione di quel dataset.

Tra una famiglia e l'altra la varietà è più ampia, e diversi checkpoint
pubblicati non possono essere usati in un prodotto commerciale:

- SegFormer è la separazione più netta tra i due livelli. L'implementazione è
  un port Apache-2.0 del codice di Hugging Face Transformers. I checkpoint
  ADE20K pubblicati sono convertiti dalla release di NVIDIA sotto la NVIDIA
  Source Code License, che permette la ridistribuzione ma limita l'uso alla
  ricerca o alla valutazione non commerciali, e trasferisce quel limite alle
  opere derivate. Quei checkpoint non sono coperti dai termini permissivi di
  LibreYOLO.
- I checkpoint di OV-DEIM sono CC BY-NC 4.0, come confermato dall'autore
  upstream. Ogni predizione carica anche la text tower MobileCLIP-B(LT) di
  Apple, la cui licenza limita l'uso alla ricerca, un termine più restrittivo
  di quello del checkpoint stesso.
- Il codice di SenseNova-Vision è Apache-2.0 e i suoi pesi sono CC BY-NC 4.0.
  Il loader stampa l'avviso di uso non commerciale prima di ogni download
  automatico.

Alcune famiglie non hanno alcun checkpoint ospitato da LibreYOLO, e le loro
pagine lo dicono nella riga Pesi. SAM 3 è ad accesso condizionato su Hugging
Face sotto la SAM License personalizzata di Meta e viene scaricato direttamente
da Meta. Gli asset delle release di MiDaS vengono scaricati dagli URL ufficiali
e verificati tramite hash, anziché essere riospitati. Dome-DETR rimanda a
upstream perché la sua model card non indica alcuna licenza nei metadati mentre
il testo dichiara Apache-2.0 e allo stesso tempo limita l'uso alla ricerca
accademica, e le due cose non concordano. Le architetture TEED e DexiNed sono
MIT, ma i checkpoint rilasciati dagli autori sono stati addestrati su BIPED, i
cui termini di dataset sono non commerciali, quindi LibreYOLO non li include né
li scarica automaticamente.

Diversi checkpoint di torchvision non hanno un proprio file di licenza.
LibreYOLO ne fa il mirror sotto la licenza usata dal progetto che li rilascia,
dichiara su ogni model card che la base è implicita anziché concessa per
singolo checkpoint, e ripete l'avvertenza di torchvision stesso secondo cui i
termini di un modello preaddestrato possono derivare dai dati di addestramento.

## Trovare i termini di un singolo modello

La pagina del modello ha una riga **Licenze** nell'intestazione, nella forma
`Code X, weights Y`, che rimanda alla sezione Licenze della pagina. Quella
sezione elenca l'opera originale e i suoi autori, la licenza upstream, la
sorgente upstream, la licenza del codice LibreYOLO, i pesi e
un'interpretazione di ciò che i termini consentono. La tabella Checkpoint sulla
stessa pagina ha una colonna **Licenza dei pesi**, con una riga per ogni file
pubblicato, così una famiglia con termini misti li mostra file per file.

Tutto questo viene generato dagli stessi dati con cui la libreria viene
verificata, ed è per questo che questa pagina non li ripete sotto forma di
tabella. Una matrice delle licenze scritta a mano è sbagliata nel giro di una
release, e sbagliarsi qui costa caro.

Nell'albero dei sorgenti gli equivalenti sono `NOTICE` per il codice incluso,
`THIRD_PARTY_NOTICES.txt` per i progetti upstream e i testi delle loro licenze,
e [`weights/LICENSE_NOTICE.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/weights/LICENSE_NOTICE.txt)
per un riepilogo per famiglia dei checkpoint pubblicati.

Poi controlla il repository Hugging Face del file esatto che stai per
scaricare. È la fonte autorevole, e può cambiare senza che una pagina della
documentazione cambi con esso.

## Uso commerciale

Il codice è raramente il problema. MIT, Apache-2.0 e BSD-3-Clause permettono
tutte l'uso commerciale e closed-source. Ognuna chiede di conservare il testo
della licenza e gli avvisi di attribuzione con le copie che ridistribuisci,
Apache-2.0 concede anche una licenza sui brevetti, e nessuna di esse pone
condizioni sul codice della tua applicazione.

È sui checkpoint che i prodotti si bloccano. Un checkpoint non commerciale
resta non commerciale per quanto permissivo sia il codice attorno, e convertire
il file non cambia i termini applicabili, come `weights/LICENSE_NOTICE.txt`
afferma esplicitamente. Un artefatto ONNX o TensorRT costruito da un checkpoint
soggetto a restrizioni eredita la restrizione.

Quando una licenza trasferisce la sua restrizione alle opere derivate, come fa
la NVIDIA Source Code License, nemmeno il fine-tuning permette di sfuggirle.
Addestrare la stessa architettura da zero su dati che hai il diritto di usare
sì: il codice è permissivo, quindi un modello che addestri tu è tuo, e i
termini del checkpoint preaddestrato non vi entrano mai. La pagina di SegFormer
lo esplicita per i propri pesi; leggi la riga Interpretazione sulla pagina
della famiglia che intendi mettere in produzione.

Decidi la questione della licenza quando scegli il modello, non quando vai in
produzione, e leggi i termini del file che hai effettivamente scaricato, perché
una famiglia con un checkpoint permissivo può averne accanto uno soggetto a
restrizioni.

## Non è una consulenza legale

Questa pagina descrive le licenze coinvolte. È una descrizione, non una
consulenza legale, e non crea alcuna garanzia. Se la risposta ha rilievo
commerciale, leggi tu stesso le licenze e rivolgiti a un legale di fiducia.
