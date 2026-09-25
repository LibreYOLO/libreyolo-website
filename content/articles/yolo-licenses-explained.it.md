---
title: "Licenze YOLO spiegate: da v1 a YOLO26 (guida 2026)"
description: "La mappa completa delle licenze YOLO nel 2026: da YOLOv1 a YOLOv13, YOLO26, YOLO-World e YOLOE, repository originali e riscritture permissive, link a paper e GitHub e cosa puoi davvero distribuire in un prodotto commerciale."
date: 2026-07-11
author: Xuban
tags: [LibreYOLO, yolo-license, agpl, gpl, apache, mit-license, yolov9, yolov10, yolo11, yolov12, yolov13, yolo26]
faq:
  - q: "YOLOv9 è gratuito per uso commerciale?"
    a: "Dipende dal repository che usi. Il repository del paper (WongKinYiu/yolov9) è GPL-3.0. Lo stesso laboratorio distribuisce anche un'implementazione separata di YOLOv9 e YOLOv7 con licenza MIT in MultimediaTechLab/YOLO, scritta dopo che utenti commerciali avevano chiesto un'opzione permissiva. È una riscrittura, non una riassegnazione della licenza dei file GPL. La licenza del codice è chiara; nessuno dei due repository dichiara separatamente la licenza dei pesi preaddestrati, quindi considera la provenienza dei pesi una questione a sé."
  - q: "Quali versioni di YOLO sono gratuite per l'uso commerciale a codice chiuso?"
    a: "Esiste almeno un repository permissivo per: da YOLOv1 a YOLOv4 (Darknet originale, pubblico dominio), YOLOv7 e YOLOv9 (riscrittura MIT del laboratorio), YOLOX e PP-YOLOE (Apache-2.0) e le reimplementazioni MIT di LibreYOLO. A luglio 2026 non esiste un'implementazione permissiva di YOLOv5, v6, v8, v10, YOLO11, v12, v13, YOLO26, YOLO-World e YOLOE. Prima di farci affidamento, controlla il file LICENSE corrente: le licenze cambiano e queste sono informazioni generali, non consulenza legale."
  - q: "Qual è la licenza di YOLOv10?"
    a: "AGPL-3.0. YOLOv10 è una release accademica della Tsinghua University, ma il suo codice deriva dalla codebase di Ultralytics e quindi eredita la licenza AGPL-3.0. Non esiste una reimplementazione permissiva."
  - q: "Qual è la licenza di YOLOv12 e YOLOv13?"
    a: "Entrambi sono AGPL-3.0, come confermano i rispettivi file LICENSE. Come YOLOv10, sono paper accademici il cui codice di riferimento deriva dal repository Ultralytics, quindi si applica la AGPL indipendentemente da chi abbia scritto il paper. Le pagine di terze parti che indicano YOLOv13 come Apache-2.0 sono sbagliate."
  - q: "Quando Ultralytics è passata alla licenza AGPL-3.0 per YOLO?"
    a: "Il 14 aprile 2023, non nel 2022 come viene spesso ripetuto. Nella storia di ultralytics/yolov5 il file LICENSE è stato modificato solo tre volte, e il commit AGPL (34cf749, PR #11359) è datato 2023-04-14. L'ultima release del 2022, YOLOv5 v7.0, era ancora GPL-3.0. Lo stesso giorno è stato riassegnato alla nuova licenza il repository ultralytics/ultralytics. YOLOv8 è quindi stato lanciato a gennaio 2023 sotto GPL-3.0: le release PyPI dalla 8.0.0 alla 8.0.76 dichiarano GPL-3.0, mentre la 8.0.80 (16 aprile 2023) è la prima a dichiarare AGPL-3.0."
  - q: "YOLO26 è gratuito per uso commerciale?"
    a: "Non per i prodotti a codice chiuso. YOLO26 è distribuito con licenza AGPL-3.0 e offre come alternativa una licenza Enterprise a pagamento, alle stesse condizioni di YOLOv8 e YOLO11."
  - q: "YOLO originale di Darknet è davvero di pubblico dominio?"
    a: "Sì. La LICENSE di Darknet di Joseph Redmon dichiara 'Darknet is public domain. Do whatever you want with it.' Il fork di YOLOv4 di AlexeyAB contiene lo stesso testo sul pubblico dominio. Il problema è che i port PyTorch usati davvero hanno licenze proprie, da Apache-2.0 ad AGPL-3.0 fino all'assenza di licenza."
  - q: "Posso usare commercialmente i pesi preaddestrati di YOLO-NAS?"
    a: "No. Il codice di super-gradients è Apache-2.0, ma i pesi ufficiali di YOLO-NAS sono distribuiti con una licenza separata che afferma: 'may not use the Software for any commercial use, including in connection with any models used in a production environment.'"
  - q: "I pesi di una rete neurale ereditano la licenza del codice di addestramento?"
    a: "La questione non è risolta. Ultralytics afferma che la AGPL-3.0 copre 'the training code and the models produced by that training code' e, in quanto titolare del copyright, stabilisce i termini per ciò che pubblica. Non è mai stato verificato in tribunale se i pesi che addestri tu siano un'opera derivata dal codice di addestramento. Considera soggetti a vincoli i pesi AGPL pubblicati e fai attenzione quando li carichi in una reimplementazione permissiva."
  - q: "Perché implementazioni diverse dello stesso YOLO possono avere licenze diverse?"
    a: "Il copyright protegge l'espressione, non le idee (17 U.S.C. 102(b)). Un'architettura descritta in un paper può essere implementata indipendentemente e concessa in licenza come preferisce chi la implementa. Non puoi copiare file sorgente GPL o AGPL e riassegnarli a un'altra licenza. Nota che questo è un argomento sul copyright: un'implementazione indipendente non è una difesa contro i brevetti."
---

Ogni anno arrivano nuovi YOLO e ogni anno torna la domanda sulle licenze, di solito cinque minuti prima di una decisione sul prodotto. La confusione nasce dal fatto che "YOLO" non è un singolo progetto. È un nome condiviso da una ventina di famiglie di modelli di autori diversi, e c'è un dettaglio che sfugge alla maggior parte delle guide sulle licenze: **la licenza appartiene a un repository, non a un modello.** La stessa versione di YOLO può esistere contemporaneamente in un repository GPL e in uno MIT, anche dello stesso gruppo di autori. È il caso di YOLOv9 e cambia completamente la decisione.

Questa guida mappa il panorama repository per repository, con link a ogni paper e codebase, aggiornata a luglio 2026. Abbiamo verificato ogni licenza qui riportata nel file LICENSE del repository corrispondente, perché una quantità sorprendente di ciò che si legge sulle licenze YOLO, comprese le guide meglio posizionate nei risultati di ricerca, è sbagliata.

Se ti interessa solo la questione della AGPL di Ultralytics, la analizziamo in dettaglio in [YOLO è gratuito per uso commerciale?](/articles/yolo-commercial-license). Qui trovi la mappa completa.

**Due dichiarazioni prima di iniziare.** Primo, gestiamo LibreYOLO, una libreria con licenza MIT che compete con diversi progetti descritti qui sotto, e l'ultima sezione di questo articolo parla di LibreYOLO. È un motivo per verificare quanto scriviamo, non per fidarsi sulla parola: per questo ogni affermazione sulle licenze rimanda alla fonte primaria. Secondo, questo articolo riporta informazioni generali sui testi delle licenze pubblicati e sulle dichiarazioni pubbliche, aggiornate alla data indicata sopra. Non è consulenza legale e non sostituisce il parere di un avvocato nella tua giurisdizione. Le licenze e le posizioni dei maintainer cambiano. Prima di decidere cosa distribuire, leggi il file LICENSE corrente di ogni repository.

## La tabella delle licenze

Per "opzione permissiva" intendiamo un repository che implementa il modello e che puoi usare in un prodotto commerciale a codice chiuso, senza rendere open source la tua applicazione e senza pagare una licenza.

| Modello | Anno | Codice originale | Opzione permissiva | Paper |
| --- | --- | --- | --- | --- |
| YOLOv1 | 2015 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (pubblico dominio) | L'originale stesso | [arXiv](https://arxiv.org/abs/1506.02640) |
| YOLOv2 | 2016 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (pubblico dominio) | L'originale stesso | [arXiv](https://arxiv.org/abs/1612.08242) |
| YOLOv3 | 2018 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (pubblico dominio) | L'originale; attenzione al port PyTorch AGPL | [arXiv](https://arxiv.org/abs/1804.02767) |
| YOLOv4 | 2020 | [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) (pubblico dominio) | L'originale; [pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) (Apache-2.0) | [arXiv](https://arxiv.org/abs/2004.10934) |
| YOLOv5 | 2020 | [ultralytics/yolov5](https://github.com/ultralytics/yolov5) (AGPL-3.0) | Nessuna | nessun paper |
| YOLOv6 | 2022 | [meituan/YOLOv6](https://github.com/meituan/YOLOv6) (GPL-3.0) | Nessuna | [arXiv](https://arxiv.org/abs/2209.02976) |
| YOLOv7 | 2022 | [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) (GPL-3.0) | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (MIT, stesso laboratorio) | [arXiv](https://arxiv.org/abs/2207.02696) |
| YOLOv8 | 2023 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0; GPL-3.0 al lancio, vedi sotto) | Nessuna (il port Keras è stato abbandonato, vedi sotto) | nessun paper |
| YOLOv9 | 2024 | [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) (GPL-3.0) | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (MIT, stesso laboratorio) | [arXiv](https://arxiv.org/abs/2402.13616) |
| YOLOv10 | 2024 | [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10) (AGPL-3.0) | Nessuna | [arXiv](https://arxiv.org/abs/2405.14458) |
| YOLO11 | 2024 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | Nessuna | nessun paper |
| YOLOv12 | 2025 | [sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12) (AGPL-3.0) | Nessuna | [arXiv](https://arxiv.org/abs/2502.12524) |
| YOLOv13 | 2025 | [iMoonLab/yolov13](https://github.com/iMoonLab/yolov13) (AGPL-3.0) | Nessuna | [arXiv](https://arxiv.org/abs/2506.17733) |
| YOLO26 | 2026 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | Nessuna | [arXiv](https://arxiv.org/abs/2606.03748) |
| YOLOX | 2021 | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0) | L'originale stesso | [arXiv](https://arxiv.org/abs/2107.08430) |
| PP-YOLOE | 2022 | [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) (Apache-2.0) | L'originale, da PaddleDetection (non PaddleYOLO) | [arXiv](https://arxiv.org/abs/2203.16250) |
| YOLO-NAS | 2023 | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) (codice Apache-2.0, pesi non commerciali) | Codice sì, pesi ufficiali no | nessun paper |
| YOLO-World | 2024 | [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World) (GPL-3.0) | Nessuna | [arXiv](https://arxiv.org/abs/2401.17270) |
| YOLOE | 2025 | [THU-MIG/yoloe](https://github.com/THU-MIG/yoloe) (AGPL-3.0) | Nessuna | [arXiv](https://arxiv.org/abs/2503.07465) |
| MMYOLO | 2022 | [open-mmlab/mmyolo](https://github.com/open-mmlab/mmyolo) (GPL-3.0, fermo dal 2024) | Nessuna | nessun paper |

Leggendo la tabella colonna per colonna, una cosa diventa evidente: il numero di versione non dice nulla sulla licenza. Lo dice il repository.

## Stesso modello, due licenze: il caso YOLOv9

YOLOv9 dimostra nel modo più chiaro che chiedere "qual è la licenza di YOLOvN?" è la domanda sbagliata. Vale la pena ricostruire la cronologia con precisione perché è l'unico caso nella storia di YOLO in cui la pressione della community ha davvero cambiato l'esito.

- **18 febbraio 2024:** [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) viene pubblicato insieme al [paper](https://arxiv.org/abs/2402.13616).
- **22 febbraio 2024:** un utente apre la issue #10 per chiedere quale sia la licenza. Chien-Yao Wang (WongKinYiu) risponde "I think it should be GPL3" e aggiunge il file GPL-3.0 quattro giorni dopo.
- **26 febbraio 2024:** un secondo utente apre la [issue #82, "An Apache/MIT rewrite"](https://github.com/WongKinYiu/yolov9/issues/82). Nelle due settimane e mezzo successive intervengono una dozzina di utenti commerciali.
- **14 marzo 2024:** Wang scrive nella discussione: *"Okay I create a new repo for mit rewrite... I think I can handle most of implementation of architectures and loss functions. And need someone to give great help about dataloader and ddp training."*

Quel repository, inizialmente chiamato `yolov9mit`, oggi è **[MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO)**: ha licenza MIT, copyright "Kin-Yiu, Wong and Hao-Tang, Tsui" e si descrive come *"the official implementation of YOLOv7 and YOLOv9, YOLO-RD."* La maggior parte della riscrittura non è stata realizzata dai volontari intervenuti nella discussione, ma da Hao-Tang Tsui, collega di Wang, che ha scritto circa il 90 percento dei commit.

Quindi YOLOv9 è allo stesso tempo "non distribuibile" e "distribuibile", a seconda del repository che cloni. Stessa architettura, stesso laboratorio, due licenze.

**Prima di basarci un prodotto, considera tre aspetti che il repository non mette in evidenza nella sua pagina principale:**

1. **È ancora in fase di maturazione.** Una [issue aperta](https://github.com/MultimediaTechLab/YOLO/issues/231) di febbraio 2026 segnala che i checkpoint distribuiti non riproducono i risultati COCO del paper e hanno sensibilmente più parametri di quelli dichiarati nel paper. L'addestramento per segmentazione e keypoint [non è ancora implementato](https://github.com/MultimediaTechLab/YOLO/issues/232). Dopo il tag v1.0 di dicembre 2025 non è stato integrato nulla nel branch main.
2. **I pesi non hanno una dichiarazione di licenza separata.** Per convenzione, la LICENSE MIT del repository copre i suoi asset di release e gli indizi tecnici suggeriscono che i checkpoint siano stati addestrati da questo progetto, anziché copiati dal repository GPL: i file differiscono per dimensioni e numero di parametri dai checkpoint del repository GPL. Ma nessun documento dichiara che i file .pt siano MIT e una [richiesta della community per una verifica formale di eventuali contaminazioni AGPL](https://github.com/MultimediaTechLab/YOLO/issues/51) è ancora aperta. È una lacuna nella documentazione: nessuno ha dichiarato che siano "verificati e privi di contaminazioni" e nessuno ha segnalato un problema.
3. **È una codebase diversa**, non un sostituto compatibile pronto all'uso degli script del repository GPL.

Questo non significa che sia inutilizzabile. Significa che va valutato come progetto, non trattato come una casella da spuntare.

## Era 1: gli anni del pubblico dominio (da YOLOv1 a YOLOv4)

Joseph Redmon ha pubblicato [YOLOv1](https://arxiv.org/abs/1506.02640), [YOLOv2](https://arxiv.org/abs/1612.08242) (pubblicato come paper YOLO9000) e [YOLOv3](https://arxiv.org/abs/1804.02767) nel framework Darknet. Il file [LICENSE](https://github.com/pjreddie/darknet/blob/master/LICENSE) è famoso per le sue tre righe:

> 0. Darknet is public domain.
> 1. Do whatever you want with it.
> 2. Stop emailing me about it!

È una vera dichiarazione di pubblico dominio. [YOLOv4](https://arxiv.org/abs/2004.10934), mantenuto da Alexey Bochkovskiy in un [fork di Darknet](https://github.com/AlexeyAB/darknet), contiene lo *stesso* file, non la Unlicense che diverse guide sulle licenze gli attribuiscono. Il classificatore di GitHub rinuncia a identificarlo e riporta "Other". È utile saperlo se i tuoi strumenti di compliance leggono quel campo: uno scanner lo segnalerà come licenza non identificata e non come licenza permissiva, anche se il testo è difficilmente più permissivo.

Anche Darknet non è del tutto abbandonato. Il repository di Redmon è inattivo dal 2022, ma il README di AlexeyAB ora indica [hank-ai/darknet](https://github.com/hank-ai/darknet) come successore C/C++ consigliato e mantenuto attivamente.

**La trappola sono i port.** Nel 2026 quasi nessuno addestra in Darknet; si tende a usare una reimplementazione PyTorch, che ha una propria licenza:

- [ultralytics/yolov3](https://github.com/ultralytics/yolov3), il port YOLOv3 più popolare, è **AGPL-3.0**. Un'architettura di pubblico dominio ridistribuita con la licenza copyleft più restrittiva tra quelle comuni in questo ambito. La licenza che ottieni dipende interamente dal port che hai installato con pip.
- [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) è **GPL-3.0**.
- [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) è **Apache-2.0**, quindi per v4 esiste ancora un'opzione permissiva in PyTorch.
- [WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4), realizzato dal coautore di YOLOv4, non contiene **alcun file LICENSE**. È peggio del copyleft: senza licenza non vengono concessi diritti e vale la regola predefinita che tutti i diritti sono riservati.

Controlla la licenza del repository che hai clonato, non il numero di versione del paper.

## Era 2: GPL-3.0 (YOLOv6, v7 e v9)

[YOLOv6](https://github.com/meituan/YOLOv6) (Meituan), [YOLOv7](https://github.com/WongKinYiu/yolov7) e [YOLOv9](https://github.com/WongKinYiu/yolov9) sono stati pubblicati con licenza GPL-3.0.

La GPL-3.0 è copyleft: se distribuisci una versione modificata o combini codice GPL e il tuo codice in un unico programma, devi fornire il codice sorgente completo corrispondente dell'opera combinata sotto GPL-3.0. Ci sono due limiti da tenere presenti. **La mera aggregazione** indica programmi separati e indipendenti distribuiti insieme, che non vengono combinati automaticamente. Inoltre, l'obbligo scatta con la **distribuzione**: a differenza della AGPL, se non distribuisci mai l'opera combinata, la GPL semplice non impone di divulgare il codice. Per questo alcune aziende tengono un modello GPL strettamente dietro la propria API.

Questa strada è più limitata di quanto sembri e presenta tre problemi che raramente si considerano in anticipo. Quando il modello arriva sul dispositivo di un cliente o in un'installazione on-prem, lo stai distribuendo. L'espressione "solo uso interno" non è più vera quando una copia esce dalla tua entità legale: consegnare una build a un consulente, a un team di sviluppo esterno o a una società affiliata con personalità giuridica distinta costituisce distribuzione, mentre copiare il software all'interno della stessa azienda no. E questa protezione regge solo se *tutto* ciò che è raggiungibile tramite quell'API è GPL o più permissivo, perché un singolo componente AGPL in qualsiasi punto del lavoro fornito tramite il servizio fa scattare la Sezione 13 per l'intero lavoro, indipendentemente da quanto dichiari il file GPL.

**YOLOv7 e YOLOv9 hanno l'opzione MIT descritta sopra.** YOLOv6 no. La prova più forte viene da Baidu: PaddleDetection è Apache-2.0 e i suoi maintainer hanno isolato intenzionalmente YOLOv5, YOLOv6, YOLOv7 e YOLOv8 in un repository GPL-3.0 separato, [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO), dichiarando che il codice di quei modelli "will not be merged into PaddleDetection." Quando un'azienda grande come Baidu crea una barriera per tenere YOLOv6 fuori dal proprio framework permissivo, capisci quanto valga quella licenza.

## Era 3: AGPL-3.0 (YOLOv5, v8, v10, 11, v12, v13, 26 e YOLOE)

Ultralytics è passata alla AGPL-3.0 il **14 aprile 2023**, non nel 2022, come ripetono moltissime fonti, comprese le guide che si trovano in prima pagina per questa domanda. Tutto ciò che è arrivato dopo ([YOLOv8](https://github.com/ultralytics/ultralytics), YOLO11 e [YOLO26](https://arxiv.org/abs/2606.03748)) è AGPL-3.0, con una licenza Enterprise a pagamento come alternativa. La [pagina delle licenze](https://ultralytics.com/license) dichiara che il livello Enterprise copre "YOLO26, earlier YOLO versions, and any future YOLO models", quindi la politica è intenzionale e guarda al futuro. I prezzi non sono pubblici.

La data si può verificare e vale la pena farlo, perché la versione diffusa di questa storia è sbagliata su un punto importante:

- Nella sua cronologia, il file `LICENSE` di `ultralytics/yolov5` è stato modificato esattamente tre volte. Il terzo intervento è il commit [`34cf749`](https://github.com/ultralytics/yolov5/commit/34cf749958d2dd3ed1205f6bb07e0f20f6e2372d), "Update LICENSE to AGPL-3.0 (#11359)", datato **2023-04-14**, che sostituisce `GNU GENERAL PUBLIC LICENSE` con `GNU AFFERO GENERAL PUBLIC LICENSE` in 101 file.
- L'ultima release di YOLOv5 del 2022, **v7.0 (22 novembre 2022), era ancora GPL-3.0**, come le versioni v6.2, v6.1 e v6.0 precedenti. Lo stesso vale per le migliaia di fork che hanno smesso di sincronizzarsi prima di aprile 2023: se ne scegli uno qualsiasi, GitHub riporta ancora GPL-3.0.
- Lo stesso giorno, 41 minuti dopo, anche `ultralytics/ultralytics` ha ricevuto [la stessa modifica](https://github.com/ultralytics/ultralytics/commit/2c6fc0a4443b9cf805ef17b1cfdd71a98693b4d4) (PR #2031).

Ecco il fatto che quasi nessuno conosce: **YOLOv8 non è stato lanciato sotto AGPL.** È uscito a gennaio 2023 sotto **GPL-3.0** e ha cambiato licenza tre mesi dopo. Il record PyPI non lascia dubbi: `ultralytics` dalla versione 8.0.0 (10 gennaio 2023) alla 8.0.76 (13 aprile 2023) dichiara sempre `GPL-3.0`. La versione 8.0.80, caricata il 16 aprile 2023, è la prima a dichiarare `AGPL-3.0`.

Non è una scappatoia. Una licenza già concessa non viene revocata retroattivamente, quindi quelle vecchie versioni restano disponibili alle condizioni con cui sono state pubblicate. Ma sono vecchie di tre anni, non sono più mantenute né aggiornate con patch, la GPL-3.0 resta copyleft (hai sostituito una clausola di rete con una clausola di distribuzione, senza uscire dal copyleft) e la posizione commerciale di Ultralytics è comunque più ampia del testo della licenza. La lezione utile è più circoscritta e generale: **la licenza si applica alla versione che hai ricevuto, non al nome del progetto.** Blocca le dipendenze a una versione e registra cosa diceva la licenza il giorno in cui hai acquisito la copia: il progetto può cambiarla domani e la tua documentazione di compliance dipende dalla versione che hai effettivamente ottenuto.

La AGPL-3.0 è la GPL più la Sezione 13, la clausola di rete: *se modifichi il Programma*, agli utenti che interagiscono in rete con la tua versione modificata deve essere offerto il relativo sorgente. Questo chiude la possibilità di usare il software come servizio che la GPL semplice lascia aperta.

Nota la precisazione, spesso omessa negli articoli. La Sezione 0 della AGPL definisce "modificare" come copiare o adattare l'opera *in un modo che richiede il permesso sul copyright*. Eseguire codice di addestramento non modificato sul tuo dataset significa usare il programma, non adattarne il sorgente, proprio come compilare il tuo codice con un compilatore GPL non modificato non modifica il compilatore. Quindi il solo addestramento non fa scattare automaticamente l'obbligo; chi te lo dice sta ignorando la definizione.

Rientri nell'ambito della licenza quando combini o incorpori il pacchetto nella tua codebase in modo che i due vengano distribuiti o forniti come un unico programma. Secondo l'interpretazione della FSF, collegare un'opera GPL o AGPL ad altri moduli crea un'opera combinata; non è mai stato verificato in tribunale se questo principio si estenda fino a un import Python. Inoltre, le condizioni commerciali di Ultralytics richiedono una licenza Enterprise per piattaforme SaaS, API e sistemi cloud che usano YOLO dietro le quinte, come indicato nella loro [pagina delle licenze](https://ultralytics.com/license). La posizione del fornitore è quindi più ampia del testo della licenza in sé. Considera rischiosa l'integrazione in un prodotto che distribuisci o fornisci come servizio; non dare per scontato che il solo addestramento isolato abbia lo stesso effetto.

Per YOLO26, le date sono confuse ovunque, quindi ecco la cronologia: è stato **presentato in anteprima a settembre 2025** durante YOLO Vision, **rilasciato effettivamente a gennaio 2026** (release `ultralytics` 8.4.0, le cui note dicono "Ultralytics YOLO26 has arrived") e il **paper è stato pubblicato a giugno 2026**. Il modello si trova nel repository principale `ultralytics/ultralytics`; `ultralytics/yolo26` è solo una pagina di presentazione.

**Ecco la sorpresa:** anche i modelli YOLO accademici degli ultimi due anni sono AGPL, e non perché li abbia scritti Ultralytics.

- [YOLOv10](https://arxiv.org/abs/2405.14458) viene dalla Tsinghua University. Il suo README dice *"The code base is built with ultralytics."*
- [YOLOv12](https://arxiv.org/abs/2502.12524): *"The code is based on ultralytics."*
- [YOLOv13](https://arxiv.org/abs/2506.17733): *"The code is based on Ultralytics."*
- [YOLOE](https://arxiv.org/abs/2503.07465), il modello a vocabolario aperto "see anything" del gruppo YOLOv10, è anch'esso AGPL-3.0 e deriva da Ultralytics.

Il copyleft si trasmette: un'opera derivata da codice AGPL deve essere distribuita sotto AGPL, quindi quando uno di questi fork viene distribuito o fornito come servizio, la licenza si applica anche a esso. (Se lo modifichi e lo tieni solo per te, non scatta alcun obbligo, perciò l'uso nella ricerca non è coinvolto.) La ricerca è indipendente, la licenza no. Dal 2024 pubblicare "il prossimo YOLO" come derivato di Ultralytics è diventato il flusso di lavoro predefinito, e la AGPL si propaga automaticamente ai numeri di versione successivi.

Se trovi una pagina che dichiara che YOLOv13 è Apache-2.0, è sbagliata. Il file LICENSE, l'API di GitHub e la scheda del modello indicano tutti AGPL-3.0.

### L'opzione permissiva per YOLOv8 che non esiste più

Molte guide ancora online, inclusa una bozza precedente di questo articolo, indicano YOLOv8 di KerasCV con licenza Apache-2.0 come alternativa pulita alla AGPL di YOLOv8. **Oggi non puoi farci affidamento.** Ecco cosa risulta dalle fonti pubbliche:

- Nella discussione KerasCV [Clarifications regarding YOLOv8 licensing](https://github.com/keras-team/keras-cv/discussions/2032), un collaboratore ha scritto che "many parts are derived form ultralytics." Nessun maintainer ha risposto, quindi la questione di quanto fosse indipendente l'implementazione non è mai stata chiarita pubblicamente.
- La issue KerasHub [add YOLOV8](https://github.com/keras-team/keras-hub/issues/1760) è stata chiusa a luglio 2025 da un maintainer Keras, che ha scritto: "Hey all!! Because of license issues, we have decided to drop this." In seguito, due persone hanno chiesto cosa significasse per chi già usava la versione KerasCV in prodotti commerciali. Nessuno ha risposto.
- KerasCV è ora archiviato e KerasHub, il suo successore mantenuto, **non distribuisce YOLOv8**.

Non sappiamo se queste due circostanze siano collegate e non sosteniamo che sia successo qualcosa di scorretto. Non sono basi solide su cui costruire un prodotto.

## Le eccezioni permissive: YOLOX, PP-YOLOE e YOLO-NAS

Non tutti i modelli YOLO hanno seguito la strada del copyleft.

- **[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)** (Megvii, 2021) è Apache-2.0 senza eccezioni, sia per il codice sia per i pesi. È ancora il modello YOLO classico più semplice da usare in ambito commerciale, anche se il repository non riceve commit dalla metà del 2025 e una [domanda sulla distribuzione dei pesi preaddestrati in un'app commerciale](https://github.com/Megvii-BaseDetection/YOLOX/issues/1865) è aperta da marzo 2026 senza risposta. Il testo della licenza è chiaro. Al momento, però, non c'è nessuno che risponda alle domande. Abbiamo scritto di [come usare YOLOX con LibreYOLO](/articles/yolox-with-libreyolo).
- **[PP-YOLOE](https://arxiv.org/abs/2203.16250)** (Baidu) è Apache-2.0 **all'interno di PaddleDetection**. Prendilo da lì, non da PaddleYOLO, che contiene una cartella di configurazione quasi identica ed è GPL-3.0. Stesso modello, stessa azienda, due repository, due licenze. È il caso YOLOv9 al contrario.
- **[YOLO-NAS](https://github.com/Deci-AI/super-gradients)** (Deci, 2023) è quello che tende a trarre in inganno. Il codice è Apache-2.0, i pesi ufficiali no. La [licenza separata](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md) stabilisce che *"may not use the Software for any commercial use, including in connection with any models used in a production environment."* Da quando NVIDIA ha acquisito Deci nel 2024, non abbiamo trovato nuovo lavoro sulle funzionalità nel repository e gli URL originali dei pesi non funzionano più (i checkpoint sono ora ospitati altrove). A volte si suggerisce come soluzione alternativa che addestrare l'architettura da zero permetta di restare coperti da Apache, ma Deci e NVIDIA non lo hanno confermato e l'idea si concilia male con il testo della licenza, che si applica a "any components comprising the model". Altre informazioni su YOLO-NAS [qui](/articles/yolo-nas-with-libreyolo).
- **[MMYOLO](https://github.com/open-mmlab/mmyolo)** viene talvolta indicato come Apache-2.0 nelle guide più vecchie. Non è così: MMYOLO è GPL-3.0 (il suo progetto affine MMDetection è quello Apache) e non riceve commit da luglio 2024.
- **[YOLO-World](https://github.com/AILab-CVC/YOLO-World)** (Tencent) è **GPL-3.0**, non AGPL e non permissivo. Vale la pena dirlo perché spesso si presume che i modelli YOLO a vocabolario aperto siano permissivi per associazione con l'ecosistema in stile CLIP da cui prendono spunto.

Questi casi coprono i tre errori più frequenti sulle licenze in questo ambito: presumere che i pesi abbiano la stessa licenza del codice, che tutto ciò che proviene dalla stessa organizzazione condivida una licenza e che la licenza di un modello si possa dedurre dal suo nome.

## Le licenze coprono il codice, non le idee (ma attenzione ai brevetti)

Un principio giuridico è alla base di ogni opzione permissiva descritta sopra: **il copyright protegge l'espressione, non le idee.** Non è uno slogan, ma una norma consolidata su entrambe le sponde dell'Atlantico.

Negli Stati Uniti è [17 U.S.C. 102(b)](https://www.law.cornell.edu/uscode/text/17/102), che esclude dalla protezione del copyright qualsiasi "idea, procedura, processo, sistema, metodo operativo, concetto, principio o scoperta", secondo un principio che risale a *Baker v. Selden* (1879). Nell'UE, la cui legge si applica a noi e probabilmente a molti lettori, l'articolo 1(2) della [direttiva sul software (2009/24/EC)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32009L0024) afferma che le "idee e i principi alla base di qualsiasi elemento di un programma per computer" non sono protetti. La Corte di giustizia ha confermato in *SAS Institute v World Programming* ([C-406/10](https://curia.europa.eu/juris/liste.jsf?num=C-406/10), 2012) che le funzionalità di un programma, il suo linguaggio di programmazione e i suoi formati dati non sono espressione protetta. Lo è soltanto il codice.

Un'architettura descritta in un paper arXiv è un'idea pubblicata. Chiunque può implementarla. Non puoi copiare o adattare i file sorgente GPL/AGPL di qualcuno e riassegnarli a un'altra licenza. Un'implementazione indipendente, scritta da persone che partono dal paper e non dal sorgente, è una nuova opera e il suo autore può scegliere la licenza.

Due precisazioni che molti fornitori tralasciano:

- **L'indipendenza deve essere reale.** Il metodo difendibile è un processo clean-room: chi scrive il codice lavora partendo dal paper e da una specifica, non dal sorgente copyleft. Leggere attentamente il repository GPL e poi "riscriverlo" non equivale a questo, e la differenza conta se qualcuno dovesse esaminare il lavoro.
- **Questo è un argomento sul copyright, non sui brevetti.** Un'invenzione indipendente *non* è una difesa contro un brevetto. Una reimplementazione clean-room risolve la questione del copyright e lascia invariata quella dei brevetti. Oggi non c'è un brevetto ampiamente rivendicato sulle architetture YOLO, ma "nessuno ha ancora fatto causa" non significa "non c'è nulla per cui fare causa".

## I pesi hanno una seconda licenza

L'errore più costoso in questo ambito è controllare il codice e dimenticare il checkpoint. Ci sono tre livelli soggetti a licenza e possono avere condizioni diverse:

**1. Il codice.** Tutto ciò che abbiamo visto finora.

**2. I pesi.** A volte la licenza è esplicita, come per YOLO-NAS. Altre volte viene dichiarata: la [pagina delle licenze](https://ultralytics.com/license) di Ultralytics afferma che la AGPL-3.0 "covers the training code and the models produced by that training code" e applica questa condizione anche ai modelli che addestri tu usando il loro codice. In quanto titolare del copyright di ciò che pubblica, l'azienda può stabilire i termini dei propri checkpoint. È una questione ancora aperta se i pesi che *produci tu* siano giuridicamente un'opera derivata dal codice di addestramento: non esistono precedenti giudiziari e questa interpretazione contrasta con la regola generale della FSF, secondo cui l'output di un programma non è automaticamente coperto dal copyright del programma. Il fatto che la questione non sia risolta non significa che sia sicura. In pratica, considera soggetti a vincoli i pesi AGPL pubblicati.

Il punto più delicato per chi segue i consigli di questo articolo: **i repository GPL di YOLOv7 e YOLOv9 non dicono nulla sui propri pesi.** I loro checkpoint sono asset di release in un repository GPL-3.0 senza una concessione separata. Se carichi quei file .pt in una reimplementazione MIT, ottieni codice pulito, ma lo stato del checkpoint resta da chiarire. Se hai cambiato implementazione per la licenza, usa i pesi addestrati dal progetto permissivo oppure addestra i tuoi.

**3. I dati di addestramento.** Le *annotazioni* di COCO sono CC BY 4.0, quindi le etichette sono a posto. Le *immagini*, invece, non sono di COCO e il consorzio non può concedere loro una licenza: sono foto Flickr soggette a una varietà di condizioni individuali, che il consorzio COCO dichiara esplicitamente di non possedere. Gran parte del settore considera questo un rischio basso e va avanti, ma dire "COCO è a posto" si riferisce solo alle annotazioni. Fuori da COCO le condizioni diventano presto più restrittive: molti dataset aerei, medici e di guida non sono commerciali, e la licenza permissiva del codice di un modello non li rende utilizzabili.

Verifica tutti e tre i livelli, ogni volta.

## Una nota su MIT e Apache-2.0

Distribuiamo una libreria con licenza MIT, quindi sarebbe comodo dirti che la MIT è semplicemente la licenza migliore. Un confronto onesto è più interessante.

La MIT non significa zero obblighi: devi conservare l'avviso di copyright e il testo della licenza nelle copie che distribuisci. È una condizione reale, anche se facile da rispettare. Apache-2.0 offre inoltre qualcosa che la MIT non ha: una **concessione esplicita di brevetto** da ogni contributore e una clausola di ritorsione che revoca la concessione a chi fa causa per l'opera. La MIT non si pronuncia sui brevetti. Per un'azienda preoccupata soprattutto dall'esposizione ai brevetti, Apache-2.0 può essere la scelta *più sicura*. Molti progetti permissivi di object detection (YOLOX, RT-DETR, D-FINE, DEIM) usano Apache-2.0 proprio per questo motivo.

La MIT offre semplicità e compatibilità. Nessuna delle due licenze ti obbliga a pubblicare il codice sorgente, che è il criterio di cui tratta davvero questo articolo. Se qualcuno ti dice che la scelta importante è tra MIT e Apache, sta cercando di venderti qualcosa. La scelta importante è tra permissiva e copyleft.

## Quale YOLO puoi usare davvero?

- **Prodotto commerciale a codice chiuso:** YOLOv7 o YOLOv9 tramite il [repository MIT](https://github.com/MultimediaTechLab/YOLO) del laboratorio, YOLOX o PP-YOLOE (da PaddleDetection) tra i modelli originali, oppure un framework MIT mantenuto come LibreYOLO se preferisci non assemblare e verificare tutto da solo. Evita GPL e AGPL a meno che tu non voglia rendere open source l'intera applicazione o acquistare la licenza Enterprise.
- **Progetto open source:** qualsiasi modello, se la sua licenza è compatibile con quella del progetto. Se il tuo progetto è AGPL, la famiglia Ultralytics è una scelta naturale e la ricerca più recente (YOLOv13, YOLO26, YOLOE) arriva lì per prima.
- **Ricerca e benchmark:** le licenze impongono pochi vincoli. Usa ciò che ha usato il paper con cui fai il confronto.
- **Il piano "ci pensiamo più avanti":** non funziona. Rimuovere una dipendenza AGPL quando il prodotto è già costruito significa riaddestrare, rivalidare e riesportare tutto, compresi i pesi. Scegli prima il repository e quindi la licenza.

Per confrontare i framework, invece delle loro licenze, leggi [Le migliori alternative a Ultralytics nel 2026](/articles/best-ultralytics-alternatives).

## L'opzione MIT: cosa distribuisce LibreYOLO

Dichiarazione: LibreYOLO è un nostro progetto e la sua licenza è il motivo per cui esiste questo articolo.

**[LibreYOLO](https://github.com/LibreYOLO/libreyolo) è un'unica codebase con licenza MIT** che include i rilevatori seguenti tramite un'unica API, con addestramento ed esportazione in ONNX, TensorRT, OpenVINO e NCNN integrati. Niente copyleft, nessuna clausola di rete, nessun livello Enterprise.

Ecco l'intera gamma di modelli per il rilevamento di oggetti, con il progetto upstream di ciascuna famiglia e la relativa licenza, così puoi vedere da dove proviene ogni opzione permissiva:

| In LibreYOLO | Modello | Upstream | Licenza upstream |
| --- | --- | --- | --- |
| `LibreYOLO2`, `LibreYOLO3`, `LibreYOLO4` | I modelli YOLO dell'era Darknet, in PyTorch | [pjreddie/darknet](https://github.com/pjreddie/darknet), [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) | Pubblico dominio |
| `LibreYOLO7` | YOLOv7 | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT (riscrittura del laboratorio stesso, non il repository GPL) |
| `LibreYOLO9`, `LibreYOLO9E2E` | YOLOv9 e una variante end-to-end senza NMS | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT (riscrittura del laboratorio stesso, non il repository GPL) |
| `LibreYOLO9P2` | YOLOv9 con una testa stride-4 per oggetti piccoli | Originale LibreYOLO | MIT |
| `LibreYOLOX` | YOLOX | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache-2.0 |
| `LibreYOLONAS` | YOLO-NAS, rilevamento e posa | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) | Codice Apache-2.0. I pesi upstream soggetti a restrizioni **non** vengono ridistribuiti |
| `LibreRTDETR`, `LibreRTDETRv2` | RT-DETR e v2 | [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | Apache-2.0 |
| `LibreRTDETRv4` | RT-DETRv4 | [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | Apache-2.0 |
| `LibreRFDETR` | RF-DETR, rilevamento, segmentazione, posa, OBB | [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | Apache-2.0 per N/S/M/L. XL/2XL sono soggetti alla Platform Model License di Roboflow, quindi distribuiamo solo le dimensioni Apache |
| `LibreDFINE` | D-FINE | [Peterande/D-FINE](https://github.com/Peterande/D-FINE) | Apache-2.0 |
| `LibreDEIM` | DEIM | [Intellindust-AI-Lab/DEIM](https://github.com/Intellindust-AI-Lab/DEIM) | Apache-2.0 |
| `LibreDEIMv2` | DEIMv2 | [Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) | Codice Apache-2.0. Le versioni più grandi usano il backbone DINOv3 di Meta, soggetto alla licenza non OSI di Meta, che consente la ridistribuzione ma **vieta gli usi militari, nucleari, di spionaggio e relativi alle armi**. Queste versioni sono pubblicate come `other`, non Apache. Leggi la licenza prima di distribuire un modello per uso duale |
| `LibreRTMDet` | RTMDet ([senza MMDetection](/articles/rtmdet-without-mmdetection)) | [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) | Apache-2.0 (il progetto affine Apache, non mmyolo GPL-3.0) |
| `LibrePICODET` | PP-PicoDet | Architettura da [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), checkpoint tramite il port [Picodet_Pytorch](https://github.com/Bo396543018/Picodet_Pytorch) | Apache-2.0 (entrambi) |
| `LibreEC` | EdgeCrafter, rilevamento, posa, segmentazione | [Intellindust-AI-Lab/EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter) | Apache-2.0 |
| `LibreFOMO` | Rilevatore di centroidi per hardware della classe dei microcontrollori | Originale LibreYOLO | MIT |
| `LibreGroundingDINO` | Grounding DINO, vocabolario aperto (inferenza) | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | Apache-2.0 |
| `LibreOWLv2` | OWLv2, vocabolario aperto (inferenza) | Google, tramite `transformers` | Apache-2.0 |

Quella colonna mette volutamente in chiaro due aspetti.

**Nella codebase non viene copiato codice GPL o AGPL.** Ogni famiglia qui sopra deriva da un progetto upstream di pubblico dominio o con licenza MIT o Apache-2.0, condizione che mantiene il framework sotto MIT. Quando un'architettura ha sia una versione permissiva sia una copyleft, partiamo da quella permissiva: YOLOv9 e YOLOv7 derivano dal repository MIT del laboratorio, non dagli originali GPL; RTMDet deriva da MMDetection con licenza Apache, non da MMYOLO con licenza GPL. YOLO-World non è incluso, anche se ce lo chiedono spesso, perché è GPL-3.0 e non esiste una fonte permissiva da cui partire.

**Permissivo non significa senza restrizioni, ed è meglio dirlo ora che fartelo scoprire durante un audit.** Le versioni più grandi di DEIMv2 ereditano i termini di DINOv3 di Meta, compreso il divieto di uso militare, nucleare, di spionaggio e relativo alle armi, un vincolo concreto se lavori in un settore vicino al dual use. Alcuni checkpoint di anteprima per la ricerca sono addestrati su dataset non commerciali (DOTA, VisDrone) e su Hugging Face sono contrassegnati `cc-by-nc`, invece di essere distribuiti senza indicazioni come se fossero liberi da vincoli. E la cautela che applichiamo ai pesi degli altri vale anche per i nostri: i checkpoint YOLO9 sono convertiti dai checkpoint del repository MIT del laboratorio, quindi ereditano le condizioni applicabili a quei pesi, che come detto sono "dichiarate, ma non verificate formalmente". La licenza MIT copre il codice che abbiamo scritto. Ogni checkpoint viene pubblicato con le condizioni relative ai dati e alla provenienza dell'addestramento.

Oltre al rilevamento, la stessa API supporta segmentazione di istanze e semantica, posa, box orientati, classificazione (MobileNetV4, ConvNeXt, EfficientNetV2, ResNet, CLIP), stima monoculare della profondità, ripristino delle immagini e stima dello sguardo.

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
```

Lo stesso flusso di lavoro YOLO, senza tutte le verifiche sulle licenze.

Aggiungilo ai preferiti su GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentazione: [libreyolo.com/docs](https://libreyolo.com/docs)

---

*Marchi e affiliazione: YOLO, YOLOv8, YOLO11, YOLO26 e Ultralytics sono marchi dei rispettivi titolari, tra cui Ultralytics Inc. LibreYOLO è un progetto indipendente e non è affiliato, sponsorizzato né approvato da Ultralytics o da qualsiasi altra organizzazione citata in questo articolo. I nomi di prodotti e repository sono usati solo per identificare e confrontare il software discusso.*

*Questo articolo contiene informazioni generali, non costituisce consulenza legale, è aggiornato all'11 luglio 2026 ed è scritto da uno dei fornitori inclusi nel confronto. Verifica la licenza corrente prima di distribuire il software.*
