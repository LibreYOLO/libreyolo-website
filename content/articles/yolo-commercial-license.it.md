---
title: "YOLO è gratis per uso commerciale? Licenze YOLOv8, YOLO11 e YOLO26"
description: "YOLOv5, YOLOv8, YOLO11 e YOLO26 sono distribuiti con licenza AGPL-3.0, quindi non sono gratuiti per l'uso commerciale con codice chiuso. Ecco cosa richiede davvero la licenza, versione per versione, e qual è l'alternativa con licenza MIT."
date: 2026-07-04
author: Xuban
tags: [LibreYOLO, yolo-license, agpl, commercial-use, yolov8, yolo11, yolo26, mit-license]
faq:
  - q: "YOLOv8 è gratis per uso commerciale?"
    a: "Non per prodotti con codice chiuso. YOLOv8 è distribuito con licenza AGPL-3.0, quindi devi pubblicare il codice sorgente dell'intera applicazione con licenza AGPL-3.0 oppure acquistare la Enterprise License a pagamento del fornitore. Per un uso commerciale gratuito e senza restrizioni, usa una libreria con licenza MIT come LibreYOLO."
  - q: "Con quale licenza è distribuito YOLO?"
    a: "Le release principali, YOLOv5, YOLOv8, YOLO11 e YOLO26, sono distribuite per impostazione predefinita con licenza AGPL-3.0. AGPL-3.0 è una licenza copyleft forte: è open source, ma non è gratuita da usare all'interno di un prodotto proprietario."
  - q: "Posso usare YOLO in un prodotto commerciale con codice chiuso?"
    a: "Non con AGPL-3.0 senza rispettarne gli obblighi, il che significa rendere open source l'intero prodotto con licenza AGPL-3.0. Per mantenere privato il tuo codice, devi acquistare la Enterprise License del fornitore oppure passare a un framework con licenza permissiva come LibreYOLO (MIT)."
  - q: "Esiste una versione di YOLO senza AGPL?"
    a: "Sì. LibreYOLO è distribuito con licenza MIT, quindi non prevede copyleft AGPL né una clausola sull'uso in rete. Puoi usarlo liberamente in software commerciale e con codice chiuso, senza costi."
  - q: "Quanto costa una licenza commerciale di YOLO?"
    a: "La Enterprise License del fornitore è un accordo a pagamento per singola azienda, con prezzi non pubblici. LibreYOLO non costa nulla: la licenza MIT concede gratuitamente l'uso commerciale."
  - q: "Eseguire YOLO solo sui miei server evita l'AGPL?"
    a: "No. La Sezione 13 della licenza AGPL-3.0 riguarda l'interazione in rete. Se gli utenti accedono alla tua versione modificata tramite una rete, devi offrire loro il codice sorgente corrispondente. Eseguirla solo sui tuoi server non costituisce un'esenzione."
---

**No. I modelli YOLO più diffusi, YOLOv5, YOLOv8, YOLO11 e il nuovo YOLO26, non sono gratuiti per l'uso commerciale con codice chiuso.** Sono distribuiti con licenza **AGPL-3.0**, una licenza copyleft forte. Se realizzi un prodotto basato su uno di essi, devi pubblicare l'intera applicazione con licenza AGPL-3.0 oppure acquistare la Enterprise License a pagamento del fornitore. Se nessuna delle due opzioni fa al caso tuo, esiste un'alternativa con licenza permissiva: **[LibreYOLO](/) è distribuito con licenza MIT ed è gratuito per qualsiasi uso commerciale, senza condizioni aggiuntive.**

Se hai cercato "is YOLOv8 free for commercial use", "YOLO license" o "YOLO without AGPL", questa pagina risponde con precisione, versione per versione.

## Le licenze YOLO in breve

| Modello | Licenza predefinita | Gratuito per uso commerciale con **codice chiuso**? | Cosa devi fare |
| --- | --- | --- | --- |
| YOLOv5 | AGPL-3.0 | No | Pubblicare il codice sorgente dell'intera app oppure acquistare una Enterprise License |
| YOLOv8 | AGPL-3.0 | No | Pubblicare il codice sorgente dell'intera app oppure acquistare una Enterprise License |
| YOLO11 | AGPL-3.0 | No | Pubblicare il codice sorgente dell'intera app oppure acquistare una Enterprise License |
| YOLO26 | AGPL-3.0 | No | Pubblicare il codice sorgente dell'intera app oppure acquistare una Enterprise License |
| **LibreYOLO** | **MIT** | **Sì** | **Niente. Integralo nei prodotti proprietari, gratis.** |

In breve: i modelli YOLO principali sono open source, ma con licenza **AGPL-3.0**, una licenza copyleft che la maggior parte delle aziende non può rispettare in un prodotto proprietario. LibreYOLO offre lo stesso workflow YOLO con licenza **MIT**, che non impone questo obbligo.

## Con quale licenza è distribuito YOLO?

I modelli YOLO più usati, **YOLOv5, YOLOv8, YOLO11 e il più recente YOLO26**, sono distribuiti dal loro fornitore con la **GNU Affero General Public License v3.0 (AGPL-3.0)**.

AGPL-3.0 è una licenza *copyleft forte*. Non è una licenza open source permissiva del tipo "fai quello che vuoi", come MIT o Apache-2.0. Impone un obbligo specifico e di ampia portata, proprio quello che mette in difficoltà gli utenti commerciali.

## YOLOv8 è gratis per uso commerciale?

Non nel senso che intende la maggior parte delle persone. **Puoi** scaricare YOLOv8 e usarlo a fini commerciali, ma solo rispettando AGPL-3.0. In pratica significa che:

- Se distribuisci, rilasci o ospiti un prodotto che include YOLOv8, devi **pubblicare il codice sorgente completo del prodotto**, cioè la tua app, il codice di integrazione, gli script e la configurazione, con licenza AGPL-3.0.
- L'obbligo scatta non solo quando distribuisci software agli utenti, ma anche quando permetti loro di interagirvi **tramite una rete**, in base alla Sezione 13, la clausola caratteristica di AGPL. Un backend SaaS o API non ne è esente.

Per un progetto personale o una ricerca accademica di solito va bene. Per un prodotto commerciale proprietario, pubblicare l'intero codice sorgente è quasi sempre impraticabile. Tutto questo vale allo stesso modo per **YOLO11** e **YOLO26**: stessa licenza, stesso obbligo.

## E YOLOv5, YOLO11 e YOLO26?

Vale lo stesso discorso. Per impostazione predefinita, sono tutti distribuiti con licenza **AGPL-3.0**. La Enterprise License del fornitore copre esplicitamente "YOLO26, earlier YOLO versions, and any future YOLO models", il che indica che la licenza è una scelta intenzionale e coerente per l'intera famiglia. Scegliere una versione più recente non ti dà una licenza più permissiva.

Per completezza: non *tutti* i modelli con "YOLO" nel nome sono distribuiti con AGPL. Alcune varianti della community usano GPL-3.0 o Apache-2.0, e le licenze variano in base all'autore. Ma le release a cui ci si riferisce quando si cerca "YOLOv8" o "YOLO11" sono distribuite con AGPL-3.0.

## Cosa richiede davvero AGPL-3.0, in parole semplici

AGPL-3.0 estende GPL con un'aggiunta fondamentale: la **clausola sull'uso in rete**. Ecco cosa comporta in pratica:

1. **Distribuisci il codice sorgente.** Se distribuisci software che include codice AGPL, devi rendere disponibile, con licenza AGPL-3.0, il *codice sorgente completo corrispondente* dell'opera combinata nel suo insieme.
2. **Anche l'uso in rete conta.** Se gli utenti interagiscono con la tua versione modificata tramite una rete, per esempio un'app web, un'API o un SaaS, devi offrire loro lo stesso codice sorgente. Non esiste una scappatoia del tipo "lo eseguiamo solo sui nostri server".
3. **Il copyleft si estende all'opera derivata.** L'obbligo riguarda l'*intera* opera derivata, non solo i file YOLO. Anche la logica aziendale rientra nell'ambito di AGPL.

Queste sono informazioni generali, non consulenza legale, quindi rivolgiti a un avvocato per il tuo caso specifico. Ma il punto è semplice: **AGPL-3.0 e software commerciale con codice chiuso non vanno d'accordo.**

## L'opzione Enterprise License e il suo limite

Il fornitore offre una **Enterprise License** a pagamento che rimuove gli obblighi AGPL e ti permette di integrare YOLO in un prodotto proprietario senza pubblicare il codice sorgente. È una strada valida e per alcune aziende è quella giusta.

Gli aspetti da considerare:

- **Ha un costo**, una tariffa commerciale ricorrente, negoziata per azienda, con prezzi non pubblici.
- **Dipende dalle condizioni di un solo fornitore**, che possono cambiare e riguardano solo il codice di quel fornitore.
- **Paghi per ottenere il permesso** di usare un'architettura di modello che, alla base, è pubblicata come ricerca.

Se una tariffa ricorrente e il lock-in del fornitore sono accettabili, la Enterprise License funziona. Se preferisci non pagare o evitare del tutto questi obblighi, continua a leggere.

## L'alternativa MIT: LibreYOLO

Trasparenza: LibreYOLO è un nostro progetto. È anche il motivo per cui questa pagina può concludersi con una risposta netta.

**LibreYOLO è un framework YOLO distribuito con la licenza permissiva MIT.** Questo singolo fatto cambia tutto per l'uso commerciale:

- **Gratuito per l'uso commerciale**, anche in prodotti proprietari con codice chiuso.
- **Niente copyleft né clausola sull'uso in rete.** Non devi mai pubblicare il codice sorgente della tua app.
- **Nessun costo per utente o Enterprise.** MIT significa MIT.
- **Lo stesso workflow che già conosci**, quindi la migrazione è semplice.

LibreYOLO è un framework reale e mantenuto, non un wrapper: rilevamento di oggetti, segmentazione, stima della posa, classificazione, profondità e altro, con una sola API familiare, oltre all'addestramento e all'esportazione in ONNX/TensorRT/OpenVINO/NCNN integrati. Il workflow non cambia. Cambia la licenza.

Se stai cercando una risposta più ampia alla domanda "lasciare YOLO con licenza AGPL", abbiamo scritto un confronto più completo dell'ecosistema in [Le migliori alternative a Ultralytics nel 2026](/articles/best-ultralytics-alternatives).

Per le licenze di tutti gli altri YOLO, dagli originali Darknet di pubblico dominio fino a YOLOv9, YOLOv10, YOLOv12, YOLOv13 e YOLO26, consulta [Tutte le licenze YOLO spiegate](/articles/yolo-licenses-explained).

### Perché MIT conta per un'azienda

La licenza MIT ti permette di usare, modificare, integrare e vendere software basato su LibreYOLO **senza l'obbligo di divulgare il codice sorgente** e **senza costi**. È la licenza di gran parte dello stack software moderno proprio perché è adatta all'adozione commerciale. Il prodotto è tuo e non devi nulla.

## Provalo

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

Mantieni l'esperienza di sviluppo YOLO: addestra sui tuoi dati, fai predizioni ed esporta per il deployment. Ti liberi dell'obbligo AGPL e della fattura Enterprise.

LibreYOLO è distribuito con licenza MIT, funziona su Linux, Mac e Windows e gira su GPU, Apple Silicon e CPU standard senza modifiche al codice. Una sola API copre YOLO9, RF-DETR, RTMDet, YOLOX, D-FINE, la linea RT-DETR, segmentazione, stima della posa, profondità e altro, tutto con licenze permissive.

Aggiungi una stella su GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentazione: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
