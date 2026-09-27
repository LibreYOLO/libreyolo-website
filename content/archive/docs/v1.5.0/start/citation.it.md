---
title: Citazione
seo_title: Citare LibreYOLO e gli autori originali
description: >-
  Come citare LibreYOLO in un paper e come citare gli autori della famiglia di
  modelli che hai eseguito. Entrambe le cose vanno nella stessa sezione dei
  metodi.
lead: >-
  Una citazione completa di LibreYOLO ha due parti: la libreria e il lavoro
  pubblicato dietro la famiglia di modelli che ha prodotto il risultato.
keywords:
  - citare libreyolo
  - libreyolo bibtex
  - citation cff
  - come citare un modello
  - citazione computer vision
  - bibtex object detection
last_verified: 1.5.0
source_hash: 0f3f23e4e85e38be
---

## Citare LibreYOLO

Il repository pubblica i propri metadati di citazione come
[`CITATION.cff`](https://github.com/LibreYOLO/libreyolo/blob/release/CITATION.cff),
non come blocco BibTeX. GitHub legge quel file e mostra un pulsante Cite this
repository nella pagina del repository, che a partire da lì genera le voci APA e
BibTeX. Prendi la voce da lì invece di scriverne una a mano.

Il file per intero:

```yaml
cff-version: 1.2.0
message: "If you use LibreYOLO in your research or software, please cite it as below."
title: "LibreYOLO"
type: software
authors:
  - family-names: Ceccon
    given-names: Xuban
  - name: "The LibreYOLO contributors"
license: MIT
url: "https://github.com/LibreYOLO/libreyolo"
repository-code: "https://github.com/LibreYOLO/libreyolo"
```

Non riporta né versione né data di rilascio, di proposito.
[`RELEASING.md`](https://github.com/LibreYOLO/libreyolo/blob/release/RELEASING.md)
dice ai maintainer di non incrementare, datare o rinominare mai `CITATION.cff` o
`.zenodo.json` durante una release, così che ogni citazione finisca su un unico
record invece di disperdersi tra le versioni. Indica nel tuo testo la versione
che hai eseguito, e lascia stare la citazione.

## Citare la famiglia di modelli

LibreYOLO è un port. Eseguire `LibreRFDETRm.pt` significa eseguire RF-DETR, e le
persone che hanno scritto RF-DETR sono quelle che un revisore si aspetta di
vedere citate. Citare solo la libreria attribuisce il loro lavoro al progetto
sbagliato.

Tutto il necessario si trova nella pagina della famiglia. La riga Upstream
nell'intestazione indica il lavoro originale e l'organizzazione che c'è dietro, e
rimanda al paper e al repository sorgente. La sezione Citation più in basso
contiene il BibTeX.

Quel BibTeX è copiato alla lettera dal blocco di citazione degli autori stessi,
di norma la sezione Citation del README originale o un `CITATION.cff`, e viene
mostrato con un link al blocco da cui proviene, così puoi confrontarlo con la
fonte. Non viene mai assemblato a partire dai metadati del paper. Una voce
ricostruita a mano fallisce in modo silenzioso e costoso: un coautore perso, la
sede sbagliata, il tipo di voce sbagliato, un anno che appartiene al preprint.
Anche i preprint vengono accettati, quindi una voce può essere un
`@inproceedings` anche se la versione che hai letto era su arXiv.

Copia il blocco così com'è. Se il tuo stile bibliografico richiede un tipo di
voce diverso, converti la voce invece di riscriverla, e mantieni l'ordine
originale degli autori.

## Cosa serve in una sezione dei metodi

Tre cose rendono un risultato LibreYOLO riproducibile e correttamente attribuito:

- La libreria, citata da `CITATION.cff`, insieme alla versione che hai eseguito.
  `libreyolo version` la stampa, insieme alle versioni di Python, torch e CUDA
  con cui sta girando.
- Il lavoro originale, citato dalla sezione Citation della pagina della famiglia.
- Il nome esatto del file di checkpoint, ad esempio `LibreRFDETRm.pt`. Le
  dimensioni all'interno di una stessa famiglia si comportano in modo diverso, e
  diverse famiglie pubblicano checkpoint addestrati su dataset differenti sotto
  lo stesso prefisso, quindi il nome della famiglia da solo non identifica cosa è
  stato eseguito.

L'attribuzione è anche una clausola di licenza per buona parte di ciò che
LibreYOLO pubblica. Sia Apache-2.0 sia la famiglia CC BY richiedono che l'avviso
viaggi insieme ai pesi che ridistribuisci, un obbligo distinto dal citare un
paper. Vedi [licenze](/docs/licensing) per sapere quali termini si applicano a
quale checkpoint.
