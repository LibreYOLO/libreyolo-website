---
title: libreyolo doctor
seo_title: riferimento del comando libreyolo doctor
description: >-
  Controlla un dataset di rilevamento prima dell'addestramento: gli argomenti
  con i valori predefiniti, le famiglie di controlli che puoi saltare o
  selezionare e i codici di uscita su cui la CI può fallire.
lead: >-
  Esegue un insieme di controlli di salute su un dataset di rilevamento e
  segnala ciò che danneggerebbe un addestramento: file mancanti, etichette
  rotte, immagini corrotte, fuga tra split e sbilanciamento delle classi.
keywords:
  - libreyolo doctor cli
  - controllo salute dataset yolo
  - validare dataset detection
  - data leakage tra split
  - libreyolo doctor strict
last_verified: 1.5.0
meta:
  - label: Comando
    value: libreyolo doctor
    mono: true
  - label: Obbligatorio
    value: data
    mono: true
  - label: Output
    value: >-
      Un report delle segnalazioni su stdout. Esce con 1 quando vengono trovati
      errori
snippets:
  examples:
    - label: Base
      language: bash
      code: >
        # download=true permette al coco8.yaml incluso di scaricare le sue
        immagini se mancano.

        libreyolo doctor coco8.yaml download=true
    - label: 'Passata veloce, senza decodifica delle immagini'
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true fast=true
    - label: Gate di CI su controlli selezionati
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true strict=true json=true \
          only=labels,files,config
source_hash: 79e0ef471d567ea3
---

## Sinossi

```bash
libreyolo doctor <data.yaml> [key=value ...]
```

Il dataset è posizionale, e `data=<path>` è accettato come alternativa. Passarli
entrambi con valori diversi esce con `config_conflict`. Tutto il resto sono
coppie `key=value`, e funziona anche la forma POSIX, quindi `imgsz=1024` e
`--imgsz 1024` sono lo stesso argomento.

## Argomenti

| Argomento | Default | Significato |
|---|---|---|
| `data` | | Posizionale. YAML del dataset in formato di rilevamento YOLO, ad es. `coco8.yaml`. Obbligatorio |
| `imgsz` | `640` | Dimensione dell'immagine di addestramento usata nei controlli basati sui pixel, come quello degli oggetti minuscoli |
| `fast` | `false` | Salta la decodifica delle immagini, il che elimina i controlli su corruzione, duplicati e fuga |
| `skip` | | Id di controllo o famiglie separati da virgole da saltare, ad es. `images,labels.tiny_object` |
| `only` | | Id di controllo o famiglie separati da virgole da eseguire in esclusiva |
| `strict` | `false` | Anche gli avvisi fanno fallire il codice di uscita, per i gate di CI |
| `download` | `false` | Consente il download del dataset da URL se manca. Mai script |
| `json` | `false` | Output JSON su stdout |
| `quiet` | `false` | Silenzia stderr |
| `help_json` | `false` | Stampa lo schema del comando come JSON ed esce |

### Famiglie di controlli

`skip` e `only` accettano sia un id di controllo completo sia un prefisso di
famiglia, quindi `images` seleziona tutti i controlli `images.*`.

| Famiglia | Copre |
|---|---|
| `config` | Il file YAML del dataset in sé: `names` mancante, `nc` rispetto a `names`, split mancanti, `path` non risolvibile, nomi di classe duplicati |
| `files` | L'abbinamento tra immagini ed etichette: etichette mancanti, immagini mancanti, etichette orfane, estensioni non supportate, collisioni tra maiuscole e minuscole |
| `labels` | Il contenuto delle etichette: sintassi, righe di poligono, id di classe fuori intervallo, coordinate fuori intervallo, bounding box degeneri, oggetti minuscoli, box enormi, rapporti d'aspetto estremi, bounding box duplicati, immagini affollate, file identici |
| `images` | I dati dei pixel: file corrotti, orientamento EXIF, modalità di colore insolite, dimensioni minuscole o estreme, immagini uniformi, duplicati esatti e approssimati |
| `splits` | Fuga tra split, esatta e approssimata |
| `balance` | La distribuzione delle classi: classi con zero o poche istanze, sbilanciamento, copertura degli split, proporzione di sfondo, asimmetria tra split |

## Esempi

<code-tabs name="examples" />

## Note

### Codici di uscita

`0` quando non è stato trovato nessun errore, `1` quando una segnalazione è un
errore. Con `strict=true`, anche gli avvisi alzano il codice di uscita a `1`,
che è l'impostazione che vuole un gate di CI.

I problemi d'uso hanno codici propri: `2` per un id di controllo o una famiglia
sconosciuti in `skip` o `only`, `3` quando il dataset non viene trovato e `3`
quando il dataset non ha la forma di un dataset di rilevamento.

### La selezione si risolve prima della scansione

`skip` e `only` vengono risolti contro il registro dei controlli prima che venga
letto qualsiasi cosa dal disco, quindi un errore di battitura fallisce subito
invece che dopo una lunga passata sulle immagini. Un selettore che non
corrisponde a nulla è un errore, e il messaggio elenca le famiglie note.

Se la combinazione di `skip`, `only` e `fast` non lascia nessun controllo da
eseguire, anche questo è un errore invece di una passata silenziosa.

### Download

Il dataset non viene scaricato a meno che `download=true`, e vengono eseguiti
solo download da URL. Uno script di download Python incorporato nello YAML di un
dataset non viene mai eseguito da questo comando, qualunque sia il valore del
flag.

### Ambito

I controlli sono scritti per i dataset di rilevamento. Un dataset le cui
etichette hanno la forma di pose, di segmentazione o di box orientati viene
riconosciuto e rifiutato con `data_invalid` invece di essere valutato con le
regole sbagliate.

### Output

Il report leggibile dalle persone va su stdout, e `json=true` lo sostituisce con
un oggetto strutturato che contiene i conteggi del riepilogo, le statistiche del
dataset, tutte le segnalazioni e l'elenco dei controlli che sono stati saltati.

Correlato: [`libreyolo train`](/docs/cli/train), il comando prima del quale
questo è pensato per essere eseguito.
