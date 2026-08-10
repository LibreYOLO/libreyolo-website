---
title: libreyolo label
seo_title: Riferimento del comando libreyolo label
description: >-
  Avvia lo strumento locale di annotazione dei bounding box: argomenti con i
  valori predefiniti, l'interruttore di assistenza IA e che cosa espone il
  binding su un'interfaccia di rete.
lead: >-
  Avvia uno strumento web locale per disegnare e modificare i bounding box.
  Scrive file di etichette nel formato nativo di LibreYOLO, quindi un dataset
  annotato qui si addestra senza alcun passaggio di conversione.
keywords:
  - libreyolo label cli
  - strumento annotazione bounding box
  - etichettare dataset yolo
  - auto labeling cli
  - condividere libreyolo label
last_verified: 1.5.0
meta:
  - label: Comando
    value: libreyolo label
    mono: true
  - label: Output
    value: >-
      Un URL del server su stdout; le etichette vengono scritte come
      labels/*.txt accanto alle immagini
snippets:
  examples:
    - label: Base
      language: bash
      code: |
        # Apre la home del progetto; scegli o crea un dataset nel browser.
        libreyolo label
    - label: 'Solo manuale, porta fissa'
      language: bash
      code: |
        libreyolo label no_assist=true port=9200 no_browser=true
    - label: Far entrare i colleghi
      language: bash
      code: |
        libreyolo label share=true
source_hash: bddad245877793b1
---

## Sinossi

```bash
libreyolo label [data=<dataset.yaml|folder>] [key=value ...]
```

Gli argomenti sono coppie `key=value`, e funziona anche la forma POSIX, quindi
`port=9200` e `--port 9200` sono lo stesso argomento.

## Argomenti

| Argomento | Predefinito | Significato |
|---|---|---|
| `data` | | YAML o cartella del dataset da aprire direttamente. Se non impostato parte dalla home del progetto |
| `host` | `127.0.0.1` | Host o interfaccia su cui fare il binding |
| `port` | `8000` | Porta su cui fare il binding. Passa alla successiva libera se è occupata |
| `device` | `auto` | Dispositivo per l'auto-etichettatura IA: `0`, `cpu`, `mps`, `auto` |
| `no_assist` | `false` | Disattiva l'auto-etichettatura IA, lasciando un etichettatore manuale |
| `no_browser` | `false` | Non aprire automaticamente il browser |
| `share` | `false` | Fa il binding su `0.0.0.0` così i colleghi sulla tua rete possono entrare |
| `json` | `false` | Output JSON su stdout |
| `quiet` | `false` | Silenzia stderr |
| `verbose` | `false` | Output dettagliato su stderr |

## Esempi

<code-tabs name="examples" />

## Note

### Che cosa scrive

I box vengono salvati come file `labels/*.txt` nel formato nativo di LibreYOLO,
che è il formato letto da `libreyolo train`, quindi dopo non c'è niente da
convertire. Questa versione gestisce solo i bounding box. Le modifiche si salvano
mentre ti sposti tra le immagini.

### Aprire un dataset

Senza `data`, lo strumento parte dalla home del progetto e il dataset si sceglie
o si crea dal browser. Passare `data=path/to/data.yaml` apre subito quel dataset,
e la riga di avvio riporta il numero di immagini, il numero di classi e se il
dataset è scrivibile. Un dataset in sola lettura si apre comunque e indica perché
non ci si può scrivere.

### Condivisione, e che cosa fa `host`

`share=true` fa il binding sull'indirizzo jolly, il che permette ad altre
macchine sulla tua rete di raggiungere lo strumento, mentre le azioni
amministrative (cambiare o eliminare progetti e avviare calcoli) restano su
questa macchina.

Impostare `host` su un'interfaccia specifica fa una cosa diversa e meno sicura:
l'host diventa indistinguibile da un client di rete, quindi ogni client ottiene
i diritti amministrativi. Quando lo fai, il comando stampa un avviso su stderr.
Meglio `share=true`.

### Porte e spegnimento

Una porta occupata passa alla successiva, fino a venti oltre quella richiesta. Se
falliscono tutte e venti, si esce con `io_error`. L'URL stampato su stdout è la
porta a cui ci si è effettivamente collegati. Con `share=true`, il risultato
riporta anche `lan_url`, l'indirizzo che devono aprire i colleghi.

Il comando resta in primo piano fino a Ctrl+C.

Correlati: [`libreyolo doctor`](/docs/cli/doctor) per controllare il dataset
etichettato prima dell'addestramento, e [`libreyolo train`](/docs/cli/train) per
addestrare su di esso.
