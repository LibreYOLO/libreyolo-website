---
title: libreyolo monitor
seo_title: riferimento del comando libreyolo monitor
description: >-
  Serve una dashboard live per gli addestramenti: argomenti con i valori
  predefiniti, cosa legge il server dal disco e come un solo server copre molti
  run.
lead: >-
  Serve una dashboard web per gli addestramenti, leggendo gli artefatti che un
  run scrive su disco. Non si aggancia mai al processo di addestramento, quindi
  i run live, quelli finiti e quelli andati in crash vengono mostrati allo
  stesso modo.
keywords:
  - libreyolo monitor cli
  - dashboard training
  - monitorare addestramento in tempo reale
  - libreyolo monitor porta
  - visualizzare metriche training
last_verified: 1.5.0
meta:
  - label: Comando
    value: libreyolo monitor
    mono: true
  - label: Output
    value: 'Una URL del server su stdout, poi il processo resta in primo piano'
snippets:
  examples:
    - label: Base
      language: bash
      code: |
        # Sorveglia runs/ ed elenca tutti i run che ci sono sotto.
        libreyolo monitor
    - label: Un'altra radice dei run
      language: bash
      code: |
        libreyolo monitor experiments/
    - label: 'Un solo run, porta fissa, senza browser'
      language: bash
      code: |
        libreyolo monitor runs/train/exp port=9100 no_browser=true
source_hash: 4aa178141d451728
---

## Sinossi

```bash
libreyolo monitor [<run-dir|runs-root>] [key=value ...]
```

La directory è posizionale. Tutto il resto è una coppia `key=value`, e funziona
anche la forma POSIX, quindi `port=9100` e `--port 9100` sono lo stesso
argomento.

## Argomenti

| Argomento | Predefinito | Significato |
|---|---|---|
| `run_dir` | `runs` | Posizionale. Una radice dei run da sorvegliare, o una singola directory di run da aprire direttamente. In entrambi i casi vengono elencati tutti i run sotto la radice |
| `host` | `127.0.0.1` | Host o interfaccia su cui mettersi in ascolto |
| `port` | `8420` | Porta su cui mettersi in ascolto. Passa alla prima libera se è occupata |
| `no_browser` | `false` | Non aprire automaticamente il browser |
| `json` | `false` | Output JSON su stdout |
| `quiet` | `false` | Silenzia stderr |
| `verbose` | `false` | Output dettagliato su stderr |

## Esempi

<code-tabs name="examples" />

## Note

### Un server, molti run

Il server sorveglia una radice dei run invece di un singolo run, e indirizza
ogni run tramite URL, quindi più run sulla stessa macchina condividono una sola
porta. Apri la URL radice per l'indice, oppure una scheda per run; il parametro
`?run=` in ogni URL indica quale.

Puntare il comando su una singola directory di run mette la radice del server
sulla directory padre, così i run fratelli compaiono comunque nell'indice, e il
link porta direttamente a quello indicato.

### Cosa legge

La dashboard è costruita a partire dai file che scrive `libreyolo train`:
`status.json`, `metrics.jsonl`, `train.log` e le immagini del run. Non viene
letto nulla dal processo di addestramento stesso, quindi un run che è finito, o
che è morto, viene mostrato esattamente come uno live.

### Prerequisiti e porte

Deve esistere già almeno un run. Senza argomento e senza directory `runs/`, il
comando esce con `source_not_found`; succede lo stesso quando la directory
indicata non contiene nessun run.

Una porta occupata passa alla successiva, fino a venti oltre quella richiesta.
Se falliscono tutte e venti, esce con `io_error`. La URL stampata su stdout è la
porta effettivamente occupata.

Il comando serve in primo piano fino a Ctrl+C. `json=true` stampa la URL, la
radice sorvegliata e il numero di run trovati, come un unico oggetto con
`schema_version`.

Correlato: [`libreyolo train`](/docs/cli/train), i cui argomenti `project` e
`name` decidono dove finiscono queste directory di run.
