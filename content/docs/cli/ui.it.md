---
title: libreyolo ui
seo_title: riferimento del comando libreyolo ui
description: >-
  Avvia la web UI locale per l'inferenza: indirizzo di bind, comportamento della
  porta, scelta del dispositivo e come termina il comando.
lead: >-
  Avvia un server web locale che accetta immagini trascinate o incollate, ci
  esegue sopra il modello scelto e mostra i risultati nel browser.
keywords:
  - libreyolo ui cli
  - libreyolo web ui
  - ui web inferenza locale
  - inferenza drag and drop immagini
  - libreyolo ui porta
last_verified: 1.5.0
meta:
  - label: Comando
    value: libreyolo ui
    mono: true
  - label: Output
    value: 'Un URL del server su stdout, poi il processo resta in primo piano'
snippets:
  examples:
    - label: Base
      language: bash
      code: |
        libreyolo ui
    - label: 'Porta fissa, senza browser'
      language: bash
      code: |
        libreyolo ui port=9000 no_browser=true
    - label: 'Su CPU, output leggibile da un programma'
      language: bash
      code: |
        libreyolo ui device=cpu json=true
source_hash: b0eebd33fd0f463b
---

## Sintassi

```bash
libreyolo ui [key=value ...]
```

Gli argomenti sono coppie `key=value`, e funziona anche la forma POSIX, quindi
`port=9000` e `--port 9000` sono lo stesso argomento.

## Argomenti

| Argomento | Default | Significato |
|---|---|---|
| `host` | `127.0.0.1` | Host o interfaccia su cui mettersi in ascolto |
| `port` | `8000` | Porta su cui mettersi in ascolto. Passa alla successiva libera se è occupata |
| `device` | `auto` | Dispositivo: `0`, `cpu`, `mps`, `auto` |
| `no_browser` | `false` | Non aprire automaticamente il browser |
| `json` | `false` | Output JSON su stdout |
| `quiet` | `false` | Silenzia stderr |
| `verbose` | `false` | Output dettagliato su stderr |

## Esempi

<code-tabs name="examples" />

## Note

Di default il bind è sul loopback, quindi la UI è raggiungibile solo da questa
macchina.

Se la porta richiesta è occupata, il comando prova la successiva e continua
fino a venti porte oltre quella richiesta. Se falliscono tutte e venti, esce
con `io_error` e il suggerimento di passare una porta diversa. L'URL stampato
su stdout è la porta a cui il server si è effettivamente legato, quindi leggilo
invece di dare per scontata quella che hai chiesto.

A meno che tu non passi `no_browser=true`, poco dopo il bind si apre una scheda
del browser su quell'URL.

Il comando resta poi in primo piano a servire le richieste fino a Ctrl+C, che
spegne il server in modo pulito. Non esiste una modalità detached; mandalo in
background con la shell se vuoi riavere il terminale.

`json=true` stampa URL e dispositivo come un unico oggetto con
`schema_version` prima che il server parta, ed è così che uno script recupera
la porta a cui si è legato.

Correlati: [`libreyolo label`](/docs/cli/label) per disegnare box e salvare
etichette, [`libreyolo monitor`](/docs/cli/monitor) per seguire gli
addestramenti. Sono entrambi server web locali con lo stesso comportamento su
porta e browser.
