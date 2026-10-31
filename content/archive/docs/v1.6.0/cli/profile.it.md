---
title: libreyolo profile
seo_title: Riferimento del comando libreyolo profile
description: >-
  Misura la velocità di addestramento e inferenza e leggi il risultato: ogni
  sottocomando di profile, i suoi argomenti e i valori predefiniti, e cosa
  riporta ogni lente.
lead: >-
  Un gruppo di comandi che misura dove va il tempo in uno step di addestramento
  o in una chiamata di inferenza, scrive un profilo autonomo e rilegge quel
  profilo attraverso diverse lenti.
keywords:
  - libreyolo profile cli
  - profiler training yolo
  - profiling latenza inferenza
  - gpu kernel profiling pytorch
  - misurare velocità inferenza yolo
last_verified: 1.5.0
meta:
  - label: Comando
    value: libreyolo profile
    mono: true
  - label: Output
    value: profile.json and profile_trace.json under runs/profile
    mono: true
snippets:
  examples:
    - label: Misurare l'inferenza
      language: bash
      code: |
        # Senza argomento source viene usata l'immagine di esempio inclusa.
        libreyolo profile infer --device cpu --warmup 5 --runs 20
    - label: Leggere il verdetto
      language: bash
      code: |
        libreyolo profile summary runs/profile/infer/profile.json
    - label: Confrontare due misurazioni
      language: bash
      code: >
        libreyolo profile infer --device cpu --warmup 5 --runs 20 --project
        runs/profile/a

        libreyolo profile infer --device cpu --warmup 5 --runs 20 --batch 4
        --project runs/profile/b


        libreyolo profile compare runs/profile/a/infer/profile.json \
          runs/profile/b/infer/profile.json
source_hash: b967e869fd9ba418
---

## Sinossi

```bash
libreyolo profile <subcommand> [<positional>] [--flag value ...]
```

Questo gruppo non accetta argomenti `key=value`. I suoi sottocomandi usano
argomenti posizionali e flag POSIX, quindi si scrive `--weights
LibreYOLO9t.pt`, non `weights=LibreYOLO9t.pt`. Eseguire `libreyolo profile`
senza sottocomando stampa l'elenco.

Due sottocomandi misurano e scrivono un profilo; gli altri lo leggono. `run` e
`infer` producono entrambi lo stesso `profile.json` autonomo, quindi ogni
sottocomando di lettura funziona con l'uno o con l'altro.

## profile run

Esegue un breve addestramento profilato e scrive un profilo.

```bash
libreyolo profile run <data> [--flag value ...]
```

| Argomento | Default | Significato |
|---|---|---|
| `data` | | Posizionale. YAML o nome del dataset, ad esempio `coco128`. Obbligatorio |
| `--weights` | `LibreYOLO9t.pt` | File o nome dei pesi del modello |
| `--size` | `t` | Variante di dimensione del modello |
| `--batch` | `16` | Micro-batch. `-1` si adatta automaticamente a circa il 70% della VRAM |
| `--imgsz` | `640` | Dimensione delle immagini di addestramento |
| `--workers` | `8` | Worker del dataloader |
| `--amp` | `true` | Usa il percorso AMP della famiglia. `--no-amp` lo disattiva |
| `--steps` | `20` | Step profilati, cioè misurati |
| `--warmup` | `5` | Step di warmup prima della misurazione |
| `--repeat` | `1` | Ripete N volte per ottenere media e deviazione standard |
| `--device` | `0` | Dispositivo |
| `--project` | `runs/profile` | Radice della directory di output |
| `--json` | `false` | Output JSON su stdout |

La finestra misurata è `--warmup` più `--steps` iterazioni. Un dataset troppo
piccolo per riempirla non produce alcun profilo e il comando esce con codice
`3`, indicando le tre vie d'uscita: un dataset più grande, meno step o un batch
più piccolo.

`--repeat` maggiore di 1 scrive un `runs/profile/profile_repeat.json` aggregato,
le cui metriche scalari sono mediate tra le prove, mentre gli elenchi dei kernel
vengono dall'ultima prova. È anche il prerequisito per un verdetto di
significatività in `compare`: una singola esecuzione non può fornirlo.

## profile infer

Profila il percorso di inferenza e scrive un profilo.

```bash
libreyolo profile infer [<source>] [--flag value ...]
```

| Argomento | Default | Significato |
|---|---|---|
| `source` | | Posizionale. Immagine o directory. Se omesso, l'immagine di esempio inclusa |
| `--weights` | `LibreYOLO9t.pt` | File o nome dei pesi del modello |
| `--size` | `t` | Variante di dimensione del modello |
| `--batch` | `1` | Immagini per passaggio forward |
| `--imgsz` | `640` | Dimensione delle immagini in input |
| `--half` | `false` | Forward in autocast, solo CUDA. `--no-half` lo disattiva |
| `--amp-dtype` | `float16` | dtype dell'autocast CUDA: `float16` o `bfloat16` |
| `--warmup` | `20` | Iterazioni di warmup prima della misurazione |
| `--runs` | `100` | Iterazioni misurate |
| `--repeat` | `1` | Ripete N volte per ottenere media e deviazione standard |
| `--conf` | `0.25` | Soglia di confidenza, che cambia quanto lavoro fa NMS |
| `--iou` | `0.45` | Soglia IoU di NMS |
| `--max-det` | `300` | Numero massimo di rilevamenti per immagine, che cambia quanto lavoro fa NMS |
| `--device` | `0` | Dispositivo |
| `--trace` | `true` | Produce un trace Chrome per analizzare in dettaglio kernel e op. `--no-trace` lo salta |
| `--project` | `runs/profile` | Radice della directory di output |
| `--json` | `false` | Output JSON su stdout |

Riporta la latenza a p50, p90 e p99, il throughput in immagini al secondo e la
ripartizione per fase tra preprocess, forward e postprocess. I tre argomenti di
soglia sono qui perché spostano il numero del postprocess.

## profile summary

```bash
libreyolo profile summary <trace> [--json]
```

| Argomento | Default | Significato |
|---|---|---|
| `trace` | | Posizionale. Percorso di un `profile.json` o `profile_trace.json`. Obbligatorio |
| `--json` | `false` | Output JSON su stdout |

La lettura ad alto livello: tempo per step, throughput, utilizzo della GPU,
quota di Tensor Core, VRAM di picco, overhead dell'host, lanci di kernel per
step, il verdetto sul collo di bottiglia con la sua motivazione, il mix di
kernel per categoria e i kernel principali per step. Su un profilo di inferenza
stampa anche i percentili di latenza e la ripartizione per fase.

Un profilo preso in condizioni di thrashing della VRAM viene segnalato, perché
l'utilizzo e il throughput misurati lì non sono affidabili.

## profile get

```bash
libreyolo profile get <trace> [<field>] [--json]
```

| Argomento | Default | Significato |
|---|---|---|
| `trace` | | Posizionale. Percorso di un profilo. Obbligatorio |
| `field` | | Posizionale. Nome della metrica. Ometti per elencare le metriche disponibili |
| `--json` | `false` | Output JSON su stdout |

Stampa una sola metrica e nient'altro, per i cicli negli script. Un campo
sconosciuto esce con codice `2` e rimanda alla forma che elenca le metriche.

## profile phases

```bash
libreyolo profile phases <trace> [--json]
```

| Argomento | Default | Significato |
|---|---|---|
| `trace` | | Posizionale. Percorso di un profilo. Obbligatorio |
| `--json` | `false` | Output JSON su stdout |

Millisecondi GPU, millisecondi reali, numero di kernel e numero di op per fase:
forward, backward, dataload, to_device, optimizer.

## profile kernels

```bash
libreyolo profile kernels <trace> [--flag value ...]
```

| Argomento | Default | Significato |
|---|---|---|
| `trace` | | Posizionale. Percorso di un profilo. Obbligatorio |
| `--top` | `20` | Mostra i primi N per tempo GPU |
| `--category` | | Filtra per sottostringa di categoria: `gemm`, `layout`, `norm`, `elementwise` |
| `--grep` | | Filtra per espressione regolare sul nome del kernel |
| `--tensorcore` | `false` | Solo kernel Tensor Core |
| `--sort` | `time` | `time`, `count` o `name` |
| `--phase` | | Limita a una sola fase: `forward`, `backward`, `dataload`, `to_device`, `optimizer` |
| `--json` | `false` | Output JSON su stdout |

Il fondo dell'analisi: i singoli kernel GPU con la loro quota di tempo GPU, i
millisecondi per step, le invocazioni per step e la categoria. Un `--phase`
sconosciuto esce con codice `2` ed elenca le fasi presenti nel profilo.

## profile ops

```bash
libreyolo profile ops <trace> [--flag value ...]
```

| Argomento | Default | Significato |
|---|---|---|
| `trace` | | Posizionale. Percorso di un profilo. Obbligatorio |
| `--top` | `20` | Mostra i primi N per tempo CPU |
| `--phase` | | Limita a una sola fase |
| `--json` | `false` | Output JSON su stdout |

La vista del framework anziché quella del dispositivo: op `aten` e di autograd
ordinate per tempo CPU, che è dove si manifesta il costo dei lanci dall'host.

## profile compare

```bash
libreyolo profile compare <before> <after> [--json]
```

| Argomento | Default | Significato |
|---|---|---|
| `before` | | Posizionale. Profilo di riferimento. Obbligatorio |
| `after` | | Posizionale. Nuovo profilo. Obbligatorio |
| `--json` | `false` | Output JSON su stdout |

Confronta throughput, millisecondi per immagine, utilizzo della GPU, overhead
dell'host, lanci di kernel per step e verdetto sul collo di bottiglia.

Il giudizio di significatività richiede che entrambi i lati siano misurati con
`--repeat` di almeno 2. A quel punto una differenza conta come significativa
quando supera il doppio dell'errore standard combinato, e l'output stampa il
confronto che ha fatto. Senza, la riga dice che una singola esecuzione non può
sostenere il giudizio.

## profile what-if

```bash
libreyolo profile what-if <trace> [--flag value ...]
```

| Argomento | Default | Significato |
|---|---|---|
| `trace` | | Posizionale. Percorso di un profilo. Obbligatorio |
| `--remove-category` | | Proietta la rimozione di una categoria di kernel: `gemm`, `layout`, `norm`, `elementwise` |
| `--remove-launches` | | Proietta la rimozione di N lanci di kernel per step, per esempio un guadagno da fusione di op |
| `--json` | `false` | Output JSON su stdout |

Stima cosa porterebbe una modifica prima che la modifica venga scritta. Una
delle due opzioni è obbligatoria; nessuna delle due esce con codice `2`.

La proiezione segue il verdetto del profilo stesso. Sotto l'80% di utilizzo
della GPU modella il risparmio come meno lanci moltiplicati per il costo host
per lancio misurato; sopra, come meno lavoro GPU. Il risultato porta con sé un
campo di avvertenza, perché il costo per lancio è un'approssimazione e l'unica
prova è una seconda misurazione.

## Esempi

<code-tabs name="examples" />

## Note

Il profiler misura e riporta. Non cambia nulla: leggere il verdetto, modificare
la configurazione o il codice, rieseguire e confrontare è il ciclo per cui è
costruito.

`--device` vale `0` per impostazione predefinita, cioè il dispositivo CUDA 0.
Passare `--device cpu` misura sulla CPU e produce un profilo che i sottocomandi
di lettura accettano comunque, senza il dettaglio dei kernel GPU.

Ogni sottocomando supporta `--json`, e quelli di lettura stampano solo su
stdout, ed è questo che rende il gruppo utilizzabile da uno script.

I codici di uscita qui sono propri del gruppo: `2` per un file che non esiste o
un argomento che non si risolve, `3` quando `run` non ha prodotto alcun profilo
e `1` quando un trace non può essere analizzato.

Correlato: [`libreyolo train`](/docs/cli/train), i cui argomenti sono ciò che di
solito un profilo di addestramento serve a regolare.
