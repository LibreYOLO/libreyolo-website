---
title: utility di libreyolo
seo_title: "riferimento dei comandi di utility della CLI di libreyolo"
description: "I comandi piccoli di LibreYOLO: version, checks, models, formats, cfg, info, metadata, enroll e compare, ognuno con i suoi argomenti e i suoi valori predefiniti."
lead: "Nove comandi che riportano o ispezionano invece di calcolare. Stampano i dati dell'ambiente, l'inventario di modelli e formati, i valori predefiniti già risolti e i dettagli di un checkpoint, e costruiscono e interrogano una galleria di volti."
keywords: [libreyolo version, libreyolo checks, elencare modelli libreyolo, formati di esportazione libreyolo, vedere metadati checkpoint yolo, galleria volti libreyolo enroll]
last_verified: "1.5.0"
meta:
  - label: Comandi
    value: version, checks, models, formats, cfg, info, metadata, enroll, compare
    mono: true
  - label: Output
    value: "stdout, in testo oppure con json=true come un unico oggetto che porta schema_version"
snippets:
  examples:
    - label: Ambiente
      language: bash
      code: |
        libreyolo version
        libreyolo checks
    - label: Cosa è disponibile
      language: bash
      code: |
        libreyolo models
        libreyolo formats family=yolo9 task=detect
    - label: Ispezionare un checkpoint
      language: bash
      code: |
        libreyolo info model=LibreYOLO9s.pt
        libreyolo metadata path=weights/LibreYOLO9s.pt
---

## Sinossi

```bash
libreyolo <command> [key=value ...]
```

Gli argomenti sono coppie `key=value`, e funziona anche la forma POSIX, quindi
`model=x` e `--model x` sono lo stesso argomento. Tutti i comandi di questa
pagina scrivono i risultati su stdout e accettano `json=true` e `quiet=true`.

Il comando radice porta un flag proprio, `libreyolo --version`, che stampa la
stringa di versione ed esce. È un output più piccolo di quello del comando
`version` qui sotto.

## version

Stampa la versione di LibreYOLO più le versioni di Python, torch e CUDA con cui
sta girando.

```bash
libreyolo version
```

| Argomento | Predefinito | Significato |
|---|---|---|
| `json` | `false` | Output JSON su stdout |
| `quiet` | `false` | Silenzia stderr |

## checks

Stampa l'ambiente in modo più dettagliato: Python, torch, CUDA, cuDNN, ogni GPU
rilevata con il suo nome e la sua memoria, e la versione installata di ogni
pacchetto opzionale usato dai percorsi di esportazione.

```bash
libreyolo checks
```

| Argomento | Predefinito | Significato |
|---|---|---|
| `json` | `false` | Output JSON su stdout |
| `quiet` | `false` | Silenzia stderr |

L'elenco dei pacchetti copre `onnx`, `onnxruntime`, `tensorrt`, `openvino`,
`paddlepaddle`, `x2paddle`, `mnn`, `ncnn`, `onnx2tf`, `ai-edge-litert`,
`transformers` e `scipy`. Un pacchetto che non è installato viene segnalato come
tale invece di essere omesso, così un'esportazione fallita si può ricondurre a
una dipendenza mancante con questo unico comando.

## models

Elenca ogni famiglia di modelli con i suoi task, le sue dimensioni, i nomi CLI
che risolvono ai suoi checkpoint e la risoluzione di input di ogni dimensione.

```bash
libreyolo models
```

| Argomento | Predefinito | Significato |
|---|---|---|
| `json` | `false` | Output JSON su stdout |
| `quiet` | `false` | Silenzia stderr |

Una famiglia la cui dipendenza opzionale non è installata compare come non
disponibile insieme alla riga `pip install` che la renderebbe disponibile. I
nomi CLI sono ciò che `model=` accetta come forma abbreviata: `yolox-s` risolve
a `LibreYOLOXs.pt`, e i task che non sono di rilevamento portano il suffisso del
loro task.

## formats

Elenca i formati di esportazione che l'ambiente installato è in grado di
produrre, con l'estensione di file di ogni formato e se supporta FP16 e INT8.

```bash
libreyolo formats [family=<family>] [task=<task>]
```

| Argomento | Predefinito | Significato |
|---|---|---|
| `family` | | Mostra i livelli di supporto per una famiglia di modelli. `model=` è accettato come la stessa opzione |
| `task` | | Task canonico del modello. Il task predefinito della famiglia se non indicato |
| `json` | `false` | Output JSON su stdout |
| `quiet` | `false` | Silenzia stderr |

Senza `family`, l'output è il solo inventario dei formati. Con esso, ogni
formato guadagna il livello di supporto per quella famiglia e quel task, il
motivo dietro al livello e ogni vincolo associato. Una famiglia sconosciuta, o
un task che la famiglia non supporta, è un errore d'uso.

Gli alias di formato compaiono accanto al loro nome canonico: `engine` per
`tensorrt`, `litert` per `tflite`.

## cfg

Stampa la configurazione predefinita già risolta: i valori predefiniti di
addestramento, quelli di validazione, quelli di predizione e gli override per
famiglia.

```bash
libreyolo cfg
```

| Argomento | Predefinito | Significato |
|---|---|---|
| `json` | `false` | Output JSON su stdout |
| `quiet` | `false` | Silenzia stderr |

I valori si leggono dalle dataclass di configurazione, non da una copia, quindi
questa è l'autorità su ciò che userà un addestramento quando non passi un
argomento. `family_overrides` è la sezione che risponde al perché una famiglia
si è addestrata con impostazioni che non hai chiesto. Vedi
[`libreyolo train`](/docs/cli/train) per come vengono applicati quegli override.

## info

Carica un modello sulla CPU e riporta la sua famiglia, la sua dimensione, il suo
numero di parametri, le sue classi e il livello di supporto all'esportazione per
ogni formato.

```bash
libreyolo info model=<name|path>
```

| Argomento | Predefinito | Significato |
|---|---|---|
| `model` | | Nome del modello o percorso dei pesi. Obbligatorio |
| `detailed` | `false` | Include i dettagli per parametro |
| `json` | `false` | Output JSON su stdout |
| `quiet` | `false` | Silenzia stderr |

## metadata

Legge i metadati di un checkpoint senza costruire un modello, e li valida
rispetto allo schema dei checkpoint di LibreYOLO.

```bash
libreyolo metadata path=<checkpoint.pt>
```

| Argomento | Predefinito | Significato |
|---|---|---|
| `path` | | Percorso di un checkpoint `.pt`. Obbligatorio |
| `json` | `false` | Output JSON su stdout |
| `quiet` | `false` | Silenzia stderr |

Le voci grandi che portano tensori vengono riassunte invece che stampate, così
l'output resta leggibile su un checkpoint completo di addestramento. Un
checkpoint che non esiste esce con `checkpoint_not_found`, e uno i cui metadati
non passano la validazione stampa gli errori ed esce con `1`.

## enroll

Costruisce una galleria di volti a partire da un albero con una cartella per
persona, così che le predizioni successive possano dare un nome ai volti che
trovano.

```bash
libreyolo enroll model=<embedder> source=<people-dir> gallery=<gallery.npz>
```

| Argomento | Predefinito | Significato |
|---|---|---|
| `model` | | Modello di embedding dei volti, percorso o nome. Obbligatorio |
| `source` | | Albero con una cartella per persona, `source/<identity>/*.jpg`. Obbligatorio |
| `gallery` | | File `.npz` della galleria di output. Viene esteso sul posto se esiste già. Obbligatorio |
| `face_detector` | | Rilevatore di volti: un `.onnx` di YuNet o un rilevatore LibreYOLO. Il rilevatore predefinito della famiglia se non indicato |
| `device` | `auto` | Dispositivo: `0`, `cpu`, `mps`, `auto` |
| `json` | `false` | Output JSON su stdout |
| `quiet` | `false` | Silenzia stderr |

```bash
# people/ contiene una cartella per identità; il nome della cartella diventa l'identità.
libreyolo enroll model=librefacerec-l.onnx source=people/ gallery=people.npz
```

Il nome della sottocartella è l'identità. Un'immagine di riferimento senza alcun
volto rilevabile viene saltata con una riga su stderr e le altre proseguono; una
sorgente senza sottocartelle di identità, o una in cui non è stato trovato
nessun volto, è un errore.

Passa il file risultante a
[`libreyolo predict`](/docs/cli/predict) come `gallery=people.npz` per fare in
modo che i rilevamenti portino un'identità e un punteggio di corrispondenza.

## compare

Riporta la similarità coseno tra due immagini di volti e se supera la soglia di
stessa identità.

```bash
libreyolo compare model=<embedder> source=<a.jpg> source2=<b.jpg>
```

| Argomento | Predefinito | Significato |
|---|---|---|
| `model` | | Modello di embedding dei volti, percorso o nome. Obbligatorio |
| `source` | | Prima immagine. Obbligatorio |
| `source2` | | Seconda immagine con cui confrontare. Obbligatorio |
| `face_detector` | | Rilevatore di volti: un `.onnx` di YuNet o un rilevatore LibreYOLO |
| `threshold` | `0.4` | Soglia di similarità coseno per la decisione di stessa identità |
| `device` | `auto` | Dispositivo: `0`, `cpu`, `mps`, `auto` |
| `json` | `false` | Output JSON su stdout |
| `quiet` | `false` | Silenzia stderr |

```bash
libreyolo compare model=librefacerec-l.onnx source=a.jpg source2=b.jpg
```

`libreyolo verify` è registrato come secondo nome di questo comando e prende gli
stessi argomenti.

Sia `compare` sia `enroll` hanno bisogno di un modello il cui task sia
l'embedding dei volti. Qualsiasi altra cosa esce con `config_unsupported`. Come
sorgenti sono accettati sia i percorsi locali delle immagini sia gli URL `http`
o `https`.

## Esempi

<code-tabs name="examples" />

## Note

stdout porta il risultato; l'avanzamento e gli avvisi vanno su stderr.
`json=true` stampa un unico oggetto con `schema_version`, che è la forma da
leggere da uno script. L'output in testo è quello predefinito ed è pensato per
essere letto da una persona.

I codici di uscita seguono la stessa mappa del resto della CLI: `0` se va tutto
bene, `2` per un errore d'uso o di configurazione, `3` quando una sorgente non
si trova, `4` quando un modello o un checkpoint non si può caricare, e `1` per
gli altri fallimenti a runtime.

Correlati: [`libreyolo doctor`](/docs/cli/doctor), che è il comando di ispezione
dal lato dataset, e [`libreyolo profile`](/docs/cli/profile), quello dal lato
prestazioni.
