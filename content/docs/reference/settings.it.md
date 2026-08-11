---
title: Impostazioni
seo_title: Variabili d'ambiente e directory di LibreYOLO
description: >-
  Tutte le variabili d'ambiente che LibreYOLO legge, le directory in cui scrive,
  i token di cui ha bisogno e gli interruttori che cambiano quale percorso di
  codice viene eseguito.
lead: >-
  LibreYOLO non ha un file di configurazione. Il comportamento che non è un
  argomento di funzione è controllato da variabili d'ambiente e da un piccolo
  numero di directory convenzionali, tutte elencate qui.
keywords:
  - LIBREYOLO_DATASETS_DIR
  - LIBREYOLO_KERNELS
  - LIBREYOLO_FASTER_COCO_EVAL
  - HF_TOKEN
  - variabili ambiente libreyolo
  - cartella pesi libreyolo
last_verified: 1.5.0
verification: >-
  Variabili individuate cercando os.environ e os.getenv in libreyolo/**/*.py
  alla v1.5.0; semantica letta in ogni punto di utilizzo. Convenzioni sulle
  directory lette da libreyolo/data/utils.py, libreyolo/utils/download.py,
  libreyolo/export/exporter.py, libreyolo/models/base/model.py e
  libreyolo/models/sam3dbody/mhr_body.py.
snippets:
  usage:
    - label: Puntare la radice dei dataset altrove
      language: bash
      code: |
        export LIBREYOLO_DATASETS_DIR=/data/datasets
        python -c "from libreyolo.data import DATASETS_DIR; print(DATASETS_DIR)"
    - label: Leggere il valore risolto da Python
      language: python
      code: >
        from libreyolo.data import DATASETS_DIR


        # Il valore predefinito è ~/datasets; LIBREYOLO_DATASETS_DIR lo
        sovrascrive al momento dell'import.

        print(DATASETS_DIR)
source_hash: 462f1288582225ce
---

## Variabili d'ambiente

| Variabile | Valore predefinito | Effetto |
|---|---|---|
| `LIBREYOLO_DATASETS_DIR` | `~/datasets` | Radice dei dataset. Letta una sola volta all'import, dentro `libreyolo.data.DATASETS_DIR` |
| `LIBREYOLO_FASTER_COCO_EVAL` | non impostata | Sovrascrive il flag di validazione `faster_coco_eval`. `1`, `true`, `yes` o `on` forza l'attivazione del backend più veloce, qualsiasi altro valore lo disattiva, se non impostata si affida al flag della configurazione |
| `LIBREYOLO_KERNELS` | non impostata | Selezione dei kernel. `off` o `reference` forza le implementazioni di riferimento; qualsiasi altro valore seleziona solo le implementazioni registrate sotto quel nome |
| `LIBREYOLO_QUANT_KERNELS` | non impostata | Alias legacy di `LIBREYOLO_KERNELS`, letto solo quando quest'ultima non è impostata |
| `LIBREYOLO_HUB_KERNELS` | non impostata | `0`, `false`, `off` o `no` disabilita il caricamento dei kernel dall'Hub di Hugging Face. Qualsiasi altro valore, compresa l'assenza, lo lascia abilitato |
| `LIBREYOLO_MHR_PATH` | `~/.cache/libreyolo/mhr/mhr_model.pt` | Posizione del modello MHR body usato dal task `mesh` |
| `LIBRELABEL_ENABLE_LOCATE` | non impostata | Deve valere esattamente `1`, `true`, `yes` o `on` per esporre l'assistente LocateAnything nello strumento di etichettatura. Qualsiasi altro valore lo tiene disattivato |
| `SAM_3D_BODY_PATH` | non impostata | Percorso del pacchetto SAM 3D Body per la famiglia mesh, quando non viene passato al costruttore |
| `HF_TOKEN` | non impostata | Token di accesso Hugging Face, usato per i repository ad accesso limitato |

<code-tabs name="usage" />

`LIBREYOLO_DATASETS_DIR` viene letta al momento dell'import, quindi impostarla
dopo aver importato `libreyolo.data` non ha alcun effetto su `DATASETS_DIR`.

I kernel dell'Hub sono un opt-in in due parti. Il recupero a runtime avviene
solo quando il pacchetto opzionale `kernels` è installato, quindi installare
`libreyolo[hub-kernels]` è l'opt-in e `LIBREYOLO_HUB_KERNELS=0` è l'opt-out.
Un'installazione senza l'extra non è influenzata né in un senso né nell'altro.

Anche la selezione dei kernel accorcia gli import: quando `LIBREYOLO_KERNELS`
forza `off` o `reference`, i provider accelerati interni non vengono importati
affatto. Il registro che queste tre variabili controllano è documentato su
[kernels](/docs/reference/kernels).

## Variabili che la libreria imposta

Queste vengono scritte anziché lette, quindi impostarle a mano non è il
percorso supportato.

| Variabile | Impostata da |
|---|---|
| `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` | L'helper di spawn DDP, un valore per ogni processo worker |
| `CUDA_VISIBLE_DEVICES` | Ristretta temporaneamente durante il setup distribuito, poi ripristinata |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Impostata a `1` dai trainer EC, con `setdefault`, così un valore esistente prevale |
| `MOMENTUM_ENABLED` | Impostata con `setdefault` dal loader della famiglia mesh |

`LOCAL_RANK` funge anche da segnale di modalità distribuita: la sua presenza
nell'ambiente è il modo in cui il codice di addestramento rileva di essere in
esecuzione sotto DDP.

## Variabili dei logger

I logger di addestramento opzionali ricadono sui valori predefiniti
dell'ambiente per il nome del progetto.

| Variabile | Valore predefinito | Usata da |
|---|---|---|
| `WANDB_PROJECT` | `libreyolo` | Il logger Weights and Biases, quando non viene passato alcun progetto |
| `COMET_PROJECT_NAME` | `libreyolo` | Il logger Comet, quando non viene passato alcun progetto |

L'autenticazione per quei servizi segue i loro strumenti, non LibreYOLO.

## Token

`HF_TOKEN` è il token di accesso a Hugging Face. Quando non è impostata, il
token viene letto da `~/.cache/huggingface/token`, che è dove lo scrive un
login con la CLI di Hugging Face. Entrambi i percorsi funzionano.

Un token serve solo per i repository ad accesso limitato. SAM 3 è l'esempio
incluso: i suoi pesi si scaricano da un repository ad accesso limitato con una
licenza personalizzata, quindi bisogna accettare i termini sulla pagina del
repository e la sessione deve essere autenticata.

## Directory

| Percorso | Contenuto |
|---|---|
| `weights/` | Checkpoint scaricati, snapshot Hugging Face scaricati e artefatti esportati |
| `~/datasets` | Radice dei dataset, a meno che `LIBREYOLO_DATASETS_DIR` non dica altrimenti |
| `~/.cache/huggingface/token` | Token Hugging Face, quando non è in `HF_TOKEN` |
| `~/.cache/libreyolo/mhr/mhr_model.pt` | Modello MHR body, a meno che `LIBREYOLO_MHR_PATH` non dica altrimenti |
| `runs/track/` | Output predefinito di `model.track(save=True)` |

`weights/` è relativa alla directory di lavoro. Un nome di file nudo si risolve
attraverso di essa, quindi `LibreYOLO("LibreYOLO9t.pt")` cerca
`weights/LibreYOLO9t.pt` e scarica lì quando non c'è. `model.export()` scrive
nella stessa directory quando `output_path` non viene indicato. I tier fratelli
scaricano snapshot multi-file in `weights/<Prefix><size>/`.

## Comportamento dei download

I download dei pesi vengono ritentati tre volte con backoff, riprendono da un
file parziale e sono protetti da un file di lock così due processi non
recuperano lo stesso checkpoint contemporaneamente. Una famiglia che scarica da
un host di terze parti può fissare un checksum e fallire in modo esplicito su
una discrepanza.

Alcuni download stampano un avviso di licenza prima di iniziare. Quegli avvisi
fanno parte del percorso di download e non si possono sopprimere tramite
configurazione.

## Backend di validazione

`model.val()` accetta `faster_coco_eval=True` per impostazione predefinita e
ricade su pycocotools quando il pacchetto non è installato, avvisando una sola
volta. Impostare `LIBREYOLO_FASTER_COCO_EVAL` sovrascrive il flag della singola
chiamata, ed è quello che dovrebbe usare un'infrastruttura di benchmark che non
può toccare le configurazioni delle singole esecuzioni. Il backend che ha
davvero girato viene riportato su `model.last_eval_backend`.

## Script di download dei dataset

Un YAML di dataset può contenere un campo `download` con del codice Python. Non
viene eseguito a meno che non si passi `allow_download_scripts=True` alla
chiamata che lo legge, ed è un argomento di funzione su `val()` e `export()`
anziché una variabile d'ambiente.
