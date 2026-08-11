---
title: Checkpoint e pesi
seo_title: Checkpoint e pesi di LibreYOLO
description: >-
  Come LibreYOLO trova, scarica e verifica i pesi dei modelli, dove sono
  ospitati, come eseguire senza rete e cosa rende sicuro il caricamento di un
  checkpoint.
lead: >-
  Un checkpoint di LibreYOLO è un dizionario torch.save che contiene uno state
  dict più i metadati necessari a identificarlo. Questa pagina spiega da dove
  arrivano quei file, dove finiscono e come vengono caricati.
keywords:
  - pesi libreyolo
  - checkpoint libreyolo
  - scaricare pesi libreyolo
  - libreyolo offline
  - libreyolo hugging face
  - metadati checkpoint
last_verified: 1.5.0
meta:
  - label: Ospitati su
    value: 'Un repository Hugging Face per ogni checkpoint:'
    links:
      - label: huggingface.co/LibreYOLO
        href: 'https://huggingface.co/LibreYOLO'
  - label: Cache locale
    value: weights/ sotto la directory di lavoro
    mono: true
  - label: Schema dei metadati
    value: v1.0
snippets:
  load:
    - label: Download automatico
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Un nome di file semplice si risolve in weights/LibreYOLO9t.pt e
        # viene scaricato lì se non è già presente.
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model(SAMPLE_IMAGE).boxes)
    - label: Percorso esplicito
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Un percorso con componente di directory viene usato esattamente
        # come è scritto e non viene mai scaricato dalla rete.
        model = LibreYOLO("/opt/models/LibreYOLO9t.pt")
        print(model.family, model.size, model.task)
  inspect:
    - label: CLI
      language: bash
      code: |
        # Legge i metadati senza costruire un modello, e riporta se
        # soddisfano lo schema.
        libreyolo metadata path=weights/LibreYOLO9t.pt
    - label: JSON
      language: bash
      code: |
        libreyolo metadata path=weights/LibreYOLO9t.pt --json
    - label: Python
      language: python
      code: >
        from libreyolo.utils.serialization import (
            load_untrusted_torch_file,
            validate_checkpoint_metadata,
        )


        loaded = load_untrusted_torch_file("weights/LibreYOLO9t.pt")


        # Restituisce una lista di problemi. Vuota significa che il file
        soddisfa la v1.0.

        print(validate_checkpoint_metadata(loaded))

        print(loaded["model_family"], loaded["size"], loaded["task"],
        loaded["nc"])
source_hash: 210a12baa1417cfb
---

## Dove viene cercato un checkpoint

Un riferimento a un modello senza componente di directory, come `LibreYOLO9t.pt`,
viene risolto rispetto a `weights/` relativo alla directory di lavoro corrente.
Se `weights/LibreYOLO9t.pt` esiste viene usato quello; se un file con quel nome
esiste nella directory di lavoro stessa viene usato quest'ultimo; altrimenti
`weights/LibreYOLO9t.pt` diventa la destinazione del download.

Un riferimento che invece contiene una directory, assoluta o relativa, viene
preso alla lettera. È la forma da usare quando i pesi stanno in una posizione
centralizzata e non si deve scaricare nulla.

<code-tabs name="load" />

## Download automatico

Quando il percorso risolto non esiste, LibreYOLO analizza il nome del file per
ricavare la famiglia, la dimensione e il task, e chiede alla famiglia
corrispondente un URL di download. La maggior parte delle famiglie lo costruisce
a partire dall'organizzazione LibreYOLO su Hugging Face, dove ogni checkpoint ha
un repository tutto suo che porta il nome del file:

```text
https://huggingface.co/LibreYOLO/<name>/resolve/main/<name>.pt
```

Un suffisso di variante del dataset resta parte del nome del repository, così un
checkpoint addestrato su qualcosa di diverso dal dataset predefinito della
famiglia si risolve nel proprio repository invece di sovrascrivere quello
predefinito.

Il trasferimento in sé è difensivo, perché un file di pesi troncato fallisce più
avanti con un errore poco utile. I download vengono scritti in streaming su un
file `.part` e spostati al loro posto in modo atomico solo quando sono completi,
così un processo interrotto non può mai lasciare un checkpoint scritto a metà nel
percorso finale. Un trasferimento interrotto riprende dal proprio offset in byte
usando un validator HTTP, e riparte da zero se il server segnala che l'oggetto è
cambiato. I fallimenti vengono ritentati tre volte con backoff esponenziale. I
processi concorrenti che puntano allo stesso percorso prendono un file di lock,
così due addestramenti che partono insieme scaricano una volta sola. Dove una
famiglia scarica da un host di terze parti invece che dall'organizzazione
LibreYOLO, può fissare un checksum e rifiutare il file in caso di mancata
corrispondenza.

Se `HF_TOKEN` è impostata, o se un token è nella cache in
`~/.cache/huggingface/token`, viene allegato come bearer token. Viene allegato
solo agli URL di `huggingface.co`, quindi una famiglia che scarica da un altro
host non lo riceve mai.

Non tutte le famiglie scaricano automaticamente. Alcune deliberatamente non
restituiscono alcun URL perché i pesi rilasciati non possono essere
ridistribuiti, e l'errore spiega allora cosa fornire al loro posto. Altre
stampano un avviso di licenza prima che il trasferimento inizi. Quell'avviso è il
segnale, a runtime, che i termini di un checkpoint sono più restrittivi di quelli
del codice, e vale la pena leggerlo invece di scorrere oltre.

## L'organizzazione su Hugging Face

I pesi pubblicati stanno su
[huggingface.co/LibreYOLO](https://huggingface.co/LibreYOLO), un repository per
ogni checkpoint. Ogni repository porta con sé una licenza, e la licenza non è
uniforme all'interno di una famiglia: una famiglia il cui codice è MIT può avere
pesi che non lo sono. Il repository fa fede. Ogni pagina di modello elenca i
checkpoint pubblicati di quella famiglia e le loro licenze nelle sezioni
Checkpoint e Licenze.

## Lavorare offline

Nulla nella libreria richiede accesso alla rete una volta che i file sono in
locale. Funzionano due approcci:

Prepopola una directory `weights/` accanto al punto in cui gira il job. Basta
scaricare i checkpoint una volta su una macchina connessa e poi copiare la
directory; il passo di risoluzione descritto sopra li trova e non arriva mai alla
rete.

Oppure passa un percorso assoluto a una posizione condivisa. Un riferimento con
componente di directory viene usato così com'è, quindi un mount in sola lettura
di pesi selezionati è una configurazione valida. Se il processo non può scrivere
accanto a un checkpoint che deve convertire, la conversione ripiega su una
directory temporanea privata invece di fallire.

I dataset seguono una regola a parte: si risolvono sotto `~/datasets`, oppure
sotto la directory indicata da `LIBREYOLO_DATASETS_DIR` quando quella variabile è
impostata.

## Sicurezza del caricamento

I checkpoint sono pickle, e un pickle può eseguire codice arbitrario quando viene
aperto. LibreYOLO tratta ogni file di pesi come non fidato e lo carica con il
percorso `weights_only=True` di PyTorch, che limita l'unpickler ai tensori e a un
piccolo insieme di tipi sicuri. Questo vale per il file che passi tu, non solo
per i file scaricati da LibreYOLO. Su una build di PyTorch troppo vecchia per
supportare quell'argomento, il caricamento viene rifiutato invece di essere
eseguito in modo non sicuro.

Alcuni checkpoint di addestramento upstream contengono oggetti che l'unpickler
ristretto rifiuta, come un oggetto di configurazione del framework con cui sono
stati addestrati. Quegli oggetti sono metadati di cui LibreYOLO non ha bisogno,
quindi durante la conversione ogni classe bloccata viene sostituita da un
sostituto inerte che soddisfa l'unpickler senza eseguire nulla, e nel file
convertito sopravvivono solo i tensori. I nomi di moduli sensibili vengono
rifiutati del tutto invece di essere sostituiti da uno stub, e il ciclo di retry
è limitato, così un file costruito ad arte per introdurre una serie infinita di
classi bloccate fallisce in sicurezza. Vedi
[importare pesi esistenti](/docs/migrate) per il resto di quel percorso.

## Metadati del checkpoint

Un checkpoint di LibreYOLO è un dizionario la cui chiave `model` contiene lo
state dict di PyTorch. Lo schema v1.0 richiede nove chiavi che, insieme,
permettono alla factory di identificare un file senza analizzarne il nome né
tirare a indovinare dalle forme dei tensori.

| Chiave | Significato |
|---|---|
| `model` | Lo state dict di PyTorch |
| `schema_version` | La versione del contratto dei metadati. La v1.0 usa la stringa `1.0` |
| `libreyolo_version` | La versione di LibreYOLO che ha prodotto il file |
| `model_family` | Un identificatore di famiglia registrato, come `yolo9` |
| `size` | La variante all'interno di quella famiglia, come `t` o `r18` |
| `task` | Un nome di task canonico |
| `nc` | Un numero di classi positivo |
| `names` | Una mappa da indice di classe a etichetta, che copre da `0` a `nc - 1` |
| `imgsz` | Una risoluzione di input positiva |

I task con struttura aggiuntiva la registrano accanto a quelle chiavi. I
checkpoint di stima della posa aggiungono `num_keypoints` e `keypoint_dim`, e
possono aggiungere le sigma OKS per keypoint. I checkpoint OCR incorporano
l'intero charset CTC, così il file è autosufficiente. I checkpoint di restore
possono registrare il tipo di degradazione e un fattore di upscale. I checkpoint
del trainer aggiungono lo stato di ripresa, come `epoch`, lo stato
dell'optimizer e i pesi EMA; i pesi di inferenza pubblicati non dovrebbero
contenerli.

Un file che soddisfa tutte e nove le chiavi viene caricato attraverso il percorso
dei metadati. Un file che non le soddisfa viene convertito, se una famiglia ne
riconosce il layout, oppure caricato attraverso il percorso di compatibilità con
un avviso che indica cosa manca.

## Ispezionare un checkpoint

<code-tabs name="inspect" />

`libreyolo metadata` non costruisce mai un modello, quindi funziona su un file la
cui famiglia non è installata e su un file di cui non sei sicuro.
