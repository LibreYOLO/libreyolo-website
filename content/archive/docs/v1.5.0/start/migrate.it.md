---
title: Importare pesi esistenti
seo_title: Caricare pesi upstream in LibreYOLO
description: >-
  Indica a LibreYOLO un checkpoint di un progetto upstream. La conversione
  automatica lo reimpacchetta al momento del caricamento, conservando numero e
  nomi delle classi.
lead: >-
  LibreYOLO porta le sue famiglie di modelli dai progetti upstream, quindi i
  checkpoint che questi rilasciano sono già quasi caricabili. Quello che manca
  sono i metadati. La conversione automatica li fornisce al momento del
  caricamento.
keywords:
  - convertire pesi libreyolo
  - caricare checkpoint upstream
  - migrazione libreyolo
  - convertire pth in libreyolo
  - conversione automatica checkpoint
last_verified: 1.5.0
meta:
  - label: Punto di ingresso
    value: LibreYOLO("path/to/upstream.pth")
    mono: true
  - label: Scritto accanto alla sorgente come
    value: '<source>-<Prefix><size>[-task].pt'
    mono: true
  - label: Convertitori da script
    value: weights/ nel repository
    mono: true
snippets:
  convert:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Sostituisci il percorso di un checkpoint che hai già. Un layout
        # upstream riconosciuto viene convertito al volo, scritto accanto
        # alla sorgente e poi caricato.
        model = LibreYOLO("path/to/upstream-checkpoint.pth")

        # Numero e nomi delle classi vengono dai tensori e dai metadati del
        # file stesso, così un fine-tune mantiene il suo set di etichette
        # invece di quello di COCO.
        print(model.family, model.size, model.task, model.nb_classes)
        print(model.names)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=path/to/upstream-checkpoint.pth \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Controllare il risultato
      language: bash
      code: |
        # Il file convertito soddisfa lo stesso schema di uno pubblicato.
        libreyolo metadata path=path/to/upstream-checkpoint-LibreYOLO9t.pt
source_hash: bf9d7c7d168fd2c0
---

Questa pagina parla dei checkpoint che arrivano da altri progetti. Se invece
stai spostando il tuo codice da una versione più vecchia di LibreYOLO, vedi
[aggiornare alla 1.5.0](/docs/upgrade).

## Cosa succede quando carichi un file esterno

`LibreYOLO()` carica qualsiasi file di pesi passando prima dal percorso
ristretto, quello weights-only. Se il risultato porta con sé metadati LibreYOLO
completi, viene usato direttamente. Se non li porta, il file passa al
convertitore automatico prima di qualsiasi altro tentativo. Se il
caricamento ristretto fallisce del tutto, cosa che succede quando in un
checkpoint è stato serializzato con pickle un oggetto di terze parti, il
convertitore automatico viene provato con un loader che neutralizza quegli
oggetti.

La conversione automatica fa quattro cose. Estrae il dizionario dei tensori dal
layout usato dal progetto upstream, qualunque esso sia. Chiede a ogni famiglia
registrata se riconosce le chiavi risultanti, rimappando i nomi là dove la
nomenclatura upstream differisce da quella del port di LibreYOLO. Impacchetta la
famiglia vincente in un checkpoint che soddisfa la versione 1.0 dello schema dei
metadati, leggendo dimensione, task e numero di classi dai tensori stessi. Poi
scrive il risultato accanto al file sorgente e carica quello.

<code-tabs name="convert" />

La conversione non è silenziosa. Un file convertito viene registrato nel log con
la famiglia, il nome della sorgente, il nome dell'output e il numero di classi
risultante, così il log di un'esecuzione riporta esattamente cosa è stato
caricato.

## I layout che spacchetta

I checkpoint upstream annidano i pesi in una manciata di posti convenzionali, e
il convertitore li prova in ordine finché uno non contiene tensori: un blocco EMA
sotto `ema.module` o un `ema` piatto, un `ema_state_dict` con il prefisso
`module.` rimosso, poi `params_ema`, `params`, `ema_net`, `net`, `model`,
`state_dict` e infine l'oggetto stesso. Provarne diversi invece di fermarsi al
primo significa che un blocco `ema` che contiene solo contatori non nasconde i
pesi veri che stanno sotto.

Anche i prefissi dei wrapper vengono rimossi: `module.` dell'addestramento
distribuito, `_orig_mod.` di un modello compilato e l'annidamento `model.model.`
che alcune redistribuzioni aggiungono.

## Cosa legge, e da dove

Dimensione, task e numero di classi vengono dai tensori, non dal nome del file,
ed è per questo che un checkpoint sottoposto a fine-tuning si converte con il
proprio numero di classi invece che con quello predefinito dell'architettura. I
nomi delle classi vengono presi dai metadati del checkpoint stesso quando ci
sono, da un blocco `args` o `hyper_parameters` se i nomi stanno lì, e vengono
tagliati al numero di classi rilevato, così un fine-tune che ha mantenuto il suo
set di etichette di base non si porta dietro indici che la sua testa non ha più.

I task densi sono gestiti in modo esplicito, invece di ricevere etichette
inventate. Un checkpoint di profondità ottiene una sola classe di nome `depth`,
un checkpoint di restauro una sola classe di nome `image`. Un checkpoint di posa
deve produrre un numero di keypoint, dai tensori o dalla famiglia; se nessuno dei
due lo fornisce, la conversione viene rifiutata invece di scrivere un file
incompleto.

RF-DETR ha un riconoscitore tutto suo, perché il rilevamento della dimensione
richiede l'intero checkpoint e perché la sua testa ha 91 uscite, dove LibreYOLO
usa la convenzione COCO a 80 classi. Un checkpoint viene normalizzato a 80 classi
quando porta esattamente 80 nomi, o dichiara un numero di classi pari a 80, o
indica COCO come proprio dataset, o non porta alcun metadato di classe o di
dataset. Un vero modello a 90 classi, identificato dai suoi nomi, da un conteggio
esplicito diverso da 80 o da un indizio di dataset non COCO, viene conservato
com'è.

## Dove finisce il file convertito

L'output viene scritto accanto al file sorgente e ne riprende il nome:

```text
<source stem>-<FilenamePrefix><size>[-<task suffix>].pt
```

Un rilevatore YOLOv9 tiny salvato come `upstream-checkpoint.pth` diventa quindi
`upstream-checkpoint-LibreYOLO9t.pt`. Dargli il nome della sorgente invece che
quello della famiglia significa che due fine-tune della stessa famiglia e della
stessa dimensione nella stessa directory non si sovrascrivono a vicenda, e che
nessuno dei due va in conflitto con un checkpoint ufficiale. Il file viene
riscritto a ogni caricamento, quindi non diventa mai obsoleto rispetto alla sua
sorgente. Se la directory è in sola lettura, il file convertito finisce invece in
una nuova directory temporanea privata e il log dice dove.

Da quel momento è un normale checkpoint LibreYOLO: si carica attraverso il
percorso dei metadati e `libreyolo metadata` lo riporta come valido.

## Casi che richiedono una mano

Due famiglie stanno fuori dal riconoscitore generico. La famiglia gaze è esclusa
del tutto: funziona solo in inferenza e i pesi rilasciati portano restrizioni di
ridistribuzione. RF-DETR è escluso perché ha il riconoscitore dedicato descritto
sopra, che è quello che se ne occupa al suo posto.

I checkpoint grezzi di PIDNet upstream vengono rifiutati, con un errore che rimanda
a `weights/convert_pidnet_weights.py`. Quello script scrive i metadati semantici
Cityscapes di cui il checkpoint ha bisogno.

D-FINE e DEIM condividono le stesse chiavi dell'architettura, quindi i soli
tensori non bastano a distinguerli. Quando entrambi rivendicano un file e in gara
non c'è nessuna famiglia sorella con un marcatore distintivo, decide il nome del
file: un nome nella forma `dfine_hgnetv2_n_coco.pth` o `deim_hgnetv2_n_coco.pth`
chiude la questione, mentre un nome che non dice nulla porta a un rifiuto con
quella spiegazione, invece che a un'ipotesi. Anche istanziare direttamente
`LibreDFINE` o `LibreDEIM` risolve il caso.

Quando più famiglie rivendicano legittimamente lo stesso file, una sottoclasse
batte la classe base che raffina, e il resto lo decide l'ordine del registro,
dato che quell'ordine codifica quanto è specifico il controllo di ogni famiglia.
Il nome del file viene consultato solo per il pareggio tra D-FINE e DEIM, quindi
il nome di un file non può mai far vincere una corrispondenza generica su una
precisa.

## I convertitori da script

Il repository contiene script di conversione per singola famiglia sotto
`weights/`, più helper condivisi per le parti ripetitive. Sono la strada da
seguire per un file che il percorso a runtime rifiuta, per produrre un checkpoint
in anticipo invece che al momento del caricamento, e per le famiglie i cui
metadati vanno forniti invece che dedotti dai tensori.

Quegli script fanno parte del repository, non del pacchetto installato, quindi
usarne uno significa clonare:

```bash
git clone https://github.com/LibreYOLO/libreyolo.git
cd libreyolo
python weights/convert_pidnet_weights.py --help
```

Ogni script scrive un checkpoint che soddisfa la versione 1.0 dello schema, cioè
lo stesso requisito che soddisfa la conversione automatica e lo stesso che
soddisfano i pesi pubblicati. Vedi [checkpoint e pesi](/docs/weights) per sapere
cosa contiene quello schema.
