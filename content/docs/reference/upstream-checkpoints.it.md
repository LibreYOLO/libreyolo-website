---
title: Checkpoint upstream
seo_title: Caricare checkpoint upstream in LibreYOLO
description: >-
  Come la conversione automatica trasforma un checkpoint upstream rilasciato in
  uno LibreYOLO v1.0: i layout che spacchetta, quali famiglie riconoscono cosa e
  dove si ferma.
lead: >-
  Le famiglie di LibreYOLO sono portate da progetti upstream i cui checkpoint
  rilasciati sono quasi caricabili ma non portano metadati LibreYOLO. La
  conversione automatica riconosce quei file, li avvolge nello schema v1.0 e
  scrive il risultato accanto al sorgente.
keywords:
  - libreyolo autoconvert
  - caricare checkpoint upstream
  - convert_upstream_state_dict
  - pesi upstream libreyolo
  - conversione checkpoint pytorch
last_verified: 1.5.0
verification: >-
  Comportamento letto da libreyolo/models/autoconvert.py e da
  BaseModel.convert_upstream_state_dict; i riconoscitori per famiglia sono stati
  controllati leggendo l'override di convert_upstream_state_dict di ogni
  famiglia, tutto alla v1.5.0. Le regole COCO di RF-DETR vengono da
  docs/checkpoint_schema.md.
snippets:
  usage:
    - label: Basta passare il file alla factory
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Un file upstream riconosciuto viene convertito al caricamento, e il
        # checkpoint convertito viene scritto accanto a lui.
        # model = LibreYOLO("yolov9-t-converted.pt")

        # Qualsiasi checkpoint LibreYOLO si carica senza modifiche.
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model.family, model.size, model.task, model.nb_classes)
source_hash: c6022771a2a207a1
---

## Cosa succede al caricamento

Quando `LibreYOLO()` incontra un file `.pt` che non è già un checkpoint v1.0
completo, chiama il convertitore automatico, che:

1. spacchetta il dizionario dei tensori dai layout upstream più comuni;
2. chiede a ogni famiglia registrata se riconosce il layout, rimappando le
   chiavi dove la nomenclatura upstream differisce dal port nativo;
3. avvolge il vincitore in un checkpoint con metadati v1.0 rigorosi, leggendo
   dimensione, task e numero di classi dai tensori stessi, così i checkpoint
   su cui è stato fatto fine-tuning si convertono correttamente;
4. lo scrive accanto al sorgente come `<source>-<Prefix><size>[-task].pt` e
   restituisce quel percorso, così la factory lo carica normalmente.

Al chiamante non viene chiesto nulla. Un file che nessuna famiglia rivendica
non restituisce niente e la factory segnala che non è riuscita a caricarlo.

<code-tabs name="usage" />

## I layout che spacchetta

Il dizionario dei tensori viene cercato in questo ordine di preferenza, prima
l'EMA, e ogni candidato viene provato finché uno contiene davvero dei tensori.
Un blocco EMA vuoto o con soli metadati non nasconde quindi i pesi validi che
stanno sotto.

| Chiave | Nota |
|---|---|
| `ema.module` | Il wrapper EMA più comune |
| `ema` | Vecchi wrapper EMA piatti che memorizzano i tensori direttamente |
| `ema_state_dict` | Alle voci sotto un prefisso `module.` il prefisso viene rimosso |
| `params_ema` | |
| `params` | |
| `ema_net` | |
| `net` | |
| `model` | |
| `state_dict` | |
| Il file stesso | Uno state dict semplice |

Ogni candidato viene poi ristretto alle voci che contengono tensori e
normalizzato: un prefisso iniziale `module.` o `_orig_mod.` viene rimosso, e a
un dizionario le cui chiavi iniziano tutte con `model.model.` viene tolto quel
prefisso.

## Quali famiglie riconoscono cosa

Il riconoscimento è un classmethod per famiglia. L'implementazione predefinita
rivendica un layout le cui chiavi corrispondono già al port nativo. Una
famiglia la cui nomenclatura delle chiavi upstream è diversa la sovrascrive
con una rimappatura, e non restituisce nulla per i layout che non riconosce.

Famiglie che includono un riconoscitore con rimappatura: `centernet`,
`deeplabv3`, `deformable_detr`, `dexined`, `moge2`, `picodet`, `rtdetr`,
`rtdetrv2`, `rtdetrv4`, `rtmdet`, `segformer`, `swin`, `teed`, `yolo7`,
`yolo9`, `yolo9_e2e`, `yolo9_p2`.

Famiglie che rifiutano del tutto la conversione automatica: `efficientdet`,
`eomt` e `pidnet` non restituiscono nulla dal riconoscitore, quindi i loro
file upstream passano invece per uno script di conversione. `l2cs` è escluso
dal riconoscitore generico perché è solo per inferenza e ha pesi con
restrizioni sulla ridistribuzione.

RF-DETR mantiene un riconoscitore proprio, perché per rilevare la dimensione e
rimappare le classi COCO gli serve l'intero checkpoint e non solo il
dizionario dei tensori. Viene registrato solo quando le sue dipendenze
opzionali sono installate.

Ogni altra famiglia registrata usa l'implementazione predefinita: rivendica il
file quando il suo stesso loader riconosce già quelle chiavi.

## Quale famiglia vince

Più famiglie possono rivendicare lo stesso file, quindi la risoluzione
rispecchia le regole di dispatch della factory.

La rivendicazione di una sottoclasse batte quella della sua classe base.
L'ordine di registrazione segue la creazione delle classi, quindi una famiglia
derivata si registra dopo la base che raffina, e i suoi marcatori positivi non
devono perdere contro il passthrough più ampio della base.

Decide poi l'ordine del registro, perché codifica la specificità: la
rivendicazione che arriva per prima è la corrispondenza più specifica.

L'unico pareggio che l'ordine del registro non riesce a sciogliere è DEIM
contro D-FINE, le cui chiavi architetturali sono identiche. Lì, e solo lì, il
nome del file è il segnale decisivo, e un file il cui nome non dà indizi viene
rifiutato invece che indovinato. Altrove il nome del file non viene consultato
di proposito, così una rivendicazione ampia e falsamente positiva non può mai
essere promossa sopra una più specifica solo per come si chiama il file.

## Caricamento sicuro

I file upstream vengono caricati attraverso l'unpickler weights-only. Alcuni
checkpoint di addestramento upstream incorporano oggetti di libreria che
quell'unpickler rifiuta. Quegli oggetti sono metadati di addestramento e non
pesi, quindi ogni global bloccato viene ritentato con una classe sostitutiva
inerte che soddisfa l'unpickler senza eseguire nulla. Il nome catturato viene
usato solo come etichetta testuale, mai importato, valutato o chiamato.

I nomi di moduli sensibili vengono rifiutati del tutto e mai sostituiti da uno
stub: `builtins`, `os`, `sys`, `posix`, `nt` e `subprocess`. Il ciclo di
ritentativi è limitato a 32 tentativi, così un file costruito apposta per
introdurre una serie illimitata di global diversi fallisce in sicurezza invece
di girare a vuoto. Nel checkpoint convertito sopravvivono solo i tensori.

## Dove finisce il file convertito

L'output viene scritto accanto al sorgente, con il nome
`<source>-<Prefix><size>[-task].pt`. Viene sempre riscritto invece che
riutilizzato: così i caricamenti ripetuti dello stesso sorgente restano
aggiornati, evitando collisioni con i pesi ufficiali o con un altro
fine-tuning della stessa famiglia, dimensione e task nella stessa directory.

Quando la directory del sorgente è di sola lettura, la conversione ripiega su
una directory temporanea privata creata a ogni chiamata, e la riga di log
indica il percorso usato. Solo se anche questo fallisce la conversione viene
abbandonata, con un avviso.

## Checkpoint LibreYOLO già esistenti

Un file che porta un marcatore specifico di LibreYOLO, `libreyolo_version` o
`model_family`, appartiene al percorso di caricamento normale e non viene
riconvertito. Il salto vale solo per una rivendicazione passthrough, cioè una
in cui l'insieme delle chiavi è rimasto invariato. Una rivendicazione la cui
conversione ha cambiato l'insieme delle chiavi è la prova di un layout
upstream estraneo e viene accettata anche su un file marcato.

`schema_version` non viene trattata come marcatore di proposito, perché altri
strumenti di addestramento e di esportazione usano quel nome generico, e non
lo sono nemmeno `names`, `nc`, `size`, `task` o `imgsz`, perché anche un
fine-tuning upstream può portarle. Un fine-tuning estraneo che porta soltanto
una chiave `names` generica non risulta quindi marcato, così la sua
rivendicazione con chiavi native si converte normalmente e ricava il numero di
classi dalla testa dei tensori invece di essere caricato per errore come
modello a 80 classi.

## I metadati che il convertitore legge

I nomi delle classi vengono presi da una chiave `names` di primo livello,
oppure da `class_names` dentro un blocco `args` o `hyper_parameters`. Una
mappa dei nomi indicizzata per etichetta invece che per indice di classe è
inutilizzabile e viene sostituita da valori predefiniti generati. Una lista di
nomi più lunga del numero di classi rilevato viene tagliata, perché gli indici
fuori intervallo farebbero fallire il validatore rigoroso e interromperebbero
la conversione senza dirlo.

Gli `args` upstream vengono riportati come semplici metadati, scartando
qualsiasi valore che non sia una stringa, un numero, un booleano, una lista o
un dizionario, così nel file salvato non finisce nulla di rischioso.

## Normalizzazione COCO per RF-DETR

I checkpoint RF-DETR upstream espongono una testa di classificazione a 91
uscite, cioè le 90 classi di COCO più lo sfondo. La conversione automatica
normalizza un RF-DETR COCO alla convenzione COCO-80, con la rimappatura
applicata nel postprocessing.

Un checkpoint viene trattato come COCO quando porta esattamente 80 nomi, o
dichiara un numero di classi pari a 80, o ha un indizio di dataset `coco`, o
non ha alcun metadato di classe o di dataset. Quest'ultimo caso conta: uno
state dict upstream nudo è il checkpoint canonico preaddestrato su COCO, ed è
l'unico RF-DETR a 91 uscite senza metadati in circolazione.

Un vero RF-DETR personalizzato a 90 classi viene conservato con 90 classi. Lo
si riconosce da una lista di nomi, da un numero di classi esplicito diverso da
80 o da un indizio di dataset non COCO, così il fallback del checkpoint nudo
non scatta per lui. I segnaposto vuoti vengono ignorati quando si decide se un
indizio di dataset è presente.

## Limiti

La conversione automatica riconosce i layout upstream rilasciati. Non riscrive
un'architettura e non rende caricabile un modello non portato. Quando nessuna
famiglia rivendica un file, la risposta è uno script di conversione e non un
argomento della factory: il repository include `weights/convert_*.py` per le
famiglie che ne hanno bisogno, tra cui EoMT, PIDNet ed EfficientDet.

La conversione inoltre non inventa i metadati che non riesce a leggere.
Dimensione, task e numero di classi vengono dai tensori; i nomi vengono dal
file quando ci sono, e altrimenti sono generati come `class_i`.
