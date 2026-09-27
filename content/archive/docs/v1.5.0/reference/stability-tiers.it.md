---
title: Tier di stabilità
seo_title: Cosa significa ogni tier di supporto di LibreYOLO
description: >-
  Il vocabolario dei tier che usa LibreYOLO: i tre tier di supporto
  all'esportazione, i quattro tier di API, i sei gruppi di copertura e ciò che
  nessuno di loro promette.
lead: >-
  LibreYOLO usa la parola tier per tre cose diverse: le prove dietro un percorso
  di esportazione, il contratto di chiamata a cui risponde una famiglia di
  modelli e il gruppo di copertura a cui la famiglia è iscritta. Questa pagina
  definisce ciascuno di essi e dice che cosa non implica.
keywords:
  - tier supporto libreyolo
  - validated available blocked
  - tier esportazione libreyolo
  - gruppi di copertura libreyolo
  - g0 g1 g2 g3 g4
  - tier modelli yolo
last_verified: 1.5.0
verification: >-
  Tier di esportazione da docs/adr/0011-export-support-tiers.md e
  libreyolo/export/support.py; gruppi di copertura e conteggi per famiglia da
  MODEL_GROUPS in libreyolo/models/registry.py; il gate da zero da
  libreyolo/models/base/model.py e libreyolo/cli/commands/train.py; l'inventario
  della CLI letto da libreyolo/models/inventory.py; i tier di API dai docstring
  dei package libreyolo/models/sam/, openvocab/ e vlm/ e dai contratti in
  base.py, tutto alla v1.5.0. Le etichette di gruppo mostrate al lettore
  (Flagship, Core, Supportate, Solo inferenza, Museo, Tier gemello) sono il
  vocabolario del sito per gli stessi gruppi, da src/data/docs/registry.json.
snippets:
  usage:
    - label: Leggere entrambe le classificazioni per una famiglia
      language: python
      code: |
        from libreyolo.models.registry import GROUPS, group_of
        from libreyolo.export.support import get_support, validated_alternatives

        family = "yolo9"

        group = group_of(family)
        print(group, GROUPS[group])

        print(get_support(family, "detect", "onnx").tier)
        print(validated_alternatives(family, "detect"))
source_hash: de545894b0d125e4
---

## Tier di supporto all'esportazione

Il tier che decide se una chiamata riesce. Vale per la tripla
`(family, task, format)`, e ogni combinazione ne ha esattamente uno.

| Tier | Significato | Cosa succede con `export()` |
|---|---|---|
| `validated` | La parità numerica è coperta in CI o da una run notturna documentata | Viene eseguita |
| `available` | La conversione è implementata, ma non è stata registrata alcuna prova di parità numerica a runtime | Viene eseguita |
| `blocked` | Nessun percorso supportato | Solleva `NotImplementedError` nel preflight, con il motivo |

Sia validated sia available procedono senza una conferma esplicita né un
avviso generico. La differenza sta nelle prove, non nel permesso: una voce
validated ha dietro un test di parità e una release `since`, una voce
available non ancora. Una conversione CoreML senza una run di predizione su
macOS, per esempio, è available e non validated.

Una combinazione blocked fallisce prima dei controlli sulle dipendenze, del
caricamento della calibrazione, del tracing o della creazione dell'artefatto,
quindi non viene scritto nulla di parziale.

Ogni cella validated porta con sé un vincolo che descrive la configurazione da
cui arriva il numero di parità, di solito un canvas di input fisso, batch 1,
FP32 e una versione di runtime specifica. Leggilo come un'affermazione su
quella configurazione e non sul formato in generale. Le regole che riempiono
le celle senza una voce esplicita sono nella pagina
[matrice di esportazione](/docs/reference/export-matrix).

<code-tabs name="usage" />

## Tier di API

Il tier che decide che aspetto ha una chiamata. Una famiglia sta in
esattamente uno, scelto in base al contratto di chiamata e non
all'architettura.

| Tier | Factory | Contratto |
|---|---|---|
| Factory dei detector | `LibreYOLO` | Un singolo forward senza prompt restituisce ogni oggetto trovato, con punteggi calibrati. I membri si registrano da soli riconoscendo un checkpoint |
| Segmentazione con prompt | `LibreSAM` | Un forward non ha senso senza un prompt spaziale o concettuale per immagine, fornito al momento della chiamata. Interattivo e con stato: codifica una volta, invia prompt molte volte |
| Rilevamento a vocabolario aperto | `LibreOpenVocab` | Detector discriminativi condizionati dal testo. L'elenco delle classi è un prompt, impostato da `set_classes` |
| Vision-language | `LibreVLM` | Un modello generativo guidato come un detector. L'elenco delle classi è un prompt e la confidenza è un segnaposto |

I tre tier gemelli deliberatamente non si registrano nella factory dei
detector, ed è per questo che `LibreYOLO("some-alias")` non li raggiunge. Si
caricano per alias di dimensione e download automatico, non tramite
l'ispezione del checkpoint.

Tutti e quattro restituiscono lo stesso `Results`, quindi il codice a valle
non cambia passando dall'uno all'altro. Cambia quali metodi funzionano: i tier
gemelli sollevano `NotImplementedError` per `train()`, `val()` ed `export()`,
e i tier SAM e a vocabolario aperto sollevano anche per `track()`. La pagina
di ogni tier elenca le proprie esclusioni.

## Gruppi di copertura

La classificazione che decide quali famiglie entrano in una run di test tra
famiglie, e quella che un lettore ha più probabilità di incontrare su una
pagina di modello. Ogni famiglia registrata è iscritta a esattamente un
gruppo, e un test fallisce quando una famiglia registrata manca
dall'iscrizione. `GROUPS` in `libreyolo/models/registry.py` è la fonte della
colonna Significato qui sotto; `MODEL_GROUPS` nello stesso file assegna ogni
famiglia, e la colonna Famiglie conta direttamente quell'assegnazione. La
colonna Etichetta è il nome più breve che il sito usa per lo stesso gruppo
nell'intestazione di una pagina di modello.

| Gruppo | Etichetta | Famiglie | Significato |
|---|---|---|---|
| `g0` | Flagship | 2 | Riferimenti flagship obbligatori nella copertura delle funzionalità condivise |
| `g1` | Core | 10 | Insieme di copertura dei detector addestrabili |
| `g2` | Supportate | 14 | Insieme di copertura aggiuntivo per le famiglie addestrabili |
| `g3` | Solo inferenza | 35 | Famiglie senza un'implementazione dell'addestramento |
| `g4` | Museo | 5 | Famiglie storiche con copertura in inferenza |
| `s` | Tier gemello | 21 | API gemelle (SAM, open-vocab, VLM, zero-shot) coperte separatamente |

In totale sono 87 famiglie in sei gruppi. Da solo `g3` contiene più famiglie
di tutti gli altri gruppi messi insieme, perché gran parte del registry è
lignaggio solo in inferenza e copertura da museo, più che detector addestrati
attivamente.

Per chi sta scegliendo un modello, il gruppo dice dove aspettarsi attenzione
ingegneristica, non quanto è accurata una famiglia. `g0` e `g1` sono i gruppi
in cui una nuova funzionalità viene progettata e arriva per prima; `g2` viene
tenuto verde in CI, ma una funzionalità ci arriva quando capita e non sulla
stessa ondata di release. `g3` dichiara un'assenza, non un limite: predizione,
validazione e, dove la famiglia lo supporta, esportazione funzionano ancora
tutte, e `train()` su una famiglia `g3` o `g4` solleva `NotImplementedError`
indicando il motivo, invece di fare qualcosa di parziale in silenzio. Le
famiglie `s` non stanno affatto in questo compromesso, perché si caricano
attraverso la loro factory e non tramite `LibreYOLO()`. Vedi
[concetti fondamentali](/docs/concepts) per capire come un gruppo si inserisce
accanto a task, famiglia e dimensione quando leggi il nome di file di un
checkpoint.

Un gruppo di per sé non concede né limita una capacità visibile all'utente. Il
supporto arriva dall'API implementata dalla famiglia e dai controlli di
capacità specifici del formato, mai dalla sola appartenenza a un gruppo. I
gruppi classificano le famiglie, non i task, quindi una run di copertura
limitata a un task nomina il task esplicitamente, come in "g1 detect".

Due punti leggono il gruppo a runtime e non solo nei test.
`collect_model_inventory()` in `libreyolo/models/inventory.py` allega il
gruppo a ogni voce che l'inventario della CLI stampa, e `pretrained=False`
attiva il percorso speciale di reinizializzazione da zero solo per le famiglie
in `g0` e `g1`. Fuori da questi due gruppi il controllo in
`libreyolo/models/base/model.py` viene saltato del tutto, quindi
`pretrained=False` arriva al `train()` della famiglia stessa come un normale
argomento keyword.

## Addestramento

Una famiglia in `g3` o `g4` non ha un'implementazione dell'addestramento, e
chiamare `train()` su una di esse solleva un errore. È una proprietà del
codice della famiglia, non del suo gruppo: il gruppo registra il fatto, non lo
causa.

Per una famiglia che invece si addestra, se una singola manopola di data
augmentation arrivi alla pipeline è una questione a parte, con un suo
vocabolario a tre valori, `used`, `gated_by_mosaic` e `ignored`. Vedi la
[matrice della data augmentation](/docs/reference/augmentation-matrix).

## Cosa un tier non ti dice

Un tier non è un'affermazione sull'accuratezza. Un'esportazione validated dice
che l'artefatto riproduce il modello nativo entro una soglia dichiarata; non
dice nulla su quanto bene il modello nativo si comporti su un dataset. I
numeri dei benchmark stanno sulle pagine dei modelli.

Un tier non è nemmeno una dichiarazione sulle licenze. Le licenze dei pesi
variano all'interno di una stessa famiglia e fa fede il repository che ospita
uno specifico checkpoint. Il fatto che una famiglia sia nella factory dei
detector non dice nulla sul fatto che i suoi pesi pubblicati permettano l'uso
commerciale.
