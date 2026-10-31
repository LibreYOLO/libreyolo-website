---
title: Congelamento dei layer
seo_title: Congelare i layer durante l'addestramento in LibreYOLO
description: >-
  Congela una parte del modello per il transfer learning: un numero intero di
  gruppi di congelamento della famiglia, una lista esplicita di indici, oppure
  selettori per nome di modulo e di parametro.
lead: >-
  Il congelamento tiene fissi i pesi selezionati mentre il resto del modello si
  addestra. I selettori fanno riferimento ai gruppi di congelamento ordinati
  propri di una famiglia o ai suoi nomi di modulo, non a numeri di layer grezzi
  presi da un grafo YAML.
keywords:
  - congelare layer yolo
  - transfer learning yolo
  - freeze backbone yolo
  - batchnorm congelato
  - gruppi di congelamento
  - addestrare solo la testa
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # I primi 10 gruppi sono l'intero backbone di YOLOv9.
        model.train(data="my-dataset.yaml", epochs=50, freeze=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=50 freeze=10
    - label: Per nome
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, freeze="backbone")
    - label: Più selettori
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", freeze=["backbone", "neck"])
  groups:
    - label: Elencare in ordine i gruppi di congelamento di una famiglia
      language: python
      code: |
        from libreyolo import LibreYOLO9
        from libreyolo.models.yolo9.trainer import YOLO9Trainer

        model = LibreYOLO9("LibreYOLO9s.pt", size="s")
        trainer = YOLO9Trainer(model=model.model, wrapper_model=model, size="s")

        for index, (name, _module) in enumerate(trainer.get_freeze_groups()):
            print(index, name)
source_hash: 9f1e7551af6b16fe
---

## Congela qualcosa

`freeze` è opzionale e di default non congela nulla.

<code-tabs name="train" />

Il congelamento avviene dopo la costruzione del modello e dopo l'eventuale
ricostruzione della testa per un nuovo numero di classi, e prima che venga creato
l'ottimizzatore, così l'ottimizzatore riceve sempre e solo parametri addestrabili.

## Che cosa può essere un selettore

| Valore | Significato |
|---|---|
| `None`, `False`, `""`, `"none"` | Addestra ogni parametro |
| `10` o `"10"` | Congela i primi dieci gruppi di congelamento della famiglia |
| `[0, 3, 7]` | Congela quei gruppi, indicizzati da zero |
| `"backbone"` | Congela il gruppo, il modulo o il prefisso di parametro corrispondente |
| `["backbone", "neck"]` | Congela ognuno dei selettori elencati |
| `["backbone", 3]` | Le liste miste funzionano |

Una stringa viene analizzata prima di essere interpretata, quindi la CLI e una
configurazione YAML accettano le stesse forme di Python. `freeze="[0, 3, 'head']"`
viene letto come una lista letterale, `freeze="backbone,neck"` viene diviso sulla
virgola, e una semplice stringa decimale diventa un conteggio.

`freeze=True` viene rifiutato perché ambiguo.

I selettori per nome corrispondono al nome di un gruppo di congelamento, al nome
di un modulo o a un prefisso di nome di parametro, e i caratteri glob `*`, `?` e
`[` funzionano. Un `model.` iniziale è trattato in modo flessibile, così sia
`backbone` sia `model.backbone` funzionano, qualunque sia la forma usata
internamente dalla famiglia.

## I gruppi li definisce la famiglia

Un intero fa riferimento alla lista ordinata di gruppi di congelamento propria
della famiglia, non a una posizione in un grafo condiviso. Le famiglie di LibreYOLO non
sono tutte un unico modello sequenziale indicizzato da YAML, quindi un numero di
layer grezzo significherebbe una cosa diversa su ciascuna di esse.

YOLOv9 ordina i suoi gruppi a partire dall'ingresso: dieci stadi di backbone, poi
sei stadi di neck, poi la testa. Per questo `freeze=10` è esattamente il backbone.
`backbone`, `neck` e `head` sono selettori per nome stabili costruiti sopra di
esso.

I gruppi di RF-DETR sono `backbone.encoder`, `backbone.projector`, `decoder`,
`queries`, `transformer.encoder_output` e `head`. Qui i nomi sono la scelta
migliore, perché i componenti di un transformer non corrispondono a un conteggio
di layer. `backbone` corrisponde per prefisso a entrambi i gruppi del backbone.

Le famiglie che non definiscono gruppi semantici ripiegano su un default
conservativo: ogni figlio diretto del modello che possiede almeno un parametro, in
ordine di dichiarazione. Di solito è una lista corta, quindi un intero grande non
troverà abbastanza gruppi:

```text
freeze index 10 is out of range for 3 available freeze groups.
```

Per vedere la lista reale invece di tirare a indovinare:

<code-tabs name="groups" />

## Gli errori si fanno sentire

Ogni modo di sbagliare solleva un errore invece di addestrare qualcosa che non hai
chiesto.

Un selettore che non corrisponde a nulla solleva un errore, indicando quali
selettori hanno mancato il bersaglio:

```text
freeze selector(s) matched no parameters: 'backbon'
```

Un congelamento che non lascerebbe nulla di addestrabile solleva un errore, sia al
momento del congelamento sia di nuovo quando viene costruito l'ottimizzatore:

```text
freeze would leave no trainable parameters. Use a smaller freeze value or
target a narrower module.
```

Che è quello che fa `freeze="all"`, dato che `all` corrisponde a ogni parametro.

Quando il congelamento riesce, una riga registra quello che è successo:

```text
Layer freezing: selectors=[10], tensors=124, params=2103776, trainable=1863456/3967232
```

## Il BatchNorm congelato smette di aggiornarsi

Un parametro congelato sta comunque dentro un modulo le cui statistiche di
running continuerebbero a muoversi. Ogni modulo di tipo BatchNorm i cui parametri
finiscono nell'insieme congelato viene messo in modalità eval, e il trainer la
riapplica dopo la chiamata a `model.train()` di ogni epoca, così le statistiche
restano fisse per tutta l'esecuzione.

È attivo di default ed è ciò che fa sì che congelare un backbone lo congeli
davvero.

## Combinare con LoRA

`freeze` e `lora=True` funzionano insieme. Su RF-DETR, DEIM e ConvNeXt i parametri
degli adapter restano addestrabili anche quando il gruppo che li contiene è
congelato, che è la combinazione che vuoi: un backbone congelato con gli adapter
che imparano sopra di esso. Vedi [Fine-tuning con LoRA](/docs/train/lora).

## Ambito

Questo è un congelamento statico deciso all'avvio. Lo scongelamento programmato e
il congelamento progressivo non fanno parte dell'interfaccia.

## Correlati

- [Iperparametri](/docs/train/hyperparameters) per il resto di `train()`.
- [Distillazione](/docs/train/distillation) per l'altro modo di trasferire la
  conoscenza di un modello grande in un addestramento.
