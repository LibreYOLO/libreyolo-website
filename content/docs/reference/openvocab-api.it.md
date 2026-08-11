---
title: API a vocabolario aperto
seo_title: 'API LibreOpenVocab: alias e argomenti'
description: >-
  La factory LibreOpenVocab, le sue quattro famiglie e tutti gli alias,
  set_classes, i valori di conf predefiniti per famiglia e le regole di
  text_threshold e iou.
lead: >-
  LibreOpenVocab è la factory dei rilevatori condizionati dal testo. L'elenco
  delle classi è un prompt invece di una testa fissa, quindi il vocabolario si
  imposta con set_classes e il modello restituisce normali Results di
  rilevamento rispetto a quell'elenco.
keywords:
  - LibreOpenVocab
  - open vocabulary detection python
  - Grounding DINO
  - OWLv2
  - OMDet-Turbo
  - OV-DEIM
  - rilevare oggetti da testo senza addestramento
last_verified: 1.5.0
verification: >-
  Alias letti da libreyolo/models/openvocab/__init__.py; repository, dimensioni
  e soglie da grounding_dino.py, owlv2.py, omdet_turbo.py e ov_deim.py; regole
  di chiamata da libreyolo/models/openvocab/base.py, tutto alla v1.5.0. Intento
  di progettazione da docs/adr/0008-open-vocab-detector-contract.md.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[openvocab]'
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-tiny")
        model.set_classes(["person", "skateboard", "handrail"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
source_hash: 64e4c641c6f8cde0
---

## Installazione

Il tier richiede l'extra `openvocab`.

<code-tabs name="install" />

## La factory

```python
LibreOpenVocab(model: str = "grounding-dino-tiny", **kwargs) -> LibreOpenVocabDetector
```

`model` è un alias, non un percorso. Gli underscore diventano trattini prima
della ricerca, quindi i nomi qualificati per famiglia che stampa l'inventario
della CLI, come `omdet_turbo-t` e `grounding_dino-t`, si caricano così come
sono. Un alias sconosciuto solleva `ValueError` elencando tutti gli alias noti.

Il costruttore accetta `size`, `nb_classes=80`, `names=None`,
`device="auto"`, `task=None` e `text_threshold=None`. Passare `names` equivale
a chiamare `set_classes` subito dopo il caricamento. Passare `text_threshold`
a una famiglia che non lo supporta solleva `TypeError`.

<code-tabs name="usage" />

## Famiglie e alias

| Famiglia | Alias | Dimensioni | Pesi |
|---|---|---|---|
| Grounding DINO | `grounding-dino`, `groundingdino`, `grounding-dino-tiny`, `groundingdino-tiny`, `grounding-dino-t`, `groundingdino-t`, `grounding-dino-base`, `groundingdino-base`, `grounding-dino-b`, `groundingdino-b` | `t`, `b` | `LibreYOLO/LibreGroundingDINOt`, `LibreYOLO/LibreGroundingDINOb` |
| OWLv2 | `owlv2`, `owl-v2`, `owlv2-base`, `owl-v2-base`, `owlv2-b16`, `owl-v2-b16`, `owlv2-large`, `owl-v2-large`, `owlv2-l14`, `owl-v2-l14` | `b16`, `l14` | `LibreYOLO/LibreOWLv2b16`, `LibreYOLO/LibreOWLv2l14` |
| OMDet-Turbo | `omdet-turbo`, `omdet`, `omdetturbo`, `omdet-turbo-tiny`, `omdet-turbo-swin-tiny`, `omdet-turbo-t` | `t` | `LibreYOLO/LibreOMDetTurbot` |
| OV-DEIM | `ov-deim`, `ovdeim`, `ov-deim-s`, `ovdeim-s`, `ov-deim-m`, `ovdeim-m`, `ov-deim-l`, `ovdeim-l` | `s`, `m`, `l` | `LibreYOLO/LibreOVDEIMs`, `LibreYOLO/LibreOVDEIMm`, `LibreYOLO/LibreOVDEIMl` |

L'alias predefinito è `grounding-dino-tiny`.

`LibreGroundingDINO`, `LibreOWLv2` e `LibreOMDetTurbo` sono esportati a livello
di pacchetto e si possono costruire direttamente con `size=`. A OV-DEIM si
arriva tramite gli alias della factory qui sopra.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreOpenVocabDetector
```

Imposta il vocabolario per ogni chiamata successiva a `predict()` e restituisce
il modello, così le chiamate si possono concatenare. L'elenco non può essere
vuoto, deve contenere solo stringhe e le sue voci devono essere uniche quando
le confronti senza distinguere maiuscole e minuscole; le etichette vuote
vengono rifiutate. Passare una stringa da sola solleva `TypeError`, perché
verrebbe enumerata in classi di un solo carattere.

Dopo la chiamata, `model.names` mappa `0..N-1` sulle etichette nell'ordine
dato, e `model.nb_classes` vale `N`.

## Argomenti di chiamata

Il tier riusa la superficie standard di predict con tre differenze.

`conf` usa come valore predefinito quello della singola famiglia invece dello
0.25 condiviso:

| Famiglia | conf predefinito | Soppressione |
|---|---|---|
| Grounding DINO | 0.25 | |
| OWLv2 | 0.1 | |
| OMDet-Turbo | 0.3 | Il suo post-processing, soglia 0.5, rispetta `iou=` |
| OV-DEIM | 0.25 | Corrispondenza uno a uno con selezione top-K, nessuna soppressione |

`iou=` significa qualcosa solo per una famiglia che esegue la soppressione.
OMDet-Turbo prende la soglia come argomento e la imposta a 0.5 quando `iou=`
non è indicato. Le altre tre non sopprimono nulla, quindi passare `iou=` lì
emette un avviso e viene ignorato.

`text_threshold=` riguarda solo Grounding DINO, dove vale 0.25 di default. Si
può passare alla costruzione per avere un valore persistente, oppure a ogni
chiamata. Un valore per chiamata non si può combinare con `stream=True`, perché
i risultati in streaming sono generati in modo pigro; in quel caso impostalo
sul costruttore. Ogni altra famiglia solleva `TypeError`.

`imgsz=` solleva `ValueError`: in questo tier è la pipeline di preprocessing a
occuparsi del ridimensionamento. Anche `augment=True` solleva un errore, dato
che la data augmentation in fase di test è fuori dallo scopo qui. Le dimensioni
di input sono registrate per famiglia solo a titolo di riferimento: Grounding
DINO 800, OWLv2 960 e 1008, OMDet-Turbo 640, OV-DEIM 640.

## Non supportato

`train()`, `val()`, `track()` e `export()` sollevano tutti
`NotImplementedError`. Fai il fine-tuning upstream e carica i pesi risultanti;
esegui `predict()` per ogni frame al posto del tracking. La validazione
richiederebbe un validatore dedicato, perché il validatore di rilevamento
condiviso chiama il modello con tensori immagine mentre questo tier richiede
input condizionati dal testo.
