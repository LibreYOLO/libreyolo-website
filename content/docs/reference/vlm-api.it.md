---
title: API di visione e linguaggio
seo_title: 'API LibreVLM: alias, set_classes e chat'
description: >-
  La factory LibreVLM, tutti gli alias dei modelli, il vocabolario persistente
  di set_classes, set_task, la via di fuga di chat e perché la confidenza è un
  segnaposto.
lead: >-
  LibreVLM carica un modello generativo di visione e linguaggio e lo guida come
  un rilevatore di oggetti. L'elenco delle classi è un prompt invece di una
  testa fissa, e il modello restituisce gli stessi Results che restituisce
  qualsiasi altra famiglia.
keywords:
  - LibreVLM
  - rilevamento oggetti con vision language model
  - Qwen3-VL
  - LFM2-VL
  - InternVL3
  - SmolVLM2
  - Florence-2
  - libreyolo chat
last_verified: 1.5.0
verification: >-
  Alias letti da libreyolo/models/vlm/__init__.py; repository, dimensioni ed
  elenchi di task dai moduli delle famiglie sotto libreyolo/models/vlm/ più
  libreyolo/models/sensenova/model.py; regole di chiamata ed eccezioni da
  libreyolo/models/vlm/base.py, tutto alla v1.5.0.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[vlm]'
  usage:
    - label: Rilevare un vocabolario aperto
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        model.set_classes(["person", "skateboard"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
    - label: Fare una domanda libera
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        print(model.chat(SAMPLE_IMAGE, "How many people are in this image?"))
source_hash: 57ddac08bc4d4e05
---

## Installazione

Il tier richiede l'extra `vlm`.

<code-tabs name="install" />

## La factory

```python
LibreVLM(model: str = "qwen3-vl-4b", **kwargs) -> LibreVLMModel
```

`model` è un alias, non un percorso. `**kwargs` arriva al costruttore della
famiglia, che accetta `device`, `names` (il vocabolario iniziale, equivalente a
chiamare `set_classes` dopo il caricamento), `prompt` (per sovrascrivere il
prompt di rilevamento) e `max_new_tokens`. Un alias sconosciuto solleva
`ValueError` elencando tutti gli alias.

<code-tabs name="usage" />

## Alias

| Famiglia | Alias | Dimensioni | Pesi |
|---|---|---|---|
| Qwen3-VL | `qwen3-vl`, `qwen3-vl-2b`, `qwen3-vl-4b`, `qwen3-vl-8b` | `2b`, `4b`, `8b` | `Qwen/Qwen3-VL-2B-Instruct`, `-4B-`, `-8B-` |
| LFM2-VL | `lfm2-vl`, `lfm2-vl-450m`, `lfm2-vl-1.6b` | `450m`, `1.6b` | `LiquidAI/LFM2.5-VL-450M`, `-1.6B` |
| InternVL3 | `internvl3`, `internvl3-1b`, `internvl3-2b`, `internvl3-8b` | `1b`, `2b`, `8b` | `OpenGVLab/InternVL3-1B-hf`, `-2B-hf`, `-8B-hf` |
| SmolVLM2 | `smolvlm2`, `smolvlm2-2.2b`, `smolvlm2-500m` | `2.2b`, `500m` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, `SmolVLM2-500M-Video-Instruct` |
| Florence-2 | `florence-2`, `florence2`, `florence-2-base`, `florence-2-large` | `base`, `large` | `florence-community/Florence-2-base`, `-large` |
| Kosmos-2 | `kosmos-2`, `kosmos2` | `224` | `microsoft/kosmos-2-patch14-224` |
| LocateAnything | `locate-anything`, `locateanything`, `locate-anything-3b`, `locateanything-3b` | `3b` | `nvidia/LocateAnything-3B` |
| SenseNova-Vision | `sensenova-vision`, `sensenova-vision-7b`, `sensenovavision` | `7b` | `LibreYOLO/SenseNovaVision7b` |
| LibreMODUS | `libremodus`, `libremodus-14b-a7b`, `modus`, `modus-14b-a7b` | `14b-a7b` | Snapshot upstream fissato |

L'alias predefinito è `qwen3-vl-4b`. Le dimensioni per l'alias predefinito di
ogni famiglia sono quelle elencate per prime: `qwen3-vl` risolve a `4b`,
`lfm2-vl` a `450m`, `internvl3` a `2b`, `smolvlm2` a `2.2b`, `florence-2` a
`base`.

`LibreVLM`, `LibreLFM2VL`, `LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`,
`LibreFlorence2`, `LibreKosmos2`, `LibreLocateAnything` e `LibreMODUS`
(scritto anche `LibreModus`) sono esportati a livello di pacchetto.

## Task

La maggior parte delle famiglie serve solo `detect`. Due ne servono di più:

| Famiglia | Task supportati |
|---|---|
| LocateAnything | `detect`, `point` |
| SenseNova-Vision | `detect`, `segment`, `panoptic`, `pose`, `point`, `depth`, `ocr` |

Poiché il task è guidato dal prompt invece di essere incorporato in un
checkpoint, si può cambiare su un modello già caricato:

```python
model.set_task(task: str) -> LibreVLMModel
```

Il task viene validato rispetto all'elenco supportato dalla famiglia, resta
valido nelle chiamate successive a `predict()` e `track()`, e il modello viene
restituito così che le chiamate si possano concatenare.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreVLMModel
```

Imposta il vocabolario aperto. Funziona qualsiasi parola, perché il modello la
riceve nel prompt invece di essere vincolato a una testa fissa. L'elenco non
deve essere vuoto e le sue voci devono essere uniche confrontandole senza
distinzione tra maiuscole e minuscole. Passare una semplice stringa solleva
`TypeError`, perché verrebbe enumerata in classi di un carattere. Il
vocabolario è persistente: impostalo una volta dopo il caricamento e rimane
finché non lo imposti di nuovo.

## chat

```python
model.chat(image, prompt, max_new_tokens=None, color_format="auto") -> str
```

Generazione multimodale grezza: immagine e prompt in ingresso, testo decodificato
in uscita, alla lettera. È la via di fuga sotto la comodità del rilevamento, per
domande libere, per contare, o per un formato di output che il wrapper di
rilevamento non copre. `max_new_tokens` ricade sul `MAX_NEW_TOKENS` della
famiglia, che è 1024 nella classe base. La decodifica è greedy con una lieve
penalità di ripetizione.

## Confidenza

L'output generato non ha una confidenza calibrata per box. Questa versione
assegna un segnaposto costante perché `predict`, il disegno e `track`
funzionino, il che rende il filtraggio con `conf=` e la mAP approssimativi
anziché significativi. È anche il motivo per cui `val()` solleva un'eccezione:
la mAP COCO su punteggi segnaposto sarebbe fuorviante.

## Predict e track

Vale la superficie standard di predict, e `track()` funziona, quindi un
rilevatore VLM si inserisce nella stessa pipeline di qualsiasi altra famiglia.
Due politiche a livello di classe differiscono da un rilevatore convoluzionale:
la test-time augmentation è disabilitata, perché l'augmentation multi-scala non
ha senso per un generatore a risoluzione fissa, e il predict in batch è
disattivato, perché la generazione è autoregressiva e il preprocessing
restituisce una codifica di testo e immagine invece di un tensore immagine
impilabile.

## Non supportato

`train()`, `val()` ed `export()` sollevano `NotImplementedError`. Fai
fine-tuning a monte e carica i pesi risultanti.

## Codice remoto

Ogni famiglia inclusa viene caricata tramite una classe di modello nativa,
quindi LibreYOLO non esegue codice di repository di terze parti per impostazione
predefinita. Una famiglia che ne ha davvero bisogno deve fare opt-in esplicito e
fissare una revisione snapshot; LocateAnything è quella che lo fa, fissata al
commit `c32291ca5e996f5a7a485845b4f57a233936bba0`.

LibreMODUS è un'eccezione esplicita allo schema dei checkpoint: il suo alias
risolve a una directory di file upstream fissati invece che a un `.pt` di
LibreYOLO, e LibreYOLO non gli aggiunge metadati v1.0 né lo ripubblica.
