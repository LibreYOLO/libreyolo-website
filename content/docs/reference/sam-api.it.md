---
title: API di segmentazione promptable
seo_title: 'API LibreSAM: prompt, alias e firme'
description: >-
  La factory LibreSAM, i suoi alias di dimensione, i tipi di prompt a punti, a
  box e a testo-concetto, il ciclo di vita set_image a codifica singola e ciò
  che il tier non supporta.
lead: >-
  LibreSAM è la factory per la segmentazione promptable. Un forward pass ha
  bisogno di un prompt per immagine fornito al momento della chiamata, quindi il
  tier possiede la propria superficie predict invece di passare per il runner di
  inferenza senza prompt.
keywords:
  - LibreSAM
  - promptable segmentation
  - prompt a punti SAM
  - prompt box SAM
  - set_image
  - segment everything
  - libreyolo sam extra
last_verified: 1.5.0
verification: >-
  Alias, dimensioni e repository della factory letti da
  libreyolo/models/sam/model.py, sam2.py, edgetam.py, sam3.py,
  libreyolo/models/mobilesam/model.py e libreyolo/models/picosam3/model.py.
  Contratto dei prompt e valori predefiniti letti da
  libreyolo/models/sam/base.py. Intento di progettazione da
  docs/adr/0007-libresam-contract.md, tutto alla v1.5.0.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[sam]'
  usage:
    - label: Prompt a punti e box
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        r = model.predict(SAMPLE_IMAGE, points=[900, 370], labels=[1])
        print(r.masks.xy)
        print(r.boxes.xyxy)

        r = model.predict(SAMPLE_IMAGE, bboxes=[100, 100, 200, 200])
        print(len(r))
    - label: 'Codifica una volta, prompt molte volte'
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")
        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[500, 375], labels=[1])
        b = model.predict(bboxes=[100, 100, 200, 200])
        print(len(a), len(b))

        model.reset_image()
source_hash: 18e8206c10ce17fd
---

## Installazione

Il tier richiede l'extra `sam`.

<code-tabs name="install" />

## La factory

```python
LibreSAM(model: str = "base", **kwargs) -> LibreSAMModel
```

`model` è un alias di dimensione, non un percorso. `**kwargs` arriva al
costruttore della famiglia, che accetta `device` e `multimask`. Un alias
sconosciuto solleva `ValueError` e il messaggio elenca tutti gli alias noti.

<code-tabs name="usage" />

## Alias

| Famiglia | Alias | Dimensioni | Pesi |
|---|---|---|---|
| SAM-1 | `base`, `large`, `huge`, `b`, `l`, `h`, `sam-base`, `sam-large`, `sam-huge`, `sam_b`, `sam_l`, `sam_h` | `base`, `large`, `huge` | `facebook/sam-vit-base`, `-large`, `-huge` |
| SAM-2 | `sam2-tiny`, `sam2-small`, `sam2-base-plus`, `sam2-baseplus`, `sam2-large`, e le forme brevi `sam2-t`, `sam2-s`, `sam2-bp`, `sam2-l`, `sam2_t`, `sam2_s`, `sam2_bp`, `sam2_l` | `tiny`, `small`, `base-plus`, `large` | `LibreYOLO/LibreSAM2tiny`, `-small`, `-base-plus`, `-large` |
| EdgeTAM | `edgetam`, `edge-tam`, `edgetam-edge` | `edge` | `LibreYOLO/LibreEdgeTAM` |
| SAM 3 | `sam3`, `sam-3`, `sam3-large` | `large` | `facebook/sam3` |
| MobileSAM | `mobilesam`, `mobilesam-tiny`, `mobilesam_t`, `mobile-sam`, `mobile-sam-tiny` | `tiny` | `LibreYOLO/LibreMobileSAM` |
| PicoSAM3 | `picosam3`, `picosam3-pico`, `picosam3_pico`, `pico-sam3` | `pico` | `LibreYOLO/LibrePicoSAM3` |

Il valore predefinito è `base`. SAM-1, SAM-2, EdgeTAM e MobileSAM lavorano su
un canvas nominale di 1024 pixel, SAM 3 su 1008, PicoSAM3 su 96.

I pesi di SAM 3 sono ad accesso condizionato. Si scaricano da `facebook/sam3`
sotto la SAM License personalizzata di Meta, che non è né MIT né Apache-2.0 e
non viene ridistribuita da LibreYOLO. Accetta i termini sulla pagina del
repository e autenticati su Hugging Face prima di caricare; il loader registra
prima di tutto l'avviso.

Anche le classi delle famiglie sono esportate, quindi `LibreSAM1`,
`LibreSAM2`, `LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM` e `LibrePicoSAM3`
si possono costruire direttamente con `size=`.

## predict

```python
model.predict(
    source=None,
    *,
    points=None,
    bboxes=None,
    labels=None,
    masks=None,
    text=None,
    conf=None,
    multimask=None,
    max_det=300,
    device=None,
    color_format="auto",
    points_per_side=None,
) -> Results
```

| Argomento | Default | Significato |
|---|---|---|
| `source` | `None` | Immagine da segmentare; `None` riusa l'immagine messa in cache da `set_image()` |
| `points` | `None` | Prompt a punti in coordinate pixel |
| `bboxes` | `None` | Prompt a box come `[x1, y1, x2, y2]`, oppure una lista di box per ottenere una maschera per box |
| `labels` | `None` | Etichette dei punti, `1` positiva e `0` negativa, di forma coerente con `points`; tutte positive se omesse |
| `masks` | `None` | Riservato; passarne una solleva `NotImplementedError` |
| `text` | `None` | Prompt di concetto; solo SAM 3 |
| `conf` | `None` | Soglia minima sull'IoU predetto della maschera |
| `multimask` | `None` | Restituisce tutte le maschere di ambiguità per ogni prompt; per default usa l'impostazione data alla costruzione |
| `max_det` | `300` | Limite alle maschere restituite |
| `device` | `None` | Sposta il modello per questa chiamata e per quelle successive, invalidando gli embedding in cache |
| `color_format` | `"auto"` | Suggerimento sul formato colore per gli array in memoria |
| `points_per_side` | `None` | Densità della griglia per segment-everything; per default 32 |

Il valore restituito è un normale `Results` che porta `masks`, più i `boxes`
aderenti derivati da quelle maschere, con la classe `0` chiamata `"object"`.

## Forme dei prompt

`points` accetta le forme annidate `[x, y]` per un oggetto, `[[x, y], ...]`
per N oggetti e `[[[x, y], ...], ...]` per punti raggruppati per oggetto. Gli
array Numpy funzionano ovunque funzioni una lista. Le coordinate sono semplici
pixel sull'immagine sorgente.

Omettere ogni prompt spaziale esegue segment-everything, un generatore
automatico di maschere a griglia con una soglia sull'IoU predetto e una
deduplicazione per IoU dei box. Il valore predefinito di `points_per_side`,
32, esegue circa 1024 passaggi del decoder, il che è lento su CPU; abbassalo
per l'uso interattivo. Il generatore non fa il filtraggio per stability score,
il multi-crop e la deduplicazione per IoU delle maschere, quindi è
un'approssimazione del percorso con prompt più che il suo equivalente.

## Confidenza

`conf` filtra in base all'IoU predetto della maschera, che è un punteggio di
qualità della maschera e non una confidenza di rilevamento. `None` mantiene
tutte le maschere nel percorso con prompt e applica la soglia di griglia della
famiglia in segment-everything. `0.0` disattiva il filtraggio in entrambe le
modalità.

Nel percorso testuale di SAM 3, `conf` è invece il punteggio di rilevamento
della Promptable Concept Segmentation. Lì `None` significa la soglia standard
di 0.3, e `0.0` mantiene tutti i candidati.

## Prompt testuali

`text=` vale solo per SAM 3; ogni famiglia a prompt spaziali solleva
`NotImplementedError` se lo riceve. Il testo è mutuamente esclusivo con punti
e box. Il `names` restituito associa la classe `0` al concetto richiesto. Una
chiamata testuale con `source=None` ricodifica l'immagine in cache, perché il
tracker e l'encoder dei concetti non condividono la cache.

Il parametro `exemplars=` è riservato a una futura estensione con esemplari
immagine e non è implementato.

## Il ciclo di vita a codifica singola

```python
model.set_image(source, color_format="auto") -> LibreSAMModel
model.reset_image() -> LibreSAMModel
```

`set_image` esegue una sola volta il pesante encoder di immagini e mette in
cache gli embedding, così ogni `predict()` successivo con `source=None` costa
poco. Entrambi i metodi restituiscono il modello, così le chiamate si possono
concatenare. Passare `device=` a `predict` sposta il modello e invalida la
cache.

## PicoSAM3

PicoSAM3 accetta solo `bboxes=`. I prompt a punti, a testo, a maschera, il
multimask e segment-everything sollevano un'eccezione. Il box viene espanso
del 10 percento e passato attraverso una rete ROI a 96 pixel, e PicoSAM3 è
l'unica famiglia del tier che esporta, e solo in ONNX.

## Non supportato

`train()`, `val()` e `track()` sollevano `NotImplementedError` su ogni
famiglia del tier. Le maschere promptable non hanno un insieme fisso di classi
su cui essere valutate, quindi qui la mAP non ha significato. `export()`
solleva un'eccezione per SAM-1, SAM-2, SAM 3, EdgeTAM e MobileSAM.

I percorsi video e di memoria per SAM-2, SAM 3 ed EdgeTAM sono fuori dallo
scopo di questa versione, così come gli esemplari immagine e i prompt a
maschera di SAM 3.
