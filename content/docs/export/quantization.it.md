---
title: Quantizzazione
seo_title: Quantizzare un modello LibreYOLO in PyTorch
description: >-
  L'API di quantizzazione PyTorch di LibreYOLO: nove ricette, dati di
  calibrazione tenuti separati da quelli di addestramento, QAT e QAD, e due
  artefatti di deployment.
lead: >-
  La quantizzazione in LibreYOLO gira interamente in PyTorch: model.quantize()
  sostituisce i moduli Conv2d e Linear di un modello con equivalenti quantizzati
  e li calibra. Il risultato mantiene il normale contratto predict, val, train e
  save, quindi un modello quantizzato viene valutato dagli stessi validatori di
  uno float.
keywords:
  - quantizzazione libreyolo
  - quantizzare yolo int8
  - quantization aware training
  - qat qad
  - nvfp4 mxfp4
  - fp8 e4m3
  - dataset di calibrazione
  - export onnx qdq int8
last_verified: 1.5.0
meta:
  - label: Chiamata
    value: 'model.quantize(recipe="int8", calib="coco128.yaml")'
    mono: true
  - label: Comando
    value: libreyolo quantize --model M.pt --recipe int8 --calib coco128.yaml
    mono: true
  - label: Extra
    value: Nessuno. La quantizzazione gira in PyTorch.
  - label: Famiglie
    value: 'yolo9, rfdetr, birefnet, feynobg'
  - label: Ricette
    value: 'fp16, bf16, fp8, int8, w4a16, w4a8, nvfp4, mxfp4, int2'
    mono: true
  - label: Artefatti di deployment
    value: >-
      export(format="pt") per un checkpoint impacchettato, export(format="onnx")
      per un grafo QDQ INT8
    mono: true
verification: >-
  Letto da libreyolo/quant/api.py, libreyolo/models/base/model.py,
  libreyolo/cli/commands/quantize.py e docs/quantization.md sul branch dev. Le
  dimensioni dei checkpoint sono i valori misurati registrati in
  docs/quantization.md.
snippets:
  quantize:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Sostituzione dei moduli più calibrazione. calib è un piccolo insieme
        di

        # immagini SENZA ETICHETTE, letto solo in avanti per ricavare intervalli
        e scale.

        qmodel = model.quantize(recipe="int8", calib="coco128.yaml",
        samples=128)


        print(qmodel.quant_info())

        qmodel.val(data="coco8.yaml")          # gli stessi validatori di un
        modello float

        qmodel.save("LibreYOLO9s-int8.pt")     # il checkpoint porta un manifest
        quant
    - label: CLI
      language: bash
      code: >
        libreyolo quantize --model LibreYOLO9s.pt --recipe int8 --calib
        coco128.yaml
    - label: Argomenti
      language: python
      code: |
        model.quantize(
            recipe="int8",
            calib="coco128.yaml",      # percorso di un data.yaml o nome integrato; None salta la calibrazione
            samples=128,               # numero massimo di immagini di calibrazione
            batch=8,                   # dimensione del batch di calibrazione
            algorithm="auto",          # auto e minmax sono la stessa cosa; percentile è l'alternativa
            keep_high_precision=None,  # None usa la policy della famiglia
            verbose=True,
        )
  reload:
    - label: Un checkpoint quantizzato si ricarica come tale
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Il manifest quant ricostruisce la struttura quantizzata e le scale
        # prima che i pesi vengano caricati.
        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
        print(qmodel.quant_info())
  train:
    - label: Il QAT è un semplice train() su un modello quantizzato
      language: python
      code: >
        from libreyolo import LibreYOLO


        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")


        # Un finetune, non un addestramento da zero: usa learning rate da
        finetune.

        qmodel.train(data="coco8.yaml", epochs=5, lr0=1e-4)
    - label: Il QAD aggiunge gli argomenti di distillazione già esistenti
      language: python
      code: |
        qmodel.train(
            data="coco8.yaml",
            epochs=5,
            lr0=1e-4,
            distill_model="LibreYOLO9m.pt",
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train --model LibreYOLO9s-int8.pt --data coco8.yaml --epochs 5
        --lr0 1e-4
  export:
    - label: Checkpoint PyTorch impacchettato
      language: python
      code: >
        from libreyolo import LibreYOLO


        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")


        # Scrive LibreYOLO9s-int8-final.pt: pesi e scale impacchettati a pochi
        bit,

        # master fp32 rimossi, il resto non quantizzato convertito in fp16.

        qmodel.export(format="pt")


        # remainder="fp32" mantiene esatti i tensori non quantizzati.

        qmodel.export(format="pt", remainder="fp32")
    - label: QDQ INT8 ONNX
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # Coppie QuantizeLinear/DequantizeLinear nel grafo, che portano le scale
        # calibrate o addestrate con QAT del modello stesso.
        qmodel.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9s-int8.pt --format onnx
  dequantize:
    - label: 'Tornare a float, mantenendo i pesi addestrati con QAT'
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
        qmodel.dequantize()

        # Ora vale qualsiasi esportatore float, a qualunque precisione supporti.
        qmodel.export(format="tensorrt", half=True)
source_hash: 4ffb06b87cad017e
---

## Installazione

La quantizzazione non ha bisogno di alcun extra. La sostituzione dei moduli, il
passaggio di calibrazione e l'aritmetica simulata girano tutti in PyTorch, quindi
`pip install libreyolo` è tutto il necessario. Gli artefatti di deployment hanno
bisogno di quello che serve al loro formato, che per la via ONNX è
`libreyolo[onnx]`.

## Quantizzare

<code-tabs name="quantize" />

`quantize()` trasforma sul posto il modello caricato e lo restituisce. Non ci sono
gradienti di mezzo: la sostituzione installa moduli quantizzati e il passaggio di
calibrazione gira solo in avanti.

Il checkpoint risultante è un normale checkpoint LibreYOLO con allegato un manifest
`quant`, quindi si ricarica con struttura e scale intatte:

<code-tabs name="reload" />

Anche i checkpoint scritti dal trainer durante una sessione di QAT portano con sé il
manifest, il che significa che il `best.pt` di una sessione simile è a sua volta un
checkpoint quantizzato.

## Ricette

Sono supportate quattro famiglie: `yolo9`, `rfdetr`, `birefnet` e `feynobg`.

| Ricetta | Cosa fa | Famiglie | Calibrazione |
|---|---|---|---|
| `fp16` | Cast a precisione half con un contratto di input e output in float32. Solo inferenza. | tutte e quattro | nessuna |
| `bf16` | Cast a bfloat16, che mantiene l'intervallo di esponente del float32. La soluzione quando fp16 va in overflow su un modello in stile DETR. Solo inferenza. | tutte e quattro | nessuna |
| `fp8` | Pesi e attivazioni E4M3 su `Conv2d` e `Linear`: scale dei pesi per canale, scale delle attivazioni per tensore calibrate. | tutte e quattro | richiesta |
| `int8` | W8A8 su `Conv2d` e `Linear`: pesi simmetrici per canale, attivazioni affini per tensore. | tutte e quattro | richiesta, oppure `calib=None` per i soli pesi |
| `w4a16` | Pesi INT4 simmetrici a gruppi, gruppo 128 lungo `in_features`, attivazioni float, su `Linear`. | rfdetr, birefnet, feynobg | non serve |
| `w4a8` | Pesi INT4 a gruppi più attivazioni INT8 calibrate, su `Linear`. | rfdetr, birefnet, feynobg | richiesta |
| `nvfp4` | NVFP4 W4A4 su `Linear`: elementi E2M1, blocchi da 16 elementi, scale di blocco FP8 E4M3, scala del tensore FP32. Scalatura dinamica delle attivazioni. | rfdetr, birefnet, feynobg | non serve |
| `mxfp4` | MXFP4 OCP su `Linear`: elementi E2M1, blocchi da 32 elementi, scale di blocco E8M0 potenze di due. Scalatura dinamica delle attivazioni. | rfdetr, birefnet, feynobg | non serve |
| `int2` | Solo per la ricerca: pesi a 2 bit a gruppi, gruppo 64, più attivazioni INT8, su `Linear`. Da sola la post-training è inutilizzabile, quindi servono QAT o QAD. | rfdetr | richiesta |

Le ricette sotto gli 8 bit puntano a `nn.Linear` e vengono rifiutate per `yolo9` di
proposito: sull'hardware attuale quell'accelerazione riguarda solo le GEMM, quindi
le convoluzioni restano a precisione più alta. YOLO9 usa `int8` o `fp8`. `int2`
viene rifiutata per `birefnet` e `feynobg` perché quelle famiglie sono solo di
inferenza, quindi il QAT di recupero da cui la ricetta dipende lì non è disponibile.

I valori predefiniti di ogni famiglia tengono in float il primo layer e le teste, e
la convoluzione DFL di YOLO9 non viene mai quantizzata: è un operatore di
aspettazione integrale fisso. Sovrascrivi con `keep_high_precision=("head.",)`
quando hai un motivo per farlo.

## I dati di calibrazione non sono dati di addestramento

`calib=` prende qualche centinaio di immagini, non legge etichette e gira solo in
avanti per stimare gli intervalli delle attivazioni. `data=` in `train()` e `val()`
è il dataset etichettato usato per gradienti e metriche. Sono argomenti diversi con
scopi diversi, e il valore predefinito di `calib` è `coco128.yaml`.

`algorithm="minmax"` mantiene gli estremi assoluti visti nei batch di calibrazione
ed è ciò che seleziona `"auto"`. `"percentile"` usa la media dei percentili 0.1 e
99.9 di ogni batch; è stato misurato che fa crollare l'accuratezza della famiglia
DETR, perché nei transformer gli outlier delle attivazioni sono portanti. Quello che
davvero risolve la sensibilità all'INT8 dei modelli piccoli è calibrare su
abbastanza batch: con il `coco128` predefinito, YOLO9-t arriva a circa un punto di
mAP dal suo punteggio float. L'algoritmo scelto viene registrato nel manifest del
checkpoint.

## Recuperare l'accuratezza

<code-tabs name="train" />

I moduli quantizzati mantengono pesi master in fp32 e applicano la fake quantization
con uno straight-through estimator, quindi i gradienti arrivano ai master e i trainer
esistenti funzionano senza modifiche: EMA, AMP, ripresa da checkpoint e gli argomenti
di distillazione si compongono tutti.

Il QAT è un finetune di un modello già addestrato. Usa learning rate da finetune
invece dei valori predefiniti per l'addestramento da zero, altrimenti una sessione
breve distruggerà i pesi preaddestrati a prescindere dalla quantizzazione. La
disponibilità del QAD segue il supporto alla distillazione di ogni famiglia, che oggi
significa `yolo9` e `rfdetr`.

I modelli quantizzati con `fp16` e `bf16` sono solo di inferenza, e il trainer li
rifiuta rimandando ad `amp=True`.

## Esportazione

<code-tabs name="export" />

`format="pt"` cristallizza il modello. Pesi e scale impacchettati a pochi bit
sostituiscono i master, e il resto non quantizzato viene convertito in fp16 a meno
che non si passi `remainder="fp32"`. L'invariante dell'impacchettamento è che lo
spacchettamento riproduce la simulazione bit per bit sul dispositivo su cui hai
finalizzato, quindi il file finalizzato ottiene esattamente il punteggio che hai
validato. Misurato: YOLO9-s int8 passa da 29.5 MB a 9.6 MB, RF-DETR-n nvfp4 da 122
MB a 26 MB. Caricarne uno dà un modello pronto per l'inferenza, e chiamarci sopra
`train()` ricostruisce automaticamente i master dai pesi impacchettati.

`format="onnx"` vale per i modelli `int8` ed emette un grafo QDQ che porta le scale
calibrate o addestrate con QAT del modello stesso, che ONNX Runtime e TensorRT
eseguono con veri kernel INT8. È una via diversa da
[`export(format="onnx", int8=True)`](/docs/export/onnx) su un modello float, dove è
ONNX Runtime a ricavare le scale da sé.

Le ricette di cast non hanno bisogno di alcun esportatore quantizzato:

<code-tabs name="dequantize" />

## Vincoli

L'aritmetica quantizzata viene eseguita in simulazione, cioè fake quantization
calcolata in isole di float32 anche sotto AMP. La simulazione è fedele nei numeri,
quindi un punteggio di `val()` su qualsiasi dispositivo è un'affermazione reale
sull'aritmetica quantizzata. Non è un'affermazione sulla velocità.

Due eccezioni vengono eseguite in modo nativo. `fp16` e `bf16` sono normali cast. I
moduli `fp8` finalizzati eseguono la loro GEMM direttamente su pesi E4M3
impacchettati tramite `torch._scaled_mm` su hardware di classe Ada, Hopper e
Blackwell, usando le stesse scale delle attivazioni calibrate della simulazione;
impostare `LIBREYOLO_KERNELS=off` ripristina ovunque l'esatta via simulata.

La copertura per il deployment è più stretta dell'elenco delle ricette. Qui solo
`int8` ha una forma ONNX distribuibile; `fp8` e le ricette lineari sotto gli 8 bit
vengono eseguite in PyTorch e si cristallizzano tramite `format="pt"`. Chiedere loro
un'esportazione ONNX solleva un errore con quella indicazione, così come chiedere un
formato diverso da ONNX a un modello `int8`: costruisci invece i motori a valle dal
grafo QDQ.

Esportare un modello `int8` le cui attivazioni non sono mai state calibrate registra
un avviso e produce un grafo che porta solo la quantizzazione dei pesi.
