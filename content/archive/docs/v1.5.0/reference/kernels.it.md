---
title: Kernels
seo_title: Registro dei kernel di LibreYOLO e kernel dell'Hub
description: >-
  Come LibreYOLO seleziona le implementazioni accelerate: il registro dei kernel
  sotto libreyolo/kernels, il kernel opzionale MS-deform-attn di Hugging Face
  Hub e l'interruttore dell'attenzione fusa.
lead: >-
  Ogni operazione accelerata di LibreYOLO ha un'implementazione portabile
  predefinita e, a volte, una variante più veloce registrata sopra di essa. La
  selezione avviene a runtime tramite un predicato, una dipendenza opzionale
  mancante è un fallback e non un errore, e un grafo esportato prende sempre il
  percorso portabile.
keywords:
  - libreyolo kernels
  - LIBREYOLO_KERNELS
  - LIBREYOLO_HUB_KERNELS
  - extra hub-kernels
  - kernel ms_deform_attn
  - set_fused_attention
  - kernel triton libreyolo cuda
last_verified: 1.5.0
verification: >-
  API del registro letta da libreyolo/kernels/__init__.py alla v1.5.0, API
  dell'attenzione da libreyolo/kernels/attention/__init__.py e sdpa.py, provider
  dell'Hub da libreyolo/kernels/attention/ms_deform_attn.py inclusa la sua
  revisione fissata e il suo predicato di idoneità. Struttura delle directory
  elencata da libreyolo/kernels/. Definizione dell'extra da pyproject.toml. Note
  di comportamento e cifre dei benchmark da docs/kernels.md. La storia del
  gating della v1.4.0 dal commit che ha collegato lo slot in RF-DETR e dalla
  voce del CHANGELOG della 1.5.0.
meta:
  - label: Pacchetto
    value: libreyolo.kernels
    mono: true
  - label: Extra opzionale
    value: 'libreyolo[hub-kernels]'
    mono: true
  - label: Forzare il riferimento
    value: LIBREYOLO_KERNELS=off
    mono: true
snippets:
  usage:
    - label: Vedere cosa è stato selezionato
      language: python
      code: |
        import libreyolo.kernels as kernels

        # Slot di op verso il nome dell'implementazione scelta, o "unavailable".
        print(kernels.active())
    - label: Forzare il percorso di riferimento
      language: bash
      code: |
        # off e reference significano la stessa cosa, e per di più evitano
        # del tutto l'import dei provider accelerati.
        LIBREYOLO_KERNELS=off python train.py
    - label: Disattivare i kernel dell'Hub senza disinstallare
      language: bash
      code: |
        LIBREYOLO_HUB_KERNELS=0 python predict.py
    - label: Passare una famiglia all'attenzione fusa
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.kernels.attention import set_fused_attention

        model = LibreYOLO("LibreSwinIRs.pt")

        # Restituisce quanti moduli di attenzione sono stati cambiati.
        print(set_fused_attention(model))
    - label: Registrare la tua implementazione
      language: python
      code: |
        import libreyolo.kernels as kernels

        kernels.register(
            "fake_quant_fp8",
            my_impl,
            name="mybackend",
            predicate=my_check,
        )
source_hash: 23d504e88b7959f8
---

## Il registro

`libreyolo/kernels/` è un piccolo registro a runtime di implementazioni
intercambiabili. Uno slot di op è un nome come `fake_quant_fp8` o
`ms_deform_attn`. Chi chiama chiede al registro uno slot e riceve la prima
implementazione registrata che supera il proprio predicato, dove vince la
registrazione più recente, ricadendo sull'implementazione di riferimento quando
non si applica nient'altro.

Quella struttura esiste perché una dipendenza opzionale non sia mai un requisito
obbligatorio. Una macchina senza Triton, senza CUDA o senza il pacchetto
`kernels` esegue lo stesso codice e produce gli stessi numeri, solo più
lentamente.

| Funzione | Scopo |
|---|---|
| `active()` | Slot di op verso il nome dell'implementazione scelta, o `"unavailable"` |
| `resolve(op)` | Il callable che verrebbe eseguito, o `None` |
| `register(op, impl, *, name, predicate=None)` | Aggiunge un'implementazione, la più recente per prima |
| `unregister(op, name)` | Ne rimuove una |
| `clear_cache()` | Scarta la risoluzione memoizzata |

<code-tabs name="usage" />

Un predicato che solleva un'eccezione viene catturato e segnalato con un
warning, mai propagato, così un'implementazione di terze parti difettosa degrada
al percorso portabile invece di rompere la predizione.

### Struttura

L'albero è organizzato prima per scopo e poi per backend, così uno slot si trova
in base a quello che calcola invece che in base alla libreria che oggi si trova a
implementarlo.

| Directory | Contenuto |
|---|---|
| `kernels/quant/simulate/` | Kernel Triton di fake-quantization, con backward straight-through, su qualsiasi dispositivo. Usati sia dal QAT sia dalla quantizzazione post-training simulata |
| `kernels/quant/execute/` | Percorsi a precisione reale solo per i modelli finalizzati, senza backward: il GEMM FP8 sui tensor core, il suo prologo ed epilogo Triton fusi, e i kernel di unpack dei pesi impacchettati |
| `kernels/attention/` | Op di attenzione condivise fra le famiglie: lo slot `ms_deform_attn` e la policy della SDPA fusa |

Il confine fra `simulate` ed `execute` è se il modello è finalizzato, non se sta
addestrando o andando in produzione. Le implementazioni di riferimento restano in
`libreyolo/quant/`, che definisce cosa significano i numeri; `kernels/` le rende
soltanto veloci. L'impacchettamento dei pesi non ha varianti di alcun tipo,
perché è il contratto del checkpoint.

Gli slot GEMM e di attenzione non hanno un'implementazione di riferimento. Chi
chiama deve verificare che `resolve()` abbia restituito qualcosa e mantenere il
proprio percorso portabile, ed è per questo che i grafi ONNX, TensorRT e
`torch.export` contengono sempre la matematica portabile.

### Override della selezione

`LIBREYOLO_KERNELS=off` o `=reference` forza le implementazioni di riferimento e
cortocircuita del tutto l'import dei provider accelerati. Qualsiasi altro valore
restringe la selezione alle implementazioni registrate sotto quel nome.
`LIBREYOLO_QUANT_KERNELS` è onorata come alias legacy dai tempi in cui il
registro stava sotto `libreyolo/quant/`, e viene letta solo quando
`LIBREYOLO_KERNELS` non è impostata. Entrambe sono elencate insieme alle altre su
[impostazioni](/docs/reference/settings).

## Kernel dell'Hub

I kernel CUDA compilati pubblicati sull'Hugging Face Hub si caricano a runtime
attraverso il pacchetto opzionale `kernels`. Niente viene incorporato dentro
LibreYOLO; l'artefatto è scaricato e messo in cache da quel pacchetto, e ogni
provider fissa una revisione di commit verificata, quindi alzare un pin richiede
un test di parità su GPU prima di entrare.

L'installazione dell'extra è l'opt-in:

```bash
pip install "libreyolo[hub-kernels]"
```

Senza il pacchetto non cambia niente e non viene fatta nessuna richiesta di rete.
`LIBREYOLO_HUB_KERNELS=0` disabilita il download senza disinstallare niente. Un
kernel che non riesce a caricarsi o a girare si disabilita da solo per il resto
del processo e ricade sul percorso portabile con un solo warning.

Oggi uno slot è servito dall'Hub: `ms_deform_attn`, il forward e il backward
compilati dell'attenzione deformabile multi-scala di Deformable DETR, sotto
licenza Apache 2.0. È collegato a tutta la linea deformable: RF-DETR, Deformable
DETR, DINO-DETR, LW-DETR, Grounding DINO, RT-DETR, RT-DETRv2, D-FINE, RT-DETRv4,
DEIM, DEIMv2, EC e OV-DEIM. Dato che anche il backward è compilato, ne beneficia
l'addestramento oltre alla predizione.

L'idoneità è ristretta di proposito. Gli input devono essere CUDA e float32, e
l'esecuzione deve essere eager: il provider si tira indietro sotto
`torch.jit.is_tracing()`, `torch.compiler.is_compiling()`,
`torch.compiler.is_exporting()` e `torch.onnx.is_in_onnx_export()`. Anche due
layout di input ricadono sul percorso portabile, un conteggio di punti per
livello che varia da livello a livello, e il campionamento a indici interi
discreti. La variante di posa di EC non è collegata.

### Questo kernel è raggiungibile solo da poco

Leggi questo prima di installare l'extra su un progetto già esistente.

Nella v1.4.0 lo slot veniva consultato dall'interno di un helper, dietro una
condizione che richiedeva l'assenza delle coppie di forme spaziali. RF-DETR fa
sempre passare quelle coppie attraverso il suo decoder, quindi la condizione non
si è mai verificata e il kernel non è mai stato eseguito in nessun forward eager.
Nella v1.5.0 la consultazione è stata spostata, e adesso il kernel gira davvero.

La conseguenza pratica è che aggiornare alla v1.5.0 *e* installare
`libreyolo[hub-kernels]` su CUDA significa che RF-DETR e la sua linea prendono il
forward da un binario compilato per la prima volta. Di conseguenza le predizioni
e le metriche possono spostarsi entro la tolleranza float. Un'installazione
standard, senza l'extra, non è toccata. Se stai confrontando le metriche prima e
dopo l'aggiornamento, tieni fisso l'extra o imposta `LIBREYOLO_HUB_KERNELS=0` da
entrambe le parti.

## Attenzione fusa

L'attenzione scaled dot-product fusa non richiede nessuna dipendenza opzionale,
solo PyTorch standard, quindi è governata da una policy invece che dalla
disponibilità. Valgono due regole.

Primo, una cattura del grafo non la usa mai. Ogni punto di chiamata sostituito
tiene disponibile l'equazione con op primitive dietro un controllo di
esportazione, che copre l'esportazione ONNX, il cui opset predefinito non ha un
simbolico per SDPA, e `torch.jit.trace`, da cui passano TorchScript, CoreML e
NCNN. Le catture di Dynamo sono deliberatamente fuori dal gate, perché
`torch.compile` fa il lowering di SDPA meglio della matematica manuale, e sia
Core AI sia ExecuTorch scompongono SDPA in core ATen per conto loro.

Secondo, l'asticella di parità per renderla predefinita è l'esattezza al byte. Le
famiglie che la superano usano SDPA di default: SegFormer, Depth Anything e
MoGe-2, BERT, Grounding DINO, SwinIR e PP-OCR. Le famiglie che non ci arrivano
tengono la matematica manuale ed espongono invece un flag `fused_attn`, che è
quello che `set_fused_attention(model)` ribalta: Swin, il backbone Swin di
DINO-DETR, BiRefNet e FeyNobg, OWLv2, LW-DETR, SigLIP 2, ZipDepth e MobileSAM.
ViT e DeiT hanno lo stesso flag ma lo tengono attivo di default, seguendo
l'upstream, quindi la stessa chiamata con `enabled=False` li disattiva.

Dove si applica, conviene. Su una RTX 5070 Ti in autocast fp16, l'attenzione a
finestre di Swin passa da 1.278 ms a 0.721 ms, un guadagno di 1.77x, e
l'attenzione visiva di OWLv2 da 6.483 ms a 1.735 ms, 3.74x.

## Hardware

| Piattaforma | Comportamento |
|---|---|
| CPU e MPS | Ogni predicato CUDA e Triton fallisce, quindi tutto gira sull'implementazione di riferimento |
| NVIDIA CUDA | Entrano in gioco i kernel Triton e i kernel dell'Hub e GEMM idonei |
| AMD ROCm | Triton può entrare in gioco, dato che le wheel ROCm includono il backend AMD di Triton, ma la parità è verificata in CI solo su NVIDIA |

## Aggiungere un'implementazione

Chiama `register()` con un nome e un predicato. I kernel compilati fuori
dall'albero possono essere distribuiti come pacchetto `libreyolo_kernels`
separato che si registra da solo all'import, e questo tiene un backend privato
completamente fuori dall'albero di LibreYOLO.

La parità è il criterio di ammissione per qualsiasi cosa dentro l'albero: una
corrispondenza esatta del forward rispetto al riferimento, e gradienti entro
1e-6 dallo straight-through estimator, sull'insieme di forme che la suite di test
contiene.

La selezione dei kernel interagisce con i [grafi CUDA](/docs/reference/cuda-graphs):
la matrice di parità dell'inferenza è stata eseguita senza il pacchetto `kernels`
installato, quindi la sicurezza della cattura con un kernel compilato attivo non
è coperta da essa.
