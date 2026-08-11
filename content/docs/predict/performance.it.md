---
title: Prestazioni dell'inferenza
seo_title: Inferenza più veloce in LibreYOLO
description: >-
  CUDA graph, mezza precisione, batching, inferenza a tasselli e test-time
  augmentation al momento della predizione, con i valori predefiniti reali e le
  famiglie che supportano ciascuna opzione.
lead: >-
  Cinque controlli al momento della predizione cambiano il throughput o
  l'accuratezza: replay dei CUDA graph, precisione, batching, tiling e test-time
  augmentation. Ognuno si applica a un insieme specifico di famiglie, e due di
  essi costano accuratezza o latenza invece di farti risparmiare.
keywords:
  - cuda graphs pytorch inferenza
  - inferenza batch yolo python
  - inferenza fp16
  - inferenza a tasselli oggetti piccoli
  - sliced inference immagini grandi
  - test time augmentation object detection
  - capture_graph
  - predict batch cartella python
last_verified: 1.5.0
verification: >-
  Valori predefiniti degli argomenti da InferenceRunner.__call__ in
  libreyolo/models/base/inference.py. API dei CUDA graph da
  BaseModel.capture_graph, graph_info, release_graphs e cuda_graph_scope in
  libreyolo/models/base/model.py; adesione per famiglia dalla variabile di
  classe SUPPORTS_CUDA_GRAPH. Comportamento della mezza precisione da
  NOOP_PREDICT_KWARGS in libreyolo/utils/predict_args.py, dall'avviso della CLI
  in libreyolo/cli/commands/predict.py e da CAST_RECIPES più SUPPORTED_FAMILIES
  in libreyolo/quant/api.py. Condizioni del batching da
  InferenceRunner._process_in_batches e _predict_batch. Tiling da _predict_tiled
  e _merge_tile_detections. Test-time augmentation da BaseModel._predict_augment
  e _merge_tta, con TTA_ENABLED, TTA_SCALES e TTA_FIXED_SIZE letti in tutto
  libreyolo/models/.
snippets:
  batch:
    - label: Inferenza in batch su una cartella
      language: python
      code: >
        from pathlib import Path

        from PIL import Image


        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        folder = Path("batch_demo")

        folder.mkdir(exist_ok=True)

        image = Image.open(SAMPLE_IMAGE)

        for index in range(8):
            image.save(folder / f"frame_{index}.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")


        # Un solo forward impilato per blocco di 4 sulle famiglie che lo
        supportano.

        results = model(str(folder), batch=4)

        print(len(results), "results")
    - label: 'Streaming, così la lista non si materializza mai'
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("batch_demo", batch=4, stream=True):
            print(len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt source=batch_demo batch=4
  graphs:
    - label: 'Cattura in anticipo, poi replay (richiede CUDA)'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # Paga warmup e cattura una volta sola, fuori dalla prima richiesta.
        model.capture_graph()

        result = model(SAMPLE_IMAGE, cuda_graph=True)
        print(len(result.boxes))
        print(model.graph_info())
    - label: Cattura solo quando una shape si ripete (richiede CUDA)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # "auto" aspetta di vedere una shape due volte, così il lavoro one-shot
        # non paga mai la cattura.
        for _ in range(3):
            model(SAMPLE_IMAGE, cuda_graph="auto")

        print(model.graph_info())
        model.release_graphs()
  precision:
    - label: Installa l'extra per l'esportazione
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 'Esporta e ricarica, alla precisione predefinita'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: Esportazione FP16 (compilala ed eseguila su una macchina CUDA)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")
        path = model.export(format="onnx", half=True)

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: 'FP16 in PyTorch, tramite una ricetta di cast (richiede CUDA)'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # Una ricetta di cast non legge dati di calibrazione.
        model.quantize(recipe="fp16", calib=None)

        result = model(SAMPLE_IMAGE)
        print(len(result.boxes))
  tiling:
    - label: Inferenza a tasselli su un'immagine grande
      language: python
      code: >
        from PIL import Image


        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Il tiling si attiva solo quando l'immagine è più grande della
        dimensione di input.

        large = Image.open(SAMPLE_IMAGE).resize((2048, 1536))

        large.save("large.jpg")


        model = LibreYOLO("LibreYOLO9s.pt")


        result = model("large.jpg", tiling=True, overlap_ratio=0.2)

        print(result.num_tiles, "tiles", len(result.boxes), "detections")
  tta:
    - label: Test-time augmentation
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        plain = model(SAMPLE_IMAGE)
        flipped = model(SAMPLE_IMAGE, augment=True)

        print(len(plain.boxes), "->", len(flipped.boxes))
source_hash: 3914665d0e7f892c
---

## I controlli e i loro valori predefiniti

Ognuno di questi è un argomento di `predict`, e ogni valore predefinito è disattivato.

| Argomento | Predefinito | Effetto |
|---|---|---|
| `batch` | `1` | Immagini per forward pass, per sorgenti cartella e lista |
| `cuda_graph` | `False` | Riesegue il forward da un CUDA graph catturato |
| `tiling` | `False` | Divide un'immagine grande in tasselli sovrapposti |
| `overlap_ratio` | `0.2` | Sovrapposizione dei tasselli quando `tiling` è attivo |
| `augment` | `False` | Esegue le viste specchiate e le unisce |
| `half` | | Accettato, segnalato con un avviso e ignorato |
| `device` | `None` | Sposta il modello prima di fare la predizione |

Anche `imgsz` incide sul costo, perché fissa la risoluzione a cui gira il
modello, ma è prima di tutto un argomento di accuratezza e sta con il modello
più che qui.

## Batching

<code-tabs name="batch" />

`batch` vale per le sorgenti cartella e lista. Con `batch=1` ogni immagine fa un
forward pass a sé. Sopra `1`, ogni blocco viene preprocessato, impilato in un
unico tensore, eseguito una volta sola e poi ri-suddiviso, così il postprocess a
immagine singola già esistente di ogni famiglia vede quello che si aspetta.

Il percorso impilato viene preso solo quando valgono tutte queste condizioni:

- `batch` è maggiore di `1`
- `tiling` è disattivato
- la test-time augmentation non è attiva
- la famiglia imposta `SUPPORTS_BATCHED_PREDICT`
- la rete sottostante non è in modalità addestramento

L'ultima condizione non è un tecnicismo. Una rete in modalità addestramento
normalizzerebbe il blocco impilato con statistiche di batch calcolate fra le
immagini, lasciando che immagini dello stesso blocco si cambino a vicenda le
predizioni, quindi quelle esecuzioni restano sequenziali.

`SUPPORTS_BATCHED_PREDICT` è vero per impostazione predefinita. Queste famiglie
si tirano fuori ed eseguono un'immagine per forward a prescindere da `batch`:
Depth Anything V2, Depth Anything 3, EoMT, Faster R-CNN, FCOS, HRNet, L2CS-Net,
LibreMODUS, MiDaS, MoGe-2, PP-OCRv5, Real-ESRGAN, RetinaNet, SAM 3D Body,
SwinIR, YOLOv1, ZipDepth, ogni rilevatore a vocabolario aperto e ogni vision
language model.

C'è un altro fallback. Se il preprocessing non restituisce tensori
`(1, C, H, W)` uniformi, con shape, dtype e device coincidenti su tutto il
blocco, il blocco viene eseguito in sequenza invece di essere impilato, così la
correttezza non dipende mai dal fatto che le immagini siano per caso della
stessa dimensione.

Combina `batch` con `stream=True` su una cartella grande per avere forward in
batch senza tenere in memoria ogni risultato.

## CUDA graph

<code-tabs name="graphs" />

Un CUDA graph registra un forward pass una volta sola e lo riesegue come un
unico lancio. I rilevatori piccoli passano gran parte del tempo a batch 1 a
lanciare kernel, quindi comprimere quei lanci fa guadagnare throughput, e
l'output del replay è identico bit per bit all'esecuzione eager.

`cuda_graph` accetta tre valori. `False` è il predefinito e non fa nulla. `True`
cattura al primo utilizzo per ogni shape di input. `"auto"` aspetta che una
shape si ripeta prima di catturare, così il lavoro one-shot o a shape variabile
non paga mai il costo della cattura.

`capture_graph(imgsz=None, batch=1, dtype=None)` sposta quel costo fuori dalla
prima richiesta. Un graph è valido solo per la shape esatta che ha catturato,
quindi `batch` qui deve corrispondere a come verrà chiamato poi `predict`.

`graph_info()` riporta i graph catturati, i conteggi dei replay e l'eventuale
motivo per cui l'esecuzione è ricaduta su eager. `release_graphs()` li libera
insieme ai loro buffer statici.

La cattura richiede CUDA e una famiglia che abbia aderito tramite
`SUPPORTS_CUDA_GRAPH`, perché serve un forward senza lavoro visibile dall'host e
questo si verifica famiglia per famiglia. Chiederla su una famiglia che non ha
aderito solleva `NotImplementedError` invece di eseguire in eager di nascosto.

Un graph registra indirizzi di memoria, non valori, quindi qualsiasi cosa
sposti i parametri lo fa decadere. Cambiare device con `predict(device=...)`,
quantizzare e dequantizzare invalidano tutti i graph catturati.

La matrice di supporto completa per famiglia, i punti di taglio e il contratto
sulla numerica sono su [CUDA graph](/docs/reference/cuda-graphs).

## Precisione

<code-tabs name="precision" />

`half=True` non fa nulla quando fai una predizione. È accettato per
compatibilità con la riga di comando, genera un avviso che dice che è un no-op
ed è scartato prima di arrivare a qualsiasi famiglia. Il flag `--half` della CLI
stampa lo stesso avviso per un modello `.pt`.

Ci sono due strade vere verso una precisione più bassa.

Per un artefatto esportato, la precisione si sceglie quando esporti, con
`export(format=..., half=True)`, e il file risultante si ricarica con
`LibreYOLO()` senza modifiche.

Per l'esecuzione in PyTorch, `model.quantize(recipe="fp16")` converte il modello
a float16 e installa hook che mantengono float32 agli ingressi e alle uscite del
modello. `"bf16"` fa lo stesso con bfloat16. Nessuno dei due cast legge dati di
calibrazione, quindi per loro `calib` viene ignorato. La quantizzazione copre
attualmente quattro famiglie: YOLOv9, RF-DETR, BiRefNet e FeyNobg. Un cast su un
device CPU registra un avviso che sarà lento, quindi queste ricette sono pensate
per una GPU.

Entrambe le strade cambiano la numerica. Nessuna delle due garantisce gli stessi
rilevamenti senza altri interventi, quindi valida prima di mettere in
produzione.

## Inferenza a tasselli

<code-tabs name="tiling" />

Il tiling ritaglia un'immagine grande in tasselli quadrati sovrapposti, fa una
predizione su ognuno e unisce i risultati. È l'opzione per gli oggetti piccoli
in immagini ad alta risoluzione, dove ridimensionare l'immagine intera
rimpicciolisce i target sotto ciò che il modello riesce a risolvere.

La dimensione del tassello è la dimensione di input del modello, oppure `imgsz`
quando lo passi, e deve essere quadrata. `overlap_ratio` vale `0.2` per
impostazione predefinita. I tasselli che si sovrappongono vengono riconciliati
con una non-maximum suppression per classe alla soglia `iou`, e la lista unita
viene poi troncata a `max_det`. Questo significa che `iou` ha un effetto sulle
predizioni a tasselli anche per le famiglie che non eseguono alcuna NMS propria.

Il tiling viene saltato, non solo reso economico, quando l'immagine ci sta già:
se entrambe le dimensioni sono pari o inferiori alla dimensione di input, viene
eseguito un normale forward. Viene saltato anche per la classificazione, la
segmentazione semantica e il task `embed`, che ricadono su un unico passaggio
perché lì il tiling non ha senso.

Solleva un errore per i task il cui payload non si può ricucire insieme:
maschere di segmentazione di istanze, box orientati, punti, profondità, bordi e
normali. Non può essere combinato con `augment`.

Il risultato porta `result.tiled` e `result.num_tiles`. Con `save=True`, le
esecuzioni a tasselli scrivono una directory sotto `runs/tiled_detections` che
contiene ogni tassello, l'immagine annotata, una visualizzazione a griglia e un
`metadata.json` che registra la dimensione del tassello, la sovrapposizione e le
soglie, con `result.tiles_path` e `result.grid_path` che puntano a essi.

## Test-time augmentation

<code-tabs name="tta" />

`augment=True` esegue l'immagine più di una volta e unisce i rilevamenti con una
non-maximum suppression per classe alla soglia `iou`. Come il tiling, questo
rende `iou` determinante per le famiglie che altrimenti lo ignorano.

In pratica si tratta di un ribaltamento orizzontale. La lista di scale
`TTA_SCALES` contiene per impostazione predefinita una sola scala, `1.0`, e
nessuna famiglia distribuita la sovrascrive, quindi ogni famiglia esegue due
passaggi: l'immagine originale e il suo specchio. Le famiglie marcate
`TTA_FIXED_SIZE` ridimensionano a un quadrato fisso, il che rende comunque il
multi-scala un no-op per loro.

La segmentazione semantica e quella panottica usano un'unione diversa. La loro
vista ribaltata viene ribaltata di nuovo e le due distribuzioni softmax vengono
mediate prima dell'argmax, invece di essere unite come box.

La test-time augmentation non è disponibile per ogni task. Solleva un errore per
box orientati, posa, punti, profondità, normali, bordi, restauro, OCR e modelli
di embedding, e non può essere combinata con il tiling.

Queste famiglie la disattivano del tutto, quindi `augment=True` esegue un
singolo passaggio normale: BiRefNet, CenterNet, CLIP, DexiNed, FOMO, HRNet,
L2CS-Net, LibreMODUS, NAFNet, PP-OCRv5, Real-ESRGAN, RetinaNet, SAM 3D Body,
SigLIP2, SwinIR, TEED, ogni variante di SAM, ogni rilevatore a vocabolario
aperto e ogni vision language model.

## Misurare

Niente in questa pagina riporta un numero di latenza, perché un millisecondo
senza il suo hardware, runtime, precisione e dimensione del batch non è un dato.
Le cifre misurate su hardware e runtime diversi sono pubblicate su
[visionanalysis.org](https://www.visionanalysis.org), e `libreyolo profile`
misura un modello specifico sulla macchina che hai davanti.
