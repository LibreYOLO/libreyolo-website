---
title: Data augmentation
seo_title: 'Data augmentation per l''addestramento in LibreYOLO, con esempi'
description: >-
  Cosa fa ogni parametro di data augmentation di LibreYOLO, mostrato su
  un'immagine reale dal codice di addestramento della libreria stessa, con il
  suo intervallo, il suo valore predefinito e le famiglie che lo leggono.
lead: >-
  La data augmentation di addestramento consiste in trasformazioni casuali
  applicate a ogni campione di addestramento. Si configura con i parametri di
  TrainConfig, e ogni famiglia di modelli esegue la propria pipeline, quindi un
  parametro modifica i campioni solo dove quella pipeline lo legge.
keywords:
  - data augmentation yolo
  - mosaic augmentation
  - mixup
  - hsv augmentation
  - random affine
  - random erasing
  - small object detection
  - rilevamento oggetti piccoli
  - esempi data augmentation
  - copy paste augmentation
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: 1.8.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_prob=1.0,
            mixup_prob=0.15,
            hsv_prob=1.0,
            flip_prob=0.5,
            no_aug_epochs=15,
        )
    - label: CLI
      language: bash
      code: |
        # La CLI scrive mosaic_prob come mosaic e mixup_prob come mixup.
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 mosaic=1.0 mixup=0.15 hsv_prob=1.0 \
          flip_prob=0.5 no_aug_epochs=15
  small:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # YOLOX parte da mosaic_scale=(0.1, 2.0), che può ridurre un
        # oggetto a un decimo della sua dimensione. (0.5, 1.5) si ferma a metà.
        model = LibreYOLO("LibreYOLOXs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_scale=(0.5, 1.5),
        )
    - label: CLI
      language: bash
      code: |
        # Sulla CLI una coppia è formata da due numeri separati da una virgola.
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=100 mosaic_scale=0.5,1.5
  support:
    - label: Leggere la tabella di supporto di una famiglia
      language: python
      code: |
        from libreyolo.data.augment.spec import AUG_KNOBS, aug_support

        for knob, description in AUG_KNOBS.items():
            support = aug_support("yolo9")[knob]
            print(f"{knob:16} {support.status:16} {support.note or description}")
    - label: Solo quelli ignorati
      language: python
      code: |
        from libreyolo.data.augment.spec import ignored_aug_params

        print(sorted(ignored_aug_params("rfdetr")))
  classify:
    - label: Pacchetto di classificazione
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(
            data="my-classification-dataset",
            epochs=50,
            auto_augment="randaugment",
            erasing=0.25,
            mixup=0.2,
            cutmix=0.2,
        )
source_hash: 72d22a38d647de0a
---

## Impostare i parametri

I parametri di augmentation sono normali argomenti di `train()`.

<code-tabs name="train" />

Tre di questi hanno una forma più breve, accettata sia da `train()` sia dalla
CLI: `mosaic` corrisponde a `mosaic_prob`, `fliplr` a `flip_prob` e, nel
rilevamento, `mixup` corrisponde a `mixup_prob`. Passare una forma breve e il
suo campo con valori diversi solleva un errore. Nella 1.8.0 `perspective` non ha
un flag sulla CLI, quindi si imposta da Python.

I parametri di classificazione vengono validati quando si costruisce la loro
pipeline, e la CLI controlla anche `flip_prob`, `flipud`, `cutmix` ed `erasing`.
`TrainConfig` non controlla l'intervallo dei parametri di rilevamento, quindi
con `degrees=100` l'addestramento usa rotazioni fino a 100 gradi. La [matrice di
data augmentation](/docs/reference/augmentation-matrix) riporta l'intervallo di
ogni parametro.

## Tre stati, non due

Che un parametro abbia effetto o meno dipende dalla famiglia. La libreria ne
tiene una tabella dichiarativa, e ogni voce è uno di tre stati.

`used` significa che il parametro arriva alla pipeline e modifica i campioni.
`ignored` significa che non arriva mai alla pipeline, quindi impostarlo non fa
nulla. `gated_by_mosaic` significa che si applica solo ai campioni passati dal
ramo del mosaico, quindi con `mosaic_prob=0` non scatta mai, anche se è
collegato.

Il terzo stato è quello che coglie di sorpresa. In una pipeline in stile YOLOX
la trasformazione affine agisce sul canvas del mosaico e MixUp fonde un campione
a mosaico, quindi `mosaic_prob=0` disattiva silenziosamente `degrees`,
`translate`, `shear`, `perspective`, `mosaic_scale`, `mixup_prob` e
`mixup_scale` tutti insieme. Il trainer registra un avviso specifico per il caso
di MixUp:

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

La CLI segnala anche i parametri ignorati, elencando solo quelli che hai
effettivamente scritto:

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

Chiamato da Python, `train()` non segnala i parametri ignorati. La [matrice di
data augmentation](/docs/reference/augmentation-matrix) riporta lo stato di ogni
parametro per ogni famiglia e il valore da cui parte ogni famiglia. La tabella è
la spec distribuita in `libreyolo/data/augment/spec.py`, che i test della
libreria stessa confrontano con le pipeline, e si può leggere direttamente:

<code-tabs name="support" />

## Cosa fa ogni parametro

Le figure qui sotto sono prodotte dalle trasformazioni di addestramento proprie
di LibreYOLO 1.8.0, eseguite sull'immagine di esempio tramite un dataset di una
sola immagine, quindi ogni tessera del mosaico e ogni immagine partner di MixUp
è la stessa foto. Il box è quello della persona che LibreYOLO9t rileva in
quell'immagine. Sotto ogni pannello sono indicati l'impostazione e la dimensione
del box più grande, in pixel alla risoluzione di addestramento di 640 pixel.
Senza augmentation la persona misura 291 × 217 px.

Diversi parametri geometrici sono intervalli: `degrees=10` significa un angolo
estratto in un punto qualsiasi tra -10 e +10 gradi. I loro pannelli fissano
l'estrazione al valore indicato nella didascalia, quindi mostrano l'effetto più
forte che un'impostazione consente invece di un campione tipico. Le figure di
rilevamento provengono dalla pipeline di YOLOv9. YOLOv7 si addestra attraverso
di essa senza modifiche, e RT-DETR e RT-DETRv2 usano il suo wrapper del mosaico
con una trasformazione per campione propria. YOLOX, RTMDet, PicoDet e FOMO
costruiscono il mosaico e la trasformazione allo stesso modo in un wrapper
separato, il cui MixUp è diverso.

### mosaic_prob

La probabilità che un campione sia un mosaico: quattro immagini disposte su un
canvas grande il doppio della dimensione di ingresso, che la trasformazione
affine riduce poi alla dimensione di ingresso. Nelle pipeline in stile YOLOX la
trasformazione e MixUp agiscono solo sui campioni a mosaico, quindi entrambi si
fermano quando questo parametro vale 0. RT-DETR, RT-DETRv2, PicoDet e FOMO lo
hanno a 0 di default.

<aug-figure name="mosaic" />

Il secondo pannello fissa la scala della trasformazione a 0.5×, dove il canvas
entra per intero. Gli ultimi due sono estrazioni ordinarie con i valori
predefiniti di YOLOv9, e la didascalia indica la scala estratta da ciascuno.

### mosaic_scale

L'intervallo da cui si estrae lo zoom della trasformazione. A 1× un oggetto
mantiene la dimensione che ha in inferenza, e a 0.5× ne ha la metà. Il valore
predefinito di base è `(0.1, 2.0)`, che YOLOX, YOLOv7, RTMDet, PicoDet, RT-DETR,
RT-DETRv2 e FOMO mantengono, quindi i loro oggetti possono ridursi a un decimo
della loro dimensione. YOLOv9 usa di default `(0.5, 1.5)`. YOLO-NAS e PP-YOLOE
non hanno il mosaico e riusano questo intervallo come scala della loro
trasformazione affine per campione.

<aug-figure name="mosaic_scale" />

Sopra 1× l'output mostra solo la parte in alto a sinistra del canvas, quindi gli
oggetti vengono tagliati sul bordo o escono dall'inquadratura.

### degrees

L'intervallo di rotazione: ogni campione trasformato ruota di un angolo estratto
tra `-degrees` e `+degrees`. La rotazione fa perno sul vertice in alto a
sinistra del canvas, quindi un angolo ampio porta gran parte dell'immagine fuori
dall'inquadratura. YOLOX e YOLOv7 mantengono il valore predefinito di base di
10, e YOLOv9 usa di default 0.

<aug-figure name="degrees" />

### translate

L'intervallo di spostamento, come frazione del lato di ingresso: ogni campione
trasformato si sposta fino a `translate` volte 640 px su ciascun asse con
`imgsz=640`. Il valore predefinito di base è 0.1. YOLO-NAS e PP-YOLOE usano di
default 0.25, e RTMDet, PicoDet e FOMO 0.

<aug-figure name="translate" />

### shear

L'intervallo di shear in gradi, estratto separatamente per ciascun asse. YOLOX e
YOLOv7 mantengono il valore predefinito di base di 2, e tutte le altre famiglie
che lo leggono usano di default 0.

<aug-figure name="shear" />

### perspective

Una trasformazione proiettiva sopra a rotazione, scala, shear e traslazione.
Ogni vertice del canvas si sposta verso l'interno di una quantità casuale fino a
`min(perspective * 100, 0.2)` volte metà del lato del canvas, per asse, quindi
`perspective=0.001` sposta un vertice al massimo del 5 percento del lato a
qualsiasi risoluzione, e ogni valore da `0.002` in su raggiunge lo stesso
massimo. Il valore predefinito `0` mantiene la trasformazione affine, e un
valore negativo solleva `ValueError` quando viene trasformato il primo campione.
I pannelli condividono una stessa estrazione casuale, scalata in base al valore.
YOLOv9 e le sue varianti E2E e P2, YOLOX, YOLOv7, YOLO-NAS, PP-YOLOE, RTMDet,
PicoDet, RT-DETR e RT-DETRv2 la applicano.

<aug-figure name="perspective" />

### hsv_prob

La probabilità di un jitter di colore HSV. Solo la probabilità è un parametro.
Il jitter sposta la tonalità fino a 5 sulla scala della tonalità da 0 a 179 di
OpenCV, e saturazione e valore fino a 30 sulla scala da 0 a 255, e ciascuno dei
tre spostamenti viene saltato con probabilità pari a un mezzo. Ogni pipeline di
rilevamento che legge `hsv_prob` usa queste quantità. Gli ultimi tre pannelli
sono i più visibili su 119 estrazioni.

<aug-figure name="hsv" />

### flip_prob e flipud

Le probabilità di flip orizzontale e verticale. `flip_prob` parte da 0.5 in ogni
famiglia tranne FOMO, che lo ha a 0 di default. `flipud` parte da 0. RTMDet,
PicoDet, RT-DETR, RT-DETRv2, FOMO, PP-YOLOE e le famiglie in stile DETR non
hanno il flip verticale e lo ignorano. La ricetta di PP-YOLOE usa invece una
rotazione casuale di 90 gradi.

<aug-figure name="flip" />

### mixup_prob e mixup_scale

MixUp fonde un secondo campione in quello corrente e mantiene i box di
entrambi. Nella pipeline di YOLOv9 il partner è un campione intero senza
mosaico, il rapporto di fusione viene estratto in modo uniforme tra 0 e 1, e
`mixup_scale` non viene mai letto, anche se la spec della libreria lo elenca
come condizionato al mosaico. Nel wrapper di YOLOX, usato da YOLOX, RTMDet,
PicoDet e FOMO, e in quello di YOLO-NAS, il partner viene prima riscalato di un
fattore estratto da `mixup_scale` e le due immagini vengono mediate con un
rapporto fisso di 0.5. Il valore predefinito di base di `mixup_prob` è 1.0, e
YOLOv9 usa di default 0.

<aug-figure name="mixup" />

Gli ultimi due pannelli fondono un secondo campione in un mosaico a 0.5×. La
loro didascalia indica il rapporto di fusione λ, il peso del campione a
mosaico.

### no_aug_epochs

`no_aug_epochs` è `used` ovunque, ma non significa la stessa cosa ovunque. Sulle
pipeline a mosaico disattiva il mosaico e MixUp per le epoche finali. Sulle
pipeline in stile DETR ferma le augmentation fotometriche, di zoom-out e di
ritaglio, e dà forma alla coda dello schedule. Sulle pipeline di classificazione
disattiva auto-augmentation, erasing, MixUp e CutMix, mentre ritaglio e flip
restano.

## Ricette in stile DETR

D-FINE, Dome-DETR, DEIM, DEIMv2, RT-DETRv4 e TinyFormer si addestrano con
un'unica trasformazione di torchvision, senza mosaico e senza trasformazione
affine, e lo stesso vale per EdgeCrafter (EC) nel rilevamento. Le tre operazioni
forti di questa trasformazione sono costanti della ricetta e non parametri:

- `RandomPhotometricDistort` applica ogni variazione di colore (luminosità da
  0.875 a 1.125, contrasto e saturazione da 0.5 a 1.5, tonalità ±0.05 e una
  permutazione casuale dei canali) con probabilità 0.5.
- `RandomZoomOut`, con probabilità 0.5, colloca l'immagine su un canvas nero
  fino a quattro volte il suo lato, quindi gli oggetti possono ridursi a un
  quarto della loro dimensione.
- `RandomIoUCrop`, applicato con probabilità 0.8, ritaglia una regione tra 0.3 e
  1.0 di ciascun lato, scelta in modo da conservare almeno una parte dei box.

Dopo questi passaggi vengono scartati i box di 1 px o meno su uno qualsiasi dei
lati. Tutte e tre le operazioni si fermano all'epoca di arresto della famiglia,
al più tardi quando iniziano le `no_aug_epochs` epoche finali. Gli unici
parametri che questa pipeline legge sono `flip_prob` e `no_aug_epochs`.

<aug-figure name="detr_ops" />

Tre campioni dall'inizio di un addestramento di D-FINE, con tutte e tre le
operazioni attive:

<aug-figure name="detr_samples" />

RF-DETR ha una pipeline tutta sua: un flip orizzontale, poi, con probabilità
`crop_resize_prob`, un ridimensionamento del lato più corto a 400, 500 o 600 px
e un ritaglio quadrato casuale da 384 a 600 px, e infine un ridimensionamento
diretto alla risoluzione di addestramento quadrata, che stira l'immagine invece
di applicare il letterbox. `crop_resize_prob` vale 0.5 di default e si imposta
da Python.

<aug-figure name="rfdetr_crop" />

## Oggetti piccoli

`erasing` non si applica al rilevamento. Random Erasing appartiene al pacchetto
di classificazione e nessuna pipeline di rilevamento ha un passaggio di erasing,
quindi un rilevatore si addestra allo stesso modo con qualsiasi valore di questo
parametro, e non c'è nulla da disattivare. Questi sono i passaggi che
rimpiccioliscono o eliminano gli oggetti piccoli:

- `mosaic_scale`. Sulle pipeline a mosaico la dimensione di un oggetto viene
  moltiplicata per la scala estratta, quindi il limite inferiore stabilisce la
  dimensione minima che raggiunge. Con il valore predefinito di base
  `(0.1, 2.0)` un oggetto di 20 px può risultare di 2 px. La pipeline di YOLOv9
  scarta poi i box di 2 px o meno su uno qualsiasi dei lati, e la sua
  trasformazione finale, come quella di YOLOX, scarta quelli di 1 px o meno su
  uno qualsiasi dei lati. I pixel dell'oggetto restano nell'immagine senza
  etichetta. Un limite inferiore
  di 0.5, come in `(0.5, 1.5)` di YOLOv9, mantiene ogni oggetto a metà della sua
  dimensione di inferenza o più.
- `degrees`, `translate`, `shear` e `perspective` spingono gli oggetti verso il
  bordo, dove i loro box vengono ritagliati e passano dallo stesso filtro sulla
  dimensione.
- Nelle ricette in stile DETR, `RandomZoomOut` può ridurre gli oggetti a un
  quarto della loro dimensione e `RandomIoUCrop` può lasciarli fuori dal
  ritaglio. Nessuno dei due è un parametro, ed entrambi si fermano all'epoca di
  arresto.
- Il ritaglio di RF-DETR può tagliare gli oggetti sul suo bordo.
  `crop_resize_prob=0` lo disattiva.
- La risoluzione di addestramento scala ogni oggetto: il letterbox ridimensiona
  un'immagine di un fattore pari a `imgsz` diviso per il suo lato più lungo,
  quindi con `imgsz=1280` ogni oggetto ha il doppio della larghezza e
  dell'altezza che ha a 640. [YOLOv9](/docs/models/yolov9) ha anche una variante
  P2 con una testa a stride 4 per gli oggetti piccoli.

<code-tabs name="small" />

## Il pacchetto di classificazione

Sei parametri agiscono solo sulla pipeline di classificazione. Le famiglie di
rilevamento li ignorano tutti e sei.

<code-tabs name="classify" />

`scale` è l'intervallo di area del ritaglio casuale. `scale=0.5` significa
`(0.5, 1.0)`, e una coppia esplicita imposta entrambi i limiti. Un limite
inferiore piccolo può lasciare l'oggetto del tutto fuori dal ritaglio, come nel
terzo pannello.

<aug-figure name="cls_crop" />

`auto_augment` accetta `"randaugment"`, `"autoaugment"`, `"augmix"` o `None`, e
applica un'estrazione di quella policy a ogni campione.

<aug-figure name="auto_augment" />

`erasing` è la probabilità di Random Erasing, in `[0, 1)`. Viene eseguito dopo
la normalizzazione e riempie di zeri un rettangolo che copre dal 2 al 33
percento dell'immagine, il che, dopo la normalizzazione, corrisponde al colore
medio di ImageNet.

<aug-figure name="erasing" />

`mixup` e `cutmix` sono probabilità per batch che producono etichette soft. Al
massimo uno dei due viene applicato per batch, MixUp per primo, quindi i due
sono additivi e la loro somma deve essere al massimo 1. Nei pannelli, A e B sono
due ritagli della stessa foto, e ogni didascalia indica l'etichetta soft su cui
si addestra l'immagine mescolata.

<aug-figure name="batch_mix" />

`auto_augment`, `erasing`, `mixup` e `cutmix` sono disattivati di default,
quindi l'addestramento di classificazione non cambia se non lo chiedi.
`train()` e la CLI instradano `mixup` in base al task: la classificazione usa il
mixing a livello di batch, mentre il rilevamento usa `mixup_prob`.

`crop_pct=None` conserva il rapporto di ridimensionamento di valutazione della
famiglia. Un valore esplicito cambia il ritaglio di valutazione del passaggio di
validazione eseguito durante l'addestramento, mentre l'esportazione mantiene il
preprocessing nativo della famiglia.

La CLI espone `auto_augment`, `erasing`, `cutmix`, `scale`, `crop_pct`,
`fliplr` e `flipud`. Nella classificazione `mixup` indica il mixing a livello di
batch e vale 0.0 di default. I flip orizzontali valgono 0.5 di default e quelli
verticali 0.0.

`no_aug_epochs` disattiva auto-augmentation, erasing, MixUp e CutMix per le
epoche finali, mantenendo ritaglio e flip. Le ricette di classificazione
impostano questa coda a 0 di default.

## Parametri specifici per famiglia

Alcuni parametri risiedono nella sottoclasse di configurazione di una famiglia
invece che nella classe base, quindi esistono solo per quella famiglia e non
hanno un flag sulla CLI.

| Famiglia | Parametro | Effetto |
|---|---|---|
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste` | Probabilità di augmentation copy-paste sulle istanze, solo `task="segment"` |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste_mode` | `"flip"` riusa lo stesso campione specchiato, `"mixup"` preleva un secondo campione |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `rot90` | Probabilità di rotazione casuale di 90 gradi |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `max_labels` | Limite di ground truth per immagine nelle trasformazioni di addestramento, default 300 |
| RF-DETR | `copy_paste`, `copy_paste_mode` | Copy-paste per `task="segment"`, solo modalità `"flip"` |
| RF-DETR | `crop_resize_prob` | Probabilità di crop-resize casuale nella pipeline nativa, default 0.5 |
| D-FINE, EdgeCrafter | `crop_resize_prob` | Probabilità di crop-resize casuale, solo `task="segment"` |
| EdgeCrafter, YOLO-NAS | `brightness_contrast_prob`, `affine_prob` | Probabilità del jitter sul percorso pose e della trasformazione affine che tiene conto dei keypoint |

`max_labels` è quello che perde dati. I box oltre il limite vengono scartati con
un avviso unico invece che con un errore, quindi con immagini dense come la
fotografia aerea va alzato. La memoria dell'assegnazione delle etichette segue
l'immagine più densa del batch, non il limite.

Il mosaico e MixUp sono disattivati per l'addestramento con box orientati
indipendentemente dai parametri, perché l'augmentation che tiene conto dei
vertici per i box ruotati non è implementata.

## Correlati

- [Matrice di data augmentation](/docs/reference/augmentation-matrix) per lo
  stato di ogni famiglia per ogni parametro, gli intervalli e i valori
  predefiniti di ogni famiglia.
- [Iperparametri](/docs/train/hyperparameters) per `no_aug_epochs` come
  argomento dello schedule e per il resto di `train()`.
- [Dataset](/docs/train/datasets) per i formati di etichette che queste
  trasformazioni accettano in ingresso.
