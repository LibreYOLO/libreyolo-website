---
title: Installazione leggera
seo_title: Esegui l'inferenza ONNX di LibreYOLO senza PyTorch
description: >-
  Installa LibreYOLO con --no-deps ed esegui il rilevamento ONNX solo con numpy,
  senza torch su disco. La tecnica, i suoi limiti e l'elenco esatto dei
  pacchetti.
lead: >-
  Il percorso di inferenza ONNX di LibreYOLO è numpy dall'inizio alla fine,
  decode e NMS inclusi. Niente su quel percorso ha bisogno di PyTorch a runtime,
  quindi un'installazione che salta la risoluzione delle dipendenze può eseguire
  il rilevamento con torch assente dalla macchina.
keywords:
  - inferenza senza torch
  - libreyolo senza pytorch
  - onnx inference senza torch
  - installazione leggera libreyolo
  - pip install no-deps
  - libreyolo spazio su disco
  - onnxruntime python
  - torch solo cpu
last_verified: 1.5.0
meta:
  - label: Si applica a
    value: 'Rilevamento ONNX, sette famiglie di modelli'
  - label: Punto di ingresso
    value: libreyolo.backends.onnx.OnnxBackend
    mono: true
  - label: Livello di supporto
    value: 'Best effort, non è una distribuzione a parte'
snippets:
  install:
    - label: Leggera
      language: bash
      code: >
        # Installa il pacchetto senza il suo elenco di dipendenze, poi fornisci

        # i quattro pacchetti che il percorso di rilevamento ONNX importa
        davvero.

        pip install --no-deps libreyolo

        pip install numpy pillow opencv-python-headless onnxruntime
    - label: Torch solo CPU
      language: bash
      code: |
        # Prova prima questo. Mantiene ogni funzionalità ed evita il wheel CUDA,
        # che è dove se ne va la maggior parte dello spazio su disco.
        pip install libreyolo --index-url https://download.pytorch.org/whl/cpu
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo.backends.onnx import OnnxBackend


        model = OnnxBackend("libreyolo9t.onnx")

        result =
        model.predict("https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg")


        # qui xyxy è un ndarray di numpy, non un tensore torch.

        print(result.boxes.xyxy)

        print(result.boxes.conf)

        print(result.boxes.cls)
source_hash: e60e83d32d13026e
---

## Perché funziona

`pip install --no-deps libreyolo` installa il pacchetto e salta del tutto il suo
elenco di dipendenze. Nulla viene risolto al posto tuo, e diventi responsabile
di installare ciò che usi davvero.

Questo è utile solo se il percorso di codice che ti interessa davvero non ha
bisogno delle dipendenze che hai saltato, e per il rilevamento ONNX non ne ha
bisogno. Il decode, non-maximum suppression inclusa, è numpy. Le ricette di
preprocessing sono numpy. PyTorch è una dipendenza per l'addestramento e per
l'inferenza eager, e su questo percorso non viene mai chiamato.

Prima di questa release l'import falliva comunque: importare qualsiasi cosa
sotto `libreyolo.models` costruiva ogni classe di modello per popolare il
registro di rilevamento automatico dei checkpoint, e quelle classi sono
sottoclassi di `torch.nn.Module`. Le ricette di preprocessing ora vivono in un
pacchetto tutto loro, `libreyolo.preprocess`, e l'import di torch è rimandato
finché qualcosa non tocca un attributo di torch, così il percorso ONNX si
importa con torch assente dalla macchina. Quel pacchetto contiene un
preprocessore nativo in numpy per famiglia: `yolo9`, `yolonas`, `yolox`, `ec`,
`rtdetr`, `rfdetr`, `dfine`, `deim` e `deimv2`, due in più delle sette famiglie
verificate end to end qui sotto. Ogni
`libreyolo/models/<family>/utils.py` ri-esporta da lì, così i percorsi di import
esistenti continuano a funzionare.

## Prova prima il wheel solo CPU

Quasi tutti quelli che chiedono questa cosa vogliono evitare un'installazione da
diversi gigabyte, e la dimensione è concentrata in un punto solo: il wheel
`torch` predefinito include CUDA. Una build solo CPU è una frazione di quella e
non richiede nessun percorso di installazione speciale.

<code-tabs name="install" />

L'opzione solo CPU mantiene ogni funzionalità di LibreYOLO: addestramento,
validazione, ogni task, ogni famiglia, la CLI. Prendi il percorso leggero quando
vuoi zero torch sulla macchina, non semplicemente meno torch.

## Cosa copre l'installazione leggera

| | |
|---|---|
| Task | Rilevamento |
| Formato | ONNX |
| Punto di ingresso | `OnnxBackend` |
| Interfaccia | Libreria Python |

Su questo percorso sono state verificate sette famiglie:
[YOLOv9](/docs/models/yolov9), [YOLO-NAS](/docs/models/yolo-nas),
[EdgeCrafter](/docs/models/edgecrafter), [RT-DETR](/docs/models/rt-detr),
[RF-DETR](/docs/models/rf-detr), [D-FINE](/docs/models/d-fine) e
[DEIM](/docs/models/deim), contando in ciascuna anche le sue varianti.

Questo è l'ambito verificato, non un confine che la libreria impone. Altri task
e altre famiglie sono semplicemente fuori da ciò che è stato controllato: alcuni
tireranno dentro torch quando li chiami, e qualcuno potrebbe funzionare per
caso. Tratta tutto ciò che va oltre questo elenco come non testato, piuttosto
che come supportato o come rotto.

Al suo interno i risultati sono identici a quelli dell'installazione normale,
non semplicemente vicini. Ogni famiglia è stata esportata in ONNX ed eseguita
due volte, una normalmente e una con torch bloccato; box, punteggi e classi
coincidevano esattamente. Un test di parità nella suite impedisce a quel
contratto di andare alla deriva.

## Le cinque cose che colgono di sorpresa

**Usa `OnnxBackend`, non le classi dei modelli.** `LibreYOLO9("model.onnx")`
richiede comunque torch, perché `LibreYOLO9` è a sua volta una sottoclasse di
`nn.Module`. È l'errore più probabile, dato che ogni altra pagina di questa
documentazione carica un modello attraverso la sua classe o attraverso
`LibreYOLO()`.

**Esporta da un'altra parte.** Produrre il file `.onnx` richiede torch, quindi
la macchina leggera non può crearne uno. Esporta su una macchina di sviluppo o
di CI e spedisci l'artefatto al target snello.

**I risultati portano array numpy.** Qui `result.boxes.xyxy` è un `ndarray`. I
contenitori accettano entrambi i tipi, quindi i nomi degli attributi non
cambiano, ma il codice che chiama `.cpu()` o `.numpy()` su un risultato
fallirà.

**Una singola immagine restituisce un singolo `Results`.** `predict()`
restituisce un `Results` per una sola immagine e una lista per più immagini.
Indicizzare un singolo risultato con `[0]` seleziona il primo rilevamento, non
la prima immagine, il che ti dà silenziosamente un risultato con un box solo
invece di sollevare un errore.

**La CLI non funzionerà.** `typer` e `click` non sono tra i quattro pacchetti,
quindi il comando `libreyolo` non è disponibile. Questa è un'installazione come
libreria.

## Predizione

<code-tabs name="predict" />

Sostituisci `onnxruntime` con `onnxruntime-gpu` per girare su CUDA. I quattro
pacchetti sono quelli che una `predict()` davvero priva di torch importa sul
serio, registrati durante la chiamata invece che dedotti a tavolino.
`opencv-python-headless` sostituisce il dichiarato `opencv-python`: stesso
modulo, nessuna libreria GUI, meno spazio su disco.

Delle restanti dipendenze dichiarate, `requests` serve solo a caricare
un'immagine da un URL, `pycocotools` e `scipy` sono validazione e valutazione, e
`typer` e `click` sono la CLI.

## Questo elenco andrà alla deriva, per scelta

L'elenco di pacchetti qui sopra è corretto per la release indicata in cima a
questa pagina. `--no-deps` ti fa rinunciare alla risoluzione delle dipendenze,
quindi nessuno lo controlla al posto tuo, e una release successiva potrebbe
importare qualcosa che qui non è elencato.

Se incontri un `ModuleNotFoundError`, la tecnica l'hai già capita: installa il
pacchetto mancante. È il modello di manutenzione previsto, non una segnalazione
di bug. Questo percorso è best effort e non è una distribuzione supportata a
parte, motivo per cui non esiste un secondo pacchetto leggero su PyPI e non c'è
alcun piano per farne uno.

Per confermare che il tuo ambiente sia davvero senza torch e non stia
silenziosamente ripiegando su una copia installata, mettilo per iscritto con un
assert:

```python
import importlib.util

assert importlib.util.find_spec("torch") is None, "torch is installed"
```

Quel controllo vale la pena tenerlo in CI per l'immagine snella. Senza, un
ambiente che per caso ha torch passerà ogni test senza dirti nulla.
