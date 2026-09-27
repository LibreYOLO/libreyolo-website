---
title: Sorgenti di predizione
seo_title: Sorgenti di predizione in LibreYOLO
description: >-
  Tutte le sorgenti accettate da predict: immagini, cartelle, URL, file video,
  webcam, RTSP, YouTube, cattura dello schermo, liste di immagini e file
  .streams.
lead: >-
  L'argomento source viene classificato prima che qualsiasi cosa venga aperta,
  così una sola chiamata gestisce un JPEG, una cartella, un MP4, un indice di
  webcam, un URL RTSP, una regione dello schermo o una lista di telecamere.
keywords:
  - yolo video inference python
  - rtsp
  - webcam object detection python
  - predict su una cartella di immagini
  - object detection cattura schermo
  - più stream rtsp
  - file streams
  - youtube inference python
  - vid_stride
  - stream=True
last_verified: 1.5.0
verification: >-
  Classificazione delle sorgenti letta da libreyolo/utils/source.py
  (classify_source, SourceKind, StreamSource, MultiStreamSource). Tipi di
  immagine accettati ed estensioni delle directory da
  libreyolo/utils/image_loader.py. Estensioni video e percorsi di salvataggio da
  libreyolo/utils/video.py. Sintassi di screen da libreyolo/utils/screen.py.
  Forme dei valori restituiti e valori predefiniti degli argomenti da
  InferenceRunner.__call__ in libreyolo/models/base/inference.py.
snippets:
  images:
    - label: Una sola immagine
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # Una sorgente a immagine singola restituisce un Results, non una lista.
        result = model(SAMPLE_IMAGE)
        print(len(result.boxes), "detections")
    - label: Immagini in memoria
      language: python
      code: |
        import numpy as np
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        pil_image = Image.open(SAMPLE_IMAGE)
        array = np.asarray(pil_image)
        raw_bytes = open(SAMPLE_IMAGE, "rb").read()

        for source in (pil_image, array, raw_bytes):
            result = model(source)
            print(type(source).__name__, len(result.boxes))
    - label: Una cartella
      language: python
      code: >
        from pathlib import Path

        from PIL import Image


        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        folder = Path("sample_folder")

        folder.mkdir(exist_ok=True)

        image = Image.open(SAMPLE_IMAGE)

        for index in range(3):
            image.save(folder / f"frame_{index}.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")


        # Una cartella restituisce una lista, un Results per immagine, ordinata
        per percorso.

        results = model(str(folder))

        print(len(results), "images")
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  video:
    - label: Un file video (usa un tuo clip)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Sostituisci clip.mp4 con un file video presente su disco.
        for result in model("clip.mp4", stream=True):
            print(result.frame_idx, len(result.boxes))
    - label: 'Un frame su tre, scritto su disco'
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("clip.mp4", stream=True, vid_stride=3, save=True):
            pass
  live:
    - label: Webcam (serve una telecamera collegata)
      language: python
      code: >
        import itertools


        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Indice webcam 0. Le sorgenti live non finiscono mai, quindi limita il
        ciclo.

        for result in itertools.islice(model(0, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
    - label: RTSP (serve l'URL raggiungibile di una telecamera)
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        source = "rtsp://user:password@192.168.1.64:554/Streaming/Channels/101"

        for result in itertools.islice(model(source, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  streams:
    - label: Un file .streams (usa le tue telecamere)
      language: python
      code: >
        import itertools

        from pathlib import Path


        from libreyolo import LibreYOLO


        Path("cameras.streams").write_text(
            "# one source per line, blank lines and comments are skipped\n"
            "rtsp://192.168.1.64:554/Streaming/Channels/101\n"
            "rtsp://192.168.1.65:554/Streaming/Channels/101\n",
            encoding="utf-8",
        )


        model = LibreYOLO("LibreYOLO9s.pt")

        for result in itertools.islice(model("cameras.streams", stream=True),
        100):
            print(result.frame_idx, len(result.boxes))
    - label: Una lista di telecamere
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        cameras = [0, "rtsp://192.168.1.64:554/Streaming/Channels/101"]

        for result in itertools.islice(model(cameras, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  screen:
    - label: Uno screenshot (servono mss e una sessione desktop)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Senza stream=True cattura un solo frame.
        result = model("screen")
        print(len(result.boxes), "detections")
    - label: 'Una regione di un monitor, in continuo'
      language: python
      code: >
        import itertools


        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # "screen <monitor> <left> <top> <width> <height>"

        for result in itertools.islice(model("screen 1 100 200 512 256",
        stream=True), 50):
            print(len(result.boxes))
source_hash: c371965951dd0181
---

## Come viene classificata una sorgente

`classify_source` esamina il valore prima che qualsiasi cosa venga aperta o
scaricata, in quest'ordine. Vince la prima regola che corrisponde.

| Sorgente | Interpretata come |
|---|---|
| `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` | Cattura dello schermo |
| Un `int` non negativo, o una stringa di cifre senza un file con quel nome | Webcam |
| Un URL `rtsp://`, `rtmp://`, `tcp://` o `udp://` | Stream di rete |
| Un URL `http(s)://` il cui percorso termina in `.m3u8` | Stream di rete |
| L'URL di una pagina YouTube | Stream di rete |
| Una lista o tupla i cui elementi sono tutti live o video | Più stream live |
| Qualsiasi altra lista o tupla | Batch di immagini |
| Un percorso che termina in `.streams` | Più stream live |
| Un percorso con un'estensione video | File video |
| Una directory esistente | Cartella di immagini |
| Qualsiasi altra cosa | Immagine singola |

Una lista che mescola sorgenti live e immagini solleva `TypeError`. Un indice di
webcam negativo solleva `ValueError`.

Il classificatore non tocca mai la rete, quindi un URL scritto male emerge
quando si apre la cattura, non quando chiami `predict`.

## Immagini

<code-tabs name="images" />

Una sorgente a immagine singola accetta sette tipi.

| Tipo | Interpretato come |
|---|---|
| `str` o `pathlib.Path` | File locale, `http(s)://`, `s3://` o `gs://` |
| `PIL.Image.Image` | Convertita in RGB |
| `numpy.ndarray` | 2D in scala di grigi, oppure 3D HWC o CHW; un array 4D usa la sua prima immagine |
| `torch.Tensor` | CHW o NCHW, letto come RGB; un tensore in batch usa la sua prima immagine |
| `bytes` | Dati immagine codificati |
| `io.BytesIO` | Dati immagine codificati |

Tutto viene convertito in RGB prima del preprocessing. Gli array NumPy sono
l'unico caso in cui l'ordine dei canali è ambiguo, quindi lo controlla
`color_format`: `"auto"` (il valore predefinito) lascia l'array com'è, `"bgr"`
inverte i canali, che è ciò di cui ha bisogno un frame letto con OpenCV.

Gli array float vengono riscalati in base al proprio intervallo: i valori
minori o uguali a `1.0` vengono moltiplicati per 255, quelli più alti vengono
troncati dentro `[0, 255]`. Un array RGBA scarta il suo canale alpha.

I percorsi remoti richiedono un pacchetto ciascuno, e nessuno di questi è
installato di default: `requests` per `http(s)://`, `boto3` per `s3://` e
`gcsfs` per `gs://`.

## Cartelle

Una directory viene scansionata ricorsivamente e ordinata, e ogni file con uno
di questi suffissi diventa un'immagine: `.jpg`, `.jpeg`, `.png`, `.gif`,
`.webp`, `.bmp`, `.tiff`, `.tif`. Tutto il resto nella cartella viene saltato.
Una cartella vuota restituisce una lista vuota invece di sollevare un errore.

Cartelle e liste sono le due sorgenti che accettano `batch`, che esegue un
forward pass impilato per ogni blocco sulle famiglie che lo supportano. Vedi
[Prestazioni dell'inferenza](/docs/predict/performance).

## File video

<code-tabs name="video" />

Un percorso conta come video quando il suo suffisso è uno tra `.asf`, `.avi`,
`.gif`, `.m4v`, `.mkv`, `.mov`, `.mp4`, `.mpeg`, `.mpg`, `.ts`, `.wmv`,
`.webm`.

`.gif` compare in entrambe le liste. Un percorso `.gif` passato direttamente a
`predict` viene aperto come video, perché il controllo sul video viene eseguito
per primo; un `.gif` che si trova dentro una cartella scansionata viene caricato
come immagine statica.

`vid_stride` elabora un frame ogni N e vale `1` di default. Senza `stream=True`
l'intero video viene decodificato in una lista, e qualsiasi valore superiore a
500 frame dopo lo stride genera un avviso che suggerisce `stream=True`.

Ogni `Results` proveniente da un video porta con sé `frame_idx`.

## Webcam, stream di rete e YouTube

<code-tabs name="live" />

Le sorgenti live sono illimitate, quindi richiedono `stream=True`. Senza,
`predict` solleva `ValueError` invece di provare a raccogliere una lista
infinita.

I frame vengono letti su un thread in background, uno per cattura. Di default la
coda tiene solo il frame più recente, così un modello più lento della telecamera
salta dei frame invece di restare indietro. `stream_buffer=True` conserva ogni
frame catturato, il che li preserva al costo di una latenza crescente.

Un indice di webcam è un `int` o una stringa di cifre. Su Windows la cattura
viene aperta prima attraverso il backend DirectShow e ripiega sul backend
predefinito se questo fallisce.

Gli URL delle pagine YouTube vengono risolti in un URL multimediale diretto
senza scaricare il video, e questo richiede `yt-dlp`:

```bash
pip install "libreyolo[stream]"
```

Le etichette degli stream vengono oscurate prima di essere registrate nei log o
usate come nomi di file. Un URL che porta con sé delle credenziali compare come
`user:***@host`, e le query string vengono rimosse dalle etichette degli stream
diretti perché è lì che stanno gli URL firmati e i bearer token. L'id di un
video YouTube viene mantenuto, dato che non è una credenziale.

## Più telecamere insieme

<code-tabs name="streams" />

Un file `.streams` contiene una sorgente per riga. Le righe vuote e quelle che
iniziano con `#` vengono ignorate. Ogni riga rimanente deve essere a sua volta
un indice di webcam, uno stream di rete, un URL YouTube o il percorso di un file
video; qualsiasi altra cosa solleva `ValueError` indicando il numero di riga. Un
file vuoto solleva un errore invece di partire senza telecamere.

Una lista o tupla di sorgenti live fa la stessa cosa senza bisogno di un file.

Ogni cattura ottiene il proprio thread, e i frame di tutte quante vengono
multiplexati in un unico generatore. Ogni passaggio interroga ciascuno stream
attivo e restituisce quello che è pronto, così una telecamera lenta non blocca
una veloce, e i frame di telecamere diverse si alternano. Uno stream che
termina esce dalla rotazione mentre gli altri continuano.

## Cattura dello schermo

<code-tabs name="screen" />

Una sorgente schermo è la parola `screen` seguita da zero, uno, quattro o cinque
interi. Qualsiasi altro conteggio solleva `ValueError`.

| Forma | Cattura |
|---|---|
| `"screen"` | Ogni monitor, uniti |
| `"screen 1"` | Il monitor 1 |
| `"screen 100 200 512 256"` | Un riquadro sul desktop unificato |
| `"screen 1 100 200 512 256"` | Un riquadro sul monitor 1 |

Le coordinate del riquadro sono `left top width height`, relative all'angolo in
alto a sinistra del monitor scelto. Una sorgente schermo dichiara il proprio
frame rate come 30 diviso `vid_stride`, che è la velocità con cui viene scritto
un video salvato. La cattura richiede il pacchetto `mss`:

```bash
pip install mss
```

Senza `stream=True`, una sorgente schermo cattura un frame e restituisce un
singolo `Results`, che è l'equivalente da screenshot di una predizione su un
file immagine. Con `stream=True` cattura finché il ciclo non viene interrotto.

## Che cosa restituisce predict

La forma del valore restituito dipende dalla sorgente e da `stream`.

| Sorgente | `stream=False` | `stream=True` |
|---|---|---|
| Immagine singola | Un `Results` | Generatore di un `Results` |
| Lista di immagini | Lista di `Results` | Generatore |
| Cartella | Lista di `Results` | Generatore |
| File video | Lista di `Results` | Generatore |
| Schermo | Un `Results` | Generatore, illimitato |
| Webcam, stream di rete, `.streams` | `ValueError` | Generatore, illimitato |

Una singola immagine restituisce l'oggetto `Results` stesso. Indicizzarlo
seleziona un rilevamento, non un'immagine, quindi `result[0]` su una predizione
a immagine singola è il primo box e non la prima figura. Per sapere cosa
contengono quegli oggetti, vedi
[Lavorare con i risultati](/docs/predict/results).

## Dove scrive save

`save=True` scrive l'output annotato dentro una directory di run invece di
restituirlo.

Le immagini finiscono in `runs/detect/predict`, `runs/detect/predict2` e così
via, con incremento automatico, mantenendo il nome del file sorgente. Ogni
immagine di uno stesso processo finisce nella stessa directory, quindi due
cartelle di input che contengono lo stesso nome di file si sovrascrivono a
vicenda. Le immagini in memoria non hanno un nome di file da riutilizzare e
vengono numerate `image0`, `image1` e così via.

Le sorgenti video e live vengono scritte come un unico `.mp4` che prende il nome
dalla sorgente.

`output_path` sovrascrive la directory. Un percorso con un suffisso viene
trattato come un file, un percorso senza suffisso come una directory.
`output_file_format` seleziona la codifica delle immagini statiche e accetta
`jpg`, `png` o `webp`.

Dopo un salvataggio, il percorso scritto viene anche allegato al risultato come
`result.saved_path`.
