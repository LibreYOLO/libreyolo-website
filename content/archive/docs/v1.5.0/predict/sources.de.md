---
title: Vorhersagequellen
seo_title: Vorhersagequellen in LibreYOLO
description: >-
  Alle von predict akzeptierten Quellen: Bilder, Ordner, URLs, Videodateien,
  Webcams, RTSP, YouTube, Bildschirmaufnahmen, Bildlisten und .streams-Dateien.
lead: >-
  Das Argument source wird klassifiziert, bevor etwas geöffnet wird. Ein Aufruf
  verarbeitet daher ein JPEG, einen Ordner, eine MP4-Datei, einen Webcam-Index,
  eine RTSP-URL, einen Bildschirmbereich oder eine Liste von Kameras.
keywords:
  - YOLO Video Inferenz Python
  - RTSP Objekterkennung
  - Webcam Objekterkennung Python
  - Vorhersage für Bilderordner
  - Bildschirmaufnahme Objekterkennung
  - mehrere RTSP Streams
  - streams Datei
  - YouTube Inferenz
  - vid_stride
  - stream=True
last_verified: 1.5.0
verification: >-
  Quellenklassifizierung aus libreyolo/utils/source.py gelesen (classify_source,
  SourceKind, StreamSource, MultiStreamSource). Akzeptierte Bildtypen und
  Verzeichnisendungen aus libreyolo/utils/image_loader.py. Videoendungen und
  Speicherpfade aus libreyolo/utils/video.py. Bildschirmsyntax aus
  libreyolo/utils/screen.py. Rückgabeformen und Argumentstandardwerte aus
  InferenceRunner.__call__ in libreyolo/models/base/inference.py.
snippets:
  images:
    - label: Ein Bild
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")


        # Eine einzelne Bildquelle gibt ein Results-Objekt und keine Liste
        zurück.

        result = model(SAMPLE_IMAGE)

        print(len(result.boxes), "detections")
    - label: Bilder im Arbeitsspeicher
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
    - label: Ein Ordner
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


        # Ein Ordner gibt eine nach Pfad sortierte Liste mit einem
        Results-Objekt je Bild zurück.

        results = model(str(folder))

        print(len(results), "images")
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  video:
    - label: Eine Videodatei (verwende einen eigenen Clip)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Ersetze clip.mp4 durch eine lokale Videodatei.
        for result in model("clip.mp4", stream=True):
            print(result.frame_idx, len(result.boxes))
    - label: Jeden dritten Frame auf den Datenträger schreiben
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("clip.mp4", stream=True, vid_stride=3, save=True):
            pass
  live:
    - label: Webcam (angeschlossene Kamera erforderlich)
      language: python
      code: >
        import itertools


        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Webcam-Index 0. Live-Quellen enden nie, daher wird die Schleife
        begrenzt.

        for result in itertools.islice(model(0, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
    - label: RTSP (erreichbare Kamera-URL erforderlich)
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        source = "rtsp://user:password@192.168.1.64:554/Streaming/Channels/101"

        for result in itertools.islice(model(source, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  streams:
    - label: Eine .streams-Datei (verwende eigene Kameras)
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
    - label: Eine Liste von Kameras
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        cameras = [0, "rtsp://192.168.1.64:554/Streaming/Channels/101"]

        for result in itertools.islice(model(cameras, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  screen:
    - label: Ein Screenshot (mss und Desktop-Sitzung erforderlich)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Ohne stream=True wird ein einzelner Frame aufgenommen.
        result = model("screen")
        print(len(result.boxes), "detections")
    - label: Einen Bereich eines Monitors fortlaufend erfassen
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

## Klassifizierung einer Quelle

`classify_source` untersucht den Wert in der folgenden Reihenfolge, bevor etwas geöffnet oder heruntergeladen wird. Die erste passende Regel gewinnt.

| Quelle | Interpretation |
|---|---|
| `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` | Bildschirmaufnahme |
| Ein nicht negativer `int` oder eine Ziffernfolge, für die keine gleichnamige Datei existiert | Webcam |
| Eine URL mit `rtsp://`, `rtmp://`, `tcp://` oder `udp://` | Netzwerkstream |
| Eine `http(s)://`-URL, deren Pfad auf `.m3u8` endet | Netzwerkstream |
| Eine YouTube-Seiten-URL | Netzwerkstream |
| Eine Liste oder ein Tupel, deren Einträge ausschließlich Live-Quellen oder Videos sind | Mehrere Live-Streams |
| Jede andere Liste oder jedes andere Tupel | Batch aus Bildern |
| Ein Pfad mit der Endung `.streams` | Mehrere Live-Streams |
| Ein Pfad mit einer Videoendung | Videodatei |
| Ein vorhandenes Verzeichnis | Bilderordner |
| Alles andere | Einzelbild |

Eine Liste, die Live-Quellen mit Bildern mischt, löst `TypeError` aus. Ein negativer Webcam-Index löst `ValueError` aus.

Der Klassifikator greift nie auf das Netzwerk zu. Eine falsch geschriebene URL fällt daher erst beim Öffnen der Aufnahme und nicht beim Aufruf von `predict` auf.

## Bilder

<code-tabs name="images" />

Eine einzelne Bildquelle akzeptiert sieben Typen.

| Typ | Interpretation |
|---|---|
| `str` oder `pathlib.Path` | Lokale Datei, `http(s)://`, `s3://` oder `gs://` |
| `PIL.Image.Image` | Wird in RGB umgewandelt |
| `numpy.ndarray` | 2D-Graustufen oder 3D-HWC beziehungsweise CHW; ein 4D-Array verwendet sein erstes Bild |
| `torch.Tensor` | CHW oder NCHW, als RGB gelesen; ein Batch-Tensor verwendet sein erstes Bild |
| `bytes` | Codierte Bilddaten |
| `io.BytesIO` | Codierte Bilddaten |

Vor der Vorverarbeitung wird alles in RGB umgewandelt. Nur bei NumPy-Arrays ist die Kanalreihenfolge mehrdeutig, daher steuert `color_format` sie. `"auto"` ist der Standard und lässt das Array unverändert. `"bgr"` kehrt die Kanäle um, wie es ein mit OpenCV gelesener Frame benötigt.

Gleitkomma-Arrays werden anhand ihres eigenen Wertebereichs skaliert. Werte bis einschließlich `1.0` werden mit 255 multipliziert, höhere Werte auf `[0, 255]` begrenzt. Bei einem RGBA-Array wird der Alphakanal entfernt.

Entfernte Pfade benötigen jeweils ein Paket, von denen keines standardmäßig installiert ist: `requests` für `http(s)://`, `boto3` für `s3://` und `gcsfs` für `gs://`.

## Ordner

Ein Verzeichnis wird rekursiv durchsucht und sortiert. Jede Datei mit einer der folgenden Endungen wird als Bild behandelt: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`, `.tiff`, `.tif`. Alles andere wird übersprungen. Ein leerer Ordner gibt eine leere Liste zurück, statt einen Fehler auszulösen.

Ordner und Listen sind die beiden Quelltypen, die `batch` akzeptieren. Bei unterstützten Familien wird damit je Gruppe ein gestapelter Vorwärtsdurchlauf ausgeführt. Weitere Informationen findest du unter [Inferenzleistung](/docs/predict/performance).

## Videodateien

<code-tabs name="video" />

Ein Pfad gilt als Video, wenn seine Endung `.asf`, `.avi`, `.gif`, `.m4v`, `.mkv`, `.mov`, `.mp4`, `.mpeg`, `.mpg`, `.ts`, `.wmv` oder `.webm` lautet.

`.gif` steht in beiden Listen. Ein direkt an `predict` übergebener `.gif`-Pfad wird als Video geöffnet, weil die Videoprüfung zuerst erfolgt. Eine `.gif`-Datei in einem durchsuchten Ordner wird dagegen als Standbild geladen.

`vid_stride` verarbeitet jeden N-ten Frame und ist standardmäßig `1`. Ohne `stream=True` wird das gesamte Video in eine Liste decodiert. Bei mehr als 500 Frames nach Anwendung des Stride erscheint eine Warnung mit dem Vorschlag, `stream=True` zu verwenden.

Jedes `Results`-Objekt eines Videos enthält `frame_idx`.

## Webcams, Netzwerkstreams und YouTube

<code-tabs name="live" />

Live-Quellen sind unbegrenzt und benötigen deshalb `stream=True`. Ohne dieses Argument löst `predict` `ValueError` aus, statt eine endlose Liste aufzubauen.

Frames werden in einem Hintergrundthread gelesen, einer je Aufnahme. Standardmäßig enthält die Warteschlange nur den neuesten Frame. Ist ein Modell langsamer als die Kamera, überspringt es dadurch Frames, statt immer weiter zurückzufallen. `stream_buffer=True` behält jeden aufgenommenen Frame. Das erhält alle Frames auf Kosten zunehmender Latenz.

Ein Webcam-Index ist ein `int` oder eine Ziffernfolge. Unter Windows wird die Aufnahme zuerst über das DirectShow-Backend geöffnet. Schlägt dies fehl, wird auf das Standard-Backend zurückgegriffen.

YouTube-Seiten-URLs werden ohne Download des Videos in eine direkte Medien-URL aufgelöst. Dafür ist `yt-dlp` erforderlich:

```bash
pip install "libreyolo[stream]"
```

Stream-Labels werden vor der Protokollierung oder Verwendung als Dateiname bereinigt. Eine URL mit Zugangsdaten erscheint als `user:***@host`. Abfragezeichenfolgen werden aus direkten Stream-Labels entfernt, da sie häufig signierte URLs und Bearer-Tokens enthalten. Eine YouTube-Video-ID bleibt erhalten, da sie keine Zugangsdaten darstellt.

## Mehrere Kameras gleichzeitig

<code-tabs name="streams" />

Eine `.streams`-Datei enthält eine Quelle pro Zeile. Leerzeilen und mit `#` beginnende Zeilen werden ignoriert. Jede übrige Zeile muss selbst ein Webcam-Index, Netzwerkstream, eine YouTube-URL oder ein Videodateipfad sein. Alles andere löst `ValueError` unter Angabe der Zeilennummer aus. Eine leere Datei löst einen Fehler aus, statt ohne Kameras zu starten.

Eine Liste oder ein Tupel aus Live-Quellen leistet dasselbe ohne Datei.

Jede Aufnahme erhält einen eigenen Thread. Die Frames aller Aufnahmen werden in einem Generator zusammengeführt. Jeder Durchgang fragt jeden aktiven Stream ab und gibt alle bereiten Frames aus. Eine langsame Kamera hält dadurch keine schnelle auf und Frames verschiedener Kameras werden ineinander verschachtelt. Ein beendeter Stream fällt aus der Rotation, während die übrigen fortgesetzt werden.

## Bildschirmaufnahme

<code-tabs name="screen" />

Eine Bildschirmquelle besteht aus dem Wort `screen`, gefolgt von null, einer, vier oder fünf Ganzzahlen. Jede andere Anzahl löst `ValueError` aus.

| Form | Aufnahme |
|---|---|
| `"screen"` | Alle Monitore zusammengeführt |
| `"screen 1"` | Monitor 1 |
| `"screen 100 200 512 256"` | Ein Bereich auf dem zusammengeführten Desktop |
| `"screen 1 100 200 512 256"` | Ein Bereich auf Monitor 1 |

Die Bereichskoordinaten lauten `left top width height` relativ zur oberen linken Ecke des gewählten Monitors. Eine Bildschirmquelle meldet als Framerate 30 geteilt durch `vid_stride`. Mit dieser Rate wird ein gespeichertes Video geschrieben. Für die Aufnahme ist das Paket `mss` erforderlich:

```bash
pip install mss
```

Ohne `stream=True` nimmt eine Bildschirmquelle einen Frame auf und gibt ein einzelnes `Results`-Objekt zurück. Dies entspricht der Vorhersage für eine Bilddatei. Mit `stream=True` wird aufgenommen, bis die Schleife abgebrochen wird.

## Rückgabewert von predict

Die Form des Rückgabewerts hängt von der Quelle und von `stream` ab.

| Quelle | `stream=False` | `stream=True` |
|---|---|---|
| Einzelbild | Ein `Results`-Objekt | Generator eines `Results`-Objekts |
| Bilderliste | Liste von `Results` | Generator |
| Ordner | Liste von `Results` | Generator |
| Videodatei | Liste von `Results` | Generator |
| Bildschirm | Ein `Results`-Objekt | Unbegrenzter Generator |
| Webcam, Netzwerkstream, `.streams` | `ValueError` | Unbegrenzter Generator |

Ein Einzelbild gibt das `Results`-Objekt selbst zurück. Seine Indizierung wählt eine Erkennung und kein Bild aus. `result[0]` einer Einzelbildvorhersage ist daher die erste Box und nicht das erste Bild. Informationen zum Inhalt dieser Objekte findest du unter [Arbeiten mit Ergebnissen](/docs/predict/results).

## Speicherort von save

`save=True` schreibt die annotierte Ausgabe in ein Laufverzeichnis, statt sie zurückzugeben.

Bilder werden unter einem automatisch hochgezählten Pfad wie `runs/detect/predict`, `runs/detect/predict2` und so weiter gespeichert. Der Quelldateiname bleibt erhalten. Alle Bilder eines Prozesses landen im selben Verzeichnis. Enthalten zwei Eingabeordner denselben Dateinamen, überschreiben sie sich deshalb gegenseitig. Bilder im Arbeitsspeicher besitzen keinen wiederverwendbaren Namen und werden als `image0`, `image1` und so weiter nummeriert.

Video- und Live-Quellen werden als einzelne `.mp4`-Datei mit dem Namen der Quelle geschrieben.

`output_path` überschreibt das Verzeichnis. Ein Pfad mit Dateiendung wird als Datei behandelt, ein Pfad ohne Endung als Verzeichnis. `output_file_format` wählt die Codierung für Standbilder und akzeptiert `jpg`, `png` oder `webp`.

Nach dem Speichern wird der geschriebene Pfad außerdem als `result.saved_path` an das Ergebnis angehängt.
