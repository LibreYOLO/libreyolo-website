---
title: Źródła predykcji
seo_title: Źródła predykcji w LibreYOLO
description: >-
  Wszystkie źródła przyjmowane przez predict: obrazy, foldery, adresy URL, pliki
  wideo, kamery internetowe, RTSP, YouTube, przechwytywanie ekranu, listy
  obrazów i pliki .streams.
lead: >-
  Argument source jest klasyfikowany przed otwarciem czegokolwiek, dlatego jedno
  wywołanie obsługuje JPEG, folder, MP4, indeks kamery internetowej, adres URL
  RTSP, obszar ekranu albo listę kamer.
keywords:
  - YOLO wnioskowanie wideo Python
  - RTSP
  - detekcja obiektów kamera internetowa Python
  - predykcja na folderze obrazów
  - detekcja obiektów przechwytywanie ekranu
  - wiele strumieni RTSP
  - plik streams
  - wnioskowanie YouTube
  - vid_stride
  - stream=True
last_verified: 1.5.0
verification: >-
  Klasyfikację źródeł odczytano z libreyolo/utils/source.py (classify_source,
  SourceKind, StreamSource, MultiStreamSource). Akceptowane typy obrazów i
  rozszerzenia katalogów pochodzą z libreyolo/utils/image_loader.py.
  Rozszerzenia wideo i ścieżki zapisu pochodzą z libreyolo/utils/video.py.
  Składnia ekranu pochodzi z libreyolo/utils/screen.py. Kształty zwracanych
  wartości i wartości domyślne argumentów pochodzą z InferenceRunner.__call__ w
  libreyolo/models/base/inference.py.
snippets:
  images:
    - label: Jeden obraz
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # Pojedyncze źródło obrazu zwraca jeden Results, a nie listę.
        result = model(SAMPLE_IMAGE)
        print(len(result.boxes), "detections")
    - label: Obrazy w pamięci
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
    - label: Folder
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


        # Folder zwraca listę, jeden Results na obraz, posortowaną według
        ścieżki.

        results = model(str(folder))

        print(len(results), "images")
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  video:
    - label: Plik wideo (podaj własny klip)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Zastąp clip.mp4 plikiem wideo na dysku.
        for result in model("clip.mp4", stream=True):
            print(result.frame_idx, len(result.boxes))
    - label: Co trzecia klatka zapisywana na dysku
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("clip.mp4", stream=True, vid_stride=3, save=True):
            pass
  live:
    - label: Kamera internetowa (wymaga podłączonej kamery)
      language: python
      code: >
        import itertools


        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Indeks kamery internetowej 0. Źródła na żywo nigdy się nie kończą,
        więc ogranicz pętlę.

        for result in itertools.islice(model(0, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
    - label: RTSP (wymaga osiągalnego adresu URL kamery)
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        source = "rtsp://user:password@192.168.1.64:554/Streaming/Channels/101"

        for result in itertools.islice(model(source, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  streams:
    - label: Plik .streams (podaj własne kamery)
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
    - label: Lista kamer
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        cameras = [0, "rtsp://192.168.1.64:554/Streaming/Channels/101"]

        for result in itertools.islice(model(cameras, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  screen:
    - label: Jeden zrzut ekranu (wymaga mss i sesji pulpitu)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Bez stream=True pobierana jest jedna klatka.
        result = model("screen")
        print(len(result.boxes), "detections")
    - label: Ciągłe przechwytywanie obszaru jednego monitora
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

## Sposób klasyfikowania źródła

`classify_source` sprawdza wartość przed otwarciem lub pobraniem czegokolwiek,
w poniższej kolejności. Wygrywa pierwsza pasująca reguła.

| Źródło | Odczytywane jako |
|---|---|
| `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` | Przechwytywanie ekranu |
| Nieujemne `int` albo ciąg cyfr, gdy nie istnieje plik o tej nazwie | Kamera internetowa |
| Adres URL `rtsp://`, `rtmp://`, `tcp://` albo `udp://` | Strumień sieciowy |
| Adres URL `http(s)://`, którego ścieżka kończy się na `.m3u8` | Strumień sieciowy |
| Adres URL strony YouTube | Strumień sieciowy |
| Lista albo krotka, której wszystkie wpisy są źródłami na żywo lub wideo | Kilka strumieni na żywo |
| Dowolna inna lista albo krotka | Partia obrazów |
| Ścieżka kończąca się na `.streams` | Kilka strumieni na żywo |
| Ścieżka z rozszerzeniem wideo | Plik wideo |
| Istniejący katalog | Folder obrazów |
| Wszystko inne | Pojedynczy obraz |

Lista mieszająca źródła na żywo z obrazami powoduje `TypeError`. Ujemny indeks
kamery internetowej powoduje `ValueError`.

Klasyfikator nigdy nie używa sieci, dlatego literówka w adresie URL ujawnia się
dopiero przy otwieraniu przechwytywania, a nie podczas wywołania `predict`.

## Obrazy

<code-tabs name="images" />

Pojedyncze źródło obrazu przyjmuje siedem typów.

| Typ | Odczytywane jako |
|---|---|
| `str` albo `pathlib.Path` | Plik lokalny, `http(s)://`, `s3://` albo `gs://` |
| `PIL.Image.Image` | Konwertowany do RGB |
| `numpy.ndarray` | Obraz 2D w skali szarości albo 3D HWC lub CHW; tablica 4D używa pierwszego obrazu |
| `torch.Tensor` | CHW albo NCHW, odczytywany jako RGB; tensor z partią używa pierwszego obrazu |
| `bytes` | Zakodowane dane obrazu |
| `io.BytesIO` | Zakodowane dane obrazu |

Przed przetwarzaniem wstępnym wszystko jest konwertowane do RGB. Tablice NumPy
są jedynym przypadkiem niejednoznacznej kolejności kanałów, dlatego steruje nią
`color_format`: `"auto"` (wartość domyślna) pozostawia tablicę bez zmian, a
`"bgr"` odwraca kanały, co jest potrzebne dla klatki odczytanej przez OpenCV.

Tablice zmiennoprzecinkowe są skalowane według własnego zakresu: wartości nie
większe niż `1.0` są mnożone przez 255, a wyższe przycinane do `[0, 255]`.
W tablicy RGBA kanał alfa jest usuwany.

Ścieżki zdalne wymagają po jednym pakiecie, z których żaden nie jest instalowany
domyślnie: `requests` dla `http(s)://`, `boto3` dla `s3://` oraz `gcsfs` dla `gs://`.

## Foldery

Katalog jest skanowany rekurencyjnie i sortowany, a każdy plik z jednym z
następujących rozszerzeń staje się obrazem: `.jpg`, `.jpeg`, `.png`, `.gif`,
`.webp`, `.bmp`, `.tiff`, `.tif`. Wszystko inne w folderze jest pomijane. Pusty
folder zwraca pustą listę zamiast zgłaszać błąd.

Foldery i listy to dwa źródła przyjmujące `batch`, które wykonuje jeden przebieg
w przód na ułożonym stosie dla każdego fragmentu w obsługujących go rodzinach.
Zobacz [wydajność wnioskowania](/docs/predict/performance).

## Pliki wideo

<code-tabs name="video" />

Ścieżka jest uznawana za wideo, gdy jej rozszerzenie jest jednym z: `.asf`, `.avi`,
`.gif`, `.m4v`, `.mkv`, `.mov`, `.mp4`, `.mpeg`, `.mpg`, `.ts`, `.wmv`, `.webm`.

`.gif` występuje na obu listach. Ścieżka `.gif` przekazana bezpośrednio do
`predict` jest otwierana jako wideo, ponieważ kontrola wideo odbywa się jako
pierwsza. Plik `.gif` wewnątrz skanowanego folderu jest ładowany jako obraz statyczny.

`vid_stride` przetwarza co N-tą klatkę i ma domyślnie wartość `1`. Bez
`stream=True` całe wideo jest dekodowane do listy, a ponad 500 klatek po
uwzględnieniu kroku powoduje ostrzeżenie sugerujące `stream=True`.

Każdy obiekt `Results` z wideo zawiera `frame_idx`.

## Kamery internetowe, strumienie sieciowe i YouTube

<code-tabs name="live" />

Źródła na żywo są nieograniczone, dlatego wymagają `stream=True`. Bez niego
`predict` zgłasza `ValueError` zamiast próbować zebrać nieskończoną listę.

Klatki są odczytywane w wątku działającym w tle, po jednym na przechwytywanie.
Domyślnie kolejka zawiera tylko najnowszą klatkę, więc model wolniejszy od kamery
pomija klatki zamiast zwiększać opóźnienie. `stream_buffer=True` zachowuje każdą
przechwyconą klatkę, co odbywa się kosztem rosnącego opóźnienia.

Indeks kamery internetowej jest typu `int` albo ciągiem cyfr. W systemie Windows
przechwytywanie jest najpierw otwierane przez backend DirectShow, a po niepowodzeniu
wraca do backendu domyślnego.

Adresy URL stron YouTube są rozwiązywane do bezpośredniego adresu URL multimediów
bez pobierania wideo, co wymaga `yt-dlp`:

```bash
pip install "libreyolo[stream]"
```

Etykiety strumieni są redagowane przed zapisaniem w logach lub użyciem jako
nazwy plików. Adres URL z danymi uwierzytelniającymi pojawia się jako
`user:***@host`, a ciągi zapytania są usuwane z bezpośrednich etykiet strumieni,
ponieważ znajdują się w nich podpisane adresy URL i tokeny Bearer. Identyfikator
wideo YouTube pozostaje, ponieważ nie jest daną uwierzytelniającą.

## Kilka kamer jednocześnie

<code-tabs name="streams" />

Plik `.streams` zawiera jedno źródło w wierszu. Puste wiersze i wiersze zaczynające
się od `#` są ignorowane. Każdy pozostały wiersz musi być indeksem kamery
internetowej, strumieniem sieciowym, adresem URL YouTube albo ścieżką pliku wideo.
Wszystko inne powoduje `ValueError` z numerem wiersza. Pusty plik zgłasza błąd
zamiast uruchamiać się bez kamer.

Lista albo krotka źródeł na żywo robi to samo bez pliku.

Każde przechwytywanie otrzymuje własny wątek, a klatki ze wszystkich są
multipleksowane do jednego generatora. Każdy przebieg sprawdza każdy aktywny
strumień i zwraca wszystko, co jest gotowe, dlatego wolna kamera nie zatrzymuje
szybkiej, a klatki z różnych kamer się przeplatają. Strumień, który się zakończy,
wypada z rotacji, podczas gdy pozostałe działają dalej.

## Przechwytywanie ekranu

<code-tabs name="screen" />

Źródło ekranu to słowo `screen`, po którym występuje zero, jedna, cztery albo
pięć liczb całkowitych. Każda inna liczba powoduje `ValueError`.

| Postać | Przechwytywany obszar |
|---|---|
| `"screen"` | Wszystkie monitory połączone |
| `"screen 1"` | Monitor 1 |
| `"screen 100 200 512 256"` | Prostokąt na połączonym pulpicie |
| `"screen 1 100 200 512 256"` | Prostokąt na monitorze 1 |

Współrzędne prostokąta mają postać `left top width height` względem lewego
górnego rogu wybranego monitora. Źródło ekranu raportuje liczbę klatek na sekundę
jako 30 podzielone przez `vid_stride`, czyli częstotliwość zapisu wynikowego
wideo. Przechwytywanie wymaga pakietu `mss`:

```bash
pip install mss
```

Bez `stream=True` źródło ekranu przechwytuje jedną klatkę i zwraca pojedynczy
obiekt `Results`, co jest odpowiednikiem zrzutu ekranu dla predykcji na pliku
obrazu. Z `stream=True` przechwytuje do przerwania pętli.

## Wartość zwracana przez predict

Kształt zwracanej wartości zależy od źródła i `stream`.

| Źródło | `stream=False` | `stream=True` |
|---|---|---|
| Pojedynczy obraz | Jeden `Results` | Generator jednego `Results` |
| Lista obrazów | Lista `Results` | Generator |
| Folder | Lista `Results` | Generator |
| Plik wideo | Lista `Results` | Generator |
| Ekran | Jeden `Results` | Generator, nieograniczony |
| Kamera internetowa, strumień sieciowy, `.streams` | `ValueError` | Generator, nieograniczony |

Pojedynczy obraz zwraca sam obiekt `Results`. Indeksowanie go wybiera detekcję,
a nie obraz, dlatego `result[0]` w predykcji pojedynczego obrazu jest pierwszą
ramką, a nie pierwszym obrazem. Zawartość tych obiektów opisano w sekcji
[praca z wynikami](/docs/predict/results).

## Miejsce zapisu

`save=True` zapisuje wynik z adnotacjami w katalogu przebiegu zamiast go zwracać.

Obrazy trafiają do automatycznie numerowanych katalogów `runs/detect/predict`,
`runs/detect/predict2` i kolejnych, z zachowaniem nazwy pliku źródłowego. Każdy
obraz w jednym procesie trafia do tego samego katalogu, więc dwa foldery wejściowe
zawierające tę samą nazwę pliku nadpisują się. Obrazy w pamięci nie mają nazwy
do ponownego użycia i są numerowane jako `image0`, `image1` i kolejne.

Wideo i źródła na żywo są zapisywane jako pojedynczy plik `.mp4` nazwany według źródła.

`output_path` nadpisuje katalog. Ścieżka z rozszerzeniem jest traktowana jako
plik, a bez rozszerzenia jako katalog. `output_file_format` wybiera kodowanie
obrazu statycznego i przyjmuje `jpg`, `png` albo `webp`.

Po zapisie zapisana ścieżka jest również dołączana do wyniku jako `result.saved_path`.
