---
title: DEIM
families:
  - deim
seo_title: DEIM i DEIMv2 w LibreYOLO
description: >-
  Użyj DEIM i DEIMv2 w LibreYOLO do wykrywania obiektów. Instaluj, przewiduj,
  trenuj, waliduj i eksportuj, od rozmiaru pół miliona parametrów wzwyż.
lead: >-
  Transformator detekcji trenowany z gęstym dopasowaniem jeden-do-jednego, który
  zbiega w znacznie mniejszej liczbie epok niż przepisy DETR, na których się
  opiera. LibreYOLO zawiera dwie jego wersje, rozróżniane przez checkpoint,
  który wczytujesz.
keywords:
  - DEIM
  - DEIMv2
  - DINOv3
  - transformer wykrywania
  - DETR
  - detekcja obiektów
  - wykrywanie w czasie rzeczywistym
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDEIMn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDEIMn.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Wideo
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Wersja jest częścią nazwy pliku, a fabryka kieruje na

        # checkpoint, więc oba ładują się w ten sam sposób.

        model = LibreYOLO("LibreDEIMv2pico.pt")


        # Dowolne źródło akceptowane przez bibliotekę: plik, folder, URL, indeks
        kamery internetowej,

        # Strumień RTSP lub lista .streams

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDEIMn.pt")


        # coco128.yaml pobiera próbkę 128 obrazów przy pierwszym użyciu. Punkt
        `data`

        # na własnym zbiorze danych YAML do rzeczywistego uruchomienia.

        model.train(data="coco128.yaml", epochs=50, batch=8, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 batch=8 lr0=1e-4
    - label: DEIMv2
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Jeżeli nie zostaną ustawione, epochs, batch, imgsz i lr0 pochodzą z
        wydanej wersji

        # przepis na rozmiar, który został załadowany.

        model = LibreYOLO("LibreDEIMv2pico.pt")

        model.train(data="coco128.yaml", epochs=50)
    - label: LoRA
      language: python
      code: |
        # Wymaga dodatkowego lora: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # val() zwraca zwykły słownik, a nie obiekt
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDEIMn.pt data=coco128.yaml
    - label: Przeciwko COCO
      language: bash
      code: |
        # coco-val-only.yaml pobiera 5000 obrazów val2017 i pomija
        # zbiór treningowy. Zawiera osadzony skrypt pobierania, więc wymaga
        # wyraźna zgoda, chyba że zbiór danych jest już lokalny.
        libreyolo val model=LibreDEIMn.pt data=coco-val-only.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        # Wymaga dodatkowego onnx: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDEIMn.pt format=onnx
    - label: Użyj wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka kieruje na podstawie sufiksu pliku, więc eksportowany artefakt
        się ładuje

        # jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreDEIMn.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 6edaac5f05abaabe
---
## Instalacja

Żadna z wersji nie potrzebuje opcjonalnego dodatku. Wszystko, co importują, jest w podstawowej instalacji.

```bash
pip install libreyolo
```

Dostrajanie adaptera za pomocą `lora=True` jest wyjątkiem i wymaga dodatkowego `lora`.

```bash
pip install "libreyolo[lora]"
```

## Predykcja

Wagi są pobierane z Hugging Face przy pierwszym użyciu i są przechowywane w pamięci podręcznej lokalnie.

<code-tabs name="predict" />

Zwrócony obiekt `Results` jest tym, który zwraca każda rodzina, więc wymiana na inny detektor to zmiana w jednej linii. `conf` i `max_det` filtrują dekodowanie top-k dla zapytań i klas; nie ma kroku NMS do regulacji, a `iou` jest akceptowany, ale nieużywany. Zobacz [prediction](/docs/predict) dla źródeł, streaming i obsługi wyników.

## Warianty

Wersja 1 oferuje pięć rozmiarów, wszystkie przy tym samym rozmiarze wejściowym. Wersja 2 zachowuje te pięć nazw i dodaje trzy mniejsze, `atto`, `femto` i `pico`, z których dwa pierwsze są natywne przy mniejszym rozmiarze wejściowym niż pozostałe. W obu wersjach istnieje więc pięć kodów rozmiarów i oznaczają różne modele; wersja jest zapisana w nazwie pliku checkpointu.

<benchmark-table task="detect" />

<va-embed />

Wersja 1 zachowuje architekturę D-FINE i zastępuje jej cel klasyfikacji funkcją
straty uwzględniającą możliwość dopasowania z gęstego przepisu jeden-do-jednego.
Obie rodziny współdzielą więc niemal wszystkie klucze słownika stanu, a odróżniają
je metadane checkpointu. Wersja 2 zachowuje ten kontrakt trenowania i łączy
różne backbone: HGNetv2 w rozmiarach mniejszych niż `s` oraz transformer wizyjny
DINOv3 z adapterem dostrajania przestrzennego w rozmiarze `s` i większych.
Ten backbone powoduje objęcie czterech checkpointów drugą licencją, dlatego
przed ich wdrożeniem należy przeczytać sekcję
[Licencjonowanie](#licensing).

## Trenowanie

Trenowanie zaczyna się od opublikowanego checkpointu. `pretrained` nigdy nie dociera do trenera: wersja 1 ostrzega, że klucz jest nieznany i go ignoruje, wersja 2 go usuwa. Żadna z nich nie daje losowo zainicjowanego modelu.

<code-tabs name="train" />

Dla wersji 1 należy samodzielnie podać `lr0`. Sygnatura metody `train()`
w Pythonie ma wartość domyślną `4e-4`, pochodzącą z opublikowanego przepisu
COCO. Konfiguracja trenowania rodziny ustawia natomiast `1e-4` jako domyślną
wartość dostrajania i właśnie tę niższą wartość wybiera CLI, gdy argumentu nie
podano. Konfiguracja rejestruje wynik pomiaru uzasadniającego ten wybór: przy
rozmiarach batcha używanych podczas dostrajania na małych zbiorach danych
współczynnik z przepisu COCO wyraźnie pogarszał transfer.

Wersja 2 sama rozwiązuje te ustawienia domyślne. Pozostawienie `epochs`, `batch`, `imgsz` i `lr0` nieustawionych powoduje, że program odczytuje każdy z nich z opublikowanego przepisu dla wczytanego rozmiaru, dzięki czemu małe rozmiary trenują w swojej własnej rozdzielczości wejściowej bez konieczności jej podawania, a wartość, którą podasz, nadpisuje przepis. `imgsz` to argument, który ogranicza: musi być dodatnią wielokrotnością 32, a wersja 2 zgłasza błąd przed rozpoczęciem działania w przeciwnym razie.

Zobacz [trenowanie](/docs/train) dotyczące zbiorów danych, augmentacji, multi-GPU i loggerów.

## Walidacja

`val()` zwraca słownik kluczy `metrics/` obejmujących precyzję, czułość, mAP 50 oraz mAP 50-95, mierzone względem dowolnego zbioru danych w formacie, na którym trenowałeś.

<code-tabs name="val" />

Wiersze w powyższej tabeli benchmarkowej pochodzą z zestawu testowego LibreYOLO; nota pod tą tabelą zapisuje, który zbiór danych je wygenerował i łączy z zapisami uruchomień.

## Eksport

<export-matrix />

Macierz obejmuje obie wersje jako jedną stronę: tam, gdzie nie zgadzają się co do formatu, komórka pokazuje słabszą z obu, więc nic tutaj nie jest przesadnie promowane dla którejkolwiek wersji, którą załadujesz.

Eksportowany artefakt ładuje się ponownie przez `LibreYOLO()` na swoim rozszerzeniu pliku, więc plik `.onnx` lub `.engine` działa jak checkpoint i zwraca ten sam `Results`.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box> Cztery rozmiary DEIMv2 od S wzwyż korzystają z backbone DINOv3,
dlatego ich repozytoria wag są objęte zarówno licencją Apache-2.0, jak i licencją
DINOv3 firmy Meta. LibreYOLO udostępnia kod źródłowy backbone DINOv3 na tych
samych warunkach. Pozostała część rodziny, w tym wszystkie rozmiary DEIMv2
mniejsze niż S, jest objęta wyłącznie licencją Apache-2.0. </provenance-box>

## Cytowanie

<citation-block />

DEIMv2 to odrębny artykuł i ma własny blok cytowań na [github.com/Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2#5-citation); cytuj go, jeśli używałeś checkpointu wersji 2.
