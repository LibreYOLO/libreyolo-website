---
title: RT-DETR
families:
  - rtdetr
seo_title: 'RT-DETR, RT-DETRv2 i RT-DETRv4 w LibreYOLO'
description: >-
  Używaj modeli RT-DETR, RT-DETRv2 i RT-DETRv4 w LibreYOLO do detekcji obiektów,
  a także ramek zorientowanych w RT-DETRv2. Instalacja, predykcja, trenowanie,
  walidacja i eksport z wagami na licencji Apache-2.0.
lead: >-
  Transformer detekcyjny przeznaczony do wnioskowania w czasie rzeczywistym.
  Dekoduje stały zestaw zapytań zamiast gęstej siatki, dlatego nie wykonuje NMS.
  LibreYOLO zawiera trzy jego wersje rozróżniane przez ładowany punkt kontrolny,
  a wersja 2 obsługuje także ramki zorientowane.
keywords:
  - RT-DETR
  - RT-DETRv2
  - RT-DETRv4
  - transformer detekcyjny czasu rzeczywistego
  - DETR
  - detekcja obiektów
  - detekcja zorientowanych ramek ograniczających
  - OBB
  - DOTA
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTDETRr18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRTDETRr18.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Wideo
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Wersja jest częścią nazwy pliku, a fabryka wybiera ścieżkę według

        # punktu kontrolnego, dlatego wszystkie trzy ładują się tak samo.

        model = LibreYOLO("LibreRTDETRv4s.pt")


        # Dowolne źródło akceptowane przez bibliotekę: plik, folder, URL, indeks
        kamery,

        # strumień RTSP albo lista .streams

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
    - label: Ramki zorientowane
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Tylko wersja 2. Przyrostek -obb wybiera zadanie, a punkt kontrolny
        # jest rozpoznawany jako zorientowany na podstawie własnych tensorów,
        # dlatego argument task nie jest potrzebny. Te wagi to DOTA v1.0,
        # 15 klas lotniczych w rozdzielczości 1024 px.
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        result = model("aerial.png", save=True)

        obb = result.obb
        print(obb.xywhr)     # (N, 5): cx, cy, w, h, radiany
        print(obb.xyxyxyxy)  # te same wiersze jako cztery punkty narożne
        print(result.boxes.xyxy)  # otaczające ramki wyrównane do osi
    - label: 'Ramki zorientowane, CLI'
      language: bash
      code: >
        libreyolo predict model=LibreRTDETRv2n-obb.pt source=aerial.png
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRTDETRr18.pt")


        # coco128.yaml pobiera próbkę 128 obrazów przy pierwszym użyciu. Dla

        # rzeczywistego przebiegu ustaw `data` na własny plik YAML zbioru
        danych.

        model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 batch=4 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        # Wymaga dodatku lora: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: Wiele GPU
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")

        # val() zwraca zwykły słownik, a nie obiekt
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTDETRr18.pt data=coco128.yaml
    - label: Walidacja na COCO
      language: bash
      code: |
        # coco-val-only.yaml pobiera 5000 obrazów val2017 i pomija zestaw
        # treningowy. Zawiera osadzony skrypt pobierania, dlatego wymaga
        # jawnego zezwolenia, chyba że zbiór danych jest już dostępny lokalnie.
        libreyolo val model=LibreRTDETRr18.pt data=coco-val-only.yaml \
          allow_download_scripts=True
    - label: Ramki zorientowane
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Walidacja zorientowana dopasowuje za pomocą obróconego IoU, dlatego
        predykcja

        # we właściwym miejscu, lecz pod złym kątem, jest liczona jako
        chybienie.

        model = LibreYOLO("LibreRTDETRv2n-obb.pt")

        metrics = model.val(data="my-obb-dataset.yaml")


        print(metrics["metrics/mAP50-95(OBB)"])

        print(metrics["metrics/mAP50(OBB)"])
  export:
    - label: Python
      language: python
      code: |
        # Wymaga dodatku onnx: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRTDETRr18.pt format=onnx
    - label: Ramki zorientowane
      language: bash
      code: >
        # ONNX i TorchScript są zwalidowanymi formatami docelowymi dla zadania

        # zorientowanego, w FP32, z partią 1 i stałym płótnem 1024 na 1024.

        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024

        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript
        imgsz=1024
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, dlatego

        # artefakt ładuje się jak punkt kontrolny i zwraca ten sam obiekt
        Results.

        model = LibreYOLO("LibreRTDETRr18.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 8022a5a591922a90
---

## Instalacja

RT-DETR nie wymaga opcjonalnych dodatków. Wszystkie importowane przez niego
elementy znajdują się w instalacji podstawowej, a dodatek `rtdetr` jest stabilną
nazwą, która niczego do niej nie dodaje.

```bash
pip install libreyolo
```

Dostrajanie adaptera z `lora=True` jest wyjątkiem i wymaga dodatku `lora`.

```bash
pip install "libreyolo[lora]"
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej pamięci podręcznej.

<code-tabs name="predict" />

Zwracany obiekt `Results` jest taki sam dla każdej rodziny, więc zamiana na
inny detektor wymaga zmiany jednego wiersza. Parametry `conf` i `max_det`
filtrują dekodowanie top-k dla zapytań i klas. Nie ma etapu NMS do dostrojenia,
a `iou` jest przyjmowane, lecz nieużywane. Zorientowany punkt kontrolny natywnie
wypełnia `result.obb`, a także `result.boxes` otaczającymi prostokątami wyrównanymi
do osi. Informacje o źródłach, strumieniowaniu i obsłudze wyników znajdziesz
w sekcji [predykcja](/docs/predict).

## Warianty

Dostępne są trzy wersje i łącznie dwa zadania, a kody rozmiarów nie tworzą jednej
serii. Wersja 1 nazywa rozmiary według backbone, ResNet lub HGNetv2. Wersja 2
ponownie wykorzystuje tylko nazwy ResNet: wersja 1 zawiera już dwa rozmiary
HGNetv2, a wyniki wersji 2 były dla nich na tyle zbliżone, że LibreYOLO nie
publikuje zduplikowanych wag. Wersja 4 używa zwykłej serii literowej, która koliduje
z nazwami HGNetv2 z wersji 1, dlatego sam kod rozmiaru nie identyfikuje modelu.
Wersja jest zapisana w nazwie pliku punktu kontrolnego.

<benchmark-table task="detect" />

<va-embed />

Wersja 2 zachowuje architekturę i układ słownika stanu wersji 1, a zmienia sposób
próbkowania uwagi deformowalnej. Dlatego są rozróżniane na podstawie metadanych
w punkcie kontrolnym, a nie kształtu. Wersja 4 pochodzi z innej linii: ponownie
wykorzystuje architekturę i trenera D-FINE, a jej wagi powstały przez destylację
bazowego modelu wizyjnego DINOv3 z nauczyciela do ucznia HGNetv2. W LibreYOLO
`LibreRTDETRv4` jest podklasą `LibreDFINE` z trwale wyłączoną głowicą masek,
dlatego obsługuje wyłącznie detekcję.

### Ramki zorientowane w wersji 2

Wersja 2 jako jedyna zawiera drugie zadanie. Obsługuje `detect` i `obb`, które
nie korzystają ze wspólnego grafu ani serii rozmiarów. Detekcja używa rozmiarów
ResNet przy 640 px. Detekcja zorientowana korzysta z serii HGNetv2, n, s, m, l
i x, przy 1024 px, a rozmiar wejścia jest ustalany według zadania, nie rodziny.
Punkt kontrolny jest rozpoznawany jako zorientowany na podstawie własnych tensorów,
pięciowspółrzędnych głowic ramek i parametrów próbkowania wersji 2. Dzięki temu
wagi `-obb` ładują się do grafu zorientowanego bez argumentu `task`, a niezgodność
obu wariantów powoduje jawny błąd zamiast cichej reinterpretacji.

Opublikowane pliki mają nazwy od `LibreRTDETRv2n-obb.pt` do
`LibreRTDETRv2x-obb.pt`. Są to oficjalne jedno­skalowe punkty kontrolne DOTA v1.0
przekonwertowane do formatu LibreYOLO, z 15 klasami lotniczymi od samolotu i
statku po port i śmigłowiec. Nazwy klas są zapisane w punkcie kontrolnym. W
przeciwieństwie do detekcji zadanie zorientowane obsługuje wyłącznie wnioskowanie:
działają predykcja, walidacja i eksport, a `train()` na modelu zorientowanym zgłasza
błąd. Śledzenie i augmentacja podczas testowania również nie obsługują ramek
zorientowanych. Sekcja [detekcja zorientowana](/docs/tasks/oriented-detection)
opisuje zadanie, format etykiet i metryki.

## Trenowanie

Trenowanie rozpoczyna się od opublikowanego punktu kontrolnego. Parametr
`pretrained` jest przyjmowany, a następnie odrzucany we wszystkich trzech wersjach,
więc `pretrained=False` nie daje losowo zainicjalizowanego modelu. Cała ta sekcja
dotyczy detekcji. Zadanie zorientowane wersji 2 obsługuje wyłącznie wnioskowanie
i nie ma ścieżki przenoszenia wag detekcji, ponieważ oba zadania używają innych backbone'ów.

<code-tabs name="train" />

Najważniejszym argumentem jest szybkość uczenia, a każda wersja ma własną wartość
domyślną zamiast wartości wspólnej dla biblioteki. Sygnatura `train()` w Pythonie
odczytuje ją z konfiguracji trenowania tej wersji, a CLI ustala tę samą wartość,
gdy `lr0` nie zostanie przekazane. Wersje 1 i 2 przyjmują również `lr_backbone`
i domyślnie ustawiają je na jedną dwudziestą `lr0`, zgodnie z oryginalną
konfiguracją. Wersja 4 korzysta z trenera D-FINE, który skaluje grupę parametrów
backbone za pomocą `backbone_lr_mult`.

Pozostaw `imgsz` w natywnym rozmiarze punktu kontrolnego, chyba że istnieje powód,
by go zmienić. Walidacja i predykcja w innych rozmiarach działają, z jednym
zastrzeżeniem: rozmiar prostokątny, którego liczba tokenów odpowiada rozmiarowi
natywnemu, nadal używa osadzenia zbudowanego dla niewłaściwych proporcji.

Informacje o zbiorach danych, augmentacji, wielu GPU i loggerach znajdziesz w sekcji [trenowanie](/docs/train).

## Walidacja

Metoda `val()` zwraca słownik kluczy `metrics/` obejmujących precyzję, czułość,
mAP 50 i mAP 50-95, mierzone na dowolnym zbiorze danych w formacie użytym do trenowania.

<code-tabs name="val" />

Wiersze w powyższej tabeli benchmarków pochodzą z zestawu benchmarkowego
LibreYOLO. Uwaga pod tabelą wskazuje zbiór danych, który je wygenerował, i
zawiera odnośniki do zapisów przebiegów.

Walidacja zorientowana korzysta z tego samego wywołania i raportuje te same
klucze oraz cztery powtórzone z przyrostkiem `(OBB)`. Dopasowanie używa obróconego
IoU zamiast IoU otaczających prostokątów, dlatego błąd kąta jest liczony jako
chybienie. Ustawienie `augment=True` jest odrzucane w tym zadaniu.

## Eksport

<export-matrix />

Macierz obejmuje całą linię na jednej stronie. Jeśli trzy wersje różnią się
obsługą formatu, komórka pokazuje najsłabszy poziom spośród nich, więc możliwości
żadnej ładowanej wersji nie są zawyżane. Wiersz zorientowany dotyczy wyłącznie
wersji 2. ONNX i TorchScript zostały dla niej zwalidowane w FP32, z partią 1
i stałym płótnem 1024 na 1024. OpenVINO, TensorRT i ExecuTorch konwertują i
ładują ponownie model, ale nie osiągnęły zgodności surowych wyników dla całego
zestawu zapytań. Najlepsze ramki zgadzają się z dokładnością do ułamka piksela,
natomiast ogon wyników się rozbiega.

Wyeksportowany artefakt jest ponownie ładowany przez `LibreYOLO()` na podstawie
rozszerzenia pliku, więc plik `.onnx` lub `.engine` zachowuje się jak punkt
kontrolny i zwraca ten sam obiekt `Results`.

<code-tabs name="export" />

## Punkty kontrolne

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

Nazwa pliku zawiera wersję, następnie rozmiar i zadanie. Wagi detekcji to
`LibreRTDETR<size>.pt`, `LibreRTDETRv2<size>.pt` i `LibreRTDETRv4<size>.pt`,
wszystkie w rozdzielczości 640 px. Wagi zorientowane istnieją tylko dla wersji 2
i dodają przyrostek zadania, od `LibreRTDETRv2n-obb.pt` do
`LibreRTDETRv2x-obb.pt`. Wszystkie działają przy 1024 px i zostały wytrenowane
na DOTA v1.0 zamiast COCO.

## Licencjonowanie

<provenance-box></provenance-box>

## Cytowanie

<citation-block />

Powyższy blok odpowiada cytowaniu publikowanemu przez autorów dla detekcji
w wersjach 1 i 2. Wagi zorientowane wersji 2 mają trzecie źródło, repozytorium
RiO-DETR na licencji Apache-2.0 pod adresem
[github.com/RicePasteM/RiO-DETR](https://github.com/RicePasteM/RiO-DETR), z
którego pochodzą punkty kontrolne DOTA. Jeśli użyto jednego z nich, należy
zacytować ten projekt. Wersja 4 jest osobną publikacją innej grupy i ma własny
blok cytowania pod adresem
[github.com/RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4#4-citation).
Jeśli użyto punktu kontrolnego wersji 4, należy zacytować tę publikację.

