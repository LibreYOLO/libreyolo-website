---
title: Segmentacja instancji
seo_title: Segmentacja instancji w LibreYOLO
description: >-
  Segmentuj poszczególne obiekty w LibreYOLO. Poznaj rodziny obsługujące
  zadanie, format etykiet wielokątów oraz wywołania predykcji, trenowania,
  walidacji i eksportu.
lead: >-
  Segmentacja instancji lokalizuje każdą instancję obiektu i zwraca dla niej
  maskę na poziomie pikseli razem z ramką, klasą i wskaźnikiem zwracanym przez
  detektor. Kluczem zadania jest segment.
keywords:
  - segmentacja instancji python
  - predykcja maski obiektu
  - trenowanie modelu segmentacji
  - etykiety wielokątów
  - biblioteka segmentacji mit
  - mask mAP
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Sufiks -seg w nazwie pliku wybiera głowicę masek, więc argument task
        # nie jest potrzebny.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)   # (N, H, W), jedna maska na detekcję
        print(result.boxes.xyxy.shape)   # (N, 4), te same N wierszy
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDFINEn-seg.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Kontury masek
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDFINEn-seg.pt")

        result = model(SAMPLE_IMAGE)


        # .xy jest listą konturów pikselowych (P, 2), a .xyn tych samych
        znormalizowanych konturów.

        for name, contour in zip(result.boxes.cls, result.masks.xy):
            print(result.names[int(name)], contour.shape)
    - label: 'Inna rodzina, to samo wywołanie'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Kontynuuje od opublikowanych wag segmentacji, łącznie z głowicą masek.

        # data musi wskazywać zbiór danych, którego etykiety zawierają
        wielokąty.

        model = LibreYOLO("LibreDFINEn-seg.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: Z wag detekcji
      language: bash
      code: >
        # Wagi detekcji nie zawierają głowicy masek, dlatego jest to jawny
        transfer:

        # głowica rozpoczyna bez wytrenowania. Zezwala na to podanie
        task=segment.

        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])       # maski
        print(metrics["metrics/mAP50-95(M)"])    # maski, jawnie
        print(metrics["metrics/mAP50-95(B)"])    # ramki
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn-seg.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDFINEn-seg.pt format=onnx imgsz=640
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Funkcja fabrykująca wybiera ścieżkę na podstawie sufiksu pliku, więc

        # wyeksportowany artefakt wczytuje się jak checkpoint i zwraca ten sam
        obiekt Results.

        model = LibreYOLO("LibreDFINEn-seg.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.masks.data.shape)
source_hash: 33e331eac0f9b0af
---

## Definicja

Segmentacja instancji jest detekcją rozszerzoną o kształt. Każda instancja
obiektu nadal otrzymuje ramkę, klasę i wskaźnik, a dodatkowo binarną maskę
obejmującą należące do niej piksele. Maski mogą się nakładać, a piksele
nienależące do żadnego obiektu pozostają nieprzypisane. Odróżnia to zadanie od
[segmentacji semantycznej](/docs/tasks/semantic-segmentation) i [segmentacji
panoptycznej](/docs/tasks/panoptic-segmentation).

`segment` jest kanonicznym kluczem zadania, a sufiks `-seg` w nazwie pliku
checkpointu je wybiera. Dlatego podczas wczytywania opublikowanych wag nie
trzeba podawać `task=`.

Funkcja `predict()` wypełnia `result.masks` razem z `result.boxes`. `.data` jest
stosem `(N, H, W)` na płótnie oryginalnego obrazu, którego wiersze są wyrównane
z ramkami. Maska `i` należy więc do ramki `i`. `.xy` przekształca każdą maskę w
jej największy kontur zewnętrzny jako tablicę pikseli `(P, 2)`, a `.xyn` zwraca
ten sam znormalizowany kontur.

## Modele

Cztery rodziny obsługują zarówno trenowanie, jak i predykcję masek:
[RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter),
[D-FINE](/docs/models/d-fine) oraz [RTMDet](/docs/models/rtmdet). RF-DETR wymaga
własnego zestawu zależności `pip install "libreyolo[rfdetr]"`, a pozostałe trzy
działają z pakietem bazowym.

[Mask R-CNN](/docs/models/mask-rcnn) przewiduje, waliduje i eksportuje maski,
ale jego funkcja `train()` zgłasza `NotImplementedError`.

[EoMT](/docs/models/eomt) przewiduje i waliduje maski, ale również nie obsługuje
trenowania, a zakres jego eksportu jest jeszcze węższy. `export()` przyjmuje
wyłącznie zadanie semantyczne i zgłasza `NotImplementedError` dla `segment` oraz
`panoptic`, ponieważ nie zdefiniowano wymaganego przez te zadania kontraktu
masek zapytań w środowisku uruchomieniowym. EoMT należy używać do masek
instancji w Pythonie, a nie przez wyeksportowany graf.

Osobna grupa wykonuje segmentację na podstawie promptu zamiast listy klas.
Kliknięcie, ramka lub fraza wybiera obiekt, a model zwraca jego maskę. W ten
sposób działają [SAM](/docs/models/sam), [SAM 2](/docs/models/sam-2),
[SAM 3](/docs/models/sam-3), [MobileSAM](/docs/models/mobilesam),
[EdgeTAM](/docs/models/edgetam) i [PicoSAM3](/docs/models/picosam3), a także
[SenseNova-Vision](/docs/models/sensenova-vision), którego segmentacja jest
referencyjna i przyjmuje frazę nazywającą jeden obiekt. Modele te są wczytywane
przez własne funkcje fabrykujące i zestawy zależności, a ich strony zawierają
dokładne wywołania.

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

`conf` i `max_det` kształtują dane wyjściowe tak samo jak w detekcji, a maski są
filtrowane razem z ramkami, do których należą. Informacje o źródłach, streamingu
i obsłudze wyników zawiera strona [predykcji](/docs/predict).

## Format zbioru danych

Układ jest taki sam jak dla detekcji. Każdy obraz ma jeden plik etykiet `.txt`,
znajdowany przez zamianę `images` na `labels` w ścieżce obrazu i zmianę
rozszerzenia.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

Zmienia się zawartość wiersza. Segment składa się z indeksu klasy i następującego
po nim płaskiego wielokąta:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

Wymagane są co najmniej trzy punkty, więc liczba współrzędnych po indeksie klasy
musi być parzysta i wynosić co najmniej sześć, a wielokąt nie może być
zdegenerowany. Współrzędne są wartościami float w zakresie `[0, 1]` względem
szerokości i wysokości oryginalnego obrazu. Pięciopolowy wiersz detekcji jest
również akceptowany w zbiorze danych segmentacji i odczytywany jako segment
prostokątny. Dzięki temu zbiór zawierający wyłącznie ramki można wczytać bez
etapu konwersji.

Plik YAML jest taki sam jak dla detekcji:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

Natywny format JSON COCO również działa. Należy dodać mapowanie `annotations`
od nazwy podziału do pliku JSON, a ścieżka podziału wskazuje katalog główny
obrazów.

## Trenowanie

<code-tabs name="train" />

Domyślnie trenowanie jest kontynuowane z opublikowanego checkpointu `-seg`.
Można rozpocząć od wag detekcji, ale jest to celowy transfer. Wagi te nie
zawierają głowicy masek, więc rozpoczyna ona bez wytrenowania, a przekazanie
`task=segment` zezwala na tę zamianę. Informacje o zbiorach danych, augmentacji,
wielu GPU i modułach rejestrujących zawiera strona [trenowania](/docs/train).

## Walidacja

Funkcja `val()` zwraca zwykły słownik kluczy `metrics/`. Ramki i maski są
oceniane osobno za pomocą oceny COCO, a wyniki masek są wynikami głównymi.

<code-tabs name="val" />

Klucze bez sufiksu zawierają wyniki masek: `metrics/mAP50-95`,
`metrics/mAP50`, `metrics/mAP75`, następnie `metrics/mAP_small`,
`metrics/mAP_medium` i `metrics/mAP_large` według powierzchni obiektu oraz
`metrics/AR1`, `metrics/AR10`, `metrics/AR100`, `metrics/AR_small`,
`metrics/AR_medium` i `metrics/AR_large` dla średniego recall.
`metrics/AR_max_det` i `metrics/max_det` zapisują limit detekcji użyty w
uruchomieniu.

Cztery wartości są również publikowane z jawnym sufiksem, `(M)` dla maski i
`(B)` dla ramki, dzięki czemu porównanie nie zależy od tego, którą wartość dana
rodzina uznała za główną: `metrics/mAP50-95(M)` i `metrics/mAP50-95(B)`,
`metrics/mAP50(M)` i `metrics/mAP50(B)`, `metrics/precision(M)` i
`metrics/precision(B)`, `metrics/recall(M)` i `metrics/recall(B)`. To zadanie
nie ma klucza `metrics/precision` ani `metrics/recall` bez sufiksu.

Klucze precision i recall należy interpretować ostrożnie. Zachowano je dla
zgodności wstecznej i są aliasami, a nie punktem pracy.
`metrics/precision(M)` zawiera tę samą wartość co `metrics/mAP50-95(M)`, a
`metrics/recall(M)` tę samą wartość co AR masek przy 100 detekcjach. Sufiks
`(B)` działa tak samo dla ramek. Wykreślenie tej pary zgłasza jedną wartość
dwukrotnie.

## Eksport

<code-tabs name="export" />

Wyeksportowany artefakt wczytuje się ponownie przez `LibreYOLO()` na podstawie
sufiksu pliku. Plik `.onnx` lub `.engine` działa więc jak checkpoint i zwraca
ten sam obiekt `Results`. Zakres segmentacji jest węższy niż zakres detekcji w
tej samej rodzinie. Macierz na stronie każdego modelu jest generowana ze
zweryfikowanego zestawu i podaje przyczynę niedostępności formatu. Formaty, ich
zestawy zależności i ograniczenia opisuje strona [eksportu i
wdrożenia](/docs/export).
