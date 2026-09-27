---
title: Detekcja obiektów
seo_title: Detekcja obiektów w LibreYOLO
description: >-
  Wykrywaj obiekty jako ramki wyrównane do osi w LibreYOLO. Poznaj rodziny
  obsługujące zadanie, format etykiet oraz wywołania predykcji, trenowania,
  walidacji i eksportu.
lead: >-
  Detekcja obiektów lokalizuje każdą instancję obiektu na obrazie i zwraca dla
  niej prostokąt wyrównany do osi, etykietę klasy oraz wskaźnik. Kluczem zadania
  jest detect.
keywords:
  - detekcja obiektów python
  - wykrywanie obiektów na obrazie
  - bounding box detection
  - biblioteka detekcji obiektów mit
  - alternatywa yolo
  - trenowanie detektora obiektów
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9t.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 'Inna rodzina, to samo wywołanie'
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Funkcja fabrykująca wybiera ścieżkę na podstawie checkpointu, a każdy
        detektor

        # zwraca ten sam obiekt Results, więc zmiana rodziny wymaga zmiany
        jednego wiersza.

        model = LibreYOLO("LibreDFINEn.pt")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy.shape)
    - label: Wideo i strumienie
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # Dowolne źródło przyjmowane przez bibliotekę: plik, folder, adres URL,
        indeks

        # kamery internetowej, strumień RTSP lub lista .streams.

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # coco128.yaml pobiera próbkę ze 128 obrazami przy pierwszym użyciu. W
        przypadku

        # rzeczywistego uruchomienia skieruj data do pliku YAML własnego zbioru
        danych.

        model.train(data="coco128.yaml", epochs=50, imgsz=640, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 imgsz=640 batch=8
    - label: Wiele GPU
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() zwraca zwykły słownik, a nie obiekt.
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/AR100"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9t.pt data=coco128.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9t.pt format=onnx imgsz=640
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Funkcja fabrykująca wybiera ścieżkę na podstawie sufiksu pliku, więc

        # wyeksportowany artefakt wczytuje się jak checkpoint i zwraca ten sam
        obiekt Results.

        model = LibreYOLO("LibreYOLO9t.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: c735b6e3de78dd2b
---

## Definicja

Detekcja obiektów odpowiada, gdzie znajduje się każdy obiekt i czym jest. Jeden
obraz na wejściu daje jeden wiersz na instancję na wyjściu: cztery liczby
opisujące prostokąt, indeks klasy i wskaźnik. Nie zawiera informacji o kształcie
na poziomie pikseli, orientacji ani częściach. Odróżnia ją to od [segmentacji
instancji](/docs/tasks/instance-segmentation), [obróconych
ramek](/docs/tasks/oriented-detection) i [estymacji pozy](/docs/tasks/pose-estimation).

`detect` jest kanonicznym i domyślnym kluczem zadania. Checkpoint, którego nazwa
pliku nie zawiera sufiksu zadania, jest wczytywany jako detektor.

Funkcja `predict()` wypełnia `result.boxes`. `.xyxy` podaje narożniki w pikselach
na płótnie oryginalnego obrazu, `.conf` wskaźnik, a `.cls` indeks klasy w
`result.names`. `.xywh`, `.xyxyn` i `.xywhn` są widokami tych samych wierszy, a
`.id` zawiera identyfikator śledzenia po dołączeniu trackera. Iterowanie po
obiekcie `Boxes` zwraca wycinki z jednym wierszem, dlatego `box.cls`, `box.conf`
i `box.xyxy` działają osobno dla każdej detekcji.

## Modele

Dwanaście rodzin obsługuje zarówno trenowanie, jak i predykcję:
[YOLOv9](/docs/models/yolov9), [RF-DETR](/docs/models/rf-detr),
[EdgeCrafter](/docs/models/edgecrafter), [RT-DETR](/docs/models/rt-detr),
[D-FINE](/docs/models/d-fine), [DEIM](/docs/models/deim),
[Dome-DETR](/docs/models/dome-detr), [YOLO-NAS](/docs/models/yolo-nas),
[YOLOX](/docs/models/yolox), [YOLOv7](/docs/models/yolov7),
[RTMDet](/docs/models/rtmdet) oraz [PicoDet](/docs/models/picodet). YOLOv9 i
RF-DETR są dwiema głównymi rodzinami, które jako pierwsze otrzymują nowe
funkcje. RF-DETR wymaga własnego zestawu zależności
`pip install "libreyolo[rfdetr]"`, a pozostałe działają z pakietem bazowym.

Kolejnych jedenaście rodzin obsługuje predykcję, walidację i eksport, ale ich
funkcja `train()` zgłasza `NotImplementedError`: [LW-DETR](/docs/models/lw-detr),
[DETR](/docs/models/detr), [Deformable DETR](/docs/models/deformable-detr),
[DINO-DETR](/docs/models/dino-detr), [Faster R-CNN](/docs/models/faster-rcnn),
[Mask R-CNN](/docs/models/mask-rcnn), [FCOS](/docs/models/fcos),
[RetinaNet](/docs/models/retinanet), [SSD](/docs/models/ssd),
[CenterNet](/docs/models/centernet) oraz
[EfficientDet](/docs/models/efficientdet).

Linia Darknet, czyli [YOLOv1](/docs/models/yolov1),
[YOLOv2](/docs/models/yolov2), [YOLOv3](/docs/models/yolov3) i
[YOLOv4](/docs/models/yolov4), jest zachowana jako niezmienny eksponat.
Predykcja, walidacja i eksport działają, ale trenowanie nie.

Osobna grupa przyjmuje listę klas w czasie działania zamiast z checkpointu,
dzięki czemu wykrywa nazwy niewidziane podczas trenowania:
[Grounding DINO](/docs/models/grounding-dino), [OWLv2](/docs/models/owlv2),
[OMDet-Turbo](/docs/models/omdet-turbo) i [OV-DEIM](/docs/models/ov-deim), a
także rodziny wizualno-językowe [Florence-2](/docs/models/florence-2),
[Kosmos-2](/docs/models/kosmos-2), [Qwen3-VL](/docs/models/qwen3-vl),
[SmolVLM2](/docs/models/smolvlm2), [InternVL3](/docs/models/internvl3),
[LFM2-VL](/docs/models/lfm2-vl), [LocateAnything](/docs/models/locate-anything),
[SenseNova-Vision](/docs/models/sensenova-vision) oraz
[LibreMODUS](/docs/models/libremodus). Modele te są wczytywane przez własne
funkcje fabrykujące i zestawy zależności, a ich strony zawierają dokładne
wywołania.

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

`conf` ustawia próg pewności, a `max_det` ogranicza liczbę wierszy. `iou` jest
progiem NMS, więc ma wpływ wyłącznie na rodziny korzystające z NMS. RF-DETR i
głowica end-to-end YOLOv9 dekodują stały zestaw predykcji i go ignorują.
Informacje o źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Format zbioru danych

Każdy obraz ma jeden plik etykiet `.txt`, znajdowany przez zamianę `images` na
`labels` w ścieżce obrazu i zmianę rozszerzenia.

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

Każdy wiersz ma dokładnie pięć pól. Po indeksie klasy znajduje się
znormalizowana ramka opisana przez środek i rozmiar:

```text
<class_id> <cx> <cy> <w> <h>
```

Współrzędne są wartościami float w zakresie `[0, 1]` względem szerokości i
wysokości oryginalnego obrazu. `w` i `h` muszą być dodatnie. Brakujący lub pusty
plik etykiet oznacza, że obraz nie zawiera obiektów. Wiersze nie zawierają
pewności ani identyfikatora śledzenia.

Plik YAML podaje podziały i klasy:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

`train` i `val` mogą być katalogami obrazów, plikami `.txt` z listą obrazów lub
listami obu rodzajów wartości. `nc` jest opcjonalne, a gdy występuje, musi być
zgodne z `names`. Natywny format JSON COCO również działa. Należy dodać
mapowanie `annotations` od nazwy podziału do pliku JSON, a ścieżka podziału
wskazuje wtedy katalog główny obrazów. Jeśli występuje `names`, definiuje
identyfikatory etykiet, więc nazwy kategorii JSON muszą być z nim zgodne.

## Trenowanie

<code-tabs name="train" />

Argumenty `epochs`, `imgsz`, `batch` i `lr0` należy dostosować w pierwszej
kolejności. `lr0` jest tym, którego nie można przenosić między rodzinami.
Współczynnik uczenia tolerowany przez detektor konwolucyjny spowoduje
rozbieżność transformera, dlatego należy przyjąć wartość ze strony modelu, a
nie z przykładu innej rodziny. Rodzina może również całkowicie ignorować
argument, co jest wskazane na jej stronie. Informacje o zbiorach danych,
augmentacji, wielu GPU i modułach rejestrujących zawiera strona
[trenowania](/docs/train).

## Walidacja

Funkcja `val()` zwraca zwykły słownik kluczy `metrics/`, obliczany za pomocą
oceny COCO na podziale wskazanym przez `val` w pliku YAML zbioru danych.

<code-tabs name="val" />

`metrics/mAP50-95` jest średnią average precision uśrednioną dla progów IoU od
0.50 do 0.95 i stanowi główny wynik. `metrics/mAP50` i `metrics/mAP75` są
wersjami dla pojedynczego progu. `metrics/mAP_small`, `metrics/mAP_medium` i
`metrics/mAP_large` dzielą tę samą średnią według powierzchni obiektu, a
`metrics/AR1`, `metrics/AR10`, `metrics/AR100`, `metrics/AR_small`,
`metrics/AR_medium` i `metrics/AR_large` są odpowiadającymi wartościami
średniego recall. `metrics/AR_max_det` i `metrics/max_det` zapisują limit
detekcji użyty w uruchomieniu.

Klucze `metrics/precision` i `metrics/recall` w tym zadaniu należy interpretować
ostrożnie. Zachowano je dla zgodności wstecznej i są aliasami, a nie punktem
pracy. `metrics/precision` zawiera tę samą wartość co `metrics/mAP50-95`, a
`metrics/recall` tę samą wartość co `metrics/AR100`. Wykreślenie ich jako pary
precision-recall zgłasza jedną wartość dwukrotnie. Cztery klucze są również
powtórzone z sufiksem `(B)` oznaczającym ramkę, dzięki czemu klucz detekcji ma
taką samą postać w modelu, który przewiduje również maski:
`metrics/mAP50-95(B)`, `metrics/mAP50(B)`, `metrics/precision(B)` i
`metrics/recall(B)`.

## Eksport

<code-tabs name="export" />

Wyeksportowany artefakt wczytuje się ponownie przez `LibreYOLO()` na podstawie
sufiksu pliku. Plik `.onnx` lub `.engine` działa więc jak checkpoint i zwraca
ten sam obiekt `Results`. Zakres formatów różni się między rodzinami, a macierz
na stronie każdego modelu jest generowana ze zweryfikowanego zestawu, a nie
wpisywana ręcznie. Formaty, ich zestawy zależności i ograniczenia opisuje strona
[eksportu i wdrożenia](/docs/export).
