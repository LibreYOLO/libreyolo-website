---
title: Segmentacja semantyczna
seo_title: Segmentacja semantyczna w LibreYOLO
description: >-
  Oznacz każdy piksel klasą w LibreYOLO: poznaj rodziny obsługujące to zadanie,
  format gęstych masek oraz wywołania predykcji, trenowania, walidacji i
  eksportu.
lead: >-
  Segmentacja semantyczna przypisuje klasę każdemu pikselowi obrazu i nie
  rozróżnia instancji tej samej klasy. Klucz zadania to semantic.
keywords:
  - segmentacja semantyczna python
  - klasyfikacja pikseli
  - dense prediction
  - trenowanie modelu segmentacji
  - mIoU
  - biblioteka segmentacji MIT
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Sufiks -sem w nazwie pliku wybiera zadanie, dlatego argument

        # task nie jest potrzebny.

        model = LibreYOLO("LibreSegformerb0-sem.pt")

        result = model(SAMPLE_IMAGE, save=True)


        mask = result.semantic_mask

        print(mask.data.shape)   # identyfikatory klas (H, W) na oryginalnym
        obszarze roboczym

        print(mask.classes)      # posortowane obecne identyfikatory klas, z
        pominięciem 255
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSegformerb0-sem.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Jedna klasa naraz
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreSegformerb0-sem.pt")(SAMPLE_IMAGE)
        mask = result.semantic_mask

        for class_id in mask.classes:
            pixels = mask.class_mask(class_id)   # wartości logiczne (H, W)
            print(result.names[class_id], int(pixels.sum()))
    - label: 'Inna rodzina, to samo wywołanie'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePIDNets-sem.pt")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: Na ADE20K
      language: bash
      code: |
        # ade20k.yaml zawiera osadzony skrypt pobierania archiwum około 1 GB,
        # dlatego wymaga jawnej zgody, chyba że dane są dostępne lokalnie.
        libreyolo train model=LibreSegformerb0-sem.pt data=ade20k.yaml \
          epochs=160 imgsz=512 batch=8 allow_download_scripts=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")

        # val() zwraca zwykły słownik, a nie obiekt.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512
    - label: Używanie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie sufiksu pliku, więc
        wyeksportowany artefakt

        # wczytuje się jak checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreSegformerb0-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: 44b92d8ba6062f04
---

## Definicja

Segmentacja semantyczna etykietuje piksele, a nie obiekty. Każdy piksel otrzymuje
jeden identyfikator klasy, a dwa samochody stykające się na obrazie stają się
jednym obszarem klasy samochodu bez granicy między nimi. Zliczanie instancji
realizuje [segmentacja instancji](/docs/tasks/instance-segmentation), natomiast
jednoczesne etykietowanie każdego piksela i rozdzielanie instancji realizuje
[segmentacja panoptyczna](/docs/tasks/panoptic-segmentation).

`semantic` jest kanonicznym kluczem zadania, a sufiks `-sem` w nazwie pliku
checkpointu wybiera to zadanie, więc `task=` nie jest potrzebne przy wczytywaniu
opublikowanych wag.

`predict()` wypełnia `result.semantic_mask`. Pole `.data` jest
całkowitoliczbową mapą klas `(H, W)` na obszarze roboczym oryginalnego obrazu,
`.classes` zawiera posortowaną listę obecnych identyfikatorów, a
`.class_mask(id)` zwraca logiczne zaznaczenie `(H, W)` jednej klasy. Wartość
`255` jest etykietą ignorowaną. Nigdy nie jest klasą, jest wyłączona z funkcji
straty i metryk, a `.classes` ją pomija.

## Modele

Trzy rodziny zarówno trenują, jak i przewidują:
[SegFormer](/docs/models/segformer),
[LingBot-Vision](/docs/models/lingbot-vision) oraz
[DINOv2](/docs/models/dinov2). SegFormer i LingBot-Vision działają z pakietem
podstawowym i udostępniają opublikowane wagi. DINOv2 wymaga
`pip install "libreyolo[rfdetr]"` i nie ma checkpointu hostowanego przez
LibreYOLO. Wczytuje nadrzędny backbone, a jego gęsta głowica rozpoczyna od
losowej inicjalizacji, dlatego jest punktem wyjścia do trenowania, a nie gotowym
modelem predykcyjnym.

Cztery kolejne rodziny przewidują, walidują i eksportują, ale ich `train()`
zgłasza `NotImplementedError`: [FCN](/docs/models/fcn),
[DeepLabv3](/docs/models/deeplabv3), [PIDNet](/docs/models/pidnet) oraz
[EoMT](/docs/models/eomt).

Zestawy klas różnią się zależnie od checkpointu, a nie rodziny. Opublikowane
wagi pochodzą ze zbiorów danych o bardzo odmiennych przestrzeniach etykiet, na
przykład 150 klas ADE20K wobec 19 klas Cityscapes. Pole `names` checkpointu
określa więc, co może on etykietować, a dwa checkpointy są porównywalne tylko
wtedy, gdy wytrenowano je na tym samym zbiorze.

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane lokalnie w
pamięci podręcznej.

<code-tabs name="predict" />

Mapa jest wynikiem argmax dla każdego piksela, dlatego nie ma etapu NMS, a `iou`
nigdy nie wpływa na wynik. `conf` i `max_det` są akceptowane dla zgodności API,
ale nie mają wpływu na SegFormer, PIDNet ani inne gęste predyktory. Wyjątkiem
jest EoMT, w którym `conf` filtruje wybór zapytań. Informacje o źródłach,
streamingu i obsłudze wyników znajdują się w sekcji
[predykcja](/docs/predict).

## Format zbioru danych

Każdy obraz jest sparowany z gęstą maską jednokanałową zamiast pliku etykiet
`.txt`. Znajduje się ją przez zastąpienie `images` katalogiem masek w ścieżce
obrazu.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  masks/
    train/000001.png
    val/000101.png
```

Maski są bezstratnymi obrazami jednokanałowymi, zwykle PNG, a pliki PNG w trybie
palety są odczytywane jako indeksy palety. Każda wartość piksela jest
identyfikatorem klasy w zakresie `0..nc-1`, wartość `255` oznacza ignorowanie, a
rozdzielczość maski musi być równa rozdzielczości sparowanego obrazu.

Plik YAML dodaje dwa klucze do wspólnego kontraktu:

```yaml
path: dataset
train: images/train
val: images/val
masks_dir: masks
nc: 19
names:
  0: road
  1: sidewalk
```

`masks_dir` jest nazwą katalogu zastępującego `images` i domyślnie przyjmuje
wartość `masks`. `label_mapping` jest opcjonalnym mapowaniem
`{source_id: train_id}`, stosowanym do wartości pikseli maski podczas
wczytywania. W ten sposób zbiór danych numerowany od 1 do 150 zmienia się w
zakres od 0 do 149. Każda niezmapowana wartość źródłowa staje się etykietą
ignorowaną, a każdy identyfikator trenowania musi mieścić się w `0..nc-1`.

Pominięcie `masks_dir` przełącza moduł wczytujący na ścieżkę rezerwową. Maski są
rasteryzowane podczas wczytywania z etykiet wielokątów rozwiązywanych zgodnie ze
zwykłą konwencją `images` do `labels`, a klasa `background` jest dołączana po
klasach obiektów, więc `nc` rośnie o jeden.

Kanoniczny moduł wczytujący to `libreyolo.data.SemanticDataset`.

## Trenowanie

<code-tabs name="train" />

`imgsz` jest tutaj ograniczone w sposób, który nie występuje w detektorze. Każda
rodzina deklaruje dzielnik, którego wielokrotnością musi być rozmiar wejścia.
Wynika on z siatki patchy lub kroku danych wyjściowych. Zarówno trenowanie, jak i
walidacja zgłaszają `ValueError` przed rozpoczęciem przebiegu, gdy `imgsz` nie
dzieli się bez reszty. Dzielnik wynosi 32 dla SegFormer, 16 dla LingBot-Vision i
EoMT, 14 dla DINOv2 oraz 8 dla FCN i PIDNet. Informacje o zbiorach danych,
augmentacji, wielu GPU i loggerach znajdują się w sekcji
[trenowanie](/docs/train).

## Walidacja

`val()` zwraca zwykły słownik kluczy `metrics/`, obliczanych dla podziału
wskazanego przez `val` w pliku YAML zbioru danych.

<code-tabs name="val" />

`metrics/mIoU` to średni współczynnik intersection over union. Dla każdej klasy
jest to nałożenie przewidzianych i prawdziwych pikseli podzielone przez ich sumę
zbiorów, uśrednione dla klas. Jest głównym wynikiem i służy do wyboru najlepszej
epoki podczas trenowania. `metrics/pixel_accuracy` jest udziałem pikseli, którym
przypisano prawidłową klasę. Duża klasa tła może zawyżyć tę wartość, dlatego do
porównań należy używać mIoU. Piksele oznaczone `255` nie są uwzględniane w
żadnej z nich. Słownik zawiera również `fitness`, czyli kopię wartości mIoU.

## Eksport

<code-tabs name="export" />

Wyeksportowany artefakt jest ponownie wczytywany przez `LibreYOLO()` na podstawie
sufiksu pliku, więc plik `.onnx` lub `.engine` zachowuje się jak checkpoint i
zwraca ten sam obiekt `Results`. Zakres formatów różni się zależnie od rodziny.
Macierz na stronie każdego modelu jest generowana ze zweryfikowanego zestawu, a
nie wpisywana ręcznie. Formaty, ich dodatki i ograniczenia opisano w sekcji
[eksport i wdrożenie](/docs/export).

