---
title: Klasyfikacja obrazów
seo_title: Klasyfikacja obrazów w LibreYOLO
description: >-
  Przypisuj etykietę do całego obrazu w LibreYOLO. Poznaj rodziny obsługujące
  zadanie, układ zbioru danych ImageFolder oraz wywołania predykcji, trenowania,
  walidacji i eksportu.
lead: >-
  Klasyfikacja obrazów przypisuje jeden rozkład etykiet do całego obrazu i nie
  lokalizuje żadnych elementów. Kluczem zadania jest classify.
keywords:
  - klasyfikacja obrazów python
  - trenowanie klasyfikatora obrazów
  - zbiór danych ImageFolder
  - top-1 accuracy
  - klasyfikacja zero-shot
  - biblioteka klasyfikacji mit
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Sufiks -cls w nazwie pliku wybiera zadanie, więc argument task
        # nie jest potrzebny.
        model = LibreYOLO("LibreResNet50-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.names[result.probs.top1], float(result.probs.top1conf))
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreResNet50-cls.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Pełny rozkład
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        result = LibreYOLO("LibreResNet50-cls.pt")(SAMPLE_IMAGE)

        probs = result.probs


        # .data jest pełnym wektorem (C,), a top5/top5conf to uporządkowane
        widoki.

        print(probs.data.shape)

        for index, score in zip(probs.top5, probs.top5conf):
            print(result.names[index], float(score))
    - label: Zero-shot bez trenowania
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # CLIP ocenia obraz względem promptów tekstowych, dlatego zestaw etykiet

        # ustala się przy wywołaniu, zamiast zapisywać w checkpointcie.

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        model.set_classes(["a person jumping", "an empty street", "a parked
        car"])

        result = model(SAMPLE_IMAGE)


        print(model.names[result.probs.top1], float(result.probs.top1conf))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # imagenette160 jest znaną nazwą zbioru danych i zostanie pobrany przy
        pierwszym użyciu.

        # W przypadku własnych danych przekaż katalog zawierający podział
        train/.

        model = LibreYOLO("LibreResNet50-cls.pt")

        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 epochs=5
    - label: Wiele GPU
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")

        # val() zwraca zwykły słownik, a nie obiekt.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreResNet50-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreResNet50-cls.pt format=onnx
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Funkcja fabrykująca wybiera ścieżkę na podstawie sufiksu pliku, więc

        # wyeksportowany artefakt wczytuje się jak checkpoint i zwraca ten sam
        obiekt Results.

        model = LibreYOLO("LibreResNet50-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1, result.probs.top1conf)
source_hash: 836bea76cd2cdf92
---

## Definicja

Klasyfikacja obrazów tworzy jeden wskaźnik na klasę dla całego obrazu i nie
zwraca żadnych współrzędnych. Odpowiada na pytanie, co znajduje się na obrazie,
a nigdy gdzie, co odróżnia ją od [detekcji
obiektów](/docs/tasks/object-detection).

`classify` jest kanonicznym kluczem zadania, a sufiks `-cls` w nazwie pliku
checkpointu je wybiera. Ten sufiks jest wymagany, a nie opcjonalny dla rodzin
klasyfikacyjnych. Dlatego `LibreResNet50.pt` nie jest interpretowany jako
klasyfikator, a `LibreResNet50-cls.pt` już tak.

Funkcja `predict()` wypełnia `result.probs` i pozostawia `boxes` puste. `.data`
jest pełnym wektorem wskaźników, `.top1` indeksem największego wskaźnika, a
`.top1conf` jego wartością. `.top5` zawiera pięć największych indeksów w
kolejności malejącej, a `.top5conf` ich wskaźniki. Indeksy odwołują się do
`result.names`. Wycinanie obiektu `Results` nigdy nie skraca `probs`, ponieważ
wektor należy do obrazu, a nie do pojedynczego wiersza.

## Modele

Pięć rodzin obsługuje zarówno trenowanie, jak i predykcję:
[ResNet](/docs/models/resnet), [ConvNeXt](/docs/models/convnext),
[MobileNetV4](/docs/models/mobilenetv4),
[EfficientNetV2](/docs/models/efficientnetv2) oraz
[DINOv2](/docs/models/dinov2). Pierwsze cztery działają z pakietem bazowym i
mają opublikowane wagi. DINOv2 wymaga `pip install "libreyolo[rfdetr]"` i nie ma
checkpointu hostowanego przez LibreYOLO. Wczytuje backbone projektu źródłowego
z losowo inicjowaną głowicą liniową, dlatego stanowi punkt początkowy do
dostrajania, a nie gotowy model predykcyjny.

Kolejnych pięć rodzin obsługuje predykcję, walidację i eksport, ale ich funkcja
`train()` zgłasza `NotImplementedError`: [ViT](/docs/models/vit),
[Swin](/docs/models/swin), [VGG](/docs/models/vgg),
[AlexNet](/docs/models/alexnet) oraz [DeiT](/docs/models/deit).

[CLIP](/docs/models/clip) i [SigLIP2](/docs/models/siglip2) klasyfikują bez
stałego zestawu etykiet. Oceniają obraz względem promptów tekstowych, dlatego
`set_classes()` definiuje klasy podczas wywołania i w ogóle nie ma etapu
trenowania dla nowego zestawu etykiet. Obie rodziny obsługują również zadanie
`embed`.

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

Argumenty `conf`, `iou` i `max_det` nie mają tutaj wpływu. Nie ma kandydatów do
filtrowania progiem ani tłumienia, a jedynie jeden rozkład. Informacje o
źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Format zbioru danych

Klasyfikacja korzysta z drzewa katalogów, a nie z plików etykiet ani pliku YAML.
`data` jest katalogiem głównym zbioru danych.

```text
dataset/
  train/
    tench/000001.jpg
    parachute/000002.jpg
  val/
    tench/000101.jpg
    parachute/000102.jpg
```

`train/` jest wymagany do trenowania i definiuje mapowanie klas na indeksy na
podstawie posortowanych nazw folderów. Pierwszy folder alfabetycznie staje się
klasą 0. `val/` jest wymagany do walidacji. Podział `test/` może być obecny,
ale domyślne polecenia trenowania i walidacji go nie używają. Każdy podział
inny niż `train` musi zawierać te same nazwy folderów klas co oczekiwany zestaw
klas. Dzięki temu niezgodność powoduje wyraźny błąd, a nie ocenę jako błędna
predykcja. Akceptowane rozszerzenia obrazów to `.jpg`, `.jpeg`, `.png`, `.bmp`,
`.webp`, `.tif` i `.tiff`.

`data` przyjmuje trzy rodzaje wartości: ścieżkę do katalogu zawierającego
podział `train/`, adres URL pliku `.zip` lub jedną ze znanych nazw zbiorów
danych, `imagenette160` i `smoke10`, które są pobierane i zapisywane w pamięci
podręcznej przy pierwszym użyciu.

Kanonicznym modułem wczytującym jest `libreyolo.data.classify_dataset`.

## Trenowanie

<code-tabs name="train" />

Nie deklaruje się `nc`. Liczba klas pochodzi z nazw folderów w `train/`, a
końcowa warstwa liniowa jest przebudowywana odpowiednio do tej liczby, podczas
gdy backbone jest przenoszony bez zmian. Informacje o zbiorach danych,
augmentacji, wielu GPU i modułach rejestrujących zawiera strona
[trenowania](/docs/train).

## Walidacja

Funkcja `val()` zwraca zwykły słownik kluczy `metrics/`, obliczany na podziale
`val/` katalogu głównego zbioru danych.

<code-tabs name="val" />

`metrics/accuracy_top1` jest udziałem obrazów, dla których klasa z największym
wskaźnikiem jest klasą prawdziwą. To główny wynik używany podczas trenowania do
wyboru najlepszej epoki. `metrics/accuracy_top5` jest udziałem obrazów, dla
których prawdziwa klasa znajduje się wśród pięciu klas o największym wskaźniku.
Metryka mówi tym mniej, im mniej klas zawiera zbiór danych. Słownik obejmuje
również `fitness`, kopię wartości top-1.

## Eksport

<code-tabs name="export" />

Wyeksportowany artefakt wczytuje się ponownie przez `LibreYOLO()` na podstawie
sufiksu pliku. Plik `.onnx` lub `.engine` działa więc jak checkpoint i zwraca
ten sam obiekt `Results`. Zakres formatów różni się między rodzinami, a macierz
na stronie każdego modelu jest generowana ze zweryfikowanego zestawu, a nie
wpisywana ręcznie. Formaty, ich zestawy zależności i ograniczenia opisuje strona
[eksportu i wdrożenia](/docs/export).
