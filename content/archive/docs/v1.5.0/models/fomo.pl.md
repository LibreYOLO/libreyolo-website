---
title: FOMO
families:
  - fomo
seo_title: 'FOMO: lokalizacja punktów, trenowanie i eksport w LibreYOLO'
description: >-
  Uruchamiaj FOMO (Faster Objects, More Objects) w LibreYOLO: mały detektor
  lokalizujący punkty, przeznaczony do zliczania wielu małych obiektów.
  Instaluj, przewiduj, trenuj i eksportuj.
lead: >-
  FOMO to lokalizator punktów oparty na siatce: każda komórka siatki o niskiej
  rozdzielczości jest klasyfikowana jako tło lub środek obiektu, bez regresji
  ramek ograniczających. LibreYOLO obsługuje go w zadaniu punktowym.
keywords:
  - FOMO
  - Faster Objects More Objects
  - lokalizacja punktów
  - detekcja centroidów
  - detekcja małych obiektów
  - edge AI
  - detekcja na mikrokontrolerze
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Wagi LibreFOMO nie są pobierane automatycznie (zobacz sekcję
        Checkpointy poniżej).

        # Wskaż checkpoint pobrany wcześniej na dysk lokalny.

        model = LibreYOLO("./LibreFOMOs-point.pt")

        result = model(SAMPLE_IMAGE, save=True)


        for point in result.points:
            print(point.cls, point.conf, point.xy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=./LibreFOMOs-point.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=40, batch=32, lr0=3e-4,
        )
    - label: CLI
      language: bash
      code: >
        # Należy podać imgsz: wartość domyślna CLI to 640, a checkpoint s

        # przyjmuje tylko własną natywną wartość 96.

        libreyolo train model=./LibreFOMOs-point.pt data=my-dataset.yaml
        imgsz=96 epochs=40 batch=32 lr0=3e-4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/grid_F1"])
        print(metrics["metrics/grid_precision"], metrics["metrics/grid_recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=./LibreFOMOs-point.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=./LibreFOMOs-point.pt format=onnx
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("./LibreFOMOs-point.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.points.xy)
source_hash: 03015f2bcd9fe99d
---

## Instalacja

FOMO nie wymaga żadnego dodatku poza pakietem podstawowym.

```bash
pip install libreyolo
```

## Predykcja

W przeciwieństwie do każdej innej rodziny na tej stronie wagi LibreFOMO nie są
pobierane automatycznie. `LibreYOLO("LibreFOMOs-point.pt")` szuka tego pliku na
dysku i zgłasza `ValueError` z jego nazwą zamiast pobierać go z Hugging Face.
Najpierw należy pobrać checkpoint z [organizacji LibreYOLO](https://huggingface.co/LibreYOLO)
i wczytać go ze ścieżki lokalnej albo wytrenować własny (zobacz sekcję
Trenowanie poniżej).

<code-tabs name="predict" />

Wynik zawiera `points` zamiast `boxes`. Każdy wiersz ma postać
`x, y, class, confidence` i jest dostępny jako `result.points.data` albo przez
akcesory `.xy`, `.xyn`, `.cls` i `.conf`. Nie ma progu `iou`, ponieważ nie ma
ramek do tłumienia. `predict(..., nms_radius=1)` określa minimalny odstęp w
komórkach siatki, przy którym dwie detekcje mogą zostać zachowane. Nazwa pliku
musi zawierać sufiks zadania punktowego FOMO `-point`, aby loader go rozpoznał.
Więcej informacji o źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Warianty

Trzy rozmiary, `s`, `m` i `l`, używają coraz szerszych backbone w stylu
MobileNetV2 przy odpowiednio większych, stałych rozdzielczościach wejściowych,
każdy z pojedynczą głowicą klasyfikacyjną 1x1. Ta rodzina nie ma tutaj tabeli
benchmarku. Rozmiar pliku checkpointu w poniższej tabeli jest obecnie
najbardziej czytelną opublikowaną wskazówką dla poszczególnych rozmiarów.

## Trenowanie

<code-tabs name="train" />

`imgsz` nie jest dowolnym wyborem. Domyślnie przyjmuje natywną rozdzielczość
wczytanego checkpointu, a przekazanie innej wartości zgłasza `ValueError` z
oczekiwanym rozmiarem. Rozmiary te wynoszą 96 dla `s`, 192 dla `m` i 224 dla
`l`. CLI domyślnie ustawia `imgsz` na 640, dlatego polecenie `libreyolo train`
musi jawnie podać wartość zgodną z checkpointem.

Poza tym bez zmian konfiguracji trener wykonuje 40 epok z batchem 32,
optymalizatorem Adam przy `lr0=3e-4`, bez zaniku wag i z klasą pierwszego planu
ważoną 100 razy silniej niż tło w funkcji straty cross-entropy dla każdej
komórki, ponieważ w typowej scenie prawie każda komórka siatki jest tłem. EMA i
mieszana precyzja są domyślnie wyłączone. Nie jest stosowana żadna augmentacja
geometryczna ani kolorystyczna używana w innych częściach LibreYOLO: mosaic,
mixup, przesunięcie HSV, odbicie, obrót, przesunięcie przestrzenne i ścinanie
mają wartość zero.

Tą ścieżką wytrenowano opublikowane checkpointy LibreFOMO, od zera na COCO.

Informacje o zbiorach danych i loggerach zawiera strona
[trenowania](/docs/train).

## Walidacja

`val()` wybiera walidator na poziomie siatki zbudowany dla tej rodziny. Oprócz
kluczy `metrics/precision`, `metrics/recall` i `metrics/mAP@` z dopasowaniem
punktów, współdzielonych z innymi zadaniami punktowymi, przeszukuje progi
pewności i wartości `nms_radius`, a najlepszą kombinację F1 publikuje pod
kluczami `metrics/grid_F1`, `metrics/grid_precision`, `metrics/grid_recall` i
`metrics/grid_mean_distance`. Próg i promień, które ją uzyskały, znajdują się
pod `decode/threshold` i `decode/nms_radius`.

<code-tabs name="val" />

## Eksport

<export-matrix />

Wyeksportowany artefakt jest ponownie ładowany przez `LibreYOLO()` na podstawie
rozszerzenia pliku, więc plik `.onnx` lub `.engine` zachowuje się jak checkpoint
i zwraca ten sam obiekt `Results`. Obsługiwane jest także uruchomienie grafu w
samym środowisku uruchomieniowym bez zainstalowanego LibreYOLO, ale wtedy należy
samodzielnie napisać przetwarzanie wstępne i końcowe.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny. Żaden nie jest pobierany
automatycznie. Należy pobrać wybrany plik z podanej strony Hugging Face i
przekazać jego lokalną ścieżkę do `LibreYOLO()`.

<checkpoint-table />

## Licencja

<provenance-box>

Nie istnieje repozytorium kodu źródłowego FOMO, do którego można podać odnośnik.
Edge Impulse opisuje technikę we wpisie na blogu i dokumentacji produktu, ale
nie opublikowało kodu trenowania ani inferencji FOMO. Architektura i trenowanie
są własną implementacją tego opublikowanego opisu w LibreYOLO, a checkpointy
LibreFOMO wytrenowano od zera na COCO. Zarówno kod, jak i te wagi podlegają więc
własnej licencji MIT projektu LibreYOLO. Nazwa FOMO i opisywana przez nią
technika pozostają własnością Edge Impulse.

</provenance-box>
