---
title: Szybki start
seo_title: Szybki start z LibreYOLO
description: >-
  Uruchom detektor na obrazie, dostrój go na małym zbiorze danych i wyeksportuj
  do TorchScript lub ONNX, wszystko na CPU i w około dziesięciu wierszach kodu
  Pythona.
lead: >-
  Najkrótsza ścieżka przez LibreYOLO: predykcja na jednym obrazie, trenowanie na
  małym zbiorze danych, a następnie eksport wyniku. Każde z tych poleceń działa
  na CPU.
keywords:
  - libreyolo szybki start
  - libreyolo poradnik
  - libreyolo predykcja
  - libreyolo trenowanie
  - libreyolo eksport
  - yolo przykład python
last_verified: 1.5.0
meta:
  - label: Instalacja
    value: pip install libreyolo
    mono: true
  - label: Checkpoint
    value: LibreYOLO9t.pt
    mono: true
  - label: Sprzęt
    value: CPU wystarcza do wszystkiego na tej stronie
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Przy pierwszym użyciu pobiera checkpoint, a następnie zapisuje go w
        weights/.

        model = LibreYOLO("LibreYOLO9t.pt")


        # Pojedynczy obraz zwraca jeden obiekt Results.

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy.tolist())
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=yolo9-t save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Wideo i strumienie
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # stream=True zwraca po jednym obiekcie Results na klatkę zamiast
        tworzyć listę.

        # Zastąp ścieżkę indeksem kamery internetowej, adresem URL RTSP lub
        folderem.

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # coco8 to dołączony do biblioteki zbiór danych z 8 obrazami. Przy
        pierwszym

        # użyciu jest pobierany z adresu URL, więc nie trzeba uruchamiać żadnego
        skryptu.

        results = model.train(
            data="coco8.yaml",
            epochs=1,
            imgsz=640,
            batch=4,
            device="cpu",
        )


        print(results["save_dir"])

        print(results["best_checkpoint"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=yolo9-t data=coco8.yaml \
          epochs=1 imgsz=640 batch=4 device=cpu
    - label: Walidacja
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() zwraca zwykły słownik, a nie obiekt.
        metrics = model.val(data="coco8.yaml", device="cpu")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
  export:
    - label: TorchScript
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9t.pt")


        # export() zwraca ścieżkę do zapisanego pliku.

        path = model.export(format="torchscript")

        print(path)


        # Funkcja fabrykująca wybiera ścieżkę na podstawie sufiksu pliku, więc
        artefakt

        # wczytuje się ponownie jak checkpoint i zwraca ten sam obiekt Results.

        exported = LibreYOLO(path)

        result = exported(SAMPLE_IMAGE)

        print(len(result.boxes))
    - label: ONNX
      language: bash
      code: |
        pip install "libreyolo[onnx]"
        libreyolo export model=yolo9-t format=onnx imgsz=640
source_hash: c11b6bdbf0b6fdf1
---

## Instalacja

```bash
pip install libreyolo
```

To wszystko, czego wymagają poniższe sekcje dotyczące predykcji i trenowania.
Eksport do ONNX dodaje jedną zależność. Pełną listę zawiera strona
[instalacji](/docs/install).

## Predykcja

<code-tabs name="predict" />

`LibreYOLO()` jest funkcją fabrykującą. Odczytuje plik, ustala rodzinę, do której
należą wagi, i zwraca model tej rodziny. Dzięki temu zmiana detektora wymaga
zmiany jednego wiersza. Podanie `LibreYOLO9t.pt` bez katalogu powoduje
wyszukanie `weights/LibreYOLO9t.pt` względem katalogu roboczego i pobranie go w
to miejsce, jeśli brakuje pliku. Zasady pobierania i pracę offline opisuje
strona [checkpointy i wagi](/docs/weights).

Opcja `save=True` zapisuje kopię z adnotacjami w katalogu `runs/detect/`, w
podkatalogu `predict`, którego numer zwiększa się przy każdym uruchomieniu.
Zwracany obiekt `Results` zawiera `boxes`, a `names` mapuje indeks klasy na jej
etykietę. Ścieżka do jednego obrazu zwraca jeden obiekt `Results`. Katalog,
lista obrazów lub `stream=True` zwraca odpowiednio listę albo generator takich
obiektów.

## Trenowanie

<code-tabs name="train" />

`data` oznacza plik YAML zbioru danych. Plik `coco8.yaml` jest dołączony do
biblioteki, dlatego fragment kodu działa bez zmian. Nazwa niedołączonego pliku
jest interpretowana jako ścieżka. Zbiory danych są wyszukiwane w `~/datasets`
lub w katalogu wskazanym przez `LIBREYOLO_DATASETS_DIR`, jeśli ustawiono tę
zmienną.

Uruchomienie zapisuje dane w `project/name`, domyślnie w katalogu pod
`runs/train`, z plikami `weights/best.pt` i `weights/last.pt`. Funkcja `train()`
zwraca słownik obejmujący `save_dir`, `best_checkpoint`, `last_checkpoint`,
funkcje straty dla każdej epoki oraz metryki walidacji dla każdej epoki.
Wytrenowany checkpoint wczytuje się przez `LibreYOLO()` dokładnie tak samo jak
wstępnie wytrenowany.

Nie każdą rodzinę można trenować. Gdy rodzina obsługuje tylko inferencję,
`train()` zgłasza `NotImplementedError` i podaje tę informację. Znaczenie
poszczególnych poziomów wsparcia wyjaśniają [podstawowe pojęcia](/docs/concepts).

## Eksport

<code-tabs name="export" />

TorchScript nie wymaga niczego poza instalacją bazową. Każdy z pozostałych
formatów ma własny zestaw opcjonalnych zależności, a zakres obsługi zależy od
rodziny i zadania zamiast być jednolity. Szczegóły zawiera strona [eksport i
wdrożenie](/docs/export).

Argumenty przyjmowane przez każdy format obejmują `imgsz` (liczbę całkowitą lub
parę wysokości i szerokości), `batch` (domyślnie 1), `half`, `int8` z plikiem
YAML `data` do kalibracji, `dynamic` (domyślnie True), `simplify` (domyślnie
True), `opset`, `device` oraz `output_path`. Jeśli pominięto `output_path`, plik
jest zapisywany w `weights/` pod nazwą wyprowadzoną z checkpointu.

## Co dalej

- [Podstawowe pojęcia](/docs/concepts) opisują zadania, rodziny, rozmiary i nazwy checkpointów.
- [Checkpointy i wagi](/docs/weights) opisują automatyczne pobieranie, pracę offline i bezpieczne wczytywanie.
- [Import istniejących wag](/docs/migrate) przydaje się, jeśli istnieje już checkpoint z projektu źródłowego.
- [Wszystkie modele](/docs/models) pomagają wybrać rodzinę odpowiednią do problemu.
- Strony [Trenowanie](/docs/train), [Predykcja](/docs/predict) i [Eksport](/docs/export) zawierają pełne przepływy pracy.
