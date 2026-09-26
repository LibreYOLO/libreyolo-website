---
title: YOLOv7
families:
  - yolo7
seo_title: 'YOLOv7 w LibreYOLO: predykcja, trenowanie i eksport na licencji MIT'
description: >-
  Uruchamiaj YOLOv7 w LibreYOLO do detekcji obiektów: instaluj, przewiduj,
  trenuj, waliduj i eksportuj kod i wagi na licencji MIT.
lead: >-
  YOLOv7 to jednostopniowy detektor oparty na kotwicach, którego głowica dodaje
  wyuczone przesunięcia wiedzy ukrytej przed końcowym splotem. LibreYOLO
  obsługuje jego jeden opublikowany rozmiar do detekcji.
keywords:
  - YOLOv7
  - detekcja obiektów
  - detekcja oparta na kotwicach
  - wiedza ukryta
  - ImplicitA
  - detekcja obiektów w czasie rzeczywistym
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO7b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO7b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO7b.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO7b.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
    - label: Ciepły start nowego modelu
      language: python
      code: >
        from libreyolo import LibreYOLO7


        # pretrained=True zawsze wczytuje opublikowany checkpoint
        LibreYOLO7b.pt,

        # niezależnie od sposobu utworzenia tej instancji. Bezpośrednie
        utworzenie

        # klasy zamiast przez LibreYOLO() rozpoczyna bez wczytanych wag.

        model = LibreYOLO7(None, size="b")

        model.train(data="my-dataset.yaml", epochs=300, pretrained=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO7b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLO7b.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLO7b.pt format=tensorrt imgsz=640
        half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreYOLO7b.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 361e81de5614a571
---

## Instalacja

YOLOv7 nie wymaga żadnego dodatku poza pakietem podstawowym.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

Zwracany obiekt `Results` jest taki sam dla każdej rodziny, więc zmiana
detektora wymaga zmiany jednego wiersza. Argument `conf` ustawia próg pewności,
a `iou` próg NMS stosowany po zdekodowaniu głowicy opartej na kotwicach. Więcej
informacji o źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Warianty

LibreYOLO udostępnia jeden rozmiar, `b`. Projekt źródłowy publikuje jeden model
YOLOv7, dlatego nie ma rozmiaru do wyboru.

## Trenowanie

<code-tabs name="train" />

`pretrained` jest odczytywane, w przeciwieństwie do argumentu o tej samej nazwie
w niektórych innych rodzinach. Przekazanie `True` rozpoczyna od opublikowanego
checkpointu `LibreYOLO7b.pt` (pobieranego automatycznie), a ścieżka lub nazwa
wskazuje inne źródło. Opublikowany checkpoint ma 80 klas COCO. Żądanie go dla
modelu przebudowanego już do innej liczby klas najpierw przywraca 80 klas,
wczytuje checkpoint, a następnie po odczytaniu liczby klas zbioru danych
przenosi każdy tensor o zgodnym kształcie do docelowej liczby głowic.
`resume=True` nie można łączyć z `pretrained`. Przy domyślnej wartości `None`
trenowanie jest kontynuowane od wag użytych do utworzenia modelu albo od losowej
inicjalizacji, jeśli nie wczytano żadnych.

Poza tym bez zmian konfiguracji trener wykonuje 300 epok z `lr0=0.01`, momentum
SGD 0.937, 3 epokami rozgrzewki oraz tym samym przypisywaniem SimOTA i końcową
fazą 15 epok bez augmentacji co YOLOX, dostosowanymi do głowicy opartej na
kotwicach. Jest jedna różnica: YOLOX dodaje udoskonalanie regresji ramek L1 w
tych ostatnich epokach, a v7 je pomija, ponieważ funkcja straty SimOTA v7 nie ma
gałęzi L1 surowego przesunięcia do udoskonalania.

Informacje o zbiorach danych, augmentacji, wielu GPU i loggerach zawiera strona
[trenowania](/docs/train).

## Walidacja

`val()` zwraca słownik kluczy `metrics/`, który obejmuje precision, recall,
mAP 50 i mAP 50-95, zmierzone względem dowolnego zbioru danych w formacie użytym
do trenowania.

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

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
