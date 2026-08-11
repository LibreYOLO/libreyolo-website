---
title: SSD
families:
  - ssd
seo_title: 'SSD (SSD300): detekcja obiektów w LibreYOLO'
description: >-
  Uruchamiaj SSD300 w LibreYOLO: jednostrzałowy detektor VGG16 do predykcji,
  walidacji i eksportu ONNX na licencji BSD-3-Clause. Bez ścieżki trenowania.
lead: >-
  SSD (Single Shot MultiBox Detector) przewiduje każdą ramkę i wynik klasy z
  gęstej siatki domyślnych ramek w jednym przebiegu w przód, bez osobnego etapu
  propozycji regionów. LibreYOLO udostępnia checkpoint SSD300 z backbone VGG16
  jako detektor wyłącznie do inferencji.
keywords:
  - SSD
  - SSD300
  - Single Shot MultiBox Detector
  - detekcja obiektów
  - VGG16
  - detektor oparty na kotwicach
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSSD300.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSSD300.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSSD300.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSSD300.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSSD300.pt")


        # imgsz celowo pominięto: SSD300 jest śledzony na natywnym obszarze
        checkpointu,

        # a każda inna wartość zgłasza błąd przed rozpoczęciem eksportu.

        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSSD300.pt format=onnx
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreSSD300.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 3b3f9ea72291c4fa
---

## Instalacja

SSD nie wymaga żadnego opcjonalnego dodatku. Wszystkie importowane elementy
znajdują się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

Zwracany obiekt `Results` jest taki sam dla każdej rodziny, więc zmiana
detektora wymaga zmiany jednego wiersza. SSD dekoduje siatkę domyślnych ramek z
wynikami dla każdej klasy, a następnie uruchamia tłumienie niemaksymalne.
Argumenty `conf`, `iou` i `max_det` mają tu zatem rzeczywisty wpływ, inaczej niż
w detektorach opartych na zapytaniach w tej bibliotece. Więcej informacji o
źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Warianty

SSD udostępnia jeden checkpoint: sieć SSD300 z backbone VGG16 na jej stałym
natywnym obszarze. Ta rodzina nie oferuje wyboru rozmiaru ani skali. Predykcja,
walidacja i eksport używają tego jednego grafu.

Plik wag nazywa się `LibreSSD300.pt`, czyli prefiks rodziny z jej jedynym
kluczem rozmiaru `"300"`. Odpowiada mu klasa `LibreSSD`, dlatego bezpośrednie
utworzenie ma postać `LibreSSD(size="300")`, a nie klasy nazwanej jak plik.

## Walidacja

`val()` zwraca słownik kluczy `metrics/`, który obejmuje precision, recall,
mAP 50 i mAP 50-95, zmierzone względem dowolnego zbioru danych w formacie użytym
do trenowania.

<code-tabs name="val" />

## Eksport

<export-matrix />

SSD eksportuje wyłącznie do ONNX. Wszystkie inne formaty są obecnie blokowane
dla tej rodziny. Eksport zawsze używa natywnego obszaru checkpointu, a graf
udostępnia surową, spakowaną głowicę SSD zamiast wyjścia ze scalonym tłumieniem
niemaksymalnym, dlatego `nms=True` nie jest przyjmowane podczas eksportu. Własne
backendy LibreYOLO wykonują etap dekodowania i tłumienia po ponownym wczytaniu
grafu.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box>

Kod SSD300 w LibreYOLO nie został przeniesiony z własnego wydania Caffe autorów
publikacji. Pochodzi z implementacji SSD300 w torchvision na licencji
BSD-3-Clause i to repozytorium jest wskazane powyżej jako źródło. Wagi backbone
VGG16 wywodzą się z w pełni konwolucyjnego, zredukowanego VGGNet grupy Oxford,
wydanego przez Karen Simonyan i Andrew Zisserman na licencji CC BY 4.0.

</provenance-box>

## Cytowanie

<citation-block />
