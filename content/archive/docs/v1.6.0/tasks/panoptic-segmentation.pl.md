---
title: Segmentacja panoptyczna
seo_title: Segmentacja panoptyczna w LibreYOLO
description: >-
  Przypisz każdy piksel do jednego segmentu w LibreYOLO: poznaj rodziny
  obsługujące to zadanie, format zbioru danych COCO-panoptic oraz wywołania
  predykcji i walidacji.
lead: >-
  Segmentacja panoptyczna przypisuje każdy piksel do dokładnie jednego
  nienakładającego się segmentu, łącząc policzalne instancje obiektów z
  bezkształtnymi obszarami tła. Klucz zadania to panoptic.
keywords:
  - segmentacja panoptyczna python
  - panoptic quality
  - segmentacja things stuff
  - format COCO panoptic
  - mapa identyfikatorów segmentów
  - metryka PQ
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Sufiks -panoptic w nazwie pliku wybiera zadanie, dlatego argument
        # task nie jest potrzebny.
        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # identyfikatory segmentów (H, W)
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreEoMTl-panoptic.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Jeden segment naraz
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreEoMTl-panoptic.pt")(SAMPLE_IMAGE)
        pan = result.panoptic

        for segment in pan.segments_info:
            pixels = pan.segment_mask(segment["id"])   # wartości logiczne (H, W)
            print(result.names[segment["category_id"]], int(pixels.sum()))
    - label: Mniejszy checkpoint
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTs-panoptic.pt")
        result = model(SAMPLE_IMAGE)

        print(len(result.panoptic.segment_ids))
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")

        # val() zwraca zwykły słownik, a nie obiekt.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
        print(metrics["metrics/PQ_things"], metrics["metrics/PQ_stuff"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-panoptic.pt data=my-dataset.yaml
source_hash: b8adc9ccde7a4e6c
---

## Definicja

Segmentacja panoptyczna łączy dwa pozostałe zadania segmentacji. Każdy piksel
otrzymuje dokładnie jeden segment, segmenty nigdy się nie nakładają, a segment
jest obiektem, czyli policzalną instancją, albo tłem, czyli bezkształtnym
obszarem takim jak niebo lub droga. Jest więc bardziej rygorystyczna niż
[segmentacja instancji](/docs/tasks/instance-segmentation), która pozostawia
nieprzypisane piksele tła i pozwala maskom się nakładać, oraz bardziej
rygorystyczna niż [segmentacja semantyczna](/docs/tasks/semantic-segmentation),
która etykietuje każdy piksel, ale łączy stykające się instancje tej samej klasy.

`panoptic` jest kanonicznym kluczem zadania, a sufiks `-panoptic` w nazwie pliku
checkpointu wybiera to zadanie, więc `task=` nie jest potrzebne przy wczytywaniu
opublikowanych wag.

`predict()` wypełnia `result.panoptic`. Pole `.data` jest całkowitoliczbową mapą
identyfikatorów segmentów `(H, W)` na obszarze roboczym oryginalnego obrazu.
Pole `.segments_info` jest listą słowników, po jednym na segment, z których każdy
zawiera co najmniej `{"id", "category_id"}`. Wartość `id` odpowiada wartości na
mapie, a `category_id` indeksuje `result.names`. Pole `.segment_ids` zawiera
posortowaną listę obecnych identyfikatorów, a `.segment_mask(id)` zwraca logiczne
zaznaczenie `(H, W)` jednego segmentu. Identyfikator segmentu `0` oznacza obszar
pusty: nieoznaczone piksele wyłączone z metryki i pomijane w `.segment_ids`.

Rozróżnienie obiektu i tła jest właściwością kategorii, a nie pojedynczego
segmentu. Znajduje się w metadanych kategorii zestawu etykiet. Dla wygody dane
predykcji mogą skopiować je do każdego segmentu jako `"isthing"`, ale metadane
kategorii pozostają źródłem nadrzędnym.

## Modele

[EoMT](/docs/models/eomt) jest rodziną obsługującą to zadanie przez
`LibreYOLO()`. Działa z pakietem podstawowym i udostępnia checkpointy
panoptyczne w trzech rozmiarach: s, b i l, wytrenowane na zbiorze COCO.

[SenseNova-Vision](/docs/models/sensenova-vision) również generuje mapy
panoptyczne. Jest to generatywny model sterowany promptem, z własną fabryką
`LibreVLM` i własnym dodatkiem. Jeśli słownik nie jest ustawiony, wraca do
kategorii panoptycznych COCO, na których został dostrojony. Jego wagi są
przeznaczone do użytku niekomercyjnego. Opóźnienie dla pojedynczego obrazu jest
znacznie większe niż w wyspecjalizowanym segmentatorze, ponieważ każda
predykcja jest dekodowaniem dyfuzyjnym.

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane lokalnie w
pamięci podręcznej.

<code-tabs name="predict" />

`conf` filtruje wybór zapytań. Informacje o źródłach, streamingu i obsłudze
wyników znajdują się w sekcji [predykcja](/docs/predict).

## Format zbioru danych

LibreYOLO przyjmuje format COCO-panoptic bez zmian, zgodnie z pracą Kirillova i
współautorów, CVPR 2019. Nie istnieje układ panoptyczny specyficzny dla
LibreYOLO.

```text
dataset/
  data.yaml
  images/
    val/000000000139.jpg
  annotations/
    panoptic_val.json
    panoptic_val/000000000139.png
```

Każdy obraz jest sparowany z jednym plikiem RGB PNG o tej samej rozdzielczości,
w którym kolor każdego piksela koduje identyfikator segmentu, do którego należy:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

Identyfikator segmentu `0`, czyli czarny RGB, oznacza obszar pusty. Są to
nieoznaczone piksele, które ani nie poprawiają, ani nie pogarszają wyniku
predykcji. Każdy inny piksel należy do dokładnie jednego segmentu.

Plik JSON zawiera dla każdego obrazu plik PNG z identyfikatorami segmentów oraz
segmenty znajdujące się w tym pliku:

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1}]
}
```

`annotations[].file_name` wskazuje plik PNG w katalogu panoptycznym, a
`segments_info[].id` odpowiada wartości w tym pliku PNG. Pole `iscrowd` oznacza
obszary grupowe. Nigdy nie są one liczone jako fałszywie ujemne, a predykcja,
która w większości pokrywa taki obszar, nie jest fałszywie dodatnia. Pole
`isthing` znajduje się w `categories`, nigdy w pojedynczym segmencie.

Plik YAML wskazuje oba elementy:

```yaml
path: dataset
val: images/val
annotations:
  val: annotations/panoptic_val.json
panoptic_dir:
  val: annotations/panoptic_val
names:
  0: person
  1: bicycle
```

Pola `annotations` i `panoptic_dir` przyjmują pojedynczą ścieżkę lub mapowanie
ścieżek dla poszczególnych podziałów. Surowe identyfikatory kategorii COCO są
zwykle nieciągłe, podczas gdy modele przewidują ciągły zakres `0..nc-1`, dlatego
identyfikatory są mapowane przez `names` na podstawie nazwy kategorii. Brak
kategorii JSON w `names` powoduje błąd, a nie ciche pominięcie, ponieważ jej
usunięcie byłoby oceniane jako trwały wynik fałszywie ujemny.

Kanoniczny moduł wczytujący to `libreyolo.data.PanopticDataset`.

## Trenowanie

Obecnie żadna rodzina nie trenuje segmentacji panoptycznej w LibreYOLO. Metoda
`train()` rodziny EoMT zgłasza `NotImplementedError`, dlatego checkpointów
panoptycznych używa się w opublikowanej postaci.

## Walidacja

`val()` zwraca zwykły słownik kluczy `metrics/`, obliczanych w rozdzielczości
danych referencyjnych dla podziału wskazanego przez `val` w pliku YAML zbioru
danych. Przewidziany i prawdziwy segment tej samej kategorii są dopasowywane,
gdy ich IoU przekracza 0.5, a takie dopasowanie jest unikatowe.

<code-tabs name="val" />

`metrics/PQ` oznacza Panoptic Quality i jest głównym wynikiem. W obrębie jednej
kategorii jest iloczynem dwóch czynników. Jakość segmentacji to średnie IoU dla
dopasowanych segmentów, które określa zgodność dopasowanych kształtów. Jakość
rozpoznawania to `TP / (TP + 0.5 FP + 0.5 FN)`, czyli wynik F1 samego
dopasowania, określający, ile segmentów w ogóle znaleziono. Następnie wszystkie
trzy wartości są uśredniane dla występujących kategorii i raportowane jako
`metrics/PQ`, `metrics/SQ` oraz `metrics/RQ`. Raportowane PQ jest więc średnią
iloczynów dla poszczególnych kategorii, a nie iloczynem dwóch raportowanych
średnich.

`metrics/PQ_things` i `metrics/PQ_stuff` uśredniają te same wartości PQ dla
poszczególnych kategorii, oddzielnie dla kategorii obiektów i tła, a
`metrics/categories` zlicza kategorie, które wystąpiły i zostały uwzględnione w
średniej. Słownik zawiera również `fitness`, czyli kopię wartości PQ.

## Eksport

Checkpointów panoptycznych nie można eksportować. `export()` zgłasza
`NotImplementedError` dla tego zadania, ponieważ dane wyjściowe masek zapytań
nie mają jeszcze kontraktu eksportu do środowiska uruchomieniowego. Zadanie
semantyczne EoMT obsługuje eksport. Zobacz
[segmentację semantyczną](/docs/tasks/semantic-segmentation) oraz
[eksport i wdrożenie](/docs/export).

