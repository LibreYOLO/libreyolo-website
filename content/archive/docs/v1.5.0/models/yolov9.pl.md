---
title: YOLOv9
families:
  - yolo9
seo_title: 'YOLOv9: predykcja, trenowanie i eksport na licencji MIT'
description: >-
  Uruchamiaj YOLOv9 w LibreYOLO, w tym kompleksową głowicę bez NMS oraz głowicę
  do małych obiektów z krokiem 4. Instalacja, predykcja, trenowanie, walidacja i
  eksport.
lead: >-
  Jednoetapowy detektor konwolucyjny: jeden przebieg ocenia gęstą siatkę ramek,
  a NMS usuwa duplikaty. LibreYOLO zawiera trzy jego warianty, z których jeden
  nie korzysta z NMS.
keywords:
  - YOLOv9
  - YOLO9
  - detekcja obiektów
  - detekcja bez NMS
  - detekcja end-to-end
  - detekcja małych obiektów
  - programowalna informacja gradientowa
  - GELAN
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Bez NMS
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # To samo wywołanie, inny punkt kontrolny. Głowica end-to-end zwraca
        własne

        # predykcje z najwyższymi wynikami, więc NMS nie jest wykonywane, a iou
        jest ignorowane.

        model = LibreYOLO("LibreYOLO9E2Es.pt")

        result = model(SAMPLE_IMAGE, conf=0.25, max_det=300)


        print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: Małe obiekty
      language: python
      code: >
        from libreyolo import LibreYOLO9P2


        # Wariant z krokiem 4 nie ma własnego punktu kontrolnego COCO, dlatego
        należy

        # wskazać bazowy punkt kontrolny detekcji: backbone i neck zostaną
        załadowane

        # bez zmian, a wieża głowicy z krokiem 4 rozpocznie od losowej
        inicjalizacji.

        model = LibreYOLO9P2(None, size="s")

        model.train(data="my-dataset.yaml", epochs=100,
        pretrained="LibreYOLO9s.pt")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=my-dataset.yaml
    - label: Walidacja na COCO
      language: bash
      code: >
        # Dołączony plik yaml COCO zawiera osadzony skrypt pobierania, dlatego
        wymaga

        # jawnego zezwolenia, chyba że zbiór danych jest już dostępny lokalnie.

        libreyolo val model=LibreYOLO9c.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: Z NMS w grafie
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx nms=True \
          conf=0.25 iou=0.45 max_det=300
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, dlatego

        # artefakt ładuje się jak punkt kontrolny i zwraca ten sam obiekt
        Results.

        model = LibreYOLO("LibreYOLO9s.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: eaa6023a4a0b9e71
---

## Instalacja

YOLOv9 nie wymaga niczego poza pakietem podstawowym.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej pamięci podręcznej.

<code-tabs name="predict" />

Zwracany obiekt `Results` jest taki sam dla każdej rodziny, więc zamiana na
inny detektor wymaga zmiany jednego wiersza. W modelach bazowym i z krokiem 4
parametr `conf` ustawia próg pewności, a `iou` próg NMS. Model end-to-end nie
wykonuje NMS i ignoruje `iou`, dlatego jego wynik kształtują `conf` i `max_det`.
Informacje o źródłach, strumieniowaniu i obsłudze wyników znajdziesz w sekcji
[predykcja](/docs/predict).

## Warianty

Trzy warianty korzystają ze wspólnego backbone. Wszystkie trzy służą wyłącznie
do detekcji i przyjmują te same argumenty.

Model bazowy wykonuje predykcję na trzech skalach cech i usuwa zduplikowane ramki za pomocą NMS.

Model end-to-end zachowuje tę głowicę i dodaje obok niej gałąź dopasowania jeden
do jednego. Wnioskowanie odczytuje wyłącznie gałąź jeden do jednego i wybiera
jej predykcje z najwyższymi wynikami, dlatego NMS nie jest wykonywane. Wybierz
ten model, gdy docelowe środowisko uruchomieniowe nie ma operatora NMS.

Model z krokiem 4 udostępnia jeden poziom wcześniej w backbone, rozszerza do
niego moduł neck i wykonuje predykcję na czterech skalach zamiast trzech.
Dodatkowa skala jest przeznaczona dla obiektów obejmujących niewiele pikseli.
Jedyny opublikowany dla niej punkt kontrolny wytrenowano na obrazach lotniczych.
Bazowe punkty kontrolne detekcji można do niego przenieść: backbone i neck są
ładowane bez zmian, trzy wstępnie wytrenowane wieże głowicy przesuwają się o
jedną pozycję, a wieża z krokiem 4 zaczyna od losowej inicjalizacji.

<benchmark-table task="detect" />

<va-embed />

## Trenowanie

<code-tabs name="train" />

Parametr `pretrained` określa punkt początkowy przebiegu. Przekaż `True`, aby
załadować opublikowany punkt kontrolny tego samego modelu i rozmiaru, albo nazwę
lub ścieżkę do dowolnego innego. Tensory o niedopasowanym kształcie są pomijane,
a nie odrzucane, a przebieg zapisuje liczbę załadowanych tensorów. Dzięki temu
punkt kontrolny wytrenowany z inną liczbą klas nadal nadaje się na punkt początkowy.

Model z krokiem 4 nie ma własnego opublikowanego punktu kontrolnego COCO, dlatego
`True` wskazuje w tym przypadku nieistniejący plik, a pobieranie kończy się
niepowodzeniem. Zamiast tego wskaż bazowy punkt kontrolny detekcji.

Informacje o zbiorach danych, augmentacji, wielu GPU i loggerach znajdziesz w sekcji [trenowanie](/docs/train).

## Walidacja

Metoda `val()` zwraca słownik kluczy `metrics/` obejmujących precyzję, czułość,
mAP 50 i mAP 50-95, mierzone na dowolnym zbiorze danych w formacie użytym do trenowania.

<code-tabs name="val" />

## Eksport

<export-matrix />

Znacznik wyboru dotyczy wszystkich trzech wariantów. Jeśli ich obsługa formatu
się różni, macierz podaje najsłabszy poziom spośród trzech.

Wyeksportowany artefakt jest ponownie ładowany przez `LibreYOLO()` na podstawie
rozszerzenia pliku, więc plik `.onnx` lub `.engine` zachowuje się jak punkt
kontrolny i zwraca ten sam obiekt `Results`. Obsługiwane jest także uruchamianie
grafu w samym środowisku uruchomieniowym bez zainstalowanego LibreYOLO, ale wówczas
samodzielnie trzeba zaimplementować przetwarzanie wstępne i końcowe.

Dla bazowego modelu detekcji część przetwarzania końcowego można przenieść do
grafu. Ustawienie `nms=True` podczas eksportu ONNX umieszcza tłumienie wewnątrz
modelu, a pierwszy wynik staje się stałym tensorem `(1, max_det, 6)`, którego
wiersze mają postać `x1, y1, x2, y2, score, class` i są dopełniane zerami po
liczbie detekcji. Taki graf ma partię 1 i nie zawiera osi dynamicznych. Modele
end-to-end i z krokiem 4 nie przyjmują tej flagi.

Każdy format instaluje inny dodatek i przyjmuje kilka własnych argumentów.
Obie informacje znajdują się na stronie danego formatu.

<code-tabs name="export" />

## Punkty kontrolne

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box>

Jeden punkt kontrolny nie podlega tutaj licencji MIT. Model z krokiem 4 wytrenowany
na VisDrone2019-DET dziedziczy warunki CC BY-NC-SA 3.0 tego zbioru danych: wyłącznie
użytek niekomercyjny, udostępnianie utworów pochodnych na tych samych zasadach
i wyłączenie z liberalnej licencji obejmującej resztę tej rodziny. Przewiduje
klasy lotnicze VisDrone zamiast klas COCO. Biblioteka wyświetla wszystkie te
informacje przed pobraniem pliku.

</provenance-box>

## Cytowanie

<citation-block />

