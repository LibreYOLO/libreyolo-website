---
title: YOLO-NAS
families:
  - yolonas
seo_title: 'YOLO-NAS: predykcja, trenowanie i eksport w LibreYOLO'
description: >-
  Używaj modelu YOLO-NAS w LibreYOLO do detekcji i estymacji pozy. Wagi Deci.AI
  są własnościowe i przeznaczone do użytku niekomercyjnego, a LibreYOLO nie
  publikuje żadnej z nich.
lead: >-
  Detektor konwolucyjny, którego backbone i moduł neck powstały w wyniku
  wyszukiwania architektury przez Deci.AI, zbudowany z bloków RepVGG świadomych
  kwantyzacji. Jego wagi należą do Deci.AI i są licencjonowane wyłącznie do
  użytku niekomercyjnego. LibreYOLO nie publikuje żadnej z nich.
keywords:
  - YOLO-NAS
  - YOLONAS
  - Deci AI
  - SuperGradients
  - detekcja obiektów
  - estymacja pozy
  - detektor świadomy kwantyzacji
  - AutoNAC
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Nazwa, której nie ma jeszcze na dysku, jest pobierana z CDN Deci.
        Przed

        # pobraniem wyświetlane są warunki licencji Deci; pobranie pliku je
        akceptuje.

        model = LibreYOLO("LibreYOLONASs.pt")

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLONASs.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Poza
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Przyrostek -pose wybiera głowicę pozy i jej własny zestaw wag.
        model = LibreYOLO("LibreYOLONASs-pose.pt")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLONASs.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: Od podstaw
      language: python
      code: |
        from libreyolo import LibreYOLONAS

        # Nie są używane punkty kontrolne Deci: model zaczyna od losowych wag,
        # więc wynik trenowania pochodzi wyłącznie z własnych danych.
        model = LibreYOLONAS(None, size="s")
        model.train(data="my-dataset.yaml", imgsz=640, batch=16)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLONASs.pt data=my-dataset.yaml
    - label: Walidacja na COCO
      language: bash
      code: >
        # Dołączony plik yaml COCO zawiera osadzony skrypt pobierania, dlatego
        wymaga

        # jawnego zezwolenia, chyba że zbiór danych jest już dostępny lokalnie.

        libreyolo val model=LibreYOLONASl.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLONASs.pt format=onnx imgsz=640
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, dlatego

        # artefakt ładuje się jak punkt kontrolny i zwraca ten sam obiekt
        Results.

        model = LibreYOLO("LibreYOLONASs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 47c30d6e44024ce7
---

## Instalacja

YOLO-NAS nie wymaga niczego poza pakietem podstawowym.

```bash
pip install libreyolo
```

## Predykcja

Punkt kontrolny, którego nazwy nie ma jeszcze na dysku, jest pobierany z
publicznego CDN Deci, a nie z organizacji LibreYOLO, która nie udostępnia
żadnych z tych wag. Przed rozpoczęciem transferu biblioteka raz na proces
wyświetla warunki licencji Deci, a przed otwarciem pobranego pliku jego SHA-256
jest porównywane z przypiętą wartością. Dozwolone zastosowania opisano w sekcji
[licencjonowanie](#licensing).

<code-tabs name="predict" />

Zwracany obiekt `Results` jest taki sam dla każdej rodziny, więc zamiana na
inny detektor wymaga zmiany jednego wiersza. Parametr `conf` ustawia próg
pewności, a `iou` próg NMS. Informacje o źródłach, strumieniowaniu i obsłudze
wyników znajdziesz w sekcji [predykcja](/docs/predict).

## Warianty

Detekcja i estymacja pozy korzystają z tej samej architektury z różnymi głowicami
i przyjmują te same argumenty. Rozmiary w poniższej tabeli dotyczą detekcji.
Estymacja pozy jest dostępna w tych rozmiarach oraz w jednym mniejszym. Głowica
pozy przewiduje zestaw punktów kluczowych COCO.

<benchmark-table task="detect" />

<va-embed />

## Trenowanie

<code-tabs name="train" />

Gdy `epochs`, `lr0` i `amp` nie zostaną podane, są ustalane osobno dla każdego
zadania, więc przebieg estymacji pozy rozpoczyna się z innymi wartościami
domyślnymi niż przebieg detekcji. Domyślnym optymalizatorem jest AdamW. Liczba
klas pochodzi z pliku YAML zbioru danych, a głowica jest przebudowywana przed
pierwszą epoką. W głowicy pozy liczba punktów kluczowych jest obsługiwana tak samo,
więc punkt kontrolny pozy COCO można dostroić do szkieletu o innej liczbie punktów.

Dostrajanie zaczyna się od wag Deci, których dotyczy licencja Deci. Trenowanie
losowo zainicjalizowanego modelu nie korzysta z żadnego punktu kontrolnego Deci,
jak pokazuje trzeci fragment powyżej.

Informacje o zbiorach danych, augmentacji, wielu GPU i loggerach znajdziesz w sekcji [trenowanie](/docs/train).

## Walidacja

Metoda `val()` zwraca słownik kluczy `metrics/` obejmujących precyzję, czułość,
mAP 50 i mAP 50-95, mierzone na dowolnym zbiorze danych w formacie użytym do trenowania.

<code-tabs name="val" />

## Eksport

<export-matrix />

Wyeksportowany artefakt jest ponownie ładowany przez `LibreYOLO()` na podstawie
rozszerzenia pliku, więc plik `.onnx` lub `.engine` zachowuje się jak punkt
kontrolny i zwraca ten sam obiekt `Results`. Obsługiwane jest także uruchamianie
grafu w samym środowisku uruchomieniowym bez zainstalowanego LibreYOLO, ale
wówczas samodzielnie trzeba zaimplementować przetwarzanie wstępne i końcowe.
Każdy format instaluje inny dodatek i przyjmuje kilka własnych argumentów.
Obie informacje znajdują się na stronie danego formatu.

Eksport jest kolejną kopią tych samych wag w innym kontenerze. Eksportowanie
punktu kontrolnego Deci nie zmienia ani pochodzenia wag, ani obejmującej je licencji.

<code-tabs name="export" />

## Punkty kontrolne

Nie ma żadnych do wyświetlenia. Licencja Deci zabrania redystrybucji, dlatego
organizacja LibreYOLO nie publikuje żadnych wag YOLO-NAS, a pobieranie jest
kierowane gdzie indziej. Nazwa w postaci `LibreYOLONAS<size>.pt` albo
`LibreYOLONAS<size>-pose.pt` dla estymacji pozy jest mapowana na odpowiedni
obiekt w publicznym CDN Deci.

W ten sposób można pobierać tylko punkty kontrolne, których SHA-256 jest
przypięte w bibliotece. Każdy inny przypadek jest bezpiecznie odrzucany zamiast
otwierania niezweryfikowanego pliku pickle innej firmy. Taki plik trzeba pobrać
ręcznie i przekazać jako ścieżkę. Plik znajdujący się już na dysku jest ładowany
ze swojej ścieżki, bez pobierania i kontroli sumy. Dotyczy to również pliku `.pth`
Deci pod oryginalną nazwą, który jest rozpoznawany przez moduł ładujący.

## Licencjonowanie

<provenance-box>

LibreYOLO nie udostępnia ani nie tworzy kopii lustrzanych tych wag. W organizacji
LibreYOLO w Hugging Face nie ma niczego dla tej rodziny. Każde automatyczne
pobieranie jest zamiast tego kierowane do publicznego CDN Deci, przed rozpoczęciem
raz na proces wyświetla warunki Deci, a przed otwarciem pliku sprawdza go względem
przypiętej sumy SHA-256.

Alternatywą jest trenowanie losowo zainicjalizowanego modelu. Architektura ma
licencję Apache-2.0 w projekcie źródłowym i MIT tutaj, więc model wytrenowany w
ten sposób na własnych danych nie pochodzi z żadnego punktu kontrolnego Deci.

</provenance-box>

## Cytowanie

YOLO-NAS został opublikowany bez artykułu naukowego. Poniższa pozycja jest tą,
o której cytowanie proszą jego autorzy, i dotyczy SuperGradients, biblioteki,
w której model został wydany.

<citation-block />

