---
title: PicoDet
families:
  - picodet
seo_title: 'PicoDet w LibreYOLO: predykcja, trenowanie i eksport'
description: >-
  Uruchamiaj PicoDet w LibreYOLO do mobilnej detekcji obiektów. Instalacja,
  predykcja, trenowanie, walidacja i eksport na licencji Apache-2.0.
lead: >-
  PicoDet to jednoetapowy detektor przeznaczony do procesorów urządzeń mobilnych
  i brzegowych: backbone ESNet, moduł neck CSP-PAN oraz współdzielona głowica
  Generalized Focal Loss. LibreYOLO obsługuje go do detekcji.
keywords:
  - PicoDet
  - PP-PicoDet
  - detekcja obiektów
  - detekcja obiektów mobilna
  - detekcja na urządzeniach brzegowych
  - ESNet
  - Generalized Focal Loss
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePICODETs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePICODETs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePICODETs.pt data=my-dataset.yaml
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, batch=16, lr0=0.01,
        )
    - label: CLI
      language: bash
      code: >
        # Warto ustawić imgsz: interfejs CLI domyślnie używa 640, podczas gdy

        # natywna wartość punktu kontrolnego s to 320.

        libreyolo train model=LibrePICODETs.pt data=my-dataset.yaml imgsz=320
        epochs=300 batch=16 lr0=0.01
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.export(format="onnx", imgsz=320)
        model.export(format="tensorrt", imgsz=320, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibrePICODETs.pt format=onnx imgsz=320

        libreyolo export model=LibrePICODETs.pt format=tensorrt imgsz=320
        half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, dlatego

        # artefakt ładuje się jak punkt kontrolny i zwraca ten sam obiekt
        Results.

        model = LibreYOLO("LibrePICODETs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 947aa47214abc4c0
---

## Instalacja

PicoDet nie wymaga niczego poza pakietem podstawowym.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej pamięci podręcznej.

<code-tabs name="predict" />

Zwracany obiekt `Results` jest taki sam dla każdej rodziny, więc zamiana na
inny detektor wymaga zmiany jednego wiersza. Parametr `conf` ustawia próg
pewności, a `iou` próg NMS. Informacje o źródłach, strumieniowaniu i obsłudze
wyników znajdziesz w sekcji [predykcja](/docs/predict).

## Warianty

Dostępne są trzy rozmiary, każdy z własną stałą rozdzielczością wejściową:
`s` jest najmniejszy, a `l` największy. Rozdzielczość rośnie wraz z rozmiarem,
więc większe punkty kontrolne nie tylko mają więcej parametrów, lecz również
wymagają więcej obliczeń na obraz.

<benchmark-table task="detect" />

<va-embed />

## Trenowanie

<code-tabs name="train" />

Składniki funkcji straty i metoda przypisywania odpowiadają konfiguracji
źródłowej: VFL, DFL, GIoU i SimOTA, z ważeniem jakością klasyfikacji oraz
dynamicznymi celami VFL opartymi na IoU. Wnioskowanie na tym samym punkcie
kontrolnym jest bitowo równoważne z projektem źródłowym.

Zgodnie z docstringiem samej metody `train()` nie sprawdzono pełnej zbieżności
na całym zbiorze danych, działania z wieloma GPU ani żadnej augmentacji poza
odbiciem poziomym. Punkt kontrolny `s` w natywnej rozdzielczości 320 również
nie przekraczał niezawodnie progu dokładności LibreYOLO na zestawie testowym
obejmującym 30 obrazów i dwie klasy, używanym przez bibliotekę do testowania
małych procesów dostrajania. Ten rozmiar lepiej pasuje do pełnej skali COCO.

Metoda `train()` przyjmuje także argument `pretrained`, ale jego wartość nie jest
nigdzie odczytywana wewnątrz metody: trenowanie zawsze jest kontynuowane z wag,
z którymi utworzono model, więc `pretrained=False` nie inicjalizuje sieci ponownie.
Jeśli `imgsz` nie zostanie ustawione w Pythonie, przyjmowana jest natywna
rozdzielczość załadowanego punktu kontrolnego: 320 dla `s`, 416 dla `m` i 640
dla `l`. Interfejs CLI zawsze przekazuje `imgsz`, domyślnie 640, dlatego należy
ustawić ten parametr tak, aby odpowiadał punktowi kontrolnemu.

Przy pozostałych ustawieniach domyślnych trener wykonuje 300 epok z optymalizatorem
SGD, `lr0=0.01`, pędem 0,9, zanikiem wag 4e-5 i jedną epoką rozgrzewki w
harmonogramie cosinusowym. Odbicie poziome jest jedyną stosowaną augmentacją.

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
grafu w samym środowisku uruchomieniowym bez zainstalowanego LibreYOLO, ale wówczas
samodzielnie trzeba zaimplementować przetwarzanie wstępne i końcowe.

<code-tabs name="export" />

## Punkty kontrolne

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box>

Port w LibreYOLO bazuje na Bo396543018/Picodet_Pytorch, implementacji
oryginalnego PP-PicoDet z PaddleDetection w PyTorch, z usuniętym mmcv i dokładnie
dopasowanymi wszystkimi aktywacjami. Dzięki temu punkty kontrolne PaddlePaddle
przekonwertowane za pomocą procesu Bo ładują się bez dryfu numerycznego. Oba
źródła podlegają tym samym warunkom Apache-2.0 co autorzy publikacji.

</provenance-box>

## Cytowanie

<citation-block />

