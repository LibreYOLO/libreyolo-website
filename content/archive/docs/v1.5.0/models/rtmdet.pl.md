---
title: RTMDet
families:
  - rtmdet
seo_title: 'RTMDet w LibreYOLO: predykcja, trenowanie i eksport'
description: >-
  Uruchamiaj RTMDet w LibreYOLO do detekcji obiektów oraz RTMDet-Ins do
  segmentacji instancji. Instalacja, predykcja, trenowanie, walidacja i eksport
  na licencji Apache-2.0.
lead: >-
  RTMDet to jednoetapowy detektor, który wykonuje predykcję na podstawie jednego
  punktowego wzorca na położenie siatki, bez kotwic, przez głowicę ze
  współdzielonymi konwolucjami między poziomami cech. LibreYOLO obsługuje go do
  detekcji oraz segmentacji instancji RTMDet-Ins.
keywords:
  - RTMDet
  - detekcja obiektów
  - segmentacja instancji
  - RTMDet-Ins
  - detekcja bez kotwic
  - mmdetection
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRTMDets.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Segmentacja instancji
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Przyrostek -seg w nazwie pliku wybiera głowicę masek RTMDet-Ins,
        # dlatego argument task nie jest tutaj potrzebny.
        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTMDets.pt data=my-dataset.yaml
    - label: Segmentacja instancji
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # maski
        print(metrics["metrics/mAP50-95(B)"])   # ramki
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, imgsz=640, batch=16, lr0=0.004,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreRTMDets.pt data=my-dataset.yaml imgsz=640
        epochs=300 batch=16 lr0=0.004
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRTMDets.pt format=onnx imgsz=640

        libreyolo export model=LibreRTMDets.pt format=tensorrt imgsz=640
        half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, dlatego

        # artefakt ładuje się jak punkt kontrolny i zwraca ten sam obiekt
        Results.

        model = LibreYOLO("LibreRTMDets.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 2f5033bdc1c3c931
---

## Instalacja

RTMDet nie wymaga niczego poza pakietem podstawowym.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej pamięci podręcznej.

<code-tabs name="predict" />

Zwracany obiekt `Results` jest taki sam dla każdej rodziny, więc zamiana na
inny detektor wymaga zmiany jednego wiersza. Nazwa pliku z przyrostkiem `-seg`
samoczynnie wybiera zadanie RTMDet-Ins, a `result.masks` zawiera wtedy maski
instancji obok ramek. Parametr `conf` ustawia próg pewności, a `iou` próg NMS.
Informacje o źródłach, strumieniowaniu i obsłudze wyników znajdziesz w sekcji
[predykcja](/docs/predict).

## Warianty

Pięć rozmiarów, od `t` do `x`, korzysta z jednej architektury i wspólnej
rozdzielczości wejściowej. Dla tej rodziny nie ma tutaj tabeli benchmarków.
Rozmiary można porównać według wielkości pliku punktu kontrolnego w tabeli poniżej.

## Trenowanie

<code-tabs name="train" />

Detekcja jest trenowana za pomocą `train()`. Komponenty QualityFocalLoss, GIoU
i DynamicSoftLabelAssigner przeniesiono z projektu źródłowego mmdetection.
Przebieg w przód oraz eksport ONNX są z nim bitowo równoważne, a przetwarzanie
końcowe odpowiada wynikom mmdet z dokładnością do 0,001 mAP na podzbiorach val2017.

Zgodnie z docstringiem samej metody `train()` nie sprawdzono zbieżności dostrajania
na małym zbiorze danych, zgodności trenowania od podstaw z publikacją, działania
z wieloma GPU, przepustowości buforowanych augmentacji Mosaic i MixUp, ścisłego
przełączenia dwuetapowego potoku źródłowego ani nadpisań zaniku wag według
parametrów, które zerują zanik dla parametrów normalizacji i bias.

RTMDet-Ins nie ma ścieżki trenowania. Wywołanie `train()` na punkcie kontrolnym
`-seg` albo z `task="segment"` zgłasza `NotImplementedError`. Segmentacja instancji
obsługuje wyłącznie wnioskowanie i walidację.

Metoda `train()` przyjmuje także argument `pretrained`, ale jego wartość nie jest
nigdzie odczytywana wewnątrz metody: trenowanie zawsze jest kontynuowane z wag,
z którymi utworzono model, więc `pretrained=False` nie inicjalizuje sieci ponownie.

Przy pozostałych ustawieniach domyślnych trener wykonuje 300 epok z optymalizatorem
AdamW, `lr0=0.004` i `weight_decay=0.05`, jedną epoką rozgrzewki w harmonogramie
cosinusowym oraz augmentacjami Mosaic i MixUp wyłączonymi na ostatnie 20 epok.

Informacje o zbiorach danych, augmentacji, wielu GPU i loggerach znajdziesz w sekcji [trenowanie](/docs/train).

## Walidacja

Metoda `val()` zwraca słownik kluczy `metrics/` obejmujących precyzję, czułość,
mAP 50 i mAP 50-95, mierzone na dowolnym zbiorze danych w formacie użytym do trenowania.

<code-tabs name="val" />

W przypadku punktu kontrolnego `-seg` zwykły klucz `metrics/mAP50-95` zawiera
wynik masek, a ten sam przebieg raportuje także ramki pod `(B)` i maski pod `(M)`,
dzięki czemu oba wyniki są dostępne po jednym przebiegu.

## Eksport

<export-matrix />

Detekcję można wyeksportować do większości formatów. Segmentacji instancji nie
można obecnie wyeksportować do żadnego z nich, co odzwierciedla powyższa macierz.
Wyeksportowany artefakt detekcji jest ponownie ładowany przez `LibreYOLO()` na
podstawie rozszerzenia pliku, więc plik `.onnx` lub `.engine` zachowuje się jak
punkt kontrolny i zwraca ten sam obiekt `Results`. Obsługiwane jest także
uruchamianie grafu w samym środowisku uruchomieniowym bez zainstalowanego
LibreYOLO, ale wówczas samodzielnie trzeba zaimplementować przetwarzanie wstępne i końcowe.

<code-tabs name="export" />

## Punkty kontrolne

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box></provenance-box>

## Cytowanie

<citation-block />

