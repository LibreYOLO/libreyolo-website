---
title: RF-DETR
families:
  - rfdetr
seo_title: 'RF-DETR: trenowanie, dostrajanie i eksport na licencji MIT'
description: >-
  Używaj modelu RF-DETR w LibreYOLO do detekcji, segmentacji instancji,
  estymacji pozy i ramek zorientowanych. Instalacja, predykcja, trenowanie,
  walidacja i eksport, wszystko na licencji MIT.
lead: >-
  Transformer detekcyjny, który przewiduje stały zestaw obiektów zamiast gęstej
  siatki, dlatego podczas wnioskowania nie wymaga NMS. LibreYOLO obsługuje go w
  czterech zadaniach.
keywords:
  - RF-DETR
  - transformer detekcyjny czasu rzeczywistego
  - DETR
  - detekcja obiektów
  - segmentacja instancji
  - estymacja pozy
  - zorientowane ramki ograniczające
last_verified: 1.5.0
hero:
  src: /showcase/parkour-detection.mp4
  poster: /showcase/parkour-detection-poster.jpg
  caption: 'LibreRFDETRs, detekcja na wideo w rozdzielczości 512 px.'
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRFDETRs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Wideo
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")


        # Dowolne źródło akceptowane przez bibliotekę: plik, folder, URL, indeks
        kamery,

        # strumień RTSP albo lista .streams

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Wiele GPU
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # val() zwraca zwykły słownik, a nie obiekt
        metrics = model.val(data="my-dataset.yaml", imgsz=512)

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs.pt data=my-dataset.yaml imgsz=512
    - label: Walidacja na COCO
      language: bash
      code: >
        # Dołączony plik yaml COCO zawiera osadzony skrypt pobierania, dlatego
        wymaga

        # jawnego zezwolenia, chyba że zbiór danych jest już dostępny lokalnie.

        libreyolo val model=LibreRFDETRn.pt data=coco.yaml imgsz=384 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")

        model.export(format="onnx", imgsz=512)

        model.export(format="tensorrt", imgsz=512, half=True)


        # Argumenty akceptowane dla każdego formatu:

        #

        #   format    "onnx" | "torchscript" | "executorch" | "tensorrt"

        #             | "openvino" | "paddle" | "mnn" | "rknn" | "ncnn"

        #             | "tflite" | "coreml" | "coreai".

        #             "engine" jest aliasem tensorrt, a "litert" aliasem tflite.

        #   imgsz     int albo (wysokość, szerokość). Domyślnie natywna

        #             rozdzielczość punktu kontrolnego.

        #   batch     int, domyślnie 1.

        #   half      bool, eksport w FP16. Domyślnie False.

        #   int8      bool, eksport w INT8. Domyślnie False. Wymaga `data`.

        #   data      ścieżka do pliku YAML zbioru danych używanego do
        kalibracji int8.

        #   fraction  float, część zestawu kalibracyjnego do użycia. Domyślnie
        1.0.

        #   dynamic   bool, osie dynamiczne. Domyślnie True.

        #   simplify  bool, uproszczenie grafu ONNX. Domyślnie True.

        #   opset     int, zestaw operatorów ONNX. Jeśli nie podano, wybierany
        dla rodziny.

        #   device    str, urządzenie do śledzenia. Domyślnie urządzenie modelu.

        #   output_path  str, domyślnie nazwa pochodząca od punktu kontrolnego.

        #   verbose   bool, domyślnie False.

        #   allow_download_scripts  bool, domyślnie False. Zezwala na osadzony

        #             Python w pliku YAML zbioru danych, który trzeba pobrać.

        #

        # Kilka formatów przyjmuje własne dodatkowe argumenty, na przykład
        platformę

        # docelową RKNN. Są one opisane na stronie każdego formatu.
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRFDETRs.pt format=onnx imgsz=512

        libreyolo export model=LibreRFDETRs.pt format=tensorrt imgsz=512
        half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, dlatego

        # artefakt ładuje się jak punkt kontrolny i zwraca ten sam obiekt
        Results.

        model = LibreYOLO("LibreRFDETRs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
    - label: Bez LibreYOLO
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # Bezpośrednie uruchomienie grafu wymaga własnego przetwarzania
        wstępnego

        # i końcowego. Przed integracją sprawdź sygnaturę.

        session = ort.InferenceSession("LibreRFDETRs.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 512, 512),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 8c464aa759131694
---

## Instalacja

RF-DETR wymaga własnego dodatku, który instaluje `transformers` dla backbone.

```bash
pip install "libreyolo[rfdetr]"
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej pamięci podręcznej.

<code-tabs name="predict" />

Zwracany obiekt `Results` jest taki sam dla każdej rodziny, więc zamiana na
inny detektor wymaga zmiany jednego wiersza. Parametry `conf` i `max_det`
filtrują wybór zapytań. Nie ma etapu NMS do dostrojenia. Informacje o źródłach,
strumieniowaniu i obsłudze wyników znajdziesz w sekcji [predykcja](/docs/predict).

## Warianty

Dostępne są cztery rozmiary oraz cztery zadania korzystające ze wspólnej
architektury. Segmentacja, estymacja pozy i ramki zorientowane ponownie używają
dekodera detekcji z inną głowicą, dlatego przyjmują te same argumenty. Rozmiary
mają podobną liczbę parametrów i różnią się głównie rozdzielczością wejściową.

<benchmark-table task="detect" />

<va-embed />

## Trenowanie

Trenowanie dla wszystkich czterech zadań zaczyna się od opublikowanego punktu
kontrolnego. RF-DETR umieszcza `pretrained` wśród argumentów ignorowanych przez
natywny trener, więc przekazanie `pretrained=False` nie daje tutaj losowo
zainicjalizowanego modelu.

<code-tabs name="train" />

Dwa argumenty mają tutaj większe znaczenie niż w detektorze CNN. Utrzymuj `lr0`
na poziomie `1e-4` lub niższym, ponieważ detektory transformer rozbiegają się przy
szybkościach uczenia tolerowanych przez model YOLO. Pozostaw `imgsz` w natywnej
rozdzielczości punktu kontrolnego, chyba że istnieje powód, by ją zmienić. Rozmiar
wejścia musi być podzielny przez iloczyn rozmiaru fragmentu backbone i liczby
okien. LibreYOLO sprawdza to przed rozpoczęciem przebiegu i wskazuje najbliższe
prawidłowe rozmiary.

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

<provenance-box></provenance-box>

## Cytowanie

<citation-block />

