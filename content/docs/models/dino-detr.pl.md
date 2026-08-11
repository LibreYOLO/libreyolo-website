---
title: DINO-DETR
families:
  - dinodetr
seo_title: 'DINO-DETR: przewiduj i eksportuj pod Apache-2.0'
description: >-
  Uruchom DINO-DETR w LibreYOLO do wykrywania obiektów. Zainstaluj, przewiduj,
  zweryfikuj i eksportuj trzy rozmiary kotwic odszumiających, wszystkie
  licencjonowane przez Apache-2.0.
lead: >-
  DINO-DETR, opublikowany przez IDEA Research jako DINO, łączy kontrastowe
  trenowanie denoisingowe z mieszanym wyborem zapytań na wierzchu rzadkiej uwagi
  Deformowalnego DETR. LibreYOLO jest dostępny w trzech rozmiarach do detekcji,
  tylko do inferencji.
keywords:
  - DINO-DETR
  - DINO
  - transformer wykrywania
  - odszumiające ramki kotwic
  - wybór mieszanych zapytań
  - detekcja obiektów
  - Badania IDEA
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDINODETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDINODETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")

        # val() zwraca zwykły słownik, a nie obiekt
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDINODETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDINODETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDINODETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: Użyj wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka kieruje na podstawie rozszerzenia pliku, więc eksportowany
        artefakt się ładuje

        # jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreDINODETRr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: dda176ebee3a83de
---
## Instalacja

DINO-DETR nie wymaga żadnych dodatkowych opcji. Wszystko, co importuje, znajduje się w instalacji bazowej, używając tego samego czystego jądra uwagi wieloskalowej odkształcalnej PyTorch co rodzina Deformable DETR LibreYOLO.

```bash
pip install libreyolo
```

Instalacja `libreyolo[hub-kernels]` jest opcjonalna. Gdy pakiet `kernels` jest obecny, LibreYOLO pobiera skompilowane jądro uwagi wieloskalowej zdeformowanej z Hugging Face Hub w czasie wykonywania i używa go zamiast czystego rdzenia PyTorch; `LIBREYOLO_HUB_KERNELS=0` wyłącza je ponownie.

## Predykcja

Wagi są pobierane z Hugging Face przy pierwszym użyciu i są przechowywane w pamięci lokalnej.

<code-tabs name="predict" />

Zwrócony obiekt `Results` jest tym, który zwraca każda rodzina, więc zamiana na inny detektor to zmiana jednej linii. `conf` i `max_det` filtrują wybór zapytania; `iou` jest akceptowany dla zgodności z API, ale nie ma efektu, ponieważ dekoder jest przewidującym zestaw, który nie ma kroku NMS. Zobacz [predykcja](/docs/predict) dla źródeł, streaming i obsługi wyników.

DINO-DETR jest tylko do inferencji w LibreYOLO. Pociągi nadrzędne z kontrastowym denoise'owaniem i dopasowaniem węgierskim; ten przepis nie jest tutaj zaimplementowany, więc `train()` wywołuje `NotImplementedError`.

## Warianty

Trzy checkpointy, wszystkie o tej samej rozdzielczości wejściowej. `r50` i `r50s5` dzielą ResNet-50 backbone i różnią się tym, ile skal map cech zasila dekoder, cztery w porównaniu do pięciu. `swinl` zamienia backbone na Swin-L i również próbuje pięć skal.

## Walidacja

`val()` zwraca słownik kluczy `metrics/` obejmujących dokładność, recall, mAP 50 i mAP 50-95, mierzone względem dowolnego zbioru danych w formacie, na którym byłeś trenowany.

<code-tabs name="val" />

## Eksport

<export-matrix />

Eksportowany artefakt ładuje się z powrotem przez `LibreYOLO()` na podstawie jego rozszerzenia pliku, więc plik `.onnx` lub `.engine` zachowuje się jak checkpoint i zwraca ten sam `Results`. [Eksport](/docs/export) wymienia argumenty akceptowane przez każdy format.

<code-tabs name="export" />

## Checkpointy

Każdy opublikowany plik wag dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box>

Trzy oficjalne checkpointy pochodzą z folderu wydania Google Drive autorów, a nie z karty modelu Hugging Face. Repozytorium upstream deklaruje Apache-2.0 na poziomie repozytorium, ale nie załącza pliku licencji ani metadanych licencji do samych checkpointów, więc podstawa do redystrybucji opiera się na deklaracji na poziomie repozytorium, a nie na udzieleniu licencji dla konkretnego checkpointu. Każde lustrzane repozytorium LibreYOLO dostarcza dokładny tekst licencji upstream Apache-2.0 wraz z powiadomieniem objaśniającym to.

</provenance-box>

## Cytowanie

<citation-block /> 
