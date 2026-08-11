---
title: Odkształcalny DETR
families:
  - deformable_detr
seo_title: 'Deformowalny DETR: przewiduj i eksportuj, Apache-2.0'
description: >-
  Uruchom Deformable DETR w LibreYOLO do wykrywania obiektów. Zainstaluj,
  przewiduj, waliduj i eksportuj pięć rozmiarów rzadkiej uwagi, wszystkie
  licencjonowane przez Apache-2.0.
lead: >-
  Deformowalny DETR zastępuje gęstą uwagę krzyżową DETR rzadkim, wieloskalowym
  próbkowaniem wokół każdego punktu odniesienia, co sprawiło, że trenowanie
  detektorów transformatorowych stało się praktyczne. LibreYOLO oferuje pięć
  rozmiarów do detekcji, tylko do inferencji.
keywords:
  - Odkształcalny DETR
  - transformer wykrywania
  - rzadka uwaga
  - uwaga wieloskaliowa
  - detekcja obiektów
  - SenseTime
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeformableDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")

        # val() zwraca zwykły słownik, a nie obiekt
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeformableDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDeformableDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDeformableDETRr50.pt format=tensorrt
        imgsz=800 half=True
    - label: Użyj wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka kieruje na podstawie rozszerzenia pliku, więc eksportowany
        artefakt się ładuje

        # jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreDeformableDETRr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 35225efc54b5ef91
---
## Instalacja

Deformowalny DETR nie wymaga żadnych dodatkowych opcji. Wszystko, co importuje, znajduje się w podstawowej instalacji, korzystając z rdzenia uwagi wieloskalowej deformowalnej pure-PyTorch.

```bash
pip install libreyolo
```

Instalacja `libreyolo[hub-kernels]` jest opcjonalna. Gdy pakiet `kernels` jest obecny, LibreYOLO pobiera skompilowane jądro wieloskalowej deformowalnej uwagi z Hugging Face Hub w czasie wykonywania i używa go zamiast czystego rdzenia PyTorch; `LIBREYOLO_HUB_KERNELS=0` wyłącza je ponownie.

## Predykcja

Wagi są pobierane z Hugging Face przy pierwszym użyciu i są przechowywane w pamięci lokalnej.

<code-tabs name="predict" />

Zwrócony obiekt `Results` jest tym, który zwraca każda rodzina, więc zamiana na inny detektor to zmiana jednej linii. `conf` i `max_det` filtrują wybór zapytania; `iou` jest akceptowany dla zgodności z API, ale nie ma efektu, ponieważ dekoder jest przewidującym zestaw, który nie ma kroku NMS. Zobacz [prediction](/docs/predict) dla źródeł, streaming i obsługi wyników.

Deformowalny DETR jest tylko do wnioskowania w LibreYOLO. W górę strumienia trenowany jest z dopasowaniem węgierskim i stratą klasyfikacyjną fokalną; ten przepis nie jest tutaj zaimplementowany, więc `train()` powoduje `NotImplementedError`.

## Warianty

Pięć checkpointów obejmuje opublikowane konfiguracje, wszystkie przy tym samym rozdzielczości wejściowej. `r50ss` koncentruje uwagę na jednej skali cech; `r50ssdc5` dodaje rozszerzoną fazę C5 backbone na jej szczycie. `r50` to domyślna konfiguracja wieloskalowa, próbkująca z czterech poziomów map cech. `r50refine` dodaje iteracyjne dopracowywanie obszarów ograniczających w warstwach dekodera, a `r50twostage` generuje swoje początkowe propozycje regionów z wyjścia enkodera zamiast z wyuczonych zapytań.

## Walidacja

`val()` zwraca słownik kluczy `metrics/` obejmujących precyzję, czułość, mAP 50 oraz mAP 50-95, mierzone względem dowolnego zbioru danych w formacie, na którym trenowałeś.

<code-tabs name="val" />

## Eksport

<export-matrix />

Eksportowany artefakt ładuje się z powrotem przez `LibreYOLO()` na podstawie jego rozszerzenia pliku, więc plik `.onnx` lub `.engine` zachowuje się jak checkpoint i zwraca ten sam `Results`. [Eksport](/docs/export) wymienia argumenty akceptowane przez każdy format.

<code-tabs name="export" />

## Checkpointy

Każdy opublikowany plik wagowy dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
