---
title: Deformable DETR
families:
  - deformable_detr
seo_title: 'Deformable DETR: predykcja i eksport na licencji Apache-2.0'
description: >-
  Uruchamiaj Deformable DETR w LibreYOLO do detekcji obiektów. Instaluj,
  uruchamiaj predykcję, waliduj i eksportuj pięć wariantów z rzadką uwagą,
  wszystkie na licencji Apache-2.0.
lead: >-
  Deformable DETR zastępuje gęstą uwagę krzyżową DETR rzadkim, wieloskalowym
  próbkowaniem wokół każdego punktu odniesienia, co sprawiło, że trenowanie
  detektorów transformatorowych stało się praktyczne. LibreYOLO oferuje pięć
  rozmiarów do detekcji, tylko do inferencji.
keywords:
  - Deformable DETR
  - transformer detekcyjny
  - rzadka uwaga
  - uwaga wieloskalowa
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

Zwracany obiekt `Results` jest taki sam dla każdej rodziny, dlatego zamiana
detektora wymaga zmiany jednego wiersza. Argumenty `conf` i `max_det` filtrują
wybór zapytań. Argument `iou` jest akceptowany dla zgodności API, ale nie ma
wpływu na wynik, ponieważ dekoder jest predyktorem zbioru bez etapu NMS. Zobacz
stronę [predykcji](/docs/predict), aby poznać źródła, streaming i obsługę
wyników.

Deformable DETR służy w LibreYOLO wyłącznie do inferencji. Model upstream jest
trenowany z dopasowaniem węgierskim i ogniskową funkcją straty klasyfikacji.
Tego przepisu nie zaimplementowano tutaj, dlatego `train()` zgłasza
`NotImplementedError`.

## Warianty

Pięć checkpointów obejmuje wszystkie opublikowane konfiguracje i korzysta
z tej samej rozdzielczości wejściowej. `r50ss` skupia uwagę na jednej skali
cech, a `r50ssdc5` dodaje do niego etap C5 backbone z dylatacją. `r50` jest
domyślną konfiguracją wieloskalową, która próbkuje cztery poziomy map cech.
`r50refine` dodaje iteracyjne udoskonalanie ramek między warstwami dekodera,
a `r50twostage` tworzy początkowe propozycje regionów z wyjścia enkodera zamiast
z wyuczonych zapytań.

## Walidacja

`val()` zwraca słownik kluczy `metrics/` obejmujących precyzję, czułość, mAP 50 oraz mAP 50-95, mierzone względem dowolnego zbioru danych w formacie, na którym trenowałeś.

<code-tabs name="val" />

## Eksport

<export-matrix />

Eksportowany artefakt ładuje się z powrotem przez `LibreYOLO()` na podstawie jego rozszerzenia pliku, więc plik `.onnx` lub `.engine` zachowuje się jak checkpoint i zwraca ten sam `Results`. [Eksport](/docs/export) wymienia argumenty akceptowane przez każdy format.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
