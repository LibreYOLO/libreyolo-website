---
title: CenterNet
families:
  - centernet
seo_title: 'CenterNet: detekcja obiektów w LibreYOLO'
description: >-
  Uruchamiaj CenterNet (Objects as Points) w LibreYOLO z backbone ResDCN-18 i
  DLA-34. Wykonuj predykcję, walidację i eksport do ONNX na licencji MIT. Bez
  ścieżki trenowania.
lead: >-
  CenterNet przedstawia obiekt jako punkt środkowy jego ramki ograniczającej i
  wyznacza każdą inną właściwość ze szczytu mapy cieplnej. Dzięki temu nie
  wymaga kotwic ani etapu non-maximum suppression. LibreYOLO udostępnia go jako
  detektor przeznaczony tylko do inferencji.
keywords:
  - CenterNet
  - Objects as Points
  - detekcja punktów kluczowych
  - detektor bez kotwic
  - ResDCN-18
  - DLA-34
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreCenterNetresdcn18.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: DLA-34
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetdla34.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCenterNetresdcn18.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreCenterNetresdcn18.pt")


        # Eksport ONNX wymaga opset 16 lub nowszego: etap upsamplingu

        # z deformowalnym splotem jest przekształcany do GridSample, dodanego w
        opset 16.

        model.export(format="onnx", opset=18)

        model.export(format="tensorrt")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreCenterNetresdcn18.pt format=onnx opset=18
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Fabryka wybiera ścieżkę na podstawie sufiksu pliku, więc
        wyeksportowany

        # artefakt wczytuje się jak każdy checkpoint i zwraca ten sam obiekt
        Results.

        model = LibreYOLO("LibreCenterNetresdcn18.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 20aaef83cc95590d
---

## Instalacja

CenterNet nie wymaga opcjonalnego dodatku. Wszystkie importowane elementy
znajdują się w instalacji bazowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane lokalnie
w pamięci podręcznej.

<code-tabs name="predict" />

Zwracany obiekt `Results` jest taki sam dla każdej rodziny, dlatego zamiana
detektora wymaga zmiany jednego wiersza. Argumenty `conf` i `max_det` filtrują
uszeregowane szczyty mapy cieplnej. Argument `iou` jest akceptowany dla
zgodności API, lecz nie ma wpływu na wynik, ponieważ dekodowanie szczytów top-k
w CenterNet nie wymaga etapu tłumienia ramek na podstawie IoU. Zobacz stronę
[predykcji](/docs/predict), aby poznać źródła, streaming i obsługę wyników.

## Warianty

Dostępne są dwa backbone. `resdcn18` łączy główną część ResNet-18 z upsamplingiem
wykorzystującym deformowalny splot. `dla34` łączy główną część DLA-34
z upsamplingiem wykorzystującym iteracyjną głęboką agregację. Oba warianty
zasilają te same trzy gęste głowice (mapa cieplna, szerokość/wysokość,
przesunięcie) i korzystają z tego samego obszaru wejściowego.

## Walidacja

Metoda `val()` zwraca słownik kluczy `metrics/` obejmujących precyzję, recall,
mAP 50 i mAP 50-95, zmierzonych względem dowolnego zbioru danych w formacie
użytym podczas trenowania.

<code-tabs name="val" />

## Eksport

<export-matrix />

Eksport ONNX wymaga opset 16 lub nowszego. Etap upsamplingu z deformowalnym
splotem w obu backbone jest przekształcany do operatora ONNX `GridSample`,
który wprowadzono w opset 16. Wybranie starszego opset powoduje zgłoszenie
błędu przed rozpoczęciem śledzenia grafu.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box>

Graf ResDCN-18 zawiera również kod projektu human-pose-estimation.pytorch
firmy Microsoft na licencji MIT, a graf DLA-34 kod implementacji DLA autorstwa
Fishera Yu na licencji BSD-3-Clause. LibreYOLO nie dołącza oryginalnego
rozszerzenia DCNv2 używanego przez projekt upstream. Natywne wykonanie korzysta
z funkcji `deform_conv2d` z torchvision na licencji BSD-3-Clause, natomiast
przenośną implementację przeznaczoną wyłącznie do eksportu opracowano osobno
dla LibreYOLO.

</provenance-box>

## Cytowanie

<citation-block />
