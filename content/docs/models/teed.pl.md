---
title: TEED
families:
  - teed
seo_title: 'TEED: detekcja krawędzi z własnym checkpointem'
description: >-
  Używaj TEED w LibreYOLO do przewidywania gęstych map prawdopodobieństwa
  krawędzi. Przekonwertuj checkpoint objęty odpowiednią licencją, a następnie
  wykonuj predykcję, walidację i eksport.
lead: >-
  TEED (Tiny and Efficient Edge Detector) to mała sieć konwolucyjna, która
  przewiduje gęstą mapę prawdopodobieństwa krawędzi na podstawie jednego obrazu
  RGB. LibreYOLO udostępnia jej architekturę wyłącznie do detekcji krawędzi.
  Biblioteka nie zawiera checkpointu.
keywords:
  - TEED
  - Tiny and Efficient Edge Detector
  - detekcja krawędzi
  - BIPED
  - gęsta predykcja
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        result = model(SAMPLE_IMAGE, save=True)

        edges = result.edges
        print(edges.array.shape)        # (H, W) float32 w zakresie [0, 1]
        print(edges.binary(0.5).sum())  # liczba pikseli krawędzi po progowaniu
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=weights/LibreTEEDt-edge.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("weights/LibreTEEDt-edge.pt")

        metrics = model.val(data="my-dataset.yaml", imgsz=352)


        print(metrics["metrics/ODS"])   # miara F w optymalnej skali zbioru
        danych

        print(metrics["metrics/OIS"])   # miara F w optymalnej skali obrazu
    - label: CLI
      language: bash
      code: >
        libreyolo val model=weights/LibreTEEDt-edge.pt data=my-dataset.yaml
        imgsz=352
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        model.export(format="onnx", imgsz=352)
        model.export(format="tensorrt", imgsz=352, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=weights/LibreTEEDt-edge.pt format=onnx imgsz=352

        libreyolo export model=weights/LibreTEEDt-edge.pt format=tensorrt
        imgsz=352 half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreTEEDt-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: c7203b254e460258
---

## Instalacja

TEED nie wymaga żadnego opcjonalnego dodatku. Wszystkie importowane elementy
znajdują się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

LibreYOLO nie udostępnia checkpointu TEED. Oficjalnie opublikowane wagi
wytrenowano na BIPED, którego warunki udostępnienia ograniczają użycie do celów
niekomercyjnych, dlatego LibreYOLO nie tworzy ich kopii lustrzanej. Checkpoint,
na którego użycie zezwala posiadana licencja, należy przekonwertować za pomocą
`weights/convert_teed_weights.py`. Skrypt przed zapisaniem pliku, który
LibreYOLO może bezpośrednio wczytać, sprawdza klucze tensorów względem
architektury środowiska uruchomieniowego:

```bash
python weights/convert_teed_weights.py upstream.pth weights/LibreTEEDt-edge.pt --verify
```

<code-tabs name="predict" />

`result.edges` zawiera wynik: tablicę float32 `(H, W)` z wartościami w zakresie
`[0, 1]`. Metoda `.binary(threshold)` zwraca logiczną maskę krawędzi. Model nie
zwraca ramek, dlatego `conf`, `iou` i `max_det` nie mają wpływu na wynik. Więcej
informacji o źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Warianty

W LibreYOLO model TEED jest dostępny w jednym rozmiarze. Zestaw testowy
LibreYOLO nie wykonał pomiarów tej rodziny, dlatego nie ma opublikowanych
wyników, z którymi można ją porównać.

## Walidacja

`val()` zwraca miary F ODS i OIS w stylu BSDS, obliczone względem sparowanego
zbioru danych krawędzi. Obrazy są umieszczone obok map krawędzi o tej samej
nazwie bazowej. Opcjonalna maska poprawności sprawia, że dopełnione piksele nie
są nigdy uwzględniane. `imgsz` musi być podzielne przez krok zmniejszania
rozdzielczości sieci. Jeśli tak nie jest, LibreYOLO zgłasza czytelny błąd.

<code-tabs name="val" />

## Eksport

<export-matrix />

Eksport krawędzi korzysta z kontraktu środowiska uruchomieniowego o stałej
rozdzielczości i rozmiarze batcha 1. Argument `dynamic` oraz `batch` inny niż 1
są odrzucane, a wyeksportowany graf zwraca jedną scaloną mapę prawdopodobieństwa.
Wyeksportowany artefakt jest ponownie ładowany przez `LibreYOLO()` na podstawie
rozszerzenia pliku, dlatego plik `.onnx` zachowuje się jak checkpoint i zwraca
ten sam obiekt `Results`.

<code-tabs name="export" />

## Licencja

<provenance-box>

LibreYOLO nie publikuje checkpointu TEED. Organizacja LibreYOLO nie tworzy
lustrzanej kopii żadnego checkpointu. Zamiast tego należy przekonwertować
checkpoint objęty posiadaną licencją za pomocą
`weights/convert_teed_weights.py`.

</provenance-box>

## Cytowanie

<citation-block />
