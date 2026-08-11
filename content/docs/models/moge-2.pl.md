---
title: MoGe-2
families:
  - moge2
seo_title: 'MoGe-2: predykcja, walidacja i eksport normalnych powierzchni'
description: >-
  Używaj MoGe-2 w LibreYOLO do gęstej predykcji normalnych powierzchni.
  Instaluj, przewiduj, waliduj i eksportuj oficjalne checkpointy ViT-S, ViT-B i
  ViT-L.
lead: >-
  MoGe-2 to monokularny model geometrii działający w jednym przebiegu w przód,
  który przewiduje gęste pole normalnych powierzchni na podstawie jednego obrazu
  RGB. LibreYOLO obsługuje go wyłącznie do estymacji normalnych przez oficjalne
  checkpointy ViT-S, ViT-B i ViT-L.
keywords:
  - MoGe-2
  - MoGe 2
  - estymacja normalnych powierzchni
  - geometria monokularna
  - mapa normalnych
  - gęsta predykcja
  - DINOv2
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE, save=True)

        normal = result.normal_map
        print(normal.array.shape)   # (H, W, 3), wektory jednostkowe float32
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMoGe2s-normal.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])   # stopnie
        print(metrics["metrics/median_angular_error"])
        print(metrics["metrics/within_11_25"])          # procent pikseli
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMoGe2s-normal.pt data=my-dataset.yaml imgsz=518
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
        model.export(format="tensorrt", imgsz=518, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreMoGe2s-normal.pt format=onnx imgsz=518

        libreyolo export model=LibreMoGe2s-normal.pt format=tensorrt imgsz=518
        half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.normal_map.array.shape)
source_hash: ddfacf6b7e9729f6
---

## Instalacja

MoGe-2 nie wymaga żadnego opcjonalnego dodatku. Wszystkie importowane elementy
znajdują się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane automatycznie. LibreYOLO pobiera
odpowiedni rozmiar bezpośrednio z oficjalnych checkpointów i zapisuje go w
lokalnej pamięci podręcznej.

<code-tabs name="predict" />

MoGe-2 zwraca gęste pole zamiast zbioru detekcji, dlatego `result.boxes` jest
puste, a `conf`, `iou` i `max_det` nie mają wpływu na wynik.
`result.normal_map` zawiera tablicę wektorów jednostkowych `(H, W, 3)` w układzie
kamery OpenCV, gdzie `+x` wskazuje w prawo, `+y` w dół, `+z` w głąb sceny, a
powierzchnia skierowana do kamery ma wartość `(0, 0, -1)`. Predykcja listy
obrazów wykonuje jeden przebieg w przód na obraz. Ta rodzina nie ma szybkiej
ścieżki ze złożonym batchem. Więcej informacji o źródłach, streamingu i obsłudze
wyników zawiera strona [predykcji](/docs/predict).

## Warianty

Trzy rozmiary enkodera są dostępne jako osobne checkpointy: ViT-S, ViT-B i
ViT-L, wszystkie przy tej samej rozdzielczości wejściowej. Zestaw testowy
LibreYOLO nie wykonał pomiarów tej rodziny, dlatego nie ma opublikowanych
wyników dokładności do ich porównania. Rozmiar należy dobrać do własnego budżetu
obliczeniowego.

## Walidacja

`val()` mierzy błąd kątowy względem sparowanego zbioru map normalnych. Obrazy
znajdują się obok 16-bitowych plików PNG normalnych o tej samej nazwie bazowej,
a opcjonalna maska poprawności sprawia, że dopełnione i nieprawidłowe piksele nie
są uwzględniane. Zwracany jest średni i medianowy błąd kątowy w stopniach oraz
procent pikseli mieszczących się w zakresie 11.25, 22.5 i 30 stopni.

<code-tabs name="val" />

## Eksport

<export-matrix />

Eksport normalnych korzysta z kontraktu środowiska uruchomieniowego o stałej
rozdzielczości i rozmiarze batcha 1. Argument `dynamic` oraz `batch` inny niż 1
są odrzucane, a `imgsz` musi być podzielne przez rozmiar patcha enkodera ViT, co
LibreYOLO sprawdza przed rozpoczęciem przebiegu. Wyeksportowany artefakt jest
ponownie ładowany przez `LibreYOLO()` na podstawie rozszerzenia pliku, dlatego
plik `.onnx` zachowuje się jak checkpoint i zwraca ten sam obiekt `Results`.

<code-tabs name="export" />

## Licencja

<provenance-box>

LibreYOLO nie kopiuje tych checkpointów do własnej organizacji.
`LibreYOLO("LibreMoGe2s-normal.pt")` pobiera odpowiedni rozmiar bezpośrednio z
oficjalnych repozytoriów Hugging Face przy przypiętej rewizji i przed użyciem
weryfikuje plik względem zapisanego skrótu SHA-256.

</provenance-box>

## Cytowanie

<citation-block />
