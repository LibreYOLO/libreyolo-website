---
title: libreyolo predict
seo_title: dokumentacja polecenia libreyolo predict
description: >-
  Uruchamianie inferencji z wiersza poleceń: każdy argument, jego wartość
  domyślna odczytana z definicji CLI oraz flagi zmieniające to, co trafia na
  stdout.
lead: >-
  Uruchamia wczytany model na jednym źródle i wypisuje predykcje. Źródłem może
  być obraz, katalog, wideo, adres URL lub strumień na żywo; modelem może być
  checkpoint lub wyeksportowany artefakt.
keywords:
  - libreyolo predict cli
  - inferencja yolo z wiersza poleceń
  - polecenie libreyolo predict
  - argumenty libreyolo predict
  - yolo json na stdout
last_verified: 1.5.0
meta:
  - label: Polecenie
    value: libreyolo predict
    mono: true
  - label: Wymagane
    value: source
    mono: true
  - label: Wyjście
    value: >-
      Predykcje na stdout. Przy save=true pliki z adnotacjami w
      runs/detect/predict
snippets:
  examples:
    - label: Podstawowe użycie
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Zapis obrazów z adnotacjami
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=true \
          project=runs/detect name=parkour exist_ok=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 'Filtrowanie klas, JSON na stdout'
      language: bash
      code: >
        # klasa 0 to person na liście klas COCO dołączonej do checkpointu.

        libreyolo predict model=LibreYOLO9s.pt classes="[0]" conf=0.4 max_det=50
        \
          json=true quiet=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: 7e46c7ed7dd9e6c4
---

## Składnia

```bash
libreyolo predict source=<path|url|index> [model=<name|path>] [key=value ...]
```

Argumenty to pary `key=value`. To samo polecenie przyjmuje również formę POSIX,
więc `conf=0.4` i `--conf 0.4` są wymienne, a wartość logiczna zapisana jako
`save=true` staje się `--save`. Nazwy z podkreśleniem przyjmują obie pisownie:
`max_det=50` i `--max-det 50` trafiają do tej samej opcji.

`libreyolo detect predict ...` jest akceptowane i działa identycznie; słowo
zadania jest usuwane przed parsowaniem.

## Argumenty

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `source` | | Ścieżka do obrazu, katalog lub adres URL. Wymagane |
| `model` | `yolox-s` | Nazwa lub ścieżka modelu |
| `conf` | `0.25` | Próg pewności |
| `iou` | `0.45` | Próg IoU dla NMS |
| `imgsz` | | Rozmiar obrazu wejściowego: `640` (kwadrat) lub `480x640` (HxW). Gdy nie ustawiono, własny rozmiar wejściowy modelu |
| `classes` | | Filtrowanie po identyfikatorach klas, np. `[0,2,5]`. Akceptowana jest też sama liczba całkowita |
| `max_det` | `300` | Maksymalna liczba detekcji na obraz |
| `half` | `false` | Inferencja FP16 (tylko CUDA, wymaga obsługi po stronie modelu) |
| `save` | `false` | Zapis obrazów z adnotacjami |
| `batch` | `1` | Liczba obrazów na jedno przejście w przód dla źródeł katalogowych. Powyżej 1 uruchamia prawdziwą inferencję batchową na modelach, które ją obsługują |
| `stream` | `false` | Zwracanie wyników przyrostowo. Włączane automatycznie dla kamer internetowych i strumieni na żywo |
| `stream_buffer` | `false` | Buforowanie każdej klatki na żywo zamiast zachowywania tylko najnowszej |
| `vid_stride` | `1` | Przetwarzanie co N-tej klatki wideo lub klatki na żywo |
| `show` | `false` | Wyświetlanie wyników wideo i na żywo; `q` zatrzymuje |
| `tiling` | `false` | Inferencja kafelkowa dla dużych obrazów |
| `overlap_ratio` | `0.2` | Współczynnik nakładania się kafelków |
| `output_path` | | Jawna ścieżka wyjściowa. W przeciwnym razie `project/name`, gdy `save=true` |
| `color_format` | `auto` | Kolor wejściowy: `auto`, `rgb`, `bgr` |
| `output_file_format` | | Format wyjściowy: `jpg`, `png`, `webp` |
| `device` | `auto` | Urządzenie: `0`, `cpu`, `mps`, `auto` |
| `face_detector` | | Model detektora twarzy (ścieżka lub nazwa CLI). Wymagany dla modeli estymacji spojrzenia (gaze) |
| `gallery` | | Galeria twarzy `.npz` z `libreyolo enroll`, względem której identyfikowane są twarze. Tylko modele embeddingów twarzy |
| `gallery_threshold` | `0.4` | Próg cosinusowy dopasowania tożsamości w galerii |
| `project` | `runs/detect` | Katalog główny wyjścia |
| `name` | `predict` | Nazwa eksperymentu |
| `exist_ok` | `false` | Ponowne użycie istniejącego katalogu wyjściowego |
| `json` | `false` | Wyjście JSON na stdout |
| `quiet` | `false` | Wyciszenie stderr |
| `verbose` | `false` | Szczegółowe wyjście na stderr |
| `help_json` | `false` | Zrzut schematu polecenia w formacie JSON i zakończenie |

## Przykłady

<code-tabs name="examples" />

## Uwagi

Wyeksportowany artefakt wczytuje się tak samo jak checkpoint, więc
`model=weights/LibreYOLO9s.onnx` i `model=weights/LibreYOLO9s.engine` są
poprawnymi wartościami dla `model`. Trzy opcje są w tych środowiskach
uruchomieniowych odrzucane, a nie ignorowane: `tiling`, `overlap_ratio` i
`output_file_format` kończą działanie z `config_unsupported`, gdy backend
środowiska uruchomieniowego nie może ich obsłużyć.

`half` działa odwrotnie. Wyeksportowane środowiska uruchomieniowe przyjmują tę
opcję i działają w FP16; natywna inferencja w PyTorch zapisuje w logach, że
została zignorowana, i kontynuuje w FP32.

Modele estymacji spojrzenia są dwuetapowe i nie mają własnego detektora,
dlatego `face_detector` jest dla nich wymagany. `gallery` dotyczy tylko modeli,
których zadaniem jest `embed`; przekazanie tej opcji czemukolwiek innemu kończy
działanie z `config_unsupported`.

stdout niesie wyniki i nic poza nimi; postęp, ostrzeżenia i błędy trafiają na
stderr. `json=true` wypisuje jeden obiekt JSON na wywołanie lub jeden na klatkę
przy streamingu, każdy z polem `schema_version`. `quiet=true` wycisza stderr.
Obie opcje razem dają czytnikowi maszynowemu czysty strumień stdout.

Kod wyjścia to `0` przy powodzeniu, `2` przy błędzie użycia lub konfiguracji,
`3`, gdy nie można znaleźć źródła, `4`, gdy nie można wczytać modelu, oraz `1`
przy innych awariach w czasie działania.

`help_json=true` wypisuje parametry polecenia, typy, wartości domyślne i flagi
w formacie JSON, nie uruchamiając niczego, co jest niezawodnym sposobem na
odczytanie tej tabeli z zainstalowanej wersji.

Powiązane: [`libreyolo val`](/docs/cli/val) dla zmierzonych metryk na zbiorze
danych, [`libreyolo export`](/docs/cli/export) do wytworzenia wymienionych wyżej
artefaktów środowiska uruchomieniowego.
