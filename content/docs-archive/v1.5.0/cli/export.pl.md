---
title: libreyolo export
seo_title: dokumentacja polecenia libreyolo export
description: >-
  Eksport checkpointu do formatu do wdrożenia: każdy argument z wartością
  domyślną, gdzie trafia artefakt i jakie kombinacje polecenie odrzuca.
lead: >-
  Zamienia jeden checkpoint w jeden format do wdrożenia i zapisuje artefakt w
  weights/. To format decyduje, które z poniższych argumentów mają zastosowanie.
keywords:
  - libreyolo export cli
  - eksport yolo do onnx
  - polecenie libreyolo export
  - eksport yolo do tensorrt
  - argumenty libreyolo export
last_verified: 1.5.0
meta:
  - label: Polecenie
    value: libreyolo export
    mono: true
  - label: Wymagany
    value: model
    mono: true
  - label: Wyjście
    value: 'weights/<checkpoint-stem>[_fp16|_int8]<format-suffix>'
    mono: true
snippets:
  examples:
    - label: Podstawowy przykład
      language: bash
      code: |
        # Zapisuje weights/LibreYOLO9s.onnx
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: NMS wewnątrz grafu
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx \
          nms=true conf=0.25 iou=0.45 max_det=300
    - label: Uruchomienie artefaktu
      language: bash
      code: >
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640


        # Fabryka rozpoznaje sufiks pliku, więc eksport wczytuje się jak
        checkpoint.

        libreyolo predict model=weights/LibreYOLO9s.onnx \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: ef2ca20af3814109
---

## Składnia

```bash
libreyolo export model=<name|path> [format=<format>] [key=value ...]
```

Argumenty to pary `key=value`, działa też forma POSIX, więc `format=onnx` i
`--format onnx` to ten sam argument.

## Argumenty

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `model` | | Wagi modelu `.pt`. Wymagane |
| `format` | `onnx` | Format eksportu: `onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`, `rknn`, `ncnn`, `tflite`, `coreml`, `coreai` |
| `name` | | Platforma docelowa RKNN, obecnie wyłącznie `rk3588`. Odrzucane przy każdym innym formacie |
| `imgsz` | | Rozmiar obrazu wejściowego: `640` lub `480x640` (HxW). Akceptowane jest także `480,640`. Gdy nie ustawiono, własny rozmiar modelu |
| `batch` | `1` | Rozmiar batcha przy eksporcie |
| `half` | `false` | Precyzja FP16 |
| `int8` | `false` | Kwantyzacja INT8 |
| `dynamic` | `false` | Dynamiczne kształty wejścia (ONNX) |
| `simplify` | `true` | Upraszczanie grafu ONNX |
| `nms` | `false` | Osadzenie NMS w modelu. Tylko ONNX i CoreML |
| `conf` | `0.25` | Próg pewności dla osadzonego NMS |
| `iou` | `0.45` | Próg IoU dla osadzonego NMS |
| `max_det` | `300` | Maksymalna liczba detekcji dla osadzonego NMS w ONNX |
| `opset` | | Wersja opset ONNX. Gdy nie ustawiono, wybierana automatycznie |
| `data` | | Dane kalibracyjne dla INT8 |
| `fraction` | `1.0` | Część danych kalibracyjnych, która zostanie użyta |
| `device` | `auto` | Urządzenie do tracingu |
| `allow_download_scripts` | `false` | Zezwolenie na osadzony kod Pythona w blokach pobierania w pliku YAML zbioru danych |
| `json` | `false` | Wynik w formacie JSON na stdout |
| `quiet` | `false` | Wyciszenie stderr |
| `verbose` | `false` | Szczegółowe logowanie eksportu |
| `verify` | `false` | Uruchomienie symulatora PC z RKNN Toolkit2 i porównanie z ONNX Runtime. Tylko RKNN |
| `help_json` | `false` | Zrzut schematu polecenia w formacie JSON i wyjście |

`engine` to alias dla `tensorrt`, a `litert` alias dla `tflite`. Oba są
rozwiązywane do nazwy kanonicznej, zanim cokolwiek zostanie zapisane, więc wynik
JSON i linia logu zawsze podają `tensorrt` lub `tflite`.

## Przykłady

<code-tabs name="examples" />

## Uwagi

### Gdzie trafia plik

Polecenie nie przyjmuje ścieżki wyjściowej. Artefakt jest zapisywany w
`weights/`, a jego nazwa to rdzeń nazwy źródłowego checkpointu plus sufiks
formatu, z wstawionym `_fp16` lub `_int8`, gdy zażądano jednej z tych precyzji.
`LibreYOLO9s.pt` wyeksportowany do ONNX w FP16 staje się
`weights/LibreYOLO9s_fp16.onnx`. Wynik JSON zawiera rozwiązaną wartość
`output_path`, rozmiar pliku w MB oraz kształt wejścia w postaci
`[batch, 3, height, width]`.

### Kombinacje, które są odrzucane

`nms=true` jest akceptowane dla ONNX i CoreML, a dla każdego innego formatu
odrzucane z `nms_unsupported_format`. W ONNX wymusza wyłączenie `dynamic`,
ponieważ osadzony graf jest ustalony na batch 1, i informuje o tym na stderr.
W CoreML przyjmuje `conf` i `iou`, ale nie `max_det`, więc niedomyślna wartość
`max_det` razem z `format=coreml nms=true` kończy się wyjściem z
`config_unsupported`.

`half=true` razem z `int8=true` nie jest błędem. INT8 wygrywa, `half` zostaje
pominięte, a ostrzeżenie trafia na stderr.

`name` i `verify` są dziś opcjami RKNN. Przekazanie którejkolwiek z nich z innym
formatem kończy się wyjściem z `config_unsupported`, zamiast zostać
zignorowane.

### Które formaty obsługuje dana rodzina

Wsparcie jest określane per rodzina i per zadanie, nie globalnie. `libreyolo
formats family=<family> task=<task>` wypisuje poziom wsparcia dla każdego
formatu w tej kombinacji, wraz z uzasadnieniem i ewentualnym ograniczeniem.
Argumenty opisuje [`libreyolo formats`](/docs/cli/utilities).

Niektóre formaty wymagają opcjonalnej instalacji, a niektóre gotowego
toolchainu. Brakująca zależność Pythona kończy się wyjściem z
`export_dep_missing`; precyzja, której format nie potrafi wygenerować, kończy
się wyjściem z `format_precision_unsupported`.

### Uruchamianie tego, co wyeksportowano

Wyeksportowane artefakty są wczytywane przez tę samą fabrykę modeli co
checkpointy, na podstawie sufiksu pliku, więc
`libreyolo predict model=weights/LibreYOLO9s.onnx` działa bez żadnej dalszej
konwersji. Wyjątkiem są trzy opcje predykcji, odrzucane na backendach runtime:
`tiling`, `overlap_ratio` i `output_file_format`.

Dwa cele wdrożenia mają własne strony:
[NVIDIA DeepStream](/docs/export/deepstream) i
[NVIDIA Jetson](/docs/export/jetson).

### Wynik i kody wyjścia

Wynik trafia na stdout; postęp na stderr. Kod wyjścia to `0` przy powodzeniu,
`2` przy błędzie użycia lub konfiguracji, `4` gdy modelu nie da się wczytać, `5`
przy nieznanym formacie, brakującej zależności eksportu, nieobsługiwanej
precyzji lub odrzuconym żądaniu osadzenia NMS, oraz `1` przy pozostałych błędach
w trakcie działania.

Powiązane: [`libreyolo quantize`](/docs/cli/quantize), które pozostaje w
środowisku PyTorch i zapisuje checkpoint, a nie artefakt do wdrożenia.
