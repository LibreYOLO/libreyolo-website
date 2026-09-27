---
title: libreyolo quantize
seo_title: libreyolo quantize opis polecenia
description: >-
  Kwantyzacja checkpointu w PyTorch z wiersza poleceń: przepisy, argumenty
  kalibracji, wartości domyślne i rodziny modeli akceptowane przez każdy
  przepis.
lead: >-
  Zastępuje moduły float modelu ich skwantyzowanymi odpowiednikami, kalibruje je
  na obrazach bez etykiet, gdy przepis potrzebuje statystyk, i zapisuje wynik
  jako checkpoint PyTorch.
keywords:
  - libreyolo quantize cli
  - kwantyzacja int8 yolo
  - kwantyzacja fp8
  - kwantyzacja po trenowaniu
  - libreyolo quantize argumenty
last_verified: 1.5.0
meta:
  - label: Polecenie
    value: libreyolo quantize
    mono: true
  - label: Wymagane
    value: model
    mono: true
  - label: Wynik
    value: 'Ścieżka źródłowa z -<recipe> przed sufiksem, np. LibreYOLO9s-int8.pt'
    mono: true
snippets:
  examples:
    - label: Podstawowe użycie
      language: bash
      code: |
        # Kalibruje na coco128 i zapisuje LibreYOLO9s-int8.pt
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8
    - label: 'Samo rzutowanie typu, bez kalibracji'
      language: bash
      code: |
        libreyolo quantize model=LibreYOLO9s.pt recipe=fp16 calib=none \
          out=weights/LibreYOLO9s-fp16.pt
    - label: 'Szersza kalibracja, potem odzyskiwanie dokładności'
      language: bash
      code: >
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8 \
          calib=coco128.yaml samples=256 batch=16 algorithm=minmax

        # Trenowanie z uwzględnieniem kwantyzacji na skwantyzowanym checkpoincie
        przywraca dokładność.

        libreyolo train model=LibreYOLO9s-int8.pt data=coco8.yaml epochs=10
        lr0=0.001
source_hash: 7ae663e9f117826e
---

## Składnia

```bash
libreyolo quantize model=<name|path> [recipe=<recipe>] [key=value ...]
```

Argumenty podaje się w parach `key=value`, forma POSIX również działa, więc
`recipe=int8` i `--recipe int8` to ten sam argument.

## Argumenty

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `model` | | Wagi modelu `.pt`. Wymagany |
| `recipe` | `int8` | Przepis kwantyzacji: `fp16`, `bf16`, `fp8`, `int8`, `w4a16`, `w4a8`, `nvfp4`, `mxfp4`, `int2` |
| `calib` | `coco128.yaml` | Obrazy do kalibracji: plik YAML ze zbiorem danych lub nazwa wbudowanego zbioru danych. Bez etykiet, tylko przejście w przód. `none` pomija kalibrację |
| `samples` | `128` | Maksymalna liczba obrazów kalibracyjnych |
| `batch` | `8` | Rozmiar batcha przy kalibracji |
| `algorithm` | `auto` | Estymacja zakresu aktywacji: `auto`, które wybiera minmax, albo `minmax`, albo `percentile` |
| `out` | | Ścieżka wyjściowego checkpointu. Domyślnie ścieżka źródłowa z `-<recipe>` przed sufiksem |
| `device` | `auto` | Urządzenie |
| `allow_download_scripts` | `false` | Zezwolenie na osadzony kod Pythona w blokach pobierania w pliku YAML zbioru danych |
| `json` | `false` | Wyjście JSON na stdout |
| `quiet` | `false` | Wyciszenie stderr |
| `help_json` | `false` | Zrzut schematu polecenia w formacie JSON i wyjście |

## Przykłady

<code-tabs name="examples" />

## Uwagi

### Które rodziny ją przyjmują

Kwantyzacja obejmuje cztery rodziny: `yolo9`, `rfdetr`, `birefnet` i
`feynobg`. Każda inna rodzina kończy się błędem `quantize_failed`, który zawiera
tę listę.

### Czego dotyka każdy przepis

`fp16` i `bf16` to rzutowania typów. Zmieniają wyłącznie dtype, nie wymagają
kalibracji, a właściwym ustawieniem jest dla nich `calib=none`.

`int8` i `fp8` kwantyzują moduły `Conv2d` i `Linear`, dlatego pasują do rodzin
splotowych.

`w4a16`, `w4a8`, `nvfp4`, `mxfp4` i `int2` kwantyzują wyłącznie `nn.Linear`,
więc są przeznaczone dla rodzin transformerowych. Żądanie któregokolwiek z nich
dla `yolo9` jest odrzucane z wyjaśnieniem, zamiast po cichu dawać
nieskwantyzowany model, ponieważ przyspieszenie poniżej 8 bitów działa tam tylko
dla GEMM, a sploty pozostałyby w wyższej precyzji.

`int8`, `fp8`, `w4a8` i `int2` potrzebują statystyk kalibracji dla swoich
aktywacji. `int2` wymaga dodatkowo trenowania, aby odzyskać dokładność po
kwantyzacji, dlatego jest odrzucany dla `birefnet` i `feynobg`, które nie mają
modułu trenowania.

Każda rodzina niezależnie od przepisu zostawia część modułów w float: pierwsze
warstwy, głowice predykcji oraz, w modelu YOLOv9, splot DFL, czyli stały operator
całkowej wartości oczekiwanej, którego nie wolno kwantyzować.

### Dane kalibracyjne to nie dane treningowe

`calib` wskazuje niewielki zbiór obrazów bez etykiet, używany tylko w przejściu
w przód, aby wyznaczyć zakresy aktywacji. Nie liczy się na nim metryk, a jego
etykiety nigdy nie są czytane. Domyślny `coco128.yaml` jest pobierany przy
pierwszym użyciu z adresu URL, więc nie wymaga dodatkowych uprawnień; plik YAML
z osadzonym skryptem pobierania w Pythonie wymaga `allow_download_scripts=true`.

`algorithm=percentile` jest dostępny i może obniżać dokładność w rodzinach
transformerowych, dlatego `auto` wybiera minmax.

### Odzyskiwanie dokładności

Wynikiem jest zwykły checkpoint PyTorch, więc
[`libreyolo train`](/docs/cli/train) przyjmuje go bezpośrednio. Trenowanie
skwantyzowanego checkpointu to trenowanie z uwzględnieniem kwantyzacji
(quantization-aware training); dodanie `distill_model=<teacher>` zmienia je
w destylację z uwzględnieniem kwantyzacji.

### Wynik i kody wyjścia

Wypisywane są: ścieżka zapisu, przepis, tryb wykonania, informacja o tym, czy
kalibracja została przeprowadzona, oraz liczba podmienionych modułów w podziale
na rodzaje. Kod wyjścia to `0` przy powodzeniu, `4`, gdy modelu nie da się
wczytać, `5`, gdy nie powiedzie się kwantyzacja lub zapis, oraz `1` przy
pozostałych błędach wykonania.

Powiązane: [`libreyolo export`](/docs/cli/export), polecenie, które opuszcza
PyTorch i zapisuje zamiast tego artefakt do wdrożenia.
