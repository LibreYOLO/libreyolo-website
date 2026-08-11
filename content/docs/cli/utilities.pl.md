---
title: narzędzia libreyolo
seo_title: Dokumentacja poleceń narzędziowych CLI libreyolo
description: >-
  Małe polecenia biblioteki LibreYOLO: version, checks, models, formats, cfg,
  info, metadata, enroll i compare, każde z argumentami i wartościami
  domyślnymi.
lead: >-
  Dziewięć poleceń, które raportują lub sprawdzają, zamiast liczyć. Wypisują
  informacje o środowisku, spis modeli i formatów, rozwiązane wartości domyślne
  oraz szczegóły checkpointu, a także budują i przeszukują galerię twarzy.
keywords:
  - libreyolo version
  - libreyolo checks
  - lista modeli libreyolo
  - formaty eksportu libreyolo
  - domyślne ustawienia libreyolo cfg
  - metadane checkpointu yolo
  - rozpoznawanie twarzy libreyolo enroll
  - porównanie twarzy libreyolo
last_verified: 1.5.0
meta:
  - label: Polecenia
    value: 'version, checks, models, formats, cfg, info, metadata, enroll, compare'
    mono: true
  - label: Wyjście
    value: >-
      stdout, w postaci tekstu lub z json=true jako jeden obiekt z
      schema_version
snippets:
  examples:
    - label: Środowisko
      language: bash
      code: |
        libreyolo version
        libreyolo checks
    - label: Co jest dostępne
      language: bash
      code: |
        libreyolo models
        libreyolo formats family=yolo9 task=detect
    - label: Sprawdzenie checkpointu
      language: bash
      code: |
        libreyolo info model=LibreYOLO9s.pt
        libreyolo metadata path=weights/LibreYOLO9s.pt
source_hash: 7b5b53c46df00c06
---

## Składnia

```bash
libreyolo <command> [key=value ...]
```

Argumenty to pary `key=value`, a forma POSIX też działa, więc `model=x` i
`--model x` to ten sam argument. Każde polecenie z tej strony wypisuje wyniki na
stdout i przyjmuje `json=true` oraz `quiet=true`.

Polecenie główne ma jedną własną flagę, `libreyolo --version`, która wypisuje
ciąg z wersją i kończy działanie. Jest to mniejsze wyjście niż w przypadku
opisanego niżej polecenia `version`.

## version

Wypisuje wersję biblioteki LibreYOLO oraz wersje Pythona, biblioteki torch i
CUDA, z którymi działa.

```bash
libreyolo version
```

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `json` | `false` | Wyjście JSON na stdout |
| `quiet` | `false` | Wyciszenie stderr |

## checks

Wypisuje środowisko bardziej szczegółowo: Python, torch, CUDA, cuDNN, każde
wykryte GPU z nazwą i pamięcią oraz zainstalowaną wersję każdego opcjonalnego
pakietu, z którego korzystają ścieżki eksportu.

```bash
libreyolo checks
```

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `json` | `false` | Wyjście JSON na stdout |
| `quiet` | `false` | Wyciszenie stderr |

Lista pakietów obejmuje `onnx`, `onnxruntime`, `tensorrt`, `openvino`,
`paddlepaddle`, `x2paddle`, `mnn`, `ncnn`, `onnx2tf`, `ai-edge-litert`,
`transformers` i `scipy`. Pakiet, który nie jest zainstalowany, jest zgłaszany
jako brakujący, a nie pomijany, więc nieudany eksport można powiązać z brakującą
zależnością na podstawie tego jednego polecenia.

## models

Wypisuje każdą rodzinę modeli wraz z jej zadaniami, rozmiarami, nazwami CLI,
które wskazują na jej checkpointy, oraz rozdzielczością wejściową każdego
rozmiaru.

```bash
libreyolo models
```

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `json` | `false` | Wyjście JSON na stdout |
| `quiet` | `false` | Wyciszenie stderr |

Rodzina, której opcjonalna zależność nie jest zainstalowana, jest wypisana jako
niedostępna razem z linią `pip install`, która ją udostępni. Nazwy CLI to
skróty przyjmowane przez `model=`: `yolox-s` wskazuje na `LibreYOLOXs.pt`, a
zadania inne niż detekcja mają sufiks swojego zadania.

## formats

Wypisuje formaty eksportu, które potrafi wygenerować zainstalowane środowisko,
wraz z rozszerzeniem pliku każdego formatu i informacją, czy obsługuje on FP16
i INT8.

```bash
libreyolo formats [family=<family>] [task=<task>]
```

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `family` | | Pokazanie poziomów wsparcia dla jednej rodziny modeli. `model=` jest przyjmowane jako ta sama opcja |
| `task` | | Kanoniczne zadanie modelu. Gdy nie podano, domyślne zadanie rodziny |
| `json` | `false` | Wyjście JSON na stdout |
| `quiet` | `false` | Wyciszenie stderr |

Bez `family` wyjściem jest sam spis formatów. Z nim każdy format zyskuje poziom
wsparcia dla tej rodziny i tego zadania, uzasadnienie tego poziomu oraz
ewentualne powiązane z nim ograniczenie. Nieznana rodzina albo zadanie, którego
rodzina nie obsługuje, to błąd użycia.

Aliasy formatów pojawiają się obok nazwy kanonicznej: `engine` dla `tensorrt`,
`litert` dla `tflite`.

## cfg

Wypisuje rozwiązaną konfigurację domyślną: domyślne ustawienia trenowania,
domyślne ustawienia walidacji, domyślne ustawienia predykcji oraz nadpisania
dla poszczególnych rodzin.

```bash
libreyolo cfg
```

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `json` | `false` | Wyjście JSON na stdout |
| `quiet` | `false` | Wyciszenie stderr |

Wartości są odczytywane z dataclass konfiguracji, a nie z kopii, więc jest to
źródło prawdy o tym, czego użyje trenowanie, gdy nie poda się argumentu.
`family_overrides` to sekcja, która odpowiada na pytanie, dlaczego rodzina
trenowała się z ustawieniami, o które nikt nie prosił. Zobacz
[`libreyolo train`](/docs/cli/train), aby poznać sposób stosowania tych
nadpisań.

## info

Ładuje model na CPU i podaje jego rodzinę, rozmiar, liczbę parametrów, klasy
oraz poziom wsparcia eksportu dla każdego formatu.

```bash
libreyolo info model=<name|path>
```

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `model` | | Nazwa modelu lub ścieżka do wag. Wymagane |
| `detailed` | `false` | Dołączenie szczegółów dla poszczególnych parametrów |
| `json` | `false` | Wyjście JSON na stdout |
| `quiet` | `false` | Wyciszenie stderr |

## metadata

Odczytuje metadane checkpointu bez budowania modelu i sprawdza ich zgodność ze
schematem checkpointu biblioteki LibreYOLO.

```bash
libreyolo metadata path=<checkpoint.pt>
```

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `path` | | Ścieżka do checkpointu `.pt`. Wymagane |
| `json` | `false` | Wyjście JSON na stdout |
| `quiet` | `false` | Wyciszenie stderr |

Duże wpisy zawierające tensory są podsumowywane, a nie wypisywane w całości,
więc wyjście pozostaje czytelne także dla pełnego checkpointu z trenowania.
Nieistniejący checkpoint kończy działanie kodem `checkpoint_not_found`, a taki,
którego metadane nie przechodzą walidacji, wypisuje błędy i kończy działanie
kodem `1`.

## enroll

Buduje galerię twarzy z drzewa katalogów, w którym jeden katalog odpowiada
jednej osobie, dzięki czemu późniejsze predykcje mogą nazwać znalezione twarze.

```bash
libreyolo enroll model=<embedder> source=<people-dir> gallery=<gallery.npz>
```

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `model` | | Model do embeddingów twarzy, ścieżka lub nazwa. Wymagane |
| `source` | | Drzewo z jednym katalogiem na osobę, `source/<identity>/*.jpg`. Wymagane |
| `gallery` | | Wyjściowy plik galerii `.npz`. Jeśli istnieje, jest rozszerzany w miejscu. Wymagane |
| `face_detector` | | Detektor twarzy: plik `.onnx` z YuNet albo detektor LibreYOLO. Gdy nie podano, domyślny detektor rodziny |
| `device` | `auto` | Urządzenie: `0`, `cpu`, `mps`, `auto` |
| `json` | `false` | Wyjście JSON na stdout |
| `quiet` | `false` | Wyciszenie stderr |

```bash
# people/ zawiera jeden katalog na tożsamość; nazwa katalogu staje się tożsamością.
libreyolo enroll model=librefacerec-l.onnx source=people/ gallery=people.npz
```

Nazwa podkatalogu jest tożsamością. Obraz referencyjny, na którym nie da się
wykryć twarzy, jest pomijany z komunikatem na stderr, a reszta jest
przetwarzana dalej; źródło bez podkatalogów z tożsamościami albo takie, w
którym nie znaleziono żadnej twarzy, to błąd.

Powstały plik przekaż do
[`libreyolo predict`](/docs/cli/predict) jako `gallery=people.npz`, aby
detekcje niosły tożsamość i wynik dopasowania.

## compare

Podaje podobieństwo kosinusowe między dwoma obrazami twarzy oraz to, czy
przekracza ono próg tej samej tożsamości.

```bash
libreyolo compare model=<embedder> source=<a.jpg> source2=<b.jpg>
```

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `model` | | Model do embeddingów twarzy, ścieżka lub nazwa. Wymagane |
| `source` | | Pierwszy obraz. Wymagane |
| `source2` | | Drugi obraz do porównania. Wymagane |
| `face_detector` | | Detektor twarzy: plik `.onnx` z YuNet albo detektor LibreYOLO |
| `threshold` | `0.4` | Próg podobieństwa kosinusowego dla decyzji o tej samej tożsamości |
| `device` | `auto` | Urządzenie: `0`, `cpu`, `mps`, `auto` |
| `json` | `false` | Wyjście JSON na stdout |
| `quiet` | `false` | Wyciszenie stderr |

```bash
libreyolo compare model=librefacerec-l.onnx source=a.jpg source2=b.jpg
```

`libreyolo verify` jest zarejestrowane jako druga nazwa tego polecenia i
przyjmuje te same argumenty.

Zarówno `compare`, jak i `enroll` wymagają modelu, którego zadaniem jest
embedding twarzy. Wszystko inne kończy działanie kodem `config_unsupported`.
Jako źródła przyjmowane są zarówno lokalne ścieżki do obrazów, jak i adresy URL
`http` oraz `https`.

## Przykłady

<code-tabs name="examples" />

## Uwagi

Wynik trafia na stdout; postęp i ostrzeżenia idą na stderr. `json=true` wypisuje
jeden obiekt z polem `schema_version` i to jest forma przeznaczona do odczytu ze
skryptu. Wyjście tekstowe jest domyślne i jest przeznaczone do czytania przez
człowieka.

Kody wyjścia są zgodne z tą samą mapą co w reszcie CLI: `0` przy powodzeniu, `2`
przy błędzie użycia lub konfiguracji, `3` gdy nie można znaleźć źródła, `4` gdy
nie można wczytać modelu lub checkpointu i `1` przy innych błędach w trakcie
działania.

Powiązane: [`libreyolo doctor`](/docs/cli/doctor), czyli polecenie do inspekcji
po stronie zbioru danych, oraz [`libreyolo profile`](/docs/cli/profile), po
stronie wydajności.
