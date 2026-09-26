---
title: libreyolo profile
seo_title: dokumentacja polecenia libreyolo profile
description: >-
  Pomiar szybkości trenowania i inferencji oraz odczyt wyniku: wszystkie
  podpolecenia profile, ich argumenty i wartości domyślne oraz to, co pokazuje
  każde ujęcie.
lead: >-
  Grupa poleceń, która mierzy, na co schodzi czas w kroku trenowania lub w
  wywołaniu inferencji, zapisuje samodzielny profil i odczytuje go z powrotem
  przez kilka ujęć.
keywords:
  - libreyolo profile cli
  - profilowanie trenowania yolo
  - pomiar opóźnienia inferencji
  - profilowanie kerneli cuda pytorch
  - porównanie wydajności libreyolo
last_verified: 1.5.0
meta:
  - label: Polecenie
    value: libreyolo profile
    mono: true
  - label: Wyjście
    value: profile.json i profile_trace.json w runs/profile
    mono: true
snippets:
  examples:
    - label: Pomiar inferencji
      language: bash
      code: |
        # Brak argumentu source oznacza dołączony przykładowy obraz.
        libreyolo profile infer --device cpu --warmup 5 --runs 20
    - label: Odczyt werdyktu
      language: bash
      code: |
        libreyolo profile summary runs/profile/infer/profile.json
    - label: Porównanie dwóch pomiarów
      language: bash
      code: >
        libreyolo profile infer --device cpu --warmup 5 --runs 20 --project
        runs/profile/a

        libreyolo profile infer --device cpu --warmup 5 --runs 20 --batch 4
        --project runs/profile/b


        libreyolo profile compare runs/profile/a/infer/profile.json \
          runs/profile/b/infer/profile.json
source_hash: b967e869fd9ba418
---

## Składnia

```bash
libreyolo profile <subcommand> [<positional>] [--flag value ...]
```

Ta grupa nie przyjmuje argumentów `key=value`. Jej podpolecenia korzystają z
argumentów pozycyjnych i flag POSIX, więc jest to `--weights LibreYOLO9t.pt`,
a nie `weights=LibreYOLO9t.pt`. Uruchomienie `libreyolo profile` bez
podpolecenia wypisuje ich listę.

Dwa podpolecenia mierzą i zapisują profil; pozostałe go odczytują. `run` i
`infer` emitują ten sam samodzielny plik `profile.json`, więc każde
podpolecenie odczytujące działa z dowolnym z nich.

## profile run

Uruchamia krótkie profilowane trenowanie i zapisuje profil.

```bash
libreyolo profile run <data> [--flag value ...]
```

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `data` | | Pozycyjny. Plik YAML zbioru danych lub nazwa, np. `coco128`. Wymagany |
| `--weights` | `LibreYOLO9t.pt` | Plik wag modelu lub nazwa |
| `--size` | `t` | Wariant rozmiaru modelu |
| `--batch` | `16` | Mikro-batch. `-1` automatycznie dobiera około 70% VRAM |
| `--imgsz` | `640` | Rozmiar obrazu przy trenowaniu |
| `--workers` | `8` | Procesy robocze dataloadera |
| `--amp` | `true` | Użycie ścieżki AMP danej rodziny. `--no-amp` to wyłącza |
| `--steps` | `20` | Kroki profilowane, czyli mierzone |
| `--warmup` | `5` | Kroki rozgrzewki przed pomiarem |
| `--repeat` | `1` | Powtórzenie N razy dla średniej i odchylenia standardowego |
| `--device` | `0` | Urządzenie |
| `--project` | `runs/profile` | Katalog główny wyników |
| `--json` | `false` | Wyjście JSON na stdout |

Mierzone okno to `--warmup` plus `--steps` iteracji. Zbiór danych zbyt mały,
aby je wypełnić, nie daje profilu, a polecenie kończy się kodem `3` i wskazuje
trzy wyjścia: większy zbiór danych, mniej kroków lub mniejszy batch.

`--repeat` powyżej 1 zapisuje zagregowany plik
`runs/profile/profile_repeat.json`, w którym metryki skalarne są uśredniane po
próbach, a listy kerneli pochodzą z ostatniej próby. Jest to również warunek
konieczny werdyktu o istotności w `compare`: pojedynczy przebieg nie może go
dostarczyć.

## profile infer

Profiluje ścieżkę inferencji i zapisuje profil.

```bash
libreyolo profile infer [<source>] [--flag value ...]
```

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `source` | | Pozycyjny. Obraz lub katalog. Po pominięciu dołączony przykładowy obraz |
| `--weights` | `LibreYOLO9t.pt` | Plik wag modelu lub nazwa |
| `--size` | `t` | Wariant rozmiaru modelu |
| `--batch` | `1` | Liczba obrazów na jedno przejście w przód |
| `--imgsz` | `640` | Rozmiar obrazu wejściowego |
| `--half` | `false` | Przejście w przód z autocastem, tylko CUDA. `--no-half` to wyłącza |
| `--amp-dtype` | `float16` | Typ danych autocastu CUDA: `float16` lub `bfloat16` |
| `--warmup` | `20` | Iteracje rozgrzewki przed pomiarem |
| `--runs` | `100` | Mierzone iteracje |
| `--repeat` | `1` | Powtórzenie N razy dla średniej i odchylenia standardowego |
| `--conf` | `0.25` | Próg pewności, który zmienia ilość pracy NMS |
| `--iou` | `0.45` | Próg IoU dla NMS |
| `--max-det` | `300` | Maksymalna liczba detekcji na obraz, co zmienia ilość pracy NMS |
| `--device` | `0` | Urządzenie |
| `--trace` | `true` | Emisja śladu Chrome do analizy kerneli i operacji. `--no-trace` to pomija |
| `--project` | `runs/profile` | Katalog główny wyników |
| `--json` | `false` | Wyjście JSON na stdout |

Raportuje opóźnienie na poziomie p50, p90 i p99, przepustowość w obrazach na
sekundę oraz podział na etapy: preprocess, forward i postprocess. Trzy
argumenty progowe są tutaj dlatego, że zmieniają wartość dla postprocess.

## profile summary

```bash
libreyolo profile summary <trace> [--json]
```

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `trace` | | Pozycyjny. Ścieżka do pliku `profile.json` lub `profile_trace.json`. Wymagana |
| `--json` | `false` | Wyjście JSON na stdout |

Odczyt ogólny: czas kroku, przepustowość, wykorzystanie GPU, udział Tensor
Core, szczytowe zużycie VRAM, narzut hosta, liczba uruchomień kerneli na krok,
werdykt o wąskim gardle wraz z uzasadnieniem, rozkład kerneli według kategorii
oraz najkosztowniejsze kernele na krok. Dla profilu inferencji wypisuje
dodatkowo percentyle opóźnienia i podział na etapy.

Profil zebrany w warunkach przeciążenia pamięci VRAM jest oznaczany, ponieważ
zmierzonemu tam wykorzystaniu i przepustowości nie można ufać.

## profile get

```bash
libreyolo profile get <trace> [<field>] [--json]
```

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `trace` | | Pozycyjny. Ścieżka do profilu. Wymagana |
| `field` | | Pozycyjny. Nazwa metryki. Pominięcie wypisuje dostępne metryki |
| `--json` | `false` | Wyjście JSON na stdout |

Wypisuje jedną metrykę i nic więcej, z myślą o pętlach w skryptach. Nieznane
pole kończy się kodem `2` i wskazuje formę wypisującą listę.

## profile phases

```bash
libreyolo profile phases <trace> [--json]
```

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `trace` | | Pozycyjny. Ścieżka do profilu. Wymagana |
| `--json` | `false` | Wyjście JSON na stdout |

Milisekundy GPU, milisekundy zegarowe, liczba kerneli i liczba operacji na
fazę: forward, backward, dataload, to_device, optimizer.

## profile kernels

```bash
libreyolo profile kernels <trace> [--flag value ...]
```

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `trace` | | Pozycyjny. Ścieżka do profilu. Wymagana |
| `--top` | `20` | Pokazanie N pozycji o najdłuższym czasie GPU |
| `--category` | | Filtrowanie po fragmencie nazwy kategorii: `gemm`, `layout`, `norm`, `elementwise` |
| `--grep` | | Filtrowanie wyrażeniem regularnym po nazwie kernela |
| `--tensorcore` | `false` | Tylko kernele Tensor Core |
| `--sort` | `time` | `time`, `count` lub `name` |
| `--phase` | | Ograniczenie do jednej fazy: `forward`, `backward`, `dataload`, `to_device`, `optimizer` |
| `--json` | `false` | Wyjście JSON na stdout |

Najniższy poziom analizy: pojedyncze kernele GPU wraz z ich udziałem w czasie
GPU, milisekundami na krok, liczbą wywołań na krok i kategorią. Nieznana
wartość `--phase` kończy się kodem `2` i wypisuje fazy obecne w profilu.

## profile ops

```bash
libreyolo profile ops <trace> [--flag value ...]
```

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `trace` | | Pozycyjny. Ścieżka do profilu. Wymagana |
| `--top` | `20` | Pokazanie N pozycji o najdłuższym czasie CPU |
| `--phase` | | Ograniczenie do jednej fazy |
| `--json` | `false` | Wyjście JSON na stdout |

Widok frameworku zamiast widoku urządzenia: operacje `aten` i autograd
uszeregowane według czasu CPU, w którym ujawnia się koszt uruchamiania po
stronie hosta.

## profile compare

```bash
libreyolo profile compare <before> <after> [--json]
```

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `before` | | Pozycyjny. Profil bazowy. Wymagany |
| `after` | | Pozycyjny. Nowy profil. Wymagany |
| `--json` | `false` | Wyjście JSON na stdout |

Porównuje przepustowość, milisekundy na obraz, wykorzystanie GPU, narzut hosta,
liczbę uruchomień kerneli na krok i werdykt o wąskim gardle.

Ocena istotności wymaga, aby obie strony zmierzono z `--repeat` co najmniej 2.
Przy takim pomiarze różnicę uznaje się za istotną, gdy przekracza dwukrotność
łącznego błędu standardowego, a wyjście wypisuje wykonane porównanie. Bez tego
w tym wierszu widnieje informacja, że pojedynczy przebieg nie pozwala na taką
ocenę.

## profile what-if

```bash
libreyolo profile what-if <trace> [--flag value ...]
```

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `trace` | | Pozycyjny. Ścieżka do profilu. Wymagana |
| `--remove-category` | | Prognoza usunięcia kategorii kerneli: `gemm`, `layout`, `norm`, `elementwise` |
| `--remove-launches` | | Prognoza usunięcia N uruchomień kerneli na krok, na przykład zysku z fuzji operacji |
| `--json` | `false` | Wyjście JSON na stdout |

Szacuje, co dałaby zmiana, zanim zostanie ona napisana. Jedna z dwóch opcji
jest wymagana; brak którejkolwiek kończy się kodem `2`.

Prognoza podąża za werdyktem samego profilu. Poniżej 80% wykorzystania GPU
modeluje oszczędność jako mniejszą liczbę uruchomień pomnożoną przez zmierzony
koszt hosta na jedno uruchomienie; powyżej tego progu jako mniejszą pracę GPU.
Wynik zawiera pole z zastrzeżeniem, ponieważ koszt na jedno uruchomienie jest
przybliżeniem, a jedynym dowodem pozostaje drugi pomiar.

## Przykłady

<code-tabs name="examples" />

## Uwagi

Profiler mierzy i raportuje. Niczego nie zmienia: odczyt werdyktu, edycja
konfiguracji lub kodu, ponowne uruchomienie i porównanie to pętla, do której
został zbudowany.

`--device` domyślnie ma wartość `0`, czyli urządzenie CUDA 0. Podanie
`--device cpu` wykonuje pomiar na CPU i daje profil, który podpolecenia
odczytujące nadal przyjmują, ale bez szczegółów dotyczących kerneli GPU.

Każde podpolecenie obsługuje `--json`, a te odczytujące wypisują wyłącznie na
stdout, co czyni tę grupę użyteczną w skrypcie.

Kody wyjścia są tutaj własne dla tej grupy: `2` dla nieistniejącego pliku lub
argumentu, którego nie da się rozwiązać, `3` gdy `run` nie wyprodukował
profilu, oraz `1` gdy śladu nie da się przeanalizować.

Powiązane: [`libreyolo train`](/docs/cli/train), którego argumenty zwykle stroi
się na podstawie profilu trenowania.
