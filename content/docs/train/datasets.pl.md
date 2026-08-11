---
title: Zbiory danych
seo_title: Zbiory danych do trenowania w LibreYOLO
description: >-
  Plik YAML zbioru danych odczytywany przez LibreYOLO, oczekiwany układ
  katalogów, automatyczne pobieranie oraz polecenie doctor sprawdzające zbiór
  przed trenowaniem.
lead: >-
  Zbiór danych LibreYOLO jest plikiem YAML wskazującym katalog główny, podziały
  i nazwy klas. Wszystkie pozostałe informacje, w tym położenie plików etykiet,
  są wyprowadzane z tego pliku według konwencji.
keywords:
  - format zbioru danych yolo
  - data.yaml
  - trenowanie na własnym zbiorze danych
  - format etykiet yolo
  - zbiór danych coco json
  - automatyczne pobieranie zbioru danych
  - libreyolo doctor
  - sprawdzanie nierównowagi klas
  - wyciek między train val
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Działa nazwa dołączona do pakietu, ścieżka względna lub ścieżka
        bezwzględna.

        model.train(data="coco8.yaml", epochs=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10
  doctor:
    - label: Sprawdzenie zbioru danych
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml
    - label: Kończenie zadania CI błędem także przy ostrzeżeniach
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml strict=true json=true
    - label: Pomijanie dekodowania obrazów
      language: bash
      code: >
        # Odczytuje tylko etykiety i YAML. Sprawdzenia uszkodzeń, duplikatów i
        wycieku

        # między podziałami wymagają pikseli, dlatego są pomijane.

        libreyolo doctor my-dataset.yaml fast=true
    - label: Python
      language: python
      code: |
        from libreyolo import doctor

        report = doctor.diagnose("my-dataset.yaml", imgsz=640)

        for finding in report.findings:
            print(finding.severity.value, finding.check_id, finding.message)

        raise SystemExit(report.exit_code(strict=False))
source_hash: 9a12a0551c8b56e9
---

## Wskazywanie zbioru danych do trenowania

`data=` przyjmuje ścieżkę YAML lub nazwę konfiguracji dołączonej do pakietu.

<code-tabs name="train" />

Nazwa jest rozwiązywana w stałej kolejności: istniejąca ścieżka bezwzględna,
następnie podana nazwa względem katalogu roboczego, ta sama nazwa z dołączonym
`.yaml`, a na końcu katalog dołączonych konfiguracji. Jeśli nic nie pasuje,
komunikat błędu wskazuje każdy przeszukany katalog i wymienia dołączone
konfiguracje.

## Dołączone konfiguracje

Pakiet zawiera trzynaście konfiguracji zbiorów danych w
`libreyolo/config/datasets/`.

| Konfiguracja | Zadanie | Uwagi |
|---|---|---|
| `coco8.yaml` | detect | 8 obrazów, pobieranie ze zwykłego adresu URL |
| `coco128.yaml` | detect | 128 obrazów |
| `coco1000.yaml` | detect | 800 train, 200 val |
| `coco5000.yaml` | detect | 4000 train, 1000 val |
| `coco.yaml` | detect | pełny COCO 2017 |
| `coco-val-only.yaml` | detect | tylko val2017 |
| `coco8-pose.yaml` | pose | 8 obrazów, punkty kluczowe COCO-17 |
| `coco-pose.yaml` | pose | punkty kluczowe COCO 2017 |
| `ade20k.yaml` | semantic | 150 klas |
| `cityscapes.yaml` | semantic | 19 klas, pobieranie ręczne |
| `cocostuff.yaml` | semantic | 182 klasy, pobieranie ręczne |
| `gopro.yaml` | restore | pary do usuwania rozmycia |
| `sr8.yaml` | restore | pary do superrozdzielczości |

Tylko `coco8.yaml` i `coco128.yaml` zawierają zwykły adres URL pobierania.
Pozostałe zawierają blok pobierania w Pythonie, który wymaga zgody opisanej
poniżej, albo oczekują, że dane już znajdują się na dysku.

## Położenie zbioru danych na dysku

Klucz `path` w pliku YAML wskazuje katalog główny zbioru danych. Ścieżka
bezwzględna jest używana bez zmian. Ścieżka względna jest najpierw wyszukiwana w
katalogu zbiorów danych, a następnie obok samego pliku YAML. Zbiór przeznaczony
do pobrania trafia do katalogu zbiorów danych.

Katalogiem tym jest `~/datasets`, co można zastąpić zmienną środowiskową
`LIBREYOLO_DATASETS_DIR`. Nie ma dla niego pliku ustawień.

## Klucze YAML

```yaml
path: my-dataset        # katalog główny zbioru danych
train: images/train     # wymagane do trenowania
val: images/val         # wymagane do walidacji
test: images/test       # opcjonalne
nc: 3                   # opcjonalne; musi być zgodne z names
names:
  0: person
  1: helmet
  2: vest
download: https://example.com/my-dataset.zip   # opcjonalne
```

Pola `train`, `val` i `test` przyjmują katalog obrazów, plik `.txt` z jedną
ścieżką obrazu w każdym wierszu albo listę łączącą oba warianty. Wiersze listy
`.txt` mogą być względne. Są wtedy rozwiązywane względem katalogu samego pliku
listy. Wiersze zaczynające się od `#` są pomijane.

`names` może być listą lub mapowaniem z kluczami całkowitoliczbowymi. `nc` jest
opcjonalne. Gdy oba pola są obecne i niezgodne, doctor zgłasza błąd.

## Układ katalogów i pliki etykiet

Detekcja, segmentacja, estymacja pozy i obrócone ramki używają wspólnego układu.
Ścieżka etykiety jest wyprowadzana ze ścieżki obrazu przez zastąpienie składnika
katalogu `images` składnikiem `labels` i zmianę rozszerzenia na `.txt`:

```text
my-dataset/
  images/train/0001.jpg   ->   labels/train/0001.txt
  images/val/0002.jpg     ->   labels/val/0002.txt
```

Zastępowany jest tylko pełny składnik ścieżki `images`, dlatego katalog o nazwie
`images_old` pozostaje bez zmian.

Wiersz detekcji zawiera pięć pól, wszystkie znormalizowane do `[0, 1]` względem
szerokości i wysokości oryginalnego obrazu:

```text
<class_id> <cx> <cy> <w> <h>
```

Brakujący lub pusty plik etykiet oznacza, że obraz nie zawiera obiektów. Jest
trenowany jako tło zamiast powodować błąd. Wiersz z więcej niż pięcioma polami
jest odczytywany jako wielokąt, a jego ramką staje się zewnętrzny obrys
wielokąta. Dzięki temu eksport segmentacji użyty do trenowania detekcji wczytuje
się bez ostrzeżeń. Doctor raportuje, ile wierszy przeszło tą ścieżką.

## Inne zadania

Segmentacja zachowuje ten sam układ z wierszami wielokątów
`<class_id> <x1> <y1> ... <xN> <yN>` zawierającymi co najmniej trzy punkty.
Pięciopolowy wiersz detekcji jest akceptowany i oznacza prostokątną instancję.

Estymacja pozy dodaje do pliku YAML `kpt_shape: [K, D]` i opcjonalną permutację
`flip_idx`. Każdy wiersz zawiera dokładnie `5 + K * D` pól: ramkę, a następnie
`K` punktów kluczowych `x y` lub `x y v`, z widocznością `0`, `1` albo `2`.

Obrócone ramki używają dokładnie dziewięciu pól: klasy, po której następują
cztery punkty narożne w znormalizowanych współrzędnych. Kąt nie jest zapisywany
w pliku.

Segmentacja semantyczna paruje każdy obraz z jednokanałową maską o tej samej
rozdzielczości, rozwiązywaną przez zastąpienie `images` wartością `masks_dir`
(domyślnie `masks`). Wartość piksela `255` oznacza ignorowanie. `label_mapping`
mapuje identyfikatory źródłowe na identyfikatory trenowania podczas wczytywania.

Klasyfikacja używa drzewa ImageFolder zamiast plików etykiet. Katalogi `train/`
i `val/` zawierają po jednym katalogu na klasę. Mapowanie klas na indeksy wynika
z posortowanej kolejności nazw katalogów.

Rekonstrukcja paruje zdegradowane wejście z czystym celem o identycznej
rozdzielczości przez `input_dir` i `target_dir`. Głębia, normalne powierzchni i
krawędzie parują obraz z gęstą mapą przez własny klucz katalogu.

Pełny kontrakt dla poszczególnych zadań, w tym konwencje skali głębi i kodowanie
identyfikatorów segmentów w plikach PNG segmentacji panoptycznej, znajduje się w
`docs/dataset_schema.md` w repozytorium biblioteki.

## Natywny format COCO JSON

Pliku adnotacji COCO JSON można użyć bezpośrednio. Dodaj mapowanie `annotations`,
a ścieżka podziału stanie się katalogiem głównym obrazów:

```yaml
path: my-dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

Jeśli `names` jest obecne, nazwy kategorii w pliku JSON muszą być z nim zgodne,
a `names` definiuje identyfikatory etykiet przewidywane przez model. Bez `names`
identyfikatory kategorii COCO są sortowane i gęsto mapowane do `0..N-1`.

Ta ścieżka oczekuje jednego katalogu obrazów na podział. Lista ścieżek lub lista
obrazów `.txt` zgłasza błąd zamiast po cichu wczytać inny zestaw.

## Automatyczne pobieranie

Zbiór danych jest uznawany za obecny, gdy jego ścieżka `train` lub `val`
rozwiązuje się do niepustego katalogu albo istniejącego pliku. Jeśli tak nie
jest, a plik YAML ma klucz `download`, jego wartość określa dalsze działanie.

Adres URL `http` lub `https` jest pobierany, a jeśli prowadzi do pliku zip,
archiwum jest rozpakowywane w katalogu głównym zbioru danych. Każda inna wartość
jest traktowana jako osadzony skrypt Pythona i uruchamiana tylko przy
`allow_download_scripts=True`. Bez tego skrypt jest pomijany z ostrzeżeniem, a
trenowanie jest kontynuowane na danych znajdujących się na dysku.

```bash
libreyolo train model=LibreYOLO9s.pt data=coco.yaml allow_download_scripts=true
```

Flaga jest bramą wykonania kodu, a nie dostępu do sieci. Pobieranie z adresu URL
odbywa się niezależnie od niej. Wymagają jej bloki `download: |`. CLI wyświetla
ostrzeżenie, gdy flaga jest włączona, a doctor nigdy jej nie włącza.

## Sprawdzanie zbioru danych przed trenowaniem

`libreyolo doctor` odczytuje zbiór danych detekcji i zgłasza potencjalne problemy
przed użyciem GPU. Kończy działanie kodem 1 po znalezieniu błędów, dlatego może
służyć jako brama CI.

<code-tabs name="doctor" />

Sprawdzenia dzielą się na sześć rodzin:

| Rodzina | Wyszukiwane problemy |
|---|---|
| `config` | brak `names`, niezgodność `nc` z `names`, brakujące lub puste podziały, zduplikowane nazwy klas |
| `files` | obrazy bez pliku etykiet, etykiety bez obrazu, brakujące obrazy wymienione w podziale, kolizje nazw bazowych |
| `labels` | nieprawidłowe wiersze, identyfikatory klas poza `[0, nc)`, współrzędne poza `[0, 1]`, ramki o zerowym polu, bardzo małe lub duże ramki, zduplikowane ramki, pliki etykiet identyczne bajt po bajcie |
| `balance` | klasy bez instancji lub z małą ich liczbą, współczynnik nierównowagi klas, klasy obecne tylko w jednym podziale, udział obrazów tła |
| `images` | pliki niemożliwe do zdekodowania, obrót EXIF, nietypowe układy kanałów, jednorodne obrazy, dokładne i zbliżone duplikaty |
| `splits` | ten sam obraz występujący w dwóch podziałach, dokładnie lub niemal identycznie |

`--only` i `--skip` przyjmują identyfikator sprawdzenia lub prefiks rodziny,
dlatego `skip=images,labels.tiny_object` jest poprawne. `--fast` usuwa wszystkie
sprawdzenia wymagające dekodowania pikseli, czyli rodziny `images` i `splits`.

Warto znać dwa zachowania. `--strict` sprawia, że kod zakończenia jest błędny
zarówno dla ostrzeżeń, jak i błędów. Ponadto doctor obsługuje wyłącznie zbiory
danych detekcji. Zbiór estymacji pozy, segmentacji lub obróconych ramek jest
odrzucany z komunikatem wskazującym wykryty typ, zamiast być sprawdzany według
niewłaściwego kontraktu.

## Powiązane strony

- [Hiperparametry](/docs/train/hyperparameters) opisują argumenty przyjmowane
  przez `train()` po przygotowaniu danych.
- [Walidacja i metryki](/docs/train/validation) opisują ewaluację na podziale
  `val` lub `test`.

