---
title: Podstawowe pojęcia
seo_title: Podstawowe pojęcia LibreYOLO
description: >-
  Wzajemne powiązania zadań, rodzin modeli, rozmiarów i nazw plików checkpointów
  w LibreYOLO oraz gwarancje poszczególnych poziomów obsługi.
lead: >-
  Każdy model w LibreYOLO opisują cztery pojęcia: wykonywane zadanie, rodzina,
  rozmiar w obrębie rodziny oraz poziom obsługi rodziny. Nazwa pliku checkpointu
  koduje pierwsze trzy.
keywords:
  - pojęcia LibreYOLO
  - zadania LibreYOLO
  - rodziny modeli LibreYOLO
  - nazwy checkpointów LibreYOLO
  - poziomy obsługi LibreYOLO
last_verified: 1.5.0
meta:
  - label: Schemat nazwy pliku
    value: 'Libre<FAMILY><size>[-<task>].pt'
    mono: true
  - label: Zadania kanoniczne
    value: 17
  - label: Poziomy obsługi
    value: 'Flagowy, Rdzeń, Obsługiwany, Tylko inferencja, Muzeum, Sąsiedni poziom'
snippets:
  inspect:
    - label: Lista rodzin
      language: bash
      code: >
        # Zadania, rozmiary i rozdzielczości wejściowe każdej zarejestrowanej
        rodziny.

        libreyolo models
    - label: Jeden model
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        print(model.family, model.size, model.task)
        print(model.input_size)
        print(model.nb_classes, model.names[0])
    - label: Wybór zadania
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Aliasy są normalizowane na granicy API: "keypoints" rozwiązuje się do
        # "pose", "det" do "detect", a "semantic-segmentation" do "semantic".
        model = LibreYOLO("LibreYOLO9t.pt", task="det")
        print(model.task)
source_hash: 23d045463a6a8411
---

## Zadania

Zadanie określa, co zwraca model. LibreYOLO ma siedemnaście kanonicznych nazw
zadań, a każda odpowiada polu obiektu `Results`, które zawiera wynik.

| Zadanie | Zwraca |
|---|---|
| `detect` | Ramki wyrównane do osi wraz z klasą i pewnością |
| `segment` | Maski poszczególnych instancji, po jednej na wykryty obiekt |
| `semantic` | Jedną etykietę klasy na piksel, bez rozdzielania instancji |
| `panoptic` | Jedną nienakładającą się etykietę na piksel, łączącą policzalne elementy thing z amorficznymi obszarami stuff |
| `pose` | Punkty kluczowe poszczególnych instancji, wyrównane wierszami z ramkami |
| `classify` | Rozkład prawdopodobieństwa na zbiorze etykiet dla całego obrazu |
| `obb` | Obrócone ramki z kątem obrotu |
| `point` | Jedną współrzędną obrazu na detekcję zamiast ramki |
| `depth` | Gęstą mapę względnej odwrotności głębi |
| `normal` | Gęste pole jednostkowych wektorów normalnych powierzchni |
| `edge` | Gęstą mapę prawdopodobieństwa krawędzi |
| `restore` | Przywrócony obraz RGB po usunięciu rozmycia, odszumieniu lub zwiększeniu rozdzielczości |
| `matte` | Miękką mapę pierwszego planu od 0 do 1 do usuwania tła |
| `ocr` | Czworokąty tekstu z transkrypcjami w kolejności czytania |
| `embed` | Wektor znormalizowany normą L2, którego iloczyn skalarny mierzy zgodność |
| `gaze` | Kierunek spojrzenia dla każdej wykrytej twarzy |
| `mesh` | Ustawione w pozie ciało 3D każdej wykrytej osoby |

Są to nazwy występujące w metadanych checkpointów i nazwach plików. Znane
aliasy są akceptowane wszędzie tam, gdzie przekazuje się zadanie, i
normalizowane przed każdą inną operacją: `detection` i `det` stają się
`detect`, `keypoints` staje się `pose`, `cls` staje się `classify`, `deblur`,
`denoise` i `super-resolution` stają się `restore`, a `face-recognition`
i `reid` stają się `embed`. Nierozpoznana nazwa zgłasza błąd zamiast po cichu
wybrać wartość domyślną.

`segment`, `semantic` i `panoptic` to trzy różne zadania, a nie trzy określenia
tego samego. Maski instancji, etykiety poszczególnych pikseli oraz połączona
mapa thing i stuff mają inne dane referencyjne, inne metryki i inne pola wyniku.

## Rodziny modeli

Rodzina jest jedną linią architektury z własnym kodem wczytywania,
przetwarzania wstępnego i końcowego. Każda rodzina deklaruje identyfikator
`FAMILY`, taki jak `yolo9`, `rfdetr` lub `dfine`, obsługiwane zadania oraz
rozdzielczość wejściową każdego dostarczanego rozmiaru.

`LibreYOLO()` jest fabryką, a nie klasą. Po otrzymaniu ścieżki wczytuje plik,
identyfikuje rodzinę na podstawie metadanych checkpointu lub, jeśli to się nie
uda, na podstawie samych kluczy tensorów, i zwraca instancję modelu tej rodziny.
Dlatego zmiana detektora wymaga jednej linii: zwrócony obiekt udostępnia ten sam
interfejs `predict`, `train`, `val` i `export` oraz zwraca ten sam typ `Results`.

<code-tabs name="inspect" />

Rodzina obsługująca więcej niż jedno zadanie zwykle publikuje osobny checkpoint
dla każdego zadania, często z innym zestawem rozmiarów. Kilka rodzin współdzieli
zamiast tego jeden artefakt między dwoma zadaniami środowiska uruchomieniowego.
W obu przypadkach obsługiwane zadania tworzą stałą listę. Żądanie zadania spoza
niej zgłasza błąd z listą obsługiwanych zadań w komunikacie, zamiast wczytać
przybliżony odpowiednik.

Pełna lista z benchmarkami poszczególnych rodzin i opublikowanymi wagami
znajduje się na stronie [wszystkich modeli](/docs/models).

## Rozmiary

Rozmiar jest wariantem w obrębie rodziny, zapisanym jako kod małymi literami
dołączony bezpośrednio do prefiksu rodziny. Popularne litery to `n` dla nano,
`t` dla tiny, `s` dla small, `m` dla medium, `l` dla large i `x` dla xlarge,
ale kody są właściwe dla rodzin, a kilka rodzin używa zupełnie innych. Mogą to
być kody z nazwą backbone, takie jak `r50` lub `r101`, gdy rozmiar oznacza
głębokość ResNet, kody skalowania złożonego od `b0` do `b3` albo nazwa
identyfikująca jedyny wydany checkpoint. YOLOv9 używa `c` dla compact tam,
gdzie inne rodziny używają `l`.

Rozmiar ustala również rozdzielczość wejściową, a dla rodzin z kilkoma zadaniami
rozdzielczość może zależeć od zadania. Obie wartości są odczytywane z rodziny,
nigdy przyjmowane z góry. Polecenie `libreyolo models` je wyświetla.

## Nazwy plików checkpointów

Każdy opublikowany plik wag jest zgodny z jednym schematem:

```text
Libre<FAMILY><size>[-<task>].pt
```

Prefiks rodziny jest stałym ciągiem dla rodziny, rozmiar jest zapisany małymi
literami i dołączony bez separatora, a sufiks zadania jest poprzedzony łącznikiem.
Detekcja nie ma sufiksu, zgodnie z konwencją zawsze stosowaną w checkpointach
YOLO. Dlatego `LibreYOLO9t.pt` jest detektorem, a `LibreRFDETRn-seg.pt` modelem
segmentacji z tej samej rodziny.

| Zadanie | Sufiks |
|---|---|
| `detect` | |
| `segment` | `-seg` |
| `semantic` | `-sem` |
| `panoptic` | `-panoptic` |
| `pose` | `-pose` |
| `classify` | `-cls` |
| `gaze` | `-gaze` |
| `obb` | `-obb` |
| `point` | `-point` |
| `depth` | `-depth` |
| `edge` | `-edge` |
| `normal` | `-normal` |
| `restore` | `-restore` |
| `matte` | `-matte` |
| `ocr` | `-ocr` |
| `embed` | `-embed` |
| `mesh` | `-mesh` |

Rodzina bez zadania działającego bez sufiksu może wymagać sufiksu, dlatego nazwa
bez niego nie jest akceptowana jako prawidłowy checkpoint tej rodziny. Rodzina
publikująca wagi trenowane na zbiorze innym niż domyślny dołącza nazwę zbioru
danych jako kolejny sufiks, a ten wariant pozostaje częścią nazwy repozytorium,
z którego pobierany jest plik.

Trzy poziomy nie korzystają z tego schematu. Rodziny segmentacji sterowanej
podpowiedziami, rodziny wizyjno-językowe i detektory z otwartym słownikiem nie są
rejestrowane w fabryce checkpointów i nie emitują pliku
`Libre<FAMILY><size>.pt`. Ich prefiks nazywa zamiast tego pobraną migawkę
Hugging Face lub checkpoint sterowany podpowiedziami, a wielkość liter marki ze
źródła nadrzędnego jest tam celowo zachowywana.

## Sposób ustalania zadania

Gdy zadanie może określać kilka sygnałów, są sprawdzane w stałej kolejności,
a pierwszy obecny wygrywa: przekazany argument `task`, następnie zadanie zapisane
w metadanych checkpointu, sufiks zadania w nazwie pliku i domyślne zadanie
rodziny. Wynik jest sprawdzany względem zadań obsługiwanych przez rodzinę przed
zbudowaniem modelu, dlatego niezgodność kończy się błędem podczas wczytywania,
a nie nieprawidłowym wynikiem później.

## Poziomy obsługi

Rodziny są przypisane dokładnie do jednego poziomu. Poziom jest stwierdzeniem
o uwadze inżynieryjnej, a nie o dokładności. Wskazuje, gdzie nowa funkcja trafia
najpierw i co jest utrzymywane w stanie poprawnym.

| Poziom | Znaczenie |
|---|---|
| Flagowy | Funkcje są projektowane i najpierw w pełni walidowane na GPU tutaj |
| Rdzeń | Główne detektory obsługujące trenowanie. Funkcje podążają za rodzinami flagowymi w tej samej fali wydania |
| Obsługiwany | Dodatkowe rodziny obsługujące trenowanie. Utrzymywane w stanie poprawnym w CI, funkcje trafiają zależnie od możliwości |
| Tylko inferencja | Predykcja, walidacja i eksport. Funkcje trenowania nie mają zastosowania |
| Muzeum | Zamrożony eksponat. Tylko poprawki błędów |
| Sąsiedni poziom | Osobny interfejs produktu z własną fabryką i kontraktem |

Każda strona modelu podaje poziom rodziny w nagłówku. Dwie rodziny flagowe to
[YOLOv9](/docs/models/yolov9) dla detektorów CNN oraz
[RF-DETR](/docs/models/rf-detr) dla detektorów transformerowych. Jeśli nie ma
powodu, aby wybrać inaczej, warto zacząć od nich.

Tylko inferencja określa brakującą funkcję, czyli pętlę trenowania w LibreYOLO.
Predykcja, walidacja i, jeśli rodzina go obsługuje, eksport działają. Wywołanie
`train()` dla takiej rodziny zgłasza `NotImplementedError` z przyczyną.
