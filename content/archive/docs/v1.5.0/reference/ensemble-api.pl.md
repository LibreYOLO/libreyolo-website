---
title: API zespołów modeli
seo_title: API LibreEnsemble i operacje fuzji
description: >-
  LibreEnsemble, ExternalDetector oraz trzy operacje fuzji w libreyolo.ops:
  ważona fuzja ramek, jej wariant z ziarnami i fuzja NMS uwzględniająca klasy.
lead: >-
  LibreEnsemble uruchamia kilka detektorów na tym samym obrazie i scala ich
  detekcje w jeden obiekt Results. Fuzja następuje po przetwarzaniu końcowym
  każdego elementu, dlatego każdy z nich zachowuje własny rozmiar wejścia,
  normalizację i tłumienie.
keywords:
  - LibreEnsemble
  - ważona fuzja ramek
  - weighted boxes fusion
  - ExternalDetector
  - libreyolo.ops.fusion
  - min_votes konsensus
last_verified: 1.5.0
verification: >-
  Sygnatury i wartości domyślne odczytano z libreyolo/ensemble/model.py i
  libreyolo/ops/fusion.py w wersji 1.5.0. Założenia projektowe pochodzą z
  docs/adr/0004-model-ensembling.md.
snippets:
  usage:
    - label: Dwa elementy i domyślna fuzja
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])

        # Pojedyncze źródło obrazu zwraca jeden obiekt Results, a nie listę.
        result = ens(SAMPLE_IMAGE, conf=0.25)

        print(result.boxes.xyxy)
        print(result.speed)
    - label: Konsensus i progi poszczególnych elementów
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(
            ["LibreYOLO9t.pt", "LibreYOLO9s.pt"],
            weights=[1.0, 2.0],
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,
        )
        result = ens(SAMPLE_IMAGE, conf=[0.25, 0.4])
        print(len(result))
  ops:
    - label: Operacja fuzji bez udziału modelu
      language: python
      code: >
        import torch

        from libreyolo.ops import weighted_boxes_fusion


        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [12.0, 11.0, 51.0,
        49.0]])

        scores = torch.tensor([0.9, 0.8])

        labels = torch.tensor([0, 0])

        model_ids = torch.tensor([0, 1])


        fused = weighted_boxes_fusion(
            boxes, scores, labels, model_ids, num_models=2, iou_thr=0.55
        )

        print(fused)
source_hash: 3834f628efb1193d
---

## LibreEnsemble

```python
LibreEnsemble(
    members,
    *,
    weights=None,
    fusion="wbf",
    fusion_iou=0.55,
    min_votes=1,
)
```

| Argument | Wartość domyślna | Znaczenie |
|---|---|---|
| `members` | | Co najmniej dwa detektory |
| `weights` | `None` | Współczynniki zaufania dla poszczególnych elementów; w razie pominięcia wszystkie wynoszą `1.0` |
| `fusion` | `"wbf"` | `"wbf"`, `"wbf_seeded"`, `"nms"` lub obiekt wywoływalny |
| `fusion_iou` | `0.55` | Próg IoU grupowania podczas fuzji |
| `min_votes` | `1` | Zachowanie tylko ramek potwierdzonych przez co najmniej tyle elementów |

Elementem może być ścieżka wag rozwiązywana przez fabrykę `LibreYOLO()`, już
utworzony model, wyeksportowany backend lub `ExternalDetector`. Każdy element
musi być modelem zadania detect.

<code-tabs name="usage" />

Konstruktor odrzuca mniej niż dwa elementy, listę `weights` o niewłaściwej
długości, wagę niedodatnią, `min_votes` niebędące dodatnią liczbą całkowitą oraz
`min_votes` większe niż liczba elementów. Ustawienie `fusion="nms"` przy
`min_votes > 1` również zgłasza błąd, ponieważ NMS odrzuca informację
o przynależności do klastra i nie może liczyć głosów.

`weights` skaluje zaufanie przypisane każdemu elementowi. Większa waga przyciąga
scalone współrzędne i wskaźniki pewności w stronę danego elementu. Zgodnie
z konwencją powinny być proporcjonalne do walidacyjnego mAP.

## Przestrzenie klas

Elementy o identycznym `names` są przekazywane bezpośrednio. W przeciwnym razie
przestrzenie klas są łączone według nazw, identyfikatory klas elementów są
mapowane za pomocą tabel wyszukiwania, a scalony `Results.names` jest ich sumą.
Fuzja łączy ramki tylko w obrębie tej samej ujednoliconej klasy, dlatego klasa
znana tylko jednemu elementowi przechodzi bez scalenia. Niezgodność powoduje
zapisanie ostrzeżenia podczas tworzenia.

`min_votes` jest ograniczane dla każdej klasy do liczby elementów, których
przestrzenie etykiet zawierają tę klasę, dzięki czemu konsensus zachowuje sens
przy częściowo wspólnych słownikach.

## Wywoływanie zespołu

```python
ens(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    output_path=None,
    color_format="auto",
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    **kwargs,
)
```

`predict` jest aliasem `__call__`. Zwracany jest zwykły `Results`, którego pole
`speed` rozbija koszt według elementów i dodaje wpis `fusion`. Pojedyncze źródło
obrazu zwraca jeden taki obiekt, lista lub katalog zwraca listę, a `stream=True`
zwraca generator.

Pola `conf`, `iou` i `device` są rozgłaszane do każdego elementu, ale przyjmują
również po jednej wartości na element. Dlatego `conf=[0.25, 0.4]` ustawia próg
0.25 dla elementu 0 i 0.4 dla elementu 1. `imgsz` jest rozgłaszane, gdy jest
liczbą całkowitą lub krotką, a dotyczy poszczególnych elementów tylko jako
lista. Dlatego `imgsz=(480, 640)` oznacza jeden prostokątny rozmiar dla
wszystkich, natomiast `imgsz=[480, 640]` oznacza 480 dla elementu 0 i 640 dla
elementu 1. Każdy wpis musi być prawidłowy dla rodziny danego elementu.

`augment` jest rozgłaszane do elementów obsługujących augmentację podczas testu,
a wyeksportowane backendy je ignorują. `classes` przyjmuje identyfikatory klas
sumy, natomiast `max_det` dotyczy scalonego wyniku. Elementy działają więc
z wysokim limitem, a zespół przycina wynik jeden raz. `batch` jest przyjmowane
dla zgodności API, ale obrazy są przetwarzane kolejno.

`val()` i `export()` zgłaszają `NotImplementedError`. Elementy należy walidować
i eksportować osobno.

## ExternalDetector

```python
ExternalDetector(fn: Callable, names: dict[int, str])
```

Dostosowuje dowolny obiekt wywoływalny detekcji do roli elementu. `fn` przyjmuje
obraz PIL i zwraca `(boxes, scores, labels)`, gdzie ramki mają format xyxy
w pikselach pierwotnego obrazu, a etykiety są identyfikatorami klas poprawnymi
w `names`. Obsługiwane są tensory, tablice i zagnieżdżone listy. LibreYOLO nie
importuje niczego z zewnętrznego kodu.

Adapter waliduje zwracaną wartość: musi to być trójka, ramki muszą mieć kształt
`(N, 4)`, trzy tablice muszą mieć tę samą długość, a każdy identyfikator klasy
musi występować w `names`. Detekcje o wartości mniejszej lub równej `conf` są
odrzucane przed fuzją.

## Operacje fuzji

Prymitywy fuzji są samodzielnymi operacjami torch w `libreyolo.ops`. Nie zależą
od modelu i można je importować osobno, dlatego są eksportowane niezależnie od
zespołu.

<code-tabs name="ops" />

Wszystkie trzy przyjmują te same argumenty pozycyjne: `boxes, scores, labels,
model_ids`, i zwracają `(boxes, scores, labels)`.

| Operacja | Klucz rejestru | Zachowanie |
|---|---|---|
| `weighted_boxes_fusion` | `wbf` | Sekwencyjna ważona fuzja ramek zgodna z publikacją |
| `wbf_seeded` | `wbf_seeded` | Równoległy, jednoprzebiegowy wariant tej samej redukcji |
| `nms_fusion` | `nms` | Łączy wszystko i stosuje NMS uwzględniający klasy |

`FUSIONS` mapuje trzy klucze rejestru na obiekty wywoływalne, a `LibreEnsemble`
wyszukuje tam wartość `fusion=`.

```python
weighted_boxes_fusion(
    boxes, scores, labels, model_ids,
    *,
    weights=None,
    num_models=None,
    iou_thr=0.55,
    skip_box_thr=0.0,
    conf_type="avg",
    min_votes=1,
    models_per_label=None,
    label_weights=None,
)
```

`wbf_seeded` ma identyczną sygnaturę. `nms_fusion` przyjmuje te same argumenty
z wyjątkiem `conf_type` i zgłasza `ValueError`, gdy `min_votes > 1`.

W `weighted_boxes_fusion` detekcje są odwiedzane w kolejności malejącej pewności
przeskalowanej wagą. Każda dołącza do istniejącego klastra, którego bieżącą
scaloną ramkę nakłada najlepiej przy IoU powyżej `iou_thr` i tej samej etykiecie,
albo rozpoczyna nowy klaster. Scalona ramka klastra jest średnią współrzędnych
jego elementów ważoną pewnością, a jej wynik jest ważoną średnią lub wartością
maksymalną ich pewności, przeskalowaną tak, aby ramki potwierdzone przez mniej
modeli otrzymywały niższy wynik.

`wbf_seeded` wybiera ziarna klastrów za pomocą NMS uwzględniającego klasy przy
`iou_thr`, przypisuje każdą detekcję do ziarna tej samej etykiety o najlepszym
IoU, a następnie redukuje każdy klaster w ten sam sposób. Kształty klastrów nie
zmieniają się w trakcie przebiegu, więc cała operacja jest matematyką tensorów
o stałym kształcie. Oba warianty dają ten sam wynik, gdy klastry są jednoznaczne,
i mogą nieznacznie różnić się dla nakładających się łańcuchów klastrów.

`nms_fusion` zachowuje bez zmian ramkę o najwyższej pewności z każdej nakładającej
się grupy. Pole `weights` dla poszczególnych modeli skaluje pewność tylko na
potrzeby rankingu tłumienia, a zachowane ramki utrzymują pierwotne wyniki.

## Własna fuzja

`fusion=` przyjmuje również obiekt wywoływalny o tej samej sygnaturze co powyższe
operacje. Jego nazwa jest zapisywana w `ens.fusion`, a gdy jej nie ma, używana
jest wartość `"custom"`. Zwracana wartość jest walidowana: musi być trójką
`(boxes, scores, labels)` o zgodnych kształtach.
