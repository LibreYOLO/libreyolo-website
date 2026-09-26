---
title: Progi i filtrowanie
seo_title: 'conf, iou i max_det w LibreYOLO'
description: >-
  Co naprawdę robią conf, iou, max_det i classes podczas predykcji, które
  rodziny ignorują iou, ponieważ nie wykonują NMS, oraz dlaczego agnostic_nms
  nic nie robi.
lead: >-
  Cztery argumenty decydują, które predykcje pozostają: conf, iou, max_det i
  classes. Tylko dwa z nich dotyczą każdej rodziny, ponieważ predyktor zbioru
  dekoduje stały zestaw zapytań i nigdy nie wykonuje NMS.
keywords:
  - próg conf YOLO
  - próg iou NMS
  - max_det
  - filtrowanie klas detekcji Python
  - agnostic nms
  - DETR bez NMS
  - próg pewności detekcji
  - filtrowanie klas podczas wnioskowania
last_verified: 1.5.0
verification: >-
  Wartości domyślne pochodzą z InferenceRunner.__call__ w
  libreyolo/models/base/inference.py. Zachowanie NMS dla poszczególnych rodzin
  odczytano ze wszystkich modułów w libreyolo/postprocess/ i porównano z
  _is_nms_free_family w libreyolo/backends/base.py. Filtrowanie klas pochodzi z
  InferenceRunner._apply_classes_filter i _wrap_results. Stan agnostic_nms
  pochodzi z NOOP_PREDICT_KWARGS w libreyolo/utils/predict_args.py. Obsługę
  otwartego słownika zaczerpnięto z NMS_THRESHOLD w
  libreyolo/models/openvocab/base.py. Wartości domyślne walidacji pochodzą z
  BaseModel.val.
snippets:
  basic:
    - label: Cztery argumenty
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(
            SAMPLE_IMAGE,
            conf=0.25,      # zachowaj predykcje z tym wynikiem lub wyższym
            iou=0.45,       # próg nakładania NMS tam, gdzie NMS jest wykonywane
            max_det=300,    # limit na obraz
            classes=None,   # albo lista identyfikatorów klas
        )
        print(len(result.boxes))
    - label: Przeszukiwanie wartości conf
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        for conf in (0.1, 0.25, 0.5, 0.75):
            result = model(SAMPLE_IMAGE, conf=conf)
            print(conf, len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt conf=0.4 iou=0.5 max_det=100 \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  classes:
    - label: Filtrowanie do określonych klas
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # Identyfikatory klas indeksują model.names. W COCO 0 oznacza osobę.
        result = model(SAMPLE_IMAGE, classes=[0])

        print({result.names[int(c)] for c in result.boxes.cls.tolist()})
    - label: Znajdowanie identyfikatora według nazwy
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        wanted = {"person", "backpack"}
        ids = [i for i, name in result.names.items() if name in wanted]
        print(ids)

        filtered = model(SAMPLE_IMAGE, classes=ids)
        print(len(filtered.boxes))
  nmsfree:
    - label: iou w rodzinie bez NMS
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # RF-DETR dekoduje stały zestaw zapytań, więc iou niczego tutaj nie
        zmienia.

        model = LibreYOLO("LibreRFDETRs.pt")


        loose = model(SAMPLE_IMAGE, iou=0.9)

        tight = model(SAMPLE_IMAGE, iou=0.1)


        # Liczba jest taka sama w obu przypadkach. Działają parametry conf i
        max_det.

        print(len(loose.boxes), len(tight.boxes))
source_hash: 0b978963c356027d
---

## Cztery argumenty

| Argument | Wartość domyślna | Zastosowanie |
|---|---|---|
| `conf` | `0.25` | Każda rodzina |
| `iou` | `0.45` | Rodziny wykonujące tłumienie niemaksymalne |
| `max_det` | `300` | Każda rodzina |
| `classes` | `None` | Każda rodzina |

<code-tabs name="basic" />

Dwa z tych argumentów są uniwersalne, a dwa nie. To najważniejsza rzecz, którą
warto wiedzieć przed rozpoczęciem dostrajania.

Walidacja celowo używa innych wartości domyślnych: `val()` działa z `conf=0.001`
i `iou=0.6`, ponieważ średnia precyzja jest obliczana dla pełnej krzywej precyzja-czułość,
a próg 0,25 skróciłby ją.

## conf

`conf` to wynik, poniżej którego predykcja jest odrzucana. Dotyczy każdej rodziny,
w tym tych, które nigdy nie wykonują NMS, i jest pierwszym parametrem do zmiany,
gdy detekcji jest zbyt wiele lub zbyt mało.

Wartość domyślna `0.25` nadaje się do oglądania obrazów. System przekazujący
wyniki dalej zwykle wymaga wartości wyższej, a pomiar dokładności znacznie niższej.

## iou

`iou` określa nakładanie, powyżej którego tłumienie niemaksymalne usuwa ramkę
o niższym wyniku spośród dwóch ramek tej samej klasy. Ma znaczenie tylko wtedy,
gdy rodzina w ogóle wykonuje tłumienie.

Predyktor zbioru dekoduje stałą liczbę zapytań i wybiera te z najwyższymi wynikami.
Duplikaty są tłumione wewnątrz architektury podczas trenowania, a nie na etapie
przetwarzania końcowego, więc nie ma progu do ustawienia. Następujące rodziny
przyjmują `iou` dla zgodności API i je ignorują:

CenterNet, DEIM, DETR, Deformable DETR, D-FINE, DINO-DETR, EdgeCrafter,
Faster R-CNN, LW-DETR, Mask R-CNN, RF-DETR, RT-DETR oraz głowica end-to-end YOLOv9.
Warianty zbudowane na tych dekoderach dziedziczą to zachowanie.

<code-tabs name="nmsfree" />

Większość z nich informuje o tym w docstringach przetwarzania końcowego, ale
podczas działania nie jest zgłaszane ostrzeżenie. Dlatego przeszukiwanie wartości
`iou` dla RF-DETR daje płaską linię zamiast błędu. Faster R-CNN i Mask R-CNN są
nieco innym przypadkiem: oba wykonały już NMS wewnątrz modelu ze stałym progiem
projektu źródłowego, którego `iou` nie może zmienić w obsługiwany sposób.

Następujące rodziny używają tego parametru: od YOLOv1 do YOLOv4, YOLOv7, YOLOv9,
YOLOX, YOLO-NAS, RTMDet, PicoDet, EfficientDet, FCOS, RetinaNet i SSD.

Dwie opcje predykcji sprawiają, że `iou` ma znaczenie nawet dla predyktora zbioru,
ponieważ obie scalają ramki po zakończeniu działania modelu:

- `tiling=True` uzgadnia nakładające się kafelki za pomocą NMS na klasę przy `iou`
- `augment=True` scala odbite widoki za pomocą NMS na klasę przy `iou`

Obie opisano w sekcji [wydajność wnioskowania](/docs/predict/performance).

Detektory z otwartym słownikiem mają własną zasadę. Rodzina, której procesor
wykonuje NMS, deklaruje własny domyślny próg i respektuje `iou`, jak OMDet-Turbo.
Rodziny bez tłumienia, Grounding DINO, OWLv2 i OV-DEIM, zgłaszają ostrzeżenie po
przekazaniu `iou`. Jest to jedyne takie ostrzeżenie w bibliotece.

## max_det

`max_det` ogranicza liczbę predykcji zwracanych dla jednego obrazu. Dotyczy
wszystkich rodzin, ale jest realizowane za pomocą różnych mechanizmów. Rodzina
z NMS przycina wynik po tłumieniu, a predyktor zbioru używa go jako rozmiaru
wyboru top-k.

Niektóre rodziny ograniczają wartość poniżej żądanej, ponieważ tak działa ich
konfiguracja referencyjna. SSD ogranicza ją do 200, segmentacja instancji RTMDet
do 100, a FCOS do własnego limitu detekcji na obraz. Zwiększanie `max_det` ponad
te wartości nie ma wpływu.

Jedynym miejscem, w którym `max_det` jest stosowane centralnie zamiast osobno dla
rodziny, jest wnioskowanie kafelkowe. Scalona lista jest wtedy przycinana po
uzgodnieniu kafelków.

## Filtrowanie klas

<code-tabs name="classes" />

`classes` przyjmuje listę identyfikatorów klas i zachowuje tylko predykcje, których
klasa się na niej znajduje. Identyfikatory indeksują `result.names`, a najpewniejszą
metodą ich uzyskania jest odczytanie `names` z wyniku zamiast zakładania kolejności
zbioru danych.

Filtrowanie odbywa się centralnie, po przetwarzaniu końcowym każdej rodziny,
w jednym punkcie, przez który przechodzi każda ścieżka predykcji. Ma to dwie
ważne konsekwencje. Działa dla każdej rodziny, w tym tych bez NMS. Filtruje też
dane dopasowane do ramek, więc maski, punkty kluczowe i ramki zorientowane są
przycinane razem z nimi, a nie pozostawiane w niezgodnym stanie.

W wierszu poleceń `classes` przyjmuje pojedynczą liczbę całkowitą, listę albo
ciąg rozdzielany przecinkami:

```bash
libreyolo predict model=LibreYOLO9s.pt classes=0 source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
libreyolo predict model=LibreYOLO9s.pt classes="[0,2,5]" source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
```

Filtrowanie nie zapewnia bezpłatnie większej dokładności. Model nadal wykorzystuje
zasoby na przewidywanie klas, które później są odrzucane, a `max_det` jest stosowane
przez rodzinę przed filtrem. Obraz z dużą liczbą niepożądanych klas może więc
osiągnąć limit przed dotarciem do właściwej klasy. W takim przypadku zmniejsz
`conf` lub zwiększ `max_det`.

## agnostic_nms

`agnostic_nms` jest przyjmowane i nic nie robi. Przekazanie tego argumentu zgłasza
ostrzeżenie, że jest to opcja bez działania zachowana dla zgodności z wierszem
poleceń, po czym argument jest odrzucany.

Nie ma trybu tłumienia niezależnego od klasy. Każde wywołanie NMS w bibliotece
uwzględnia klasy, dlatego dwie nakładające się ramki różnych klas pozostają przy
dowolnym `iou`. Jeśli stanowi to problem, najpierw filtruj za pomocą `classes`
albo samodzielnie wykonaj tłumienie między klasami na `result.boxes`.

## Argumenty odrzucane przez predict

Dwa argumenty zgłaszają błąd zamiast ostrzeżenia: `visualize` i `embed` zgłaszają
`NotImplementedError`. Aby uzyskać osadzenia, załaduj model z `task="embed"`
i normalnie wywołaj `predict` lub `embed`.

Każdy nierozpoznany argument zgłasza `TypeError` z nazwami obsługiwanych opcji,
więc literówka powoduje natychmiastowy błąd, zamiast zostać po cichu zignorowana.

Następujące argumenty są przyjmowane, powodują ostrzeżenie i są odrzucane:
`agnostic_nms`, `boxes`, `dnn`, `half`, `line_width`, `retina_masks`, `show_conf`,
`show_labels` i `verbose`.

