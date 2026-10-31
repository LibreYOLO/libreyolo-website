---
title: Progi i filtrowanie
seo_title: 'conf, iou i max_det w LibreYOLO'
description: >-
  Co robią conf, iou, max_det, classes i agnostic_nms podczas predykcji oraz
  które rodziny ignorują iou, ponieważ nie wykonują własnego NMS.
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
last_verified: 1.6.0
verification: >-
  Wartości domyślne pochodzą z InferenceRunner.__call__ w
  libreyolo/models/base/inference.py. Zachowanie NMS dla poszczególnych rodzin
  odczytano ze wszystkich modułów w libreyolo/postprocess/ i porównano z
  _is_nms_free_family w libreyolo/backends/base.py. Filtrowanie klas pochodzi z
  InferenceRunner._apply_classes_filter i _wrap_results. agnostic_nms pochodzi z
  check_agnostic_nms i postprocess_max_det w libreyolo/utils/predict_args.py
  oraz z kroku agnostic w InferenceRunner._wrap_results. Obsługę otwartego
  słownika zaczerpnięto z NMS_THRESHOLD w libreyolo/models/openvocab/base.py.
  Wartości domyślne walidacji pochodzą z BaseModel.val.
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
  agnostic:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")


        per_class = model(SAMPLE_IMAGE)

        agnostic = model(SAMPLE_IMAGE, agnostic_nms=True)


        # Nakładające się ramki różnych klas zostają zredukowane do tej z
        najwyższym wynikiem.

        print(len(per_class.boxes), len(agnostic.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt agnostic_nms=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: fa0e6cee02b3e336
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

Walidacja detekcji udostępnia `metrics/best_conf` i progi poszczególnych klas na podstawie F1 przy IoU 0.50. Można ich użyć jako zmierzonych punktów wyjścia dla progów wdrożenia; zobacz [walidację](/docs/train/validation). Segmentacja nie udostępnia tych kluczy.

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

Trzy opcje predykcji sprawiają, że `iou` ma znaczenie nawet dla predyktora zbioru,
ponieważ wszystkie trzy działają na ramkach po zakończeniu działania modelu:

- `tiling=True` uzgadnia nakładające się kafelki za pomocą NMS na klasę przy `iou`
- `augment=True` scala odbite widoki za pomocą NMS na klasę przy `iou`

- `agnostic_nms=True` tłumi nakładające się ramki między klasami przy `iou`,
  opisane [poniżej](#agnostic_nms)

Pierwsze dwie opisano w sekcji [wydajność wnioskowania](/docs/predict/performance).

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
konfiguracja referencyjna. SSD ogranicza ją do 200, a FCOS do własnego limitu
detekcji na obraz. Zwiększanie `max_det` ponad te wartości nie ma wpływu.

`max_det` jest stosowane centralnie zamiast osobno dla rodziny w dwóch
przypadkach: we wnioskowaniu kafelkowym, gdzie scalona lista jest przycinana po
uzgodnieniu kafelków, oraz w wywołaniach z `classes` lub `agnostic_nms=True`,
gdzie cięcie następuje po filtrze albo po tłumieniu.

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
zasoby na przewidywanie klas, które później są odrzucane. Przy ustawionym
`classes` rodzina zachowuje budżet kandydatów równy 300 ramkom lub `max_det`,
gdy jest wyższe, a cięcie do `max_det` następuje po filtrze, więc małe `max_det`
nie wypycha właściwej klasy. Wyeksportowane backendy używają tego samego
budżetu, łącznie z wsadowymi silnikami TensorRT. Obraz z więcej niż 300
wyżej ocenionymi ramkami niepożądanych klas nadal może wypełnić budżet przed
dotarciem do właściwej klasy; jeśli tak się stanie, zwiększ `max_det` powyżej 300.

## agnostic_nms

NMS każdej rodziny uwzględnia klasy, więc domyślnie dwie nakładające się ramki
różnych klas pozostają obie, przy dowolnym `iou`. `agnostic_nms=True` dodaje
jeden krok niezależny od klas: spośród ramek nakładających się powyżej `iou`
zostaje tylko ta z najwyższym wynikiem, bez względu na ich klasy.

<code-tabs name="agnostic" />

Krok działa na gotowych detekcjach rodziny, po filtrze `classes` i przed cięciem
do `max_det`. Dlatego zachowuje się tak samo w predyktorach zbioru, które nie
wykonują własnego NMS, w wyeksportowanych backendach oraz z `augment=True` i
`tiling=True`. Maski i punkty kluczowe są odrzucane razem ze swoją ramką, a
obrócone ramki są porównywane za pomocą obróconego IoU. Obejmuje zadania
zwracające ramki: detekcję, segmentację, pozę i obb.

Rodzina zachowuje budżet kandydatów równy 300 ramkom lub `max_det`, gdy jest
wyższe, na którym działa ten krok. Przy `max_det` poniżej 300 kandydaci o niższym
wyniku uzupełniają miejsca zwolnione przez tłumienie. Przy 300 lub więcej, co
obejmuje wartość domyślną, te miejsca pozostają puste.

Gdzie to nie działa:

- Na standardowej ścieżce predykcji zadanie bez ramek zgłasza `ValueError`.
- `LibreEnsemble` i [HRNet](/docs/models/hrnet) zgłaszają `NotImplementedError`.
- Modele `LibreSAM` i [detektory 3D](/docs/tasks/3d-object-detection), czyli
  WildDet3D, 3D-MOOD, FCOS3D i DetAny3D, nie przyjmują takiego argumentu i
  zgłaszają `TypeError`.
- [L2CS-Net](/docs/models/l2cs), [PP-OCRv5](/docs/models/pp-ocrv5),
  [SAM 3D Body](/docs/models/sam-3d-body) i
  [LibreFaceRec](/docs/models/librefacerec) mają własną ścieżkę predykcji i
  ignorują go.

`val(agnostic_nms=True)` stosuje ten sam krok w walidacji detekcji i
segmentacji; zobacz [walidacja i metryki](/docs/train/validation).

## Argumenty odrzucane przez predict

Dwa argumenty zgłaszają błąd zamiast ostrzeżenia: `visualize` i `embed` zgłaszają
`NotImplementedError`. Aby uzyskać osadzenia, załaduj model z `task="embed"`
i normalnie wywołaj `predict` lub `embed`. `agnostic_nms=True` zgłasza błąd tam,
gdzie nie działa, jak wymieniono powyżej.

Każdy nierozpoznany argument zgłasza `TypeError` z nazwami obsługiwanych opcji,
więc literówka powoduje natychmiastowy błąd, zamiast zostać po cichu zignorowana.

Następujące argumenty są przyjmowane, powodują ostrzeżenie i są odrzucane:
`boxes`, `dnn`, `half`, `line_width`, `retina_masks`, `show_conf`,
`show_labels` i `verbose`.
