---
title: Łączenie detektorów w zespół
seo_title: Łączenie detektorów w zespół w LibreYOLO
description: >-
  Uruchamiaj kilka detektorów na jednym obrazie i scalaj ich ramki za pomocą
  ważonego scalania ramek albo NMS, także dla modeli z różnymi listami klas.
lead: >-
  LibreEnsemble uruchamia co najmniej dwa detektory na tym samym zdekodowanym
  obrazie i scala ich ramki w jeden obiekt Results. Elementy zachowują własne
  wagi, progi, urządzenia i listy klas.
keywords:
  - zespół modeli detekcji obiektów
  - weighted boxes fusion
  - WBF Python
  - łączenie dwóch detektorów
  - scalanie ramek ograniczających
  - LibreEnsemble
  - ensemble detekcji Python
  - min_votes
last_verified: 1.5.0
verification: >-
  Sygnatury konstruktora i wywołania, wartości domyślne, błędy walidacji,
  ujednolicanie przestrzeni klas, liczenie głosów i zwracany obiekt Results
  odczytano z libreyolo/ensemble/model.py. Algorytmy scalania i ich argumenty
  pochodzą z libreyolo/ops/fusion.py. Założenia projektowe pochodzą z
  docs/adr/0004-model-ensembling.md. Wzorce użycia porównano z
  tests/unit/test_ensemble.py i tests/unit/test_ops_fusion.py.
snippets:
  basic:
    - label: Dwa scalone detektory
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE
        
        # Elementami mogą być ścieżki checkpointów lub już wczytane modele.
        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])
        
        result = ensemble(SAMPLE_IMAGE)
        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: Wagi i wymaganie liczby głosów
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ensemble = LibreEnsemble(
            ["LibreYOLO9s.pt", "LibreRFDETRs.pt"],
            weights=[1.0, 1.3],   # umownie proporcjonalnie do walidacyjnego mAP
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,          # zachowaj tylko ramki znalezione przez oba elementy
        )

        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes), "agreed detections")
    - label: Progi poszczególnych elementów
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE
        
        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])
        
        # Skalar dotyczy każdego elementu; lista jest odczytywana osobno dla każdego z nich.
        result = ensemble(SAMPLE_IMAGE, conf=[0.3, 0.5], iou=0.5)
        print(len(result.boxes))
  external:
    - label: Dodanie detektora niezaładowanego przez LibreYOLO
      language: python
      code: |
        from libreyolo import ExternalDetector, LibreEnsemble, SAMPLE_IMAGE

        def my_detector(pil_image):
            # Zwróć (boxes, scores, labels): xyxy w pikselach oryginalnego obrazu.
            return ([[100.0, 100.0, 200.0, 300.0]], [0.9], [0])

        external = ExternalDetector(my_detector, names={0: "person"})

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", external])
        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes))
  sources:
    - label: Te same źródła co dla pojedynczego modelu
      language: python
      code: |
        from libreyolo import LibreEnsemble

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        # Zastąp clip.mp4 plikiem wideo na dysku.
        for result in ensemble("clip.mp4", stream=True, vid_stride=2):
            print(result.frame_idx, len(result.boxes))
source_hash: 4f4c54c52b295795
---

## Czym jest zespół

`LibreEnsemble` przyjmuje co najmniej dwa detektory, uruchamia każdy na tym samym
obrazie i scala ich ramki w jeden obiekt `Results`. Jest to konstrukcja używana
podczas predykcji. Nie ma niczego do trenowania, a elementy pozostają niezależnymi
modelami, które można osobno walidować i eksportować.

Detekcja jest jedynym obsługiwanym zadaniem. Element wykonujący inne zadanie
powoduje w konstruktorze `ValueError` z indeksem elementu i nazwą zadania.

Obie nazwy są importowane leniwie, więc nic nie kosztują do chwili użycia:

```python
from libreyolo import LibreEnsemble, ExternalDetector
```

## Budowanie zespołu

<code-tabs name="basic" />

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

`members` to sekwencja co najmniej dwóch elementów. Wpis typu `str` albo `Path`
jest ładowany przez `LibreYOLO()`. Każdy inny musi być wywoływalny i udostępniać
słownik `names`. Mniej niż dwa elementy powodują `ValueError`, a przekazanie samego
ciągu powoduje `TypeError` zamiast iterowania po jego znakach.

`weights` ma domyślnie wartość `None`, co oznacza równomierne ważenie. Podane wagi
muszą występować po jednej na element i być ściśle dodatnie. Waga zerowa powoduje
błąd zamiast cichego usunięcia elementu. Udokumentowana konwencja zakłada wartości
proporcjonalne do walidacyjnego mAP każdego elementu.

`fusion_iou` ma domyślnie wartość `0.55` i jest IoU, przy którym ramki różnych
elementów są grupowane. Jest to inny próg niż `iou` przekazywane przy wywołaniu,
które ustawia NMS osobno dla każdego elementu.

`min_votes` ma domyślnie wartość `1`, co oznacza, że pojedynczy element może
zachować ramkę. Zwiększenie wartości pozostawia tylko klastry potwierdzone przez
tyle różnych elementów. Musi to być dodatnia liczba całkowita nieprzekraczająca
liczby elementów. Dla każdej klasy jest ograniczana do liczby elementów, które
rzeczywiście ją znają, dzięki czemu klasa wytrenowana tylko w jednym elemencie
nie zostaje po cichu usunięta.

## Metody scalania

Akceptowane są trzy nazwy oraz obiekt wywoływalny.

| `fusion` | Zachowanie |
|---|---|
| `"wbf"` | Ważone scalanie ramek, sekwencyjne i zgodne z publikacją. Wartość domyślna |
| `"wbf_seeded"` | Jednoprzebiegowe ważone scalanie ramek; NMS uwzględniające klasy wybiera zalążki klastrów |
| `"nms"` | Połączenie ramek wszystkich elementów, a następnie NMS uwzględniające klasy |

Ważone scalanie ramek uśrednia współrzędne klastra z wagami wynikającymi z pewności,
tworząc ramkę, której nie zaproponował żaden pojedynczy element. Oba warianty
ważone są zgodne, gdy klastry są jednoznaczne, i mogą nieznacznie różnić się dla
łańcuchów nakładających się klastrów. `"nms"` wybiera jedną ramkę zamiast uśredniać,
więc ramki, które przetrwały, zachowują oryginalne wyniki, a wagi wpływają tylko
na to, która ramka wygra. Ponieważ ta metoda wybiera zamiast grupować, nie może
liczyć głosów. Połączenie `fusion="nms"` z `min_votes` większym niż `1` powoduje
`ValueError`.

Ważone scalanie ramek skaluje wynik klastra według udziału wagi elementów, które
go poparły. Przy dwóch elementach o jednakowych wagach ramka znaleziona tylko
przez jeden zachowuje połowę wyniku: `0.9` staje się `0.45`. Scalona pewność może
więc spaść poniżej `conf`, z którym uruchomiono każdy element. Filtruj według
wyniku po scaleniu, zamiast zakładać, że próg elementu nadal obowiązuje.

## Elementy z różnymi listami klas

Elementy nie muszą mieć wspólnej listy klas. Ich przestrzenie etykiet są łączone
według nazw, a każdy element otrzymuje tabelę odwzorowującą własne identyfikatory
klas do sumy. `ensemble.names` jest tą sumą i trafia do zwracanego obiektu `Results`.

Ramki są scalane wyłącznie w obrębie tej samej nazwy klasy. Klasa znana tylko
jednemu elementowi przechodzi bez scalania i nie jest za to karana. Skalowanie
wyniku używa mianownika osobnego dla klasy, więc klasa znana pojedynczo zachowuje
swój wynik.

Częściowe pokrywanie się list powoduje ostrzeżenie wymieniające klasy, które nie
są wspólne dla wszystkich elementów. Należy przeczytać je uważnie, ponieważ punkt
kontrolny z nazwami zastępczymi, takimi jak `class_0`, tworzy sumę rozłączną z
każdym innym elementem i w ogóle nie dochodzi do scalania między elementami.

Element zwracający identyfikator klasy spoza własnego `names` powoduje `RuntimeError`.

## Detektory zewnętrzne

<code-tabs name="external" />

`ExternalDetector(fn, names)` opakowuje dowolny obiekt wywoływalny, który przyjmuje
obraz PIL i zwraca `(boxes, scores, labels)`, z ramkami xyxy w pikselach oryginalnego
obrazu. Sprawdza liczbę elementów wyniku, kształt ramek, zgodność długości oraz to,
czy każdy identyfikator klasy występuje w `names`, a także sam stosuje próg `conf`.

W ten sposób detektor niezaładowany przez LibreYOLO uczestniczy w scalaniu.

## Wywoływanie

<code-tabs name="sources" />

Sygnatura wywołania odpowiada pojedynczemu modelowi i przyjmuje te same źródła:
obrazy, foldery, listy, wideo, przechwytywanie ekranu, kamery internetowe i
strumienie sieciowe. Źródła na żywo wymagają `stream=True` z tego samego powodu
co w innych miejscach.

| Argument | Wartość domyślna | Uwagi |
|---|---|---|
| `conf` | `0.25` | Osobno dla elementu; skalar jest rozgłaszany albo można podać po jednej wartości na element |
| `iou` | `0.45` | Własny próg NMS każdego elementu, nie próg scalania |
| `imgsz` | `None` | `list` jest odczytywana osobno dla elementów; `int` lub krotka są rozgłaszane |
| `device` | `None` | Skalar albo po jednej wartości na element, dzięki czemu elementy mogą działać na różnych urządzeniach |
| `classes` | `None` | Filtruje scalony wynik według identyfikatorów sumy klas |
| `max_det` | `300` | Dotyczy scalonego wyniku |

Ponieważ `list` dla `imgsz` oznacza wartości osobne dla elementów, `imgsz=[480, 640]`
ustawia 480 dla pierwszego i 640 dla drugiego elementu, natomiast `imgsz=(480, 640)`
jest jednym prostokątnym rozmiarem dla wszystkich. Łatwo pomylić te przypadki.

Elementy są wywoływane z `max_det` wynoszącym co najmniej 300 niezależnie od
żądanej wartości, dzięki czemu każdy działa szeroko, a zespół przycina wynik raz na końcu.

Obraz jest dekodowany raz, po czym ten sam obiekt jest przekazywany każdemu
elementowi. `batch` jest przyjmowane dla zgodności i ignorowane. Obrazy są
przetwarzane sekwencyjnie.

## Zwracany wynik

Zwykły obiekt `Results`, tego samego typu co wynik pojedynczego modelu, z `names`
ustawionym na sumę przestrzeni klas. Wszystkie informacje z sekcji
[praca z wynikami](/docs/predict/results) pozostają aktualne.

Jedyną różnicą jest `result.speed`, które zespół faktycznie wypełnia. Klucze to
`member_0`, `member_1` i kolejne oraz `fusion`, a wartości podano w milisekundach.
Jest to jedyne miejsce w bibliotece, w którym `speed` jest wypełniane.

Wiersze zawierające nieskończone albo nieokreślone ramki lub wyniki są odrzucane
przed scalaniem. Gdy elementy działają na różnych urządzeniach, scalanie odbywa
się na urządzeniu pierwszego elementu, który cokolwiek zwrócił.

## Ograniczenia zespołu

Zarówno `val()`, jak i `export()` zgłaszają `NotImplementedError` i wskazują
elementy. Każdy z nich należy walidować i eksportować osobno. Metoda `train` nie
istnieje, więc jej wywołanie powoduje `AttributeError`.

Połowiczna precyzja nie jest obsługiwana na poziomie zespołu. `half=True` trafia
do tej samej ścieżki ostrzeżenia i braku działania co wszędzie indziej. Precyzję
należy skonfigurować osobno dla każdego elementu.

Łączenie w zespół nie ma interfejsu wiersza poleceń. Jest to API Python.
