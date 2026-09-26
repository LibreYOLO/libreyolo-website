---
title: Estymacja pozy
seo_title: Estymacja pozy w LibreYOLO
description: >-
  Przewiduj punkty kluczowe dla każdej instancji w LibreYOLO: poznaj rodziny
  obsługujące to zadanie, format etykiet oraz wywołania predykcji, trenowania,
  walidacji i eksportu.
lead: >-
  Estymacja pozy lokalizuje każdą instancję i zwraca dla niej uporządkowany
  zestaw nazwanych punktów kluczowych. Dane wyjściowe opisują więc wewnętrzną
  strukturę obiektu, a nie tylko jego zasięg. Klucz zadania to pose.
keywords:
  - estymacja pozy python
  - detekcja punktów kluczowych
  - model pozy człowieka
  - COCO keypoints
  - OKS mAP
  - trenowanie modelu pose
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Sufiks -pose w nazwie pliku wybiera głowicę punktów kluczowych, więc
        # argument task nie jest potrzebny.
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy.shape)   # współrzędne pikselowe (N, K, 2)
        print(result.boxes.xyxy.shape)     # (N, 4), te same N instancji
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreECs-pose.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Tylko widoczne punkty kluczowe
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        result = LibreYOLO("LibreECs-pose.pt")(SAMPLE_IMAGE)

        kpts = result.keypoints


        # .has_visible jest wyprowadzane z trzeciej kolumny punktów kluczowych i
        ma

        # same wartości prawdziwe, gdy checkpoint przewiduje tylko (x, y).

        for person, visible in zip(kpts.xy, kpts.has_visible):
            print(person[visible])
    - label: Zamiast tego top-down
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # HRNet działa top-down: najpierw wycina każdą osobę. Gdy nie podano
        źródła

        # osób, łączy się z detektorem LibreYOLO9t i zapisuje ten wybór w logu.

        model = LibreYOLO("LibreHRNetw32-pose.pt")

        result = model(SAMPLE_IMAGE)


        print(result.keypoints.xy.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # coco8-pose.yaml zawiera osadzony skrypt pobierania, dlatego wymaga
        # jawnej zgody, chyba że dane są już dostępne lokalnie.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="coco8-pose.yaml",
            epochs=50,
            imgsz=640,
            batch=4,
            allow_download_scripts=True,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreECs-pose.pt data=coco8-pose.yaml \
          epochs=50 imgsz=640 batch=4 allow_download_scripts=True
    - label: Własny zbiór danych
      language: python
      code: |
        from libreyolo import LibreYOLO

        # data.yaml musi deklarować kpt_shape, a wiersze etykiet muszą zawierać
        # dokładnie 5 + K * D pól.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(data="my-pose-dataset.yaml", epochs=50, imgsz=640, batch=8)
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreECs-pose.pt")


        # val() zwraca zwykły słownik, a nie obiekt.

        metrics = model.val(data="coco8-pose.yaml", allow_download_scripts=True)


        print(metrics["metrics/keypoints_mAP50-95"])

        print(metrics["metrics/keypoints_mAP50"],
        metrics["metrics/keypoints_mAP75"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs-pose.pt data=coco8-pose.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
    - label: Używanie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie sufiksu pliku, więc
        wyeksportowany artefakt

        # wczytuje się jak checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreECs-pose.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.keypoints.xy)
source_hash: 9de01d1f615bdf33
---

## Definicja

Estymacja pozy zwraca strukturę, a nie tylko zasięg. Każda instancja nadal
otrzymuje ramkę, klasę i wynik, a ponadto `K` punktów kluczowych w stałej
kolejności. Dzięki temu indeks 5 oznacza tę samą część ciała dla każdej
instancji i na każdym obrazie. Kolejność definiuje zestaw etykiet. Żaden element
danych wyjściowych nie identyfikuje punktu kluczowego nazwą.

`pose` jest kanonicznym kluczem zadania, a sufiks `-pose` w nazwie pliku
checkpointu wybiera to zadanie, więc `task=` nie jest potrzebne przy wczytywaniu
opublikowanych wag.

`predict()` wypełnia `result.keypoints` obok `result.boxes`. Pole `.data` ma
postać `(N, K, 2)` lub `(N, K, 3)` i jest wyrównane wierszami z ramkami, więc
instancja `i` w jednym polu jest instancją `i` w drugim. `.xy` wybiera
współrzędne pikselowe, a `.xyn` normalizuje je według rozmiaru oryginalnego
obrazu. Pole `.conf` jest trzecią kolumną, gdy checkpoint ją przewiduje, albo
`None`, gdy tego nie robi. `.has_visible` jest wyprowadzoną z niego maską
logiczną, zawierającą same wartości prawdziwe, gdy nie ma trzeciej kolumny.

Do tych danych wyjściowych prowadzą dwie architektury. Model jednoetapowy
przewiduje ramki i punkty kluczowe w jednym przebiegu. Model top-down najpierw
uruchamia detektor, następnie wycina każdą instancję i wykonuje regresję punktów
kluczowych wewnątrz wycinka, więc jego dokładność zależy od poprzedzającego go
detektora.

## Modele

Trzy rodziny zarówno trenują, jak i przewidują:
[RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter) oraz
[YOLO-NAS](/docs/models/yolo-nas). Wszystkie są jednoetapowe. RF-DETR wymaga
własnego dodatku `pip install "libreyolo[rfdetr]"`. RF-DETR i EdgeCrafter
udostępniają opublikowane checkpointy pozy i oba można dostrajać na zbiorach
danych z jedną klasą zawierającą wyłącznie osoby. Głowica punktów kluczowych
EdgeCrafter jest ustalana podczas tworzenia i odrzuca zbiór danych deklarujący
inną liczbę, natomiast RF-DETR ponownie inicjalizuje dla niego głowicę. YOLO-NAS
pobiera wagi z własnej sieci CDN Deci.AI na licencji niekomercyjnej, a LibreYOLO
nie publikuje żadnych z nich. Jego głowica pozy również jest przebudowywana dla
nowej liczby punktów kluczowych. Jako jedyna z tych trzech rodzin nie ma stałej
liczby klas równej jeden, dlatego nadaje się do szkieletu wieloklasowego lub
innego niż ludzki, na przykład do pozy zwierząt.

[HRNet](/docs/models/hrnet) jest opcją top-down. Przewiduje, waliduje i
eksportuje, a jego `train()` zgłasza `NotImplementedError`. Jeśli nie podano
źródła osób, automatycznie łączy się z detektorem LibreYOLO9t. `cropped=True`
traktuje cały obraz jako jedną instancję, `person_boxes=` przyjmuje już posiadane
ramki, a `person_detector=` wskazuje inny detektor.

[SenseNova-Vision](/docs/models/sensenova-vision) również generuje punkty
kluczowe. Jest to generatywny model sterowany promptem, z własną fabryką
`LibreVLM` i własnym dodatkiem. Gdy nie ustawiono słownika, `set_task("pose")`
wraca do kategorii osoby. Jego wagi są przeznaczone do użytku niekomercyjnego,
a opóźnienie dla pojedynczego obrazu jest znacznie większe niż w
wyspecjalizowanej głowicy pozy, ponieważ każda predykcja jest dekodowaniem
dyfuzyjnym.

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane lokalnie w
pamięci podręcznej.

<code-tabs name="predict" />

Liczba i kolejność punktów kluczowych są właściwościami checkpointu, a nie
biblioteki, dlatego model wytrenowany na innym szkielecie zwraca inną wartość
`K` i inne znaczenie każdego indeksu. Zawartość trzeciej kolumny punktów
kluczowych również jest właściwością checkpointu. EdgeCrafter zapisuje tam stałą
wartość zamiast wyniku dla poszczególnych punktów i w ogóle nie ma głowicy
ramek, dlatego każda z jego ramek pozy jest zewnętrznym obrysem punktów
kluczowych danej instancji. Informacje o źródłach, streamingu i obsłudze wyników
znajdują się w sekcji [predykcja](/docs/predict).

## Format zbioru danych

Układ odpowiada detekcji: na każdy obraz przypada jeden plik etykiet `.txt`,
znajdowany przez zastąpienie `images` ciągiem `labels` w ścieżce obrazu i zmianę
rozszerzenia.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

Wiersz jest wierszem detekcji z dołączonymi punktami kluczowymi:

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

Liczba pól wynosi dokładnie `5 + K * D`, gdzie `D` jest drugą wartością
`kpt_shape`. Współrzędne ramki i punktów kluczowych są znormalizowanymi liczbami
zmiennoprzecinkowymi względem szerokości i wysokości oryginalnego obrazu.
Widoczność `v`, obecna tylko wtedy, gdy `D` wynosi 3, ma wartość `0`, `1` lub `2`.

Plik YAML dodaje dwa klucze do wspólnego kontraktu:

```yaml
path: dataset
train: images/train
val: images/val
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
names:
  0: person
```

`kpt_shape` jest wymagane i ma postać `[K, 2]` lub `[K, 3]`. Opcjonalne
`flip_idx` jest permutacją `0..K-1`, która dla każdego punktu kluczowego podaje
indeks przyjmowany po odbiciu poziomym. Dzięki temu lewy nadgarstek pozostaje
lewym nadgarstkiem. Pominięcie tego pola wyłącza augmentację odbicia poziomego
dla punktów kluczowych zamiast zastosować ją z błędną kolejnością indeksów.

## Trenowanie

<code-tabs name="train" />

Trenowanie jest kontynuowane z opublikowanego checkpointu `-pose`, który ma już
głowicę punktów kluczowych. Zadanie jest odczytywane z wczytanego checkpointu,
a nie z flagi przekazanej podczas trenowania, dlatego żądanie zadania pozy nie
zmienia checkpointu detekcji w takie trenowanie. W przypadku EdgeCrafter
`kpt_shape` w pliku YAML musi dokładnie odpowiadać głowicy, ponieważ jest ona
ustalana podczas tworzenia. RF-DETR i YOLO-NAS zmieniają natomiast rozmiar
głowicy dla innej liczby punktów. Informacje o zbiorach danych, augmentacji,
wielu GPU i loggerach znajdują się w sekcji [trenowanie](/docs/train).

## Walidacja

`val()` zwraca zwykły słownik kluczy `metrics/`. Ocena jest ewaluacją punktów
kluczowych COCO względem Object Keypoint Similarity, które waży błąd odległości
każdego punktu kluczowego skalą instancji i tolerancją dla danego punktu. Pełni
więc rolę IoU dla ramek. Wymaga `pycocotools`, które znajduje się w instalacji
podstawowej.

<code-tabs name="val" />

`metrics/keypoints_mAP50-95` jest głównym wynikiem, czyli średnią precyzją
uśrednioną dla progów OKS od 0.50 do 0.95. Trenowanie używa go do wyboru
najlepszej epoki. `metrics/keypoints_mAP50` i `metrics/keypoints_mAP75` są
wariantami z jednym progiem, a `metrics/keypoints_mAP_M` i
`metrics/keypoints_mAP_L` dzielą średnią według obszaru instancji na średnie i
duże. Ewaluacja punktów kluczowych COCO nie definiuje małej kategorii. Odpowiednie
wartości średniego recall to `metrics/keypoints_AR50-95`,
`metrics/keypoints_AR50`, `metrics/keypoints_AR75`, `metrics/keypoints_AR_M` i
`metrics/keypoints_AR_L`. Każdy klucz w tym zadaniu ma prefiks `keypoints_`,
dlatego nie pojawiają się klucze `mAP` ramek zwracane przez detektor.

## Eksport

<code-tabs name="export" />

Wyeksportowany artefakt jest ponownie wczytywany przez `LibreYOLO()` na podstawie
sufiksu pliku, więc plik `.onnx` lub `.engine` zachowuje się jak checkpoint i
zwraca ten sam obiekt `Results`. Zakres formatów różni się zależnie od rodziny.
Macierz na stronie każdego modelu jest generowana ze zweryfikowanego zestawu, a
nie wpisywana ręcznie. Formaty, ich dodatki i ograniczenia opisano w sekcji
[eksport i wdrożenie](/docs/export).

