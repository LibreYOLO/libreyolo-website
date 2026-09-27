---
title: Typy Results
seo_title: Dokumentacja obiektu Results w LibreYOLO
description: >-
  Wszystkie dane, które może zawierać obiekt Results LibreYOLO, po jednym polu
  na kształt zadania: ramki, maski, punkty kluczowe, probs, obb, głębia, ocr,
  embeddingi i dziesięć kolejnych.
lead: >-
  Results jest pojedynczym typem zwracanym dla każdego obrazu przez każdy model
  LibreYOLO. Zawiera osiemnaście opcjonalnych pól danych, po jednym na kształt
  zadania, i wypełnia tylko te, które utworzył model.
keywords:
  - obiekt Results LibreYOLO
  - Results.boxes
  - Results.masks
  - Results.probs
  - Results.depth_map
  - Results.summary
  - LibreYOLO Results to_json
last_verified: 1.5.0
verification: >-
  Nazwy pól, kształty, właściwości i wartości domyślne odczytano z
  libreyolo/utils/results.py w wersji 1.5.0. Semantyka pochodzi z docstringów
  klas danych.
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        print(result.orig_shape, result.path)
        print(result.boxes.xyxy)
        print(result.boxes.conf)
        print(result.names[int(result.boxes.cls[0])])
  convert:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        # Wszystkie dane są przenoszone razem.
        result = result.cpu().numpy()

        # Wiersze jako zwykłe słowniki, a następnie jako JSON.
        print(result.summary()[:1])
        print(result.to_json())
source_hash: 16f654364ae6448a
---

## Obiekt Results

Jeden `Results` opisuje jeden obraz. Pojedyncze źródło obrazu zwraca jeden taki
obiekt, źródło będące listą lub katalogiem zwraca listę, a `stream=True` zwraca
generator, który je emituje.

| Atrybut | Typ | Znaczenie |
|---|---|---|
| `orig_shape` | `(int, int)` | Pierwotna wysokość i szerokość obrazu |
| `path` | `str` | Ścieżka źródłowa, gdy wejście pochodziło z dysku |
| `names` | `dict[int, str]` | Mapowanie indeksu klasy na jej nazwę |
| `speed` | `dict[str, float]` | Milisekundy dla poszczególnych etapów |
| `track_id` | tensor | Identyfikatory śledzenia, gdy wynik pochodzi z `track()` |
| `frame_idx` | `int` | Indeks klatki dla źródeł wideo i strumieni |
| `restore_scale` | `int` | Współczynnik powiększenia wyjścia względem wejścia dla wyniku przywracania; `1` we wszystkich pozostałych przypadkach |

<code-tabs name="usage" />

## Pola danych

Każde pole ma wartość `None`, jeśli model go nie utworzył. Rodzina wypełnia
pole zależne od swojego zadania.

| Pole | Klasa | Zadanie |
|---|---|---|
| `boxes` | `Boxes` | detect |
| `masks` | `Masks` | segment |
| `keypoints` | `Keypoints` | pose |
| `probs` | `Probs` | classify |
| `obb` | `OBB` | obb |
| `gaze` | `Gaze` | gaze |
| `points` | `Points` | point |
| `semantic_mask` | `SemanticMask` | semantic |
| `panoptic` | `PanopticSegmentation` | panoptic |
| `depth_map` | `DepthMap` | depth |
| `normal_map` | `NormalMap` | normal |
| `edges` | `EdgeMap` | edge |
| `restored` | `RestoredImage` | restore |
| `matte` | `Matte` | matte |
| `ocr` | `OCRRegions` | ocr |
| `embeddings` | `Embeddings` | embed |
| `identities` | `Identities` | embed, z galerią |
| `meshes` | `Meshes` | mesh |

`result.normals` jest aliasem do odczytu i zapisu dla `result.normal_map`.

Jednocześnie może być ustawione więcej niż jedno pole. Model segmentacji
wypełnia zarówno `boxes`, jak i `masks`; model gaze wypełnia `boxes` ramkami
twarzy i `gaze` kątami; model siatki wypełnia `boxes` ramkami osób oraz
`meshes` wyrównanymi do nich według wierszy.

## Boxes

Ramki detekcji dla jednego obrazu.

| Element | Zwraca |
|---|---|
| `xyxy` | Współrzędne narożników w pikselach pierwotnego obrazu |
| `xywh` | Środek i rozmiar w pikselach |
| `xyxyn` | Narożniki znormalizowane do `[0, 1]` |
| `xywhn` | Środek i rozmiar znormalizowane do `[0, 1]` |
| `conf` | Pewność każdej ramki |
| `cls` | Indeks klasy każdej ramki |
| `id` | Identyfikator śledzenia każdej ramki albo `None` |
| `is_track` | `True`, gdy występują identyfikatory śledzenia |
| `data` | Spakowany tensor |

`with_id(id)` i `with_orig_shape(orig_shape)` zwracają nowy obiekt `Boxes`
z zastąpionym odpowiednim polem.

## Masks

Maski instancji dla jednego obrazu. `data` jest tensorem maski, `xy` zwraca
kontury poszczególnych instancji w pikselach, a `xyn` zwraca je znormalizowane.

## Keypoints

Punkty kluczowe pozy wyrównane wierszami z `boxes`. `xy` jest parą współrzędnych
każdego punktu kluczowego, a `xyn` parą znormalizowaną. `conf` jest trzecim
kanałem, gdy dane go zawierają, w przeciwnym razie ma wartość `None`.
`has_visible` jest tablicą logiczną, której wartość jest true tam, gdzie
`conf > 0`, a gdy nie ma kanału pewności, wszystkie wartości są true.

## Points

Lokalizacja punktów dla jednego obrazu. `data` ma kształt `(N, 4)` z wierszami
`x, y, class, confidence`. Współrzędne są bezwzględnymi pikselami; `xy`, `cls`
i `conf` rozdzielają kolumny, a `xyn` normalizuje współrzędne.

## Probs

Wyniki klasyfikacji. `top1` jest zwycięskim indeksem, `top5` pięcioma
najlepszymi indeksami, a `top1conf` i `top5conf` ich wynikami.

## OBB

Obrócone ramki. `data` zawiera 7 lub 8 wartości w każdym wierszu: `xywhr`,
opcjonalny identyfikator śledzenia, a następnie pewność i klasę.

| Element | Zwraca |
|---|---|
| `xywhr` | Środek, rozmiar i obrót w radianach |
| `xyxyxyxy` | Cztery narożniki w pikselach |
| `xyxyxyxyn` | Cztery znormalizowane narożniki |
| `xyxy` | Obwiednię wyrównaną do osi w pikselach |
| `conf`, `cls`, `id`, `is_track` | Tak samo jak w `Boxes` |

## Gaze

Kąty spojrzenia dla poszczególnych twarzy w radianach, o kształcie `(N, 2)`,
wyrównane wierszami z ramkami twarzy w `boxes`. Kolumna 0 to nachylenie pitch,
a kolumna 1 to odchylenie yaw, zgodnie z konwencją L2CS: dodatnie yaw obraca
spojrzenie w lewo osoby, a dodatnie pitch obraca je w dół. `pitch_deg`
i `yaw_deg` konwertują wartości na stopnie, a `direction_3d` zwraca jednostkowy
wektor kierunku.

## SemanticMask

Gęsta mapa semantyczna o kształcie `(H, W)` z całkowitymi identyfikatorami klas
na pierwotnym obszarze obrazu. `255` jest wartością ignorowaną i nigdy nie jest
liczone jako klasa (`SemanticMask.IGNORE_INDEX`). `classes` wymienia obecne
identyfikatory klas, a `class_mask(class_id)` zwraca maskę logiczną jednej klasy.

## PanopticSegmentation

Każdy piksel otrzymuje dokładnie jeden nienakładający się segment, co łączy
regiony stuff i instancje thing. `data` jest całkowitoliczbową mapą
identyfikatorów segmentów `(H, W)`. Identyfikator segmentu `0` oznacza brak
etykiety (`PanopticSegmentation.IGNORE_INDEX`). `segments_info` jest listą
słowników, po jednym na segment, a każdy zawiera co najmniej `{"id": int,
"category_id": int}`, gdzie `id` odpowiada wartości w mapie, a `category_id`
indeksuje `names`. `segment_ids` wymienia obecne identyfikatory,
a `segment_mask(segment_id)` zwraca maskę logiczną jednego segmentu.

Podział thing i stuff jest właściwością kategorii, a nie segmentu. Dane mogą
zdenormalizować tę informację na każdy segment jako `"isthing": bool`. Jeśli
to robią, wartość musi być zgodna z mapą na poziomie kategorii.

## DepthMap

Gęsta mapa względnej odwrotności głębi, o kształcie zmiennoprzecinkowym
`(H, W)` na pierwotnym obszarze obrazu. Wyższe wartości oznaczają położenie
bliżej kamery. Wartości są względne, a nie metryczne w metrach. `min`, `max`
i `mean` są obliczane dla wartości skończonych, a `normalized()` skaluje mapę
do `[0, 1]`.

## NormalMap

Gęste pole normalnych powierzchni, float32 `(H, W, 3)` na pierwotnym obszarze
obrazu, w układzie kamery OpenCV: `+x` w prawo, `+y` w dół, `+z` w głąb sceny.
Normalne są skierowane do kamery, więc powierzchnia równoległa do płaszczyzny
obrazu ma wartość `(0, 0, -1)`. Każdy piksel jest wektorem jednostkowym.
`assert_normalized(atol=1e-4)` sprawdza ten niezmiennik.

## EdgeMap

Gęsta mapa prawdopodobieństwa krawędzi, float32 `(H, W)` na pierwotnym obszarze
obrazu, gdzie `0` oznacza brak krawędzi, a `1` oznacza krawędź. Zachowywana jest
mapa ciągła, aby wybór progu należał do kodu wywołującego:
`binary(threshold=0.5)` stosuje próg, a `array` zwraca widok numpy.

## RestoredImage

Przywrócony obraz RGB, `(H, W, 3)` uint8. Dla super-rozdzielczości obszar jest
`Results.restore_scale` razy większy od wejścia. `array` zwraca widok numpy,
a `save(path)` zapisuje obraz.

## Matte

Miękki matte krycia, float32 `(H, W)` w zakresie `[0, 1]` na pierwotnym obszarze
obrazu. `1` oznacza w pełni pierwszy plan, a `0` w pełni tło. Miękki matte
obejmuje twardą maskę usuwania tła progowaną przy 0.5 i zachowuje wygładzone
krawędzie, które odrzuca maska binarna. `array` zwraca widok numpy.

Dla wyniku matte `Results.cutout(image=None)` zwraca tablicę RGBA uint8
`(H, W, 4)`, której czwarty kanał jest matte, a `Results.save(path, image=None)`
zapisuje ten wycięty obraz jako PNG z przezroczystym tłem. Obie metody pobierają
RGB z `image`, jeśli je podano, a w przeciwnym razie wczytują je ponownie
z `Results.path`.

## OCRRegions

Zlokalizowany tekst wraz z transkrypcjami. `data` to wielokąty
zmiennoprzecinkowe `(N, 4, 2)` w pikselach pierwotnego obrazu, uporządkowane jako
lewy górny, prawy górny, prawy dolny i lewy dolny. Regiony występują w kolejności
czytania, od góry do dołu, a następnie od lewej do prawej. `texts` jest listą N
transkrypcji. `conf` jest wynikiem rozpoznawania dla każdego regionu, a
`det_conf` wynikiem detekcji. Oba mają kształt `(N,)`.

Czworokąty detekcji są rzeczywistymi wielokątami, więc nie wypełniają
`Results.boxes`. `xyxy` zwraca ich obwiednie wyrównane do osi.

## Embeddings

Wektory znormalizowane normą L2 z zadania `embed`, zawsze o kształcie `(N, D)`.
Wynik całego obrazu zawiera jeden wiersz i nie ma ramek, a embeddingi regionów
są wyrównane wierszami z `boxes`. Ponieważ każdy wiersz jest znormalizowany,
podobieństwo cosinusowe jest iloczynem skalarnym.

| Element | Zwraca |
|---|---|
| `dim` | `D` |
| `normalized` | Wiersze ponownie znormalizowane |
| `similarity(other)` | Podobieństwo cosinusowe każdej pary względem innego `Embeddings` lub tensora |
| `verify(i, j, threshold=0.4)` | `True`, gdy wiersze `i` i `j` są zgodne |

## Identities

Nazwane dopasowania z galerii, wyrównane wierszami z `embeddings`. Są tworzone,
gdy `Gallery` zostanie przekazane do predykcji `embed`. `name` jest listą,
w której wpis ma wartość `None` poniżej progu dopasowania, a nazwa najbliższa,
lecz niespełniająca progu, nigdy nie jest zgadywana. `score` jest tablicą
wyników dopasowania, a `data` łączy ją z nazwami.

## Meshes

Parametryczne siatki ludzkiego ciała, wyrównane wierszami z ramkami osób
w `boxes`. Wszystko znajduje się w układzie kamery pierwotnego obrazu. `transl`
jest wartością metryczną w metrach, gdzie `+z` wskazuje od kamery. `vertices`
i `joints3d` są metryczne i zawierają już `transl`. `joints2d` jest wyrażone
w pikselach pierwotnego obszaru obrazu, a nie wycięcia widzianego przez sieć.
Żadne pole nie zawiera układu świata ani grawitacji.

Układy parametrów różnią się między modelami ciała, dlatego kształty nie są
zakodowane na stałe. `body_model` nazywa parametryzację, a liczby są odczytywane
z tensorów: `num_vertices`, `num_joints`, `num_betas` i `has_vertices`.
`params` zwraca słownik parametrów, a `save_obj(path, index=0)` zapisuje jedną
siatkę. Pola to `global_orient`, `body_pose`, `betas`, `transl`, `vertices`,
`faces`, `joints3d`, `joints2d`, `conf`, `focal_length` i `extras`.

Dla `body_model="mhr"` obroty są kątami Eulera w radianach, a nie osią i kątem,
`body_pose` jest płaskim wektorem parametrów poszczególnych stawów zamiast
jednej trójki na staw, a `betas` są współczynnikami kształtu tożsamości. Skala
szkieletu, poza dłoni i wyraz twarzy znajdują się w `extras`.

## Konwersja i wybór

Każde pole danych ma metody `to(*args, **kwargs)`, `cpu()`, `cuda()` i `numpy()`,
a wywołanie jednej z nich na `Results` stosuje ją jednocześnie do każdego
wypełnionego pola.

<code-tabs name="convert" />

`result[idx]` wybiera wiersze ze wszystkich danych wyrównanych wierszami.
`len(result)` jest liczbą detekcji albo liczbą punktów, gdy nie ma ramek.
`result.update(...)` zwraca kopię z zastąpionymi nazwanymi polami. Przyjmuje
każde pole oraz `track_id` i `restore_scale`.

## summary i to_json

`summary(normalize=False, decimals=5, embeddings=False)` zwraca listę zwykłych
słowników, po jednym wierszu na detekcję, segment, punkt lub region, zależnie od
ustawionych pól. `to_json(**kwargs)` przekazuje argumenty do `summary` i zwraca
ciąg JSON.

`plot()` renderuje gęsty wynik normalnych lub krawędzi w jego kanonicznej
wizualizacji, a dla innych typów wyników zgłasza błąd. Obrazy z adnotacjami dla
pozostałych zadań pochodzą z `predict(save=True)`.
