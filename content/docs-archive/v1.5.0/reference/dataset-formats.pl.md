---
title: Formaty zbiorów danych
seo_title: Formaty zbiorów danych LibreYOLO dla każdego zadania
description: >-
  Kontrakt plików zbioru danych dla każdego kanonicznego zadania: klucze YAML,
  układy katalogów, wiersze etykiet, konwencje masek i map oraz loader
  odczytujący każdy format.
lead: >-
  Ta strona odzwierciedla kontrakt plików zbioru danych opisany w pliku
  docs/dataset_schema.md biblioteki. Obejmuje klucze YAML i układ na dysku
  wymagane przez każde kanoniczne zadanie.
keywords:
  - format zbioru danych LibreYOLO
  - format etykiet YOLO
  - data.yaml
  - zbiór masek segmentacji
  - format COCO panoptic
  - zbiór danych głębi
  - pose kpt_shape
last_verified: 1.5.0
verification: >-
  Odpowiada plikowi docs/dataset_schema.md w repozytorium libreyolo w wersji
  1.5.0, a nazwy loaderów sprawdzono z libreyolo/data/.
snippets:
  usage:
    - label: Parsowanie jednego wiersza etykiety detekcji
      language: python
      code: >
        from libreyolo.data import parse_yolo_label_line


        # class_id cx cy w h, znormalizowane do [0, 1]

        row = parse_yolo_label_line("0 0.5 0.5 0.25 0.5", 640, 480,
        num_classes=80)


        # (class_id, x1, y1, x2, y2, area) w pikselach

        print(row)
source_hash: a8282c079624044d
---

## Wspólna konfiguracja YAML

Dotyczy zadań `detect`, `segment`, `pose` i `obb`.

| Klucz | Wymagany | Znaczenie |
|---|---|---|
| `path` | | Katalog główny zbioru danych |
| `train` | Do trenowania | Obrazy treningowe |
| `val` | Do walidacji | Obrazy walidacyjne |
| `test` | | Obrazy testowe |
| `names` | Tak | Lista klas lub mapowanie z kluczami całkowitymi |
| `nc` | | Liczba klas; jeśli występuje, musi być zgodna z `names` |
| `download` | | Instrukcje pobierania; skrypty Pythona wymagają jawnej zgody |
| `annotations` | | Mapowanie splitu na natywny plik COCO JSON dla zadań detect, segment i obb |

Pola `train`, `val` i `test` mogą wskazywać katalogi obrazów, pliki `.txt`
z listami obrazów albo listy takich elementów. Ścieżki etykiet wynikają z jednej
zamiany:

```text
images/.../image.jpg -> labels/.../image.txt
```

Dla zbioru danych w natywnym formacie COCO JSON pole `annotations` mapuje split
na jego plik JSON, a ścieżka splitu wskazuje katalog główny obrazów:

```yaml
path: dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

Jeśli występuje `names`, nazwy kategorii w natywnym pliku COCO JSON muszą być
zgodne z nazwami klas w YAML, które wyznaczają identyfikatory etykiet modelu.
Bez `names` identyfikatory kategorii COCO są sortowane i gęsto mapowane na
`0..N-1`.

Plik YAML zbioru danych nie zawiera klucza `task`. Pierwszeństwo ma jawny wybór
modelu i zadania.

Reguły wspólne dla każdego tekstowego pliku etykiet:

- jeden plik etykiet `.txt` na obraz;
- brakujący lub pusty plik etykiet oznacza brak obiektów;
- `class_id` jest liczbą całkowitą w zakresie `0..nc-1`;
- współrzędne są skończonymi, znormalizowanymi liczbami zmiennoprzecinkowymi w zakresie `[0, 1]`;
- współrzędne odnoszą się do pierwotnej szerokości i wysokości obrazu;
- wiersze nie zawierają pewności ani identyfikatora śledzenia.

<code-tabs name="usage" />

## detect

Dokładnie pięć pól w każdym wierszu:

```text
<class_id> <cx> <cy> <w> <h>
```

`cx cy w h` opisuje znormalizowaną ramkę wyrównaną do osi, a `w` i `h` muszą
być dodatnie.

## segment

Wiersz wielokąta:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

`N` wynosi co najmniej 3, liczba współrzędnych po `class_id` musi być parzysta,
a wielokąt nie może być zdegenerowany. Akceptowany jest również pięciopolowy
wiersz detekcji, który reprezentuje prostokątny segment.

## pose

YAML dodaje wymagane pole `kpt_shape` o wartości `[K, 2]` lub `[K, 3]` oraz
opcjonalne `flip_idx`, czyli całkowitoliczbową permutację zakresu `0..K-1`.

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

Liczba pól wynosi dokładnie `5 + K * D`, gdzie `D` jest drugą wartością
`kpt_shape`. Współrzędne punktów kluczowych są znormalizowane. Widoczność `v`,
jeśli występuje, ma wartość `0`, `1` lub `2`.

## obb

Dokładnie dziewięć pól:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

Cztery punkty są znormalizowanymi współrzędnymi obrazu w zakresie `[0, 1]`
i tworzą niezdegenerowany obrócony prostokąt. Plik etykiety nie przechowuje
kąta.

Kanoniczny parser jest domyślnie ścisły i odrzuca współrzędne spoza zakresu.
Podczas wczytywania zbioru danych i walidacji współrzędne prawidłowych poza tym
etykiet na granicy przycięcia mogą zostać ograniczone do `[0, 1]`, po czym
zdegenerowane ramki nadal są odrzucane. Parsowanie uwzględnia zadanie: dziewięć
pól oznacza `obb` tylko w trybie `obb`, natomiast w trybie `segment` może
oznaczać wielokąt z czterema punktami.

Wewnętrznie znormalizowane narożniki są konwertowane na kanoniczny format
`xywhr`, gdzie kąt w radianach opisuje obrót boku szerokości wokół środka ramki.
Publiczne wyniki udostępniają detekcje OBB jako wiersze `xywhr, conf, cls`.

Natywne wczytywanie OBB z COCO JSON akceptuje adnotacje w następującej
kolejności pierwszeństwa: `obb` jako osiem narożników w przestrzeni pikseli;
`obb` jako `[cx, cy, w, h, angle]` z kątem w radianach; wielokąt lub RLE COCO
`segmentation`, ponownie dopasowane do prostokąta o minimalnym polu; oraz `bbox`
COCO, odczytane jako wyrównane do osi i poddane kanonizacji.

Mosaic i mixup są wyłączone dla trenowania OBB do czasu udostępnienia augmentacji
OBB uwzględniającej narożniki.

Kanoniczny parser wiersza to `libreyolo.data.parse_yolo_obb_label_line`.

## semantic

Każdemu obrazowi odpowiada gęsta, jednokanałowa maska w formacie bezstratnym,
zwykle PNG, zamiast pliku `.txt`:

```text
images/.../image.jpg -> <masks_dir>/.../image.png
```

Maska jest jednokanałowa, a pliki PNG w trybie palety są odczytywane jako
indeksy palety. Każda wartość piksela jest identyfikatorem klasy z zakresu
`0..nc-1`. Wartość piksela `255` oznacza ignorowanie i jest wyłączona z funkcji
straty oraz metryk, a rozdzielczość maski musi być równa rozdzielczości obrazu.

Kontrakt wspólny rozszerzają dwa opcjonalne klucze YAML. `masks_dir` to nazwa
katalogu masek podstawiana za `images` w każdej ścieżce obrazu, domyślnie
`masks`. `label_mapping` to mapowanie `{source_id: train_id}` stosowane do
wartości pikseli maski podczas wczytywania. Nieodwzorowane wartości źródłowe
stają się ignorowane, a identyfikatory treningowe muszą mieścić się w zakresie
`0..nc-1`.

Gdy pominięto `masks_dir`, maski są rasteryzowane podczas wczytywania z etykiet
wielokątów `segment`, rozwiązywanych zgodnie z konwencją zamiany `images` na
`labels`. Po klasach obiektów dodawana jest klasa `background`, więc `nc`
wzrasta o jeden.

Kanoniczny loader: `libreyolo.data.SemanticDataset`.

## panoptic

LibreYOLO przyjmuje format COCO-panoptic bez zmian (Kirillov i in., CVPR 2019).
Nie istnieje format panoptyczny właściwy dla LibreYOLO.

Jeden plik RGB PNG na obraz, w rozdzielczości obrazu, koduje identyfikator
segmentu każdego piksela w jego kolorze:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

Każdy piksel należy dokładnie do jednego segmentu, a segmenty nigdy się nie
nakładają. Identyfikator segmentu `0`, czarny kolor RGB, oznacza void, czyli
nieopisane piksele wyłączone z metryki.

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1, "supercategory": "person"}]
}
```

`annotations[].file_name` wskazuje plik PNG identyfikatorów segmentów w
`panoptic_dir`, a `segments_info[].id` odpowiada wartości w tym pliku PNG.
`iscrowd` oznacza obszary grupowe: nigdy nie są one fałszywie ujemne, a predykcja
pokrywająca w większości taki obszar nie jest fałszywie dodatnia.

Podział thing i stuff jest właściwością kategorii. `isthing` znajduje się
w `categories`, nigdy w `segments_info`.

Wartości `category_id` COCO-panoptic są surowymi identyfikatorami zbioru danych
i zwykle nie są ciągłe. Modele przewidują ciągły zakres `0..nc-1`, dlatego
surowe identyfikatory są mapowane przez YAML `names` według nazwy kategorii,
zgodnie z tą samą regułą, której używa natywny loader detekcji COCO JSON.
Kategoria JSON nieobecna w `names` powoduje błąd zamiast cichego pominięcia,
ponieważ w przeciwnym razie byłaby zawsze oceniana jako fałszywie ujemna.

```yaml
path: coco
val: images/val2017
annotations:
  val: annotations/panoptic_val2017.json
panoptic_dir:
  val: annotations/panoptic_val2017
names: {0: person, 1: bicycle, 132: rug-merged}
```

Pola `annotations` i `panoptic_dir` przyjmują pojedynczą ścieżkę albo mapowanie
dla poszczególnych splitów.

Walidacja raportuje jakość panoptyczną, obliczaną w rozdzielczości danych
referencyjnych i uśrednianą dla występujących kategorii, a następnie dzieloną
na `PQ_things` i `PQ_stuff`. Dopasowanie jest jednoznaczne: przewidywany segment
i segment referencyjny tej samej kategorii są dopasowane, gdy IoU przekracza
0.5.

Kanoniczny loader: `libreyolo.data.PanopticDataset`.

## depth

Każdemu obrazowi odpowiada gęsta, jednokanałowa mapa głębi:

```text
images/.../image.jpg -> <depths_dir>/.../image.png
```

Mapa jest jednokanałowym plikiem PNG lub TIF albo plikiem `.npy` o rozdzielczości
obrazu. Wartości są zwykłą głębią w jednostce spójnej dla zbioru danych. Zero,
wartości ujemne, NaN i wartości nieskończone oznaczają nieprawidłowe piksele
i są wyłączone z funkcji straty oraz metryk.

| Klucz | Wartość domyślna | Znaczenie |
|---|---|---|
| `depths_dir` | `depths` | Katalog głębi podstawiany za `images` |
| `depth_stem_suffix` | | Sufiks dołączany do rdzenia nazwy obrazu; w razie pominięcia sprawdzany jest zarówno ten sam rdzeń, jak i sufiks `_depth` |
| `depth_mask_suffix` | `_mask` | Sufiks maski poprawności; wartości maski mniejsze lub równe zero, NaN i nieskończone unieważniają piksel głębi |
| `depth_scale` | `256.0` | Dzielnik map głębi o typie całkowitym, zgodny z popularną konwencją 16-bitowego PNG |

Mapy zmiennoprzecinkowe `.npy` są używane bez zmian i nie stosują
`depth_scale`.

Kanoniczny loader: `libreyolo.data.DepthDataset`.

## edge

Każdemu obrazowi RGB odpowiada jednokanałowa, bezstratna mapa o tym samym
rdzeniu nazwy i opcjonalna maska poprawności:

```text
images/val/scene.jpg -> edges/val/scene.png
                     -> masks/val/scene.png
```

Mapa jest jednokanałowym plikiem PNG lub TIF, a nie wizualizacją RGB,
w rozdzielczości obrazu. Mapy całkowitoliczbowe są dzielone przez maksimum
danego typu, a mapy zmiennoprzecinkowe muszą już zawierać skończone wartości
w zakresie `[0, 1]`. `0` oznacza brak krawędzi, a `1` oznacza krawędź.
Piksele opcjonalnej maski są prawidłowe, gdy mają wartość różną od zera.
Zmiana rozmiaru celów i masek używa interpolacji najbliższego sąsiada,
a piksele dopełnienia są nieprawidłowe i nie wpływają na walidację.

| Klucz | Wartość domyślna | Znaczenie |
|---|---|---|
| `edges_dir` | `edges` | Katalog map krawędzi podstawiany za `images` |
| `edge_stem_suffix` | | Sufiks dołączany do rdzeni nazw obrazów |
| `edge_extension` | `.png` | Bezstratne rozszerzenie celu |
| `edge_invert` | | Ustawione na true, gdy mapy źródłowe przechowują czarne krawędzie na białym tle |
| `masks_dir` | `masks` | Opcjonalny katalog masek poprawności |

```yaml
path: edge-dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

Walidacja pocienia ciągłe predykcje za pomocą tłumienia niemaksymalnych wartości
gradientu w czterech kierunkach oraz raportuje miary F ODS i OIS dla
konfigurowalnego przeglądu progów. Piksele przewidywane i referencyjne są
jednoznacznie dopasowywane w promieniu `edge_max_dist * image_diagonal`,
z domyślną znormalizowaną tolerancją `0.0075`.

Kanoniczny loader: `libreyolo.data.EdgeDataset`. Loader obsługuje wyłącznie
format i nie pobiera ani nie rozpowszechnia danych benchmarkowych.

## normal

Każdemu obrazowi odpowiada trzykanałowy 16-bitowy plik PNG o tym samym rdzeniu
nazwy oraz opcjonalna maska poprawności o tym samym rdzeniu:

```text
images/val/room.jpg -> normals/val/room.png
                    -> masks/val/room.png
```

Plik PNG ma dokładnie trzy kanały `uint16`, zapisane jako RGB, i rozdzielczość
obrazu. Należy go dekodować wzorem `n = png / 65535 * 2 - 1`, a następnie
ponownie znormalizować każdy wektor. Zdekodowane wektory używają układu kamery
OpenCV: `+x` w prawo, `+y` w dół, `+z` w głąb sceny, i są skierowane do kamery.
Opcjonalna maska jest jednokanałowym plikiem PNG, gdzie wartość różna od zera
oznacza poprawność. Bez maski każdy skończony, niezerowy zdekodowany wektor jest
poprawny. Nieprawidłowe i dopełnione piksele celu są wewnętrznie reprezentowane
przez `(0, 0, 0)`. Zmiana rozmiaru interpoluje trzy składowe dwuliniowo,
a następnie normalizuje je ponownie. Maski poprawności używają interpolacji
najbliższego sąsiada, a odbicie poziome dodatkowo neguje składową x.

| Klucz | Wartość domyślna | Znaczenie |
|---|---|---|
| `normals_dir` | `normals` | Katalog map normalnych podstawiany za `images` |
| `masks_dir` | `masks` | Opcjonalny katalog masek poprawności |

Walidacja raportuje średni i medianowy błąd kątowy w stopniach oraz odsetek
poprawnych pikseli mieszczących się w 11.25, 22.5 i 30 stopniach.

Kanoniczny loader: `libreyolo.data.NormalDataset`.

## restore

Każdemu zdegradowanemu obrazowi wejściowemu odpowiada czysty cel RGB:

```text
inputs/.../image.jpg -> targets/.../image.jpg
```

Wejście i cel są plikami obrazów zgodnymi z RGB, a ich rozdzielczości muszą być
identyczne. Walidacja zachowuje natywną rozdzielczość i stosuje tylko dopełnienie
potrzebne do ułożenia batcha, a metryki są obliczane na pierwotnym obszarze
obrazu. Trenowanie stosuje sprzężone przycięcie i odbicie poziome do pary wejścia
i celu.

| Klucz | Wartość domyślna | Znaczenie |
|---|---|---|
| `input_dir` | `inputs` | Katalog zdegradowanych wejść używany w ścieżkach splitów |
| `target_dir` | `targets` | Katalog czystych celów podstawiany za `input_dir` |
| `target_stem_suffix` | | Sufiks dołączany do rdzenia nazwy wejścia przed wyszukaniem celu |
| `target_stem_suffixes` | | Postać listy dla `target_stem_suffix` |
| `degradation` | | Etykieta metadanych, taka jak `deblur` lub `denoise` |
| `dataset` | | Etykieta zbioru danych lub pochodzenia |

Pola YAML podobne do klas są polami zastępczymi schematu. Należy użyć `nc: 1`
i `names: {0: image}`. Modele przywracania udostępniają `Results.restored`,
a nie detekcje.

Kanoniczny loader: `libreyolo.data.RestoreDataset`.

## matte

Każdemu obrazowi RGB odpowiada jednokanałowy referencyjny matte o tym samym
rdzeniu nazwy, gdzie 0 oznacza tło, a 255 pierwszy plan:

```text
images/subject.jpg -> mattes/subject.png
```

Akceptowane są dwa układy. Pierwszy to katalog główny zawierający `images/`
oraz katalog matte, automatycznie wykrywany wśród `mattes/`, `matte/`, `gt/`,
`masks/`, `mask/` i `alpha/`, przekazany jako `data=`. Drugi to plik YAML
z `path` oraz właściwymi dla splitów polami `val_images` i `val_mattes`,
a opcjonalnie także `train_images` i `train_mattes`. Każda ścieżka może być
względna wobec `path` lub bezwzględna.

Matte jest obrazem w skali szarości, odczytywanym jako krycie w zakresie
`[0, 1]`. Gdy kształty się różnią, jego rozmiar jest zmieniany do obszaru
predykcji za pomocą interpolacji dwuliniowej. Metryki to MAE i S-measure
(Fan i in., ICCV 2017) na pierwotnym obszarze obrazu, a S-measure jest miarą
fitness wybierającą najlepszy checkpoint.

Pola YAML podobne do klas są polami zastępczymi schematu. Należy użyć `nc: 1`
i `names: {0: matte}`. Modele matte udostępniają `Results.matte`.

W tej wersji walidacja obsługuje tylko inferencję. Kanoniczny moduł rozwiązywania
par: `libreyolo.data.matte_dataset.resolve_matte_pairs`.

## ocr

Etykiety mają postać jednego pliku JSONL na split, z jednym obiektem JSON na
obraz:

```text
images/val/receipt.jpg -> labels/val.jsonl
```

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon` jest czworokątem z czterech punktów w bezwzględnych współrzędnych
pikseli, uporządkowanych jako lewy górny, prawy górny, prawy dolny i lewy dolny.
Regiony z nieczytelnym tekstem używają `"text": "###"`, zgodnie z konwencją
ICDAR do-not-care. Są wyłączone z oceny rozpoznawania, a predykcje nakładające
się na nie są ignorowane zamiast karane podczas dopasowania detekcji.

Metryki obejmują hmean detekcji z jednoznacznym dopasowaniem wielokątów powyżej
IoU 0.5, F1 od początku do końca wymagające zarówno IoU powyżej 0.5, jak
i dokładnej transkrypcji po normalizacji NFKC i usunięciu białych znaków,
z rozróżnianiem wielkości liter, oraz 1-NED dla dopasowanych par. Miara fitness
wybierająca najlepszy checkpoint to F1 od początku do końca.

Akceptowane są dwa układy: katalog główny zawierający `images/<split>/`
i `labels/<split>.jsonl`, przekazany jako `data=`, albo YAML z `path` oraz
opcjonalnymi nazwami katalogów `images` i `labels`.

Pola YAML podobne do klas są polami zastępczymi schematu. Należy użyć `nc: 1`
i `names: {0: text}`. Modele OCR udostępniają `Results.ocr`.

W tej wersji walidacja obsługuje tylko inferencję. Kanoniczny moduł rozwiązywania
próbek: `libreyolo.data.ocr_dataset.resolve_ocr_samples`.

## classify

Drzewo katalogów w stylu ImageFolder, a nie pliki etykiet:

```text
dataset_root/
  train/
    class_a/*.jpg
    class_b/*.jpg
  val/
    class_a/*.jpg
    class_b/*.jpg
```

Katalog `train/` jest wymagany do trenowania i wyznacza mapowanie klas na
indeksy według posortowanych nazw katalogów. Katalog `val/` jest wymagany do
walidacji. Katalog `test/` może występować, ale domyślne polecenia trenowania
i walidacji go nie używają. Splity inne niż treningowy muszą zawierać te same
nazwy katalogów klas co oczekiwany zbiór klas treningowych lub klas
checkpointu. Obsługiwane rozszerzenia obrazów są zdefiniowane w
`libreyolo.data.classify_dataset.IMAGE_EXTENSIONS`.

## gaze i point

Dla `gaze` nie zaimplementowano kontraktu plików zbioru danych do trenowania
ani walidacji.

`point` jest zadaniem wyjściowym modelu, a nie schematem etykiet zbioru danych.
Rodziny point mogą wewnętrznie dostosowywać istniejące etykiety, na przykład
wyznaczając środki obiektów z wierszy ramek, ale nie zdefiniowano tekstowego
formatu etykiet przeznaczonego wyłącznie dla point.
