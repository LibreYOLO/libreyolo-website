---
title: Detekcja obróconych obiektów
seo_title: Detekcja obróconych obiektów w LibreYOLO
description: >-
  Wykrywaj obrócone obiekty w LibreYOLO: poznaj rodziny obsługujące obrócone
  ramki, format etykiety z czterema narożnikami oraz wywołania predykcji,
  trenowania, walidacji i eksportu.
lead: >-
  Detekcja obróconych obiektów lokalizuje każdą instancję za pomocą obróconego
  prostokąta zamiast prostokąta wyrównanego do osi. Dzięki temu pochylony obiekt
  jest ciasno ograniczony, a ramka nie zawiera dużej ilości tła. Klucz zadania
  to obb.
keywords:
  - detekcja obróconych ramek
  - rotated object detection
  - OBB python
  - zbiór danych DOTA
  - detekcja obiektów lotniczych
  - rotated IoU
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        # Wymaga dodatku rfdetr: pip install "libreyolo[rfdetr]"

        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Sufiks -obb w nazwie pliku wybiera zadanie, dlatego argument

        # task nie jest potrzebny.

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        result = model(SAMPLE_IMAGE, save=True)


        obb = result.obb

        print(obb.xywhr)   # (N, 5): środek x, środek y, szerokość, wysokość,
        radiany

        print(obb.conf, obb.cls)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRFDETRs-obb.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Narożniki zamiast kątów
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreRFDETRs-obb.pt")(SAMPLE_IMAGE)
        obb = result.obb

        print(obb.xyxyxyxy.shape)    # punkty narożne (N, 4, 2) w pikselach
        print(obb.xyxyxyxyn.shape)   # te same punkty po normalizacji
        print(obb.xyxy.shape)        # otaczająca ramka (N, 4) wyrównana do osi
    - label: Mniejszy checkpoint
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRn-obb.pt")
        result = model(SAMPLE_IMAGE)

        print(result.obb.xywhr.shape)
    - label: RT-DETRv2
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Wagi DOTA v1.0, 15 klas lotniczych przy 1024 px. Graf obróconych ramek

        # jest rozpoznawany z tensorów checkpointu, więc argument task jest
        zbędny.

        model = LibreYOLO("LibreRTDETRv2n-obb.pt")

        result = model("aerial.png", save=True)


        obb = result.obb

        print(obb.xywhr)

        print(result.names)   # samolot, statek, port, helikopter i 11 innych
        klas
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Kontynuuje od opublikowanych wag obróconych ramek. data musi wskazywać

        # zbiór danych, którego wiersze etykiet zawierają cztery narożniki.

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        model.train(data="my-obb-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: Z wag detekcji
      language: bash
      code: |
        # Wagi detekcji nie przewidują kąta, więc jest to jawny transfer.
        # Żądanie task=obb udziela na niego zezwolenia.
        libreyolo train model=LibreRFDETRs.pt data=my-obb-dataset.yaml \
          task=obb epochs=50 imgsz=512
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        # val() zwraca zwykły słownik, a nie obiekt.
        metrics = model.val(data="my-obb-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml
    - label: RT-DETRv2
      language: bash
      code: |
        libreyolo val model=LibreRTDETRv2n-obb.pt data=my-obb-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRFDETRs-obb.pt format=onnx imgsz=512
    - label: RT-DETRv2
      language: bash
      code: >
        # ONNX i TorchScript są tutaj zweryfikowanymi celami, z FP32,

        # batchem 1 oraz stałym obszarem roboczym 1024 na 1024.

        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024

        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript
        imgsz=1024
    - label: Używanie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie sufiksu pliku, więc
        wyeksportowany artefakt

        # wczytuje się jak checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreRFDETRs-obb.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.obb.xywhr)
source_hash: 0d605d956f3ea025
---

## Definicja

Detekcja obróconych obiektów dodaje do detekcji jedną liczbę: kąt. Każda
instancja otrzymuje obrócony prostokąt, klasę i wynik. Korzyścią jest ścisłe
dopasowanie. Statek ustawiony pod kątem 45 stopni, dach magazynu lub rząd
zaparkowanych ciężarówek zostałby otoczony ramką wyrównaną do osi, która w
większości zawierałaby tło. Dwie sąsiednie ramki nakładałyby się nawet wtedy,
gdy obiekty się nie stykają. Dlatego zadanie to jest standardem w obrazowaniu
lotniczym i analizie układu dokumentów, a jego referencyjnym zbiorem danych jest
DOTA.

`obb` jest kanonicznym kluczem zadania, a sufiks `-obb` w nazwie pliku
checkpointu wybiera to zadanie, więc `task=` nie jest potrzebne przy wczytywaniu
opublikowanych wag.

`predict()` wypełnia `result.obb`. Pole `.xywhr` ma kanoniczną postać `(N, 5)`:
środek x, środek y, szerokość, wysokość i kąt w radianach określający obrót boku
szerokości wokół środka. Pola `.conf` i `.cls` zawierają wynik i indeks klasy w
`result.names`, a `.id` identyfikator ścieżki podczas śledzenia. Pole
`.xyxyxyxy` przekształca każdy wiersz w cztery punkty narożne jako piksele
`(N, 4, 2)`, `.xyxyxyxyn` normalizuje te narożniki, a `.xyxy` zwraca otaczającą
ramkę wyrównaną do osi. Tej ostatniej postaci należy użyć, gdy dalszy kod
rozumie tylko prostokąty. `result.boxes` również jest wypełnione postacią
wyrównaną do osi.

## Modele

To zadanie obsługują dwie rodziny. Wybór zależy od tego, czy potrzebne jest
trenowanie.

[RF-DETR](/docs/models/rf-detr) jest rodziną, którą można trenować. Przewiduje,
trenuje, waliduje i eksportuje obrócone ramki oraz udostępnia opublikowane
checkpointy obróconych ramek w czterech rozmiarach: n, s, m i l. Wymaga własnego
dodatku `pip install "libreyolo[rfdetr]"`, a strona modelu zawiera informacje o
licencji i pochodzeniu wag.

Przed zaplanowaniem użycia tych checkpointów należy przeczytać poniższą sekcję,
która wyjaśnia, co faktycznie przewidują.

[RT-DETRv2](/docs/models/rt-detr) jest rodziną z wagami dla obrazów lotniczych.
Udostępnia modele od `LibreRTDETRv2n-obb.pt` do `LibreRTDETRv2x-obb.pt`, czyli
oficjalne checkpointy DOTA v1.0 w jednej skali, przekonwertowane do formatu
LibreYOLO. Obejmują one 15 klas DOTA przy 1024 px. Rodzina nie wymaga dodatków
poza pakietem podstawowym, graf obróconych ramek jest rozpoznawany z tensorów
checkpointu, a predykcja, walidacja oraz eksport do ONNX i TorchScript są w
pełni obsługiwane. Trenowanie nie jest obsługiwane. Zadanie obróconych ramek w
tej rodzinie służy tylko do inferencji, `train()` zgłasza błąd i nie ma transferu
z wag detekcji, które używają innego backbone. Śledzenie i augmentacja w czasie
testu również nie są dostępne dla obróconych ramek.

Podsumowując, do gotowych klas DOTA służy RT-DETRv2, a do własnych etykiet
obróconych ramek RF-DETR.

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane lokalnie w
pamięci podręcznej.

<code-tabs name="predict" />

Przed uruchomieniem opublikowanych checkpointów RF-DETR warto wiedzieć, czego
dotyczą. Choć DOTA jest referencyjnym benchmarkiem tego zadania, wagi te nie
zostały na nim wytrenowane. Wszystkie cztery zainicjowano wagami detekcji RF-DETR
i dostrojono na jednym zbiorze danych Roboflow Universe z nagraniami z dronów,
obejmującym sześć klas pojazdów: bike, bus, car, other_vehicle, taxi i truck.
Karty modeli opisują je jako wagi rozwojowe utworzone podczas walidacji obsługi
trenowania obróconych ramek i zaznaczają, że nie należy ich traktować jako wag
produkcyjnych ani oficjalnych dla benchmarku.

W praktyce są działającym punktem wyjścia dla obróconych ramek pojazdów
widzianych z góry oraz do sprawdzenia, czy pipeline działa od początku do końca.
Każda inna dziedzina wymaga trenowania na własnych etykietach obróconych ramek.
Dla kategorii lotniczych znanych ze zbioru DOTA właściwym wyborem są checkpointy
RT-DETRv2, rzeczywiście wytrenowane na tych danych. `conf` i `max_det` kształtują
dane wyjściowe tak samo jak w detekcji. Informacje o źródłach, streamingu i
obsłudze wyników znajdują się w sekcji [predykcja](/docs/predict).

## Format zbioru danych

Układ odpowiada detekcji: na każdy obraz przypada jeden plik etykiet `.txt`,
znajdowany przez zastąpienie `images` ciągiem `labels` w ścieżce obrazu i zmianę
rozszerzenia.

```text
dataset/
  data.yaml
  images/
    train/P0001.png
    val/P0101.png
  labels/
    train/P0001.txt
    val/P0101.txt
```

Wiersz zawiera dokładnie dziewięć pól: indeks klasy, po którym następują cztery
punkty narożne w kolejności:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

Cztery punkty są znormalizowanymi liczbami zmiennoprzecinkowymi z zakresu
`[0, 1]` i muszą tworzyć niezdegenerowany obrócony prostokąt. W pliku etykiet
nie zapisuje się kąta. Moduł wczytujący wyprowadza z narożników kanoniczną postać
`xywhr`. Parser jest domyślnie rygorystyczny i odrzuca współrzędne spoza zakresu,
natomiast podczas wczytywania zbioru danych i walidacji może najpierw ograniczyć
do `[0, 1]` poprawne poza tym etykiety na granicy wycinka, po czym nadal odrzuci
zdegenerowane ramki.

Analiza wierszy uwzględnia zadanie. Dziewięć pól oznacza obróconą ramkę tylko w
trybie `obb`. W trybie `segment` ten sam wiersz jest odczytywany jako wielokąt z
czterema punktami.

Plik YAML ma format używany w detekcji:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: plane
  1: ship
```

Można także wczytać natywny plik COCO JSON z mapowaniem `annotations` między
nazwą podziału a plikiem JSON. Adnotacje są odczytywane w następującej
kolejności priorytetów: pole `obb` z ośmioma narożnikami w przestrzeni pikseli,
pole `obb` w postaci `[cx, cy, w, h, angle]` z kątem w radianach, wielokąt
`segmentation` lub RLE dopasowane ponownie do prostokąta o minimalnym polu albo
zwykłe pole COCO `bbox`, traktowane jako prostokąt wyrównany do osi i
kanonizowane do `xywhr`.

Kanoniczny parser wiersza to `libreyolo.data.parse_yolo_obb_label_line`.

## Trenowanie

<code-tabs name="train" />

Trenowanie w tym zadaniu oznacza użycie RF-DETR. Domyślnie jest kontynuowane z
opublikowanego checkpointu `-obb`. Rozpoczęcie od wag detekcji jest świadomym
transferem. Wagi te nie przewidują kąta, a przekazanie `task=obb` zezwala na
zamianę. Wartość `lr0` należy utrzymać na poziomie `1e-4` lub niższym, podobnie
jak w pozostałych zadaniach tej rodziny. Checkpointów obróconych ramek RT-DETRv2
nie można dostrajać. Należy używać ich bez zmian lub wytrenować model RF-DETR na
własnych etykietach. Informacje o zbiorach danych, augmentacji, wielu GPU i
loggerach znajdują się w sekcji [trenowanie](/docs/train).

## Walidacja

`val()` zwraca zwykły słownik kluczy `metrics/`. Dopasowanie używa obróconego
IoU, obliczanego między obróconymi prostokątami, a nie między ich otaczającymi
ramkami wyrównanymi do osi. Dlatego predykcja o prawidłowym położeniu, ale
błędnym kącie, jest uznawana za chybioną.

<code-tabs name="val" />

`metrics/mAP50-95` to średnia precyzja uśredniona dla progów IoU od 0.50 do 0.95
z krokiem 0.05 i jest głównym wynikiem. W przeciwieństwie do ścieżki COCO
używanej w detekcji to zadanie respektuje `iou_thresholds` w konfiguracji
walidacji, więc zakres można zmienić. `metrics/mAP50` i `metrics/mAP75` to
warianty z jednym progiem. `metrics/precision` i `metrics/recall` są rzeczywistą
precyzją i wartością recall przy IoU 0.50, odczytywanymi w najluźniejszym punkcie
pracy. Liczy się każda predykcja, która przetrwała próg pewności, a próg ten
domyślnie wynosi 0.001 podczas walidacji. Podniesienie `conf` zmienia więc te
wartości, podczas gdy wyniki mAP, które używają całej krzywej precision-recall,
pozostają niezmienione. Cztery z tych metryk powtarzają się z sufiksem `(OBB)`:
`metrics/mAP50-95(OBB)`, `metrics/mAP50(OBB)`, `metrics/precision(OBB)` i
`metrics/recall(OBB)`. Dzięki temu kod wywołujący odróżnia wynik obrócony od
wyrównanego do osi, gdy oba znajdują się w tej samej tabeli. Metryka
`metrics/mAP75` nie ma odpowiednika z sufiksem.

Dwie opcje nie mają wpływu na to zadanie. `save_json` i `save_plots` są
akceptowane i powodują zapisanie ostrzeżenia w logu. Zrzuty predykcji obróconych
ramek i wykresy walidacji nie są zaimplementowane.

## Eksport

<code-tabs name="export" />

Wyeksportowany artefakt jest ponownie wczytywany przez `LibreYOLO()` na podstawie
sufiksu pliku, więc plik `.onnx` lub `.engine` zachowuje się jak checkpoint i
zwraca ten sam obiekt `Results`. Zakres formatów różni się między zadaniami w
tej samej rodzinie. Macierz na stronie modelu jest generowana ze zweryfikowanego
zestawu i podaje przyczynę niedostępności celu. Formaty, ich dodatki i
ograniczenia opisano w sekcji [eksport i wdrożenie](/docs/export).

