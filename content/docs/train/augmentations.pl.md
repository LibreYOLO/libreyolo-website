---
title: Augmentacje
seo_title: 'Augmentacje podczas trenowania w LibreYOLO, z przykładami'
description: >-
  Co robi każdy parametr augmentacji w LibreYOLO, pokazane na prawdziwym obrazie
  przez własny kod trenowania biblioteki, wraz z zakresem, wartością domyślną i
  rodzinami, które go odczytują.
lead: >-
  Augmentacje podczas trenowania to losowe transformacje stosowane do każdej
  próbki treningowej. Konfiguruje się je parametrami w TrainConfig, a każda
  rodzina modeli uruchamia własny pipeline, więc parametr zmienia próbki tylko
  tam, gdzie ten pipeline go odczytuje.
keywords:
  - augmentacja danych yolo
  - augmentacja mosaic
  - mixup
  - augmentacja hsv
  - random affine
  - random erasing
  - detekcja małych obiektów
  - przykłady augmentacji
  - augmentacja copy paste
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: 1.8.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_prob=1.0,
            mixup_prob=0.15,
            hsv_prob=1.0,
            flip_prob=0.5,
            no_aug_epochs=15,
        )
    - label: CLI
      language: bash
      code: |
        # W CLI mosaic_prob zapisuje się jako mosaic, a mixup_prob jako mixup.
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 mosaic=1.0 mixup=0.15 hsv_prob=1.0 \
          flip_prob=0.5 no_aug_epochs=15
  small:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # YOLOX zaczyna od mosaic_scale=(0.1, 2.0), co może zmniejszyć obiekt
        # do jednej dziesiątej rozmiaru. (0.5, 1.5) zatrzymuje się na połowie.
        model = LibreYOLO("LibreYOLOXs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_scale=(0.5, 1.5),
        )
    - label: CLI
      language: bash
      code: |
        # W CLI para to dwie liczby oddzielone przecinkiem.
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=100 mosaic_scale=0.5,1.5
  support:
    - label: Odczytywanie tabeli obsługi rodziny
      language: python
      code: |
        from libreyolo.data.augment.spec import AUG_KNOBS, aug_support

        for knob, description in AUG_KNOBS.items():
            support = aug_support("yolo9")[knob]
            print(f"{knob:16} {support.status:16} {support.note or description}")
    - label: Tylko ignorowane parametry
      language: python
      code: |
        from libreyolo.data.augment.spec import ignored_aug_params

        print(sorted(ignored_aug_params("rfdetr")))
  classify:
    - label: Pakiet klasyfikacji
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(
            data="my-classification-dataset",
            epochs=50,
            auto_augment="randaugment",
            erasing=0.25,
            mixup=0.2,
            cutmix=0.2,
        )
source_hash: 72d22a38d647de0a
---

## Ustawianie parametrów

Parametry augmentacji są zwykłymi argumentami `train()`.

<code-tabs name="train" />

Trzy z nich mają krótszą pisownię, akceptowaną zarówno przez `train()`, jak i przez CLI: `mosaic` odpowiada `mosaic_prob`, `fliplr` odpowiada `flip_prob`, a w detekcji `mixup` odpowiada `mixup_prob`. Przekazanie krótkiej pisowni i odpowiadającego jej pola z różnymi wartościami zgłasza wyjątek. `perspective` nie ma flagi CLI w wersji 1.8.0, dlatego ustawia się go z poziomu Pythona.

Parametry klasyfikacji są walidowane podczas budowania ich pipeline'u, a CLI dodatkowo sprawdza `flip_prob`, `flipud`, `cutmix` i `erasing`. `TrainConfig` nie sprawdza zakresów parametrów detekcji, więc przy `degrees=100` trenowanie odbywa się z obrotami do 100 stopni. [Macierz augmentacji](/docs/reference/augmentation-matrix) podaje zakres każdego parametru.

## Trzy stany, nie dwa

To, czy parametr cokolwiek zmienia, zależy od rodziny. Biblioteka przechowuje deklaratywną tabelę tych zależności, a każdy wpis przyjmuje jeden z trzech stanów.

`used` oznacza, że parametr dociera do pipeline'u i zmienia próbki. `ignored` oznacza, że nigdy nie dociera do pipeline'u, więc jego ustawienie niczego nie zmienia. `gated_by_mosaic` oznacza, że dotyczy tylko próbek, które przeszły przez gałąź mozaiki, więc przy `mosaic_prob=0` nigdy nie zadziała, mimo że jest podłączony.

To właśnie ten trzeci stan zaskakuje. W pipelinie w stylu YOLOX transformacja afiniczna działa na płótnie mozaiki, a MixUp miesza próbkę mozaikową, więc `mosaic_prob=0` po cichu wyłącza jednocześnie `degrees`, `translate`, `shear`, `perspective`, `mosaic_scale`, `mixup_prob` i `mixup_scale`. Trener zapisuje w logu ostrzeżenie dotyczące konkretnie przypadku MixUp:

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

CLI ostrzega również o ignorowanych parametrach, wymieniając tylko te, które faktycznie wpisano:

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

Funkcja `train()` wywołana z Pythona nie ostrzega o ignorowanych parametrach. [Macierz augmentacji](/docs/reference/augmentation-matrix) podaje stan każdego parametru dla każdej rodziny oraz wartość, od której zaczyna każda rodzina. Tabela to dostarczana specyfikacja w pliku `libreyolo/data/augment/spec.py`, którą własne testy biblioteki sprawdzają względem pipeline'ów, i można ją odczytać bezpośrednio:

<code-tabs name="support" />

## Działanie poszczególnych parametrów

Poniższe ilustracje powstały przy użyciu własnych transformacji trenowania biblioteki LibreYOLO 1.8.0, uruchomionych na przykładowym obrazie przez zbiór danych złożony z jednego obrazu, więc każdy kafelek mozaiki i każdy partner MixUp to to samo zdjęcie. Ramka oznacza osobę, którą model LibreYOLO9t wykrywa na tym obrazie. Pod każdym panelem podano użytą wartość oraz rozmiar największej ramki w pikselach przy rozdzielczości trenowania 640 pikseli. Bez augmentacji osoba zajmuje 291 × 217 px.

Kilka parametrów geometrycznych to zakresy: `degrees=10` oznacza kąt losowany z dowolnego miejsca przedziału od -10 do +10 stopni. Ich panele ustalają losowanie na wartości z podpisu, więc pokazują najsilniejszy efekt, na jaki pozwala dana wartość, a nie typową próbkę. Ilustracje dla detekcji pochodzą z pipeline'u YOLOv9. YOLOv7 trenuje przez niego bez zmian, a RT-DETR i RT-DETRv2 używają jego wrappera mozaiki z własną transformacją dla każdej próbki. YOLOX, RTMDet, PicoDet i FOMO budują mozaikę i transformację w ten sam sposób w osobnym wrapperze, którego MixUp działa inaczej.

### mosaic_prob

Prawdopodobieństwo, że próbka jest mozaiką: czterema obrazami umieszczonymi na płótnie dwa razy większym od rozmiaru wejścia, które transformacja afiniczna następnie sprowadza do rozmiaru wejścia. W pipeline'ach w stylu YOLOX transformacja i MixUp działają tylko na próbkach mozaikowych, więc obie operacje przestają działać, gdy ta wartość wynosi 0. RT-DETR, RT-DETRv2, PicoDet i FOMO mają ją domyślnie ustawioną na 0.

<aug-figure name="mosaic" />

Drugi panel ustala skalę transformacji na 0.5×, przy której mieści się całe płótno. Dwa ostatnie to zwykłe losowania z domyślnymi wartościami modelu YOLOv9, a podpis podaje skalę wylosowaną w każdym z nich.

### mosaic_scale

Zakres, z którego losowany jest zoom transformacji. Przy 1× obiekt zachowuje rozmiar, jaki ma podczas inferencji, a przy 0.5× ma połowę tego rozmiaru. Bazowa wartość domyślna to `(0.1, 2.0)`, którą zachowują YOLOX, YOLOv7, RTMDet, PicoDet, RT-DETR, RT-DETRv2 i FOMO, więc ich obiekty mogą zmaleć do jednej dziesiątej swojego rozmiaru. YOLOv9 ma domyślnie `(0.5, 1.5)`. YOLO-NAS i PP-YOLOE nie mają mozaiki i ponownie używają tego zakresu jako skali swojej transformacji afinicznej dla każdej próbki.

<aug-figure name="mosaic_scale" />

Powyżej 1× wynik pokazuje tylko lewą górną część płótna, więc obiekty są przycinane na krawędzi albo wychodzą poza kadr.

### degrees

Zakres obrotu: każda transformowana próbka obraca się o kąt losowany z przedziału od `-degrees` do `+degrees`. Obrót odbywa się wokół lewego górnego narożnika płótna, więc duży kąt wyprowadza znaczną część obrazu poza kadr. YOLOX i YOLOv7 zachowują bazową wartość domyślną 10, a YOLOv9 ma domyślnie 0.

<aug-figure name="degrees" />

### translate

Zakres przesunięcia jako ułamek boku wejścia: przy `imgsz=640` każda transformowana próbka przesuwa się na każdej osi o co najwyżej `translate` razy 640 px. Bazowa wartość domyślna to 0.1. YOLO-NAS i PP-YOLOE mają domyślnie 0.25, a RTMDet, PicoDet i FOMO 0.

<aug-figure name="translate" />

### shear

Zakres ścinania w stopniach, losowany osobno dla każdej osi. YOLOX i YOLOv7 zachowują bazową wartość domyślną 2, a każda inna rodzina, która odczytuje ten parametr, ma domyślnie 0.

<aug-figure name="shear" />

### perspective

Transformacja projekcyjna nakładana na obrót, skalowanie, ścinanie i przesunięcie. Każdy narożnik płótna przesuwa się do środka o losową odległość wynoszącą maksymalnie `min(perspective * 100, 0.2)` połowy boku płótna, na każdej osi, więc `perspective=0.001` przesuwa narożnik najwyżej o 5 procent boku przy dowolnej rozdzielczości, a każda wartość od `0.002` w górę osiąga to samo maksimum. Domyślne `0` pozostawia transformację afiniczną, a wartość ujemna zgłasza `ValueError` przy transformacji pierwszej próbki. Panele korzystają z jednego wspólnego losowania, przeskalowanego przez wartość. Stosują ją YOLOv9 i jego warianty E2E i P2, YOLOX, YOLOv7, YOLO-NAS, PP-YOLOE, RTMDet, PicoDet, RT-DETR i RT-DETRv2.

<aug-figure name="perspective" />

### hsv_prob

Prawdopodobieństwo losowego zaburzenia kolorów HSV. Parametrem jest tylko prawdopodobieństwo. Zaburzenie przesuwa odcień maksymalnie o 5 na skali odcienia OpenCV od 0 do 179, a nasycenie i wartość (V) maksymalnie o 30 na skali od 0 do 255, przy czym każde z trzech przesunięć jest pomijane z prawdopodobieństwem równym jednej drugiej. Każdy pipeline detekcji, który odczytuje `hsv_prob`, używa tych wartości. Trzy ostatnie panele pokazują najbardziej widoczne spośród 119 losowań.

<aug-figure name="hsv" />

### flip_prob i flipud

Prawdopodobieństwa odbicia poziomego i pionowego. `flip_prob` zaczyna od 0.5 w każdej rodzinie oprócz rodziny FOMO, która ma domyślnie 0. `flipud` zaczyna od 0. RTMDet, PicoDet, RT-DETR, RT-DETRv2, FOMO, PP-YOLOE oraz rodziny w stylu DETR nie mają odbicia pionowego i ignorują ten parametr. Receptura PP-YOLOE używa zamiast tego losowego obrotu o 90 stopni.

<aug-figure name="flip" />

### mixup_prob i mixup_scale

MixUp miesza drugą próbkę z bieżącą i zachowuje ramki obu. W pipelinie YOLOv9 partnerem jest cała próbka bez mozaiki, współczynnik mieszania jest losowany jednostajnie z przedziału od 0 do 1, a `mixup_scale` nigdy nie jest odczytywany, choć specyfikacja biblioteki oznacza go jako zależny od mozaiki. We wrapperze YOLOX, używanym przez YOLOX, RTMDet, PicoDet i FOMO, oraz we wrapperze YOLO-NAS partner jest najpierw przeskalowywany o współczynnik losowany z `mixup_scale`, a oba obrazy są uśredniane ze stałą wagą 0.5. Bazowa wartość domyślna `mixup_prob` to 1.0, a YOLOv9 ma domyślnie 0.

<aug-figure name="mixup" />

Dwa ostatnie panele mieszają drugą próbkę z mozaiką przy skali 0.5×. Ich podpis podaje współczynnik mieszania λ, czyli wagę próbki mozaikowej.

### no_aug_epochs

`no_aug_epochs` jest `used` wszędzie, lecz nie wszędzie oznacza to samo. W pipeline'ach z mozaiką wyłącza mozaikę i MixUp na ostatnie epoki. W pipeline'ach w stylu DETR zatrzymuje augmentacje fotometryczne, pomniejszanie i wycinanie oraz kształtuje końcówkę harmonogramu. W pipeline'ach klasyfikacji wyłącza automatyczną augmentację, wymazywanie, MixUp i CutMix, a wycinanie i odbicia pozostają.

## Receptury w stylu DETR

D-FINE, Dome-DETR, DEIM, DEIMv2, RT-DETRv4 i TinyFormer trenują przez jedną transformację torchvision bez mozaiki i bez transformacji afinicznej, podobnie jak EdgeCrafter (EC) w detekcji. Jej trzy silne operacje są stałymi receptury, a nie parametrami:

- `RandomPhotometricDistort` stosuje każdą zmianę koloru (jasność od 0.875 do 1.125, kontrast i nasycenie od 0.5 do 1.5, odcień ±0.05 oraz permutację kanałów) z prawdopodobieństwem 0.5.
- `RandomZoomOut` z prawdopodobieństwem 0.5 umieszcza obraz na czarnym płótnie o boku do czterech razy dłuższym niż bok obrazu, więc obiekty mogą zmaleć do jednej czwartej swojego rozmiaru.
- `RandomIoUCrop`, stosowany z prawdopodobieństwem 0.8, wycina obszar, którego boki mają od 0.3 do 1.0 długości odpowiednich boków obrazu, wybrany tak, aby zachować przynajmniej część ramek.

Po tych krokach odrzucane są ramki, których którykolwiek bok ma 1 px lub mniej. Wszystkie trzy operacje zatrzymują się na epoce zatrzymania rodziny, najpóźniej z początkiem ostatnich `no_aug_epochs` epok. Jedyne parametry, które odczytuje ten pipeline, to `flip_prob` i `no_aug_epochs`.

<aug-figure name="detr_ops" />

Trzy próbki z początku trenowania modelu D-FINE, ze wszystkimi trzema aktywnymi operacjami:

<aug-figure name="detr_samples" />

RF-DETR ma własny pipeline: odbicie poziome, następnie z prawdopodobieństwem `crop_resize_prob` zmiana rozmiaru krótszego boku do 400, 500 lub 600 px i losowy kwadratowy wycinek od 384 do 600 px, a na końcu bezpośrednia zmiana rozmiaru do kwadratowej rozdzielczości trenowania, która rozciąga obraz zamiast dopasowywać go metodą letterbox. `crop_resize_prob` ma domyślnie wartość 0.5 i ustawia się go z poziomu Pythona.

<aug-figure name="rfdetr_crop" />

## Małe obiekty

`erasing` nie dotyczy detekcji. Random Erasing należy do pakietu klasyfikacji i żaden pipeline detekcji nie ma kroku wymazywania, więc trenowanie detektora przebiega tak samo przy dowolnej wartości tego parametru i nie ma czego wyłączać. Oto kroki, które zmniejszają lub usuwają małe obiekty:

- `mosaic_scale`. W pipeline'ach z mozaiką rozmiar obiektu jest mnożony przez wylosowaną skalę, więc dolna granica wyznacza, jak mały może się on stać. Przy bazowej wartości domyślnej `(0.1, 2.0)` obiekt o rozmiarze 20 px może zmaleć do 2 px. Pipeline YOLOv9 odrzuca następnie ramki, których którykolwiek bok ma 2 px lub mniej, a jego końcowa transformacja, podobnie jak w rodzinie YOLOX, odrzuca ramki, których którykolwiek bok ma 1 px lub mniej. Piksele obiektu pozostają na obrazie bez etykiety. Dolna granica 0.5, jak w `(0.5, 1.5)` modelu YOLOv9, utrzymuje każdy obiekt na poziomie co najmniej połowy jego rozmiaru podczas inferencji.
- `degrees`, `translate`, `shear` i `perspective` przesuwają obiekty ku krawędzi, gdzie ich ramki są przycinane i przechodzą przez ten sam filtr rozmiaru.
- W recepturach w stylu DETR `RandomZoomOut` może zmniejszyć obiekty do jednej czwartej ich rozmiaru, a `RandomIoUCrop` może je odciąć. Żadna z tych operacji nie jest parametrem i obie zatrzymują się na epoce zatrzymania.
- Wycinanie w RF-DETR może przecinać obiekty na krawędzi wycinka. `crop_resize_prob=0` je wyłącza.
- Rozdzielczość trenowania skaluje każdy obiekt: letterbox zmienia rozmiar obrazu o współczynnik równy `imgsz` podzielonemu przez jego dłuższy bok, więc przy `imgsz=1280` każdy obiekt ma dwa razy większą szerokość i wysokość niż przy 640. [YOLOv9](/docs/models/yolov9) ma też wariant P2 z głowicą o stride 4 dla małych obiektów.

<code-tabs name="small" />

## Pakiet klasyfikacji

Sześć parametrów działa tylko na pipeline klasyfikacji. Rodziny detekcji ignorują wszystkie sześć.

<code-tabs name="classify" />

`scale` to zakres powierzchni losowego wycinka. `scale=0.5` oznacza `(0.5, 1.0)`, a jawna para ustawia obie granice. Mała dolna granica może całkowicie odciąć obiekt, jak na trzecim panelu.

<aug-figure name="cls_crop" />

`auto_augment` przyjmuje `"randaugment"`, `"autoaugment"`, `"augmix"` lub `None` i stosuje do każdej próbki jedno losowanie z tej polityki.

<aug-figure name="auto_augment" />

`erasing` to prawdopodobieństwo Random Erasing, w przedziale `[0, 1)`. Działa po normalizacji i wypełnia zerami prostokąt pokrywający od 2 do 33 procent obrazu, co po normalizacji odpowiada średniemu kolorowi ImageNet.

<aug-figure name="erasing" />

`mixup` i `cutmix` to prawdopodobieństwa na poziomie batcha, które tworzą miękkie etykiety. W jednym batchu działa najwyżej jeden z nich, najpierw MixUp, więc oba się sumują, a ich suma musi wynosić najwyżej 1. Na panelach A i B to dwa wycinki tego samego zdjęcia, a każdy podpis podaje miękką etykietę, na której trenowany jest zmieszany obraz.

<aug-figure name="batch_mix" />

`auto_augment`, `erasing`, `mixup` i `cutmix` są domyślnie wyłączone, więc trenowanie klasyfikacji pozostaje niezmienione, dopóki nie włączy się ich jawnie. `train()` i CLI kierują `mixup` według zadania: klasyfikacja używa mieszania batchy, a detekcja `mixup_prob`.

`crop_pct=None` zachowuje używany przez rodzinę współczynnik zmiany rozmiaru podczas ewaluacji. Nadpisanie zmienia wycinek ewaluacyjny w przebiegu walidacji wykonywanym podczas trenowania, a eksport zachowuje natywne przetwarzanie wstępne rodziny.

CLI udostępnia `auto_augment`, `erasing`, `cutmix`, `scale`, `crop_pct`, `fliplr` i `flipud`. `mixup` w klasyfikacji oznacza mieszanie batchy i ma domyślną wartość 0.0. Odbicia poziome mają domyślnie wartość 0.5, a pionowe 0.0.

`no_aug_epochs` wyłącza automatyczną augmentację, wymazywanie, MixUp i CutMix w końcowych epokach, zachowując wycinanie i odbicia. Receptury klasyfikacji domyślnie ustawiają tę końcówkę na 0.

## Parametry specyficzne dla rodzin

Niektóre parametry znajdują się w podklasie konfiguracji rodziny, a nie w klasie bazowej, więc istnieją tylko dla tej rodziny i nie mają flagi CLI.

| Rodzina | Parametr | Działanie |
|---|---|---|
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste` | Prawdopodobieństwo augmentacji instancji metodą copy-paste, tylko dla `task="segment"` |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste_mode` | `"flip"` ponownie używa tej samej próbki w odbiciu lustrzanym, `"mixup"` pobiera drugą próbkę |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `rot90` | Prawdopodobieństwo losowego obrotu o 90 stopni |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `max_labels` | Limit danych referencyjnych (ground truth) na obraz w transformacjach trenowania, domyślnie 300 |
| RF-DETR | `copy_paste`, `copy_paste_mode` | Copy-paste dla `task="segment"`, tylko tryb `"flip"` |
| RF-DETR | `crop_resize_prob` | Prawdopodobieństwo losowego wycięcia ze zmianą rozmiaru w natywnym pipelinie, domyślnie 0.5 |
| D-FINE, EdgeCrafter | `crop_resize_prob` | Prawdopodobieństwo losowego wycięcia ze zmianą rozmiaru, tylko dla `task="segment"` |
| EdgeCrafter, YOLO-NAS | `brightness_contrast_prob`, `affine_prob` | Prawdopodobieństwa zaburzenia w ścieżce pozy oraz transformacji afinicznej uwzględniającej punkty kluczowe |

`max_labels` jest tym parametrem, który powoduje utratę danych. Ramki powyżej limitu są odrzucane z jednorazowym ostrzeżeniem zamiast błędu, dlatego dla gęstych obrazów, takich jak fotografia lotnicza, trzeba go zwiększyć. Pamięć przypisywania etykiet zależy od najgęstszego obrazu w batchu, a nie od limitu.

Mozaika i MixUp są wyłączone podczas trenowania z obróconymi ramkami niezależnie od parametrów, ponieważ augmentacja uwzględniająca narożniki obróconych ramek nie jest zaimplementowana.

## Powiązane strony

- [Macierz augmentacji](/docs/reference/augmentation-matrix) podaje stan każdego parametru w każdej rodzinie, zakresy i wartości domyślne każdej rodziny.
- [Hiperparametry](/docs/train/hyperparameters) opisują `no_aug_epochs` jako argument harmonogramu oraz pozostałe argumenty `train()`.
- [Zbiory danych](/docs/train/datasets) opisują formaty etykiet używane przez te transformacje.
