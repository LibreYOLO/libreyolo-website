---
title: Macierz augmentacji
seo_title: Która rodzina LibreYOLO obsługuje którą augmentację
description: >-
  Każdy parametr augmentacji z zakresem i zapisem w CLI, stan każdego parametru
  w każdej z 31 rodzin obsługujących trenowanie oraz wartości domyślne, z
  którymi dostarczana jest każda rodzina.
lead: >-
  Ustawienie parametru augmentacji nie gwarantuje, że trafi on do pipeline'u. Ta
  strona opisuje, jak każda rodzina obsługująca trenowanie traktuje każdy
  parametr TrainConfig i od jakiej wartości zaczyna każda rodzina, na podstawie
  danych wygenerowanych z biblioteki w opisywanym wydaniu.
keywords:
  - augmentacja libreyolo
  - parametry augmentacji
  - domyślne wartości augmentacji
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - mosaic_scale
  - no_aug_epochs
  - macierz obsługi augmentacji
  - parametry TrainConfig
last_verified: 1.8.0
verification: >-
  Wygenerowano skryptem scripts/build-augmentation/collect.py z biblioteki
  LibreYOLO 1.8.0: listę parametrów, stany i uwagi z pliku
  libreyolo/data/augment/spec.py, który testy biblioteki wiążą z rzeczywistymi
  pipeline'ami, wartości domyślne z podklasy TrainConfig każdej rodziny oraz
  zapisy w CLI z polecenia libreyolo train. Zakresy oraz jedyne miejsce, w
  którym specyfikacja i pipeline się różnią, ten sam skrypt sprawdza względem
  biblioteki.
snippets:
  usage:
    - label: Bezpośrednie odpytywanie specyfikacji
      language: python
      code: |
        from libreyolo.data.augment.spec import (
            AUG_KNOBS,
            aug_support,
            ignored_aug_params,
            uses_mosaic_gating,
        )

        print(sorted(AUG_KNOBS))

        table = aug_support("yolo9")
        print(table["mixup_prob"].status, table["mixup_prob"].note)

        print(sorted(ignored_aug_params("dfine")))
        print(uses_mosaic_gating("yolo9"), uses_mosaic_gating("yolonas"))
    - label: Wartości domyślne rodziny
      language: python
      code: |
        from libreyolo.training.config import YOLO9Config

        config = YOLO9Config()
        print(config.mosaic_scale, config.degrees, config.mixup_prob)
source_hash: a88efd1537732e6c
---

## Parametry

To nazwy pól `TrainConfig`. `train()` i CLI akceptują też krótkie zapisy z kolumny CLI, więc `mosaic` ustawia `mosaic_prob`. W CLI `mixup` jest kierowany według zadania: do mieszania batchy klasyfikacji w przypadku klasyfikatorów, do `mixup_prob` w przypadku detektorów. `perspective` nie ma flagi CLI i ustawia się go z poziomu Pythona.

<aug-knobs />

Bazowa wartość domyślna to wartość ustawiona w samym `TrainConfig`. Większość rodzin nadpisuje niektóre z nich, a sekcja [Wartości domyślne według rodziny](#defaults-by-family) podaje wartości, od których faktycznie zaczyna się trenowanie. Ilustracje działania każdego parametru zawiera strona [Augmentacje](/docs/train/augmentations).

Zakresy w nawiasach to przedziały, więc `[0, 1)` obejmuje 0 i wyklucza 1. Parametry klasyfikacji są sprawdzane podczas budowania ich pipeline'u i poza swoim zakresem zgłaszają `ValueError`. Parametry detekcji nie są sprawdzane przez `TrainConfig`, więc trenowanie z wartością spoza zakresu przebiega bez żadnego komunikatu. Wyjątkami są wylosowana skala równa 0 lub mniejsza oraz ujemne `perspective`, które zgłaszają `ValueError` przy transformacji próbki.

<code-tabs name="usage" />

## Trzy statusy

| Status | Znaczenie |
|---|---|
| `used` | Parametr trafia do pipeline'u trenowania rodziny i zmienia próbki |
| `gated_by_mosaic` | Parametr dotyczy tylko próbek, które przeszły przez gałąź mozaiki, więc przy `mosaic_prob == 0` nigdy nie działa |
| `ignored` | Parametr nigdy nie trafia do pipeline'u; jego ustawienie niczego nie zmienia |

Status `ignored` warto sprawdzić przed uruchomieniem trenowania, ponieważ nic nie zgłasza błędu. CLI ostrzega, gdy jawnie ustawiony parametr trenowania należy do tych, które wybrana rodzina ignoruje, a trener ostrzega, gdy `mixup_prob > 0` nie może zadziałać, ponieważ rodzina uzależnia MixUp od mozaiki, a `mosaic_prob` wynosi zero.

## Która rodzina uwzględnia który parametr

<aug-matrix />

Każda ujęta rodzina korzysta z jednego z sześciu pipeline'ów. W pipelinie w stylu YOLOX przetwarzanie wstępne każdej próbki stosuje zaburzenie HSV i odbicia, natomiast transformacja afiniczna i MixUp działają tylko w gałęzi mozaiki. YOLO-NAS zamiast tego wykonuje zawsze aktywną transformację afiniczną dla każdej próbki, ignoruje mozaikę i stosuje MixUp niezależnie, używając ponownie `mosaic_scale` jako zakresu skali transformacji afinicznej.

Pipeline w stylu DETR to transformacja przepuszczająca, bez mozaiki. Jego zniekształcenie fotometryczne, pomniejszenie i wycinek IoU są stałymi receptury, a nie konfigurowalnymi parametrami, dlatego `hsv_prob` i parametry geometryczne nigdy do niego nie docierają. Klasyfikacja używa `flip_prob` do odbić poziomych i `flipud` do odbić pionowych. Losowa zmiana skali i HSV w segmentacji semantycznej pochodzą z atrybutów klas rodzin, a nie z parametrów konfiguracji, a odbicia w restauracji obrazu są sprzężonymi operacjami na wejściu i celu o stałym prawdopodobieństwie 0.5.

`no_aug_epochs` jest uwzględniany wszędzie, choć to, co wyłącza, się różni: mozaikę i MixUp w stylu YOLOX, transformację afiniczną i MixUp w YOLO-NAS, silne augmentacje fotometryczne i wycinanie oraz końcówkę harmonogramu współczynnika uczenia w stylu DETR, a także automatyczną augmentację, wymazywanie, MixUp i CutMix w klasyfikacji. Wycinanie i odbicia w klasyfikacji pozostają włączone.

Tam, gdzie rodzina odbiega od reszty swojego pipeline'u, przyczyna jest jedną z poniższych:

| Rodzina | Różnica |
|---|---|
| `rtmdet`, `picodet`, `rtdetr`, `rtdetrv2` | `flipud` ignorowany: ich transformacje nie mają odbicia pionowego |
| `fomo` | `perspective` i `flipud` ignorowane |
| `ppyoloe` | `flipud` ignorowany: receptura źródłowa używa zamiast tego losowego obrotu o 90 stopni |
| `yolo9`, `yolo9_e2e`, `yolo9_p2`, `yolo7`, `rtdetr`, `rtdetrv2` | `mixup_scale` ignorowany: ich MixUp miesza całe obrazy w losowej proporcji bez przeskalowania żadnego z nich |
| `gtr` | `mosaic_prob` i `mixup_prob` używane przez pierwsze `mosaic_epochs` epok; `degrees`, `translate` i `mosaic_scale` kształtują transformację afiniczną następującą po mozaice |
| `rfdetr` | `hsv_prob` ignorowany: natywny pipeline nie ma zaburzenia HSV |
| `ec` | `hsv_prob`, `degrees` i `translate` używane tylko dla `task="pose"`; detect i segment używają stałych receptur fotometrycznych |
| `dinov2` | Pakiet klasyfikacji jest używany tylko dla `task="classify"` |

`ec` i `dinov2` to rodziny wielozadaniowe, więc parametr jest oznaczany jako ignorowany tylko wtedy, gdy ignorują go wszystkie zadania tej rodziny obsługujące trenowanie. Dzięki temu ostrzeżenie CLI nigdy nie jest błędne dla jednego zadania, a jednocześnie poprawne dla innego.

Dome-DETR dziedziczy transformacje D-FINE bez zmian. Jedyną rzeczą, której nie obsługuje, jest trenowanie wieloskalowe, które wyłącza jego konfiguracja, a nie specyfikacja augmentacji.

Uwzględniono trzydzieści jeden rodzin. Dla rodziny spoza tej listy `ignored_aug_params` zwraca pusty zbiór, więc nie jest dla niej emitowane żadne ostrzeżenie.

## Wartości domyślne według rodziny

Wartość, od której zaczyna konfiguracja każdej rodziny detekcji. Pusta komórka oznacza, że rodzina ignoruje ten parametr, więc jego wartość domyślna niczego nie zmienia.

<aug-defaults />

Dwa wzorce są ważniejsze niż jakakolwiek pojedyncza wartość. RT-DETR, RT-DETRv2, PicoDet i FOMO zaczynają od `mosaic_prob=0`, więc w tych rodzinach parametry oznaczone w macierzy jako zależne od mozaiki nic nie robią, dopóki mozaika nie zostanie włączona. YOLOv9 zaczyna z wyłączonym obrotem, ścinaniem i MixUp oraz z `mosaic_scale=(0.5, 1.5)`, podczas gdy bazowa wartość domyślna to `(0.1, 2.0)`.

Rodzina EdgeCrafter odczytuje `hsv_prob`, `degrees` i `translate` tylko dla `task="pose"`, dlatego jej wiersz pokazuje dla tych trzech parametrów wartości domyślne dla pozy.

Rodziny klasyfikacji zaczynają od bazowych wartości domyślnych: `flip_prob=0.5`, `flipud=0.0`, `scale=(0.5, 1.0)`, własnego `crop_pct` rodziny oraz wyłączonych `auto_augment`, `erasing`, `mixup` i `cutmix`.

## Parametry specyficzne dla rodzin

Niektóre rodziny mają parametry augmentacji we własnej podklasie `TrainConfig`, a nie w klasie bazowej. CLI ich nie udostępnia; należy je ustawiać przez API Pythona.

| Rodzina | Parametr | Znaczenie |
|---|---|---|
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste` | Prawdopodobieństwo augmentacji instancji metodą copy-paste, tylko dla `task="segment"` |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste_mode` | Źródło copy-paste: `flip` używa lustrzanego odbicia tej samej próbki, `mixup` używa drugiej próbki |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `rot90` | Prawdopodobieństwo losowego obrotu o 90 stopni |
| `rfdetr` | `copy_paste` | Prawdopodobieństwo copy-paste dla `task="segment"`, tylko tryb `flip` |
| `rfdetr` | `copy_paste_mode` | Tryb źródła copy-paste dla `task="segment"` |
| `rfdetr` | `crop_resize_prob` | Prawdopodobieństwo losowego wycięcia ze zmianą rozmiaru w natywnym pipelinie |
| `dfine` | `crop_resize_prob` | Prawdopodobieństwo losowego wycięcia ze zmianą rozmiaru, `task="segment"` |
| `ec` | `crop_resize_prob` | Prawdopodobieństwo losowego wycięcia ze zmianą rozmiaru, `task="segment"` |
| `ec`, `yolonas` | `brightness_contrast_prob` | Prawdopodobieństwo losowej zmiany jasności i kontrastu, `task="pose"` |
| `ec`, `yolonas` | `affine_prob` | Prawdopodobieństwo transformacji afinicznej uwzględniającej punkty kluczowe, `task="pose"` |

`rot90` dotyczy detect i OBB w `yolo9`.

## Odpytywanie specyfikacji

| Funkcja pomocnicza | Zwraca |
|---|---|
| `aug_support(family)` | Tabelę przypisującą parametrom obiekty `Support` albo `None` dla nieznanej rodziny |
| `ignored_aug_params(family)` | Zbiór nazw parametrów ignorowanych przez rodzinę; pusty dla nieznanej rodziny |
| `uses_mosaic_gating(family)` | Informację, czy MixUp rodziny działa tylko na próbkach mozaikowych |
| `display_name(family)` | Czytelną dla człowieka nazwę rodziny używaną w ostrzeżeniach |
| `mixup_gating_warning(family, mosaic_prob, mixup_prob)` | Tekst ostrzeżenia, gdy MixUp nigdy nie może zadziałać, w przeciwnym razie `None` |

`Support` to nazwana krotka złożona z `status` i `note`, gdzie notatka wyjaśnia, dlaczego parametr jest w danej rodzinie ignorowany lub zależny od mozaiki. W wersji 1.8.0 specyfikacja nadal oznacza `mixup_scale` jako zależny od mozaiki dla rodzin trenujących przez wrapper mozaiki YOLOv9, więc funkcja `aug_support` zgłasza go tam jako zależny od mozaiki, choć powyższa macierz, sprawdzona względem pipeline'u, oznacza go jako ignorowany.

## Uzależnienie od mozaiki

Dla rodziny w stylu YOLOX ustawienie `mixup_prob=0.5` przy `mosaic_prob=0` całkowicie wyłącza MixUp, ponieważ MixUp jest stosowany tylko do próbek mozaikowych. Do takiej kombinacji łatwo dojść, wyłączając mozaikę pod koniec trenowania. Trener zapisuje w logu ostrzeżenie z nazwą rodziny, a `mixup_gating_warning` to czysta funkcja, która za nim stoi.
