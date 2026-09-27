---
title: Macierz augmentacji
seo_title: Która rodzina LibreYOLO obsługuje którą augmentację
description: >-
  Obsługa parametrów augmentacji w poszczególnych rodzinach: szesnaście
  parametrów TrainConfig, trzy statusy, sześć archetypów pipeline'u i parametry
  po cichu ignorowane przez rodzinę.
lead: >-
  Ustawienie parametru augmentacji nie gwarantuje, że trafi on do pipeline'u. Na
  tej stronie opisano, jak każda rodzina obsługująca trenowanie traktuje
  poszczególne parametry TrainConfig, na podstawie deklaratywnej tabeli
  dostarczanej z biblioteką jako jedynego źródła prawdy.
keywords:
  - augmentacja LibreYOLO
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - no_aug_epochs
  - macierz obsługi augmentacji
  - parametry TrainConfig
last_verified: 1.5.0
verification: >-
  Listę parametrów, statusy, archetypy, odchylenia poszczególnych rodzin i
  funkcje pomocnicze odczytano z libreyolo/data/augment/spec.py w wersji 1.5.0.
  Powiązanie tej tabeli z rzeczywistymi pipeline'ami jest sprawdzane przez
  tests/unit/test_augment_spec.py.
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
source_hash: d2e1b9f5c81072e1
---

## Parametry

Są to nazwy pól `TrainConfig`, a nie ich zapis w CLI. CLI mapuje własne aliasy
na te pola, dlatego `--mosaic` ustawia `mosaic_prob`.

| Parametr | Znaczenie |
|---|---|
| `mosaic_prob` | Prawdopodobieństwo utworzenia próbki mosaic z 4 obrazów |
| `mixup_prob` | Prawdopodobieństwo zmieszania z drugą próbką |
| `hsv_prob` | Prawdopodobieństwo losowej zmiany kolorów HSV |
| `flip_prob` | Prawdopodobieństwo odbicia poziomego |
| `degrees` | Zakres losowego obrotu dla transformacji afinicznej, w stopniach |
| `translate` | Ułamek losowego przesunięcia dla transformacji afinicznej |
| `mosaic_scale` | Zakres losowej zmiany skali dla transformacji afinicznej |
| `mixup_scale` | Zakres losowej zmiany skali stosowany do obrazu partnerskiego MixUp |
| `shear` | Zakres losowego ścinania dla transformacji afinicznej, w stopniach |
| `perspective` | Wielkość transformacji projekcyjnej dla transformacji afinicznej |
| `flipud` | Prawdopodobieństwo odbicia pionowego |
| `no_aug_epochs` | Końcowe epoki trenowane z wyłączoną silną augmentacją |
| `auto_augment` | Polityka AutoAugment dla klasyfikacji: randaugment, autoaugment lub augmix |
| `erasing` | Prawdopodobieństwo RandomErasing dla klasyfikacji |
| `mixup` | Prawdopodobieństwo batch-MixUp dla klasyfikacji, z miękkimi etykietami |
| `cutmix` | Prawdopodobieństwo batch-CutMix dla klasyfikacji, z miękkimi etykietami |

Ostatnie cztery parametry tworzą pakiet klasyfikacyjny. Rodziny detekcyjne je
ignorują. `mixup` jest parametrem dostępnym tylko przez API: `--mixup` w CLI to
alias detekcyjnego parametru `mixup_prob`.

<code-tabs name="usage" />

## Trzy statusy

| Status | Znaczenie |
|---|---|
| `used` | Parametr trafia do pipeline'u trenowania rodziny i zmienia próbki |
| `gated_by_mosaic` | Parametr dotyczy tylko próbek, które trafiły do gałęzi mosaic, więc przy `mosaic_prob == 0` nigdy nie jest stosowany |
| `ignored` | Parametr nigdy nie trafia do pipeline'u, a jego ustawienie niczego nie zmienia |

Przed uruchomieniem warto sprawdzić zwłaszcza status `ignored`, ponieważ nie
dochodzi do żadnego błędu. CLI ostrzega, gdy jawnie ustawiony parametr trenowania
jest ignorowany przez wybraną rodzinę, a trener ostrzega, gdy `mixup_prob > 0`
nie może zadziałać, ponieważ rodzina uzależnia MixUp od mosaic, a `mosaic_prob`
wynosi zero.

## Archetypy pipeline'ów

Każda ujęta rodzina korzysta z jednego z sześciu pipeline'ów, z kilkoma
odchyleniami dla poszczególnych rodzin opisanymi poniżej.

| Parametr | Styl YOLOX | YOLO-NAS | Styl DETR | Klasyfikacja | Semantyczny | Przywracanie |
|---|---|---|---|---|---|---|
| `mosaic_prob` | używany | ignorowany | ignorowany | ignorowany | ignorowany | ignorowany |
| `mixup_prob` | zależny | używany | ignorowany | ignorowany | ignorowany | ignorowany |
| `hsv_prob` | używany | używany | ignorowany | ignorowany | ignorowany | ignorowany |
| `flip_prob` | używany | używany | używany | ignorowany | ignorowany | ignorowany |
| `degrees` | zależny | używany | ignorowany | ignorowany | ignorowany | ignorowany |
| `translate` | zależny | używany | ignorowany | ignorowany | ignorowany | ignorowany |
| `mosaic_scale` | zależny | używany | ignorowany | ignorowany | ignorowany | ignorowany |
| `mixup_scale` | zależny | używany | ignorowany | ignorowany | ignorowany | ignorowany |
| `shear` | zależny | używany | ignorowany | ignorowany | ignorowany | ignorowany |
| `perspective` | zależny | używany | ignorowany | ignorowany | ignorowany | ignorowany |
| `flipud` | używany | używany | ignorowany | ignorowany | ignorowany | ignorowany |
| `no_aug_epochs` | używany | używany | używany | używany | używany | używany |
| `auto_augment` | ignorowany | ignorowany | ignorowany | używany | ignorowany | ignorowany |
| `erasing` | ignorowany | ignorowany | ignorowany | używany | ignorowany | ignorowany |
| `mixup` | ignorowany | ignorowany | ignorowany | używany | ignorowany | ignorowany |
| `cutmix` | ignorowany | ignorowany | ignorowany | używany | ignorowany | ignorowany |

W pipelinie w stylu YOLOX przetwarzanie wstępne każdej próbki stosuje losową
zmianę HSV i odbicia, natomiast transformacja afiniczna oraz MixUp są wykonywane
tylko w gałęzi mosaic. YOLO-NAS zamiast tego zawsze wykonuje transformację
afiniczną dla każdej próbki, ignoruje mosaic i niezależnie stosuje MixUp,
używając ponownie `mosaic_scale` jako zakresu skali transformacji afinicznej.

Pipeline w stylu DETR jest transformacją przepuszczającą bez mosaic. Jego
zniekształcenie fotometryczne, oddalenie i przycięcie IoU są stałymi receptury,
a nie konfigurowalnymi parametrami, dlatego `hsv_prob` ani parametry geometrii
do niego nie trafiają. Pipeline klasyfikacji używa transformacji ImageFolder,
której odbicie poziome ma stałe prawdopodobieństwo 0.5 zamiast `flip_prob`.
Losowa zmiana skali semantycznej i HSV pochodzą z atrybutów klas rodzin, a nie
z parametrów konfiguracji, natomiast odbicia przy przywracaniu są sprzężonymi
operacjami na wejściu i celu o stałym prawdopodobieństwie 0.5.

`no_aug_epochs` jest uwzględniany wszędzie, choć wyłącza różne elementy: mosaic
i MixUp w stylu YOLOX, transformację afiniczną i MixUp w YOLO-NAS, silne
augmentacje fotometryczne i przycinanie wraz z końcową częścią współczynnika
uczenia w stylu DETR oraz końcową część harmonogramu w pozostałych przypadkach.

## Rodziny według archetypu

| Archetyp | Rodziny |
|---|---|
| Styl YOLOX | `yolox`, `yolo7`, `yolo9`, `yolo9_e2e`, `yolo9_p2`, `rtmdet`, `picodet`, `rtdetr`, `rtdetrv2`, `fomo` |
| YOLO-NAS | `yolonas` |
| Styl DETR | `dfine`, `domedetr`, `deim`, `deimv2`, `rtdetrv4`, `rfdetr`, `ec`, `dinov2` |
| Klasyfikacja | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` |
| Semantyczny | `segformer` |
| Przywracanie | `nafnet` |

Uwzględniono dwadzieścia pięć rodzin. Rodzina spoza tej listy zwraca pusty
zbiór ignorowanych parametrów, więc nie jest dla niej emitowane ostrzeżenie.

## Odchylenia

| Rodzina | Różnica względem archetypu |
|---|---|
| `rtmdet` | `flipud` ignorowany: jego transformacja nie obejmuje odbicia pionowego |
| `picodet` | `flipud` ignorowany |
| `rtdetr` | `flipud` ignorowany |
| `rtdetrv2` | `flipud` ignorowany |
| `fomo` | `perspective` i `flipud` ignorowane |
| `ec` | `hsv_prob`, `degrees` i `translate` używane tylko dla `task="pose"`; detect i segment korzystają ze stałych receptur fotometrycznych |
| `dinov2` | Pakiet klasyfikacyjny używany tylko dla `task="classify"` |

`ec` i `dinov2` są rodzinami wielozadaniowymi, więc parametr jest oznaczany
jako ignorowany tylko wtedy, gdy ignorują go wszystkie zadania rodziny
obsługujące trenowanie. Dzięki temu ostrzeżenie CLI nie jest błędne dla jednego
zadania, a poprawne dla innego.

Dome-DETR dziedziczy transformacje D-FINE bez zmian. Jedyną niedostępną
funkcją jest trenowanie wieloskalowe, które wyłącza jego konfiguracja, a nie
specyfikacja augmentacji.

## Parametry właściwe dla rodzin

Niektóre rodziny mają parametry augmentacji we własnej podklasie `TrainConfig`,
a nie w klasie bazowej. CLI ich nie udostępnia. Należy ustawiać je przez API
Pythona.

| Rodzina | Parametr | Znaczenie |
|---|---|---|
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste` | Prawdopodobieństwo augmentacji instancji metodą kopiowania i wklejania, tylko dla `task="segment"` |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste_mode` | Źródło kopiowania i wklejania: `flip` odzwierciedla tę samą próbkę, `mixup` używa drugiej próbki |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `rot90` | Prawdopodobieństwo losowego obrotu o 90 stopni |
| `rfdetr` | `copy_paste` | Prawdopodobieństwo kopiowania i wklejania dla `task="segment"`, tylko tryb `flip` |
| `rfdetr` | `copy_paste_mode` | Tryb źródła kopiowania i wklejania dla `task="segment"` |
| `rfdetr` | `crop_resize_prob` | Prawdopodobieństwo losowego przycięcia ze zmianą rozmiaru w natywnym pipelinie |
| `dfine` | `crop_resize_prob` | Prawdopodobieństwo losowego przycięcia ze zmianą rozmiaru, `task="segment"` |
| `ec` | `crop_resize_prob` | Prawdopodobieństwo losowego przycięcia ze zmianą rozmiaru, `task="segment"` |
| `ec`, `yolonas` | `brightness_contrast_prob` | Prawdopodobieństwo losowej zmiany jasności i kontrastu, `task="pose"` |
| `ec`, `yolonas` | `affine_prob` | Prawdopodobieństwo transformacji afinicznej uwzględniającej punkty kluczowe, `task="pose"` |

`rot90` dotyczy detect i OBB w `yolo9`.

## Odpytywanie specyfikacji

| Funkcja pomocnicza | Zwraca |
|---|---|
| `aug_support(family)` | Tabelę mapującą parametry na `Support` albo `None` dla nieznanej rodziny |
| `ignored_aug_params(family)` | Zbiór nazw parametrów ignorowanych przez rodzinę; pusty dla nieznanej rodziny |
| `uses_mosaic_gating(family)` | Informację, czy MixUp rodziny działa tylko na próbkach mosaic |
| `display_name(family)` | Czytelną dla człowieka nazwę rodziny używaną w ostrzeżeniach |
| `mixup_gating_warning(family, mosaic_prob, mixup_prob)` | Tekst ostrzeżenia, gdy MixUp nigdy nie może zadziałać, w przeciwnym razie `None` |

`Support` to nazwana krotka zawierająca `status` i `note`, gdzie notatka wyjaśnia,
dlaczego parametr jest ignorowany lub zależny w danej rodzinie.

## Bramka mosaic

Dla rodziny w stylu YOLOX ustawienie `mixup_prob=0.5` przy `mosaic_prob=0`
całkowicie wyłącza MixUp, ponieważ MixUp jest stosowany tylko do próbek mosaic.
Do takiego połączenia łatwo doprowadzić, wyłączając mosaic pod koniec trenowania.
Trener zapisuje ostrzeżenie z nazwą rodziny, a za jego działaniem stoi czysta
funkcja `mixup_gating_warning`.
