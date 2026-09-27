---
title: Augmentacje
seo_title: Augmentacje podczas trenowania w LibreYOLO
description: >-
  Parametry augmentacji w TrainConfig, cztery stojące za nimi warianty
  pipeline'u oraz tabela dla każdej rodziny wskazująca, które parametry są
  używane, warunkowe lub ignorowane.
lead: >-
  Augmentację konfiguruje się parametrami w TrainConfig, ale każda rodzina
  modeli uruchamia własny pipeline trenowania. Pipeline bez gałęzi mozaiki
  ignoruje mosaic_prob, zamiast próbować go przybliżać.
keywords:
  - augmentacja danych yolo
  - augmentacja mosaic
  - mixup
  - hsv jitter
  - random affine
  - augmentacja copy paste
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: 1.5.0
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
      code: >
        # W CLI parametr mosaic_prob zapisuje się jako mosaic, a mixup_prob jako
        mixup.

        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 mosaic=1.0 mixup=0.15 hsv_prob=1.0 \
          flip_prob=0.5 no_aug_epochs=15
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
source_hash: 47461cd13aab580c
---

## Ustawianie parametrów

Parametry augmentacji są zwykłymi argumentami `train()`.

<code-tabs name="train" />

Dwa z nich mają krótszą pisownię w CLI: `mosaic` odpowiada `mosaic_prob`, a
`mixup` odpowiada `mixup_prob`. Każdy inny parametr zapisuje się identycznie w
obu miejscach.

## Trzy stany, nie dwa

To, czy parametr ma wpływ, zależy od rodziny. Biblioteka przechowuje deklaratywną
tabelę tych informacji, a każdy wpis ma jeden z trzech stanów.

`used` oznacza, że parametr dociera do pipeline'u i zmienia próbki. `ignored`
oznacza, że nigdy nie dociera do pipeline'u, więc jego ustawienie niczego nie
zmienia. `gated_by_mosaic` oznacza, że parametr dotyczy tylko próbek, które
trafiły do gałęzi mozaiki. Przy `mosaic_prob=0` nigdy się więc nie uruchamia,
mimo że jest podłączony.

Ten trzeci stan często zaskakuje. W pipelinie w stylu YOLOX transformacja
afiniczna działa na obszarze mozaiki, a MixUp łączy próbkę mozaikową. Wartość
`mosaic_prob=0` po cichu wyłącza więc jednocześnie `degrees`, `translate`,
`shear`, `perspective`, `mosaic_scale`, `mixup_prob` i `mixup_scale`. Trener
zapisuje ostrzeżenie specjalnie dla przypadku MixUp:

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

CLI ostrzega również o ignorowanych parametrach, wymieniając tylko te, które
faktycznie podano:

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

## Cztery warianty pipeline'u

Rodziny grupują się w cztery pipeline'y trenowania, a wybrany pipeline określa
prawie wszystkie odpowiedzi.

Pipeline mozaikowy w stylu YOLOX stosuje zaburzenie HSV i odbicia dla każdej
próbki, a następnie uruchamia transformację afiniczną oraz MixUp wewnątrz gałęzi
mozaiki. Obejmuje YOLOX, YOLOv7, YOLOv9 i jego warianty E2E oraz P2, RTMDet,
PicoDet, RT-DETR, RT-DETRv2 i FOMO.

Przekazujący pipeline w stylu DETR nie ma mozaiki ani transformacji afinicznej.
Jego zniekształcenie fotometryczne, pomniejszenie i wycinek IoU są stałymi
receptury, a nie parametrami konfiguracji, dlatego aktywne są tylko `flip_prob`
i `no_aug_epochs`. Obejmuje D-FINE, Dome-DETR, DEIM, DEIMv2, RT-DETRv4, EC oraz,
z jedną zmianą, RF-DETR.

Pipeline klasyfikacji ImageFolder ignoruje każdy parametr detekcji. Jego odbicie
poziome ma stałe prawdopodobieństwo 0.5, na które `flip_prob` nie wpływa. Ma
zamiast tego własny pakiet parametrów opisany poniżej.

YOLO-NAS ma własny wariant: bez mozaiki, z zawsze aktywną transformacją
afiniczną dla każdej próbki i z MixUp stosowanym niezależnie zamiast warunkowo.
Wartość `mosaic_scale` jest ponownie używana jako zakres skali transformacji
afinicznej.

SegFormer i NAFNet uruchamiają pipeline specyficzny dla zadania, którego
losowość jest ustalona w rodzinie, a nie konfigurowalna. Dla SegFormer aktywnymi
parametrami są atrybuty klasy `semantic_scale_jitter` i `semantic_hsv_prob`, a
nie `mosaic_scale` oraz `hsv_prob`. Wycinanie i odbicia NAFNet są sprzężonymi
operacjami wejścia i celu o stałym prawdopodobieństwie 0.5.

## Parametry obsługiwane przez poszczególne rodziny

Poniższa tabela przedstawia dostarczaną specyfikację z
`libreyolo/data/augment/spec.py`, której zgodność z rzeczywistym połączeniem
pipeline'u sprawdzają testy biblioteki. Należy ją odczytywać z tego miejsca,
zamiast wnioskować na podstawie architektury.

<code-tabs name="support" />

Podsumowanie według pipeline'u dla parametrów podstawowych:

| Parametr | Styl YOLOX | YOLO-NAS | Styl DETR | Klasyfikacja |
|---|---|---|---|---|
| `mosaic_prob` | używany | ignorowany | ignorowany | ignorowany |
| `mixup_prob` | warunkowy od mozaiki | używany | ignorowany | ignorowany |
| `hsv_prob` | używany | używany | ignorowany | ignorowany |
| `flip_prob` | używany | używany | używany | ignorowany |
| `flipud` | używany | używany | ignorowany | ignorowany |
| `degrees` | warunkowy od mozaiki | używany | ignorowany | ignorowany |
| `translate` | warunkowy od mozaiki | używany | ignorowany | ignorowany |
| `shear` | warunkowy od mozaiki | używany | ignorowany | ignorowany |
| `perspective` | warunkowy od mozaiki | używany | ignorowany | ignorowany |
| `mosaic_scale` | warunkowy od mozaiki | używany | ignorowany | ignorowany |
| `mixup_scale` | warunkowy od mozaiki | używany | ignorowany | ignorowany |
| `no_aug_epochs` | używany | używany | używany | używany |

Wyjątki wewnątrz tych kolumn, wszystkie zawężające zakres:

- RTMDet, PicoDet, RT-DETR, RT-DETRv2 i FOMO nie mają odbicia pionowego, dlatego
  `flipud` jest ignorowany. Wrapper mozaiki FOMO również nie obsługuje
  perspektywy.
- Natywny pipeline RF-DETR nie ma zaburzenia HSV, dlatego `hsv_prob` jest
  ignorowany dodatkowo względem kolumny w stylu DETR.
- EC respektuje `hsv_prob`, `degrees` i `translate`, ale tylko dla
  `task="pose"`, którego transformacja uwzględniająca punkty kluczowe odczytuje
  te wartości. Ścieżki detect i segment używają stałych receptur
  fotometrycznych.
- DINOv2 stosuje kolumnę w stylu DETR dla zadań detect i semantic oraz dodaje
  pakiet klasyfikacji dla `task="classify"`.

`no_aug_epochs` jest `used` wszędzie, lecz nie wszędzie oznacza to samo. W
pipeline'ach mozaikowych wyłącza mozaikę i MixUp na ostatnie epoki. W
pipeline'ach w stylu DETR zatrzymuje augmentacje fotometryczne, pomniejszenie i
wycinanie oraz kształtuje końcówkę harmonogramu. W pipeline'ach klasyfikacji i
semantycznym tylko kształtuje końcówkę.

## Pakiet klasyfikacji

Cztery parametry sterują pipeline'em klasyfikacji i żadnym innym. Rodziny
detekcji ignorują wszystkie cztery.

<code-tabs name="classify" />

`auto_augment` przyjmuje `"randaugment"`, `"autoaugment"`, `"augmix"` lub
`None`. `erasing` jest prawdopodobieństwem RandomErasing. `mixup` i `cutmix` są
prawdopodobieństwami dla każdego batcha, które tworzą miękkie etykiety. W jednym
batchu uruchamiany jest najwyżej jeden z nich, najpierw MixUp, więc wartości się
sumują i ich suma powinna wynosić najwyżej 1.

Wszystkie cztery są domyślnie wyłączone, dlatego trenowanie klasyfikacji nie
zmienia się bez jawnego żądania.

Jedną kolizję nazw warto opisać wprost. W CLI `mixup` jest aliasem parametru
detekcji `mixup_prob`. Pole klasyfikacji `mixup` nie ma własnej pisowni w CLI i
jest dostępne wyłącznie przez `model.train(mixup=...)` w Pythonie.

## Parametry specyficzne dla rodziny

Niektóre parametry znajdują się w podklasie konfiguracji rodziny, a nie w klasie
podstawowej, dlatego istnieją tylko dla tej rodziny i nie mają flagi CLI.

| Rodzina | Parametr | Działanie |
|---|---|---|
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste` | Prawdopodobieństwo augmentacji instancji copy-paste, tylko dla `task="segment"` |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste_mode` | `"flip"` ponownie używa tej samej odbitej próbki, a `"mixup"` pobiera drugą próbkę |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `rot90` | Prawdopodobieństwo losowego obrotu o 90 stopni |
| YOLOv9 | `max_labels` | Limit danych referencyjnych na obraz w transformacjach trenowania, domyślnie 100 |
| RF-DETR | `copy_paste`, `copy_paste_mode` | Copy-paste dla `task="segment"`, tylko tryb `"flip"` |
| RF-DETR, D-FINE, EC | `crop_resize_prob` | Prawdopodobieństwo losowego wycinania ze zmianą rozmiaru |
| EC, YOLO-NAS | `brightness_contrast_prob`, `affine_prob` | Prawdopodobieństwa zaburzenia jasności i kontrastu oraz transformacji afinicznej uwzględniającej punkty kluczowe w ścieżce pozy |

`max_labels` jest jedynym parametrem, który po cichu powoduje utratę danych.
Ramki powyżej limitu są odrzucane bez błędu, dlatego dla gęstych obrazów, takich
jak fotografia lotnicza, trzeba go zwiększyć.

Mozaika i MixUp są wyłączone podczas trenowania obróconych ramek niezależnie od
parametrów, ponieważ augmentacja uwzględniająca narożniki obróconych ramek nie
jest zaimplementowana.

## Powiązane strony

- [Hiperparametry](/docs/train/hyperparameters) opisują `no_aug_epochs` jako
  argument harmonogramu oraz pozostałe argumenty `train()`.
- [Zbiory danych](/docs/train/datasets) opisują formaty etykiet używane przez te
  transformacje.

