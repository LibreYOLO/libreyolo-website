---
title: Aktualizacja do 1.5.0
seo_title: Aktualizacja LibreYOLO z 1.4.0 do 1.5.0
description: >-
  Cztery wymagane zmiany w kodzie dla wersji 1.5.0, trzy zmiany wpływające na
  metryki i mniejsze różnice w zachowaniu, które warto znać przed porównaniem
  uruchomień.
lead: >-
  Z publicznego API modeli niczego nie usunięto: każdą klasę i funkcję
  działającą w 1.4.0 nadal można zaimportować. Zmieniła się postać czterech
  argumentów, a trzy wartości domyślne wpływają na porównywane wyniki.
keywords:
  - aktualizacja libreyolo
  - migracja libreyolo 1.5.0
  - usunięte allow_experimental
  - libreyolo zmiany niezgodne wstecznie
  - yolox bn eps
  - faster-coco-eval domyślnie
last_verified: 1.5.0
meta:
  - label: Dotyczy
    value: Od 1.4.0 do 1.5.0
  - label: Wymagane zmiany w kodzie
    value: 'Cztery, wszystkie niewielkie'
  - label: 'Wyniki, które się zmieniają'
    value: 'Backend COCO, eps BN w YOLOX, wieloskalowość D-FINE'
  - label: Usunięcia z publicznego API
    value: Brak
source_hash: ab38d8ef7b53f596
---

Ta strona dotyczy aktualizacji samego LibreYOLO. Instrukcje wczytywania
checkpointu z projektu źródłowego znajdują się na osobnej stronie [import
istniejących wag](/docs/migrate).

Pełny wpis dotyczący wydania znajduje się w [dzienniku zmian](/docs/changelog).
Poniżej opisano wyłącznie część wymagającą działania.

## Wymagane zmiany w kodzie

### `allow_experimental=True` już nie istnieje

Usunięto bramkę potwierdzenia wraz ze stojącym za nią mechanizmem
`ddp_aware(experimental_key=...)`. Trenowanie i eksport rodzin EC, RTMDet,
PicoDet oraz FOMO wymagały wcześniej tego argumentu, więc zmiana dotyczy każdego
skryptu trenującego jedną z tych rodzin.

```python
# 1.4.0
model.train(data="data.yaml", epochs=100, allow_experimental=True)

# 1.5.0: usuń argument
model.train(data="data.yaml", epochs=100)
```

Nie ma warstwy zgodności sygnalizującej wycofanie. Wywołanie, które nadal
przekazuje ten argument, zgłasza `TypeError`. Wraz z nim usunięto
`BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES`. Hook `get_download_notice()` pozostał
i nadal jest zastępowany przez MiDaS, SegFormer oraz YOLO9-P2.

Poziomy wsparcia są nadal publikowane, ale nie stanowią już argumentu. Zobacz
[poziomy stabilności](/docs/reference/stability-tiers).

### Poziom eksportu `"experimental"` już nie istnieje

```python
from libreyolo.export.support import Tier

# 1.4.0: Literal["validated", "experimental", "blocked"]
# 1.5.0: Literal["validated", "available", "blocked"]
```

Kod rozgałęziający działanie na podstawie ciągu poziomu powinien teraz
odczytywać `"available"` w miejscu, w którym wcześniej odczytywał
`"experimental"`. `BaseExporter` nie emituje już `RuntimeWarning` dla tych
formatów. Stan poszczególnych formatów zawiera [macierz
eksportu](/docs/reference/export-matrix).

### Połączenie `pretrained=False` z `resume` jest teraz odrzucane

Wcześniej ta kombinacja prowadziła do niespójnego działania. Teraz zgłasza:

```
ValueError: pretrained=False cannot be combined with resume.
```

Należy wybrać jedną opcję. `pretrained=False` rozpoczyna od nowej,
deterministycznie inicjowanej konfiguracji, która w 1.5.0 działa dla każdej
rodziny obsługującej trenowanie, a nie tylko dla trzech z nich. `resume`
kontynuuje przerwane uruchomienie od jego checkpointu. Obie opcje opisano w
sekcji [trenowanie](/docs/train).

### `--imgsz` w CLI jest ciągiem znaków, a nie liczbą całkowitą

Zmiana ma węższy zakres, niż może się wydawać. Oba poniższe przypadki pozostają
bez zmian:

```bash
libreyolo predict --model yolo9-t --source img.jpg --imgsz 640   # nadal działa
```

```python
model.predict("img.jpg", imgsz=640)   # nadal działa
```

Zmiany wymaga tylko kod, który bezpośrednio wywołuje funkcje poleceń
[CLI](/docs/cli) z Pythona. Typ `--imgsz` w `predict`, `train` i `val`
rozszerzono z `int` do `str`, aby akceptować prostokątne rozmiary:

```python
from libreyolo.cli.commands.predict import predict_cmd

predict_cmd(..., imgsz=640)      # 1.4.0
predict_cmd(..., imgsz="640")    # 1.5.0, teraz działa też "480x640"
```

Wartością domyślną `train` jest teraz ciąg `"640"`. Argument `export --imgsz`
był już ciągiem znaków, a `profile` pozostaje bez zmian.

## Zmieniające się wartości

Trzy zmiany wpływają na metryki przy ustawieniach domyślnych. Przed porównaniem
uruchomienia 1.5.0 z uruchomieniem 1.4.0 należy zapoznać się z tymi informacjami.

### faster-coco-eval jest domyślnym backendem metryk COCO

Funkcja `val()` i walidacja w każdej epoce trenowania obliczają teraz metryki
COCO za pomocą backendu C++ faster-coco-eval zamiast pycocotools.

Decyzję podjęto na podstawie zmierzonej zgodności we wszystkich 100 podziałach
testowych RF100-VL. Spośród 1400 wartości metryk 1381 było identycznych bitowo,
maksymalne odchylenie wyniosło 2.22e-16, różnice głównych wskaźników dokładnie
0, a całość działała 15.6 razy szybciej i 56 razy szybciej na zbiorach danych z
dużą liczbą detekcji. Wyniki nie powinny się zmienić. Mimo to tworzy je inna
implementacja i właśnie dlatego zmiana znajduje się na tej liście.

pycocotools pozostaje automatycznym rozwiązaniem zapasowym, gdy
faster-coco-eval nie jest zainstalowany. Aby go wymusić:

```bash
libreyolo val --model yolo9-t --data coco.yaml --no-faster-coco-eval
```

```python
model.val(data="coco.yaml", faster_coco_eval=False)
```

Ustawienie `LIBREYOLO_FASTER_COCO_EVAL=0` daje ten sam efekt globalnie.
Faktycznie użyty backend jest zapisywany w logu na poziomie INFO, dostępny jako
`model.last_eval_backend` po wywołaniu `val()` i zawarty jako `eval_backend` w
danych JSON [CLI](/docs/cli/val). Szybką ścieżkę można zainstalować poleceniem
`pip install libreyolo[fast-eval]`.

### Checkpointy YOLOX wytrenowane przed 1.5.0 wymagają nadpisania eps

To pułapka tego wydania. Należy ją uwzględnić w przypadku dostrojonego modelu
[YOLOX](/docs/models/yolox).

YOLOX określa parametry BatchNorm jako `eps=1e-3` i `momentum=0.03`. Do wersji
1.5.0 wartości te stosowano jako późniejszą poprawkę, która nie zachowywała się
podczas przebudowy liczby klas wykonywanej przez `train()`, gdy `nc` zbioru
danych różniło się od wartości checkpointu. Taki dostrojony model był trenowany
i walidowany w trakcie trenowania z domyślnym dla torch `eps=1e-5`, a następnie
ponownie wczytywany do inferencji z wartością `1e-3`. Te same tensory działały
więc z inną normalizacją.

Rozmiary ze zwykłymi konwolucjami zmieniają się nieznacznie. Rozmiar depthwise
`n` zmienia się znacznie, ponieważ jego `running_var` dla poszczególnych kanałów
jest na tyle małe, że zaczyna dominować eps. W podziale `ball` zbioru RF100-VL
ten sam checkpoint nano osiąga **0.566** mAP50-95 przy eps użytym podczas
trenowania i **0.151** po standardowym ponownym wczytaniu.

Checkpoint wytrenowany przed 1.5.0 ma semantykę eps=1e-5. Aby uzyskać dla niego
wiarygodne wyniki, można przeprowadzić ocenę z nadpisanym eps BN na 1e-5:

```python
import torch
from libreyolo import LibreYOLOX

model = LibreYOLOX("my-yolox-finetune.pt")
for module in model.model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eps = 1e-5

model.val(data="data.yaml")
```

Można też jednorazowo włączyć `sqrt((var + 1e-3) / (var + 1e-5))` do wag BN i
zapisać wynik. Checkpointy wytrenowane w wersji 1.5.0 lub nowszej nie wymagają
żadnej z tych czynności.

### Trenowanie wieloskalowe D-FINE korzysta z procedury projektu źródłowego dla każdego rozmiaru

Wartość `base_size_repeat` była ustawiona na stałe na 3 dla każdego rozmiaru.
Teraz jest wyznaczana dla poszczególnych rozmiarów zgodnie z projektem
źródłowym: **n** jest trenowany przy stałym rozmiarze z wyłączoną
wieloskalowością, **s** ma wartość 20, **m** 6, **l** 4, a **x** 3. Wcześniej
zgodny był tylko x, więc n, s, m i l otrzymują inny rozkład skal i zbiegają do
innych wartości metryk.

Aby przywrócić stare działanie, ustaw wartość jawnie:

```python
from libreyolo.training.config import DFINEConfig

config = DFINEConfig(base_size_repeat=3)
```

DEIM nadal używa stałej wartości 3. Szczegóły rodziny znajdują się na stronie
[D-FINE](/docs/models/d-fine).

## Warto wiedzieć, działanie nie jest wymagane

- **Wyniki prostokątnego `imgsz` zmieniły się, ponieważ wcześniej były
  błędne.** Współrzędne ramek, zmiana rozmiaru masek RTMDet, przeskalowanie
  YOLO-NAS i skalowanie danych referencyjnych w walidatorze używają teraz
  osobnych wartości wysokości i szerokości zamiast jednego skalara. Kwadratowy
  `imgsz` nie zmienił się na poziomie bitów. Prostokątna inferencja lub walidacja
  w wersji 1.4.0 była nieprawidłowo skalowana. YOLO-NAS odrzuca teraz
  prostokątny `imgsz`, zamiast po cichu tworzyć błędne dane wyjściowe.
- **Słowniki metryk zyskały nowe klucze.** Są to `max_det`, `ar_max_det` i
  `AR_max_det` z modułu oceniającego COCO oraz `metrics/loss` i
  `metrics/loss/ce` z FOMO. Wartości przy ustawieniach domyślnych pozostają bez
  zmian, ale każdy mechanizm iterujący po kluczach metryk, w tym niestandardowe
  [moduły rejestrujące](/docs/train/loggers) i nagłówki CSV, zobaczy nowe
  kolumny.
- **Deterministyczne uruchomienia YOLO9, które powodują przebudowę głowicy,**
  rozpoczynają od innej inicjalizacji, ponieważ ziarno jest teraz stosowane
  przed przebudową, a nie po niej. Dostrajania z wersji 1.4.0 z ustalonym ziarnem
  do innej liczby klas nie można odtworzyć bit po bicie w wersji 1.5.0.
- **`libreyolo[hub-kernels]` na CUDA rzeczywiście włącza teraz natywny kernel
  MS-deform-attn.** Wersja 1.4.0 uzależniała go od warunku, którego RF-DETR
  nigdy nie spełniał, więc kernel nie był uruchamiany. Predykcje dla RF-DETR i
  pozostałych rodzin z deformowalną uwagą mogą zmienić się w granicach
  tolerancji float. Nie dotyczy to standardowych instalacji, a ustawienie
  `LIBREYOLO_HUB_KERNELS=0` wyłącza kernel.
- **`libreyolo predict` pomija nieobsługiwane opcje zamiast zgłaszać wyjątek.**
  CLI filtruje argumenty nazwane na podstawie sygnatury `__call__` modelu, więc
  opcja nieprzyjmowana przez rodzinę jest ignorowana zamiast zgłoszenia
  `TypeError`. Literówka w nazwie flagi jest teraz po cichu ignorowana.
- **Źródła na żywo zmieniają strukturę danych wyjściowych JSON.** Kamery
  internetowe, strumienie RTSP i przechwytywanie ekranu domyślnie włączają
  streaming, który emituje jeden rekord na klatkę zamiast jednego na wywołanie.
  Te [źródła](/docs/predict/sources) są nowe w wersji 1.5.0, więc zmiana nie
  dotyczy żadnego skryptu 1.4.0.
- **Ponowny eksport `rfdetr-pose` lub `yolonas-pose` do ONNX tworzy inne nazwy
  wyjściowe.** Wersja 1.4.0 błędnie interpretowała ich wielotensorowe głowice
  pozy jako segmentację za pomocą heurystyki liczby wyjść. Istniejące pliki
  `.onnx` na dysku pozostają bez zmian.
- **W instalacji bez torch** wyniki zawierają tablice numpy zamiast
  `torch.Tensor`, dlatego `.boxes.data` zwraca inny typ, a rozstrzyganie remisów
  przez NMS może różnić się od torchvision. Gdy torch jest zainstalowany,
  działanie pozostaje identyczne bajt po bajcie. Zobacz [lekką
  instalację](/docs/lightweight-install).
- **Obiekty konfiguracji sprawdzają więcej danych podczas tworzenia.**
  `TrainConfig` otrzymał metodę `__post_init__`, której wcześniej nie miał,
  dlatego konfiguracja, która już wcześniej była nieprawidłowa, zgłasza wyjątek
  natychmiast zamiast na późnym etapie uruchomienia. Serializacja
  `ValidationConfig` otrzymała klucz `edge_thresholds`, co uniemożliwia ścisły
  cykl `ValidationConfig(**dump)` z danych wersji 1.4.0.
- **Nazwy plików wag dla rodzin z sufiksem zadania są rozwiązywane inaczej.**
  `segformer-b0` prowadzi teraz do `LibreSegformerb0-sem.pt`. Naprawia to błędy
  404 automatycznego pobierania, ale psuje skrypty z wpisaną na stałe starą
  nazwą bez sufiksu.
- **Znacznik pytest `experimental_backend` nazywa się teraz
  `extended_backend`.** Ma to znaczenie tylko w przypadku uruchamiania zestawu
  testów z opcją `-m`.

## Checkpointy i zbiory danych

Checkpointy zapisane przez wersję 1.4.0 wczytują się bez zmian.
[Schemat](/docs/reference/checkpoint-schema) otrzymał pola `imgsz_h` i `imgsz_w`
dla modeli prostokątnych i nadal zapisuje skalar `imgsz = max(h, w)` dla
starszych modułów odczytujących. Eksporty [ExecuTorch](/docs/export/executorch)
i [MNN](/docs/export/mnn) wymagają teraz pliku towarzyszącego, odpowiednio
`<program>.pte.json` i `<model>.mnn.json`, a eksporty HRNet zawierają
`pose_input: "person_crop"`. Formaty zbiorów danych pozostają bez zmian.
