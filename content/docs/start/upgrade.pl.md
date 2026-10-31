---
title: Aktualizacja do 1.7.0
seo_title: Aktualizacja LibreYOLO z 1.6.0 do 1.7.0
description: >-
  Migracja checkpointów, nazw modułów i receptur trenowania YOLO9, katalogów
  uruchomień CLI, macierzy pomyłek i nowych błędów w LibreYOLO 1.7.0.
lead: >-
  Wersja 1.7.0 zachowuje wczytywanie starszych checkpointów YOLO9 z tymi samymi
  ramkami zmiennoprzecinkowymi. Zmienia trenowanie rodziny YOLO9, nazwy widoczne
  dla kodu wewnątrz sieci oraz kilka kontraktów walidacji i predykcji.
keywords:
  - aktualizacja libreyolo
  - migracja libreyolo 1.6.0
  - libreyolo zmiany niezgodne wstecznie
  - nazwy modułów yolo9
  - clip_max_norm
  - macierz pomyłek
last_verified: 1.7.0
source_hash: 895bcd7425f0052f
---

## Z 1.6.0 do 1.7.0

- Checkpointy YOLO9, YOLO9-E2E i YOLO9-P2 zapisane w 1.7.0 używają nowych nazw tensorów i nie wczytują się w 1.6.x: 1.6.x wczytuje checkpointy YOLO9 i YOLO9-P2 z ostrzeżeniem o brakujących kluczach i nie przewiduje żadnych ramek, a dla checkpointów YOLO9-E2E zgłasza błąd niezgodności `model_family`. Należy zaktualizować każde środowisko, które je odczytuje. Checkpointy zmiennoprzecinkowe, kwantyzowane i treningowe zapisane przez wcześniejsze wydania wczytują się w 1.7.0, a eksporty ONNX pozostają bez zmian.

- Kod sięgający do `model.model` modelu z rodziny YOLO9 widzi nazwy modułów z MultimediaTechLab/YOLO, bez aliasów. `DDetect` to `YOLO9Head` (`anchor_convs` / `class_convs`), `RepConvN` to `RepConv`, `RepNBottleneck` to `Bottleneck`, `DFL` to `Anchor2Vec` (`anchor2vec`), `nc` to `num_classes`, a podwarstwy bloków to `conv1`/`conv2`/`bottleneck`. W YOLO9-E2E `YOLO9E2EDetect` to `YOLO9E2EHead`, a `one2one_cv2` / `one2one_cv3` to `one_to_one_anchor_convs` / `one_to_one_class_convs`. `LibreYOLO9Model.fuse()`, `RepConvN.fuse_convs()`, `Conv.forward_without_bn()`, `Concat` i `yolo9.utils.decode_boxes` zostały usunięte. [Lista zmian](/docs/changelog) wymienia każdą zmianę nazwy.

- Nazwy węzłów ONNX w głowicy, backbone i neck odpowiadają nowym nazwom modułów, na przykład `/head/anchor_convs.0/...`. Należy zaktualizować narzędzia oparte na nazwach węzłów, takie jak konfiguracje węzłów końcowych Hailo. Wejścia modelu, wyjścia i ich nazwy pozostają bez zmian.

- Dostrajanie YOLO9 z wag bez tensorów PGI, w tym z opublikowanych `LibreYOLO9{t,s,m,c}.pt`, trenuje tylko główną głowicę, jak w 1.5.0; wersja 1.6.0 dołączała losowo zainicjalizowaną gałąź PGI z `aux_weight=0.25`. Wagi, które zawierają tensory PGI, oraz uruchomienia od zera nadal trenują z PGI. Należy przekazać `aux_weight=0.25`, aby mimo to dołączyć nową gałąź, albo `aux_weight=0`, aby ją wyłączyć.

- Gdy PGI jest dołączone, YOLO9-m i YOLO9-c budują pomocniczą gałąź z MultimediaTechLab: 12.8M (m) i 25.7M (c) parametrów używanych tylko podczas trenowania przy 80 klasach, wobec 7.5M i 10.2M wcześniej, więc trenowanie tych rozmiarów z PGI wymaga więcej pamięci i czasu na krok. Checkpoint treningowy m/c z 1.6.0 wznawia trenowanie i dostrajanie z gałęzią, z którą był trenowany. Inferencja oraz YOLO9-t/s pozostają bez zmian.

- YOLO9, YOLO9-E2E i YOLO9-P2 obcinają normę L2 gradientu do 10, a ich przypisywanie w trenowaniu ocenia kotwice w fp32, więc pozytywy wybrane przy fp16 AMP mogą się różnić. Należy przekazać `clip_max_norm=0`, aby uzyskać obcinanie z 1.6.0; uruchomienie rozpoczęte w 1.6.0 i wznowione w 1.7.0 również obcina, chyba że się to przekaże. Przypisywanie w fp32 nie ma przełącznika.

- Niezerowe `perspective` zniekształca obraz przez przesunięcie narożników w każdej rodzinie używającej wspólnej transformacji afinicznej (YOLO9 i jego warianty E2E i P2, YOLOX, YOLOv7, YOLO-NAS, PP-YOLOE, RTMDet, PicoDet, RT-DETR i RT-DETRv2), w tym w uruchomieniach wznowionych z checkpointu 1.6.0, który je ustawiał. Wartości `0.002` i wyższe dają to samo maksymalne zniekształcenie, a wartości ujemne zgłaszają `ValueError`. Żadne ustawienie nie odtwarza zniekształcenia z 1.6.0; domyślne `perspective=0` pozostaje bez zmian.

- Pipeline mosaic YOLO9 losuje współczynnik mixup z rozkładu Beta(1, 1) zamiast Beta(32, 32), bez ustawienia dla starej wartości. Domyślne trenowanie YOLOv7 się zmienia, ponieważ YOLOv7 domyślnie włącza mixup. YOLO9, YOLO9-E2E, YOLO9-P2, RT-DETR i RT-DETRv2 zmieniają się tylko wtedy, gdy mixup jest włączony.

- `libreyolo train` bez `name=` zapisuje uruchomienia EC, D-FINE, SegFormer i LingBot-Vision w `runs/train/ec_exp`, `dfine_exp`, `segformer_exp` i `lingbotvision_exp` zamiast w `runs/train/exp`, sondy V-JEPA 2 w `runs/train/train`, a uruchomienia YOLO9-P2 w `runs/train/yolo9_exp`. Należy przekazać `name=exp` dla starej ścieżki albo `name=yolo9_p2_exp` dla YOLO9-P2.

- `libreyolo train` uruchamia D-FINE, SegFormer i LingBot-Vision z ich własnych receptur. Dla D-FINE oznacza to 132 epoki, `lr0=2e-4`, `weight_decay=1e-4`, 2 epoki rozgrzewki oraz brak augmentacji mosaic, mixup, HSV i afinicznej. Aby przekazać wartości, które wysyłała wersja 1.6.0, należy podać `epochs=300 optimizer=sgd lr0=0.01 weight_decay=5e-4 warmup_epochs=5 mosaic=1.0 mixup=1.0 hsv_prob=1.0 degrees=10 translate=0.1 shear=2.0`.

- `libreyolo.validation.ConfusionMatrix` indeksuje `matrix[predicted, true]` (wcześniej `[actual, predicted]`), liczy detekcje przy `conf_thres=0.25` (wcześniej 0.15), przyjmuje `names=` i `task=` wyłącznie jako argumenty nazwane, a jego `matrix` nie może już być przypisywane na nowo. Należy transponować kod, który indeksował starą macierz, i przekazać `conf_thres=0.15` dla starych liczności. `plots=True` zapisuje liczności w `confusion_matrix.png`, a wykres znormalizowany w `confusion_matrix_normalized.png`.

- `predict(agnostic_nms=True)` tłumi nakładające się ramki między klasami zamiast ostrzegać. Zadania bez ramek zgłaszają `ValueError`, `LibreEnsemble` i HRNet zgłaszają `NotImplementedError`, a poziom SAM odrzuca ten argument z `TypeError`; należy usunąć ten argument z takich wywołań.

- Nieznana wartość `interpolation` klasyfikatora zgłasza `ValueError`, w tym tryby torchvision `LANCZOS`, `BOX`, `HAMMING` i `NEAREST_EXACT`. Należy użyć `bilinear`, `bicubic`, `nearest` lub `bilinear_no_antialias`.

- Przed porównaniem z wynikami 1.6.0 należy ponownie uruchomić walidację dla `val(single_cls=True)` na modelu trenowanym na kilku klasach, dla zbiorów segmentacji mieszających wiersze tylko z ramkami i z wielokątami oraz dla RTMDet-Ins, który domyślnie zachowuje do `max_det=300` masek na obraz; dla ustawienia z MMDetection należy przekazać `max_det=100`.

- Bez `libreyolo[rfdetr]` wywołania CLI z nazwą modelu RF-DETR lub DINOv2 kończą działanie z kodem błędu JSON `model_load_failed` zamiast `model_not_found`, a komunikat podaje `pip install libreyolo[rfdetr]`.

Pełne wydanie opisuje [lista zmian](/docs/changelog).

## Z 1.5.0 do 1.6.0

- Przed instalacją dodatku ONNX należy zaktualizować środowiska z ONNX Runtime poniżej 1.18 do `onnxruntime>=1.18.0`.

- SAM 3D Body przyjmuje sprawdzony zestaw zasobów snapshotu. Należy zainstalować `libreyolo[hf]`, uzyskać dostęp do modelu i użyć automatycznego pobierania albo podać niezmieniony katalog snapshotu i ustalony zasób MHR w zaufanym lokalnym systemie plików.

- Po poprawkach zmiany rozmiaru RF-DETR dla zadań innych niż estymacja pozy oraz normalizacji klasyfikatorów należy ponownie uruchomić walidację. Przed porównaniem z wynikami 1.5.0 należy sprawdzić wdrożone progi pewności; nie ma flagi dawnego przetwarzania wstępnego.

- D-FINE, DEIM, RT-DETRv4 i detekcja YOLO-NAS domyślnie włączają FP16 AMP. Aby zachować FP32, należy przekazać `amp=False`.

- Aby odtworzyć dawne ustawienia dostrajania YOLO9, należy ustawić `aux_weight=0`, `max_labels=100` i `warmup_momentum=0.937`. `letterbox_pad="topleft"` jest potrzebne, gdy nowa konwersja ze znacznikiem wyśrodkowania ma odtworzyć starą geometrię. Stare checkpointy z jedną głowicą wznawiają trenowanie z oryginalnym grafem. Nowe przebiegi trenowania YOLO9 zużywają więcej pamięci GPU, więc należy obniżyć `batch`, jeśli rozmiar batcha z wersji 1.5.0 wyczerpuje pamięć.

- RF-DETR i DINOv2 tworzą katalogi uruchomień z nazwą rodziny i kolejnym numerem. Należy zaktualizować kod korzystający ze ścieżek artefaktów lub ustawić `output_dir="runs/train", exist_ok=True`, aby zachować dawną lokalizację i ponowne używanie katalogu.

- W klasyfikacji należy zachować `mixup + cutmix <= 1`; nieprawidłowe kombinacje zgłaszają teraz błąd podczas konfiguracji.

- QAT wyłącza EMA, SyncBatchNorm i uśrednianie checkpointów. Należy używać checkpointów QAT best/last bez polegania na tych stanach.

- Własne loadery z hookami zmieniającymi zbiór danych muszą używać `persistent_workers=False` lub odtwarzać procesy robocze po zmianie. Niezgodne trwałe kopie przy wielu procesach roboczych zgłaszają teraz błąd.

- Tablice obrazów NumPy są odczytywane jako BGR, w kolejności używanej przez OpenCV. Przy przekazywaniu tablicy RGB, takiej jak `np.asarray(pil_image)`, należy podać `color_format="rgb"`; wynik `cv2.imread()` i klatki wideo nie wymagają zmian.

- Tablica NumPy lub tensor 4D to batch: `predict()` zwraca listę z jednym `Results` na obraz, zamiast używać tylko pierwszego obrazu.

- `train(resume=True)` i `train(resume="path/to/last.pt")` przywracają zapisane argumenty trenowania uruchomienia i nadal zapisują do jego katalogu uruchomienia; argumenty przekazane jawnie mają pierwszeństwo. Wznowienie opublikowanych wag albo uruchomienia, które osiągnęło już swoją wartość `epochs`, zgłasza `ValueError`.

- Przy włączonej walidacji ostatnia epoka jest zawsze walidowana, dlatego uruchomienia krótsze niż `eval_interval` raportują teraz metryki i zapisują `best.pt`. `val=False` wyłącza walidację, łącznie z ostatnią epoką. Walidacja w trakcie trenowania zapisuje do `<run>/val` zamiast do `runs/val/`.

- Niepowodzenia polecenia `train` w CLI raportują typ błędu, dlatego błędy konfiguracji kończą się kodem 2 zamiast kodem 1 z `io_error`.

Pełny opis wydania zawiera [lista zmian](/docs/changelog), a konwersję checkpointów opisano w sekcji [importowania wag](/docs/migrate).

## Z 1.4.0 do 1.5.0

Ta strona dotyczy aktualizacji samego LibreYOLO. Instrukcje wczytywania
checkpointu z projektu źródłowego znajdują się na osobnej stronie [import
istniejących wag](/docs/migrate).

Pełny wpis dotyczący wydania znajduje się w [dzienniku zmian](/docs/changelog).
Poniżej opisano wyłącznie część wymagającą działania.

### Wymagane zmiany w kodzie

#### `allow_experimental=True` już nie istnieje

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
przekazuje ten argument, otrzymuje ostrzeżenie
`Unknown training config keys (ignored)`, a sam argument nie ma żadnego efektu.
Wraz z nim usunięto
`BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES`. Hook `get_download_notice()` pozostał
i nadal jest zastępowany przez MiDaS, SegFormer oraz YOLO9-P2.

Poziomy wsparcia są nadal publikowane, ale nie stanowią już argumentu. Zobacz
[poziomy stabilności](/docs/reference/stability-tiers).

#### Poziom eksportu `"experimental"` już nie istnieje

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

#### Połączenie `pretrained=False` z `resume` jest teraz odrzucane

Wcześniej ta kombinacja prowadziła do niespójnego działania. Teraz zgłasza:

```
ValueError: pretrained=False cannot be combined with resume.
```

Należy wybrać jedną opcję. `pretrained=False` rozpoczyna od nowej,
deterministycznie inicjowanej konfiguracji, która w 1.5.0 działa dla każdej
rodziny obsługującej trenowanie, a nie tylko dla trzech z nich. `resume`
kontynuuje przerwane uruchomienie od jego checkpointu. Obie opcje opisano w
sekcji [trenowanie](/docs/train).

#### `--imgsz` w CLI jest ciągiem znaków, a nie liczbą całkowitą

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

### Zmieniające się wartości

Trzy zmiany wpływają na metryki przy ustawieniach domyślnych. Przed porównaniem
uruchomienia 1.5.0 z uruchomieniem 1.4.0 należy zapoznać się z tymi informacjami.

#### faster-coco-eval jest domyślnym backendem metryk COCO

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

#### Checkpointy YOLOX wytrenowane przed 1.5.0 wymagają nadpisania eps

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
from libreyolo import LibreYOLO

model = LibreYOLO("my-yolox-finetune.pt")
for module in model.model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eps = 1e-5

model.val(data="data.yaml")
```

Można też jednorazowo włączyć `sqrt((var + 1e-3) / (var + 1e-5))` do wag BN i
zapisać wynik. Checkpointy wytrenowane w wersji 1.5.0 lub nowszej nie wymagają
żadnej z tych czynności.

#### Trenowanie wieloskalowe D-FINE korzysta z procedury projektu źródłowego dla każdego rozmiaru

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

### Warto wiedzieć, działanie nie jest wymagane

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
  `TypeError`. Nieznana nazwa flagi jest nadal odrzucana z komunikatem
  `No such option`.
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

### Checkpointy i zbiory danych

Checkpointy zapisane przez wersję 1.4.0 wczytują się bez zmian.
[Schemat](/docs/reference/checkpoint-schema) otrzymał pola `imgsz_h` i `imgsz_w`
dla modeli prostokątnych i nadal zapisuje skalar `imgsz = max(h, w)` dla
starszych modułów odczytujących. Eksporty [ExecuTorch](/docs/export/executorch)
i [MNN](/docs/export/mnn) wymagają teraz pliku towarzyszącego, odpowiednio
`<program>.pte.json` i `<model>.mnn.json`, a eksporty HRNet zawierają
`pose_input: "person_crop"`. Formaty zbiorów danych pozostają bez zmian.
