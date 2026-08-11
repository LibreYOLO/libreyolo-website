---
title: Hiperparametry
seo_title: Hiperparametry trenowania w LibreYOLO
description: >-
  Najważniejsze argumenty train(): epochs, batch, lr0, optimizer, EMA,
  autobatch, akumulacja gradientów i wznawianie oraz przyczyny różnic między
  wartościami domyślnymi rodzin.
lead: >-
  Każdy argument trenowania jest polem dataclass TrainConfig. Klasa podstawowa
  definiuje pole i jego wartość domyślną, a każda rodzina modeli dziedziczy po
  niej i zastępuje wartości domyślne zmienione w swojej opublikowanej
  recepturze.
keywords:
  - argumenty trenowania
  - współczynnik uczenia
  - rozmiar batcha
  - autobatch
  - wykładnicza średnia ruchoma
  - akumulacja gradientów
  - wznawianie trenowania
  - early stopping patience
  - amp bfloat16
  - train config yaml
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        results = model.train(
            data="my-dataset.yaml",
            epochs=100,
            batch=16,
            imgsz=640,
            lr0=0.01,
        )

        print(results["best_mAP50_95"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 batch=16 imgsz=640 lr0=0.01
  defaults:
    - label: Odczytywanie rozwiązanych wartości domyślnych rodziny
      language: python
      code: |
        from dataclasses import fields

        from libreyolo import LibreYOLO9
        from libreyolo.training.config import TrainConfig

        family_cfg = LibreYOLO9.TRAIN_CONFIG()
        base_cfg = TrainConfig()

        for f in fields(family_cfg):
            family_value = getattr(family_cfg, f.name)
            base_value = getattr(base_cfg, f.name, None)
            if not hasattr(base_cfg, f.name) or family_value != base_value:
                print(f"{f.name}: {family_value}")
    - label: CLI
      language: bash
      code: >
        # Wyświetla wartości domyślne train, val i predict, w tym zastąpienia
        rodziny.

        libreyolo cfg
  autobatch:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # batch=-1 sprawdza pamięć GPU i rozwiązuje wartość do konkretnej potęgi
        dwójki.

        model.train(data="my-dataset.yaml", batch=-1, imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml batch=-1
  accumulate:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 4 mikrobatche po 16 na krok optymalizatora, efektywny batch 64.
        model.train(data="my-dataset.yaml", batch=16, nbs=64)
  resume:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Wczytaj checkpoint przerwanego przebiegu, a następnie zażądaj
        wznowienia.

        model = LibreYOLO("runs/train/exp/weights/last.pt")

        model.train(data="my-dataset.yaml", epochs=100, resume=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=runs/train/exp/weights/last.pt \
          data=my-dataset.yaml epochs=100 resume=true
  cfg:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Klucze w yaml są nazwami pól TrainConfig. Jawne argumenty nazwane mają
        pierwszeństwo.

        model = LibreYOLO("LibreYOLO9s.pt")

        model.train(data="my-dataset.yaml", cfg="my-recipe.yaml", epochs=50)
source_hash: d838d1abd45af40f
---

## Ustawianie argumentów

`train()` przyjmuje argumenty nazwane, a CLI te same nazwy w postaci
`key=value`.

<code-tabs name="train" />

Obie ścieżki prowadzą do tego samego miejsca. Argumenty nazwane są przekazywane
do `TrainConfig.from_kwargs()`, które buduje dataclass konfiguracji rodziny.

## Literówka nie powoduje błędu

`from_kwargs()` odrzuca każdy klucz, który nie jest polem konfiguracji, i
generuje `UserWarning` z jego nazwą. Trenowanie rozpoczyna się wtedy z wartością
domyślną:

```python
# UserWarning: Nieznane klucze konfiguracji trenowania (zignorowano): ['learning_rate']
model.train(data="my-dataset.yaml", learning_rate=0.001)
```

Nic nie ulega awarii, przebieg zostaje ukończony, a współczynnik uczenia nigdy
nie ma wartości żądanej przez kod wywołujący. W pierwszej epoce nowej receptury
należy przeczytać ostrzeżenia. CLI jest bardziej rygorystyczne, ponieważ
waliduje nazwy flag przed zbudowaniem konfiguracji, więc błędnie napisana flaga
CLI zostaje od razu odrzucona.

## Wartości domyślne zależą od rodziny

`TrainConfig` definiuje pole i podstawową wartość domyślną. Każda rodzina
dziedziczy po tej klasie i zastępuje elementy zmienione w swojej opublikowanej
recepturze. Nie istnieje więc jedna prawidłowa odpowiedź na pytanie „jaki jest
domyślny współczynnik uczenia”.

Podstawowe wartości domyślne to `optimizer="sgd"`, `lr0=0.01`,
`momentum=0.937`, `weight_decay=5e-4`, `scheduler="yoloxwarmcos"`, `epochs=300`,
`batch=16`, `imgsz=640` i `amp=True`. Trzy przykłady odchyleń rodziny od tych
wartości:

| Pole | Podstawa | YOLOv9 | D-FINE | YOLO-NAS |
|---|---|---|---|---|
| `optimizer` | `sgd` | `sgd` | `adamw` | `adamw` |
| `lr0` | `0.01` | `0.01` | `2e-4` | `5e-4` |
| `weight_decay` | `5e-4` | `5e-4` | `1e-4` | `1e-5` |
| `scheduler` | `yoloxwarmcos` | `linear` | `flat_cosine` | `cos` |
| `epochs` | `300` | `300` | `132` | `300` |
| `amp` | `True` | `True` | `False` | `False` |

D-FINE i DEIM są dostarczane z `amp=False`, ponieważ dekoder D-FINE ogranicza
aktywacje do 65504, największej skończonej wartości float16. YOLO-NAS i FOMO
również domyślnie wyłączają tę opcję. Flaga `--amp` w CLI ma wartość domyślną
`True` dla każdej rodziny, dlatego jest uznawana za podaną przez użytkownika i
zastępuje wartość domyślną rodziny. Nie należy jej zmieniać bez świadomego celu.

Aby odczytać rzeczywiste wartości domyślne rodziny zamiast zgadywać:

<code-tabs name="defaults" />

## Rozmiar batcha

`batch` jest globalnym batchem. Podczas trenowania na wielu GPU każdy proces
wczytuje `batch // world_size`, więc podana liczba jest liczbą obrazów na krok
optymalizatora niezależnie od liczby używanych GPU. Zobacz
[trenowanie na wielu GPU](/docs/train/multi-gpu).

`batch=-1` włącza autobatch. Trener sprawdza model w trybie trenowania z
rzeczywistym przebiegiem wstecznym dla kolejnych potęg dwójki, dopasowuje prostą
do krzywej pamięci i wybiera największą potęgę dwójki ściśle mniejszą niż
ekstrapolowana wartość mieszcząca się w 60 procentach całkowitej pamięci VRAM.

<code-tabs name="autobatch" />

Istotą jest sprawdzanie w trybie trenowania z przebiegiem wstecznym. Test w
trybie inferencji pomija zachowane aktywacje i tensory gradientów, które dla
głębokiej sieci CNN zajmują kilka razy więcej pamięci niż inferencja. RF-DETR
obniża docelowy udział do 45 procent, ponieważ syntetyczny przebieg wsteczny w
teście nadal zaniża koszt kryterium i pomocniczych warstw dekodera.

Autobatch jest funkcją CUDA. Na CPU lub MPS zapisuje jeden wiersz w logu i
pozostawia domyślny batch.

## Akumulacja gradientów

`nbs` ustawia nominalny, czyli efektywny, rozmiar batcha. Trener akumuluje
`round(nbs / batch)` mikrobatchy na krok optymalizatora.

<code-tabs name="accumulate" />

Pozostawienie wartości domyślnej `None` wyłącza akumulację i nie zmienia
trenowania.

## Współczynnik uczenia i harmonogram

`lr0` jest początkowym współczynnikiem uczenia, a `optimizer` przyjmuje `sgd`,
`adam` lub `adamw`. `momentum` jest momentem SGD albo beta1 algorytmu Adam,
`weight_decay` składnikiem L2, a `nesterov` dotyczy SGD.

Kształt harmonogramu określają `scheduler`, `warmup_epochs`, `warmup_lr_start` i
`min_lr_ratio`. `no_aug_epochs` ustawia liczbę końcowych epok bez silnej
augmentacji. Kilka harmonogramów używa go również do kształtowania swojej
końcówki, więc nie jest to wyłącznie parametr augmentacji. Sposób wykorzystania
części augmentacyjnej przez każdą rodzinę opisano na stronie
[augmentacje](/docs/train/augmentations).

Niektóre rodziny dodają własne parametry współczynnika uczenia.
`backbone_lr_mult` skaluje grupę backbone względem głowicy, `clip_max_norm`
ustawia przycinanie gradientów, a SegFormer używa `head_lr_mult`, aby uruchamiać
głowicę dekodującą z dziesięciokrotnie większym współczynnikiem niż backbone.
Parametry te znajdują się w podklasie konfiguracji rodziny, a nie w klasie
podstawowej.

## EMA

`ema=True` utrzymuje wykładniczą średnią ruchomą wag obok wag trenowanych. Jest
domyślnie włączone wszędzie poza FOMO.

`ema_decay` jest docelowym współczynnikiem zaniku. Zanik narasta, zamiast
zaczynać od wartości docelowej. Efektywna wartość w aktualizacji `n` wynosi
`ema_decay * (1 - exp(-n / tau))`, przy domyślnej wartości `tau` równej 2000,
dzięki czemu wczesne aktualizacje dokładniej śledzą model, a późne je wygładzają.
Wartości domyślne rodzin obejmują zakres od `0.997` dla pozy YOLO-NAS przez
`0.9998` dla YOLOX do `0.9999` dla YOLOv9 i linii DETR.

Do walidacji używane są wagi EMA i to one trafiają do `best.pt` oraz `last.pt`.
Surowe wytrenowane wagi również są przechowywane pod kluczem `train_model`, aby
wznowienie kontynuowało trenowaną trajektorię, a nie średnią.

## Precyzja

`amp=True` uruchamia przebieg do przodu w ramach CUDA autocast. `amp_dtype`
wybiera `float16` (wartość domyślna) albo `bfloat16`. Akceptowane są też pisownie
`fp16` i `bf16`.

Float16 wymaga dynamicznego skalowania funkcji straty i otrzymuje aktywny
`GradScaler`. Szerszy zakres wykładnika Bfloat16 go nie wymaga, dlatego jego
skaler jest tworzony, ale wyłączony, co zachowuje identyczną ścieżkę
optymalizatora. Żądanie bfloat16 na urządzeniu CUDA bez jego obsługi powoduje
błąd podczas konfiguracji zamiast cichego obniżenia możliwości.

## Dane wyjściowe, checkpointy i zatrzymywanie

Przebiegi są zapisywane w `project/name`. `project` ma wszędzie wartość domyślną
`runs/train`, ale `name` jest jednym z pól zastępowanych dla poszczególnych
rodzin. Podstawową wartością domyślną jest `exp`, podczas gdy YOLOv9 używa
`yolo9_exp`, a D-FINE `dfine_exp`. Przy domyślnym `exist_ok=False` istniejący
katalog otrzymuje zwiększony sufiks zamiast zostać nadpisany.

`save_period` zapisuje dodatkowy plik `weights/epoch_<N>.pt` co N epok, oprócz
`weights/last.pt` po każdej epoce i `weights/best.pt` przy każdej poprawie
śledzonej metryki. `eval_interval` ustawia częstotliwość walidacji, a `patience`
zatrzymuje przebieg po tylu epokach bez poprawy. Wartość `0` wyłącza early
stopping.

`cache` przyspiesza powtarzane epoki przez przechowywanie zdekodowanych obrazów
w RAM (`True` lub `"ram"`) albo jako pliki `.npy` obok źródeł (`"disk"`). Odczyty
z pamięci podręcznej są identyczne bajt po bajcie ze świeżymi odczytami. Przy
workerach modułu wczytującego dane bezpieczniejszą opcją jest `"disk"`.

## Wznawianie

`resume=True` kontynuuje przerwany przebieg. Najpierw trzeba wczytać checkpoint,
ponieważ wznowienie odczytuje go z modelu, a nie z osobnego argumentu.

<code-tabs name="resume" />

Wznowienie przywraca wytrenowane wagi, stan optymalizatora, wagi EMA i liczbę
aktualizacji, śledzenie najlepszej metryki, skalę `GradScaler` oraz stany losowe
PyTorch, CUDA i NumPy. Rozpoczyna od epoki następującej po epoce checkpointu i
przewija harmonogram do tej pozycji.

Nie wykona dwóch czynności. `resume=True` nie można łączyć z `pretrained`, gdyż
powoduje to błąd. Jeśli klucz najlepszej metryki checkpointu różni się od klucza
bieżącego przebiegu, śledzenie najlepszej metryki jest zerowane z ostrzeżeniem,
zamiast porównywać wartości o różnych znaczeniach.

## Receptury w pliku

`cfg=` wczytuje mapowanie YAML z nazwami pól `TrainConfig` i scala je pod
jawnymi argumentami nazwanymi, dlatego argument nazwany zawsze ma pierwszeństwo
przed plikiem.

<code-tabs name="cfg" />

`size` i `num_classes` są usuwane z pliku, ponieważ instancja modelu już nimi
zarządza. CLI nie ma flagi `--cfg`. Ścieżka pliku jest argumentem Pythona.

## Powiązane strony

- [Zbiory danych](/docs/train/datasets) opisują wartości przyjmowane przez
  `data=`.
- [Augmentacje](/docs/train/augmentations) opisują parametry augmentacji i
  rodziny, które je respektują.
- [Zamrażanie warstw](/docs/train/layer-freezing) i [LoRA](/docs/train/lora)
  opisują trenowanie podzbioru wag.
- [Walidacja i metryki](/docs/train/validation) opisują raportowane wyniki
  przebiegu.

