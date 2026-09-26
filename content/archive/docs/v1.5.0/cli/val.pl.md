---
title: libreyolo val
seo_title: 'libreyolo val, dokumentacja polecenia'
description: >-
  Ocena checkpointu na splicie zbioru danych z wiersza poleceń: każdy argument z
  wartością domyślną oraz klucze metryk zwracane przez poszczególne zadania.
lead: >-
  Ocenia jeden model na jednym splicie zbioru danych i wypisuje metryki. Zestaw
  metryk zależy od zadania modelu, a liczby są tymi samymi, z których powstaje
  wiersz benchmarku.
keywords:
  - libreyolo val cli
  - walidacja modelu yolo cli
  - ewaluacja yolo z wiersza poleceń
  - jak policzyć mAP50-95
  - argumenty libreyolo val
last_verified: 1.5.0
meta:
  - label: Polecenie
    value: libreyolo val
    mono: true
  - label: Wymagane
    value: 'model, data'
    mono: true
  - label: Wyjście
    value: >-
      Metryki na stdout. Wykresy i COCO JSON w runs/val/exp, jeśli zostaną
      zażądane
snippets:
  examples:
    - label: Podstawowe wywołanie
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: Wykresy i COCO JSON
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml \
          imgsz=640 batch=8 save_json=true save_plots=true \
          project=runs/val name=yolo9s-coco8 exist_ok=true
    - label: Format maszynowy
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml json=true quiet=true
source_hash: f6507840568c3725
---

## Składnia

```bash
libreyolo val model=<name|path> data=<dataset.yaml> [key=value ...]
```

Argumenty to pary `key=value`, działa też forma POSIX, więc `batch=8` i
`--batch 8` to ten sam argument.

## Argumenty

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `model` | | Ścieżka do wag modelu lub nazwa CLI. Wymagane |
| `data` | | Ścieżka do pliku YAML zbioru danych (format YOLO, np. `coco8.yaml`). Wymagane |
| `data_dir` | | Bezpośredni katalog zbioru danych, z pominięciem ścieżki z pliku YAML |
| `split` | `val` | Split zbioru danych: `val`, `test`, `train` |
| `batch` | `16` | Rozmiar batcha |
| `imgsz` | | Rozmiar obrazu: `640` (kwadrat) lub `480x640` (HxW). Gdy nie jest ustawiony, własny rozmiar wejścia modelu |
| `conf` | `0.001` | Próg pewności |
| `iou` | `0.6` | Próg IoU dla NMS |
| `max_det` | `300` | Maksymalna liczba predykcji na obraz po NMS |
| `eval_max_det` | | Limit ewaluatora COCO. Gdy nie jest ustawiony, konwencja AP@100 z pycocotools |
| `faster_coco_eval` | `true` | Użycie szybszego backendu C++ faster-coco-eval do metryk COCO, jeśli jest zainstalowany; w przeciwnym razie pycocotools |
| `half` | `false` | Inferencja w FP16 |
| `amp_dtype` | `float16` | Typ autocast CUDA przy `half=true`: `float16` lub `bfloat16` |
| `save_json` | `false` | Zapis wyników JSON w formacie COCO |
| `save_plots` | `false` | Zapis wykresów walidacji: metryki, AP dla poszczególnych klas, macierz pomyłek, próbki |
| `workers` | `4` | Wątki robocze dataloadera |
| `device` | `auto` | Urządzenie |
| `project` | `runs/val` | Katalog główny wyjścia |
| `name` | `exp` | Nazwa eksperymentu |
| `exist_ok` | `false` | Ponowne użycie katalogu wyjściowego |
| `allow_download_scripts` | `false` | Zezwolenie na osadzony kod Pythona w blokach download w pliku YAML zbioru danych |
| `json` | `false` | Wyjście JSON na stdout |
| `quiet` | `false` | Wyciszenie stderr |
| `verbose` | `true` | Szczegółowe wyjście |
| `help_json` | `false` | Wypisanie schematu polecenia jako JSON i zakończenie |

## Przykłady

<code-tabs name="examples" />

## Uwagi

### Czym są te metryki

Wypisywany zestaw zależy od zadania modelu, a wyjście JSON używa tych samych
kluczy.

Detekcja, segmentacja i obrócone ramki zwracają `mAP50`, `mAP50_95`,
`precision` i `recall`. Gdy model przewiduje więcej niż jeden rodzaj wyjścia,
obok pojawiają się grupy dla poszczególnych rodzajów: `box_metrics`,
`mask_metrics` i `obb_metrics`, każda z tymi samymi czterema kluczami.

Klasyfikacja zwraca `accuracy_top1` i `accuracy_top5`. Detekcja punktowa
zwraca `precision`, `recall`, `f1`, `MLE`, `MAE`, `RMSE` i `mAP_sweep`.
Estymacja głębi zwraca `abs_rel`, `rmse`, `delta1`, `delta2` i `delta3`.
Segmentacja semantyczna zwraca `mIoU` i `pixel_accuracy`. Restauracja obrazu
zwraca `PSNR` i `SSIM`.

Wynik JSON zawiera też `eval_backend`, czyli nazwę i wersję biblioteki
ewaluacyjnej COCO, która wyliczyła te liczby, dzięki czemu przy porównywaniu
dwóch uruchomień wiadomo, czy oba zostały ocenione tym samym backendem.

### Progi

Wartości domyślne są tutaj wartościami dla ewaluacji, a nie dla predykcji:
`conf` wynosi `0.001`, a `iou` `0.6`, podczas gdy
[`libreyolo predict`](/docs/cli/predict) używa `0.25` i `0.45`. Podniesienie
`conf` do progu wyświetlania obniża recall, a wraz z nim mAP, więc liczba
uzyskana w ten sposób nie jest porównywalna z liczbą publikowaną.

`imgsz` domyślnie nie jest ustawiony, co oznacza własny rozmiar wejścia
modelu. Ustawienie go powoduje ewaluację w podanym rozmiarze, i tak właśnie
mierzy się checkpoint poza jego natywną rozdzielczością.

### Zbiory danych, które się pobierają

Plik YAML zbioru danych, którego pole `download` jest adresem URL, pobiera dane
przy pierwszym użyciu bez dodatkowego zezwolenia. Taki, który zawiera osadzony
skrypt pobierający w Pythonie, wymaga `allow_download_scripts=true`, a
polecenie ostrzega na stderr, że wykonywanie lokalnego kodu zostało włączone.
Dołączone pliki `coco8.yaml` i `coco128.yaml` opierają się na adresach URL,
więc nie wymagają niczego.

### Wyjście i kody wyjścia

Metryki trafiają na stdout, a postęp na stderr. `json=true` wypisuje jeden
obiekt z polem `schema_version`, a `quiet=true` wycisza stderr.

Kod wyjścia to `0` przy powodzeniu, `2` przy błędzie użycia lub konfiguracji,
`3` gdy nie można znaleźć zbioru danych, `4` gdy nie można wczytać modelu, i
`1` przy pozostałych błędach w trakcie działania.

Powiązane: [`libreyolo train`](/docs/cli/train), które uruchamia tę samą
ewaluację według własnego harmonogramu przez `eval_interval`.
