---
title: Loggery eksperymentów
seo_title: Loggery eksperymentów i callbacki w LibreYOLO
description: >-
  Wysyłaj metryki trenowania do TensorBoard, MLflow, Weights & Biases, Comet,
  ClearML, Neptune lub DVCLive i twórz własne callbacki dla czterech hooków
  trenowania.
lead: >-
  Każda rodzina możliwa do trenowania emituje cztery zdarzenia trenowania.
  Wbudowane loggery są obiektami callback nasłuchującymi tych samych zdarzeń,
  dlatego integracja backendu i niestandardowy hook używają jednego interfejsu.
keywords:
  - tensorboard trenowanie
  - śledzenie mlflow
  - weights and biases
  - clearml
  - comet ml
  - neptune
  - dvclive
  - callbacki trenowania
  - metryki trenowania csv
  - libreyolo monitor
last_verified: 1.5.0
snippets:
  logger:
    - label: Według nazwy
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, loggers="tensorboard")
    - label: Skonfigurowana instancja
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import MLflowLogger

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="coco8.yaml",
            epochs=10,
            loggers=[MLflowLogger(tracking_uri="sqlite:///mlflow.db"), "tensorboard"],
        )
  callback:
    - label: Zwykła funkcja
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import TrainEpochEvent


        def on_epoch(event: TrainEpochEvent) -> None:
            print(f"epoch {event.epoch}/{event.total_epochs} loss={event.train_loss:.4f}")


        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, callbacks=on_epoch)
    - label: Obiekt z kilkoma hookami
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.training import TrainEndEvent, TrainEpochEvent,
        TrainStartEvent



        class RunLog:
            def on_train_start(self, event: TrainStartEvent) -> None:
                print(f"{event.model_family}{event.model_size} -> {event.save_dir}")

            def on_train_epoch_end(self, event: TrainEpochEvent) -> None:
                if event.is_best:
                    print(f"new best at epoch {event.epoch}: {event.best_metric}")

            def on_train_end(self, event: TrainEndEvent) -> None:
                print(f"done in {event.total_seconds:.0f}s")


        model = LibreYOLO("LibreYOLO9s.pt")

        model.train(data="coco8.yaml", epochs=10, callbacks=RunLog())
  monitor:
    - label: Obserwowanie przebiegu w przeglądarce
      language: bash
      code: |
        libreyolo monitor                     # najnowszy przebieg w runs/
        libreyolo monitor runs/train/exp      # konkretny przebieg
source_hash: de035acbaed32804
---

## Włączanie loggera

`loggers=` przyjmuje zarejestrowaną nazwę, skonfigurowaną instancję lub element
iterowalny łączący oba warianty.

<code-tabs name="logger" />

W nazwach nie jest rozróżniana wielkość liter. Zarejestrowany zestaw obejmuje
`tensorboard`, `mlflow`, `wandb`, `comet`, `clearml`, `neptune`, `dvclive` i
`dvc`, przy czym ostatnia nazwa jest aliasem `dvclive`. Każda inna wartość
natychmiast zgłasza błąd i wymienia poprawne nazwy. Nie ma wartości włączającej
wszystkie loggery ani flagi CLI. `loggers=` jest argumentem Pythona.

## Dane rejestrowane przez każdy backend

Wszystkie zapisują te same nazwy metryk, dlatego pulpit wygląda tak samo
niezależnie od wyboru:

| Klucz | Wartość |
|---|---|
| `train/loss` | średnia funkcja straty trenowania dla epoki |
| `train/loss/<component>` | każdy składnik funkcji straty raportowany przez rodzinę |
| `lr/<group>` | współczynnik uczenia każdej grupy parametrów optymalizatora |
| `val/<metric>` | każda metryka walidacji bez prefiksu `metrics/` |
| `time/epoch_seconds` | czas zegarowy epoki |

Krok jest numerem epoki liczonym od 1. W pełni rozwiązana konfiguracja
trenowania jest rejestrowana jako parametry na początku trenowania, a domyślna
nazwa przebiegu to `<family><size>-<task>`, na przykład `yolo9s-detect`.

Po zakończeniu trenowania backendy obsługujące artefakty wysyłają `results.csv`,
`train_config.yaml` i `summary.json`, jeśli pliki istnieją, oraz
`weights/best.pt` przy `log_checkpoints=True`. TensorBoard niczego nie wysyła,
ponieważ nie ma pojęcia artefaktu. Żaden logger nie wysyła obrazów wykresów
walidacji.

## Zachowanie w razie awarii

Brakujący pakiet backendu zgłasza błąd podczas tworzenia wraz z nazwą polecenia
instalacji, ponieważ zażądanie loggera i ciche nieotrzymanie żadnych danych
ukrywa błąd.

Awaria backendu podczas przebiegu powoduje odwrotne zachowanie. Pierwszy wyjątek
z handlera wyłącza dany logger do końca przebiegu, zapisuje błąd w logu i kończy
przebieg backendu jako nieudany, ale trenowanie jest kontynuowane. Awaria serwera
śledzenia nie powoduje utraty trenowania.

## Backendy

Każdy wymaga własnego dodatku.

| Nazwa | Dodatek | Konstruktor |
|---|---|---|
| `tensorboard` | `libreyolo[tensorboard]` | `TensorBoardLogger(log_dir=None)` |
| `mlflow` | `libreyolo[mlflow]` | `MLflowLogger(tracking_uri, experiment_name, run_name, log_artifacts=True, log_checkpoints=False)` |
| `wandb` | `libreyolo[wandb]` | `WandbLogger(project, name, entity, log_checkpoints=False)` |
| `comet` | `libreyolo[comet]` | `CometLogger(project_name, workspace, name, api_key, online, log_artifacts=True, log_checkpoints=False)` |
| `clearml` | `libreyolo[clearml]` | `ClearMLLogger(project_name="LibreYOLO", task_name, tags, output_uri, log_artifacts=True, log_checkpoints=False)` |
| `neptune` | `libreyolo[neptune]` | `NeptuneLogger(project, api_token, name, run_id, tags, mode, capture_console=False, log_artifacts=True, log_checkpoints=False)` |
| `dvclive`, `dvc` | `libreyolo[dvclive]` | `DVCLiveLogger(log_dir, resume, report, save_dvc_exp=False, dvcyaml=None, monitor_system=False, log_checkpoints=False)` |

Klasy należy importować z `libreyolo.training`.

Uwagi specyficzne dla backendów, które warto poznać przed pierwszym przebiegiem:

Pliki zdarzeń TensorBoard są domyślnie zapisywane w
`<save_dir>/tensorboard`. Wyświetl je poleceniem
`tensorboard --logdir runs/train`.

MLflow 3.x oznaczył lokalny magazyn plików `./mlruns` jako przestarzały i
zgłasza błąd bez `MLFLOW_ALLOW_FILE_STORE=true`. Do lokalnego śledzenia bez
serwera należy przekazać URI bazy danych, jak w powyższym fragmencie, i odczytać
je przez `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Weights & Biases używa rezerwowo zmiennej środowiskowej `WANDB_PROJECT`, a
następnie wartości `libreyolo`. Comet używa rezerwowo `COMET_PROJECT_NAME`, a
następnie `libreyolo`, i pobiera dane uwierzytelniające z własnej konfiguracji.
`online=False` tworzy eksperyment offline. ClearML tworzy nowe zadanie, raportuje
konfigurację w `TrainConfig` i wyłącza automatyczne przechwytywanie frameworka,
aby metryki nie były raportowane dwukrotnie. Neptune używa aktualnego klienta
`neptune-scale`, a nie starszego pakietu, natomiast `mode="offline"` zapisuje
dane lokalnie.

DVCLive zapisuje dane w `<save_dir>/dvclive`. Buduje swoje drzewo podsumowania
od `/` i nie może przechowywać liczby zmiennoprzecinkowej w ścieżce, która jest
także elementem nadrzędnym. Dlatego `train/loss/box` jest zapisywane jako
`train/loss.box`, podczas gdy `train/loss` zachowuje nazwę. LibreYOLO wyłącza też
zwykłe wartości domyślne DVCLive, które zapisują eksperyment DVC i tworzą główny
plik `dvc.yaml`. Dzięki temu opcjonalny logger nie tworzy stanu kontroli wersji
poza katalogiem przebiegu. Aby je przywrócić, należy przekazać
`save_dvc_exp=True` lub jawne `dvcyaml=`.

Neptune jest celowo wyłączony z `libreyolo[all]`. Jego stabilny klient wymaga
protobuf w wersji niższej niż 7, podczas gdy dodatek TFLite wymaga protobuf 7.
`libreyolo[neptune]` należy zainstalować w środowisku bez dodatku TFLite.

## Tworzenie callbacku

Wszystkimi funkcjami sterują te same cztery zdarzenia.

<code-tabs name="callback" />

| Zdarzenie | Moment | Zawartość |
|---|---|---|
| `TrainStartEvent` | po konfiguracji, przed epoką 1 | `start_epoch`, `total_epochs`, `model_family`, `model_size`, `task`, `save_dir`, `config` |
| `TrainEpochEvent` | po każdej epoce, trenowaniu i walidacji | `epoch`, `train_loss`, `train_loss_items`, `lr`, `val_metrics`, `validated`, `is_best`, `current_metric`, `best_metric`, `best_epoch`, `epoch_seconds` |
| `TrainEndEvent` | po zakończeniu trenowania | `completed_epochs`, `final_loss`, `best_metric`, `best_epoch`, `total_seconds`, `results` |
| `TrainExceptionEvent` | po zgłoszeniu błędu przez trenowanie | `epoch`, `exception`, `exception_type`, `exception_message`, `elapsed_seconds` |

Zwykły obiekt wywoływalny otrzymuje wyłącznie `TrainEpochEvent`. Obiekt może
implementować dowolny podzbiór metod `on_train_start`, `on_train_epoch_end`,
`on_train_end` i `on_train_exception`. Brakujące metody są pomijane.

`TrainStartEvent.config` jest w pełni rozwiązaną konfiguracją, czyli mapowaniem
tylko do odczytu łączącym argumenty nazwane użytkownika z wartościami domyślnymi
rodziny. Zdarzenia są zamrożonymi obiektami dataclass, a ich mapowania są tylko
do odczytu, dlatego callback nie może zmienić przebiegu przez zapis do zdarzenia.

Wyjątek zgłoszony z `on_train_start`, `on_train_epoch_end` lub `on_train_end`
jest przekazywany dalej i kończy przebieg. Chronione jest tylko
`on_train_exception`, dzięki czemu nie może ukryć pierwotnej awarii.

Podczas trenowania na wielu GPU callbacki są wywoływane wyłącznie w procesie o
randze 0. Przy automatycznym uruchamianiu DDP muszą też nadawać się do
serializacji, co oznacza klasę lub funkcję na poziomie modułu zamiast domknięcia
albo lambdy. Zobacz [trenowanie na wielu GPU](/docs/train/multi-gpu).

## Pliki zapisywane przez każdy przebieg

Trzy pliki trafiają do katalogu przebiegu bez jakiejkolwiek konfiguracji, w
każdej rodzinie:

| Plik | Moment zapisu | Zawartość |
|---|---|---|
| `status.json` | atomowo, w każdej epoce oraz przy rozpoczęciu, zakończeniu i awarii | `state` o wartości `running`, `completed` lub `failed`, `current_epoch`, `total_epochs`, `progress`, `eta_seconds`, najnowsze `metrics`, `best_metric`, `best_epoch` oraz obiekt `error` przy awarii |
| `metrics.jsonl` | jeden wpis po każdej epoce | jeden wiersz JSON na epokę, ten sam schemat co `results.csv` |
| `train.log` | na żywo | dane wyjściowe konsoli przebiegu |

`status.json` jest niedrogim odczytem dla skryptu lub agenta odpytującego
przebieg, a zapis atomowy oznacza, że czytnik nigdy nie zobaczy częściowo
zapisanego pliku.

`results.csv` i `summary.json` są oddzielne i zależne od rodziny. Są zapisywane
dla YOLOv9, YOLOv9-E2E, YOLOv9-P2, YOLOv7, YOLO-NAS, RF-DETR, EC i DINOv2, ale
nie dla pozostałych rodzin. `results.csv` otrzymuje jeden wiersz na epokę ze
składnikami funkcji straty, metrykami walidacji i współczynnikami uczenia jako
kolumnami, a jego nagłówek rozszerza się po pojawieniu nowej kolumny. Przy
wznowieniu plik jest przycinany do wierszy sprzed wznawianej epoki, zamiast je
duplikować.

Oprócz nich podczas konfiguracji trener zawsze zapisuje `train_config.yaml` oraz
checkpointy w `weights/`.

## Obserwowanie przebiegu na żywo

<code-tabs name="monitor" />

`libreyolo monitor` udostępnia w przeglądarce pulpit oparty na powyższych plikach
i wyłącznie bibliotece standardowej. Pokazuje wykresy metryk, końcówkę logu i
wszystkie obrazy walidacji, odświeżając się podczas aktywnego przebiegu. Działa
tylko do odczytu i nigdy nie ingeruje w proces trenowania, dlatego może dołączyć
do aktywnego przebiegu, ponownie otworzyć zakończony przebieg lub sprawdzić
przebieg zakończony awarią.

## Powiązane strony

- [Walidacja i metryki](/docs/train/validation) opisują znaczenie kluczy `val/`
  oraz sposób dodawania funkcji straty walidacji.
- [Wydajność trenowania](/docs/train/performance) opisuje profiler, który jest
  innym narzędziem odpowiadającym na inne pytanie.

