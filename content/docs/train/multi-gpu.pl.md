---
title: Trenowanie na wielu GPU
seo_title: Trenowanie na wielu GPU w LibreYOLO
description: >-
  Trenowanie na kilku GPU z device="0,1". Jak biblioteka uruchamia procesy
  robocze DDP, dlaczego batch jest globalny, kiedy ustawić sync_bn i jak użyć
  torchrun.
lead: >-
  Trenowanie na wielu GPU w LibreYOLO korzysta z PyTorch
  DistributedDataParallel: na każdy GPU przypada jeden proces z pełną repliką
  modelu i fragmentem każdego batcha, a gradienty są uśredniane między rangami w
  każdym kroku.
keywords:
  - trenowanie pytorch ddp
  - trenowanie na wielu gpu
  - torchrun nproc_per_node
  - distributed data parallel pytorch
  - syncbatchnorm
  - globalny rozmiar batcha
  - backend nccl gloo
  - wiele gpu windows
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Osłona __main__ jest wymagana: każdy uruchomiony proces roboczy
        ponownie importuje ten

        # moduł, a bez osłony trenowanie zostałoby ponownie uruchomione
        rekurencyjnie.

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="my-dataset.yaml",
                epochs=100,
                batch=32,     # globalny batch: 16 obrazów na GPU przy dwóch GPU
                device="0,1",
            )
  torchrun:
    - label: train.py
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(data="my-dataset.yaml", epochs=100, batch=32)
    - label: Uruchomienie
      language: bash
      code: |
        torchrun --nproc_per_node=2 train.py
  syncbn:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreRTDETRr18.pt")
            model.train(
                data="my-dataset.yaml",
                batch=32,
                device="0,1",
                sync_bn=True,
            )
  autobatch:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            # Pomiar wykonywany raz na GPU 0, wynik skalowany do wielokrotności liczby procesów.
            model.train(data="my-dataset.yaml", batch=-1, device="0,1")
source_hash: 83c1563d68068cd0
---

## Uruchamianie na dwóch GPU

Przekaż listę urządzeń. Nic więcej się nie zmienia.

<code-tabs name="train" />

Gdy podano więcej niż jedno urządzenie i nie ma środowiska torchrun, metoda
`train()` modelu zapisuje wagi do pliku tymczasowego, w razie potrzeby ustala
autobatch i uruchamia po jednym procesie roboczym na każdy GPU za pomocą
`torch.multiprocessing.spawn`. Każdy proces roboczy ponownie importuje klasę
modelu, odtwarza go z zapisanych wag i wykonuje zwykłą ścieżkę dla jednego
urządzenia, ponieważ wewnątrz uruchomionego procesu ustawione są zmienne
środowiskowe torchrun. Po zakończeniu trenowania najlepszy checkpoint z rangi 0
jest wczytywany z powrotem do instancji modelu wywołującej metodę.

`device` przyjmuje wartości `"0,1"`, `[0, 1]`, `0`, `"cuda:0"`, `"cpu"`, `"mps"`
i `"auto"`. Tylko lista zawierająca więcej niż jeden indeks CUDA uruchamia
procesy.

## Osłona `__main__` jest obowiązkowa

Uruchomione procesy robocze ponownie importują moduł, z którego pochodzą. Bez
osłony `if __name__ == "__main__":` import ponownie wykonuje wywołanie
trenowania, a każdy proces roboczy uruchamia własne procesy. Biblioteka wykrywa
tę sytuację i zgłasza błąd, zamiast dopuścić do rekurencji:

```text
spawn_ddp_train() was called from inside a spawned subprocess. This usually
means your script calls model.train(device=...) at the top level without a
'if __name__ == "__main__":' guard.
```

Wszystko przekazywane do procesu roboczego jest serializowane przez pickle,
dlatego `callbacks=` musi nadawać się do takiej serializacji. Klasa zdefiniowana
na poziomie modułu działa, ale domknięcie ani lambda nie. Komunikat błędu to
wyjaśnia i wskazuje wbudowane loggery jako rozwiązanie alternatywne.

## `batch` jest globalnym batchem

`batch` to liczba obrazów przypadających na krok optymalizatora na wszystkich
GPU. Dataloader każdej rangi jest tworzony z `batch // world_size` i
`DistributedSampler`, więc `batch=32` na dwóch GPU oznacza 16 obrazów na GPU, a
nie 32.

Batch, którego rozmiar nie dzieli się bez reszty przez liczbę procesów, powoduje
błąd, zamiast po cichu uruchomić trenowanie z innym rozmiarem:

```text
batch=6 is the global batch and must be divisible by world_size=4: each rank
trains at batch // world_size, so this value would silently train at a
different global batch than requested. Use batch=4 or batch=8.
```

Gradienty są uśredniane przez samo DDP, dlatego funkcja straty jest przekazywana
bez skalowania. Dodatkowe pomnożenie jej przez liczbę procesów zwiększyłoby
efektywny współczynnik uczenia mniej więcej tyle razy, ile jest GPU.

## Autobatch w DDP

`batch=-1` działa i zwraca globalny batch podzielny przez liczbę procesów.

<code-tabs name="autobatch" />

Na ścieżce automatycznego uruchamiania pomiar odbywa się w procesie nadrzędnym
na pierwszym urządzeniu, zanim powstanie jakikolwiek proces roboczy. Dzięki temu
każdy proces otrzymuje konkretną liczbę całkowitą i nie jest potrzebna
koordynacja między procesami. W środowisku torchrun ranga 0 wykonuje pomiar i
rozsyła wynik jako pojedynczy tensor typu long.

Pomiar określa pojemność jednego GPU i mnoży ją przez liczbę procesów. Gdy
ustawiono `nbs`, globalny batch jest ograniczany do `nbs` i zaokrąglany w dół do
wielokrotności liczby procesów. Dodanie GPU zmniejsza więc liczbę kroków
akumulacji, zamiast zmniejszać batch przypadający na GPU. Mechanikę samego
pomiaru opisano w sekcji [Hiperparametry](/docs/train/hyperparameters).

## SyncBatchNorm

W DDP warstwy BatchNorm każdej rangi widzą tylko jej własny fragment danych.
Przy `batch // world_size` fragment ten może być na tyle mały, że statystyki
bieżące pogorszą zbieżny model względem trenowania na jednym GPU.

`sync_bn=True` przekształca każdą warstwę BatchNorm w SyncBatchNorm, aby
statystyki były obliczane dla globalnego batcha. Konwersja następuje wyłącznie
przy aktywnym trenowaniu rozproszonym, więc ta flaga nie wpływa na trenowanie na
jednym GPU.

Jest już domyślnie włączona dla rodzin konwolucyjnych intensywnie korzystających
z BatchNorm: YOLOX, YOLOv7, YOLOv9 i jego wariantów, YOLO-NAS, PicoDet, RTMDet
oraz FOMO. Dla każdej innej rodziny jest domyślnie wyłączona. Jeśli model zawiera
BatchNorm, `sync_bn` jest wyłączone, a batch na rangę jest mniejszy niż 16,
trener wyświetla ostrzeżenie.

<code-tabs name="syncbn" />

Dla `sync_bn` nie ma flagi CLI. Jest to argument Pythona.

## Uruchamianie za pomocą torchrun

torchrun również działa i jest właściwym wyborem, gdy harmonogram klastra już
zarządza uruchamianiem procesów. Napisz skrypt dla jednego urządzenia i pozwól,
aby torchrun ustawił środowisko rang.

<code-tabs name="torchrun" />

Nie łącz obu metod. Przy obecnym środowisku torchrun ustawienie `device="0,1"`
nie uruchamia procesów. Trener wybiera `cuda:LOCAL_RANK`, a torchrun zarządza
liczbą procesów.

## Zachowanie rang

Ranga 0 odpowiada za wszystkie efekty uboczne. Ustala katalog uruchomienia i
rozsyła jego ostateczną nazwę, aby wszystkie rangi były zgodne, zapisuje
checkpointy i artefakty oraz wywołuje callbacki użytkownika i loggery. Pozostałe
rangi trenują i dostarczają gradienty.

Każda ranga inicjalizuje swój dataloader i generator liczb losowych augmentacji
inną wartością wyprowadzoną ze skonfigurowanego `seed`, dzięki czemu rangi nie
losują identycznych augmentacji.

## Platforma i backend

Backend jest wybierany automatycznie: NCCL, gdy dostępne są zarówno CUDA, jak i
NCCL, a w przeciwnym razie Gloo. NCCL nie jest kompilowane w systemie Windows,
więc trenowanie w systemie Windows korzysta z Gloo bez dodatkowej konfiguracji.
Grupa procesów jest inicjalizowana z limitem czasu wynoszącym trzy godziny.

## Funkcje niedostępne w DDP

- Przechwytywanie grafu CUDA. `cuda_graph=True` zapisuje jeden komunikat i
  trenuje w trybie eager. Zobacz [Wydajność trenowania](/docs/train/performance).
- Profiler trenowania. `profile=True` jest ignorowane z ostrzeżeniem.

Nie każda rodzina obsługuje automatyczne uruchamianie. Obsługują je dwadzieścia
cztery rodziny, obejmujące trenujące rodziny do detekcji, klasyfikacji,
segmentacji semantycznej i rekonstrukcji. Przekazanie wielu GPU rodzinie, która
nie ma tej obsługi, powoduje błąd podający API modelu i polecenie torchrun,
zamiast po cichu trenować na jednym GPU.

## Powiązane

- [Hiperparametry](/docs/train/hyperparameters) dotyczące `batch`, `nbs` i
  wznawiania.
- [Loggery eksperymentów](/docs/train/loggers) dotyczące wymogu serializacji
  callbacków przez pickle.
- [GPU w chmurze](/docs/train/cloud-gpus) dotyczące wynajmowania maszyny z
  wieloma GPU.
