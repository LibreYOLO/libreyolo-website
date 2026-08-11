---
title: Trenowanie na wynajętym GPU
seo_title: Trenowanie LibreYOLO na wynajętym GPU w chmurze
description: >-
  Uruchom trenowanie LibreYOLO na wynajętym lub bezserwerowym GPU: przygotuj
  dane, zainstaluj pakiety, uruchom i obserwuj zadanie, pobierz wagi oraz
  zatrzymaj naliczanie opłat.
lead: >-
  Wynajęty GPU zmienia przebieg trenowania w zadanie z początkiem, końcem i
  rachunkiem. Sama praca wygląda tak jak podczas trenowania lokalnego. Zmienia
  się sposób przesyłania danych, zewnętrznego monitorowania, odbierania wag i
  wyłączania maszyny.
keywords:
  - trenowanie na gpu w chmurze
  - wynajem gpu
  - trenowanie vast.ai
  - modal serverless gpu
  - beam gpu
  - zdalne trenowanie modelu
  - hosting zbioru danych hugging face
  - koszt gpu na epokę
last_verified: 1.5.0
snippets:
  install:
    - label: Na maszynie
      language: bash
      code: >
        pip install libreyolo


        # Dodaj tylko dodatki potrzebne w przebiegu: rfdetr do trenowania
        RF-DETR,

        # lora do oszczędnego dostrajania parametrów, onnx do późniejszego
        eksportu.

        pip install "libreyolo[rfdetr,lora]"
    - label: Sprawdzenie GPU przed innymi czynnościami
      language: python
      code: >
        import torch


        print(torch.__version__, torch.cuda.is_available())

        print(torch.cuda.get_device_name(0))


        # Pakiet zbudowany dla innej architektury zgłasza True, a następnie
        ulega awarii

        # przy pierwszym rzeczywistym jądrze, dlatego trzeba jedno uruchomić.

        x = torch.rand(2000, 2000, device="cuda")

        print(float((x @ x).sum()))
  stage:
    - label: Jednorazowe pakowanie i wysyłanie z własnej maszyny
      language: bash
      code: >
        tar cf my-dataset.tar my-dataset/

        huggingface-cli upload my-org/my-dataset my-dataset.tar --repo-type
        dataset
    - label: Przygotowanie na wynajętej maszynie
      language: python
      code: |
        import tarfile

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            "my-org/my-dataset", "my-dataset.tar", repo_type="dataset"
        )
        with tarfile.open(path) as archive:
            archive.extractall("/root/data")
  launch:
    - label: 'Uruchomienie odłączone, odporne na utratę połączenia'
      language: bash
      code: |
        nohup libreyolo train \
          model=LibreYOLO9s.pt \
          data=/root/data/my-dataset/data.yaml \
          epochs=100 batch=-1 imgsz=640 \
          project=/root/runs name=run1 \
          > /root/train.log 2>&1 &
    - label: Wiele GPU z pliku Pythona
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="/root/data/my-dataset/data.yaml",
                epochs=100,
                batch=64,          # globalny batch na wszystkich GPU
                device="0,1,2,3",
                project="/root/runs",
                name="run1",
            )
  watch:
    - label: Jeden niedrogi odczyt
      language: bash
      code: |
        cat /root/runs/run1/status.json
    - label: Ze skryptu
      language: python
      code: |
        import json

        with open("/root/runs/run1/status.json") as handle:
            status = json.load(handle)

        print(status["state"], status["current_epoch"], status["eta_seconds"])
        print(status.get("metrics"))
    - label: W przeglądarce przez tunel SSH
      language: bash
      code: >
        # Na wynajętej maszynie (domyślnie nasłuchuje na 127.0.0.1:8420):

        libreyolo monitor /root/runs/run1 --no-browser


        # Na własnej maszynie, a następnie otwórz lokalnie
        http://localhost:8420:

        #   ssh -L 8420:localhost:8420 <user>@<host>
  push:
    - label: Wysyłanie wag do trwałej lokalizacji
      language: bash
      code: |
        huggingface-cli upload my-org/my-run \
          /root/runs/run1/weights/best.pt best.pt
source_hash: 75d314de06aca3b6
---

## Przed wynajęciem czegokolwiek

Dwie decyzje kosztują znacznie więcej później niż teraz.

Najpierw umieść zbiór danych w CDN. Spakowanie go do jednego archiwum tar w
repozytorium zbioru danych Hugging Face działa tak samo u każdego dostawcy,
zapewnia szybkie pobieranie u wszystkich i dla prywatnego repozytorium wymaga
jedynie `HF_TOKEN` w środowisku zadania. Wysyłanie zbioru danych z łącza domowego
lub pobieranie go na wynajętą maszynę z wolnego źródła oznacza płatny czas
oczekiwania GPU.

<code-tabs name="stage" />

Następnie dobierz rozmiar dysku. Dostawcy pobierający opłaty za pamięć masową
naliczają je według przydzielonej pojemności, a nie faktycznego użycia, natomiast
dysku nie można zmniejszyć po utworzeniu. Zsumuj przygotowane dane i checkpointy,
dodaj około 30 procent zapasu i na tym zakończ.

## Instalacja na maszynie

<code-tabs name="install" />

Jeżeli obraz nie zawiera już kompilacji PyTorch z CUDA pasującej do karty,
najpierw zainstaluj PyTorch, a dopiero potem LibreYOLO. Dzięki temu pip nie
rozwiąże własnego pakietu torch tylko do CPU. Drugi fragment nie jest opcjonalną
formalnością. Pakiet zbudowany dla niewłaściwej architektury GPU zgłasza
`torch.cuda.is_available() == True`, a następnie ulega awarii przy pierwszej
rzeczywistej operacji z komunikatem `CUDA error: no kernel image is available for execution
on the device`. Jedno mnożenie macierzy wykrywa to przed zmarnowaniem godziny na
konfigurację.

Jeśli dostawca oferuje wolumin, ustaw `HF_HOME` na trwałą pamięć masową, aby
pobrane checkpointy i zbiory danych przetrwały między przebiegami.

## Uruchamianie

Uruchom zadanie w trybie odłączonym. Sesja interaktywna przerwana razem z
połączeniem sieciowym zatrzyma także trenowanie.

<code-tabs name="launch" />

W tym zastosowaniu szczególnie warto użyć `batch=-1`, ponieważ zwykle pracuje
się na karcie, na której wcześniej nie trenowano. Opcja sprawdza model w trybie
trenowania z rzeczywistym przebiegiem wstecznym i wybiera największą potęgę
dwójki, która mieści się w pamięci. Jest to szybsze niż znalezienie granicy przez
błąd braku pamięci po dwudziestu minutach. Zobacz
[hiperparametry](/docs/train/hyperparameters).

Na maszynie z wieloma GPU wartość `device="0,1,2,3"` samodzielnie uruchamia
jednego workera na każdy GPU, a `batch` pozostaje globalnym batchem obejmującym
wszystkie urządzenia. Strażnik `__main__` jest obowiązkowy, ponieważ każdy worker
ponownie importuje skrypt. Ten i pozostałe elementy działania rozproszonego
opisano na stronie [trenowanie na wielu GPU](/docs/train/multi-gpu).

## Obserwowanie z zewnątrz

Każdy przebieg zapisuje `status.json` w swoim katalogu i zastępuje go atomowo po
każdej epoce. Jest to niedrogi odczyt kilkuset bajtów zawierających stan, bieżącą
epokę, szacowany czas zakończenia oraz najnowsze metryki, bez analizowania logu.

<code-tabs name="watch" />

Znajdujący się obok plik `metrics.jsonl` zawiera pełną historię każdej epoki, a
`train.log` dane wyjściowe konsoli. `libreyolo monitor` udostępnia pulpit
przeglądarkowy oparty na wszystkich trzech plikach, używając wyłącznie biblioteki
standardowej. Nie wymaga więc instalowania na maszynie niczego poza samym
LibreYOLO. Dostęp można uzyskać przez przekierowanie portu SSH.

Żadne z tych narzędzi nie ingeruje w proces trenowania. Można je dołączyć do
aktywnego przebiegu, ponownie otworzyć zakończony przebieg lub sprawdzić przebieg
zakończony awarią.

## Odbieranie wag przed zatrzymaniem opłat

Maszyna jest jednorazowa. Checkpointy należy wysyłać przy kolejnych kamieniach
milowych, a nie dopiero na końcu, ponieważ awaria, wywłaszczenie zasobów lub
wyczerpanie środków może w przeciwnym razie spowodować utratę całego przebiegu.

<code-tabs name="push" />

Pliki `weights/best.pt` i `weights/last.pt` są zapisywane w każdej epoce i przy
każdej poprawie. `save_period=N` dodaje migawki `weights/epoch_<N>.pt`, dzięki
czemu wysłanie danych w trakcie przebiegu jest niedrogie. `summary.json` i
`results.csv`, jeśli rodzina je zapisuje, są małe i również warto je pobrać.

Callback `on_train_epoch_end` jest właściwym sposobem automatyzacji wysyłania.
Zobacz [loggery eksperymentów](/docs/train/loggers), w których hostowane backendy
udostępniają także metryki bez bezpośredniego dostępu do maszyny.

## Zatrzymywanie opłat

To część, która w razie błędu kosztuje prawdziwe pieniądze, a zasada zależy od
modelu dostawcy.

Na platformie handlowej, gdzie wynajmuje się surową maszynę, rozliczenie trwa
według czasu zegarowego aż do zniszczenia instancji. Bezczynny GPU kosztuje
dokładnie tyle samo co zajęty, więc samo zakończenie procesu trenowania niczego
nie oszczędza. Zatrzymana instancja nadal generuje opłaty za dysk.

Na platformie bezserwerowej, gdzie zadanie jest dekorowaną funkcją, kontener
skaluje się do zera po zakończeniu funkcji, więc pozostawienie zapomnianej
maszyny jest znacznie mniej prawdopodobne. Zawieszone zadanie bez limitu czasu
nadal generuje opłaty, dlatego zawsze należy taki limit ustawić.

Zatrzymanie zamiast zniszczenia jest rzeczywistą metodą ograniczenia kosztów,
ale też pułapką. Pomiar wykonany 2026-07-31 na wynajętej maszynie z 8 GPU RTX
4090 i dyskiem 250 GB wykazał koszt działania 3.4828 USD za godzinę, koszt
zatrzymania 0.0694 USD za godzinę wyłącznie za dysk i brak kosztu po zniszczeniu.
Oznacza to oszczędność 98 procent przy zachowaniu środowiska, przygotowanych
danych i checkpointów.

Stawkę za zatrzymanie można obliczyć przed wynajęciem:

```text
stopped $/hr = allocated_GB * storage_cost_per_GB_per_month / 730
             = 250 * 0.20 / 730 = $0.0694/hr
```

Porównaj ją z kosztem ponownego utworzenia środowiska: kolejnego wynajmu,
pobrania obrazu, instalacji i ponownego przygotowania danych. Na tej samej
maszynie odtworzenie wymagało około 15 minut konfiguracji i 43 GB transferu
przychodzącego, łącznie około 1.00 USD. Przy stawce 0.0694 USD za godzinę powrót
w ciągu około 14 godzin przemawia za zatrzymaniem, a dłuższa przerwa za
zniszczeniem i ponownym utworzeniem z przygotowanej kopii.

Jedno ryzyko sprawia, że zatrzymanie jest niebezpieczne dla rzadko dostępnego
sprzętu: zatrzymanie zwalnia GPU. Nic ich nie rezerwuje, więc ponowne uruchomienie
powiedzie się tylko wtedy, gdy host nadal ma wolne urządzenia. Dysk jest
bezpieczny, GPU nie są.

## Tryb bezserwerowy jako funkcja

Jeśli zarządzanie maszyną nie jest pożądane, Modal i Beam uruchamiają dekorowaną
funkcję Pythona na GPU i skalują ją do zera po zakończeniu. Nocny zestaw testów
samego LibreYOLO działa w Modal, a `tools/ci/modal_nightly.py` w repozytorium
biblioteki jest działającym przykładem wewnątrz repozytorium, który można
skopiować.

```python
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")   # biblioteki systemowe OpenCV
    .pip_install("libreyolo[rfdetr]")
)
app = modal.App("libreyolo-train")
cache = modal.Volume.from_name("libreyolo-cache", create_if_missing=True)


@app.function(gpu="A100", timeout=6 * 60 * 60, volumes={"/cache": cache})
def train():
    import os

    os.environ["HF_HOME"] = "/cache/hf"          # pamięć podręczna wag między przebiegami

    from libreyolo import LibreYOLO

    model = LibreYOLO("LibreYOLO9s.pt")
    model.train(data="coco8.yaml", epochs=100, project="/cache/runs")
    cache.commit()                                # utrwalenie woluminu


@app.local_entrypoint()
def main():
    train.remote()
```

Uruchom go poleceniem `modal run modal_train.py`. System plików kontenera jest
ulotny, dlatego wszystko, co warto zachować, musi trafić do woluminu lub zostać
wysłane na zewnątrz. Ustaw `timeout=` jawnie. Jest to jedyna bariera między
zawieszonym przebiegiem a rachunkiem bez górnego limitu.

Beam ma tę samą strukturę z dekoratorem `@function`, obiektem `Volume` i
`train.remote()` wywołanym z `__main__`.

## Dobór sprzętu według kosztu zadania

Stawka za godzinę jest niewłaściwą liczbą do optymalizacji. Mały model tylko
częściowo wykorzystuje dużą kartę, dlatego tańszy i wolniejszy GPU często jest
tańszy w przeliczeniu na epokę. Przed rozpoczęciem długiego przebiegu uruchom
profiler na kilka kroków na wynajętej karcie. Jeśli werdykt to `dataloader` lub
`host / launch`, szybszy GPU niczego nie poprawi, natomiast większa liczba
workerów lub większy batch da znaczną korzyść. Zobacz
[wydajność trenowania](/docs/train/performance).

## Powiązane strony

- [Zbiory danych](/docs/train/datasets) opisują układ przygotowanego archiwum
  oraz polecenie diagnostyczne wykrywające problemy przed rozpoczęciem płatnego
  czasu GPU.
- [Trenowanie na wielu GPU](/docs/train/multi-gpu) opisuje maszyny z wieloma
  kartami.
