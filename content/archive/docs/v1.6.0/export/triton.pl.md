---
title: Triton Inference Server
seo_title: Serwowanie modelu LibreYOLO na NVIDIA Triton
description: >-
  Serwowanie eksportu ONNX z LibreYOLO przez NVIDIA Triton: układ repozytorium
  modeli, generowany config.pbtxt i predykcja pod adresem URL modelu po HTTP.
lead: >-
  Triton Inference Server hostuje repozytorium modeli i odpowiada na żądania
  inferencji po HTTP. LibreYOLO eksportuje graf ONNX, generuje config.pbtxt,
  który przenosi metadane eksportu jako jeden parametr Triton, i traktuje adres
  URL modelu jako wczytywalną ścieżkę modelu.
keywords:
  - libreyolo triton
  - triton inference server
  - config.pbtxt
  - tritonclient http
  - repozytorium modeli triton
  - zdalna inferencja yolo
last_verified: 1.5.0
meta:
  - label: Wywołanie
    value: 'LibreYOLO("http://127.0.0.1:8000/yolo9")'
    mono: true
  - label: Funkcja pomocnicza
    value: >-
      create_triton_config(onnx_path, config_path, model_name=...,
      max_batch_size=8)
    mono: true
  - label: Extra
    value: 'pip install "libreyolo[onnx,triton]"'
    mono: true
  - label: Protokół
    value: >-
      Tylko inferencja HTTP i HTTPS V2. Bez gRPC, uwierzytelniania, pamięci
      współdzielonej oraz wczytywania i wyładowywania modeli.
  - label: Limity czasu
    value: Limity czasu połączenia i sieci domyślnie wynoszą 30 sekund
verification: >-
  Odczytane z libreyolo/backends/triton.py, libreyolo/models/__init__.py,
  docs/triton.md i pyproject.toml na gałęzi dev. Polecenia kontenera są tymi
  przypiętymi w docs/triton.md.
snippets:
  install:
    - label: Instalacja
      language: bash
      code: |
        pip install "libreyolo[onnx,triton]"
  repo:
    - label: Eksport do układu repozytorium
      language: python
      code: |
        from pathlib import Path

        from libreyolo import LibreYOLO

        model_dir = Path("triton_repo/yolo9/1")
        model_dir.mkdir(parents=True, exist_ok=True)

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            output_path=str(model_dir / "model.onnx"),
            dynamic=True,
            simplify=False,
        )
    - label: Generowanie config.pbtxt
      language: python
      code: |
        from libreyolo import create_triton_config

        create_triton_config(
            "triton_repo/yolo9/1/model.onnx",
            "triton_repo/yolo9/config.pbtxt",
            model_name="yolo9",
            max_batch_size=8,
        )
    - label: Wynikowy układ
      language: text
      code: |
        triton_repo/
          yolo9/
            config.pbtxt
            1/
              model.onnx
  serve:
    - label: Uruchomienie serwera
      language: bash
      code: |
        docker run --rm --name libreyolo-triton \
          -p 8000:8000 -p 8002:8002 \
          -v "$(pwd)/triton_repo:/models:ro" \
          nvcr.io/nvidia/tritonserver:26.04-py3 \
          tritonserver --model-repository=/models --exit-on-error=true
    - label: Oczekiwanie na gotowość
      language: bash
      code: >
        until curl --fail --silent http://127.0.0.1:8000/v2/health/ready; do
        sleep 1; done
    - label: Zatrzymanie
      language: bash
      code: |
        docker stop libreyolo-triton
  run:
    - label: Predykcja na serwowanym modelu
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9")
        result = remote.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Porównanie z modelem lokalnym
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9").predict(SAMPLE_IMAGE)
        native = LibreYOLO("LibreYOLO9t.pt").predict(SAMPLE_IMAGE)

        print(len(remote.boxes), len(native.boxes))
        print(remote.boxes.xyxy[:3])
        print(native.boxes.xyxy[:3])
    - label: Przypięcie wersji lub zmiana limitu czasu
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.backends.triton import TritonBackend

        # Drugi segment ścieżki wybiera wersję modelu. Bez niego
        # decyduje skonfigurowana w Triton polityka wersji.
        pinned = LibreYOLO("http://127.0.0.1:8000/yolo9/1")

        # Limity czasu połączenia i sieci domyślnie wynoszą 30 sekund.
        patient = TritonBackend("http://127.0.0.1:8000/yolo9", timeout=120)
source_hash: 0652e4faf0224df3
---

## Instalacja

<code-tabs name="install" />

Extra `triton` instaluje `tritonclient[http]`. Extra dla gRPC i pamięci
współdzielonej są celowo pominięte: ta integracja obsługuje wyłącznie inferencję
HTTP i HTTPS V2. `onnx` jest potrzebny, ponieważ zarówno serwowany artefakt, jak
i generator konfiguracji pracują na grafie ONNX.

## Budowa repozytorium modeli

Eksport z dynamiczną osią batcha, do układu katalogów oczekiwanego przez Triton.

<code-tabs name="repo" />

Triton nie zachowuje niestandardowych metadanych ONNX w odpowiedzi z konfiguracją
modelu, więc komplet wyeksportowanych metadanych musi dotrzeć inną drogą.
`create_triton_config` koduje je jako jeden parametr tekstowy JSON o nazwie
`libreyolo_metadata` w `config.pbtxt`, wypisuje deklaracje wejść i wyjść w
kolejności z grafu, obsługuje escapowanie JSON i przypina model do `KIND_CPU`.

Funkcja pomocnicza waliduje dane przed zapisem. Wymaga dokładnie jednego wejścia
grafu ONNX, co najmniej jednego wyjścia, rozstrzygalnych kształtów tensorów oraz
metadanych, w których mapa `names` definiuje każdy indeks klasy od 0 do `nc - 1`.
Model, który nie przejdzie któregokolwiek z tych sprawdzeń, zostaje odrzucony na
etapie konfiguracji, a nie przy pierwszym żądaniu.

`max_batch_size: 8` odpowiada eksportowi dynamicznemu i pozwala serwerowi grupować
do ośmiu obrazów na żądanie. Dla grafu ONNX ze stałym batchem 1 należy użyć
`max_batch_size=0`; LibreYOLO wysyła wtedy obrazy sekwencyjnie.

## Uruchomienie serwera

<code-tabs name="serve" />

Polecenia przypinają Triton Server 26.04 i celowo pomijają flagi GPU dla Dockera,
ponieważ `KIND_CPU` w wygenerowanej konfiguracji i tak blokuje umieszczenie na GPU.

## Uruchomienie artefaktu

Adres URL modelu w Triton jest ścieżką modelu. `LibreYOLO()` sprawdza schemat
`http` lub `https` przed jakąkolwiek obsługą ścieżek lokalnych i zwraca backend
komunikujący się z serwerem, więc miejsce wywołania jest identyczne jak dla
lokalnego checkpointu, podobnie jak zwracany obiekt `Results`.

<code-tabs name="run" />

Format adresu to `http(s)://host:port/model` z opcjonalnym segmentem wersji. Port
musi być podany wprost. Osadzone dane uwierzytelniające, ciąg zapytania i fragment
są odrzucane, podobnie jak ścieżka z więcej niż dwoma segmentami.

`device` jest przyjmowany i ignorowany z wpisem w logu, ponieważ o umieszczeniu
decyduje serwer.

## Ograniczenia

Backend zgłasza wprost błąd zamiast zwracać pogorszony wynik, gdy kontrakt nie
jest spełniony: brak metadanych LibreYOLO w konfiguracji modelu, więcej niż jedno
wejście modelu, niezgodność między skonfigurowanymi wyjściami a metadanymi modelu,
nieobsługiwany typ danych wejściowych albo serwer lub model, który nie jest gotowy.

Poza kontraktem w tej wersji pozostają: gRPC, uwierzytelnianie, pamięć
współdzielona oraz wczytywanie i wyładowywanie modeli przez API.

Serwować można każdy format obsługiwany przez sam Triton, ale parametr z
metadanymi i wygenerowana konfiguracja są tutaj dopasowane do ONNX, więc ścieżką
LibreYOLO jest [ONNX](/docs/export/onnx) do repozytorium. Dla pełnego pipeline'u
wideo zamiast serwera typu żądanie-odpowiedź zobacz
[DeepStream](/docs/export/deepstream).
