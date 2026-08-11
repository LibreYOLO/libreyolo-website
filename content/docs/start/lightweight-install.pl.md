---
title: Lekka instalacja
seo_title: Inferencja LibreYOLO w ONNX bez PyTorch
description: >-
  Zainstaluj LibreYOLO z --no-deps i uruchom detekcję ONNX wyłącznie z numpy,
  bez torch na dysku. Poznaj tę technikę, jej ograniczenia i dokładną listę
  pakietów.
lead: >-
  Ścieżka inferencji ONNX w LibreYOLO korzysta od początku do końca z numpy,
  łącznie z dekodowaniem i NMS. W czasie działania nie wymaga PyTorch, dlatego
  instalacja pomijająca rozwiązywanie zależności może wykonywać detekcję bez
  torch na komputerze.
keywords:
  - inferencja bez torch
  - libreyolo bez pytorch
  - onnx inference bez torch
  - lekka instalacja libreyolo
  - pip install no-deps
  - rozmiar libreyolo na dysku
  - onnxruntime inferencja
last_verified: 1.5.0
meta:
  - label: Dotyczy
    value: 'Detekcja ONNX, siedem rodzin modeli'
  - label: Punkt wejścia
    value: libreyolo.backends.onnx.OnnxBackend
    mono: true
  - label: Poziom wsparcia
    value: 'Wsparcie w miarę możliwości, nie jest to osobna dystrybucja'
snippets:
  install:
    - label: Lekka instalacja
      language: bash
      code: |
        # Zainstaluj pakiet bez jego listy zależności, a następnie dodaj cztery
        # pakiety faktycznie importowane przez ścieżkę detekcji ONNX.
        pip install --no-deps libreyolo
        pip install numpy pillow opencv-python-headless onnxruntime
    - label: torch tylko dla CPU
      language: bash
      code: >
        # Najpierw wypróbuj tę opcję. Zachowuje wszystkie funkcje i pozwala
        uniknąć

        # pakietu wheel CUDA, który zajmuje najwięcej miejsca na dysku.

        pip install libreyolo --index-url https://download.pytorch.org/whl/cpu
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo.backends.onnx import OnnxBackend


        model = OnnxBackend("libreyolo9t.onnx")

        result =
        model.predict("https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg")


        # W tym miejscu xyxy jest tablicą ndarray z numpy, a nie tensorem torch.

        print(result.boxes.xyxy)

        print(result.boxes.conf)

        print(result.boxes.cls)
source_hash: e60e83d32d13026e
---

## Dlaczego to działa

Polecenie `pip install --no-deps libreyolo` instaluje pakiet i całkowicie pomija
jego listę zależności. Żadne zależności nie są rozwiązywane automatycznie, więc
trzeba samodzielnie zainstalować te, które będą rzeczywiście używane.

Ma to sens tylko wtedy, gdy wybrana ścieżka kodu naprawdę nie potrzebuje
pominiętych zależności. Ścieżka detekcji ONNX ich nie potrzebuje. Dekodowanie,
w tym non-maximum suppression, odbywa się w numpy. Procedury przetwarzania
wstępnego także korzystają z numpy. PyTorch jest zależnością wymaganą do
trenowania i inferencji w trybie eager, a na tej ścieżce nie jest wywoływany.

Przed tym wydaniem import mimo wszystko kończył się niepowodzeniem. Import
czegokolwiek z `libreyolo.models` budował każdą klasę modelu, aby wypełnić
rejestr automatycznego wykrywania checkpointów, a klasy te są podklasami
`torch.nn.Module`. Procedury przetwarzania wstępnego znajdują się teraz we
własnym pakiecie `libreyolo.preprocess`, a import torch jest odraczany do chwili
użycia atrybutu torch. Dzięki temu ścieżkę ONNX można zaimportować bez torch na
komputerze. Pakiet zawiera natywny dla numpy moduł przetwarzania wstępnego dla
każdej z rodzin: `yolo9`, `yolonas`, `yolox`, `ec`, `rtdetr`, `rfdetr`, `dfine`,
`deim` i `deimv2`, czyli o dwie więcej niż siedem rodzin zweryfikowanych poniżej
od początku do końca. Każdy plik `libreyolo/models/<family>/utils.py` ponownie
eksportuje elementy z tego pakietu, dzięki czemu istniejące ścieżki importu
nadal działają.

## Najpierw wypróbuj pakiet wheel tylko dla CPU

Większość osób zainteresowanych tym rozwiązaniem chce uniknąć instalacji
zajmującej kilka gigabajtów, a rozmiar skupia się w jednym miejscu: domyślny
pakiet wheel `torch` zawiera CUDA. Kompilacja tylko dla CPU zajmuje ułamek tego
miejsca i nie wymaga specjalnej ścieżki instalacji.

<code-tabs name="install" />

Opcja tylko dla CPU zachowuje wszystkie funkcje LibreYOLO: trenowanie,
walidację, każde zadanie, każdą rodzinę i CLI. Z lekkiej ścieżki warto korzystać,
gdy na komputerze nie ma być w ogóle torch, a nie tylko jego mniejsza wersja.

## Zakres lekkiej instalacji

| | |
|---|---|
| Zadanie | Detekcja |
| Format | ONNX |
| Punkt wejścia | `OnnxBackend` |
| Interfejs | Biblioteka Pythona |

Na tej ścieżce zweryfikowano siedem rodzin: [YOLOv9](/docs/models/yolov9),
[YOLO-NAS](/docs/models/yolo-nas), [EdgeCrafter](/docs/models/edgecrafter),
[RT-DETR](/docs/models/rt-detr), [RF-DETR](/docs/models/rf-detr),
[D-FINE](/docs/models/d-fine) oraz [DEIM](/docs/models/deim), wliczając w to
warianty każdej rodziny.

Jest to zweryfikowany zakres, a nie ograniczenie wymuszane przez bibliotekę.
Inne zadania i rodziny znajdują się po prostu poza zakresem testów. Niektóre
zaimportują torch po wywołaniu, a kilka może przypadkiem działać. Wszystko spoza
tej listy należy traktować jako nieprzetestowane, a nie jako obsługiwane lub
zepsute.

W ramach tego zakresu wyniki są identyczne z normalną instalacją, a nie tylko
zbliżone. Każdą rodzinę wyeksportowano do ONNX i uruchomiono dwukrotnie, raz
normalnie i raz z zablokowanym torch. Ramki, wskaźniki i klasy były dokładnie
zgodne. Test zgodności w zestawie testów chroni ten kontrakt przed zmianami.

## Pięć częstych pułapek

**Użyj `OnnxBackend`, a nie klas modeli.** Wywołanie
`LibreYOLO9("model.onnx")` nadal wymaga torch, ponieważ `LibreYOLO9` jest
podklasą `nn.Module`. To najbardziej prawdopodobna pomyłka, gdyż na każdej
innej stronie tej dokumentacji model wczytuje się przez jego klasę lub przez
`LibreYOLO()`.

**Eksport wykonaj gdzie indziej.** Utworzenie pliku `.onnx` wymaga torch, więc
lekki komputer nie może go wygenerować. Eksport należy wykonać na komputerze
deweloperskim lub w CI, a artefakt dostarczyć do odchudzonego środowiska
docelowego.

**Wyniki zawierają tablice numpy.** W tym przypadku `result.boxes.xyxy` jest
obiektem `ndarray`. Kontenery przyjmują oba typy, dlatego nazwy atrybutów się nie
zmieniają, ale kod wywołujący `.cpu()` lub `.numpy()` na wyniku zakończy się
błędem.

**Jeden obraz zwraca pojedynczy obiekt `Results`.** Funkcja `predict()` zwraca
jeden obiekt `Results` dla jednego obrazu i listę dla kilku obrazów.
Indeksowanie pojedynczego wyniku przez `[0]` wybiera pierwszą detekcję, a nie
pierwszy obraz. Zamiast zgłosić wyjątek, zwraca po cichu wynik z jedną ramką.

**CLI nie będzie działać.** Cztery podane pakiety nie obejmują `typer` ani
`click`, więc polecenie `libreyolo` jest niedostępne. Jest to instalacja
biblioteki.

## Predykcja

<code-tabs name="predict" />

Aby użyć CUDA, zamień `onnxruntime` na `onnxruntime-gpu`. Cztery wymienione
pakiety są tymi, które faktycznie importuje pełne wywołanie `predict()` bez
torch. Zarejestrowano je podczas wywołania, zamiast wyznaczać na podstawie
analizy. Pakiet `opencv-python-headless` zastępuje zadeklarowany
`opencv-python`: udostępnia ten sam moduł bez bibliotek GUI i zajmuje mniej
miejsca na dysku.

Spośród pozostałych zadeklarowanych zależności `requests` jest potrzebny tylko
do wczytywania obrazu z adresu URL, `pycocotools` i `scipy` służą do walidacji i
oceny, a `typer` i `click` są wymagane przez CLI.

## Ta lista z założenia będzie się zmieniać

Powyższa lista pakietów jest poprawna dla wydania podanego u góry tej strony.
Opcja `--no-deps` wyłącza rozwiązywanie zależności, więc nic nie sprawdza ich
automatycznie, a późniejsze wydanie może importować pakiet, którego tutaj nie
wymieniono.

Jeśli pojawi się `ModuleNotFoundError`, rozwiązanie wynika bezpośrednio z tej
techniki: należy zainstalować brakujący pakiet. Jest to zamierzony model
utrzymania, a nie błąd do zgłoszenia. Ta ścieżka jest wspierana w miarę
możliwości i nie stanowi osobnej obsługiwanej dystrybucji. Z tego powodu nie ma
drugiego lekkiego pakietu na PyPI ani planów jego utworzenia.

Aby upewnić się, że środowisko naprawdę nie zawiera torch i nie korzysta po
cichu z zainstalowanej kopii, użyj asercji:

```python
import importlib.util

assert importlib.util.find_spec("torch") is None, "torch is installed"
```

Warto zachować to sprawdzenie w CI dla odchudzonego obrazu. Bez niego
środowisko, które przypadkiem ma torch, przejdzie każdy test i nie dostarczy
żadnej użytecznej informacji.
