---
title: Poziomy stabilności
seo_title: Znaczenie poziomów obsługi LibreYOLO
description: >-
  Słownictwo poziomów używane przez LibreYOLO: trzy poziomy obsługi eksportu,
  cztery poziomy API, sześć grup pokrycia oraz to, czego żaden z nich nie
  gwarantuje.
lead: >-
  LibreYOLO używa słowa poziom w trzech różnych znaczeniach: dowodów stojących
  za ścieżką eksportu, kontraktu wywołania obsługiwanego przez rodzinę modeli
  oraz grupy pokrycia, do której zapisano rodzinę. Ta strona definiuje każde z
  nich i wyjaśnia, czego nie oznacza.
keywords:
  - poziomy obsługi LibreYOLO
  - validated available blocked
  - poziomy obsługi eksportu
  - grupy pokrycia LibreYOLO
  - g0 g1 g2 g3 g4
  - poziomy modeli
last_verified: 1.5.0
verification: >-
  Poziomy eksportu pochodzą z docs/adr/0011-export-support-tiers.md i
  libreyolo/export/support.py; grupy pokrycia i liczby rodzin z MODEL_GROUPS w
  libreyolo/models/registry.py; bramka trenowania od zera z
  libreyolo/models/base/model.py i libreyolo/cli/commands/train.py; inwentarz
  CLI z libreyolo/models/inventory.py; poziomy API z docstringów pakietów
  libreyolo/models/sam/, openvocab/ i vlm/ oraz kontraktów base.py, wszystko w
  wersji 1.5.0. Etykiety grup widoczne dla czytelnika (Flagship, Core,
  Supported, Inference only, Museum, Sibling tier) są słownictwem tej witryny
  dla tych samych grup, pochodzącym z src/data/docs/registry.json.
snippets:
  usage:
    - label: Odczyt obu klasyfikacji jednej rodziny
      language: python
      code: |
        from libreyolo.models.registry import GROUPS, group_of
        from libreyolo.export.support import get_support, validated_alternatives

        family = "yolo9"

        group = group_of(family)
        print(group, GROUPS[group])

        print(get_support(family, "detect", "onnx").tier)
        print(validated_alternatives(family, "detect"))
source_hash: de545894b0d125e4
---

## Poziomy obsługi eksportu

Poziom decydujący o powodzeniu wywołania. Dotyczy trójki
`(family, task, format)`, a każda kombinacja ma dokładnie jeden poziom.

| Poziom | Znaczenie | Zachowanie `export()` |
|---|---|---|
| `validated` | Zgodność numeryczna jest objęta CI lub udokumentowanym przebiegiem nocnym | Działa |
| `available` | Konwersja jest zaimplementowana, ale nie zapisano dowodów zgodności numerycznej w środowisku uruchomieniowym | Działa |
| `blocked` | Brak obsługiwanej ścieżki | Zgłasza `NotImplementedError` z przyczyną podczas kontroli wstępnej |

Zarówno validated, jak i available przechodzą bez potwierdzenia ani ogólnego
ostrzeżenia. Różnica dotyczy dowodów, a nie uprawnień: za wpisem validated stoi
test zgodności i wydanie `since`, a available jeszcze ich nie ma. Na przykład
konwersja CoreML bez przebiegu predykcji na macOS jest available, a nie
validated.

Zablokowana kombinacja kończy się niepowodzeniem przed sprawdzaniem zależności,
wczytywaniem kalibracji, śledzeniem lub utworzeniem artefaktu, więc nie jest
zapisywany żaden niepełny wynik.

Każda komórka validated zawiera ograniczenie opisujące konfigurację, z której
pochodzi wynik zgodności. Zwykle jest to stały obszar wejścia, batch 1, FP32
i nazwana wersja środowiska uruchomieniowego. Należy odczytywać je jako
stwierdzenie o tej konfiguracji, a nie o całym formacie. Reguły wypełniające
komórki bez jawnego wpisu opisano na stronie
[macierzy eksportu](/docs/reference/export-matrix).

<code-tabs name="usage" />

## Poziomy API

Poziom decydujący o postaci wywołania. Rodzina należy dokładnie do jednego
poziomu wybranego według kontraktu wywołania, a nie architektury.

| Poziom | Fabryka | Kontrakt |
|---|---|---|
| Fabryka detektorów | `LibreYOLO` | Jeden przebieg w przód bez podpowiedzi zwraca każdy znaleziony obiekt ze skalibrowanymi wynikami. Elementy rejestrują się przez rozpoznawanie checkpointu |
| Segmentacja sterowana podpowiedziami | `LibreSAM` | Przebieg w przód nie ma znaczenia bez przestrzennej podpowiedzi dla obrazu lub podpowiedzi koncepcji przekazanej podczas wywołania. Interaktywny i stanowy: jedno kodowanie, wiele podpowiedzi |
| Detekcja z otwartym słownikiem | `LibreOpenVocab` | Detektory dyskryminacyjne warunkowane tekstem. Lista klas jest podpowiedzią ustawianą przez `set_classes` |
| Wizyjno-językowy | `LibreVLM` | Model generatywny sterowany jak detektor. Lista klas jest podpowiedzią, a pewność polem zastępczym |

Trzy sąsiednie poziomy celowo nie rejestrują się w fabryce detektorów, dlatego
`LibreYOLO("some-alias")` do nich nie prowadzi. Są wczytywane według aliasu
rozmiaru i pobierane automatycznie, a nie rozpoznawane na podstawie checkpointu.

Wszystkie cztery zwracają ten sam `Results`, więc kod dalszego przetwarzania
pozostaje bez zmian. Różnią się działającymi metodami: sąsiednie poziomy
zgłaszają `NotImplementedError` dla `train()`, `val()` i `export()`, a poziomy
SAM oraz otwartego słownika również dla `track()`. Strona każdego poziomu
wymienia jego wykluczenia.

## Grupy pokrycia

Ta klasyfikacja decyduje, które rodziny obejmuje przebieg testów przekrojowych,
i jest najczęściej spotykana przez czytelnika na stronie modelu. Każda
zarejestrowana rodzina jest przypisana dokładnie do jednej grupy, a test kończy
się niepowodzeniem, gdy zarejestrowanej rodziny brakuje w przypisaniu. `GROUPS`
w `libreyolo/models/registry.py` jest źródłem poniższej kolumny Znaczenie,
`MODEL_GROUPS` w tym samym pliku przypisuje każdą rodzinę, a kolumna Rodziny
zlicza bezpośrednio to przypisanie. Kolumna Etykieta jest krótszą nazwą używaną
przez witrynę dla tej samej grupy w nagłówku strony modelu.

| Grupa | Etykieta | Rodziny | Znaczenie |
|---|---|---|---|
| `g0` | Flagowa | 2 | Flagowe punkty odniesienia wymagane w pokryciu wspólnych funkcji |
| `g1` | Rdzeń | 10 | Zbiór pokrycia detektorów obsługujących trenowanie |
| `g2` | Obsługiwana | 14 | Dodatkowy zbiór pokrycia rodzin obsługujących trenowanie |
| `g3` | Tylko inferencja | 35 | Rodziny bez implementacji trenowania |
| `g4` | Muzeum | 5 | Historyczne rodziny objęte inferencją |
| `s` | Sąsiedni poziom | 21 | Sąsiednie API (SAM, otwarty słownik, VLM, zero-shot) objęte osobno |

Łącznie jest to 87 rodzin w sześciu grupach. Samo `g3` zawiera więcej rodzin
niż wszystkie pozostałe grupy razem, ponieważ większość rejestru stanowią linie
tylko do inferencji i pokrycie muzealne, a nie aktywnie trenowane detektory.

Dla osoby wybierającej model grupa wskazuje, gdzie można oczekiwać uwagi
inżynieryjnej, a nie dokładność rodziny. `g0` i `g1` są miejscami projektowania
i pierwszego wdrażania nowych funkcji. `g2` jest utrzymywane w stanie poprawnym
w CI, ale funkcja trafia tam zależnie od możliwości, a nie w tej samej fali
wydania. `g3` określa brak, a nie ograniczenie: predykcja, walidacja oraz eksport,
jeśli rodzina go obsługuje, nadal działają. `train()` dla rodziny `g3` lub `g4`
zgłasza `NotImplementedError` z przyczyną zamiast po cichu wykonywać część pracy.
Rodziny `s` w ogóle nie podlegają temu kompromisowi, ponieważ są wczytywane
przez własną fabrykę zamiast `LibreYOLO()`. Zobacz
[podstawowe pojęcia](/docs/concepts), aby dowiedzieć się, jak grupa współgra
z zadaniem, rodziną i rozmiarem podczas odczytywania nazwy pliku checkpointu.

Grupa sama w sobie nie przyznaje ani nie ogranicza możliwości widocznej dla
użytkownika. Obsługa wynika z zaimplementowanego API rodziny i kontroli
możliwości właściwych dla formatu, nigdy wyłącznie z członkostwa w grupie.
Grupy klasyfikują rodziny, a nie zadania, dlatego przebieg pokrycia ograniczony
do zadania jawnie podaje jego nazwę, na przykład „g1 detect”.

Dwa miejsca odczytują grupę podczas działania, a nie tylko w testach.
`collect_model_inventory()` w `libreyolo/models/inventory.py` dołącza grupę do
każdego wpisu wyświetlanego przez inwentarz CLI, a `pretrained=False` uruchamia
specjalną ścieżkę ponownej inicjalizacji od zera tylko dla rodzin w `g0` i `g1`.
Poza tymi dwiema grupami kontrola w `libreyolo/models/base/model.py` jest
całkowicie pomijana, więc `pretrained=False` trafia do własnego `train()`
rodziny jako zwykłe słowo kluczowe.

## Trenowanie

Rodzina w `g3` lub `g4` nie ma implementacji trenowania, a wywołanie jej
`train()` zgłasza błąd. Jest to właściwość kodu rodziny, a nie jej grupy. Grupa
zapisuje ten fakt, zamiast go powodować.

Dla rodziny obsługującej trenowanie to, czy pojedynczy parametr augmentacji
trafia do pipeline'u, jest osobną kwestią z własnym słownictwem trzech wartości:
`used`, `gated_by_mosaic` i `ignored`. Zobacz
[macierz augmentacji](/docs/reference/augmentation-matrix).

## Czego poziom nie określa

Poziom nie jest deklaracją dokładności. Zweryfikowany eksport oznacza, że
artefakt odtwarza natywny model w określonym progu. Nie mówi nic o wyniku
natywnego modelu na zbiorze danych. Wyniki benchmarków znajdują się na stronach
modeli.

Poziom nie jest też deklaracją licencyjną. Licencje wag różnią się w obrębie
rodziny, a repozytorium hostujące konkretny checkpoint jest źródłem
rozstrzygającym. Obecność rodziny w fabryce detektorów nie mówi nic o tym, czy
opublikowane wagi zezwalają na użycie komercyjne.
