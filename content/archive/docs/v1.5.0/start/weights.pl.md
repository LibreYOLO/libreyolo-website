---
title: Checkpointy i wagi
seo_title: Checkpointy i wagi LibreYOLO
description: >-
  Jak LibreYOLO znajduje, pobiera i weryfikuje wagi modeli, gdzie są hostowane,
  jak pracować bez sieci i co zapewnia bezpieczne wczytywanie checkpointu.
lead: >-
  Checkpoint LibreYOLO to słownik torch.save zawierający słownik stanu oraz
  metadane potrzebne do jego identyfikacji. Ta strona opisuje pochodzenie
  plików, miejsce ich zapisu i sposób wczytywania.
keywords:
  - wagi libreyolo
  - checkpointy libreyolo
  - libreyolo pobieranie wag
  - libreyolo offline
  - libreyolo hugging face
  - metadane checkpointu
last_verified: 1.5.0
meta:
  - label: Miejsce hostowania
    value: 'Jedno repozytorium Hugging Face na checkpoint:'
    links:
      - label: huggingface.co/LibreYOLO
        href: 'https://huggingface.co/LibreYOLO'
  - label: Lokalna pamięć podręczna
    value: weights/ under the working directory
    mono: true
  - label: Schemat metadanych
    value: v1.0
snippets:
  load:
    - label: Automatyczne pobieranie
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Sama nazwa pliku prowadzi do weights/LibreYOLO9t.pt, a jeśli pliku
        # jeszcze nie ma, zostanie on pobrany do tego katalogu.
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model(SAMPLE_IMAGE).boxes)
    - label: Jawna ścieżka
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Ścieżka zawierająca katalog jest używana dokładnie w podanej postaci
        # i nigdy nie powoduje pobierania z sieci.
        model = LibreYOLO("/opt/models/LibreYOLO9t.pt")
        print(model.family, model.size, model.task)
  inspect:
    - label: CLI
      language: bash
      code: |
        # Odczytuje metadane bez tworzenia modelu i zgłasza, czy spełniają
        # wymagania schematu.
        libreyolo metadata path=weights/LibreYOLO9t.pt
    - label: JSON
      language: bash
      code: |
        libreyolo metadata path=weights/LibreYOLO9t.pt --json
    - label: Python
      language: python
      code: >
        from libreyolo.utils.serialization import (
            load_untrusted_torch_file,
            validate_checkpoint_metadata,
        )


        loaded = load_untrusted_torch_file("weights/LibreYOLO9t.pt")


        # Zwraca listę problemów. Pusta lista oznacza zgodność pliku z v1.0.

        print(validate_checkpoint_metadata(loaded))

        print(loaded["model_family"], loaded["size"], loaded["task"],
        loaded["nc"])
source_hash: 210a12baa1417cfb
---

## Miejsce wyszukiwania checkpointu

Odwołanie do modelu bez katalogu, takie jak `LibreYOLO9t.pt`, jest rozwiązywane
względem `weights/` w bieżącym katalogu roboczym. Jeśli istnieje
`weights/LibreYOLO9t.pt`, zostanie użyty ten plik. Jeśli plik o takiej nazwie
istnieje bezpośrednio w katalogu roboczym, zostanie użyty zamiast niego. W
przeciwnym razie miejscem docelowym pobierania staje się
`weights/LibreYOLO9t.pt`.

Odwołanie zawierające katalog, bezwzględne lub względne, jest interpretowane
dosłownie. Tej postaci należy używać, gdy wagi znajdują się w centralnej
lokalizacji i niczego nie należy pobierać.

<code-tabs name="load" />

## Automatyczne pobieranie

Gdy rozwiązana ścieżka nie istnieje, LibreYOLO analizuje nazwę pliku, aby
odtworzyć rodzinę, rozmiar i zadanie, a następnie prosi odpowiednią rodzinę o
adres URL pobierania. Większość rodzin buduje go na podstawie organizacji
LibreYOLO na Hugging Face, gdzie każdy checkpoint ma własne repozytorium nazwane
od pliku:

```text
https://huggingface.co/LibreYOLO/<name>/resolve/main/<name>.pt
```

Sufiks wariantu zbioru danych pozostaje częścią nazwy repozytorium. Dzięki temu
checkpoint wytrenowany na zbiorze innym niż domyślny dla rodziny prowadzi do
własnego repozytorium, zamiast nadpisywać domyślny plik.

Sam transfer jest zabezpieczony, ponieważ ucięty plik wag dopiero później
powoduje mało pomocny błąd. Pobierane dane są przesyłane strumieniowo do pliku
`.part`, który zostaje atomowo przeniesiony na miejsce dopiero po ukończeniu.
Przerwany proces nigdy nie pozostawia więc częściowo zapisanego checkpointu w
ścieżce docelowej. Przerwany transfer jest wznawiany od przesunięcia bajtowego z
użyciem walidatora HTTP, a jeśli serwer zgłosi zmianę obiektu, rozpoczyna się od
zera. Po niepowodzeniu podejmowane są trzy ponowne próby z wykładniczo rosnącym
opóźnieniem. Współbieżne procesy korzystające z tej samej ścieżki używają pliku
blokady, więc dwa uruchomienia trenowania rozpoczynające się jednocześnie
pobierają plik tylko raz. Jeśli rodzina pobiera dane z hosta innej firmy zamiast
z organizacji LibreYOLO, może ustalić sumę kontrolną i odrzucić niezgodny plik.

Jeśli ustawiono `HF_TOKEN` lub token jest zapisany w
`~/.cache/huggingface/token`, zostaje dołączony jako token bearer. Jest
dołączany wyłącznie do adresów URL `huggingface.co`, więc rodzina pobierająca z
innego hosta nigdy go nie otrzyma.

Nie każda rodzina obsługuje automatyczne pobieranie. Niektóre celowo nie
zwracają adresu URL, ponieważ wydanych wag nie wolno redystrybuować, a błąd
wyjaśnia wtedy, co należy dostarczyć. Inne wyświetlają informację o licencji
przed rozpoczęciem transferu. Informacja ta jest sygnałem w czasie działania,
że warunki checkpointu są węższe niż warunki kodu. Warto ją przeczytać zamiast
pomijać.

## Organizacja Hugging Face

Opublikowane wagi znajdują się pod adresem
[huggingface.co/LibreYOLO](https://huggingface.co/LibreYOLO), po jednym
repozytorium na checkpoint. Każde repozytorium ma licencję, która nie musi być
jednolita w obrębie rodziny. Rodzina z kodem na licencji MIT może mieć wagi
objęte innymi warunkami. Repozytorium jest rozstrzygającym źródłem. Strona
każdego modelu wymienia opublikowane checkpointy tej rodziny i ich licencje w
sekcjach Checkpointy i Licencjonowanie.

## Praca offline

Gdy pliki znajdują się już lokalnie, żaden element biblioteki nie wymaga
dostępu do sieci. Działają dwa rozwiązania:

Można z góry wypełnić katalog `weights/` obok miejsca uruchamiania zadania.
Wystarczy raz pobrać checkpointy na komputerze podłączonym do sieci, a następnie
skopiować katalog. Opisany wyżej etap rozwiązywania znajdzie pliki i nie połączy
się z siecią.

Można też przekazać ścieżkę bezwzględną do współdzielonej lokalizacji. Odwołanie
zawierające katalog jest używane w podanej postaci, więc prawidłową konfiguracją
jest również montowanie wyselekcjonowanych wag tylko do odczytu. Jeśli proces
nie może zapisać danych obok checkpointu wymagającego konwersji, konwersja
korzysta z prywatnego katalogu tymczasowego zamiast zakończyć się błędem.

Zbiory danych podlegają osobnej regule. Są rozwiązywane w `~/datasets` lub w
katalogu wskazanym przez `LIBREYOLO_DATASETS_DIR`, jeśli ustawiono tę zmienną.

## Bezpieczeństwo wczytywania

Checkpointy są plikami pickle, a taki plik może wykonać dowolny kod podczas
otwierania. LibreYOLO traktuje każdy plik wag jako niezaufany i wczytuje go
przez ścieżkę PyTorch z `weights_only=True`, która ogranicza moduł
deserializujący do tensorów i niewielkiego zbioru bezpiecznych typów. Dotyczy to
każdego przekazanego pliku, a nie tylko plików pobranych przez LibreYOLO. Jeśli
kompilacja PyTorch jest zbyt stara, aby obsługiwać ten argument, wczytywanie
zostaje odrzucone zamiast wykonane w niebezpieczny sposób.

Niektóre checkpointy trenowania z projektów źródłowych zawierają obiekty
odrzucane przez ograniczony moduł deserializujący, na przykład obiekt
konfiguracji z frameworka użytego do trenowania. LibreYOLO nie potrzebuje tych
obiektów metadanych, dlatego podczas konwersji każda zablokowana klasa jest
zastępowana nieaktywnym odpowiednikiem, który spełnia wymagania modułu
deserializującego bez uruchamiania czegokolwiek. Do przekonwertowanego pliku
trafiają wyłącznie tensory. Nazwy wrażliwych modułów są bezwarunkowo odrzucane,
a pętla ponownych prób ma ograniczenie, dzięki czemu plik przygotowany do
wprowadzania nieskończonej serii zablokowanych klas zostaje bezpiecznie
odrzucony. Pozostałą część tej ścieżki opisuje [import istniejących
wag](/docs/migrate).

## Metadane checkpointu

Checkpoint LibreYOLO jest słownikiem, którego klucz `model` zawiera słownik
stanu PyTorch. Schemat v1.0 wymaga dziewięciu kluczy. Razem pozwalają one
funkcji fabrykującej zidentyfikować plik bez analizowania jego nazwy ani
zgadywania na podstawie kształtów tensorów.

| Klucz | Znaczenie |
|---|---|
| `model` | Słownik stanu PyTorch |
| `schema_version` | Wersja kontraktu metadanych. v1.0 używa ciągu `1.0` |
| `libreyolo_version` | Wersja LibreYOLO, która utworzyła plik |
| `model_family` | Zarejestrowany identyfikator rodziny, na przykład `yolo9` |
| `size` | Wariant w obrębie rodziny, na przykład `t` lub `r18` |
| `task` | Jedna kanoniczna nazwa zadania |
| `nc` | Dodatnia liczba klas |
| `names` | Mapowanie indeksu klasy na etykietę, obejmujące wartości od `0` do `nc - 1` |
| `imgsz` | Dodatnia rozdzielczość wejściowa |

Zadania o dodatkowej strukturze zapisują ją obok tych kluczy. Checkpointy
estymacji pozy dodają `num_keypoints` i `keypoint_dim`, a opcjonalnie także
wartości sigma OKS dla poszczególnych punktów kluczowych. Checkpointy OCR
zawierają pełny zestaw znaków CTC, dzięki czemu plik jest samowystarczalny.
Checkpointy odtwarzania mogą zawierać rodzaj degradacji i współczynnik
powiększenia. Checkpointy modułu trenującego dodają stan wznowienia, na przykład
`epoch`, stan optymalizatora i wagi EMA. Opublikowane wagi do inferencji nie
powinny ich zawierać.

Plik spełniający wymagania wszystkich dziewięciu kluczy wczytuje się przez
ścieżkę metadanych. Plik niespełniający tych wymagań zostaje przekonwertowany,
jeśli rodzina rozpoznaje jego układ, albo wczytany przez ścieżkę zgodności z
ostrzeżeniem wymieniającym brakujące elementy.

## Inspekcja checkpointu

<code-tabs name="inspect" />

Polecenie `libreyolo metadata` nigdy nie tworzy modelu, więc działa na pliku,
którego rodzina nie jest zainstalowana, oraz na pliku, co do którego nie ma
pewności.
