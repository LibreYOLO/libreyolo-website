---
title: libreyolo label
seo_title: 'libreyolo label: dokumentacja polecenia'
description: >-
  Uruchamianie lokalnego narzędzia do adnotacji ramek ograniczających: argumenty
  z wartościami domyślnymi, przełącznik automatycznego etykietowania AI i to, co
  odsłania powiązanie z interfejsem sieciowym.
lead: >-
  Uruchamia lokalne narzędzie webowe do rysowania i edycji ramek
  ograniczających. Zapisuje pliki etykiet w natywnym formacie LibreYOLO, więc
  zbiór danych opisany tutaj trenuje się bez kroku konwersji.
keywords:
  - libreyolo label cli
  - narzędzie do adnotacji ramek
  - etykietowanie zbioru danych yolo
  - automatyczne etykietowanie obrazów
  - libreyolo label udostępnianie w sieci
last_verified: 1.5.0
meta:
  - label: Polecenie
    value: libreyolo label
    mono: true
  - label: Wyjście
    value: URL serwera na stdout; etykiety zapisywane jako labels/*.txt obok obrazów
snippets:
  examples:
    - label: Podstawowe użycie
      language: bash
      code: >
        # Otwiera stronę główną projektu; zbiór danych wybiera się lub tworzy w
        przeglądarce.

        libreyolo label
    - label: 'Tylko ręcznie, stały port'
      language: bash
      code: |
        libreyolo label no_assist=true port=9200 no_browser=true
    - label: Dołączenie osób z zespołu
      language: bash
      code: |
        libreyolo label share=true
source_hash: bddad245877793b1
---

## Składnia

```bash
libreyolo label [data=<dataset.yaml|folder>] [key=value ...]
```

Argumenty to pary `key=value`, działa też forma POSIX, więc `port=9200` i
`--port 9200` to ten sam argument.

## Argumenty

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `data` | | YAML zbioru danych lub folder otwierany bezpośrednio. Bez tej wartości start następuje na stronie głównej projektu |
| `host` | `127.0.0.1` | Host lub interfejs, z którym następuje powiązanie |
| `port` | `8000` | Port, z którym następuje powiązanie. Przechodzi na kolejny wolny, jeśli jest zajęty |
| `device` | `auto` | Urządzenie do automatycznego etykietowania AI: `0`, `cpu`, `mps`, `auto` |
| `no_assist` | `false` | Wyłącza automatyczne etykietowanie AI, zostaje narzędzie wyłącznie ręczne |
| `no_browser` | `false` | Nie otwiera automatycznie przeglądarki |
| `share` | `false` | Powiązanie z `0.0.0.0`, aby osoby z zespołu w tej samej sieci mogły dołączyć |
| `json` | `false` | Wyjście JSON na stdout |
| `quiet` | `false` | Wyciszenie stderr |
| `verbose` | `false` | Szczegółowe wyjście na stderr |

## Przykłady

<code-tabs name="examples" />

## Uwagi

### Co zapisuje

Ramki są zapisywane jako pliki `labels/*.txt` w natywnym formacie LibreYOLO,
czyli w tym samym, który czyta `libreyolo train`, więc później nic nie trzeba
konwertować. Ta wersja obsługuje wyłącznie ramki ograniczające. Zmiany
zapisują się przy przechodzeniu między obrazami.

### Otwieranie zbioru danych

Bez `data` narzędzie startuje na stronie głównej projektu, a zbiór danych
wybiera się lub tworzy z poziomu przeglądarki. Podanie
`data=path/to/data.yaml` otwiera ten zbiór od razu, a linia startowa podaje
liczbę obrazów, liczbę klas oraz to, czy zbiór jest zapisywalny. Zbiór tylko do
odczytu również się otwiera i podaje powód, dla którego nie można do niego
pisać.

### Udostępnianie i działanie `host`

`share=true` wiąże adres wildcard, dzięki czemu inne maszyny w tej samej sieci
docierają do narzędzia, a działania administracyjne, przełączanie i usuwanie
projektów oraz uruchamianie obliczeń pozostają na tej maszynie.

Ustawienie `host` na konkretny interfejs działa inaczej i mniej bezpiecznie:
host staje się nie do odróżnienia od klienta sieciowego, więc każdy klient
dostaje prawa administracyjne. Polecenie wypisuje wtedy ostrzeżenie na stderr.
Lepiej użyć `share=true`.

### Porty i zamykanie

Zajęty port przechodzi na kolejny, maksymalnie dwadzieścia pozycji od żądanego.
Niepowodzenie wszystkich dwudziestu kończy się kodem `io_error`. URL wypisany
na stdout wskazuje port, który został faktycznie powiązany. Przy `share=true`
wynik zawiera dodatkowo `lan_url`, adres, który powinny otworzyć osoby z
zespołu.

Polecenie działa na pierwszym planie do czasu naciśnięcia Ctrl+C.

Powiązane: [`libreyolo doctor`](/docs/cli/doctor), aby sprawdzić opisany zbiór
danych przed trenowaniem, oraz [`libreyolo train`](/docs/cli/train), aby na nim
trenować.
