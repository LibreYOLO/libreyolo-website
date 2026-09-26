---
title: libreyolo monitor
seo_title: Dokumentacja polecenia libreyolo monitor
description: >-
  Udostępnianie panelu na żywo dla przebiegów treningu: argumenty z wartościami
  domyślnymi, co serwer czyta z dysku i jak jeden serwer obsługuje wiele
  przebiegów.
lead: >-
  Udostępnia panel webowy dla przebiegów treningu, czytając artefakty, które
  przebieg zapisuje na dysku. Nigdy nie podłącza się do procesu trenowania, więc
  przebiegi trwające, zakończone i te, które padły, wyświetlają się tak samo.
keywords:
  - libreyolo monitor cli
  - dashboard treningu
  - podgląd treningu na żywo
  - libreyolo monitor port
  - metryki treningu podgląd
last_verified: 1.5.0
meta:
  - label: Polecenie
    value: libreyolo monitor
    mono: true
  - label: Wyjście
    value: 'Adres URL serwera na stdout, potem proces pozostaje na pierwszym planie'
snippets:
  examples:
    - label: Podstawowe użycie
      language: bash
      code: |
        # Obserwuje runs/ i wypisuje wszystkie przebiegi w tym katalogu.
        libreyolo monitor
    - label: Inny katalog główny przebiegów
      language: bash
      code: |
        libreyolo monitor experiments/
    - label: 'Jeden przebieg, stały port, bez przeglądarki'
      language: bash
      code: |
        libreyolo monitor runs/train/exp port=9100 no_browser=true
source_hash: 4aa178141d451728
---

## Składnia

```bash
libreyolo monitor [<run-dir|runs-root>] [key=value ...]
```

Katalog jest argumentem pozycyjnym. Wszystko poza nim to para `key=value`,
działa też forma POSIX, więc `port=9100` i `--port 9100` to ten sam argument.

## Argumenty

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `run_dir` | `runs` | Pozycyjny. Katalog główny przebiegów do obserwowania albo pojedynczy katalog przebiegu do otwarcia bezpośrednio. W obu przypadkach wypisywane są wszystkie przebiegi w tym katalogu głównym |
| `host` | `127.0.0.1` | Host lub interfejs, na którym serwer nasłuchuje |
| `port` | `8420` | Port, na którym serwer nasłuchuje. Przechodzi na kolejny wolny, jeśli jest zajęty |
| `no_browser` | `false` | Bez automatycznego otwierania przeglądarki |
| `json` | `false` | Wyjście JSON na stdout |
| `quiet` | `false` | Wyciszenie stderr |
| `verbose` | `false` | Szczegółowe wyjście na stderr |

## Przykłady

<code-tabs name="examples" />

## Uwagi

### Jeden serwer, wiele przebiegów

Serwer obserwuje katalog główny przebiegów, a nie pojedynczy przebieg, i
adresuje każdy przebieg przez URL, więc kilka przebiegów na jednej maszynie
dzieli jeden port. Otwórz adres URL katalogu głównego, aby zobaczyć indeks,
albo po jednej karcie na przebieg; parametr `?run=` w każdym adresie URL
wskazuje, o który chodzi.

Wskazanie poleceniu pojedynczego katalogu przebiegu zakotwicza serwer w
katalogu nadrzędnym, więc sąsiednie przebiegi nadal pojawiają się w indeksie, a
link prowadzi bezpośrednio do wskazanego przebiegu.

### Co jest czytane

Panel jest budowany z plików, które zapisuje `libreyolo train`: `status.json`,
`metrics.jsonl`, `train.log` oraz obrazy z przebiegu. Z samego procesu
trenowania nic nie jest czytane, więc przebieg zakończony albo taki, który
padł, wyświetla się dokładnie tak samo jak trwający.

### Wymagania wstępne i porty

Musi już istnieć co najmniej jeden przebieg. Bez argumentu i bez katalogu
`runs/` polecenie kończy się kodem `source_not_found`; to samo dzieje się, gdy
podany katalog nie zawiera żadnych przebiegów.

Zajęty port przenosi się na kolejny, maksymalnie o dwadzieścia dalej niż
żądany. Jeśli wszystkie dwadzieścia zawiodą, polecenie kończy się kodem
`io_error`. Adres URL wypisany na stdout wskazuje port, który faktycznie został
przypisany.

Polecenie działa na pierwszym planie aż do Ctrl+C. `json=true` wypisuje adres
URL, obserwowany katalog główny i liczbę znalezionych przebiegów jako jeden
obiekt z polem `schema_version`.

Powiązane: [`libreyolo train`](/docs/cli/train), którego argumenty `project` i
`name` decydują o tym, gdzie trafiają katalogi przebiegów.
