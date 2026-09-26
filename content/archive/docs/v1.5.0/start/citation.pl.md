---
title: Cytowanie
seo_title: Cytowanie LibreYOLO i autorów źródłowych
description: >-
  Jak cytować LibreYOLO w publikacji oraz jak cytować autorów użytej rodziny
  modeli. Oba cytowania powinny znaleźć się w tej samej sekcji metod.
lead: >-
  Pełne cytowanie LibreYOLO składa się z dwóch części: biblioteki oraz
  opublikowanej pracy stojącej za rodziną modeli, która wygenerowała wynik.
keywords:
  - jak cytować LibreYOLO
  - LibreYOLO BibTeX
  - LibreYOLO CITATION cff
  - cytowanie modelu
  - cytowanie wizja komputerowa
last_verified: 1.5.0
source_hash: 0f3f23e4e85e38be
---

## Cytowanie LibreYOLO

Repozytorium publikuje metadane cytowania jako
[`CITATION.cff`](https://github.com/LibreYOLO/libreyolo/blob/release/CITATION.cff),
a nie blok BibTeX. GitHub odczytuje ten plik i udostępnia przycisk Cite this
repository na stronie repozytorium, który generuje wpisy APA i BibTeX. Należy
pobrać wpis stamtąd zamiast pisać go ręcznie.

Pełna zawartość pliku:

```yaml
cff-version: 1.2.0
message: "If you use LibreYOLO in your research or software, please cite it as below."
title: "LibreYOLO"
type: software
authors:
  - family-names: Ceccon
    given-names: Xuban
  - name: "The LibreYOLO contributors"
license: MIT
url: "https://github.com/LibreYOLO/libreyolo"
repository-code: "https://github.com/LibreYOLO/libreyolo"
```

Celowo nie zawiera wersji ani daty wydania.
[`RELEASING.md`](https://github.com/LibreYOLO/libreyolo/blob/release/RELEASING.md)
nakazuje opiekunom, aby podczas wydania nigdy nie zmieniali wersji, daty ani
tytułu `CITATION.cff` lub `.zenodo.json`. Dzięki temu każde cytowanie trafia do
jednego rekordu zamiast rozpraszać się między wersjami. Użytą wersję należy
podać we własnym tekście, a cytowanie pozostawić bez zmian.

## Cytowanie rodziny modeli

LibreYOLO jest portem. Uruchomienie `LibreRFDETRm.pt` oznacza uruchomienie
RF-DETR, a recenzent oczekuje przypisania autorstwa osobom, które stworzyły
RF-DETR. Cytowanie samej biblioteki przypisuje ich pracę niewłaściwemu
projektowi.

Wszystkie potrzebne informacje znajdują się na stronie rodziny. Wiersz Upstream
w nagłówku podaje oryginalną pracę i stojącą za nią organizację oraz prowadzi do
publikacji i repozytorium źródłowego. Sekcja Citation w dalszej części zawiera
BibTeX.

Ten BibTeX jest kopiowany dosłownie z bloku cytowania samych autorów, zwykle
z sekcji Citation pliku README źródła nadrzędnego lub z `CITATION.cff`,
i wyświetlany z odnośnikiem do bloku źródłowego, aby można było go sprawdzić.
Nigdy nie jest składany z metadanych publikacji. Ręcznie odtworzony wpis może
zawierać kosztowne, lecz trudne do zauważenia błędy: pominiętego współautora,
niewłaściwe miejsce publikacji, błędny typ wpisu albo rok należący do preprintu.
Preprinty bywają również przyjmowane do publikacji, więc wpis może mieć typ
`@inproceedings`, nawet jeśli czytana wersja znajdowała się w arXiv.

Skopiuj blok w istniejącej postaci. Jeśli styl bibliografii wymaga innego typu
wpisu, należy go przekonwertować zamiast przepisywać i zachować pierwotną
kolejność autorów.

## Wymagania sekcji metod

Trzy elementy zapewniają odtwarzalność wyniku LibreYOLO i prawidłowe przypisanie
autorstwa:

- Biblioteka cytowana z `CITATION.cff` wraz z użytą wersją. Polecenie `libreyolo version` wyświetla ją razem z wersjami Python, torch i CUDA, z którymi działa.
- Praca źródłowa cytowana z sekcji Citation na stronie rodziny.
- Dokładna nazwa pliku checkpointu, na przykład `LibreRFDETRm.pt`. Rozmiary w obrębie rodziny zachowują się różnie, a kilka rodzin publikuje checkpointy trenowane na różnych zbiorach danych pod tym samym prefiksem, więc sama nazwa rodziny nie określa uruchomionego wariantu.

Przypisanie autorstwa jest także warunkiem licencji wielu zasobów publikowanych
przez LibreYOLO. Zarówno Apache-2.0, jak i rodzina CC BY wymagają dołączenia
informacji do redystrybuowanych wag, co jest obowiązkiem odrębnym od cytowania
publikacji. Na stronie [licencji](/docs/licensing) opisano warunki dotyczące
poszczególnych checkpointów.
