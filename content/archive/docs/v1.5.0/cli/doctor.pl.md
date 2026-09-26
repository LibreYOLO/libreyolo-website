---
title: libreyolo doctor
seo_title: dokumentacja polecenia libreyolo doctor
description: >-
  Sprawdzenie zbioru danych do detekcji przed trenowaniem: argumenty z
  wartościami domyślnymi, rodziny kontroli, które można pominąć lub wybrać, oraz
  kody wyjścia, na których może opierać się bramka CI.
lead: >-
  Uruchamia zestaw kontroli kondycji zbioru danych do detekcji i zgłasza to, co
  zaszkodziłoby trenowaniu: brakujące pliki, błędne etykiety, uszkodzone obrazy,
  wyciek między podziałami i nierównowagę klas.
keywords:
  - libreyolo doctor cli
  - sprawdzanie zbioru danych yolo
  - walidacja datasetu yolo
  - wyciek danych między podziałami
  - libreyolo doctor strict
last_verified: 1.5.0
meta:
  - label: Polecenie
    value: libreyolo doctor
    mono: true
  - label: Wymagane
    value: data
    mono: true
  - label: Wyjście
    value: 'Raport wykrytych problemów na stdout. Kod wyjścia 1, gdy znaleziono błędy'
snippets:
  examples:
    - label: Podstawowe użycie
      language: bash
      code: >
        # download=true pozwala dołączonemu coco8.yaml pobrać obrazy, jeśli ich
        brakuje.

        libreyolo doctor coco8.yaml download=true
    - label: 'Szybki przebieg, bez dekodowania obrazów'
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true fast=true
    - label: Bramka CI na wybranych kontrolach
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true strict=true json=true \
          only=labels,files,config
source_hash: 79e0ef471d567ea3
---

## Składnia

```bash
libreyolo doctor <data.yaml> [key=value ...]
```

Zbiór danych jest argumentem pozycyjnym, a `data=<path>` jest akceptowane jako
alternatywa. Podanie obu z różnymi wartościami kończy się błędem
`config_conflict`. Wszystko inne to pary `key=value`, a forma POSIX również
działa, więc `imgsz=1024` i `--imgsz 1024` to ten sam argument.

## Argumenty

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `data` | | Pozycyjny. Plik YAML zbioru danych w formacie detekcji YOLO, np. `coco8.yaml`. Wymagany |
| `imgsz` | `640` | Rozmiar obrazu treningowego używany w kontrolach opartych na pikselach, takich jak drobne obiekty |
| `fast` | `false` | Pomija dekodowanie obrazów, co usuwa kontrole uszkodzeń, duplikatów i wycieku |
| `skip` | | Rozdzielone przecinkami identyfikatory kontroli lub rodziny do pominięcia, np. `images,labels.tiny_object` |
| `only` | | Rozdzielone przecinkami identyfikatory kontroli lub rodziny do wyłącznego uruchomienia |
| `strict` | `false` | Ostrzeżenia również powodują niezerowy kod wyjścia, na potrzeby bramek CI |
| `download` | `false` | Zezwala na pobranie zbioru danych z adresu URL, jeśli go brakuje. Nigdy skryptów |
| `json` | `false` | Wynik w formacie JSON na stdout |
| `quiet` | `false` | Wycisza stderr |
| `help_json` | `false` | Wypisuje schemat polecenia jako JSON i kończy działanie |

### Rodziny kontroli

`skip` i `only` przyjmują pełny identyfikator kontroli albo prefiks rodziny,
więc `images` wybiera każdą kontrolę `images.*`.

| Rodzina | Obejmuje |
|---|---|
| `config` | Sam plik YAML zbioru danych: brakujące `names`, `nc` niezgodne z `names`, brakujące podziały, nierozwiązywalna ścieżka `path`, zduplikowane nazwy klas |
| `files` | Parowanie obrazów i etykiet: brakujące etykiety, brakujące obrazy, osierocone etykiety, nieobsługiwane rozszerzenia, kolizje wielkości liter |
| `labels` | Zawartość etykiet: składnia, linie wielokątów, identyfikatory klas poza zakresem, współrzędne poza zakresem, zdegenerowane ramki, drobne obiekty, ogromne ramki, ekstremalne proporcje, zduplikowane ramki, zatłoczone obrazy, identyczne pliki |
| `images` | Dane pikselowe: uszkodzone pliki, orientacja EXIF, nietypowe tryby kolorów, drobne lub ekstremalne wymiary, jednolite obrazy, duplikaty dokładne i przybliżone |
| `splits` | Wyciek między podziałami, dokładny i przybliżony |
| `balance` | Rozkład klas: klasy bez instancji lub z niewielką ich liczbą, nierównowaga, pokrycie podziałów, udział tła, skośność podziałów |

## Przykłady

<code-tabs name="examples" />

## Uwagi

### Kody wyjścia

`0`, gdy nie znaleziono błędów, `1`, gdy którykolwiek z wykrytych problemów jest
błędem. Przy `strict=true` ostrzeżenia również podnoszą kod wyjścia do `1`, co
jest ustawieniem odpowiednim dla bramki CI.

Problemy z użyciem mają własne kody: `2` dla nieznanego identyfikatora kontroli
lub rodziny w `skip` albo `only`, `3`, gdy nie można odnaleźć zbioru danych,
oraz `3`, gdy zbiór danych nie ma struktury detekcyjnej.

### Wybór jest rozstrzygany przed skanowaniem

`skip` i `only` są rozstrzygane względem rejestru kontroli, zanim cokolwiek
zostanie odczytane z dysku, więc literówka powoduje błąd natychmiast, a nie po
długim przebiegu przez obrazy. Selektor, który niczego nie dopasowuje, jest
błędem, a komunikat wymienia znane rodziny.

Jeśli kombinacja `skip`, `only` i `fast` nie pozostawia żadnych kontroli do
uruchomienia, jest to również błąd, a nie ciche przejście.

### Pobieranie

Zbiór danych nie jest pobierany, dopóki nie ustawiono `download=true`, i
wykonywane są wyłącznie pobrania z adresów URL. Osadzony w pliku YAML zbioru
danych skrypt pobierania w Pythonie nigdy nie jest uruchamiany przez to
polecenie, niezależnie od flagi.

### Zakres

Kontrole są napisane dla zbiorów danych do detekcji. Zbiór danych, którego
etykiety mają strukturę pozy, segmentacji lub obróconych ramek, jest
rozpoznawany i odrzucany z `data_invalid`, zamiast być oceniany według
niewłaściwych reguł.

### Wynik

Raport dla człowieka trafia na stdout, a `json=true` zastępuje go strukturalnym
obiektem zawierającym liczby zbiorcze, statystyki zbioru danych, każdy wykryty
problem oraz listę pominiętych kontroli.

Powiązane: [`libreyolo train`](/docs/cli/train), uruchomienie, przed którym to
polecenie ma być wykonywane.
