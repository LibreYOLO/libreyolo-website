---
title: libreyolo ui
seo_title: dokumentacja polecenia libreyolo ui
description: >-
  Uruchamianie lokalnego interfejsu webowego do inferencji: adres nasłuchu,
  zachowanie portu, wybór urządzenia i sposób kończenia polecenia.
lead: >-
  Uruchamia lokalny serwer webowy, który przyjmuje przeciągnięte lub wklejone
  obrazy, przepuszcza je przez wybrany model i pokazuje wyniki w przeglądarce.
keywords:
  - libreyolo ui cli
  - interfejs webowy libreyolo
  - lokalna inferencja w przeglądarce
  - inferencja przeciągnij i upuść
  - libreyolo ui port
last_verified: 1.5.0
meta:
  - label: Polecenie
    value: libreyolo ui
    mono: true
  - label: Wyjście
    value: 'Adres URL serwera na stdout, potem proces pozostaje na pierwszym planie'
snippets:
  examples:
    - label: Podstawowe wywołanie
      language: bash
      code: |
        libreyolo ui
    - label: 'Stały port, bez przeglądarki'
      language: bash
      code: |
        libreyolo ui port=9000 no_browser=true
    - label: 'Na CPU, wyjście maszynowe'
      language: bash
      code: |
        libreyolo ui device=cpu json=true
source_hash: b0eebd33fd0f463b
---

## Składnia

```bash
libreyolo ui [key=value ...]
```

Argumenty to pary `key=value`, działa też forma POSIX, więc `port=9000` i
`--port 9000` to ten sam argument.

## Argumenty

| Argument | Wartość domyślna | Znaczenie |
|---|---|---|
| `host` | `127.0.0.1` | Host lub interfejs do nasłuchu |
| `port` | `8000` | Port do nasłuchu. Jeśli jest zajęty, wybierany jest kolejny wolny |
| `device` | `auto` | Urządzenie: `0`, `cpu`, `mps`, `auto` |
| `no_browser` | `false` | Bez automatycznego otwierania przeglądarki |
| `json` | `false` | Wyjście JSON na stdout |
| `quiet` | `false` | Wyciszenie stderr |
| `verbose` | `false` | Szczegółowe wyjście na stderr |

## Przykłady

<code-tabs name="examples" />

## Uwagi

Domyślnie nasłuch odbywa się na loopbacku, więc interfejs jest dostępny tylko z
tej maszyny.

Jeśli żądany port jest zajęty, polecenie próbuje kolejnego i idzie tak do
dwudziestu portów powyżej żądanego. Jeśli żaden z dwudziestu nie zadziała,
polecenie kończy się błędem `io_error` i sugestią podania innego portu. W
adresie URL wypisanym na stdout znajduje się port, który faktycznie został
zajęty, więc warto go odczytać, zamiast zakładać, że jest to port żądany.

Jeśli nie podano `no_browser=true`, krótko po zajęciu portu pod tym adresem
otwiera się karta przeglądarki.

Polecenie działa dalej na pierwszym planie aż do Ctrl+C, które czysto zamyka
serwer. Nie ma trybu odłączonego; aby odzyskać terminal, należy przenieść proces
w tło środkami powłoki.

`json=true` wypisuje adres URL i urządzenie jako jeden obiekt z polem
`schema_version` jeszcze przed startem serwera, i w ten sposób skrypt poznaje
zajęty port.

Powiązane: [`libreyolo label`](/docs/cli/label) do rysowania ramek i zapisywania
etykiet, [`libreyolo monitor`](/docs/cli/monitor) do obserwowania przebiegów
trenowania. Oba to lokalne serwery webowe o tym samym zachowaniu portu i
przeglądarki.
