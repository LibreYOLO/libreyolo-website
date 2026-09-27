---
title: Rozwiązywanie problemów
seo_title: Rozwiązywanie typowych błędów LibreYOLO
description: >-
  Najczęstsze błędy zgłaszane przez LibreYOLO, ich znaczenie i sposób naprawy.
  Uwzględnia dwa problemy, które zamiast wyjątku dają nieprawidłowe dane
  wyjściowe.
lead: >-
  Błędy pogrupowane według wyświetlanego komunikatu. Dwie ostatnie pozycje
  dotyczą odwrotnego problemu: kod działa, zwraca wiarygodnie wyglądający wynik,
  ale jest on błędny.
keywords:
  - błąd libreyolo
  - modulenotfounderror libreyolo
  - libreyolo cuda brak pamięci
  - libreyolo notimplementederror
  - rozwiązywanie problemów libreyolo
last_verified: 1.5.0
source_hash: e271ab29b789865a
---

Błędy są pogrupowane według wyświetlanego tekstu. Jeśli danego komunikatu nie
ma na tej stronie, odpowiedzi na pytania niedotyczące awarii zawiera
[FAQ](/docs/faq), a `libreyolo models` informuje, co faktycznie może wczytać
bieżąca instalacja.

## ModuleNotFoundError z nazwą pakietu, którego nie zaimportowano

Niektóre rodziny wymagają opcjonalnego zestawu zależności. Komunikat podaje
nazwę brakującego pakietu zamiast zestawu, dlatego rozwiązanie nie zawsze wynika
wprost ze śladu stosu.

Uruchom `libreyolo models`. Każda rodzina z brakującą zależnością jest
wyświetlana wraz z dokładnym poleceniem pip, które ją włącza. Nie trzeba więc
samodzielnie przyporządkowywać pakietu do zestawu zależności. Polecenie
`libreyolo models --json` wyświetla te same dane jako obiekt.

Na [stronie instalacji](/docs/install) znajduje się lista wszystkich zestawów
opcjonalnych zależności i ich zastosowań.

## Inferencja ONNX wymaga onnxruntime

```
ImportError: ONNX inference requires onnxruntime. Install with: pip install onnxruntime
```

Pakiet bazowy nie zależy od środowiska uruchomieniowego, ponieważ wybór zależy
od sprzętu. Zainstaluj `onnxruntime` dla CPU albo `onnxruntime-gpu` dla CUDA.
Oba udostępniają ten sam moduł `onnxruntime`, więc należy zainstalować tylko
jeden z nich.

## Nie znaleziono modelu ONNX

```
FileNotFoundError: ONNX model not found: <path>
```

Ścieżka jest rozwiązywana względem katalogu roboczego, a nie skryptu. Komunikat
pojawia się również wtedy, gdy eksport bez wyraźnej informacji zapisał plik w
innym miejscu. Funkcja `export()` zwraca zapisaną ścieżkę, dlatego należy użyć
jej wartości zwrotnej zamiast zakładać nazwę.

## NotImplementedError z train()

Nie każdą rodzinę można trenować. Niektóre przeniesiono wyłącznie do predykcji,
walidacji i eksportu, a ich funkcja `train()` zgłasza wyjątek zamiast pozorować
działanie.

Powód wyjaśnia [wpis w FAQ](/docs/faq). Aby sprawdzić konkretną rodzinę przed
napisaniem skryptu trenującego, należy odczytać informację o trenowaniu na jej
stronie modelu.

## NotImplementedError z export()

Rodzina może obsługiwać zadanie, lecz nie pozwalać na jego eksport. Często
spotykanym przypadkiem jest EoMT: `export()` przyjmuje zadanie semantyczne, ale
zgłasza wyjątek dla `segment` i `panoptic`, ponieważ nie zdefiniowano wymaganego
przez nie kontraktu masek zapytań w środowisku uruchomieniowym.

```
NotImplementedError: LibreEoMT instance and panoptic export need query-mask runtime contracts.
```

Strona każdej rodziny zawiera macierz eksportu pokazującą zweryfikowane
połączenia zadań i formatów.

## Brak pamięci CUDA

Najpierw zmniejsz `batch`, a następnie `imgsz`. Obie wartości wpływają na
zapotrzebowanie na pamięć mniej więcej proporcjonalnie do swojego rozmiaru, ale
zmniejszenie batcha nie zmienia tego, co widzi model.

Jeśli błąd pojawia się podczas walidacji, a nie trenowania, trzeba również
zmniejszyć rozmiar batcha używany osobno przez walidację.

W systemie Windows GPU obsługujący wyświetlacz ma jeszcze jeden tryb awarii,
który przypomina losowy błąd CUDA zamiast braku pamięci. Sterownik resetuje GPU,
który nie odpowiada dłużej niż określony limit czasu, i kończy wykonywany proces.
Długie kernele uruchomione na karcie obsługującej monitor mogą wywołać taki
reset.

## Nie można pobrać wag

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej. [FAQ](/docs/faq) opisuje jej położenie i pracę całkowicie
offline.

Jeśli pobieranie zwraca błąd 404, sprawdź podaną nazwę pliku. Adres URL jest z
niej wyprowadzany razem z sufiksem zadania, dlatego nazwa niezgodna z
opublikowanym checkpointem prowadzi do nieistniejącego adresu URL. Tabela
checkpointów na stronie każdego modelu zawiera dokładne nazwy opublikowanych
plików.

## Trenowanie zawiesza się lub uruchamia ponownie w systemie Windows

System Windows nie ma mechanizmu `fork`, dlatego procesy robocze modułu
wczytującego dane rozpoczynają pracę od ponownego importu skryptu. Bez warunku
`if __name__ == "__main__":` każdy proces roboczy ponownie wykonuje wywołanie
trenowania, co powoduje zakleszczenie albo tworzenie procesów bez końca.

```python
def main():
    ...  # zbuduj model i wywołaj train()

if __name__ == "__main__":
    main()
```

Ustawienie `workers=0` również pozwala tego uniknąć kosztem przepustowości.
Warunek jest lepszym rozwiązaniem.

## Dwa problemy, które nie zgłaszają wyjątków

Pozostała część tej strony dotyczy błędów. Te dwa problemy są gorsze, ponieważ
kod działa i zwraca wynik, który wygląda poprawnie.

### Indeksowanie pojedynczego wyniku

Funkcja `predict()` zwraca jeden obiekt `Results` dla jednego obrazu i listę dla
kilku obrazów. Indeksowanie wyniku dla pojedynczego obrazu wybiera *detekcję*, a
nie obraz:

```python
result = model.predict("image.jpg")   # obiekt Results
result.boxes                          # wszystkie detekcje, poprawnie
result[0].boxes                       # JEDNA detekcja, bez ostrzeżenia
```

Nie jest zgłaszany żaden wyjątek, ponieważ indeksowanie obiektu `Results` jest
prawidłową operacją zwracającą podzbiór. Kod napisany z myślą o postaci listy
po cichu zgłasza jedną ramkę na obraz. Należy indeksować tylko wartość, o której
wiadomo, że jest listą.

### Odczytywanie metryk jako atrybutów

Funkcja `val()` zwraca zwykły słownik z kluczami będącymi nazwami metryk, a nie
obiekt z dostępem przez atrybuty:

```python
metrics = model.val(data="coco8.yaml")
metrics["metrics/mAP50-95"]   # poprawnie
metrics.box.map               # AttributeError
```

Klucze mają przestrzenie nazw `metrics/` i `speed/`. Warto raz wyświetlić
słownik, aby zobaczyć wartości utworzone dla danego zadania, ponieważ zestaw
różni się między zadaniami.

## Sprawdzenie zbioru danych przed trenowaniem

Większość niepowodzeń trenowania wynika z problemów ze zbiorem danych.
Polecenie `libreyolo doctor data.yaml` wykonuje kontrolę poprawności zbioru
danych do detekcji i zgłasza wyniki według ważności. Jest to szybsze niż
analizowanie śladu stosu z pierwszej epoki.

```python
from libreyolo import doctor

report = doctor.diagnose("data.yaml", imgsz=640)
if report.errors:
    ...
```

Katalog kontroli zawiera strona [polecenia doctor](/docs/cli/doctor).
