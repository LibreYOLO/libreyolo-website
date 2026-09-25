---
title: Czy YOLO można używać komercyjnie? Licencje YOLOv8, YOLO11 i YOLO26
description: "YOLOv5, YOLOv8, YOLO11 i YOLO26 są udostępniane na licencji AGPL-3.0, więc nie można ich bezpłatnie używać komercyjnie w oprogramowaniu o zamkniętym kodzie źródłowym. Oto wymagania licencji dla poszczególnych wersji oraz alternatywa na licencji MIT."
date: 2026-07-04
author: Xuban
tags: [LibreYOLO, yolo-license, agpl, commercial-use, yolov8, yolo11, yolo26, mit-license]
faq:
  - q: "Czy YOLOv8 można bezpłatnie używać komercyjnie?"
    a: "Nie w produktach o zamkniętym kodzie źródłowym. YOLOv8 jest objęte licencją AGPL-3.0, więc trzeba albo udostępnić cały kod źródłowy aplikacji na licencji AGPL-3.0, albo kupić płatną licencję Enterprise od dostawcy. Do bezpłatnego, nieograniczonego użytku komercyjnego można wybrać bibliotekę na licencji MIT, taką jak LibreYOLO."
  - q: "Na jakiej licencji jest YOLO?"
    a: "Główne wydania, czyli YOLOv5, YOLOv8, YOLO11 i YOLO26, są domyślnie objęte licencją AGPL-3.0. AGPL-3.0 to licencja z silnym copyleftem: jest open source, ale nie pozwala bezpłatnie używać oprogramowania w produkcie własnościowym."
  - q: "Czy można używać YOLO w komercyjnym produkcie o zamkniętym kodzie źródłowym?"
    a: "Nie na licencji AGPL-3.0 bez spełnienia jej warunków, co oznacza udostępnienie całego produktu na licencji AGPL-3.0. Aby zachować prywatność kodu, trzeba kupić licencję Enterprise od dostawcy albo przejść na framework z licencją liberalną, taki jak LibreYOLO (MIT)."
  - q: "Czy istnieje YOLO bez AGPL?"
    a: "Tak. LibreYOLO jest objęte licencją MIT, więc nie ma copyleftu AGPL ani klauzuli dotyczącej użycia przez sieć. Można go bezpłatnie używać w oprogramowaniu komercyjnym i o zamkniętym kodzie źródłowym."
  - q: "Ile kosztuje komercyjna licencja YOLO?"
    a: "Licencja Enterprise od dostawcy jest płatną umową zawieraną z firmą, a jej cena nie jest publiczna. LibreYOLO nic nie kosztuje: licencja MIT zezwala na bezpłatny użytek komercyjny."
  - q: "Czy uruchamianie YOLO wyłącznie na własnych serwerach pozwala uniknąć AGPL?"
    a: "Nie. Sekcja 13 licencji AGPL-3.0 obejmuje interakcję przez sieć. Jeśli użytkownicy korzystają przez sieć ze zmodyfikowanej wersji, trzeba udostępnić im odpowiadający jej kod źródłowy. Uruchamianie oprogramowania wyłącznie na własnych serwerach nie stanowi wyjątku."
---

**Nie. Popularne modele YOLO, czyli YOLOv5, YOLOv8, YOLO11 i nowe YOLO26, nie są bezpłatne do komercyjnego użytku w oprogramowaniu o zamkniętym kodzie źródłowym.** Są udostępniane na licencji **AGPL-3.0**, czyli licencji z silnym copyleftem. Jeśli produkt jest oparty na jednym z nich, trzeba udostępnić całą aplikację na licencji AGPL-3.0 albo kupić płatną licencję Enterprise od dostawcy. Jeśli żadna z tych opcji nie pasuje, istnieje alternatywa z licencją liberalną: **[LibreYOLO](/) jest objęte licencją MIT i można go bezpłatnie używać komercyjnie, bez dodatkowych warunków.**

Jeśli wyszukiwano frazy „czy YOLOv8 można bezpłatnie używać komercyjnie”, „licencja YOLO” lub „YOLO bez AGPL”, ta strona udziela precyzyjnej odpowiedzi dla poszczególnych wersji.

## Przegląd licencji YOLO

| Model | Domyślna licencja | Bezpłatny użytek komercyjny w oprogramowaniu o **zamkniętym kodzie źródłowym**? | Co trzeba zrobić |
| --- | --- | --- | --- |
| YOLOv5 | AGPL-3.0 | Nie | Udostępnić cały kod aplikacji albo kupić licencję Enterprise |
| YOLOv8 | AGPL-3.0 | Nie | Udostępnić cały kod aplikacji albo kupić licencję Enterprise |
| YOLO11 | AGPL-3.0 | Nie | Udostępnić cały kod aplikacji albo kupić licencję Enterprise |
| YOLO26 | AGPL-3.0 | Nie | Udostępnić cały kod aplikacji albo kupić licencję Enterprise |
| **LibreYOLO** | **MIT** | **Tak** | **Nic. Można bezpłatnie używać w produktach własnościowych.** |

W skrócie: główne modele YOLO są open source, ale na licencji **AGPL-3.0**, czyli licencji copyleft, której większość firm nie może spełnić w przypadku produktu własnościowego. LibreYOLO zapewnia ten sam sposób pracy z YOLO na licencji **MIT**, która nie nakłada takiego obowiązku.

## Na jakiej licencji jest YOLO?

Popularne modele YOLO, **YOLOv5, YOLOv8, YOLO11 i najnowszy YOLO26**, są dystrybuowane przez dostawcę na licencji **GNU Affero General Public License v3.0 (AGPL-3.0)**.

AGPL-3.0 to licencja z *silnym copyleftem*. Nie jest to liberalna licencja open source w rodzaju „rób, co chcesz”, jak MIT lub Apache-2.0. Nakłada jeden konkretny, daleko idący obowiązek, który właśnie sprawia trudności użytkownikom komercyjnym.

## Czy YOLOv8 można bezpłatnie używać komercyjnie?

Nie w znaczeniu, które ma na myśli większość osób. YOLOv8 można pobrać i używać komercyjnie, ale tylko po spełnieniu warunków AGPL-3.0. W praktyce oznacza to, że:

- Jeśli produkt zawierający YOLOv8 jest dystrybuowany, udostępniany lub hostowany, trzeba **udostępnić kompletny kod źródłowy tego produktu**, czyli aplikacji, kodu integracyjnego, skryptów i konfiguracji, na licencji AGPL-3.0.
- Obowiązek ten powstaje nie tylko przy dostarczaniu oprogramowania użytkownikom, ale także wtedy, gdy mogą oni korzystać z niego **przez sieć**. To charakterystyczna klauzula z sekcji 13 licencji AGPL. Zaplecze SaaS lub API jej nie omija.

W przypadku hobbystycznego projektu lub badań akademickich zwykle nie stanowi to problemu. W przypadku własnościowego produktu komercyjnego publikowanie całej bazy kodu jest niemal zawsze nie do przyjęcia. Te same zasady dotyczą **YOLO11** i **YOLO26**: ta sama licencja i ten sam obowiązek.

## A co z YOLOv5, YOLO11 i YOLO26?

Obowiązuje ta sama zasada. Wszystkie domyślnie są objęte licencją **AGPL-3.0**. Licencja Enterprise dostawcy wyraźnie obejmuje „YOLO26, earlier YOLO versions, and any future YOLO models”, co wskazuje na celowe i spójne podejście do licencjonowania całej rodziny. Wybór nowszej wersji nie oznacza przyjaźniejszej licencji.

Dla pełnego obrazu: nie *każdy* model z „YOLO” w nazwie podlega AGPL. Niektóre warianty społecznościowe mają licencję GPL-3.0 lub Apache-2.0, a licencja zależy od autora. Jednak wydania, które użytkownicy mają na myśli, wpisując „YOLOv8” lub „YOLO11”, są objęte AGPL-3.0.

## Co dokładnie wymaga AGPL-3.0, w prostych słowach

AGPL-3.0 rozszerza GPL o jeden istotny dodatek: **klauzulę dotyczącą użycia przez sieć**. W praktyce oznacza to, że:

1. **Należy udostępnić kod źródłowy.** Jeśli przekazywane jest oprogramowanie zawierające kod na licencji AGPL, trzeba udostępnić *kompletny odpowiadający mu kod źródłowy* całego połączonego utworu na licencji AGPL-3.0.
2. **Użycie przez sieć również się liczy.** Jeśli użytkownicy wchodzą w interakcję ze zmodyfikowaną wersją przez sieć, na przykład przez aplikację internetową, API lub SaaS, trzeba zaoferować im ten sam kod źródłowy. Nie ma wyjątku w rodzaju „uruchamiamy to tylko na własnych serwerach”.
3. **Copyleft obejmuje kod zależny.** Obowiązek dotyczy *całego utworu zależnego*, a nie tylko plików YOLO. Logika biznesowa również zostaje objęta zakresem AGPL.

To ogólne informacje, a nie porada prawna, dlatego konkretny przypadek należy skonsultować z prawnikiem. Wniosek jest prosty: **AGPL-3.0 nie łączy się z komercyjnym oprogramowaniem o zamkniętym kodzie źródłowym.**

## Opcja licencji Enterprise i jej haczyk

Dostawca oferuje płatną **licencję Enterprise**, która znosi obowiązki AGPL i pozwala osadzić YOLO we własnościowym produkcie bez udostępniania kodu źródłowego. To uzasadniona opcja, która może być właściwa dla niektórych firm.

Trzeba jednak uwzględnić, że:

- **Kosztuje**, wymaga stałej opłaty komercyjnej negocjowanej z każdą firmą, a cena nie jest publiczna.
- **Uzależnia od warunków jednego dostawcy**, które mogą ulec zmianie i obejmują wyłącznie kod tego dostawcy.
- **Płaci się za zgodę** na używanie architektury modelu, która w swojej istocie została opisana w publikacjach naukowych.

Jeśli cykliczna opłata i uzależnienie od dostawcy są akceptowalne, licencja Enterprise się sprawdzi. Jeśli lepiej nie płacić albo całkiem uniknąć tego obowiązku, warto czytać dalej.

## Alternatywa na licencji MIT: LibreYOLO

Ujawnienie: LibreYOLO jest naszym projektem. To również powód, dla którego ta strona może zakończyć się konkretną odpowiedzią.

**LibreYOLO to framework YOLO wydany na liberalnej licencji MIT.** Ten jeden fakt całkowicie zmienia sytuację w przypadku użytku komercyjnego:

- **Bezpłatny użytek komercyjny**, także w produktach własnościowych o zamkniętym kodzie źródłowym.
- **Brak copyleftu i klauzuli sieciowej.** Nie trzeba udostępniać kodu źródłowego aplikacji.
- **Brak opłat za stanowisko lub licencji Enterprise.** MIT oznacza MIT.
- **Znany już sposób pracy**, dzięki czemu migracja jest prosta.

LibreYOLO to prawdziwy, utrzymywany framework, a nie nakładka: obsługuje detekcję obiektów, segmentację, estymację pozy, klasyfikację, estymację głębi i inne zadania w ramach jednego znanego API, a także trenowanie i wbudowany eksport do ONNX/TensorRT/OpenVINO/NCNN. Różnica nie dotyczy sposobu pracy, tylko licencji.

Jeśli chodzi o szersze pytanie „jak odejść od YOLO na licencji AGPL”, w artykule [Najlepsze alternatywy dla Ultralytics w 2026 roku](/articles/best-ultralytics-alternatives) porównano cały ekosystem.

Informacje o licencjach wszystkich pozostałych modeli YOLO, od oryginalnych wersji Darknet z domeny publicznej po YOLOv9, YOLOv10, YOLOv12, YOLOv13 i YOLO26, znajdują się w artykule [Licencje wszystkich modeli YOLO: wyjaśnienie](/articles/yolo-licenses-explained).

### Dlaczego licencja MIT ma znaczenie dla firmy

Licencja MIT pozwala używać, modyfikować, osadzać i sprzedawać oprogramowanie oparte na LibreYOLO **bez obowiązku ujawniania kodu źródłowego i bez opłat**. Właśnie dlatego wiele elementów nowoczesnego stosu oprogramowania jest objętych tą licencją: bezpiecznie nadaje się do zastosowań komercyjnych. Produkt należy do firmy i nie wiążą się z nim żadne dodatkowe zobowiązania.

## Wypróbuj

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

Zachowujesz znany sposób pracy z YOLO: trenowanie na własnych danych, predykcję i eksport przed wdrożeniem. Pozbywasz się obowiązku wynikającego z AGPL i rachunku za licencję Enterprise.

LibreYOLO jest objęte licencją MIT, działa w systemach Linux, Mac i Windows oraz obsługuje GPU, Apple Silicon i zwykły CPU bez zmian w kodzie. Jedno API obejmuje YOLO9, RF-DETR, RTMDet, YOLOX, D-FINE, rodzinę RT-DETR, segmentację, estymację pozy, estymację głębi i inne zadania, a wszystkie te elementy mają licencje liberalne.

Dodaj gwiazdkę na GitHubie: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Dokumentacja: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
