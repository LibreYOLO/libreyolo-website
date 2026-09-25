---
title: "Licencje YOLO: od v1 do YOLO26, kompletny przewodnik na 2026 rok"
description: "Pełna mapa licencji YOLO na 2026 rok: od YOLOv1 do YOLOv13, YOLO26, YOLO-World i YOLOE, oryginalne repozytoria i implementacje na licencjach permisywnych, odnośniki do artykułów i GitHuba oraz informacje o tym, co można faktycznie wdrożyć w komercyjnym produkcie."
date: 2026-07-11
author: Xuban
tags: [LibreYOLO, yolo-license, agpl, gpl, apache, mit-license, yolov9, yolov10, yolo11, yolov12, yolov13, yolo26]
faq:
  - q: "Czy YOLOv9 można bezpłatnie wykorzystywać komercyjnie?"
    a: "To zależy od używanego repozytorium. Repozytorium z artykułem (WongKinYiu/yolov9) jest objęte GPL-3.0. To samo laboratorium udostępnia też osobną implementację YOLOv9 i YOLOv7 na licencji MIT w MultimediaTechLab/YOLO, napisaną po tym, jak użytkownicy komercyjni poprosili o bardziej permisywną opcję. To przepisanie kodu, a nie zmiana licencji plików GPL. Licencja kodu jest jasna, ale w żadnym z repozytoriów nie ma osobnej informacji o licencji wstępnie wytrenowanych wag, dlatego ich pochodzenie należy rozpatrywać osobno."
  - q: "Których wersji YOLO można bezpłatnie używać w zamkniętych produktach komercyjnych?"
    a: "Co najmniej w jednym repozytorium na licencji permisywnej: YOLOv1–YOLOv4 (oryginalny Darknet, domena publiczna), YOLOv7 i YOLOv9 (przepisania laboratorium na MIT), YOLOX i PP-YOLOE (Apache-2.0) oraz implementacje na MIT w LibreYOLO. Na lipiec 2026 roku nie ma permisywnej implementacji YOLOv5, v6, v8, v10, YOLO11, v12, v13, YOLO26, YOLO-World ani YOLOE. Przed podjęciem decyzji sprawdź aktualny plik LICENSE: licencje się zmieniają, a ten artykuł zawiera informacje ogólne, nie poradę prawną."
  - q: "Na jakiej licencji jest YOLOv10?"
    a: "AGPL-3.0. YOLOv10 to wydanie akademickie Uniwersytetu Tsinghua, ale jego kod bazuje na kodzie Ultralytics, dlatego obowiązuje go AGPL-3.0. Nie ma permisywnej implementacji."
  - q: "Na jakiej licencji są YOLOv12 i YOLOv13?"
    a: "Oba modele są objęte AGPL-3.0, co potwierdzają ich pliki LICENSE. Podobnie jak YOLOv10, są opisane w artykułach akademickich, a ich referencyjny kod bazuje na repozytorium Ultralytics, więc AGPL ma zastosowanie niezależnie od autorstwa artykułu. Strony innych podmiotów podające Apache-2.0 jako licencję YOLOv13 są w błędzie."
  - q: "Kiedy Ultralytics zmieniło licencję YOLO na AGPL-3.0?"
    a: "14 kwietnia 2023 roku, a nie w 2022 roku, jak często się powtarza. Plik LICENSE w ultralytics/yolov5 zmieniono w historii tylko trzy razy, a commit wprowadzający AGPL (34cf749, PR #11359) pochodzi z 2023-04-14. Ostatnie wydanie z 2022 roku, YOLOv5 v7.0, nadal było objęte GPL-3.0. Tego samego dnia zmieniono licencję repozytorium ultralytics/ultralytics. YOLOv8 zadebiutowało więc w styczniu 2023 roku na GPL-3.0: wydania PyPI od 8.0.0 do 8.0.76 deklarują GPL-3.0, a pierwszym wydaniem z AGPL-3.0 jest 8.0.80 (16 kwietnia 2023 roku)."
  - q: "Czy YOLO26 można bezpłatnie wykorzystywać komercyjnie?"
    a: "Nie w produktach o zamkniętym kodzie źródłowym. YOLO26 jest objęte AGPL-3.0, a alternatywą jest płatna licencja Enterprise. Obowiązują te same warunki co dla YOLOv8 i YOLO11."
  - q: "Czy oryginalny Darknet YOLO rzeczywiście jest w domenie publicznej?"
    a: "Tak. W pliku LICENSE projektu Darknet Josepha Redmona widnieje zapis 'Darknet is public domain. Do whatever you want with it.' Fork YOLOv4 AlexeyaAB zawiera ten sam tekst o domenie publicznej. Problem polega na tym, że używane obecnie porty PyTorch mają własne licencje, od Apache-2.0, przez AGPL-3.0, aż po brak licencji."
  - q: "Czy można komercyjnie używać wstępnie wytrenowanych wag YOLO-NAS?"
    a: "Nie. Kod super-gradients jest objęty Apache-2.0, ale oficjalne wagi YOLO-NAS podlegają osobnej licencji, która stanowi, że 'may not use the Software for any commercial use, including in connection with any models used in a production environment.'"
  - q: "Czy wagi sieci neuronowej dziedziczą licencję kodu, na którym ją wytrenowano?"
    a: "Ta kwestia nie jest rozstrzygnięta. Ultralytics twierdzi, że AGPL-3.0 obejmuje 'the training code and the models produced by that training code', a jako właściciel praw autorskich ustala warunki dotyczące publikowanych materiałów. Nie sprawdzono jeszcze, czy sąd uznałby wagi wytrenowane samodzielnie za utwór zależny od kodu treningowego. Opublikowane wagi AGPL należy traktować jako objęte ograniczeniami, a przy wczytywaniu ich do permisywnej implementacji zachować ostrożność."
  - q: "Dlaczego różne implementacje tego samego YOLO mogą mieć różne licencje?"
    a: "Prawo autorskie chroni sposób wyrażenia, a nie idee (17 U.S.C. 102(b)). Architekturę opisaną w artykule można niezależnie zaimplementować i objąć licencją wybraną przez autora. Nie można natomiast skopiować plików źródłowych GPL lub AGPL, a następnie zmienić ich licencji. Należy pamiętać, że to argument dotyczący wyłącznie prawa autorskiego: niezależna implementacja nie chroni przed patentami."
---

Każdego roku pojawia się więcej modeli YOLO i każdego roku ponownie pada pytanie o licencje, zwykle na pięć minut przed decyzją produktową. Zamieszanie wynika z tego, że „YOLO” nie oznacza jednego projektu. To nazwa współdzielona przez około dwadzieścia rodzin modeli różnych autorów. Jest tu szczegół, który umyka w większości przewodników po licencjach: **licencja dotyczy repozytorium, a nie modelu.** Ta sama wersja YOLO może jednocześnie występować w repozytorium na GPL i w repozytorium na MIT, nawet u tych samych autorów. Tak jest w przypadku YOLOv9 i zmienia to całą decyzję.

Ten przewodnik przedstawia sytuację repozytorium po repozytorium, z odnośnikami do artykułów i baz kodu. Stan na lipiec 2026 roku. Każdą wymienioną licencję sprawdzono w faktycznym pliku LICENSE danego repozytorium, ponieważ zaskakująco wiele informacji o licencjach YOLO, również w najwyżej pozycjonowanych przewodnikach, jest nieprawdziwych.

Jeśli interesuje Cię tylko kwestia AGPL w Ultralytics, szczegółowo omawia ją artykuł [Czy YOLO można bezpłatnie wykorzystywać komercyjnie?](/articles/yolo-commercial-license). Tutaj znajduje się pełna mapa.

**Dwa zastrzeżenia, zanim zaczniemy.** Po pierwsze, rozwijamy LibreYOLO, bibliotekę na licencji MIT, która konkuruje z kilkoma opisanymi niżej projektami. Ostatnia część artykułu dotyczy właśnie LibreYOLO. To powód, by sprawdzić przedstawione informacje, a nie przyjmować ich bezkrytycznie, dlatego każde twierdzenie o licencji odsyła do źródła pierwotnego. Po drugie, artykuł zawiera ogólne informacje o opublikowanych tekstach licencji i publicznych oświadczeniach, aktualne na wskazaną wyżej datę. Nie jest poradą prawną ani jej zamiennikiem udzielanym przez prawnika w danej jurysdykcji. Licencje i stanowiska opiekunów projektów się zmieniają. Przed podjęciem decyzji o wdrożeniu sprawdź aktualny plik LICENSE każdego repozytorium.

## Tabela licencji

„Liberalna ścieżka” oznacza repozytorium implementujące dany model, którego można użyć w komercyjnym produkcie o zamkniętym kodzie źródłowym, bez udostępniania kodu źródłowego aplikacji i bez opłaty licencyjnej.

| Model | Rok | Oryginalny kod | Liberalna ścieżka | Artykuł |
| --- | --- | --- | --- | --- |
| YOLOv1 | 2015 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (domena publiczna) | Sam oryginał | [arXiv](https://arxiv.org/abs/1506.02640) |
| YOLOv2 | 2016 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (domena publiczna) | Sam oryginał | [arXiv](https://arxiv.org/abs/1612.08242) |
| YOLOv3 | 2018 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (domena publiczna) | Oryginał; uwaga na port PyTorch na AGPL | [arXiv](https://arxiv.org/abs/1804.02767) |
| YOLOv4 | 2020 | [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) (domena publiczna) | Oryginał; [pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) (Apache-2.0) | [arXiv](https://arxiv.org/abs/2004.10934) |
| YOLOv5 | 2020 | [ultralytics/yolov5](https://github.com/ultralytics/yolov5) (AGPL-3.0) | Brak | brak artykułu |
| YOLOv6 | 2022 | [meituan/YOLOv6](https://github.com/meituan/YOLOv6) (GPL-3.0) | Brak | [arXiv](https://arxiv.org/abs/2209.02976) |
| YOLOv7 | 2022 | [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) (GPL-3.0) | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (MIT, to samo laboratorium) | [arXiv](https://arxiv.org/abs/2207.02696) |
| YOLOv8 | 2023 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0; przy premierze GPL-3.0, patrz niżej) | Brak (port Keras został porzucony, patrz niżej) | brak artykułu |
| YOLOv9 | 2024 | [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) (GPL-3.0) | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (MIT, to samo laboratorium) | [arXiv](https://arxiv.org/abs/2402.13616) |
| YOLOv10 | 2024 | [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10) (AGPL-3.0) | Brak | [arXiv](https://arxiv.org/abs/2405.14458) |
| YOLO11 | 2024 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | Brak | brak artykułu |
| YOLOv12 | 2025 | [sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12) (AGPL-3.0) | Brak | [arXiv](https://arxiv.org/abs/2502.12524) |
| YOLOv13 | 2025 | [iMoonLab/yolov13](https://github.com/iMoonLab/yolov13) (AGPL-3.0) | Brak | [arXiv](https://arxiv.org/abs/2506.17733) |
| YOLO26 | 2026 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | Brak | [arXiv](https://arxiv.org/abs/2606.03748) |
| YOLOX | 2021 | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0) | Sam oryginał | [arXiv](https://arxiv.org/abs/2107.08430) |
| PP-YOLOE | 2022 | [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) (Apache-2.0) | Oryginał z PaddleDetection, nie PaddleYOLO | [arXiv](https://arxiv.org/abs/2203.16250) |
| YOLO-NAS | 2023 | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) (kod Apache-2.0, wagi niekomercyjne) | Kod tak, oficjalne wagi nie | brak artykułu |
| YOLO-World | 2024 | [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World) (GPL-3.0) | Brak | [arXiv](https://arxiv.org/abs/2401.17270) |
| YOLOE | 2025 | [THU-MIG/yoloe](https://github.com/THU-MIG/yoloe) (AGPL-3.0) | Brak | [arXiv](https://arxiv.org/abs/2503.07465) |
| MMYOLO | 2022 | [open-mmlab/mmyolo](https://github.com/open-mmlab/mmyolo) (GPL-3.0, projekt wstrzymany od 2024 roku) | Brak | brak artykułu |

Wystarczy czytać kolejne kolumny, by dostrzec oczywisty wniosek: numer wersji nic nie mówi o licencji. Liczy się repozytorium.

## Ten sam model, dwie licencje: przypadek YOLOv9

YOLOv9 to najlepszy dowód na to, że pytanie „na jakiej licencji jest YOLOvN?” jest niewłaściwe. Warto dokładnie przyjrzeć się osi czasu, bo to jedyny przypadek w historii YOLO, gdy nacisk społeczności rzeczywiście zmienił sytuację.

- **18 lutego 2024:** repozytorium [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) zostaje upublicznione razem z [artykułem](https://arxiv.org/abs/2402.13616).
- **22 lutego 2024:** użytkownik otwiera zgłoszenie nr 10 z pytaniem o licencję. Chien-Yao Wang (WongKinYiu) odpowiada „I think it should be GPL3”, a cztery dni później dodaje plik GPL-3.0.
- **26 lutego 2024:** kolejny użytkownik otwiera [zgłoszenie nr 82, „An Apache/MIT rewrite”](https://github.com/WongKinYiu/yolov9/issues/82). W ciągu następnych dwóch i pół tygodnia dołącza tuzin użytkowników komercyjnych.
- **14 marca 2024:** Wang publikuje w tym wątku: *"Okay I create a new repo for mit rewrite... I think I can handle most of implementation of architectures and loss functions. And need someone to give great help about dataloader and ddp training."*

To repozytorium, początkowo nazwane `yolov9mit`, to obecnie **[MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO)**: licencja MIT, właściciele praw autorskich „Kin-Yiu, Wong and Hao-Tang, Tsui”, opis projektu: *"the official implementation of YOLOv7 and YOLOv9, YOLO-RD."* Większość przepisanego kodu nie powstała dzięki wolontariuszom z wątku, lecz za sprawą Hao-Tanga Tsui, współpracownika Wanga z laboratorium, autora około 90 procent commitów.

YOLOv9 jest więc jednocześnie „nie do wdrożenia” i „do wdrożenia”, zależnie od sklonowanego repozytorium. Ta sama architektura, to samo laboratorium, dwie licencje.

**Zanim oprzesz na tym produkt, poznaj trzy zastrzeżenia, których repozytorium nie podaje na stronie głównej:**

1. **Projekt jest nadal rozwijany.** W otwartym [zgłoszeniu](https://github.com/MultimediaTechLab/YOLO/issues/231) z lutego 2026 roku opisano, że udostępnione checkpointy nie odtwarzają wyników COCO z artykułu i mają zauważalnie więcej parametrów, niż podano w artykule. Trenowanie segmentacji i punktów kluczowych [nie zostało jeszcze zaimplementowane](https://github.com/MultimediaTechLab/YOLO/issues/232). Od oznaczenia v1.0 w grudniu 2025 roku nie scalono niczego z gałęzią main.
2. **Wagi nie mają osobnej informacji o licencji.** Zgodnie z przyjętą praktyką licencja MIT repozytorium obejmuje jego zasoby wydania, a dowody techniczne wskazują, że checkpointy wytrenowano w tym projekcie, a nie skopiowano z repozytorium GPL: pliki różnią się rozmiarem i liczbą parametrów od checkpointów GPL. Żaden dokument nie stwierdza jednak, że pliki .pt podlegają MIT, a społecznościowy [wniosek o formalny audyt pod kątem skażenia AGPL](https://github.com/MultimediaTechLab/YOLO/issues/51) pozostaje otwarty. To luka w dokumentacji: nikt nie ogłosił, że kod jest „zweryfikowany jako czysty”, ale nikt też nie zgłosił problemu.
3. **To inna baza kodu**, a nie zamiennik skryptów z repozytorium GPL, który można wstawić bez zmian.

To nie znaczy, że projekt jest bezużyteczny. Należy go ocenić, zamiast traktować jako pozycję do odhaczenia.

## Era 1: lata domeny publicznej (YOLOv1–YOLOv4)

Joseph Redmon opublikował [YOLOv1](https://arxiv.org/abs/1506.02640), [YOLOv2](https://arxiv.org/abs/1612.08242) (znany jako artykuł YOLO9000) oraz [YOLOv3](https://arxiv.org/abs/1804.02767) w ramach frameworka Darknet. Plik [LICENSE](https://github.com/pjreddie/darknet/blob/master/LICENSE) jest znany z tego, że ma tylko trzy wiersze:

> 0. Darknet is public domain.
> 1. Do whatever you want with it.
> 2. Stop emailing me about it!

To rzeczywiste przekazanie do domeny publicznej. [YOLOv4](https://arxiv.org/abs/2004.10934), którego opiekunem jest Alexey Bochkovskiy w [forku Darknet](https://github.com/AlexeyAB/darknet), zawiera ten sam plik, a nie Unlicense przypisywaną mu w kilku przewodnikach po licencjach. Własny klasyfikator GitHuba nie rozpoznaje tej licencji i raportuje „Other”. Warto o tym wiedzieć, jeśli narzędzia zgodności korzystają z tego pola: skaner oznaczy ją jako nierozpoznaną, a nie jako permisywną, choć trudno o bardziej permisywny tekst.

Sam Darknet nie jest całkiem martwy. Repozytorium Redmona nie było aktualizowane od 2022 roku, ale README AlexeyaAB wskazuje teraz [hank-ai/darknet](https://github.com/hank-ai/darknet) jako zalecanego, aktywnie rozwijanego następcę w C/C++.

**Pułapka tkwi w portach.** W 2026 roku prawie nikt nie trenuje modeli w Darknet. Zamiast tego używa implementacji w PyTorch, które mają własne licencje:

- [ultralytics/yolov3](https://github.com/ultralytics/yolov3), najpopularniejszy port YOLOv3, jest objęty **AGPL-3.0**. Architektura z domeny publicznej, ponownie rozpowszechniana na jednej z najbardziej restrykcyjnych licencji w tej dziedzinie. Otrzymana licencja zależy wyłącznie od tego, czyj port zainstalowano przez pip.
- [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) jest objęty **GPL-3.0**.
- [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) jest objęty **Apache-2.0**, więc permisywna ścieżka dla v4 jest dostępna również w PyTorch.
- [WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4), projekt współautora YOLOv4, nie ma **żadnego pliku LICENSE**. To gorsza sytuacja niż copyleft: bez licencji nie są przyznane żadne prawa, a domyślnie obowiązuje pełna ochrona praw autorskich.

Sprawdź licencję sklonowanego repozytorium, a nie numer wersji w artykule.

## Era 2: GPL-3.0 (YOLOv6, v7, v9)

[YOLOv6](https://github.com/meituan/YOLOv6) (Meituan), [YOLOv7](https://github.com/WongKinYiu/yolov7) i [YOLOv9](https://github.com/WongKinYiu/yolov9) wydano na GPL-3.0.

GPL-3.0 to copyleft: przy rozpowszechnianiu zmodyfikowanej wersji albo połączeniu kodu GPL z własnym kodem w jeden program trzeba udostępnić pełny odpowiadający mu kod źródłowy tego połączonego utworu na GPL-3.0. Istotne są dwie granice. **Samo zestawienie** oznacza, że osobne, niezależne programy dostarczone razem nie stają się automatycznie jednym utworem. Drugą granicą jest **rozpowszechnianie**: w odróżnieniu od AGPL zwykła GPL nie nakłada obowiązku ujawnienia kodu, jeśli połączony utwór nigdy nie jest przekazywany dalej. Dlatego niektóre firmy używają modelu GPL wyłącznie za własnym API.

Ta ścieżka jest węższa, niż się wydaje, i ma trzy pułapki, których rzadko uwzględnia się w planach. Gdy model trafia na urządzenie klienta lub do instalacji lokalnej, dochodzi do rozpowszechniania. Określenie „wyłącznie do użytku wewnętrznego” przestaje być prawdziwe, gdy kopia opuszcza podmiot prawny: przekazanie kompilacji wykonawcy, zewnętrznemu zespołowi programistów albo odrębnej spółce powiązanej stanowi rozpowszechnianie, choć kopiowanie wewnątrz jednej firmy nim nie jest. Ochrona działa tylko wtedy, gdy *wszystkie* elementy dostępne przez API są objęte GPL lub licencją bardziej permisywną, ponieważ pojedynczy komponent AGPL w udostępnianym utworze powoduje zastosowanie sekcji 13, niezależnie od treści pliku GPL.

**YOLOv7 i YOLOv9 mają opisaną wyżej alternatywę na MIT.** YOLOv6 jej nie ma. Najmocniejszym dowodem jest działanie Baidu: PaddleDetection jest objęte Apache-2.0, a jego opiekunowie celowo odseparowali YOLOv5, YOLOv6, YOLOv7 i YOLOv8 w osobnym repozytorium GPL-3.0, [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO), stwierdzając, że kod tych modeli „will not be merged into PaddleDetection”. Jeśli firma tak duża jak Baidu stawia barierę, by nie dopuścić YOLOv6 do swojego permisywnie licencjonowanego frameworka, wiele mówi to o konsekwencjach tej licencji.

## Era 3: AGPL-3.0 (YOLOv5, v8, v10, 11, v12, v13, 26, YOLOE)

Ultralytics przeszło na AGPL-3.0 **14 kwietnia 2023 roku**, a nie w 2022 roku, jak często się powtarza, także w przewodnikach z pierwszej strony wyników wyszukiwania. Wszystkie późniejsze modele ([YOLOv8](https://github.com/ultralytics/ultralytics), YOLO11 i [YOLO26](https://arxiv.org/abs/2606.03748)) są objęte AGPL-3.0, a alternatywą jest płatna licencja Enterprise. Na [stronie licencji](https://ultralytics.com/license) podano, że poziom Enterprise obejmuje „YOLO26, earlier YOLO versions, and any future YOLO models”, więc polityka jest celowa i obejmuje przyszłe modele. Ceny nie są publicznie dostępne.

Tę datę można sprawdzić i warto to zrobić, ponieważ popularna wersja tej historii jest błędna w istotnym szczególe:

- W historii plik `LICENSE` w `ultralytics/yolov5` zmieniono dokładnie trzy razy. Trzecia zmiana to commit [`34cf749`](https://github.com/ultralytics/yolov5/commit/34cf749958d2dd3ed1205f6bb07e0f20f6e2372d) „Update LICENSE to AGPL-3.0 (#11359)” z datą **2023-04-14**. Zmienia on `GNU GENERAL PUBLIC LICENSE` na `GNU AFFERO GENERAL PUBLIC LICENSE` w 101 plikach.
- Ostatnie wydanie YOLOv5 z 2022 roku, **v7.0 (22 listopada 2022 roku), nadal było objęte GPL-3.0**, podobnie jak wcześniejsze v6.2, v6.1 i v6.0. Tak samo jest w przypadku tysięcy forków, które przestały być synchronizowane przed kwietniem 2023 roku: GitHub nadal podaje dla dowolnego z nich GPL-3.0.
- Tego samego dnia, 41 minut później, w repozytorium `ultralytics/ultralytics` [wprowadzono tę samą zmianę](https://github.com/ultralytics/ultralytics/commit/2c6fc0a4443b9cf805ef17b1cfdd71a98693b4d4) (PR #2031).

Wynika z tego mało znany fakt: **YOLOv8 nie zadebiutowało na AGPL.** Wydano je w styczniu 2023 roku na **GPL-3.0**, a zmianę licencji wprowadzono trzy miesiące później. Dane PyPI są jednoznaczne: `ultralytics` od wersji 8.0.0 (10 stycznia 2023 roku) do 8.0.76 (13 kwietnia 2023 roku) deklaruje `GPL-3.0`. Pierwszą wersją deklarującą `AGPL-3.0` jest 8.0.80, opublikowana 16 kwietnia 2023 roku.

To nie jest luka w licencji. Udzielona licencja nie zostaje cofnięta z mocą wsteczną, więc stare wersje pozostają dostępne na warunkach obowiązujących w chwili publikacji. Są jednak przestarzałe o trzy lata, nieutrzymywane i niezałatane. GPL-3.0 nadal oznacza copyleft (klauzulę sieciową zastąpiono klauzulą dotyczącą rozpowszechniania, ale copyleft nie znika), a stanowisko komercyjne Ultralytics jest szersze niż sam tekst licencji. Praktyczny, bardziej ogólny wniosek jest taki: **licencja odnosi się do otrzymanej wersji, a nie do nazwy projektu.** Przypinaj zależności i zapisuj treść licencji z dnia pobrania kopii, bo projekt może ją zmienić następnego dnia, a kwestia zgodności zależy od tego, którą wersję faktycznie otrzymano.

AGPL-3.0 to GPL uzupełniona sekcją 13, klauzulą sieciową: *jeśli zmodyfikowano Program*, użytkownikom korzystającym z tej zmodyfikowanej wersji przez sieć trzeba udostępnić jej kod źródłowy. To zamyka ścieżkę SaaS, którą pozostawia zwykła GPL.

Warto zwrócić uwagę na to zastrzeżenie, ponieważ wiele artykułów je pomija. Sekcja 0 AGPL definiuje „modyfikację” jako kopiowanie utworu lub jego adaptację *w sposób wymagający zgody na podstawie prawa autorskiego*. Uruchamianie niezmodyfikowanego kodu treningowego na własnym zbiorze danych oznacza wykonanie programu, a nie adaptację jego źródeł. Podobnie skompilowanie własnego kodu za pomocą niezmodyfikowanego kompilatora GPL nie modyfikuje kompilatora. Samo trenowanie nie uruchamia więc automatycznie tego obowiązku. Kto twierdzi inaczej, pominął tę definicję.

Zakres licencji obejmuje połączenie lub osadzenie pakietu we własnej bazie kodu w taki sposób, że oba elementy są rozpowszechniane lub udostępniane jako jeden program. Według interpretacji FSF połączenie utworu GPL lub AGPL z innymi modułami za pomocą linkowania tworzy połączony utwór. Nie sprawdzono jeszcze, czy sąd rozszerzyłby tę interpretację na import w Pythonie. Warto również zauważyć, że własne [warunki komercyjne](https://ultralytics.com/license) Ultralytics wymagają licencji Enterprise dla platform SaaS, API i systemów chmurowych korzystających z YOLO w tle. Stanowisko dostawcy jest zatem szersze niż sam tekst licencji. Integrację z produktem udostępnianym lub rozpowszechnianym należy traktować jako ryzyko. Nie należy zakładać, że samo trenowanie w odosobnieniu je powoduje.

W przypadku YOLO26 daty są wszędzie mylone, dlatego: model **zaprezentowano we wrześniu 2025 roku** podczas YOLO Vision, **wydano w styczniu 2026 roku** (wydanie `ultralytics` 8.4.0, którego informacje o wydaniu mówią „Ultralytics YOLO26 has arrived”), a **artykuł opublikowano w czerwcu 2026 roku**. Model znajduje się w głównym repozytorium `ultralytics/ultralytics`; `ultralytics/yolo26` to tylko strona informacyjna.

**Zaskakujący szczegół:** akademickie YOLO z ostatnich dwóch lat również są objęte AGPL, i to nie dlatego, że napisało je Ultralytics.

- [YOLOv10](https://arxiv.org/abs/2405.14458) pochodzi z Uniwersytetu Tsinghua. W README napisano *"The code base is built with ultralytics."*
- [YOLOv12](https://arxiv.org/abs/2502.12524): *"The code is based on ultralytics."*
- [YOLOv13](https://arxiv.org/abs/2506.17733): *"The code is based on Ultralytics."*
- [YOLOE](https://arxiv.org/abs/2503.07465), model z otwartym słownikiem, który „widzi wszystko”, stworzony przez zespół YOLOv10, również podlega AGPL-3.0 i bazuje na Ultralytics.

Copyleft jest dziedziczony: utwór zależny od kodu AGPL musi być rozpowszechniany na AGPL, więc gdy tylko jeden z tych forków zostanie rozpowszechniony lub udostępniony, jego licencja ma zastosowanie. (Jeśli wprowadzono modyfikacje, ale kod pozostaje wyłącznie do własnego użytku, obowiązek nie powstaje, dlatego użycie badawcze pozostaje bez zmian). Badania są niezależne, licencja nie. Od 2024 roku publikowanie „kolejnego YOLO” jako utworu zależnego od Ultralytics stało się domyślną praktyką, więc AGPL automatycznie przechodzi teraz na kolejne wersje.

Strona twierdząca, że YOLOv13 jest objęte Apache-2.0, podaje błędną informację. Plik LICENSE, API GitHuba i karta modelu wskazują AGPL-3.0.

### Liberalna ścieżka dla YOLOv8, która już nie istnieje

Wiele nadal dostępnych przewodników, w tym wcześniejszy szkic tego artykułu, wskazuje YOLOv8 z KerasCV na Apache-2.0 jako prostą alternatywę dla YOLOv8 na AGPL. **Obecnie nie można na niej polegać.** Oto publicznie dostępne informacje:

- W dyskusji KerasCV [Clarifications regarding YOLOv8 licensing](https://github.com/keras-team/keras-cv/discussions/2032) jeden z uczestników napisał, że „many parts are derived form ultralytics”. Żaden opiekun projektu nie odpowiedział, więc kwestia niezależności implementacji nie została publicznie rozstrzygnięta.
- Zgłoszenie KerasHub [add YOLOV8](https://github.com/keras-team/keras-hub/issues/1760) zamknął w lipcu 2025 roku opiekun Keras z komentarzem: „Hey all!! Because of license issues, we have decided to drop this.” Dwie osoby zapytały później, co to oznacza dla dotychczasowych użytkowników komercyjnych wersji KerasCV. Żadna nie otrzymała odpowiedzi.
- KerasCV jest teraz zarchiwizowane, a jego utrzymywany następca, KerasHub, **nie udostępnia YOLOv8**.

Nie wiadomo, czy te dwie sprawy są ze sobą powiązane, i nie twierdzimy, że doszło do czegoś niewłaściwego. Nie jest to jednak solidna podstawa dla produktu.

## Liberalne wyjątki: YOLOX, PP-YOLOE i YOLO-NAS

Nie każdy model YOLO dołączył do ruchu copyleft.

- **[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)** (Megvii, 2021) jest objęte Apache-2.0 bez wyjątków, zarówno kod, jak i wagi. Nadal jest to najczystszy klasyczny model YOLO do zastosowań komercyjnych, choć repozytorium nie było aktualizowane od połowy 2025 roku, a [pytanie o dołączenie wstępnie wytrenowanych wag do aplikacji komercyjnej](https://github.com/Megvii-BaseDetection/YOLOX/issues/1865) pozostaje bez odpowiedzi od marca 2026 roku. Treść licencji jest jednoznaczna. Po prostu nie ma już nikogo, kto mógłby odpowiedzieć na pytania. Opisaliśmy [uruchamianie YOLOX z LibreYOLO](/articles/yolox-with-libreyolo).
- **[PP-YOLOE](https://arxiv.org/abs/2203.16250)** (Baidu) jest objęte Apache-2.0 **w ramach PaddleDetection**. Należy korzystać właśnie z niego, a nie z PaddleYOLO, które zawiera niemal identyczny katalog konfiguracji i jest objęte GPL-3.0. Ten sam model, ta sama firma, dwa repozytoria, dwie licencje. To odwrotna sytuacja niż w przypadku YOLOv9.
- **[YOLO-NAS](https://github.com/Deci-AI/super-gradients)** (Deci, 2023) to przypadek, który często zaskakuje. Kod jest objęty Apache-2.0, oficjalne wagi już nie. W osobnej [licencji](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md) zapisano, że *"may not use the Software for any commercial use, including in connection with any models used in a production environment."* Od przejęcia Deci przez NVIDIA w 2024 roku nie znaleźliśmy nowych prac nad funkcjami w repozytorium, a pierwotne adresy wag przestały działać (checkpointy są teraz na innym serwerze). Czasem sugeruje się obejście, według którego trenowanie architektury od zera pozwala zachować zgodność z Apache, ale Deci ani NVIDIA tego nie potwierdziły. Jest to też trudne do pogodzenia z tekstem licencji obejmującym „any components comprising the model”. Więcej o YOLO-NAS [tutaj](/articles/yolo-nas-with-libreyolo).
- **[MMYOLO](https://github.com/open-mmlab/mmyolo)** bywa w starszych przewodnikach określane jako Apache-2.0. To nieprawda: MMYOLO jest objęte GPL-3.0 (jego siostrzane MMDetection jest objęte Apache), a od lipca 2024 roku nie wprowadzono tam żadnych zmian.
- **[YOLO-World](https://github.com/AILab-CVC/YOLO-World)** (Tencent) podlega **GPL-3.0**, a nie AGPL ani licencji permisywnej. Warto to podkreślić, ponieważ często zakłada się, że modele YOLO z otwartym słownikiem są permisywnie licencjonowane, bo korzystają z ekosystemu podobnego do CLIP.

Te przypadki pokazują trzy najczęstsze błędy dotyczące licencji w tej dziedzinie: założenie, że wagi podlegają tej samej licencji co kod, że wszystkie projekty danej organizacji mają tę samą licencję oraz że licencję można wywnioskować z nazwy modelu.

## Licencje obejmują kod, nie idee (ale trzeba uważać na patenty)

Jedna zasada stanowi podstawę prawną wszystkich opisanych wyżej permisywnych ścieżek: **prawo autorskie chroni sposób wyrażenia, a nie idee.** To nie slogan, lecz jednoznaczna norma prawna po obu stronach Atlantyku.

W Stanach Zjednoczonych jest to [17 U.S.C. 102(b)](https://www.law.cornell.edu/uscode/text/17/102), które wyłącza spod ochrony prawnoautorskiej wszelkie „idea, procedure, process, system, method of operation, concept, principle, or discovery”, zgodnie z zasadą sięgającą sprawy *Baker v. Selden* (1879). W Unii Europejskiej, której prawo nas dotyczy i prawdopodobnie dotyczy też wielu czytelników, art. 1 ust. 2 [dyrektywy oprogramowania (2009/24/WE)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32009L0024) stanowi, że „ideas and principles which underlie any element of a computer program” nie podlegają ochronie. Trybunał Sprawiedliwości potwierdził w sprawie *SAS Institute przeciwko World Programming* ([C-406/10](https://curia.europa.eu/juris/liste.jsf?num=C-406/10), 2012), że funkcjonalność programu, jego język programowania i formaty danych same w sobie nie są chronionym sposobem wyrażenia. Ochrona obejmuje wyłącznie kod.

Architektura opisana w artykule na arXiv to opublikowana idea. Każdy może ją zaimplementować. Nie można natomiast kopiować ani adaptować plików źródłowych GPL/AGPL i zmieniać licencji. Niezależna implementacja napisana na podstawie artykułu, a nie kodu źródłowego, jest nowym utworem, którego autor wybiera licencję.

Dwa istotne zastrzeżenia, które pomija większość dostawców:

- **Implementacja musi być rzeczywiście „niezależna”.** Można to wiarygodnie wykazać przez proces clean-room: osoby piszące kod korzystają z artykułu i specyfikacji, a nie ze źródeł objętych copyleft. Dokładne czytanie repozytorium GPL, a następnie „przepisywanie go” to co innego. Różnica ma znaczenie, jeśli ktokolwiek kiedyś ją zbada.
- **To argument dotyczący prawa autorskiego, a nie patentów.** Niezależne wynalezienie *nie* chroni przed patentem. Implementacja clean-room rozstrzyga kwestię prawa autorskiego, ale nie zmienia sytuacji patentowej. Obecnie nie ma powszechnie wskazywanego patentu obejmującego architektury YOLO, ale „nikt jeszcze nie pozwał” nie oznacza „nie ma podstaw do pozwu”.

## Wagi podlegają osobnej licencji

Najkosztowniejszym błędem w tej dziedzinie jest audyt kodu z pominięciem checkpointu. Licencje mogą obejmować trzy warstwy i każda z nich może mieć inne warunki:

**1. Kod.** Wszystko, co omówiono wyżej.

**2. Wagi.** Czasem licencja jest podana wprost, jak w przypadku YOLO-NAS. Czasem jest to deklaracja: [strona licencji Ultralytics](https://ultralytics.com/license) twierdzi, że AGPL-3.0 „covers the training code and the models produced by that training code” i obejmuje też modele wytrenowane samodzielnie za pomocą tego kodu. Jako właściciel praw autorskich do publikowanych materiałów Ultralytics może ustalać warunki dla własnych checkpointów. Kwestia tego, czy wyprodukowane samodzielnie wagi są prawnie utworem zależnym od kodu treningowego, pozostaje otwarta: nie rozstrzygnięto jej w orzecznictwie, a stoi ona w sprzeczności z ogólną zasadą FSF, według której wynik działania programu nie jest automatycznie objęty prawami autorskimi do samego programu. Brak rozstrzygnięcia nie oznacza jednak bezpieczeństwa. W praktyce opublikowane wagi AGPL należy traktować jako objęte ograniczeniami.

Istotne ryzyko dla osób korzystających z zaleceń tego artykułu: **repozytoria GPL z YOLOv7 i YOLOv9 w ogóle nie określają licencji wag.** Ich checkpointy są zasobami wydań w repozytorium GPL-3.0, bez odrębnego zezwolenia. Wczytanie tych plików .pt do implementacji na MIT oznacza czysty kod, ale status checkpointu pozostaje niewyjaśniony. Jeśli powodem zmiany była licencja, należy użyć wag wytrenowanych przez projekt z permisywną licencją albo wytrenować własne.

**3. Dane treningowe.** *Adnotacje* COCO są objęte CC BY 4.0, więc etykiety nie sprawiają problemu. *Zdjęcia* nie należą do COCO i nie są przez nie licencjonowane: to zdjęcia z Flickr, objęte różnymi warunkami, do których konsorcjum COCO wyraźnie nie ma praw. Większość branży uznaje to za małe ryzyko i idzie dalej, ale stwierdzenie „COCO jest w porządku” dotyczy wyłącznie adnotacji. Poza COCO ograniczenia szybko rosną: wiele zbiorów danych lotniczych, medycznych i drogowych nie dopuszcza użycia komercyjnego, a permisywna licencja kodu modelu tego nie zmienia.

Za każdym razem należy sprawdzać wszystkie trzy warstwy.

## Kilka słów o MIT i Apache-2.0

Utrzymujemy bibliotekę na MIT, więc wygodnie byłoby stwierdzić, że MIT to po prostu najlepsza licencja. Uczciwe porównanie jest ciekawsze.

MIT nie oznacza braku obowiązków: w rozpowszechnianych kopiach trzeba zachować informację o prawach autorskich i tekst licencji. To rzeczywisty warunek, choć mało uciążliwy. Apache-2.0 ma coś, czego brakuje MIT: **wyraźne zezwolenie patentowe** od każdego współtwórcy i klauzulę odwetu, która odbiera to zezwolenie każdemu, kto wniesie pozew dotyczący patentu związanego z utworem. MIT milczy na temat patentów. Dla firmy, która najbardziej obawia się ryzyka patentowego, Apache-2.0 może być *bezpieczniejszą* licencją. Z tego właśnie powodu wiele projektów na licencjach permisywnych do detekcji (YOLOX, RT-DETR, D-FINE, DEIM) stosuje Apache-2.0.

MIT zapewnia prostotę i zgodność. Żadna z tych licencji nie zmusi do opublikowania kodu źródłowego. To właśnie na tej kwestii skupia się ten artykuł. Jeśli ktoś twierdzi, że ważniejszy jest wybór między MIT a Apache, próbuje coś sprzedać. Kluczowy wybór dotyczy licencji permisywnej albo copyleft.

## Którego YOLO można faktycznie użyć?

- **Komercyjny produkt o zamkniętym kodzie źródłowym:** YOLOv7 lub YOLOv9 przez repozytorium laboratorium na [MIT](https://github.com/MultimediaTechLab/YOLO), YOLOX albo PP-YOLOE (z PaddleDetection) spośród oryginalnych projektów, ewentualnie utrzymywany framework na MIT, taki jak LibreYOLO, jeśli wolisz uniknąć samodzielnego łączenia i audytowania komponentów. Należy unikać GPL i AGPL, chyba że kod całej aplikacji zostanie udostępniony albo zostanie kupiona licencja Enterprise.
- **Projekt open source:** dowolny, o ile jego licencja jest zgodna. Jeśli projekt jest na AGPL, naturalnym wyborem jest linia Ultralytics, a najnowsze badania (YOLOv13, YOLO26, YOLOE) trafiają najpierw właśnie tam.
- **Badania i benchmarki:** licencje rzadko ograniczają takie zastosowanie. Należy użyć modelu zgodnego z tym, którego wyniki są porównywane.
- **Plan „zajmiemy się tym później”:** nie zadziała. Usunięcie zależności AGPL po zbudowaniu produktu oznacza ponowne trenowanie, walidację i eksport wszystkiego, łącznie z wagami. Najpierw należy wybrać repozytorium, a tym samym licencję.

Porównanie samych frameworków, a nie ich licencji, znajduje się w artykule [Najlepsze alternatywy dla Ultralytics w 2026 roku](/articles/best-ultralytics-alternatives).

## Opcja MIT: co udostępnia LibreYOLO

Informacja: LibreYOLO to nasz projekt, a jego licencja jest powodem powstania tego artykułu.

**[LibreYOLO](https://github.com/LibreYOLO/libreyolo) to jedna baza kodu na licencji MIT**, która udostępnia poniższe detektory przez jedno API. Obejmuje trenowanie i wbudowany eksport do ONNX, TensorRT, OpenVINO oraz NCNN. Bez copyleft, klauzuli sieciowej ani poziomu Enterprise.

Poniżej znajduje się pełna lista modeli do detekcji obiektów wraz z projektami źródłowymi i ich licencjami. Dzięki temu można dokładnie sprawdzić, skąd pochodzi każda permisywna ścieżka:

| W LibreYOLO | Model | Projekt źródłowy | Licencja projektu źródłowego |
| --- | --- | --- | --- |
| `LibreYOLO2`, `LibreYOLO3`, `LibreYOLO4` | Modele YOLO z okresu Darknet, w PyTorch | [pjreddie/darknet](https://github.com/pjreddie/darknet), [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) | Domena publiczna |
| `LibreYOLO7` | YOLOv7 | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT (przepisanie laboratorium, nie repozytorium GPL) |
| `LibreYOLO9`, `LibreYOLO9E2E` | YOLOv9 i wariant end-to-end bez NMS | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT (przepisanie laboratorium, nie repozytorium GPL) |
| `LibreYOLO9P2` | YOLOv9 z głowicą o kroku 4 do małych obiektów | Oryginalny projekt LibreYOLO | MIT |
| `LibreYOLOX` | YOLOX | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache-2.0 |
| `LibreYOLONAS` | YOLO-NAS, detekcja i estymacja pozy | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) | Kod Apache-2.0. Ograniczone wagi źródłowe nie są redystrybuowane |
| `LibreRTDETR`, `LibreRTDETRv2` | RT-DETR i v2 | [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | Apache-2.0 |
| `LibreRTDETRv4` | RT-DETRv4 | [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | Apache-2.0 |
| `LibreRFDETR` | RF-DETR: detekcja, segmentacja, estymacja pozy, OBB | [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | N/S/M/L na Apache-2.0. XL/2XL podlegają Platform Model License Roboflow, dlatego udostępniamy tylko wersje objęte Apache |
| `LibreDFINE` | D-FINE | [Peterande/D-FINE](https://github.com/Peterande/D-FINE) | Apache-2.0 |
| `LibreDEIM` | DEIM | [Intellindust-AI-Lab/DEIM](https://github.com/Intellindust-AI-Lab/DEIM) | Apache-2.0 |
| `LibreDEIMv2` | DEIMv2 | [Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) | Kod Apache-2.0. Większe warianty używają backbone DINOv3 firmy Meta na jej własnej licencji niezatwierdzonej przez OSI. Zezwala ona na redystrybucję, ale **zabrania użycia w wojsku, energetyce jądrowej, szpiegostwie i przy produkcji broni**. Te rozmiary są publikowane z oznaczeniem `other`, a nie Apache. Przed wdrożeniem w zastosowaniu podwójnego przeznaczenia należy zapoznać się z licencją |
| `LibreRTMDet` | RTMDet ([bez MMDetection](/articles/rtmdet-without-mmdetection)) | [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) | Apache-2.0 (siostrzane repozytorium na Apache, a nie mmyolo na GPL-3.0) |
| `LibrePICODET` | PP-PicoDet | Architektura z [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), checkpointy z ponownego portu [Picodet_Pytorch](https://github.com/Bo396543018/Picodet_Pytorch) | Apache-2.0 (oba projekty) |
| `LibreEC` | EdgeCrafter: detekcja, estymacja pozy, segmentacja | [Intellindust-AI-Lab/EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter) | Apache-2.0 |
| `LibreFOMO` | Detektor centroidów dla sprzętu klasy mikrokontrolera | Oryginalny projekt LibreYOLO | MIT |
| `LibreGroundingDINO` | Grounding DINO, otwarty słownik (inferencja) | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | Apache-2.0 |
| `LibreOWLv2` | OWLv2, otwarty słownik (inferencja) | Google, za pośrednictwem `transformers` | Apache-2.0 |

Dwie kwestie w tej tabeli przedstawiono celowo bez zatajania ograniczeń.

**W bazie kodu nie skopiowano żadnych źródeł GPL ani AGPL.** Wszystkie wymienione rodziny bazują na projektach źródłowych z domeny publicznej, MIT lub Apache-2.0, co pozwala zachować licencję MIT frameworka. Tam, gdzie architektura ma zarówno permisywną, jak i copyleftową wersję, korzystamy z tej permisywnej: YOLOv9 i YOLOv7 pochodzą z repozytorium laboratorium na MIT, a nie z oryginalnych repozytoriów GPL. RTMDet pochodzi z MMDetection na Apache, a nie z MMYOLO na GPL. YOLO-World nie znalazło się na liście mimo częstych pytań, bo jest objęte GPL-3.0 i nie ma permisywnego źródła, z którego można byłoby skorzystać.

**Liberalna licencja nie oznacza braku ograniczeń. Wolimy o tym powiedzieć, niż pozwolić odkryć to podczas audytu.** Większe warianty DEIMv2 podlegają warunkom Meta DINOv3, w tym zakazowi użycia w wojsku, energetyce jądrowej, szpiegostwie i przy produkcji broni, co jest realnym ograniczeniem w zastosowaniach podwójnego przeznaczenia. Kilka checkpointów z wersji poglądowych do badań wytrenowano na zbiorach danych niekomercyjnych (DOTA, VisDrone) i oznaczono na Hugging Face jako `cc-by-nc`, zamiast udostępniać je bez informacji o ograniczeniach. Do naszych wag stosuje się ta sama ostrożność co do wag innych projektów: checkpointy YOLO9 konwertujemy z checkpointów repozytorium laboratorium na MIT, więc obowiązują je te same warunki, które opisano wyżej jako „deklarowane, ale niepoddane formalnemu audytowi”. Licencja MIT obejmuje napisany przez nas kod. Każdy checkpoint podlega warunkom danych i źródeł, z których powstał, publikowanym osobno dla poszczególnych wag.

Poza detekcją to samo API obejmuje segmentację instancji i semantyczną, estymację pozy, obrócone ramki, klasyfikację (MobileNetV4, ConvNeXt, EfficientNetV2, ResNet, CLIP), estymację głębi z pojedynczego obrazu, przywracanie obrazu i estymację kierunku wzroku.

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
```

Ten sam sposób pracy z YOLO, bez sprawdzania licencji.

Dodaj gwiazdkę w GitHubie: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Dokumentacja: [libreyolo.com/docs](https://libreyolo.com/docs)

---

*Znaki towarowe i powiązania: YOLO, YOLOv8, YOLO11, YOLO26 i Ultralytics są znakami towarowymi ich właścicieli, w tym Ultralytics Inc. LibreYOLO to niezależny projekt, który nie jest powiązany z Ultralytics ani inną organizacją wymienioną w tym artykule, nie jest przez nie sponsorowany ani przez nie rekomendowany. Nazwy produktów i repozytoriów służą wyłącznie do identyfikacji i porównania omawianego oprogramowania.*

*Ten artykuł zawiera ogólne informacje, nie poradę prawną. Jest aktualny na 11 lipca 2026 roku i napisał go jeden z dostawców uwzględnionych w porównaniu. Przed wdrożeniem należy sprawdzić aktualną licencję.*
