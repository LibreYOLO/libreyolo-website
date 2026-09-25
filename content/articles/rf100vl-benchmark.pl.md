---
title: "Benchmark RF100-VL: jak modele LibreYOLO uogólniają na nowych zbiorach danych?"
description: Wyniki RF100-VL pokazujące, jak YOLOv9, YOLOX, YOLO-NAS, EdgeCrafter i RF-DETR uogólniają na 100 rzeczywistych zbiorach danych po dostrojeniu.
date: 2026-09-05
author: Xuban
layout: paper
tags: [LibreYOLO, RF100-VL, benchmark, object-detection, roboflow]
---

RF100-VL sprawdza, jak dobrze detektor dostosowuje się do danych spoza COCO. Każdy model jest osobno dostrajany na 100 bardzo różnych zbiorach danych, obejmujących obrazy w podczerwieni, zdjęcia rentgenowskie, mikroskopię, sport, obrazy widma radiowego, małe obiekty i gry wideo. Zbenchmarkowaliśmy 17 konfiguracji LibreYOLO: **łącznie 1700 procesów dostrajania.**

<div>
<rf100vl-explorer></rf100vl-explorer>
</div>

## Wyniki

<div>
<rf100vl-results-chart></rf100vl-results-chart>
</div>

RF-DETR-L osiągnął w tym porównaniu wynik **61.76**. Za nim znalazł się model M z wynikiem 61.13, a następnie S z wynikiem 60.41. Cztery wyniki RF-DETR są zbliżone do [opublikowanych wyników Roboflow](https://rfdetr.roboflow.com/latest/learn/benchmarks/), co stanowi przydatne potwierdzenie poprawności trenowania RF-DETR w LibreYOLO.

Rozmiar nie pozwalał przewidzieć każdego wyniku. YOLO-NAS-S i M uzyskały praktycznie remisujące wyniki: 58.00 i 57.99. EdgeCrafter-M osiągnął 57.93, wyprzedzając S z wynikiem 55.99 i L z wynikiem 56.11. Planujemy zbadać te regresje. Ten przegląd nie jest kontrolowanym eksperymentem skalowania: rozdzielczość, wstępnie wytrenowane wagi, precyzja i receptury są różne.

RF-DETR-S z LoRA w backbone osiąga 57.19, w porównaniu z 60.41 przy pełnym dostrajaniu. Pełne dostrajanie daje lepszy wynik na 95 zbiorach danych. Mediana zarejestrowanych czasów jest tylko o siedem minut dłuższa, a łączny czas zadań pozostaje niemal bez zmian.

Wiersze YOLOv9 pochodzą sprzed istotnych poprawek trenowania. Anomalny wynik M pomógł wykryć brak PGI, inną konwencję letterbox i limit 100 etykiet podczas trenowania. Zarchiwizowane liczby opisują uruchomiony kod. Nie są werdyktem na temat architektury YOLOv9.

Wybierz model poniżej, aby sprawdzić jego wyniki na wszystkich 100 zbiorach danych.

<div>
<rf100vl-results-detail></rf100vl-results-detail>
</div>

## Metodologia

Dla każdego zbioru danych trenowanie odbywało się na `train`, następnie wybierano checkpoint o najlepszym wyniku AP50:95 na zbiorze walidacyjnym i jednokrotnie oceniano go na `test`. Wynik główny to nieważona średnia ze 100 wyników testowych.

| Ustawienie | Zasada kampanii |
|---|---|
| Trenowanie | 100 epok, bez early stopping |
| Efektywny batch | 16 |
| Seed | 0, jeden ukończony przebieg na zbiór danych |
| Checkpoint | Najlepszy wynik AP50:95 na zbiorze walidacyjnym, z użyciem wag EMA |
| Ocena na zbiorze testowym | pycocotools z `maxDets=500` |
| Dostrajanie | Jedna przypięta receptura na rodzinę, bez dostrajania dla poszczególnych zbiorów danych |

Każda rodzina miała własną recepturę trenowania:

| Rodzina | Rozdzielczość | Precyzja | Opis augmentacji |
|---|---:|---:|---|
| YOLOv9 T/S/M | 640 | FP32 | Mosaic, HSV, flip; silne augmentacje wyłączone na ostatnie 15 epok |
| YOLOX Nano/Tiny | 416 | FP32 | Mosaic, mixup, HSV, affine, flip; końcowe 15 epok |
| YOLOX S/M | 640 | FP32 | Ta sama receptura dla rodziny, ze współczynnikiem uczenia dobranym do rozmiaru |
| YOLO-NAS S/M | 640 | FP32 | Mixup, HSV i flip; mosaic wyłączone |
| EdgeCrafter S/M/L | 640 | FP16 | Flip; domyślne ustawienia rodziny LibreYOLO |
| RF-DETR N/S/M/L i S LoRA | 384/512/576/704; LoRA przy 512 | BF16 | Multi-scale, crop/resize i flip |

To wyniki z jednym seedem. Nie potwierdzono, że małe różnice oznaczają poprawę. W przypadku RF-DETR M i L zastosowano akumulację gradientów, aby zmieścić się w dostępnej pamięci, a dla niektórych gęstych zbiorów danych potrzebne były mniejsze batche fizyczne. Publiczne receptury RF-DETR również rozpoczynają od checkpointów COCO, natomiast w historycznej tabeli Roboflow wykorzystano prywatne checkpointy Objects365.

Czasy trenowania pochodzą z zapisów kampanii, a nie z kontrolowanych testów szybkości. Zadania czasami korzystały wspólnie z GPU, były ponawiane lub wznawiane. Nie przeprowadziliśmy opcjonalnego protokołu pomiaru opóźnienia pojedynczego artefaktu na T4. YOLO-NAS korzysta z objętych odrębną licencją [wag wstępnie wytrenowanych do użytku niekomercyjnego](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md) firmy Deci.

## Co poprawiła ta praca

Małe testy mogą potwierdzić, że trenowanie się rozpoczyna. Nie odtwarzają jednak wielotygodniowego, ciągłego dostrajania wielu architektur na wielu zbiorach danych. Przy tej skali trudno było przeoczyć wolne ścieżki, presję na pamięć i problemy z poprawnością.

- **Wczytywanie obrazów i walidacja.** Zapisaliśmy deterministycznie przeskalowany obraz w pamięci podręcznej przed augmentacją, ponownie wykorzystaliśmy wątki robocze walidatora i bufory oraz tam, gdzie było to obsługiwane, tworzyliśmy grafy przebiegów forward podczas walidacji. [PR #677](https://github.com/LibreYOLO/libreyolo/pull/677), [PR #682](https://github.com/LibreYOLO/libreyolo/pull/682)
- **Grafy CUDA podczas trenowania.** Obsługa przechwytywania objęła 24 rodziny, zaczynając od YOLOv9 i RF-DETR. Naprawiliśmy też wyścig związany z pin-memory w DataLoaderze oraz unieważnianie grafów przy przejściach między etapami trenowania. Test YOLOv9-T skrócił się z 428.4 do 367.7 sekund przy tym samym raportowanym AP. [PR #671](https://github.com/LibreYOLO/libreyolo/pull/671), [PR #681](https://github.com/LibreYOLO/libreyolo/pull/681), [PR #716](https://github.com/LibreYOLO/libreyolo/pull/716)
- **Ocena COCO.** Ocena na gęstych zbiorach danych czasami trwała dłużej niż trenowanie. Integracja faster-coco-eval oceniła zapisane predykcje ze wszystkich 100 podziałów testowych w 8.4 sekundy zamiast 131.4. Spośród 1,400 wartości metryk 1,381 było identycznych bit po bicie; największa różnica wyniosła 2.22e-16, a główny wynik AP się nie zmienił. [PR #708](https://github.com/LibreYOLO/libreyolo/pull/708)
- **RF-DETR i wspólne ścieżki trenowania.** Szybsze dopasowywanie L1, mniejsza liczba synchronizacji urządzenia, połączony AdamW i ograniczona pamięć matchera skróciły krok RF-DETR-S z 266 do 234 ms w zapisanym teście. To samo badanie usprawniło pięć innych matcherów, logowanie YOLOv9 i ponowne użycie tensorów w EdgeCrafter. [PR #761](https://github.com/LibreYOLO/libreyolo/pull/761), [PR #762](https://github.com/LibreYOLO/libreyolo/pull/762), [PR #765](https://github.com/LibreYOLO/libreyolo/pull/765)
- **Attention.** Skierowaliśmy obsługiwane operacje attention do PyTorch SDPA, zintegrowaliśmy implementację CUDA na licencji Apache-2.0 i naprawiliśmy wykonywanie w mieszanej precyzji. Napisaliśmy też kernel Triton dla deformable attention na potrzeby inferencji, bezpośrednio w repozytorium. W udokumentowanych testach był około cztery razy szybszy w izolacji i około 7% szybszy całościowo dla RF-DETR-N. [PR #712](https://github.com/LibreYOLO/libreyolo/pull/712), [PR #713](https://github.com/LibreYOLO/libreyolo/pull/713), [PR #760](https://github.com/LibreYOLO/libreyolo/pull/760), [PR #784](https://github.com/LibreYOLO/libreyolo/pull/784), [PR #790](https://github.com/LibreYOLO/libreyolo/pull/790)
- **Poprawność trenowania.** Naprawiliśmy rekonstrukcję BatchNorm w YOLOX oraz przywróciliśmy PGI, metadane letterbox, obsługę etykiet i rozgrzewkę momentum w YOLOv9. Benchmark dostarczył checkpointów i wzorców błędów, które ujawniły oba problemy. [PR #700](https://github.com/LibreYOLO/libreyolo/pull/700), [PR #796](https://github.com/LibreYOLO/libreyolo/pull/796)

Liczby te pochodzą z oddzielnych testów na różnych obciążeniach. Nie należy ich mnożyć, by uzyskać jeden zbiorczy wzrost szybkości. LibreYOLO jest teraz szybsze i bardziej niezawodne przy rzeczywistych obciążeniach związanych z trenowaniem niż przed tą kampanią.

## Harness i artefakty

Publiczny [harness trenowania](https://github.com/LibreYOLO/vision-analysis-benchmark/tree/rf100vl-harness) rozdziela zbiory danych między GPU, wznawia przerwane zadania, obsługuje odzyskiwanie po OOM i zapisuje receptury, wersje zbiorów danych, logi, checkpointy oraz predykcje.

Umiejętność [run-rf100vl-benchmark](https://github.com/LibreYOLO/libreyolo/blob/release/skills/run-rf100vl-benchmark/SKILL.md) opisuje protokół i instrukcje obsługi Vast.ai dla agentów AI kodujących. [Archiwum wyników](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main) zawiera każdy opublikowany przebieg.

## Podziękowania dla Roboflow

Joseph Nelson zaoferował wsparcie w postaci GPU po tym, jak napisałem, że chcę przeprowadzić benchmark wykraczający poza COCO, ale nie stać mnie na te przebiegi. Matvei Popov udostępnił konfiguracje referencyjne, objaśnił ustawienia ewaluacji i śledził pojawiające się wyniki.

To wsparcie pozwoliło zweryfikować trenowanie LibreYOLO w 17 konfiguracjach, opublikować dowody przydatne przy wyborze modelu, udostępnić harness i obciążyć bibliotekę tak, by jej słabe punkty stały się oczywiste.

Dziękuję Josephowi, Matveiowi i zespołowi Roboflow za moc obliczeniową, czas i wskazówki.
