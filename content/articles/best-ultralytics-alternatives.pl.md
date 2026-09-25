---
title: "Najlepsze alternatywy dla Ultralytics YOLO w 2026: modele open source"
description: "Praktyczny przewodnik po najlepszych otwartych alternatywach dla Ultralytics YOLO w 2026 roku: ich licencjach, obsługiwanych zadaniach i ograniczeniach wdrożeniowych oraz o tym, gdzie mieści się LibreYOLO, najbardziej kompletna biblioteka YOLO na licencji MIT."
date: 2026-07-01
author: Xuban
tags: [LibreYOLO, ultralytics-alternative, object-detection, yolo, mit-license]
faq:
  - q: "Czy Ultralytics YOLO można bezpłatnie wykorzystywać komercyjnie?"
    a: "Tylko na warunkach AGPL-3.0, która wymaga udostępnienia kodu źródłowego aplikacji zbudowanej na tym rozwiązaniu, w tym usług dostępnych przez sieć. Do komercyjnego użycia z zamkniętym kodem źródłowym potrzebna jest płatna licencja komercyjna Ultralytics. Liberalne licencje alternatyw, takie jak MIT i Apache-2.0, pozwalają tego uniknąć."
  - q: "Która alternatywa dla Ultralytics ma najbardziej przyjazną licencję?"
    a: "Jeśli potrzebny jest stos na liberalnych licencjach, gotowy do wdrożenia w dowolnym miejscu, warto wybrać LibreYOLO (kod na licencji MIT), RF-DETR (Apache-2.0) lub YOLOX (Apache-2.0). Żadne z tych rozwiązań nie podlega copyleft AGPL."
  - q: "Co zastępuje YOLO w 2026 roku?"
    a: "Coraz częściej są to detektory transformerowe działające w czasie rzeczywistym: RT-DETR, RF-DETR oraz linia D-FINE i DEIM. Na wydajnym sprzęcie osiągają dokładność YOLO lub ją przewyższają przy podobnym opóźnieniu. Sieci CNN w stylu YOLO nadal wygrywają na małych urządzeniach brzegowych, na których te transformery działają słabo i nie mają dojrzałej ścieżki dla NCNN ani CPU."
  - q: "Czy te modele działają na Raspberry Pi lub NPU?"
    a: "To zależy od możliwości eksportu. Detektory CNN, takie jak YOLOX i RTMDet, można wyeksportować do NCNN i uruchamiać na Raspberry Pi, a niektóre z nich, w tym YOLOX, działają na NPU Hailo. Detektory transformerowe, takie jak RF-DETR, nie mają takiej możliwości i wymagają GPU lub wydajnego CPU."
---

Informacja: LibreYOLO jest naszym projektem i zajmuje pierwsze miejsce na tej liście. Każdego z pozostałych narzędzi używamy w praktyce, a tam, gdzie przewyższa LibreYOLO, wyjaśniamy dlaczego.

Zespoły rezygnują z Ultralytics najczęściej z jednego z trzech powodów:

- **Licencja.** Modele YOLO firmy Ultralytics są objęte licencją AGPL-3.0 (YOLOv5 od 2023 roku, a także YOLOv8 i nowsze). Scenariusz jest dobrze znany: powstaje produkt, zostaje wdrożony, a następnie kontrola prawna lub klient zwraca uwagę na model. Trzeba wtedy zdecydować, czy udostępnić kod źródłowy całej aplikacji, czy kupić licencję komercyjną. Copyleft AGPL obejmuje kod w momencie jego dystrybucji lub udostępniania jako usługi. To częsty powód poszukiwania innego rozwiązania.
- **Otwartość.** Jeden dostawca kontroluje modele, kod treningowy i warunki licencji. Niektóre zespoły chcą korzystać z wag i kodu na liberalnych licencjach, które pozwalają rozwijać rozwiązanie bez proszenia o zgodę.
- **Model.** YOLO nie zawsze jest najlepszą architekturą do danego zadania. Detektory transformerowe działające w czasie rzeczywistym, takie jak RT-DETR, RF-DETR i D-FINE, mogą osiągać większą dokładność przy takim samym opóźnieniu. Problem w tym, że są rozsiane po repozytoriach badawczych. Jak się okaże, LibreYOLO obsługuje wszystkie trzy przez jedno API, więc jest to powód, by zmienić bibliotekę, a nie powód, by odejść od niej.

Każde z poniższych narzędzi oceniamy pod trzema względami: czy jego licencja pozwala wdrożyć rozwiązanie, czy da się je zainstalować i uruchomić z aktualną wersją PyTorch oraz czy można je wyeksportować na sprzęt używany we wdrożeniu.

W trakcie lektury powtarza się jeden motyw: każda z alternatyw jest dobra, ale używanie jej samodzielnie bywa uciążliwe. YOLOX ledwo daje się zainstalować, MMDetection oznacza utknięcie w piekle wersji wheel, rodzina RT-DETR to kod badawczy bez kanału wsparcia, a Detectron2 pozostaje zamrożony od 2021 roku. LibreYOLO zajmuje pierwsze miejsce na tej liście, ponieważ już teraz obsługuje większość z tych rozwiązań, jest rozwijane, korzysta z licencji MIT, działa na aktualnym stosie i udostępnia jeden znajomy interfejs API. To raczej warstwa nad pozycjami na tej liście niż kolejna pozycja.

## W skrócie

| Narzędzie | Licencja | Obsługiwane zadania | Najlepsze zastosowanie | Eksport na urządzenia brzegowe |
| --- | --- | --- | --- | --- |
| **LibreYOLO** | MIT (kod) | Detekcja, segmentacja, estymacja pozy, klasyfikacja, głębia, estymacja spojrzenia, śledzenie | Centrum modeli ponad 20 rodzin, dostępnych przez jedno API na licencji MIT | ONNX, TensorRT, OpenVINO, NCNN, CoreML, TFLite |
| **RF-DETR** | Apache-2.0 (N/S/M/L) | Detekcja, segmentacja, estymacja pozy (wersja zapoznawcza) | Produkcyjna detekcja i segmentacja | ONNX, TFLite, TensorRT; bez NCNN i Hailo |
| **Lightly / LightlyTrain** | MIT / AGPL-3.0 | Wstępne trenowanie, destylacja | Dane bez etykiet, destylacja DINOv2/v3 | n/a (to nie jest detektor) |
| **YOLOX** | Apache-2.0 | Detekcja | Detekcja w czasie rzeczywistym bez kotwic | Trudna instalacja wersji upstream, działa przez LibreYOLO |
| **MMDetection / fork OneDL** | Apache-2.0 | Wszystko (badania) | Szeroki wybór architektur, odtwarzanie wyników publikacji | przez eksport |
| **Detectron2** | Apache-2.0 | Detekcja, segmentacja, punkty kluczowe | Mask R-CNN i rodzina R-CNN | ręcznie |
| **D-FINE / DEIM / DEIMv2** | Apache-2.0 | Detekcja | Najnowocześniejsze detektory DETR działające w czasie rzeczywistym | ONNX, TensorRT |
| **EdgeCrafter** | Apache-2.0 | Detekcja, segmentacja, estymacja pozy | Kompaktowe ViT dla urządzeń brzegowych, destylowane z DINOv3 | w fazie badawczej |
| **RT-DETR (v1-v4)** | Apache-2.0 | Detekcja | Najnowocześniejsza detekcja w czasie rzeczywistym | ONNX, TensorRT |

## 1. LibreYOLO

Licencja: MIT (kod). Najlepszy wybór dla najbardziej kompletnej biblioteki YOLO na licencji MIT i bezpośredniej alternatywy dla Ultralytics.

**LibreYOLO to biblioteka YOLO na licencji MIT, która obsługuje każdy model przez jedno API.** To nasz projekt. Po zapoznaniu się z resztą listy jego rola powinna być jasna: jest centrum, które udostępnia opisane tu alternatywy. LibreYOLO obsługuje już przez jeden znajomy i utrzymywany interfejs API niemal każde uciążliwe repozytorium wymienione wyżej: YOLOX, RTMDet, RF-DETR, D-FINE, DEIM oraz linię RT-DETR. Model jest dostępny bez przedzierania się przez zawiłości instalacji i licencji.

Za tym rozwiązaniem stoi coś więcej niż wygoda. YOLO powstało jako otwarty projekt badawczy: Darknet Josepha Redmona, od v1 do v3, był bezpłatnie dostępny dla każdego, kto chciał go używać i rozwijać. Era AGPL zamknęła tę możliwość. LibreYOLO powstało po to, by ponownie udostępnić tę pracę na warunkach zgodnych z zamiarem twórców: z liberalną licencją, rozwojem społeczności i bez wysyłania danych do twórców. Są więc dwie historie.

**Po pierwsze: to alternatywa dla Ultralytics YOLO na licencji MIT.** Zachowuje znane już API, więc zbiory danych w formacie YOLO i skrypty można przenieść, wprowadzając niewielkie zmiany. Licencja MIT zastępuje AGPL, bez copyleft obejmującego kod aplikacji i bez konieczności kupowania licencji komercyjnej. Domyślnie nie ma też telemetrii wysyłającej dane do twórców, tak jak robi to Ultralytics. Jeśli powodem poszukiwań jest licencja, to sedno sprawy. YOLO9 i RF-DETR to dokładnie przetestowane, gotowe do produkcji modele domyślne. Pozostałe modele są nowsze i wyraźnie oznaczone jako eksperymentalne. Wolimy to jasno zaznaczyć, zamiast udawać, że wszystkie są równie sprawdzone w praktyce.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")   # or LibreRFDETRl.pt, LibreDFINEl.pt, ...
results = model("image.jpg", save=True)
```

**Po drugie: obsługuje znacznie więcej niż Ultralytics.** Jedno API obejmuje ponad 20 rodzin modeli: sieci CNN typu YOLO, transformery DETR, backbone ViT, SAM i inne, a nie tylko ofertę jednego dostawcy. Dostępne są też zadania inne niż detekcja:

- **Zadania:** segmentacja instancji i semantyczna, estymacja pozy, klasyfikacja oraz obrócone ramki, a także dwa zadania niedostępne w Ultralytics: estymacja głębi z pojedynczego obrazu (Depth Anything V2) i estymacja kierunku spojrzenia (L2CS).
- **Funkcje praktyczne:** śledzenie wielu obiektów ze stałymi identyfikatorami (ByteTrack i OC-SORT), inferencja wideo z wyborem klatek i podglądem na żywo, inferencja kafelkowa dla zdjęć z dronów i satelitów, przeglądarkowy interfejs typu przeciągnij i upuść (`libreyolo ui`) oraz polecenie `doctor`, które sprawdza zbiór danych przed rozpoczęciem kosztownego trenowania.
- **Zaawansowane trenowanie:** akumulacja gradientów, zamrażanie warstw, wznawianie trenowania, augmentacja podczas testowania, dostrajanie LoRA/DoRA, wiele GPU oraz rejestrowanie danych w TensorBoard, MLflow i Weights & Biases.
- **Eksport:** siedem formatów (ONNX, TorchScript, TensorRT, OpenVINO, NCNN, CoreML, TFLite) z kwantyzacją INT8/FP16, wbudowanym NMS i metadanymi modelu.
- **Działanie na różnych platformach:** obsługa ścieżek, adresów URL, obiektów PIL, tablic NumPy, tensorów i surowych bajtów. Model działa na CUDA, Apple Silicon (MPS) lub samym CPU bez zmian w kodzie.

Jeśli repozytorium dobrego detektora zostało porzucone albo trudno je zainstalować, LibreYOLO uruchamia jego wagi w aktualnym i utrzymywanym środowisku.

## 2. RF-DETR

Licencja: Apache-2.0 (Nano, Small, Medium, Large). Najlepszy wybór do produkcyjnej detekcji i segmentacji.

RF-DETR, stworzony przez Roboflow i zaprezentowany na ICLR 2026, jest najbardziej gotowym do produkcji modelem na tej liście. Został dobrze zaprojektowany przez zespół, który dostarcza rozwiązania, więc gdy potrzebna jest detekcja lub segmentacja działająca niezawodnie w rzeczywistym systemie, stanowi mocny pierwszy wybór. Pakiet `rfdetr` i wagi od Nano do Large są objęte licencją Apache-2.0. Większe wagi XL i 2XL podlegają licencji PML firmy Roboflow.

## 3. Lightly

Najlepszy wybór dla danych bez etykiet, samonadzorowanego wstępnego trenowania i destylacji backbone DINO.

Lightly nie jest detektorem. Pozwala wykorzystać dane, które nie zostały oznaczone etykietami, i składa się z dwóch części objętych różnymi licencjami.

- **[`lightly`](https://github.com/lightly-ai/lightly) (LightlySSL), MIT.** Framework z komponentami do uczenia samonadzorowanego: funkcjami straty, głowicami, augmentacjami, bankami pamięci i implementacjami referencyjnymi ponad dwudziestu metod (SimCLR, MoCo, BYOL, DINO, DINOv2, MAE i inne). Pętlę trenowania trzeba napisać samodzielnie. Biblioteka udostępnia wszystkie elementy potrzebne do wstępnego trenowania reprezentacji od zera na nieoznaczonych obrazach.
- **[LightlyTrain](https://github.com/lightly-ai/lightly-train), AGPL-3.0 (z opcją komercyjną i bezpłatną dla społeczności).** Gotowe narzędzie, o którym najczęściej mowa. Wystarczy wskazać model bazowy, na przykład DINOv2 lub DINOv3, oraz dane bez etykiet, a narzędzie zdestyluje ten backbone do mniejszego modelu gotowego do wdrożenia (YOLO, RT-DETR, ViT lub sieci niestandardowej). Wystarczy kilka wierszy kodu i nie są potrzebne etykiety. Należy zwrócić uwagę na licencję: AGPL-3.0 nakłada takie samo copyleft, z powodu którego wiele osób szuka alternatywy dla Ultralytics. Przy planowanym wdrożeniu z zamkniętym kodem źródłowym należy zapoznać się z warunkami albo wybrać licencję komercyjną.

Lightly warto wybrać, gdy brakuje dobrego backbone, a dostępne dane nie mają etykiet. Narzędzie uzupełnia detektory z tej listy, zamiast z nimi konkurować. Widać to też w najnowszych rozwiązaniach: RT-DETRv4 uwzględnia destylację modelu bazowego do wizji komputerowej podczas trenowania, a DEIMv2 wbudowuje backbone DINOv3 bezpośrednio w detektor.

## 4. YOLOX

Licencja: Apache-2.0. Najlepszy wybór do detekcji bez kotwic w czasie rzeczywistym, gdy uda się go zainstalować.

Repozytorium upstream pozostaje zamrożone od ostatniego wydania, v0.3.0 z kwietnia 2022 roku, a liczba otwartych zgłoszeń przekracza 700. Instalacja w stosie z 2026 roku jest trudna. Brakuje pliku `pyproject.toml`, a `requirements.txt` przypina `onnx-simplifier==0.4.10`, który nie ma gotowych wheel dla wersji Pythona nowszych niż 3.10. Dlatego w nowym środowisku interpreter buduje go ze źródeł, co wymaga cmake i kompilatora C++. Nie przypięto też wersji NumPy, co może prowadzić do konfliktów ABI NumPy 2.0 ze starszym pycocotools i torch. Podstawowa detekcja działa po rozwiązaniu łańcucha zależności, ale dotarcie do tego etapu jest dodatkowym kosztem.

Sam model nadal jest dobry: to detektor bez kotwic działający w czasie rzeczywistym, stworzony przez Megvii, a zarówno jego kod, jak i wagi są objęte licencją Apache-2.0. Stanowi rozsądną linię bazową, choć od jego premiery pojawiły się dokładniejsze detektory. Przywrócenie do działania porzuconego, ale solidnego repozytorium to obecnie zadanie na jedno popołudnie dla agenta programistycznego, więc instalacja nie jest taką przeszkodą, jak mogłoby się wydawać. LibreYOLO uruchamia również wagi YOLOX bezpośrednio w aktualnym środowisku. Niezależnie od wybranej metody, warto korzystać z tej architektury. Pełniejszy przewodnik opisaliśmy [tutaj](/articles/yolox-with-libreyolo).

## 5. Ekosystem MMDetection

Licencja: przeważnie Apache-2.0. Najlepszy wybór ze względu na szeroki wybór architektur i możliwość odtwarzania wyników publikacji.

Przez lata MMDetection zawierało oficjalne implementacje niemal każdej publikacji dotyczącej detekcji: Faster R-CNN, DINO, Grounding-DINO, RTMDet, Mask R-CNN i setki konfiguracji. W 2026 roku OpenMMLab wygasza serię mm. Ostatnie wydanie [MMDetection](https://github.com/open-mmlab/mmdetection) to v3.3.0 ze stycznia 2024 roku, zgłoszenia pozostają bez odpowiedzi, a stos jest uzależniony od przypiętej wersji `mmcv`. Gotowe wheel dla tej wersji nie nadążają za bieżącymi wydaniami PyTorch i CUDA, więc nowa instalacja często przestaje działać, dopóki nie zostaną obniżone wszystkie wersje. Opisaliśmy te problemy [tutaj](/articles/rtmdet-without-mmdetection).

Holenderska firma VBTI utrzymuje ten projekt przy życiu. Jej [fork OneDL](https://github.com/VBTI-development/onedl-mmdetection) ponownie publikuje cały stos pod prefiksem `onedl-` (`onedl-mmdetection`, `onedl-mmcv`, `onedl-mmengine` i pozostałe pakiety), przebudowany dla aktualnych wersji PyTorch 2.x, CUDA i Pythona 3.10+. Rozwiązuje to problem przypiętej wersji. Fork podlega licencji Apache-2.0 i jest aktywnie utrzymywany. W maju 2026 roku wydano wersję v3.5.1. Jeśli potrzebny jest szeroki wybór architektur MMDetection bez problemów z instalacją, warto skorzystać z tego forka. Utrzymuje go niewielki zespół, więc jeśli projekt jest od niego zależny, warto wnieść własny wkład.

Dwa powiązane narzędzia:

- **[Detectron2](https://github.com/facebookresearch/detectron2)** (Meta, Apache-2.0) jest w trybie utrzymania i nie ma oznaczonego wydania od v0.6 w 2021 roku. Pozostaje jednak niezawodny i nadal jest najczystszym źródłem implementacji Mask R-CNN oraz rodziny R-CNN do segmentacji instancji i panoptycznej.
- **[TorchVision](https://github.com/pytorch/vision)** (BSD-3-Clause, aktywnie rozwijany) zawiera Faster R-CNN, RetinaNet, FCOS, SSD, Mask R-CNN i Keypoint R-CNN. Nie ma nowoczesnych modeli YOLO ani DETR, a pętlę trenowania trzeba napisać samodzielnie. To jednak rozwiązanie bazowe bez dodatkowych zależności.

Uwaga dotycząca zastosowań komercyjnych: **MMYOLO**, centrum implementacji YOLO od OpenMMLab, podlega licencji GPL-3.0, a nie Apache-2.0, i pozostaje zamrożone od sierpnia 2023 roku.

## 6. Rodzina RT-DETR

Licencja: wyłącznie Apache-2.0. Najlepszy wybór do najnowocześniejszej detekcji w czasie rzeczywistym z kodem na poziomie projektów badawczych.

Obecna czołówka detekcji w czasie rzeczywistym to grupa powiązanych detektorów transformerowych, a nie YOLO. Większość wywodzi się z RT-DETR, korzysta ze znacznej części tego samego kodu backbone i enkodera, a niemal wszystkie modele podlegają licencji Apache-2.0. Dzięki temu łatwo się między nimi przełączać. Większość obsługuje wyłącznie detekcję. Wyjątkiem jest EdgeCrafter, który rozszerza te same metody destylacji o segmentację i estymację pozy. Wspólnym kosztem jest badawczy charakter repozytoriów: trenowanie przez pliki konfiguracyjne, ergonomia inna niż w Ultralytics i brak kanału wsparcia. Modele są wydajne, a eksport do ONNX i TensorRT jest zazwyczaj dobrze obsługiwany.

- **[D-FINE](https://github.com/Peterande/D-FINE)** (USTC, wyróżnienie Spotlight na ICLR 2025). Modeluje regresję ramek jako stopniowe udoskonalanie rozkładu z samodestylacją. Osiąga wysoką dokładność przy niskim opóźnieniu (D-FINE-X uzyskuje około 55.8 AP w około 13 ms na T4), jest dostępny w rozmiarach od N do X i został zintegrowany z Hugging Face Transformers. Dzięki temu spośród modeli na tej liście najłatwiej go wczytać i dostroić poza jego własnym repozytorium.
- **[DEIM i DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)** (Intellindust AI Lab). DEIM (CVPR 2025) to przepis na trenowanie (dopasowanie Dense O2O), który można zastosować do D-FINE lub RT-DETRv2, aby skrócić czas trenowania mniej więcej o połowę. DEIMv2 dodaje cechy DINOv3 i zmniejsza model do mobilnej wersji Atto z mniej niż 1M parametrów. DEIMv2-S to pierwszy model poniżej 10M, który przekracza 50 AP. Projekt jest aktywnie utrzymywany i aktualizowany w 2026 roku.
- **[EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter)** (Intellindust AI Lab, 2026). To najnowszy projekt tego laboratorium, a zarazem jedyna pozycja na liście wykraczająca poza detekcję. Obsługuje detekcję (ECDet), segmentację instancji (ECSeg) i estymację pozy (ECPose), a każdy model jest dostępny w rozmiarze S/M/L/X i podlega licencji Apache-2.0. Wykorzystuje duży ViT wstępnie wytrenowany na DINOv3 jako nauczyciela wyspecjalizowanego do danego zadania, a następnie destyluje go do kompaktowych backbone uczniowskich dla urządzeń brzegowych. Wyniki są dobre jak na taki rozmiar: ECDet-S osiąga 51.7 AP przy mniej niż 10M parametrów wyłącznie na COCO, ECPose-X uzyskuje 74.8 AP (więcej niż 71.6 w YOLO26Pose-X), a segmentacja instancji jest porównywalna z RF-DETR. Projekt jest zupełnie nowy, więc należy traktować go jako wydanie badawcze. To jednak rzadki kompaktowy model obsługujący wszystkie trzy zadania.
- **[RT-DETR i RT-DETRv2](https://github.com/lyuwenyu/RT-DETR)** (Baidu). Oryginalna praca pokazująca, że „DETR-y pokonują YOLO w detekcji w czasie rzeczywistym” (CVPR 2024) i początek całej rodziny: bez NMS, end-to-end i gotowy do eksportu. Wersja v2 to aktualizacja z 2024 roku zawierająca szereg usprawnień, będąca bezpośrednim ulepszeniem przy tym samym opóźnieniu. W tym samym repozytorium znajdują się oryginalna wersja Paddle i kanoniczny port PyTorch. Model jest też dostępny w HF Transformers.
- **[RT-DETRv3](https://github.com/clxia12/RT-DETRv3)** (Baidu, wystąpienie ustne na WACV 2025) oraz **[RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4)** (Peking University i Tsinghua, końcówka 2025 roku). v3 działa wyłącznie z PaddlePaddle. v4 to obecnie najlepszy model w tej rodzinie (wersja X osiąga niemal 57 AP), działa w PyTorch i destyluje model bazowy do wizji komputerowej w lekkim detektorze bez dodatkowego kosztu inferencji. Zmiany konfiguracji pozwalają też odtwarzać wyniki D-FINE, DEIM i RT-DETRv2 w tej bazie kodu.
- **[LW-DETR](https://github.com/Atten4Vis/LW-DETR)** (Baidu). DETR z czystym ViT, przedstawiany jako transformerowa alternatywa dla YOLO, w rozmiarach od tiny do xlarge, z eksportem do ONNX i TensorRT. Stanowi też podstawę modelu OVLW-DETR do detekcji z otwartym słownikiem. Aktywność ostatnio spadła, dlatego należy traktować go jako stabilne wydanie badawcze.

LibreYOLO obsługuje już wiele z tych modeli przez jedno API: D-FINE, DEIM, DEIMv2, RT-DETRv2, RT-DETRv4, RTMDet i EdgeCrafter. Najdokładniej przetestowano YOLO9 i RF-DETR. W przypadku implementacji referencyjnych i najnowszych checkpointów badawczych źródłem są powyższe repozytoria.

## FAQ

**Czy Ultralytics YOLO można bezpłatnie wykorzystywać komercyjnie?**
Tylko na warunkach AGPL-3.0, która wymaga udostępnienia kodu źródłowego aplikacji zbudowanej na tym rozwiązaniu, w tym usług dostępnych przez sieć. Do komercyjnego użycia z zamkniętym kodem źródłowym potrzebna jest płatna licencja komercyjna Ultralytics. Liberalne licencje alternatyw, takie jak MIT i Apache-2.0, pozwalają tego uniknąć.

**Która alternatywa dla Ultralytics ma najbardziej przyjazną licencję?**
Jeśli potrzebny jest stos na liberalnych licencjach, gotowy do wdrożenia w dowolnym miejscu, warto wybrać LibreYOLO (kod na licencji MIT), RF-DETR (Apache-2.0) lub YOLOX (Apache-2.0). Żadne z tych rozwiązań nie podlega copyleft AGPL.

**Co zastępuje YOLO w 2026 roku?**
Coraz częściej są to detektory transformerowe działające w czasie rzeczywistym: RT-DETR, RF-DETR oraz linia D-FINE i DEIM. Na wydajnym sprzęcie osiągają dokładność YOLO lub ją przewyższają przy podobnym opóźnieniu. Sieci CNN w stylu YOLO nadal wygrywają na małych urządzeniach brzegowych, na których te transformery działają słabo i nie mają dojrzałej ścieżki dla NCNN ani CPU.

**Czy te modele działają na Raspberry Pi lub NPU?**
To zależy od możliwości eksportu. Detektory CNN, takie jak YOLOX i RTMDet, można wyeksportować do NCNN i uruchamiać na Raspberry Pi, a niektóre z nich, w tym YOLOX, działają na NPU Hailo. Detektory transformerowe, takie jak RF-DETR, nie mają takiej możliwości i wymagają GPU lub wydajnego CPU.

## Wypróbuj

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLO ma licencję MIT, działa w systemach Linux, Mac i Windows oraz działa na GPU, Apple Silicon i zwykłym procesorze CPU bez zmian w kodzie. Jedno API obejmuje RF-DETR, D-FINE, DEIM, YOLOX, YOLO-NAS, RTMDet, RT-DETR, EdgeCrafter i inne modele do detekcji, segmentacji, estymacji pozy, klasyfikacji, estymacji głębi i spojrzenia oraz śledzenia.

Dodaj gwiazdkę w GitHubie: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Dokumentacja: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
