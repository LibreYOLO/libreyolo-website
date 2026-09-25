---
title: "69 firm Edge AI NPU: układy, SDK i YOLO (2026)"
description: "Przewodnik oparty na źródłach po 69 firmach Edge AI i ekosystemach NPU: układach, SDK, artefaktach, kwantyzacji oraz udokumentowanej obsłudze wizji komputerowej i YOLO."
date: 2026-08-15
author: Xuban
tags: [edge-ai, npu, computer-vision, yolo, hardware, ai-accelerators, amlogic, hailo, rockchip]
faq:
  - q: "Ile firm oferuje edge AI NPU?"
    a: "Nie istnieje jedna uniwersalna liczba firm, ponieważ dostawcy produktów, licencjodawcy IP, inteligentne sensory, GPU i prywatne samochodowe układy ASIC nakładają się na siebie. Ten niewyczerpujący przewodnik obejmuje 69 firm i ekosystemów platform z odpowiednimi układami Edge AI, platformami akceleratorów, licencjonowanym IP NPU lub wiarygodnymi układami z listy obserwacyjnej według stanu na sierpień 2026 r."
  - q: "Czym jest NPU?"
    a: "Neural processing unit to wyspecjalizowany układ do operacji sieci neuronowych, takich jak sploty i mnożenie macierzy. Dostawcy używają nazw NPU, AIPU, BPU, KPU, DLA, HTP, TPU i MLA, a skompilowane modele na ogół nie są przenośne między firmami."
  - q: "Które firmy oferujące NPU oficjalnie obsługują modele YOLO?"
    a: "Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Sony IMX500 za pośrednictwem oficjalnego repozytorium AI Camera firmy Raspberry Pi, STMicroelectronics, Renesas, Lattice, Himax, Microchip i kilka innych firm mają własne wpisy w katalogach modeli, samouczki lub zweryfikowane wyniki dla co najmniej jednej generacji YOLO. Nadal istotne są konkretna generacja, zadanie, układ docelowy i wersja SDK."
  - q: "Czy dowolny model ONNX można uruchomić na dowolnym NPU?"
    a: "Nie. ONNX jest formatem wymiany, a nie gwarancją zgodności ze sprzętem. Kompilator NPU musi obsługiwać każdy operator, kształt tensora i typ danych w grafie. Często stosuje się statyczne kształty, podział grafu, operacje niestandardowe i awaryjne wykonanie na CPU."
  - q: "Które NPU jest najlepsze do YOLO?"
    a: "Nie ma uniwersalnego zwycięzcy. Hailo, Rockchip, Axelera AI, DEEPX i Amlogic mają szeroką, udokumentowaną obsługę YOLO, a urządzenia Qualcomm są wyjątkowo rozpowszechnione. Wybór powinien uwzględniać dokładność po kwantyzacji, opóźnienie całego pipeline'u, moc przy długotrwałym obciążeniu, cenę, dostępność, dostęp do SDK i obsługę systemu operacyjnego."
  - q: "Dlaczego wartości TOPS dla NPU nie są bezpośrednio porównywalne?"
    a: "Dostawcy mogą inaczej liczyć precyzję, operacje rzadkie, operacje multiply-accumulate i założenia dotyczące szczytowego wykorzystania. TOPS pomija ruch danych w pamięci, nieobsługiwane operatory, przetwarzanie wstępne i końcowe oraz narzut hosta. Należy porównywać ten sam model, precyzję, rozdzielczość, dokładność i cały pipeline."
  - q: "Na czym polega kalibracja NPU?"
    a: "Podczas kalibracji reprezentatywne, zwykle nieoznaczone obrazy wdrożeniowe są przepuszczane przez model, aby kompilator mógł oszacować zakresy aktywacji dla INT8 lub innego formatu o niskiej precyzji. Losowe lub niereprezentatywne obrazy mogą nadal pozwolić utworzyć skompilowany artefakt, ale pogorszyć dokładność zadania."
  - q: "Czy LibreYOLO obsługuje edge NPU?"
    a: "LibreYOLO eksportuje ONNX i kilka formatów środowisk uruchomieniowych, ma bezpośrednią ścieżkę kompilacji RKNN dla wybranych modeli Rockchip i opisuje zewnętrzny proces kompilacji Hailo. Każdy inny natywny artefakt dostawcy nadal wymaga jego SDK i należy go traktować jako kandydata do integracji, dopóki nie zostanie skompilowany, zweryfikowany i uruchomiony na sprzęcie."
---

**Nie istnieje jeden rynek NPU. Ten niewyczerpujący przewodnik obejmuje 69 firm i ekosystemów platform: 51 dostawców układów lub platform gotowych do wdrożenia, dziewięciu dostawców IP NPU i dziewięć wyraźnie oznaczonych firm z listy obserwacyjnej.** Obejmują one kilka niezgodnych ze sobą klas akceleratorów i niemal tyle samo stosów kompilatorów. Większość obiecuje podobny proces: wyeksportować model, skwantyzować go, skompilować, skopiować zastrzeżony artefakt na płytkę i wywołać środowisko uruchomieniowe dostawcy. Szczegóły decydują, czy zajmie to popołudnie, czy kwartał pracy inżynierskiej.

Ten przewodnik mapuje firmy oferujące wiarygodny sprzęt do wizji komputerowej w 2026 r. Zawiera odpowiednie układy, nazwy SDK i kompilatorów, artefakty wdrożeniowe, poziom publicznego dostępu oraz modele faktycznie opisane przez dostawcę. Szczególną uwagę poświęcono też Amlogic, którego nowszy stos ADLA istotnie różni się od starszego ekosystemu NPU A311D.

> **Zakres badań:** Źródła sprawdzono 15 sierpnia 2026 r. Twierdzenia o nazwanych modelach pochodzą ze stron produktów, dokumentacji, katalogów modeli, repozytoriów lub samouczków dostawców, chyba że wyraźnie zaznaczono inaczej. Reklamowane wartości TOPS pochodzą od dostawców i nie są niezależnie porównywalnymi wynikami benchmarków. To przewodnik dla deweloperów, a nie porada inwestycyjna ani płatny ranking.

**Szybka nawigacja:** [tabele firm](#npu-companies-and-platforms-at-a-glance) | [szczegółowo o Amlogic](#amlogic-two-npu-generations-not-one) | [profile dostawców](#the-strongest-public-deployment-ecosystems) | [tabela skompilowanych artefaktów](#what-you-actually-deploy-the-artifact-lock-in-table) | [dostęp i cykl życia](#tool-access-and-lifecycle-signals) | [plan rozwoju LibreYOLO](#what-this-means-for-libreyolo)

## Najważniejszy wniosek

Sprzęt jest rozdrobniony, ale schemat wdrażania pozostaje zaskakująco spójny:

```text
PyTorch checkpoint
    -> ONNX or TFLite interchange graph
    -> representative calibration data
    -> vendor quantizer and graph compiler
    -> chip-specific binary or model package
    -> vendor runtime on the target
    -> application preprocessing, decode, NMS and rendering
```

[ONNX jawnie dopuszcza środowiska uruchomieniowe, generatory kodu i implementacje sprzętowe](https://onnx.ai/onnx/repo-docs/IR.html), ale plik ONNX jest tylko formatem przekazania modelu. Nie oznacza to, że każdy operator ONNX zostanie obniżony do postaci obsługiwanej przez każdy akcelerator. O tym, co rzeczywiście działa na NPU, decydują statyczne kształty wejścia, ograniczenia obsługiwanych operatorów, reguły kwantyzacji i awaryjne wykonanie na hoście.

Stos programowy jest więc częścią układu. Nominalnie szybkie NPU z kompilatorem wymagającym specjalnego dostępu lub niestabilnym może być gorszym celem wdrożenia niż mniejsze urządzenie z publicznym łańcuchem narzędzi, katalogiem modeli, symulatorem i stabilnym środowiskiem uruchomieniowym.

## Co oznacza „obsługa” w tym przewodniku

Materiały dostawców bardzo swobodnie posługują się słowem *obsługa*. W tym artykule wyróżniono cztery poziomy dowodów:

| Poziom dowodu | Co potwierdza | Czego nie potwierdza |
|---|---|---|
| **Zweryfikowany model** | Dostawca publikuje wpis modelu dla konkretnego układu, wynik lub tabelę zgodności | Zmodyfikowany checkpoint zachowa tę samą dokładność |
| **Oficjalny przykład** | Dostawca udostępnia samouczek lub demonstrację od początku do końca dla nazwanego modelu | Skompilują się inne rozmiary, zadania lub generacje |
| **Możliwości kompilatora** | SDK importuje framework lub udostępnia listę obsługiwanych operatorów | Konkretny graf YOLO działa od początku do końca |
| **Marketing lub dostęp ograniczony** | Produkt jest przeznaczony do wizji komputerowej i istnieje prywatne SDK | Zgodność modeli da się publicznie odtworzyć |

To rozróżnienie ma znaczenie. „Importuje ONNX” nie znaczy tyle samo co „obsługuje segmentację YOLO11”. Model może zostać wczytany, ale wykonywać się awaryjnie na CPU, kompilować z nieprawidłowymi wartościami wyjściowymi, przekroczyć pamięć akceleratora albo stracić dokładność podczas kwantyzacji.

## Firmy i platformy NPU w skrócie

W tabelach celowo zestawiono SoC, dyskretne akceleratory, inteligentne sensory oraz powiązane cele GPU/FPGA. Konkurują o tę samą decyzję wdrożeniową, nawet jeśli dostawcy używają nazw NPU, AIPU, BPU, KPU, DLA, HTP, TPU lub MLA.

### Wbudowane SoC, inteligentne sensory i procesory przemysłowe

| Firma | Odpowiednie układy lub platformy | SDK, kompilator i artefakt | Publicznie udokumentowane dowody dla wizji komputerowej | Dostęp |
|---|---|---|---|---|
| [Amlogic](https://github.com/Amlogic-NN/amlnn-toolkit) | A311D2, S928X, S905X5/S905D5, A311Y3, C308L/C302X/C302X2, T968D4, C305X2 i A123X | AMLNN Toolkit i `libnnsdk.so`, skompilowany `.adla`; osobny starszy stos Acuity `.nb` dla A311D | [Model playground](https://github.com/Amlogic-NN/amlnn-model-playground) dokumentuje YOLOv5/6/7/8/10/11, YOLOX, YOLOE, YOLO-World, PP-YOLOE, segmentację, estymację pozy i OBB na A311D2, S905X5, A311Y3, C305X2 i A123X; pozostałe układy z listy są celami kompilatora, a nie wpisami w tej macierzy obsługi | Publiczne GitHub i pakiety wheel; wymagany zgodny BSP/sterownik |
| [Rockchip](https://github.com/airockchip/rknn-toolkit2) | RK3562/3566/3568, RK3576, RK3588, RV1126B | RKNN-Toolkit2, RKNN Runtime, `.rknn` | YOLOv5/6/7/8/10/11, YOLOX, YOLO-World, PP-YOLOE, segmentacja, estymacja pozy i OBB w [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) | Publiczne GitHub; binarne pakiety wheel |
| [Qualcomm](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | Platformy Snapdragon i Dragonwing, QCS6490, QCS8550 i Dragonwing IQ-9075 | QAIRT/QNN, SNPE i AI Hub; pliki binarne kontekstu QNN lub DLC | Kolekcja modeli [AI Hub](https://github.com/qualcomm/ai-hub-models) obejmuje YOLOv3/5/6/7/8/9/10/11/26, YOLOX, YOLO-World, YOLOR, RF-DETR oraz modele detekcji, segmentacji i estymacji pozy | Dokumentacja publiczna; wymagania SDK/konta są różne |
| [Texas Instruments](https://github.com/TexasInstruments/edgeai-tidl-tools) | AM62A, AM67A, AM68A, AM69A i TDA4x; zapowiedziany TDA54-Q1 | Processor SDK Edge AI i TIDL | [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo) publikuje zoptymalizowane modele detekcji, segmentacji, estymacji pozy i klasyfikacji dla bieżących ścieżek AM6xA/TDA4; nie jest to jeszcze macierz celów dla zapowiadanego TDA54-Q1 | Publiczne GitHub i Processor SDK |
| [NXP](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | i.MX 8M Plus, i.MX 93, i.MX 95 oraz przejęte Kinara Ara-1/Ara240 | eIQ z TIM-VX, Vela, Neutron Converter i Ara SDK | [eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) zawiera zasoby lub przepisy dla YOLOv4-tiny/v8, NanoDet, CenterNet, FastestDet, SSD Lite i YOLACT; wpis YOLOv5 zawiera wyłącznie dokumentację | Częściowo publiczne, częściowo dostępne po założeniu konta |
| [STMicroelectronics](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) | Warianty STM32N6x7, w tym STM32N657/647 z akceleratorem Neural-ART | STM32Cube AI Studio i ST Edge AI Core | Bieżące repozytorium usług wymienia Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26 i ST-YOLOX oraz modele estymacji pozy i segmentacji | Publiczne narzędzia i repozytoria |
| [Infineon](https://documentation.infineon.com/psocedge/) | PSOC Edge E83/E84 z Cortex-M55 i Ethos-U55; osobny akcelerator NNLite po stronie M33 | ModusToolbox, [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter), TFLite Micro i Arm Vela | Materiały architektoniczne wymieniają MobileNetV1/V2 jako reprezentatywne operacje, a zestaw AI E84 zawiera kamerę, ale nie znaleziono macierzy walidacji YOLO dla PSOC Edge pochodzącej bezpośrednio od dostawcy | Publiczna dokumentacja, składniki SDK i aktualne zestawy ewaluacyjne E84 |
| [Renesas](https://github.com/renesas-rz/rzv_drp-ai_tvm) | RZ/V2L, V2M, V2MA, V2H i V2N | DRP-AI Translator, DRP-AI TVM i RUHMI | [Lista modeli RZ/V2H](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) weryfikuje warianty YOLOv5/v8/v11/26, YOLOX, estymację pozy i segmentację | Publiczne GitHub i SDK płytki |
| [Sony](https://www.aitrios.sony-semicon.com/edge-ai-devices/imx500) | Inteligentny sensor wizyjny IMX500 i Raspberry Pi AI Camera | Edge-MDT, Model Compression Toolkit i IMX500 packer; pakiet `.rpk` | Katalog modeli [Raspberry Pi IMX500](https://github.com/raspberrypi/imx500-models) obejmuje YOLOv8n, YOLO11n, EfficientDet Lite, NanoDet+ i SSD | Publiczny proces Raspberry Pi; obowiązują warunki narzędzi Sony |
| [Ambarella](https://www.ambarella.com/developer/model-garden/) | Rodziny CV72/CV75, CV5/CV52 i CV3-AD | Cooper Developer Platform, kompilator CVflow i grafy środowiska uruchomieniowego | Publiczny katalog modeli wymienia YOLOX-S, RTMDet-nano, DeepLabV3+, TopFormer, OWL-ViT i LLaVA OneVision | Dostęp głównie przez partnerów |
| [Synaptics](https://developer.synaptics.com/docs/sl/overview) | Astra SL1600/SL1680; SL2611/13/15/17/19; osobna rodzina mikrokontrolerów SR100 | SyNAP `.synap` dla SL16xx; Torq/IREE `.vmfb` dla SL261x; osobny SDK SR oparty na Ethos-U55 | Oficjalny benchmark SyNAP YOLOv8n/v8s i przykład Torq YOLOv8 dla SL261x; dowody dla SR trzeba oceniać osobno | Portal deweloperski; część plików do pobrania wymaga dostępu |
| [MediaTek](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) | Genio 360/360P/420/520/720 z NP8 i MDLA 5.3; Genio 510/700/1200 ze starszym NP6/MDLA | NeuroPilot Converter, `ncc-tflite`, Neuron Runtime i `.dla`; ścieżka ONNX Runtime w wybranych nowszych systemach | Oficjalny IoT AI Hub publikuje strony modeli i benchmarki YOLOv5 i YOLOv8 oraz modele klasyfikacji i twarzy | Publiczna dokumentacja Yocto; ważne pakiety NP8 i materiały Android wymagają bezpośredniego dostępu klienta/NDA |
| [Allwinner](https://docs.aw-ol.com/v853/en/npu/dev_npu/) | Układ SoC do wizji V853 z NPU Vivante o wydajności 1 TOPS | Konwersja Acuity/Pegasus i środowisko uruchomieniowe `viplite` | Oficjalne materiały V853 wymieniają YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet oraz sieci wykrywania twarzy i osób | Publiczna dokumentacja; pobranie SDK może wymagać konta |
| [Canaan/Kendryte](https://github.com/kendryte/nncase) | Aktualne K230/K230D; starsze KPU K210/K510 | Kompilator i środowisko uruchomieniowe `nncase`; `.kmodel` | Przewodnik nncase dla K230 opisuje kompilację, symulację i uruchomienie YOLOv5s; oficjalny katalog demonstracji i aktualny [dziennik zmian CanMV](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md) dokumentują zadania YOLOv8/11/26 | Publiczny kompilator/PyPI; wtyczka układu jest binarna |
| [Alif Semiconductor](https://alifsemi.com/support/kits/ensemble-e7appkit/) | Ensemble E7 z dwoma akceleratorami ML, w tym Ethos-U55; [E8](https://alifsemi.com/ensemble-e8-series/) z jednym Ethos-U85 i dwoma NPU Ethos-U55 | Wdrażanie oparte na Arm Vela i TFLite Micro | Alif publikuje benchmark INT8 wykrywacza twarzy YOLO-Fastest na poziomie rodziny, ale nie szeroką macierz współczesnego YOLO dla poszczególnych celów | Publiczna dokumentacja i zasoby zestawu ewaluacyjnego |
| [Himax](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) | Mikrokontroler endpoint AI HX6538 WiseEye2 z Cortex-M55 i Ethos-U55 | TFLite Micro z Arm Vela oraz awaryjnym użyciem CMSIS-NN i jąder referencyjnych | Oficjalne [przykłady WiseEye2](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) obejmują YOLOv8n do detekcji, estymacji pozy i klasyfikacji, detekcję YOLO11n, siatkę twarzy i PeopleNet | Publiczne GitHub, dokumentacja i dostępna w sprzedaży płytka partnerska |
| [Analog Devices](https://www.analog.com/en/products/max78002.html) | Mikrokontrolery AI MAX78000 i MAX78002 z energooszczędnymi akceleratorami CNN | [`ai8x-training`](https://github.com/analogdevicesinc/ai8x-training), `ai8x-synthesis`/`izer` i MSDK; wygenerowany kod C i wagi | Oficjalne materiały obejmują identyfikację/wykrywanie twarzy, RetinaNet, Visual Wake Words, klasyfikatory i rozpoznawanie aktywności; nie znaleziono wdrożenia YOLO od dostawcy | Publiczne narzędzia GitHub, dokumentacja i płytki ewaluacyjne |
| [D-Robotics](https://github.com/D-Robotics/rdk_model_zoo) | Płytki BPU RDK X3/X5/Ultra i nowsza seria S100 | OpenExplorer/Algorithm Toolchain i `hbm_runtime`; X5 `.bin`, aktualne `rdk_s` `.hbm` | Bieżąca gałąź `rdk_x5` dokumentuje YOLOv5/v5u/v8/v9/v10/11/12/13/26, YOLOE, YOLO-World, segmentację, estymację pozy, klasyfikację, OCR i CLIP dla RDK X5; X3 i seria S mają osobne gałęzie | Publiczne GitHub; niektóre pakiety toolchain są zależne od platformy |
| [AXERA](https://github.com/AXERA-TECH/ax-samples) | AX650, AX637, AX630C, AX620Q i AX615 | Kompilator Pulsar2 i AXEngine; `.axmodel` | Aktualne oficjalne przykłady obejmują YOLOv5/6/7/8/9/10/11/13/26, YOLOX i YOLO-World; zadania i cele zależą od układu | Publiczne przykłady i dokumentacja; kompilator jest dystrybuowany osobno |
| [SOPHGO](https://github.com/sophgo/tpu-mlir) | BM1684/1684X/1688/1690 oraz CV186X/CV18xx | TPU-MLIR i środowisko uruchomieniowe SOPHON; `.bmodel` lub `.cvimodel` | Oficjalne demonstracje obejmują YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, OBB, segmentację, twarze, SAM i OCR | Kompilator open source i środowisko uruchomieniowe dostawcy |
| [Huawei Ascend](https://www.hiascend.com/en/software/cann) | Układy Ascend 310/310P/310B oraz urządzenia brzegowe Atlas 200I/300I | CANN, kompilator ATC i AscendCL; `.om` | Oficjalny [Ascend ModelZoo](https://github.com/Ascend/modelzoo) obejmuje YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN i modele segmentacji | Publiczna dokumentacja; pakiety CANN i jądra muszą pasować do celu |
| [Cambricon](https://github.com/Cambricon/magicmind_cloud) | MLU370-X4/S4 w cytowanej publicznej macierzy; MLU270 to starszy sprzęt, którego macierz nie obejmuje | Neuware i MagicMind | Macierz MagicMind 1.7 w repozytorium wymienia YOLOv3/4/5/7/8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN i modele segmentacji | Historyczne przykłady modeli są publiczne; dostęp do aktualnego SDK/kontenerów jest ograniczony |
| [Sunplus](https://sunplus.atlassian.net/wiki/spaces/C3/pages/1971126471/SP7350%2BSpecification) | SP7350/C3V, około 4.1–4.6 reklamowanego TOPS | Docker Vivante Acuity NPU, środowisko uruchomieniowe SNNF i `.nb` | Oficjalna dokumentacja udostępnia YOLOv5 i niestandardowe wdrożenie [YOLOv8 od początku do końca](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350) | Publiczne źródła/dokumentacja i ekosystem płytek |
| [ESWIN Computing](https://www.eswincomputing.com/en/news/info/96.html) | SoC RISC-V EIC7700/EIC7700X i dwuukładowe [EIC7702/EIC7702X](https://www.eswincomputing.com/en/news/info/104.html) | ENNP: EsQuant, aktualny kompilator EsAAC, symulator i środowisko uruchomieniowe ESSDK; `.model`; starsza dokumentacja używała `ennc-compile` | Dokumentacja ENNP hostowana przez Milk-V obejmuje samouczek [YOLOv3 od początku do końca](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3), przykłady MobileNetV2 i ResNet | Mieszanka dokumentacji producenta i dostawców płytek; pobieranie regionalne |
| [Nuvoton](https://www.nuvoton.com/products/microcontrollers/arm-cortex-m55-mcus/m55m1-series/index.html) | Mikrokontroler M55M1 z Ethos-U55-256 | TFLite Micro, Arm Vela i NuEdgeWise/NuML | Przykłady [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise) wymieniają YOLOv8-nano, YOLOX-nano, YOLO Fastest, SSD-MobileNet i modele klasyfikacji | W większości publiczny toolchain mikrokontrolera |
| [Realtek](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | Platforma kamery AmebaPro2 RTL8735B / AMB82-mini | SDK Arduino/FreeRTOS, VoE i API NeuralNetwork; `.nb` | Dołączone modele obejmują YOLOv3-tiny, YOLOv4-tiny, YOLOv7-tiny, SCRFD i MobileFaceNet | Środowisko uruchomieniowe publiczne; dostęp do [konwertera niestandardowego](https://ameba-doc-arduino-sdk.readthedocs-hosted.com/en/latest/FAQ/offline_ai_model_conversion_steps.html) wymaga kontaktu |
| [Telechips](https://docs.topst.ai/product/p/ai) | Płytka TCC7500 / TOPST AI, reklamowane 8 TOPS | TC-NN-Toolkit / Enlight SDK; pośredni `.enlight` i skompilowany pakiet wdrożeniowy | [Oficjalny projekt TOPST](https://docs.topst.ai/blog/31) wdrożył YOLOv8s; konwersje UFLD v1/v2 nie powiodły się z powodu nieobsługiwanych warstw i zaproponowano jedynie obejście przez podział/przetwarzanie końcowe. [YOLOv4](https://community.topst.ai/t/segmentation-fault/358) i [inna ścieżka YOLOv8](https://community.topst.ai/t/segmentation-fault-error/415) pojawiają się w wątkach społeczności | Dokumentacja płytki publiczna; pobranie kompilatora/operatorów często wymaga uprawnień |
| [T-Head](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | TH1520, reklamowane 4-TOPS INT8, na LicheePi 4A | Kompilator HHB i CSI-NN2/SHL; `hhb.bm` oraz wygenerowany kod/parametry | Oficjalne przykłady dostawcy płytki uruchamiają YOLOv5n/s i MobileNetV2 na NPU | Publiczne przykłady; niepewna konserwacja starszego toolchainu |
| [SigmaStar](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | Aktualny układ SoC do kamer SSU9383CM i starsze rodziny IPU | Aktualne środowisko uruchomieniowe MI_IPU; starszy [toolchain SGS_IPU](https://wx.comake.online/doc/doc/SigmaStarDocs-SSC9381G_9351_Pudding-ULS00V040-20210913/customer/development/dla/tools.html) tworzył `.sim`, a następnie `sgsimg.img` | Starszy [oficjalny moduł postprocessingu SDK](https://wx.comake.online/doc/doc/Sigmastar_SDK_v1.2.2/module/User_Guide/Common/SigmaStar_Post_Processing_Module.html) wymienia SSD i YOLOv1/2/3; publiczna zgodność tych modeli i artefaktów z SSU9383CM nie została zweryfikowana | Aktualny układ, ale publiczne dowody dla kompilatora/modeli dotyczą różnych generacji |
| [HiSilicon](https://www.hisilicon.com/cn/products/smart-vision/machine-vision/hi3516cv610) | Układy SoC do inteligentnej wizji Hi3516CV610/DV500, Hi3519DV500 i Hi3403V100 | ATC do `.om` z środowiskiem uruchomieniowym NNN/SVP-NNN na płytce | Macierz dla konkretnych celów w HiSpark, zwłaszcza Hi3403 i Hi3591P, wymienia YOLOv3 do YOLO11, estymację pozy, segmentację, OBB, OCR i głębię; nie potwierdza to działania na każdym SoC w tym wierszu | Aktywna i odrębna od Ascend; modele repozytorium są oznaczone jako wyłącznie niekomercyjne |

### Dedykowane akceleratory brzegowe i moduły

| Firma | Odpowiedni układ | SDK i artefakt | Publicznie udokumentowane modele | Dostęp |
|---|---|---|---|---|
| [Hailo](https://github.com/hailo-ai/hailo_model_zoo) | Hailo-8, Hailo-8L, Hailo-10H i rodzina Hailo-15 | Dataflow Compiler i HailoRT; `.hef` | YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 i YOLO26, a także YOLOX, DAMO-YOLO, SSD, EfficientDet, segmentacja, estymacja pozy i OBB, z tabelami modeli zależnymi od generacji | Model Zoo publiczne; kompilator przez Developer Zone |
| [Axelera AI](https://docs.axelera.ai/sdk/reference/models/model-zoo/) | Dostępne produkty Metis; zapowiedziane produkty Europa i Titania | Publiczny Voyager SDK; Pipeline Builder w wersji alpha `.axm`/`.axe`, klasyczne pakiety `.axmodel` | Zweryfikowane YOLOv3/5/7/8/9/10/11/26, YOLOX, YOLO-NAS, OBB, estymacja pozy, segmentacja i wiele modeli spoza YOLO | SDK publiczny na GitHubie; wsparcie klienta wymaga konta |
| [DEEPX](https://developer.deepx.ai/modelzoo/) | DX-M1 i DX-M1M | DXNN SDK, DX-COM i DX-RT; `.dxnn` | 15 sierpnia 2026 r. katalog modeli zawierał 354 wpisy dla DX-COM 2.4.0/DX-RT 3.4.0, w tym YOLOv3 do YOLO11 i YOLO26, YOLOX, SSD, EfficientDet, NanoDet, segmentację, estymację pozy i OBB | Publiczne zasoby deweloperskie i pakiet narzędzi |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | Akcelerator MX3 i moduły wieloukładowe | MemryX SDK, Neural Compiler i środowisko uruchomieniowe; `.dfp` | Oficjalne przykłady i informacje o wydaniach obejmują detekcję, segmentację i estymację pozy, w tym YOLOv10, YOLO11 i YOLO26 | Publiczna dokumentacja i SDK |
| [Kneron](https://doc.kneron.com/docs/) | KL520, KL530, KL630, KL720 i KL730 mają aktualną dokumentację kompilatora; KL830 pojawia się w niektórych API PLUS, ale nie na bieżącej liście celów kompilatora | Kneron PLUS i Model Toolchain; `.nef` dla udokumentowanych celów kompilatora | Oficjalny [proces YOLO](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/) skupia się na Tiny-YOLOv3; inne materiały obejmują YOLOv5 | Publiczna dokumentacja; dostępność pakietów narzędzi zależy od układu |
| [SiMa.ai](https://docs.sima.ai/pages/palette/modelsdk.html) | MLSoC i produkcyjna platforma Modalix o wydajności 50 TOPS | Palette, ModelSDK, MLA Compiler i ModelExecutor | Publiczne wydania weryfikują YOLOv7/v8, YOLOX, estymację pozy/segmentację, DETR, Mask R-CNN i potoki EfficientDet | Publiczna dokumentacja; komercyjny SDK produktu |
| [EdgeCortix](https://www.edgecortix.com/en/hardware) | SAKURA-II, reklamowany akcelerator 60 TOPS w wersjach M.2 i PCIe | Kompilator MERA i framework | Dostawca deklaruje obsługę od wizji po generatywną AI, ale nie publikuje wystarczająco precyzyjnej, aktualnej macierzy zgodności YOLO | Zapytanie o wersję próbną/zakup; walidacja z partnerem |
| [BrainChip](https://brainchip.com/metatf-dev-tools/) | Dostępny koprocesor/moduł M.2 AKD1500; starsze platformy AKD1000; IP Akida 2 | Pakiety MetaTF: `akida-models`, `quantizeml`, `cnn2snn` i środowisko uruchomieniowe `akida` | Karta modelu Akida 2 wymienia detektor AkidaNet0.5 YOLOv2, CenterNet, AkidaUNet i rozpoznawanie twarzy; te wyniki nie potwierdzają automatycznie zgodności z AKD1500 | Publiczne podstawowe pakiety Python; mapowanie sprzętu zależy od celu |
| [Blaize](https://www.blaize.com/products/) | Produkty P1600, Pathfinder i Xplorer | SDK Picasso, NetDeploy i AI Studio | Wizja komputerowa jest rynkiem docelowym, ale nie znaleziono publicznej, możliwej do audytu macierzy zgodności nazwanych modeli | Komercyjny/dostęp ograniczony |
| [Google Coral](https://github.com/google-coral/edgetpu) | Starsze produkty Edge TPU USB, PCIe, M.2 i Dev Board; osobne aktywne open source [IP Coral NPU](https://github.com/google-coral/coralnpu) | Starszy Edge TPU Compiler/`libedgetpu`/PyCoral; nowe Coral NPU RISC-V udostępnia IP, RTL/symulację i przykłady ELF | Starsze przykłady skupiają się na SSD MobileNet, klasyfikacji, DeepLab i MoveNet; nowe IP nie ma publicznej macierzy wdrożeń YOLO i nie jest bezpośrednim następcą produktu Edge TPU | Główne starsze repozytoria zarchiwizowane; nowe IP Coral NPU jest aktywnie rozwijane |
| [Lattice Semiconductor](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | Układy FPGA ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E i Avant-X | sensAI Studio, Neural Network Compiler i konfigurowalne IP akceleratora | Aktualna tabela kompilatora wymienia YOLOv1, YOLOv5, YOLOv8 i YOLO11 w trybach zależnych od celu, a także SSD, MobileNetV2-SSD, ResNet i ENet | Kompilator i instrukcje publiczne; warunki IP różne, Advanced CNN Accelerator jest komercyjny |
| [Mobilint](https://www.mobilint.com/sdk-qb) | Akceleratory REGULUS 10 TOPS i ARIES MLA100 80 TOPS | SDK qb; kompilator INT8 i `.mxq` | Publiczny [katalog modeli](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md) wymienia detekcję YOLOv3/5/7/8/9/10/11/12/26 oraz warianty segmentacji, estymacji pozy i OBB | Publiczna dokumentacja/modele; SDK i sprzęt są komercyjne |
| [Rebellions](https://rebellions.ai/developers/) | Aktywne ATOM+ CA22 i ATOM-Max CA25; wycofane ATOM CA02/ATOM+ CA12; status narzędzi ATOM-Lite CA21 niejasny | Kompilator/środowisko uruchomieniowe/profiler RBLN; `.rbln` | Oficjalne wydania wsparcia wymieniają YOLOv3, rodziny YOLOv5/6/7/8 i aktualne samouczki YOLOv8 | Dokumentacja publiczna; wheel kompilatora wymaga danych logowania do portalu |
| [FuriosaAI](https://furiosa.ai/warboy) | Warboy Gen1 przeznaczony do wizji; odrębna generacja RNGD | Furiosa SDK; INT8 `.enf` dla Warboy, osobny `.fxb` dla RNGD | Katalog Warboy dokumentuje SSD, YOLOv5M/L i YOLOv7-w6-pose; plan RNGD oznacza obsługę YOLOv8m jako ukończoną w 2024 Q4, ale nie udostępnia szczegółowej aktualnej macierzy wizji | Dostęp przez konto/IAM; generacje należy rozdzielać |

### Powiązane platformy porównywane z NPU

| Firma | Sprzęt | Stos wdrożeniowy | Dlaczego uwzględniono ją w porównaniu |
|---|---|---|---|
| [NVIDIA](https://github.com/NVIDIA-AI-IOT/deepstream_tools/tree/main/yolo_deepstream) | GPU Jetson Orin i DLA; GPU Jetson Thor bez DLA | JetPack, TensorRT i DeepStream; `.engine` | Bardzo dojrzałe wdrażanie wizji; oficjalne narzędzia DeepStream dokumentują YOLOv4/v7/v8/v9/11, w tym konfigurację OBB YOLO11. Wiele wyników dotyczy GPU, a nieobsługiwane warstwy DLA w Orin mogą być wykonywane na GPU. |
| [AMD](https://vitisai.docs.amd.com/en/6.2/) | Cele Kria/starsze DPU; Vitis AI 6.2 GA dla Versal AI Edge Gen1 VEK280/VE2802 i Gen2 VEK385; osobne NPU klienckie Ryzen AI | Starsze `.xmodel`; migawka NPU związana z celem Gen1; katalog pamięci podręcznej kompilacji Gen2 lub pakiet produkcyjny `.rai` | Vitis AI publikuje materiały wizualne, ale starsze DPU, Versal Gen1, Versal Gen2 i Ryzen AI mają odrębne kontrakty kompilacji i wdrożenia. |
| [Microchip](https://www.microchip.com/en-us/products/fpgas-and-plds/fpga-and-soc-design-tools/vectorblox) | Układ FPGA PolarFire SoC z konfigurowalnym IP akceleratora CoreVectorBlox | Publiczny VectorBlox SDK 3.1 i Libero; zasoby wdrożeniowe `.vnnx`, `.hex` i `.ucomp` | Aktualne [samouczki](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md) obejmują YOLOv5n, YOLOv8 do detekcji/klasyfikacji/OBB/estymacji pozy/segmentacji, YOLOv9t i inne modele wizji; SDK 3.1 jest obecnie przeznaczony dla PolarFire SoC Video Kit. |
| [Intel](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) | NPU Core Ultra, CPU i zintegrowany/dyskretny GPU | OpenVINO IR lub skompilowany model cache | OpenVINO udostępnia publiczną ścieżkę między różnymi XPU; należy korzystać z kolumn NPU w [macierzy zweryfikowanych modeli](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html), zamiast zakładać, że każdy przykład YOLO w OpenVINO został zweryfikowany na NPU. |
| [Apple](https://developer.apple.com/machine-learning/core-ml/) | Apple Neural Engine w układach A-series od A11 wzwyż i układach M-series | Core ML i `coremltools`; `.mlpackage`/`.mlmodelc` | Bardzo dostępne konsumenckie NPU przez Core ML, ale Apple ukrywa dokładny podział pracy między CPU/GPU/Neural Engine i nie udostępnia kompilatora sprzętowego dla YOLO. |
| [GreenWaves Technologies](https://github.com/GreenWaves-Technologies/nn_menu_gap9) | GAP9 z silnikiem neuronowym NE16 | GAP SDK, NNTool i kod generowany przez AutoTiler | Oficjalne repozytoria obejmują MobileNet SSD, wykrywanie twarzy, klasyfikację i dedykowany [detektor osób YOLOX](https://github.com/GreenWaves-Technologies/yolox_people_detection). Pełny SDK jest dostępny dla zakwalifikowanych klientów. |
| [Syntiant](https://www.syntiant.com/hardware) | Produkowany seryjnie NDP200 i próbkowany NDP250 Neural Decision Processor | Syntiant SDK i pakiety wdrożeniowe | Ultraniskoenergetyczne procesory wizji/sensorów stale gotowe do pracy: dla NDP200 podano poniżej 1 mW, a dla rozpoznawania obrazu na NDP250 poniżej 30 mW. Nie znaleziono szerokiej publicznej macierzy YOLO. |
## Amlogic: dwie generacje NPU, a nie jedna

Amlogic wymaga dokładniejszego omówienia, ponieważ informacje w sieci często łączą niezgodne ze sobą produkty. Firma ma starszą ścieżkę opartą na VeriSilicon i nowszą, własną ścieżkę ADLA. Korzystają z innych kompilatorów, artefaktów i środowisk uruchomieniowych.

| Generacja | Przykładowe układy | NPU i toolchain | Skompilowany artefakt | Stan praktyczny |
|---|---|---|---|---|
| Starsza | A311D i S905D3, często spotykane w Khadas VIM3/VIM3L | VeriSilicon Vivante VIPNano-QI, Acuity toolkit, KSNN i starszy `aml_npu_sdk` | `.nb` w katalogu wyjściowym `nbg_unify` | Istniejące płytki i demonstracje; osobna integracja starszej platformy |
| Aktualna ADLA2 | C308L/C302X, S928X, A311D2, T968D4, S905X5/S905D5 i C302X2 | Amlogic ADLA2, AMLNN Toolkit i NNSDK2 | `.adla` dla konkretnego celu; W8A8 lub W8A16 | Aktualny stos ukierunkowany na liczby całkowite |
| Aktualna ADLA3 | A311Y3, C305X2 i A123X | Amlogic ADLA3, AMLNN Toolkit i NNSDK2 | `.adla` dla konkretnego celu; W4A8, W8A8, W4A16, W8A16 lub W16A16 | Dodaje natywną obsługę INT4, FP16 i BF16 |

[Arkusz danych A311D](https://dl.khadas.com/products/vim3/datasheet/a311d-datasheet.pdf) wskazuje pierwotne NPU INT8 o wydajności 5 TOPS. Starsza [strona aplikacji Khadas VIM3](https://docs.khadas.com/products/sbc/vim3/npu/npu-app) i [omówienie NPU](https://docs.khadas.com/products/sbc/vim3/npu/start) dokumentują DenseNet CTC, MTCNN, RetinaFace, YOLOFace, YOLOv2, YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv7-tiny i YOLOv8n; omówienie dodatkowo dokumentuje YOLOv8n-pose i oznacza FaceNet jako przestarzały. To rzeczywiste przykłady, ale dotyczą starego ekosystemu Acuity `.nb`, a nie aktualnych układów ADLA. A311D to nie A311D2, a C305X to nie C305X2.

Jest też druga pułapka sprzętowa. [Przewodnik po rewizjach Khadas VIM4](https://docs.khadas.com/products/sbc/vim4/configurations/identify-version) podaje, że pierwotna płyta V12/A311D2 w rewizji B nie ma NPU; reklamowane NPU 3.2 TOPS jest dostępne w płytach V13A i nowszych, używających układu A311D2-N0D w rewizji C. Sama nazwa produktu nie wystarcza do wyboru urządzenia testowego.

### Aktualny stos AMLNN i ADLA

Aktualny [AMLNN Toolkit](https://github.com/Amlogic-NN/amlnn-toolkit/tree/7d3cc9a36179b756ce79c953c14d43acab1f2af5) obsługuje konwersję modeli, kwantyzację, kompilację, inferencję i profilowanie. Przypięty [przewodnik użytkownika z maja 2026 r.](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/02_Amlogic_NPU_User_Guide_V0.1.pdf) dokumentuje import ONNX, zmiennoprzecinkowy lub skwantyzowany TFLite, TorchScript `.pt` i PyTorch 2 ExportedProgram/PT2. Szczegółowy przewodnik oznacza ścieżki TensorFlow, Paddle i Keras jako planowane, mimo szerszego opisu w omówieniu repozytorium. Kompilator tworzy `.adla`. Wdrożenie w Pythonie korzysta z AMLNN lub `amlnn_edge_toolkit_lite`; kod natywny C/C++ łączy `libnnsdk.so` przez `nnsdk2.h`. Programowanie po stronie hosta może wykorzystywać ADB z `nnserver`; środowisko uruchomieniowe działa też na Androidzie, Buildroot, Yocto i wybranych systemach Debian/Armbian.

Opublikowane identyfikatory platform toolchainu:

| ID celu | Platforma |
|---:|---|
| `001` | C308L / C302X |
| `002` | S928X |
| `003` | A311D2 |
| `004` | T968D4 |
| `005` | S905X5 / S905D5 |
| `006` | C302X2 |
| `007` | A311Y3 |
| `008` | C305X2 / A123X w szczegółowych dokumentach i macierzy modeli |

Pole celu nie jest kosmetyczne. Amlogic wymaga zgodności pliku `.adla`, `nnsdk2.h`, `libnnsdk.so`, BSP oraz generacji kompilatora/środowiska uruchomieniowego. Przypięty [Quick Start](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/01_Amlogic_NPU_Quick_Start_guide_V0.1.pdf) wymaga sterownika ADLA 2.0.2 lub nowszego, NNSDK 3.0.0 lub nowszego i NNSDK2 1.0.0 lub nowszego; [przewodnik wdrażania na Androidzie](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/06_Amlogic_NPU_Android_Deployment_Guide_V0.1.pdf) konkretnie opisuje relację biblioteki, nagłówka, sterownika i BSP. `.adla` należy traktować jako artefakt kompilacji związany z ABI, a nie przenośny model.

Kwantyzacja zależy też od generacji NPU. Cele ADLA2 001–006 dokumentują kompilację W8A8 i W8A16. Cele ADLA3 007 i 008 dodają kombinacje W4A8, W4A16, W8A8, W8A16 i W16A16 oraz natywną obsługę INT4, FP16 i BF16 w [przewodniku operatorów](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/05_Amlogic_NPU_Support_Operator_List_V0.1.pdf). Amlogic zaleca 200–500 reprezentatywnych próbek kalibracyjnych. Opcja danych losowych jest przeznaczona wyłącznie do testów wydajności, a nie do kwantyzacji zachowującej dokładność.

### Udokumentowane modele Amlogic

Przypięty [model playground Amlogic](https://github.com/Amlogic-NN/amlnn-model-playground/blob/03ca63ef20c4f1560722de26ab910826884f334a/README.md) jest znacznie mocniejszym dowodem niż ogólne twierdzenie o obsłudze ONNX. Opublikowana macierz obsługi/przykładów wymienia A311D2, S905X5, A311Y3, C305X2 i A123X. To, że kompilator akceptuje pozostałe identyfikatory celów, nie dowodzi, że cała ta lista modeli została na nich zweryfikowana.

| Zadanie | Udokumentowane modele |
|---|---|
| Klasyfikacja | MobileNetV2 i ResNet50-v2; DINO na ADLA3 |
| Detekcja obiektów | PP-YOLOE, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv10, YOLO11, YOLOE, YOLO-World, YOLOX i detekcja kodów QR |
| Twarze i gesty | RetinaFace i rozpoznawanie gestów |
| Segmentacja | DeepLabV3, PP-LiteSeg, YOLOv5-seg i YOLOv8-seg |
| Detekcja obróconych obiektów | YOLOv8 OBB |
| Estymacja pozy | Detektor/punkty charakterystyczne BlazePose i YOLOv8 pose |
| OCR i mowa | LPRNet, warianty PaddleOCR, Whisper Tiny i SenseVoice na wybranych nowszych układach |
| Nowsze transformatory i prace multimodalne | DETR, MobileSAM, CLIP/MobileCLIP oraz wybrane przykłady LLM/VLM o niskiej precyzji na nowszych platformach |

To samo repozytorium publikuje poniższe wyniki W8A8 dla modelu i środowiska uruchomieniowego. Są to pomiary dostawcy dla sieci neuronowej na NPU, a nie kompletne potoki kamerowe.

| Model i wejście | S905X5 | A311D2 | A311Y3 |
|---|---:|---:|---:|
| YOLOv8n, 640 x 640 | 101.72 FPS | 95.14 FPS | 191.06 FPS |
| YOLOv8s, 640 x 640 | 42.33 FPS | 42.77 FPS | 83.08 FPS |
| YOLOv8m, 640 x 640 | 19.67 FPS | 19.82 FPS | 35.30 FPS |
| YOLOv8l, 640 x 640 | 10.53 FPS | 10.12 FPS | 18.37 FPS |
| YOLO11n, 640 x 640 | 41.14 FPS | 41.48 FPS | 62.24 FPS |

Zastrzeżenia są tu wyjątkowo ważne. Przewodnik operatorów Amlogic umieszcza NMS w oprogramowaniu zarówno dla ADLA2, jak i ADLA3. README definiuje te pomiary jako wyniki natywnego modelu na NPU i pomija przetwarzanie wstępne oraz końcowe; nie ujawniono, czy uwzględniono wszystkie transfery pamięci. Wiersze dokładności YOLO wykorzystują podzbiory COCO liczące 300 obrazów, a wiersze klasyfikacji używają ImageNet `val1000`. Tabele podają też inną latencję YOLOv8n na A311Y3 niż tabela FPS, w warunkach, których repozytorium nie uzgadnia. Nie podano taktowania, trybu zasilania, chłodzenia, stanu ustalonego temperatury, dokładnej wersji SDK/BSP ani czasu testu. Wartości te służą do zrozumienia jednego stosu dostawcy, a nie do porównywania go z innym.

### Dlaczego Amlogic jest interesujący strategicznie

Amlogic łączy cztery nietypowe cechy: szeroką obecność wbudowanych SoC, nowo upubliczniony stos kompilatora, aktualną macierz modeli mocno pokrywającą się z zadaniami LibreYOLO oraz niewielką liczbę integracji pierwszej klasy w głównych bibliotekach trenowania. Aktualne repozytoria są też młode: upubliczniono je pod koniec 2025 r. i na początku 2026 r., a obszerne instrukcje pochodzą z maja–lipca 2026 r. Oznacza to realną szansę na wczesną integrację oraz ryzyko niestabilnego API.

Dobra integracja powinna wykorzystywać deterministyczny eksport ONNX LibreYOLO jako wejście kompilatora, wymagać reprezentatywnych obrazów kalibracyjnych dla kompilacji o niskiej precyzji, tworzyć `.adla` wraz z manifestem metadanych i ładować model przez adapter NNSDK2. Starszy A311D powinien mieć wyraźnie odrębny cel `.nb`, ponieważ przedstawianie `.nb` i `.adla` jako jednego backendu powodowałoby ciche błędy zgodności. Repozytoria Amlogic mają główne licencje Apache-2.0, ale przed kopiowaniem przez LibreYOLO dołączonych pakietów wheel, bibliotek współdzielonych, `nnserver` i plików binarnych Android AAR należy bezpośrednio potwierdzić prawa do redystrybucji.

## Najsilniejsze publiczne ekosystemy wdrożeniowe

Poniższe firmy oferują obecnie najbardziej przejrzyste połączenie dostępnego sprzętu, publicznych materiałów technicznych i dowodów dla nazwanych modeli. Nie oznacza to, że są zawsze szybsze. Ich deklaracje zgodności łatwiej ocenić przed zakupem sprzętu.

### Hailo

Hailo jest przykładem dedykowanego ekosystemu akceleratorów wizji. Modele są parsowane do archiwum Hailo, optymalizowane i kwantyzowane na reprezentatywnych obrazach, a następnie kompilowane do pliku Hailo Executable Format (`.hef`). Aplikacje wczytują HEF przez HailoRT.

[Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) zawiera przepisy, wstępnie wytrenowane modele, konfiguracje postprocessingu i dane wydajnościowe dla detekcji, segmentacji, klasyfikacji, estymacji pozy i innych zadań. Obsługa YOLO obejmuje YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 i YOLO26, a także YOLOX, DAMO-YOLO, SSD i EfficientDet. Przypięta wersją [tabela detekcji obiektów Hailo-8](https://github.com/hailo-ai/hailo_model_zoo/blob/v2.19.0/docs/public_models/HAILO8/HAILO8_object_detection.rst) jest lepszym źródłem zgodności niż ogólna strona produktu, a [samodzielna aplikacja](https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/python/standalone_apps/object_detection/README.md) opisuje wdrażalne potoki YOLO.

Istotna jest granica między wersjami. Hailo-8 i Hailo-8L korzystają ze starszej linii Model Zoo 2.x, Dataflow Compiler 3.x i HailoRT 4.x, natomiast Hailo-10 i Hailo-15 używają bieżącej generacji 5.x. Przepisy, zachowanie parsera i pliki HEF nie są wymienne tylko dlatego, że wszystkie urządzenia mają nazwę Hailo.

Hailo-15 ma też odrębną warstwę aplikacyjną. Używa Vision Processor Software Package i Hailo Media Library zamiast ścieżki hosta w stylu TAPPAS dla Hailo-8/10H. Publiczne repozytorium referencyjne [`hailo-camera-apps`](https://github.com/hailo-ai/hailo-camera-apps) zarchiwizowano 3 maja 2026 r., a [`hailo-apps-core`](https://github.com/hailo-ai/hailo-apps-core) pozostaje osobnym publicznym frameworkiem aplikacji. Archiwizacja repozytorium nie dowodzi końca cyklu życia produktu, ale projekty Hailo-15 powinny potwierdzić w Developer Zone aktualny pakiet aplikacyjny.

[Przewodnik wdrażania Hailo](/docs/export/hailo) LibreYOLO celowo kończy się na statycznym przekazaniu ONNX, ponieważ zastrzeżonego kompilatora nie można dołączyć jako zależności Python. To uczciwa granica między przygotowaniem odpowiedniego grafu a deklarowaniem przetestowanego HEF.

### Rockchip

Rockchip ma jeden z największych ekosystemów hobbystycznego i komercyjnego embedded Linux, zwłaszcza wokół płytek RK3588, RK3576 i RK356x. Publiczny [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) konwertuje i kwantyzuje modele na hoście x86 z Linuxem, a RKNN Runtime lub Toolkit-Lite uruchamia wynikowy plik `.rknn` na płytce.

[RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) wyjątkowo jasno opisuje układy, rodziny modeli i precyzję. Aktualna tabela wymienia ścieżki FP16 i INT8 dla YOLOv5/6/7/8/10/11, YOLOX, PP-YOLOE, YOLO-World i wariantów segmentacji YOLOv5/v8; YOLOv8 OBB i YOLOv8 pose są wymienione wyłącznie dla INT8. Uwzględniono także OCR, twarze i inne sieci segmentacji.

LibreYOLO ma już zachowawczy, bezpośredni [eksporter RKNN](/docs/export/rknn). Weryfikuje cztery konkretne warianty detekcji na RK3588, może porównać symulator kompilatora z ONNX Runtime i odrzuca nieweryfikowane rodziny zamiast utożsamiać udaną kompilację z poprawnymi predykcjami. To dobry wzorzec dla deklaracji obsługi w przyszłych backendach NPU.

### Axelera AI

Axelera AI, często błędnie zapisywana jako „Accelera”, buduje AIPU Metis i sprzedaje go w produktach M.2, PCIe i wieloukładowych. Rodzina Metis jest dostępna w sprzedaży; Europa i Titania to zapowiedziane produkty, więc ich dostępności nie należy sprowadzać do jednego statusu. [Voyager SDK jest publiczny na GitHubie](https://github.com/axelera-ai-hub/voyager-sdk) i obsługuje wybór modelu, kompilację, kwantyzację, wykonanie i potoki aplikacji; wsparcie klienta nadal wymaga konta.

Publiczny [Voyager Model Zoo](https://docs.axelera.ai/sdk/reference/models/model-zoo/) należy do najbardziej informatywnych w branży. Wymienia wariant modelu, rozmiar wejścia, precyzję, dokładność i zmierzoną wydajność dla YOLOv3, YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26 n/s/m/l/x, YOLOX, YOLO-NAS, OBB, estymacji pozy i segmentacji, a także wiele modeli klasyfikacji i predykcji gęstej. Publikowanie strat kwantyzacji i dokładności obok szybkości jest bardziej użyteczne niż podanie samego szczytowego TOPS.

Nazwy artefaktów zmieniły się wraz z generacją API. Przepływ kompilacji alpha [Pipeline Builder](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) tworzy model `.axm` i może spakować przenośny pipeline jako `.axe`. [Dokumentacja klasycznego pipeline'u](https://docs.axelera.ai/sdk/reference/pipeline/model-formats/) opisuje `.axmodel` wraz z metadanymi modelu i manifestem. Integracja powinna wykrywać zainstalowaną generację Voyager, zamiast na stałe zakładać jedno rozszerzenie.

### Qualcomm

Qualcomm oferuje ogromny zasięg wdrożeniowy, ale określenie „NPU Qualcomm” ukrywa wiele interfejsów. Deweloperzy korzystają z Hexagon HTP przez QAIRT/QNN, starszej ścieżki SNPE, usług kompilacji Qualcomm AI Hub, LiteRT lub dostawcy wykonawczego QNN dla ONNX Runtime, zależnie od urządzenia i klasy produktu.

Oficjalne repozytorium [Qualcomm AI Hub Models](https://github.com/qualcomm/ai-hub-models) dostarcza mocnych dowodów na poziomie modeli. Publikuje zoptymalizowane pakiety YOLOv3, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26 do detekcji/segmentacji/estymacji pozy, YOLOX, YOLO-World, YOLOR, RF-DETR oraz wiele modeli klasyfikacji, głębi i estymacji pozy. AI Hub udostępnia też profilowanie dla konkretnych urządzeń, co jest ważne, ponieważ model skompilowany dla jednej generacji HTP nie staje się automatycznie przenośny na inną.

Główny koszt integracji wynika z liczby kombinacji: Android kontra embedded Linux, części QCS kontra konsumencki Snapdragon, architektura HTP, wersja QNN/QAIRT i schemat kwantyzacji mają znaczenie. Aktualne strony Qualcomm są też sprzeczne w kwestii statusu [Dragonwing IQ-9075](https://www.qualcomm.com/internet-of-things/products/iq9-series/iq-9075): strona produktu określa go jako Active, a [EVK](https://www.qualcomm.com/developer/hardware/qualcomm-iq-9075-evaluation-kit-evk) jest dostępny do ewaluacji, natomiast [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program) nadal oznacza IQ-9075 jako Sampling i podaje trwałość do 2038 r. Prefiks SKU do zamówień to QCS9075; Qualcomm reklamuje konfiguracje 50 i 100 gęstych INT8 TOPS oraz obsługę Ubuntu/Yocto. Status produkcyjny trzeba potwierdzić dla konkretnego SKU, a integracja LibreYOLO powinna zapisywać ten SKU zamiast tworzyć ogólny katalog „Qualcomm”.

### DEEPX

Akceleratory DX-M1/DX-M1M firmy DEEPX korzystają z DXNN SDK. DX-COM kompiluje i kwantyzuje model, DX-RT go uruchamia, a wdrażany artefakt ma format `.dxnn`. Firma publikuje [DX-AllSuite](https://github.com/DEEPX-AI/dx-all-suite), przykłady aplikacji w [DX-APP](https://github.com/DEEPX-AI/dx_app) i obszerny [internetowy katalog modeli](https://developer.deepx.ai/modelzoo/).

Podczas sprawdzania 15 sierpnia 2026 r. publiczny katalog podawał 354 wpisy dla DX-COM 2.4.0 i DX-RT 3.4.0. Udokumentowana obsługa obejmuje YOLOv3 do YOLO11 i YOLO26, YOLOX, SSD, EfficientDet, NanoDet, DAMO-YOLO, segmentację YOLO, estymację pozy i obrócone ramki. DEEPX jest szczególnie wiarygodnym kandydatem do integracji, ponieważ granice kompilatora i środowiska uruchomieniowego oraz artefakt wdrożeniowy mają jasno określone nazwy, a publiczna obsługa wizji jest szeroka.

## Przemysłowe SoC to kilka ekosystemów, nie jeden

Dostawcy przemysłowi często oferują dłuższy cykl życia produktów oraz lepszą integrację kamer, bezpieczeństwa i czasu rzeczywistego niż SoC do płytek hobbystycznych. Ich stosy AI bywają bardziej złożone, bo jeden dostawca może jednocześnie oferować kilka niepowiązanych architektur NPU.

### Texas Instruments

Aktualne procesory TI do Edge AI obejmują AM62A, AM67A, AM68A, AM69A i powiązane urządzenia TDA4. Ścieżka programowa łączy Processor SDK Linux, kompilator/środowisko uruchomieniowe TIDL, [Edge AI TIDL Tools](https://github.com/TexasInstruments/edgeai-tidl-tools) i [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo). TIDL może współpracować z ONNX Runtime, TensorFlow Lite i innymi środowiskami aplikacyjnymi, odciążając obsługiwane podgrafy.

TI publikuje więcej niż samo twierdzenie o imporcie frameworka: katalog zawiera przekonwertowane modele i metadane wydajności dla detekcji obiektów, segmentacji, estymacji pozy, klasyfikacji, głębi i innych zadań. TI zaznacza też, że artefakty z katalogu modeli są punktem wyjścia do prac deweloperskich, a nie automatycznie gotowymi zasobami produkcyjnymi. To cenne zastrzeżenie często pomijane w porównaniach dostawców.

TI wymienia również [TDA54-Q1](https://www.ti.com/product/TDA54-Q1) jako Preview, z maksymalnie czterema NPU C7 i do 400 TOPS dla tej części; cała rodzina TDA5 jest reklamowana do 1 200 TOPS. To deklaracje dostawcy dotyczące nowej generacji, a nie podstawa do przeniesienia walidacji modeli TIDL z AM6xA/TDA4 na TDA54-Q1 przed publikacją macierzy dla konkretnego celu.

### NXP

NXP wymaga co najmniej czterech opisów backendu:

| Cel NXP | Akcelerator | Ścieżka kompilatora/środowiska uruchomieniowego |
|---|---|---|
| i.MX 8M Plus | NPU VeriSilicon Vivante 2.3 TOPS | eIQ z TIM-VX/VX delegate |
| i.MX 93 | Arm Ethos-U65 | Skwantyzowany TFLite z Arm Vela |
| i.MX 95 | NPU NXP eIQ Neutron | Neutron Converter i eIQ runtime |
| Ara-1 / [Ara240](https://www.nxp.com/products/ARA240) | Dyskretne NPU wywodzące się z Kinara; Ara240 reklamuje do 40 eTOPS | Ara SDK, integrowane z szerszym ekosystemem eIQ |

[eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) zawiera zasoby lub przepisy dla YOLOv4-tiny, YOLOv8, NanoDet, CenterNet, FastestDet, SSD Lite, YOLACT i innych modeli; wpis YOLOv5 dodano wyłącznie jako dokumentację. Wpis zweryfikowany dla jednego backendu nie stanowi dowodu dla wszystkich czterech. Własne [wskazówki NXP dotyczące eksportu YOLO dla platform i.MX](https://community.nxp.com/t5/i-MX-Processors-Knowledge-Base/Exporting-YOLO-Models-for-NXP-i-MX-Platforms/ta-p/2381361) pokazują problem konwersji zależnej od celu. NXP informuje, że monolityczny [eIQ Toolkit przestał być aktualizowany po wersji 1.17 w Q3 2025](https://community.nxp.com/t5/eIQ-Machine-Learning-Software/eIQ-FAQ/ta-p/1099741); bieżące procesy korzystają z samodzielnych pakietów, takich jak eIQ Neutron SDK.

NXP [ukończyło przejęcie Kinara w październiku 2025 r.](https://media.nxp.com/news-releases/news-release-details/nxp-completes-acquisitions-aviva-links-and-kinara-advance/). [Arkusz informacyjny Ara240](https://www.nxp.com/docs/en/fact-sheet/ARA240DNPUFS.pdf) podaje do 40 eTOPS deklarowanych przez dostawcę i wynik 313 obrazów na sekundę dla YOLOv8n. NXP wymienia Ara SDK i eIQ Toolkit na stronie produktu, ale nie dowodzi to wymienności artefaktów z odrębną [ścieżką Neutron dla i.MX 95](https://eiq.nxp.com/learning-hub/convQuant/neutron.html). Aktywny jest [moduł M.2 16 GB](https://www.nxp.com/design/design-center/development-boards-and-designs/ARA2-M2-16G-GT), a opcja USB pozostaje przedprodukcyjna. NXP definiuje „e” w eTOPS jako „equivalent”, czyli „równoważny”, a nie „effective”; nie jest to miara gęstego INT8 TOPS innego dostawcy i nie należy jej porównywać bezpośrednio.

### STMicroelectronics

[Układy STM32N6x7](https://www.st.com/en/microcontrollers-microprocessors/stm32n6-series.html), w tym STM32N657 i STM32N647, wprowadzają akcelerator Neural-ART 600 GOPS do produktów klasy MCU; linia uniwersalna N6x5 nie zawiera tego akceleratora. [STM32Cube AI Studio](https://www.st.com/en/development-tools/stedgeai-cubeai.html) i ST Edge AI Core analizują oraz optymalizują importowane modele, a [STM32 AI Model Zoo Services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) udostępnia procesy wdrażania, optymalizacji, benchmarkowania i przykłady.

Aktualne [omówienie detekcji obiektów](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/object_detection/docs/README_OVERVIEW.md) dokumentuje Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26 i ST-YOLOX oraz usługi klasyfikacji, estymacji pozy i segmentacji. To inna klasa wydajności niż karta PCIe 200 TOPS. Platforma jest interesująca, ponieważ pobieranie obrazu z kamery, inferencja i sterowanie mogą działać w mocno ograniczonym budżecie mocy i pamięci.

### Renesas

Procesory Renesas RZ/V integrują akceleratory DRP-AI. Oprogramowanie rozwijało się od DRP-AI Translator i DRP-AI TVM w kierunku [RUHMI](https://www.renesas.com/en/software-tool/ruhmi-framework), opartego na technologii MERA firmy EdgeCortix. Otwarte [repozytorium RZ/V DRP-AI TVM](https://github.com/renesas-rz/rzv_drp-ai_tvm) i [lista walidacyjna RZ/V2H](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) wymieniają warianty YOLOv5, YOLOv8, YOLO11 i YOLO26 n/s/m, YOLOX, estymację pozy i sieci segmentacji oraz modele klasyfikacji.

Renesas jest wiarygodnym celem dla robotyki i przemysłu, ale dla odtwarzalności trzeba zapisywać konkretną płytkę, generację DRP-AI, wersję translatora i postprocessing po stronie CPU.

## Inteligentne sensory i procesory projektowane z myślą o kamerach

### Sony IMX500

Sony IMX500 nie jest zwykłym procesorem aplikacyjnym. Łączy matrycę obrazową i przetwarzanie AI, dzięki czemu inferencja może odbywać się w sensorze, ograniczając ilość danych obrazu opuszczających kamerę. Raspberry Pi AI Camera sprawia, że ta architektura jest wyjątkowo dostępna.

Oficjalne [repozytorium modeli Raspberry Pi IMX500](https://github.com/raspberrypi/imx500-models) publikuje spakowane modele YOLOv8n, YOLO11n, EfficientDet Lite0, NanoDet+ i SSD MobileNetV2 FPN Lite. [Dokumentacja AI Camera](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) objaśnia konwersję, pakowanie i postprocessing w sensorze. Jednostką wdrożeniową jest pakiet RPK, a nie ogólny plik ONNX; istotnymi ograniczeniami są pamięć sensora i obsługiwane operatory. Strona produktu [Raspberry Pi](https://www.raspberrypi.com/products/ai-camera/) podaje, że kamera pozostanie w produkcji co najmniej do stycznia 2028 r.

### Ambarella

Procesory CVflow firmy Ambarella są mocno ugruntowane w kamerach, dronach i wizji samochodowej. Aktualne rodziny produktów obejmują CV72/CV75 do kamer, CV5/CV52 do systemów wizji o wyższej wydajności i samochodowe układy CV3-AD. Cooper Developer Platform oraz przepływ kompilacji/środowiska uruchomieniowego CVflow tworzą nazwaną publicznie warstwę programową.

[Publiczny katalog modeli dla deweloperów](https://www.ambarella.com/developer/model-garden/) wymienia YOLOX, RTMDet, DeepLabV3+, TopFormer, OWL-ViT i modele multimodalne. Szczegółowa dokumentacja kompilatora i pliki do pobrania pozostają jednak skierowane do partnerów. Integrację Ambarella należy więc rozpocząć od relacji z dostawcą lub OEM, a nie od założenia, że dostępny jest publiczny pakiet pip.

### Synaptics Astra

Synaptics ma trzy odpowiednie cele. Systemy Astra SL1600/SL1680 korzystają z [SyNAP toolkit](https://developer.synaptics.com/docs/synap/introduction), który konwertuje model źródłowy wraz z opisem YAML do pakietu `model.synap`. Nowsze urządzenia SL2611/13/15/17/19 korzystają z [platformy Torq](https://developer.synaptics.com/docs/torq/introduction), opartej na IREE/MLIR i tworzącej `.vmfb`. [Seria SR100](https://developer.synaptics.com/docs/sr/introduction-sr) to odrębna rodzina MCU z Cortex-M55 i Ethos-U55 oraz własnym SDK. Te artefakty i API nie są wymienne.

Oficjalny [samouczek benchmarku YOLO](https://developer.synaptics.com/docs/sl/tutorials/vision/benchmark-yolo) jest wyjątkowo konkretny: eksportuje YOLOv8 do TFLite, wykonuje asymetryczną kalibrację UINT8 w SyNAP, kompiluje model dla SL1680, a następnie uruchamia `synap_cli` i aplikację detekcji obiektów. Potwierdza YOLOv8n/v8s na SL1680 i opisuje obsługę SyNAP do YOLO11. Osobny [przewodnik detekcji obiektów SL261x](https://developer.synaptics.com/docs/sl/sl2600/getting-started/object-detection) uruchamia YOLOv8 z Torq `.vmfb`. Są to oficjalne przykłady dla konkretnych celów, a nie deklaracja, że każdy graf YOLO działa na wszystkich trzech rodzinach.

### MediaTek Genio

MediaTek zasługuje na pełny wpis, ponieważ aktualny [IoT AI Hub](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) udostępnia już macierz sprzętu i oprogramowania, pakiety modeli i tabele benchmarków, które starsze przeglądy rynku trudno było zweryfikować. Genio 360/360P/420/520/720 używa NeuroPilot 8 z MDLA 5.3; Genio 510/700 używa NP6 z MDLA 3.0; Genio 1200 używa NP6 z MDLA 2.0. MediaTek informuje, że generacja NeuroPilot i MDLA, zestaw operatorów, kompilator i środowisko uruchomieniowe są powiązane wersją przez cały cykl życia danego SoC.

Ścieżka analitycznej AI opiera się na TFLite:

```text
PyTorch model
    -> NeuroPilot Converter and representative calibration
    -> quantized .tflite
    -> version-matched ncc-tflite compiler
    -> chip-generation-specific .dla
    -> Neuron Runtime on MDLA
```

Nowsze produkty Genio udostępniają również delegowanie online LiteRT i ścieżkę ONNX Runtime CPU/NPU w obsługiwanych systemach operacyjnych. To inne tryby wykonania niż offline `.dla`. [Przewodnik po architekturze oprogramowania](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/software_architecture.html) i [macierz zasobów](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) wyraźnie rozróżniają te możliwości.

Oficjalna [tabela modeli analitycznych](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical.html) publikuje wpisy benchmarków i przewodniki konwersji YOLOv5s i YOLOv8s dla wielu generacji MDLA. Jawnie nie udostępnia wstępnie przekonwertowanych artefaktów YOLO z powodu ograniczeń AGPL-3.0. Pomiary offline Quant8 przy 640 x 640 obejmują odpowiednio 5.35 ms i 8.04 ms na Genio 720. Strona modelu [YOLOv8s](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical/YOLOv8s.html) publikuje tensory wejścia/wyjścia, czasy zależne od backendu i ostrzeżenia o niestandardowych operacjach MediaTek. Są to benchmarki dostawcy, a nie wyniki dla pełnego potoku kamery.

Ograniczeniem jest dostęp. Publiczna dokumentacja Yocto jest szczegółowa, ale aktualny pakiet NP8 all-in-one dla konwertera/kompilatora i wiele materiałów Android są oznaczone jako zasoby NDA lub dla bezpośrednich klientów. Zwykłe konto deweloperskie nie zapewnia takiego dostępu jak konto klienta MediaTek Online. LibreYOLO powinno więc traktować Genio jako platformę sprawdzoną technicznie, lecz wymagającą partnerstwa przy odtwarzalnej integracji kompilatora dla własnych modeli.

## Ekosystemy Edge AI skoncentrowane na Chinach

Kilka najbardziej wydajnych i przystępnych cenowo platform wizyjnych jest słabo reprezentowanych w anglojęzycznych zestawieniach rynku.

### D-Robotics i platformy BPU wywodzące się z Horizon

D-Robotics utrzymuje ekosystem płytek deweloperskich RDK wokół akceleratorów BPU stosowanych w RDK X3, X5, Ultra i nowszych produktach serii S100. Bieżąca gałąź `rdk_x5` w [RDK Model Zoo](https://github.com/D-Robotics/rdk_model_zoo) dokumentuje ścieżki YOLOv5/v5u, YOLOv8/9/10/11/12/13/26, YOLOE, YOLO-World, detekcję, segmentację, estymację pozy, klasyfikację, OCR i CLIP dla RDK X5. X3 korzysta z osobnej gałęzi, a bieżąca dostawa dla serii S znajduje się w gałęzi [`rdk_s`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s) głównego katalogu; starsze repozytorium [`rdk_model_zoo_s`](https://github.com/D-Robotics/rdk_model_zoo_s) zawiera historyczne demonstracje. Nowa lista X5 nie dowodzi zatem, że każdy model zweryfikowano na X3, Ultra lub S100.

OpenExplorer/Algorithm Toolchain importuje ONNX lub Caffe do PTQ i obsługuje procesy QAT z PyTorch. Wdrożenie zależy od platformy: aktualny RDK X5 używa `.bin` przez `hbm_runtime` nad `libdnn`, a bieżąca podstawowa ścieżka `rdk_s` używa `.hbm` przez API o tej samej nazwie nad `libhbucp`; X3 zachowuje starszą gałąź i interfejsy inferencji. Starsze materiały `rdk_model_zoo_s` opisują też historyczne ścieżki `.bin` i `.hbm`, więc artefakty muszą pozostać powiązane z właściwą gałęzią i toolchainem. Nieobsługiwane operatory mogą być wykonywane na CPU. Publiczne przykłady są użyteczne, ale zgodność wersji RDK OS, firmware'u płytki, toolchainu i pliku modelu pozostaje częścią kontraktu.

### AXERA

Rodziny AX650/AX630/AX620 firmy AXERA pojawiają się w kompaktowych kamerach AI i płytkach, takich jak linia MaixCAM firmy Sipeed. Pulsar2 importuje ONNX, wykonuje kalibrację i analizę precyzji, a następnie tworzy `.axmodel`; AXEngine uruchamia artefakt.

[Oficjalne przykłady AXERA](https://github.com/AXERA-TECH/ax-samples) obejmują YOLOv5/6/7/8/9/10/11/13/26, YOLOX i YOLO-World, a obsługa zadań i układów zależy od platformy. W obsługiwanych celach aktualne przykłady YOLO26 obejmują detekcję, estymację pozy, segmentację i OBB. [Przykłady konwersji Pulsar2](https://pulsar2-docs.readthedocs.io/en/latest/appendix/model_convert_examples.html) pokazują rzeczywistą pracę: dokładne nazwy tensorów, węzły wyjściowe, transformacje układu, archiwa kalibracyjne i ustawienia sprzętu docelowego.

### SOPHGO

TPU-MLIR firmy SOPHGO to jeden z ciekawszych kompilatorów do niezależnej integracji, ponieważ sam kompilator jest open source. Importuje ONNX, PyTorch, TFLite i Caffe, obniża grafy, kwantyzuje je i tworzy `.bmodel` dla układów BM lub `.cvimodel` dla sprzętu CV18xx.

Projekt [TPU-MLIR](https://github.com/sophgo/tpu-mlir) obsługuje procesy FP32, BF16, FP16 i INT8, zależnie od celu. Kolekcja [demonstracji SOPHON](https://github.com/sophgo/sophon-demo/blob/release/README_EN.md) wymienia YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, warianty OBB/segmentacji, SSD, CenterNet, RetinaFace, SAM/SAM2 i OCR. Demonstracja na BM1684X nie potwierdza działania tego samego artefaktu na BM1688 lub CV18xx.

### Huawei Ascend

Linia układów brzegowych Huawei obejmuje krzem Ascend 310/310P/310B w produktach Atlas 200I i 300I. CANN udostępnia kompilator ATC, środowisko uruchomieniowe AscendCL i jądra operatorów zależne od układu; ATC przekształca ONNX i inne reprezentacje źródłowe w model offline `.om`.

Aktualna [dokumentacja Atlas 200I DK A2](https://www.hiascend.com/en/hardware/developer-kit-a2/specification) i przykłady CANN udostępniają ścieżkę YOLOv5 `.om` dla Ascend 310B. Szerszy, starszy [Ascend ModelZoo](https://github.com/Ascend/modelzoo) zawiera YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN, estymację pozy i modele segmentacji, ale nie należy przedstawiać go tak, jakby każdy wpis ponownie zweryfikowano na 310B. CANN, jądra, firmware i SoC muszą do siebie pasować. `.om` dla Ascend310P nie jest ogólnym artefaktem Ascend.

### Cambricon

Akceleratory MLU firmy Cambricon korzystają z Neuware i silnika inferencji MagicMind. Publiczne [repozytorium MagicMind Cloud](https://github.com/Cambricon/magicmind_cloud) jest przypięte do MagicMind 1.7 i macierzy MLU370-X4/S4; to historyczne potwierdzenie zgodności, a nie macierz aktualnego SDK lub walidacji MLU270. Repozytorium dokumentuje ścieżki PyTorch, ONNX, Caffe i Paddle; obsługę TensorFlow usunięto w wersji 1.7, choć zachowały się historyczne przykłady TensorFlow. Aktualny [portal deweloperski serii 3](https://developer.cambricon.com/index/document/index/classid/3.html) wymienia nowsze wydania SDK, więc publicznego repozytorium przykładów i bieżącego stosu komercyjnego nie należy łączyć w jedną wersję.

Oficjalne [repozytorium modeli MagicMind Cloud](https://github.com/Cambricon/magicmind_cloud) publikuje wyjątkowo bezpośrednią macierz MLU370-X4/S4. Wymienia przykłady YOLOv3, Tiny-YOLOv3, YOLOv4, YOLOv5, YOLOv7, YOLOv8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN, DeepLab i UNet, w tym informację, czy istnieją przykłady C++ lub Python. Dowody są mocne; główną barierą pozostaje dostęp do SDK i kontenerów przez kanały Cambricon.

### Canaan/Kendryte

Aktualne K230/K230D firmy Kendryte i starsze układy KPU K210/K510 są ważne w segmencie niedrogich płytek i inteligentnych kamer. Kompilator [nncase](https://github.com/kendryte/nncase) importuje ONNX lub TFLite, używa danych kalibracyjnych do konwersji stałoprzecinkowej, symuluje wynik na hoście i tworzy `.kmodel`.

Oficjalny [przewodnik deweloperski nncase dla K230](https://github.com/kendryte/k230_docs/blob/main/en/01_software/board/ai/K230_nncase_Development_Guide.md) opisuje kompilację, symulację i uruchamianie YOLOv5s. Osobny [katalog demonstracji AI](https://github.com/kendryte/k230_docs/blob/main/en/02_applications/ai_demos/K230_AI_Demo_Introduction.md) zawiera artefakty YOLOv8n do detekcji, segmentacji i estymacji pozy, a aktualny [dziennik zmian CanMV](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md) dodaje zoptymalizowane przykłady klasyfikacji, detekcji, segmentacji, OBB i estymacji pozy w YOLOv8/11/26. Wynik YOLOv5s w arkuszu danych to [opublikowany benchmark](https://github.com/kendryte/k230_docs/blob/main/en/00_hardware/K230_datasheet.md); wersje kompilatora i binarnej wtyczki KPU muszą pasować do SDK płytki.

### Allwinner

Allwinner V853 łączy sprzęt multimedialny przeznaczony do kamer z NPU Vivante o wydajności 1 TOPS. [Oficjalny angielski przewodnik NPU](https://docs.aw-ol.com/v853/en/npu/dev_npu/) opisuje import Acuity/Pegasus, kwantyzację, weryfikację i wdrożenie przez `viplite`. Własne strony modeli Allwinner i [przewodnik YOLOv5](https://docs.aw-ol.com/v853/npu/npu_yolov5/) wymieniają YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet oraz sieci twarzy/osób.

To rzeczywista obsługa wizji komputerowej, ale nadal związana ze starszym toolchainem Tina Linux/Vivante i pobieraniem SDK zależnym od konta. Platforma należy do mapy rynku z widocznym tym ograniczeniem.

## Pomijane platformy koreańskie, tajwańskie i chińskie

Anglojęzyczne listy często przechodzą od Rockchip do NVIDIA, pomijając drugą grupę rzeczywistych, udokumentowanych układów. Niektóre z tych ekosystemów mają szersze aktualne macierze YOLO niż lepiej znane zachodnie startupy.

### Mobilint

Koreański dostawca akceleratorów Mobilint sprzedaje energooszczędny REGULUS i układ ARIES MLA100 o większej przepustowości. SDK qb przyjmuje dane wejściowe z PyTorch, TensorFlow, TFLite, ONNX i Keras, kwantyzuje je do INT8 i tworzy skompilowany artefakt `.mxq`. Publiczna [macierz wizji](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md) wymienia detekcję YOLOv3/v5/v7/v8/v9/v10/11/12/26, a także warianty segmentacji, estymacji pozy i OBB. [Strona ARIES](https://www.mobilint.com/aries/mla100) publikuje nawet wyniki dla YOLO11s i YOLO26m. To mocny dowód integracji, choć dostęp do SDK/sprzętu jest komercyjny.

### Sunplus

Sunplus SP7350/C3V używa stosu NPU wywodzącego się z Vivante, ale pakuje go inaczej niż Allwinner lub starszy Amlogic. Publiczny proces łączy konwersję Acuity, środowisko Docker NPU, środowisko uruchomieniowe SNNF i wynik `.nb`. Oficjalny [przewodnik niestandardowego wdrożenia YOLOv8](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350) opisuje konwersję i wdrożenie, zamiast tylko deklarować obsługę frameworka. Pokrewne IP nie czyni pliku `.nb` przenośnym między SoC opartymi na Vivante.

### ESWIN Computing

Rodzina RISC-V ESWIN obejmuje EIC7700/EIC7700X i dwurdzeniowy EIC7702/EIC7702X; EIC7700 trafia do dostępnych w sprzedaży płytek takich jak Milk-V Megrez. ENNP zawiera EsQuant, aktualny kompilator EsAAC, generowanie danych referencyjnych, symulację i środowisko uruchomieniowe ESSDK, a jego skompilowanym wynikiem jest `.model`; starsze materiały ENNP używały nazwy `ennc-compile`. Aktualna [strona EsAAC](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) dokumentuje wejście ONNX, więc inne frameworki wymagają eksportu lub konwersji do obsługiwanego IR. Hostowane przez Milk-V [omówienie ENNP](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/introduction) i [samouczek YOLOv3](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3) pokazują publiczną ścieżkę od początku do końca, ale nie szeroką aktualną macierz modeli/dokładności.

### Nuvoton M55M1

M55M1 firmy Nuvoton to konstrukcja klasy MCU z Cortex-M55 i Ethos-U55-256, a nie procesor aplikacyjny Linux. NuEdgeWise/NuML i Arm Vela przygotowują skwantyzowane modele TFLite Micro. Publiczne [repozytorium NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise) wymienia YOLOv8-nano, YOLOX-nano, YOLO Fastest v1.1, SSD-MobileNet FPNLite i kilka klasyfikatorów. Te przykłady małych sieci są wiarygodnym dowodem dla klasy urządzeń o ograniczonej pamięci i mocy, a nie potwierdzeniem standardowych wariantów YOLO 640 pikseli.

### Rebellions i FuriosaAI

Linia ATOM firmy Rebellions używa SDK RBLN i skompilowanych artefaktów `.rbln`. Oficjalne [wydanie obsługi](https://docs.rbln.ai/v0.8.2/supports/release_note.html) wymienia YOLOv3 tiny/full/SPP, YOLOv5 n–x, YOLOv6 n–l, warianty YOLOv7 i YOLOv8 n–x. Aktualna [macierz obsługi kart](https://docs.rbln.ai/latest/supports/version_matrix.html) oznacza ATOM CA02 i ATOM+ CA12 jako EoL, a ATOM+ CA22 i ATOM-Max CA25 jako Active. ATOM-Lite CA21 ma stronę produktu, ale brak go w macierzy SDK, więc status toolchainu jest niejasny. Dokumentacja jest publiczna, ale wheel kompilatora wymaga danych logowania Rebellions Portal.

FuriosaAI trzeba rozdzielić według generacji. Warboy to starszy akcelerator wizji z artefaktami INT8 `.enf` i [oficjalnym katalogiem modeli](https://developer.furiosa.ai/furiosa-models/latest/) obejmującym SSD, YOLOv5M/L i YOLOv7-w6-pose. RNGD korzysta z innego [stosu `.fxb`](https://developer.furiosa.ai/latest/en/furiosa_llm/fxb.html), przeznaczonego głównie do obciążeń LLM/VLM. Aktualny [plan rozwoju Furiosa](https://developer.furiosa.ai/latest/en/overview/roadmap.html) odnotowuje ukończenie obsługi wizji YOLOv8m w 2024 Q4, ale bieżące publiczne zestawienia obsługiwanych modeli i wydajności koncentrują się na LLM/VLM i nie przedstawiają szczegółowej macierzy dokładności/wydajności YOLOv8m. To dowód dotyczący wydanej generacji, a nie potwierdzenie zgodności z Warboy.

### Realtek, Telechips, T-Head i SigmaStar

Te cztery platformy są rzeczywiste, ale mają węższe, starsze lub ograniczone ścieżki dla modeli użytkownika:

| Platforma | Odtwarzalne publiczne dowody | Ograniczenie, które należy zachować |
|---|---|---|
| Realtek AmebaPro2 | [SDK Arduino/FreeRTOS](https://github.com/Ameba-AIoT/ameba-arduino-pro2) zawiera modele `.nb` YOLOv3/4/7-tiny, SCRFD i MobileFaceNet | Konwersja własnych modeli wymaga zgłoszenia zainteresowania/kontaktu z Realtek |
| Telechips TOPST TCC7500 | Oficjalny [projekt z 2026 r.](https://docs.topst.ai/blog/31) przekonwertował i wdrożył YOLOv8s | Konwersja UFLD v1/v2 nie powiodła się z powodu nieobsługiwanych warstw; artykuł proponuje wyłącznie obejście przez podział/postprocessing. Pełne zasoby kompilatora/operatorów często wymagają zgody udzielanej mailowo |
| T-Head TH1520 | [Przewodnik aplikacyjny Sipeed](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) uruchamia YOLOv5n/s przez HHB i CSI-NN2/SHL | Publiczny stos istnieje, ale jego utrzymanie i przejrzystość wersji odstają od aktualnych liderów |
| SigmaStar SSU9383CM | Aktualna dokumentacja opisuje API środowiska uruchomieniowego MI_IPU; starsze materiały SGS_IPU dokumentują `.sim` do `sgsimg.img`, a stary postprocessor wymienia SSD i YOLOv1/2/3 | Brak publicznego dowodu, że starsze artefakty kompilatora lub wymienione ścieżki modeli dotyczą SSU9383CM |

### Inteligentna wizja HiSilicon to nie Huawei Ascend

Układy kamerowe HiSilicon Hi3516CV610/DV500, Hi3519DV500 i Hi3403V100 udostępniają proces ATC do `.om` przez środowiska uruchomieniowe NNN/SVP-NNN. Oficjalny [HiSpark model zoo](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) wymienia YOLOv3/4/5/6/7/8/9/10/11, segmentację/estymację pozy YOLO11, OBB/World/segmentację YOLOv8, OCR, głębię i TinySAM w macierzy zależnej od celu, zwłaszcza dla Hi3403 i Hi3591P. Nie dowodzi to działania każdego wpisu na każdym wymienionym układzie do inteligentnej wizji. Część wpisów dotyczy planów rozwoju, więc wydane przykłady i planowaną obsługę należy rozdzielać; repozytorium podaje też, że udostępnione modele ModelZoo są przeznaczone wyłącznie do użytku niekomercyjnego. Wspólne nazwy ATC/`.om` nie potwierdzają zgodności artefaktów ani środowisk uruchomieniowych z linią Ascend opartą na CANN.

## Nowe i wyspecjalizowane firmy akceleratorowe

### MemryX

Moduły MX3 firmy MemryX używają architektury przepływu danych i mogą łączyć wiele układów. [Neural Compiler](https://developer.memryx.com/tools/neural_compiler.html) przyjmuje modele ONNX, TensorFlow Lite, Keras i TensorFlow, a następnie tworzy pakiet DFP obsługiwany przez środowisko akceleratora. [API środowiska uruchomieniowego](https://developer.memryx.com/api/accelerator/accelerator.html) obsługuje potoki wielomodelowe i streaming.

Dokumentacja i przykłady MemryX obejmują zoptymalizowane przetwarzanie wstępne i końcowe YOLO do detekcji, segmentacji i estymacji pozy. [Informacje o wydaniach SDK](https://developer.memryx.com/release_notes.html) jawnie wymieniają obsługę YOLOv10, YOLO11 i YOLO26. Publiczne materiały są technicznie użyteczne, choć nie ma jednej prostej tabeli modeli/dokładności/urządzeń porównywalnej z materiałami Axelera.

### Kneron

Produkty KL520/KL530/KL630/KL720/KL730 firmy Kneron obejmują małe akceleratory USB i wbudowane. Kneron PLUS udostępnia środowisko uruchomieniowe aplikacji, a bieżący Model Toolchain 0.33.1 dokumentuje konwersję, kwantyzację, ewaluację, symulację i kompilację `.nef` dla tych celów. KL830 pojawia się w niektórych [API środowiska uruchomieniowego PLUS](https://doc.kneron.com/docs/plus_c/introduction/run_examples/), ale nie na aktualnej [liście celów kompilatora](https://doc.kneron.com/docs/toolchain/manual_1_overview/); bez potwierdzenia dostawcy nie należy przypisywać ogólnej obsługi kompilacji `.nef` do KL830.

Oficjalny [przykład YOLO](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/) pokazuje proces na Tiny-YOLOv3, a inne materiały Kneron obejmują trenowanie i wdrażanie YOLOv5. Dowody są wiarygodne, lecz węższe niż szerokie macierze współczesnych YOLO u Hailo, Axelera, Rockchip lub DEEPX.

### SiMa.ai

MLSoC firmy SiMa.ai i [produkcyjna platforma Modalix 50 TOPS](https://sima.ai/press-release/sima-ai-next-gen-platform-for-physical-ai-in-production/) korzystają ze środowiska Palette. ModelSDK, kompilator MLA i ModelExecutor obsługują import, kwantyzację, kompilację, profilowanie i wykonanie. Narzędzia obsługują procesy INT8, INT16 i zorientowane na BF16, zależnie od obciążenia i produktu.

Informacje o wydaniach firmy wymieniają zweryfikowane potoki YOLOv7, YOLOv8 do detekcji/estymacji pozy/segmentacji, YOLOX i segmentacji YOLOX, DETR, Mask R-CNN i EfficientDet. Nie należy ukrywać jednego aktualnego ograniczenia: [informacje o Palette SDK 2.1](https://docs.sima.ai/v2.1.2/pages/release_notes/2.1.html) stwierdzają, że QAT nie działa z powodu regresji kwantyzacji Python/PT2E. Udokumentowanym obejściem jest utworzenie oznaczonego ONNX w SDK 2.0 i skompilowanie go w wersji 2.1. Dostęp i zakup są skierowane do przedsiębiorstw.

Udokumentowano też granicę wdrożenia: [proces kompilacji ModelSDK](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) tworzy skompilowany `.tar.gz` i może wygenerować wykonywalny ELF, a [MPK Tool](https://docs.sima.ai/v2.1.1/pages/palette/mpk_tools.html) pakuje wdrażalny plik `.mpk`. Wyniki są powiązane z celem MLSoC/Modalix i wydaniem Palette.

### EdgeCortix

SAKURA-II to reklamowany akcelerator 60 TOPS do wizji i generatywnej AI, kompilowany przez framework MERA. EdgeCortix dostarcza też technologię MERA do nowszego procesu RUHMI firmy Renesas.

Materiały produktu [SAKURA-II](https://www.edgecortix.com/en/edgecortix-sakura-ii-accelerator-brief) potwierdzają sprzęt i kompilator, a [sprzęt M.2/PCIe](https://www.edgecortix.com/en/hardware) jest oferowany w ramach wersji próbnej lub zapytania o zakup. Nie znaleziono wystarczająco precyzyjnej, aktualnej publicznej macierzy zgodności YOLO. Ta luka jest przydatną informacją przy zakupie: przed wyborem sprzętu należy zażądać walidacji modelu.

### BrainChip

Akida firmy BrainChip to neuromorficzny procesor domeny zdarzeń, a nie konwencjonalne NPU gęstych tensorów. Środowisko [MetaTF](https://brainchip.com/metatf-dev-tools/) składa się z instalowanych pakietów Python do tworzenia modeli, kwantyzacji niskobitowej, konwersji, symulacji i wykonywania na sprzęcie. BrainChip sprzedaje obecnie [dostępny koprocesor AKD1500 w formacie M.2](https://brainchip.com/brainchip-akd1500-now-available-in-compact-m-2-form-factor-enabling-fanless-edge-ai-in-industrial-and-commercial-designs/), obok starszego sprzętu AKD1000 i IP Akida 2.

Aktualna [karta modelu Akida 2](https://brainchip.com/wp-content/uploads/2025/04/Akida-2-Model-Card-V1.1-Mar.25.pdf) wymienia detektor AkidaNet0.5 YOLOv2, CenterNet, AkidaUNet i sieci rozpoznawania twarzy. Inne materiały obsługi obejmują FOMO i obciążenia oparte na zdarzeniach. Akida obsługuje wyjątkowo niskobitowe wagi i aktywacje, ale wdrożenie może wymagać konwersji uwzględniającej architekturę zamiast traktowania go jak kolejnego ogólnego celu INT8 ONNX. Wyniku katalogu modeli AKD1000 lub Akida 2 nie należy przypisywać AKD1500, dopóki raport mapowania/dopasowania tego nie potwierdzi.

### Blaize

Blaize sprzedaje produkty Pathfinder i Xplorer oparte na P1600; ścieżkę programową zapewniają Picasso SDK, NetDeploy i AI Studio. [Strony produktów](https://www.blaize.com/products/) jasno wskazują wizję i Edge AI jako rynki docelowe, ale nie ma aktualnej, publicznej macierzy zgodności modeli.

Dlatego ten przewodnik klasyfikuje Blaize jako platformę komercyjną z ograniczonym dostępem, zamiast uzupełniać lukę twierdzeniami społeczności. Warunkiem wstępnym powinien być raport kompilacji i dokładności dostawcy dla planowanego modelu.

## TinyML i wizja endpointów

### Arm Ethos-U i Alif

Arm licencjonuje firmom układy IP NPU Ethos-U55, U65 i U85. Kompilator [Vela](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) przyjmuje skwantyzowane grafy LiteRT/TFLite i przekształca obsługiwane podgrafy w niestandardowe operacje Ethos-U. Nieobsługiwane operacje mogą pozostać na CPU, dlatego twierdzenia „aplikacja działa” i „cała sieć działa na NPU” oznaczają co innego.

Rzeczywiste implementacje obejmują Ethos-U55 w Alif Ensemble E7, Ethos-U65 w NXP i.MX 93 i Ethos-U85 w nowszych systemach. [E7 AI/ML AppKit](https://alifsemi.com/support/kits/ensemble-e7appkit/) firmy Alif dokumentuje dwa akceleratory ML i pokazuje proces TFLite-do-Vela na E7; [Ensemble E8](https://alifsemi.com/ensemble-e8-series/) łączy jeden Ethos-U85 z dwoma NPU Ethos-U55. Alif publikuje też benchmark [detektora twarzy YOLO-Fastest INT8](https://alifsemi.com/faster-ai-mcu-inferencing-low-power-consumption/) na poziomie rodziny, w rozdzielczości 192 na 192, dla Cortex-M55 z Ethos-U55, ale nie szeroką macierz współczesnego YOLO dla konkretnych celów. Te systemy o ograniczonej pamięci nadają się do małych sieci klasyfikacji i detekcji, a nie dowolnych detektorów 640 pikseli tylko dlatego, że da się je wyrazić w TFLite.

### Infineon PSOC Edge, Himax WiseEye2 i Analog Devices MAX7800x

Rodziny [PSOC Edge E83](https://documentation.infineon.com/psocedge/docs/xsk1761304020519) i E84 firmy Infineon łączą Cortex-M55 z Ethos-U55 oraz dodają osobny blok NNLite po stronie energooszczędnego M33. [Podręcznik architektury E84](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) opisuje 8-bitowe wagi, aktywacje 8- lub 16-bitowe i proces Vela, w którym obsługiwane operatory stają się strumieniem poleceń NPU, a nieobsługiwana praca pozostaje na CPU. [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter) przyjmuje modele TFLite oraz ścieżki Keras i PyTorch; materiały architektoniczne podają MobileNetV1/V2 jako przykładowe jądra, a nie benchmarki celów. [Zestaw AI E84](https://documentation.infineon.com/psocedge/docs/cci1762693051052) zawiera kamerę i korzysta z ModusToolbox, ale publiczne materiały nie dostarczają jeszcze macierzy zgodności nazwanych modeli YOLO. To rzeczywista programowalna platforma wizyjna z rozwijającymi się dowodami na poziomie modeli, a nie zweryfikowany cel detekcji ogólnego przeznaczenia.

[HX6538 WiseEye2](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) firmy Himax również łączy Cortex-M55 i Ethos-U55 dla wizji stale gotowej do działania. Publiczny zestaw oprogramowania jest wyjątkowo konkretny w tej klasie poboru mocy: oficjalne [przykłady Grove Vision AI Module V2](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) obejmują detekcję obiektów YOLOv8n, estymację pozy i klasyfikację płci, detekcję YOLO11n, siatkę twarzy i PeopleNet. Wspólna ścieżka używa TFLite Micro i Vela, a następnie w razie potrzeby przechodzi na CMSIS-NN i jądra referencyjne. Są to celowo małe modele i rozdzielczości, a nie dowód, że mieści się tam standardowy graf YOLO.

Analog Devices stosuje inną metodę w akceleratorach CNN [MAX78000/MAX78002](https://www.analog.com/en/products/max78002.html). Publiczne narzędzie [`ai8x-synthesis`](https://github.com/analogdevicesinc/ai8x-synthesis) kwantyzuje wytrenowaną sieć i generuje kod C zależny od celu, wagi i konfigurację akceleratora zamiast przenośnego pakietu środowiska ONNX. Dowody pierwszej strony obejmują [identyfikację twarzy](https://www.analog.com/en/resources/app-notes/an-2616.html), wykrywanie kodów QR TinierSSD w [katalogu modeli ADI](https://ez.analog.com/dsp/software-and-development-tools/edgebench/a/docs-faqs/DF912/available-models-in-adi-model-zoo) i klasyfikatory obrazów. Urządzenia są istotne dla wizji endpointów, ale nie znaleziono aktualnej oficjalnej macierzy współczesnych modeli YOLO.

### GreenWaves GAP9

GAP9 łączy obliczenia RISC-V z silnikiem neuronowym NE16. NNTool importuje i kwantyzuje sieci, a AutoTiler i GAP SDK generują wdrażalny kod. Publiczne [menu sieci neuronowych GAP9](https://github.com/GreenWaves-Technologies/nn_menu_gap9) zawiera aplikacje MobileNet, EfficientNet, ResNet, MobileNet SSD oraz wykrywania i rozpoznawania twarzy. GreenWaves publikuje też [projekt wykrywania osób YOLOX](https://github.com/GreenWaves-Technologies/yolox_people_detection) z wynikami dokładności i symulacji.

To wyjątkowo przejrzyste dowody TinyML, ale pełny SDK jest dostępny tylko dla zakwalifikowanych klientów, a modele są znacznie mniejsze niż popularne warianty YOLO 640 na 640.

### Lattice sensAI

Lattice sensAI to oparta na FPGA alternatywa dla stałego NPU. sensAI Studio i Neural Network Compiler mapują obsługiwany graf na konfigurowalne IP akceleratora dla układów ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E i Avant-X. Wynik jest związany z wybranym FPGA, trybem akceleratora, układem pamięci i bitstreamem, a nie przenośnym modelem środowiska uruchomieniowego.

Aktualny [podręcznik Neural Network Compiler](https://www.latticesemi.com/-/media/LatticeSemi/Documents/UserManuals/MQ3/FPGA-UG-02052-8-0-Lattice-Neural-Network-Compiler-Software.ashx?document_id=52343) zawiera wyjątkowo bezpośrednią tabelę topologii. Wymienia YOLOv1 dla ECP5, YOLOv5 i YOLOv8 dla zoptymalizowanych/zaawansowanych trybów CrossLink-NX i CertusPro-NX, a YOLO11 wyłącznie z IP w trybie Advanced. Wymienia też SSD, MobileNetV2-SSD, ResNet, ENet, SqueezeDet i inne sieci wizji, pokazując nieobsługiwane komórki dla rodzin FPGA. [Advanced CNN Accelerator](https://www.latticesemi.com/en/Products/DesignSoftwareAndIP/IntellectualProperty/IPCore/IPCores05/Advanced-CNN-Accelerator-IP) jest przeznaczony dla Avant-E/Avant-X/CertusPro-NX i stanowi komercyjne IP. To dowód z tabeli kompilatora, a nie obietnica, że jeden bitstream obsłuży równocześnie każdą wymienioną topologię.

### Microchip VectorBlox

Publiczny [VectorBlox SDK 3.1](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) przygotowuje i kompiluje w pełni skwantyzowane sieci TFLite INT8 dla konfigurowalnego IP CoreVectorBlox w układach FPGA PolarFire SoC. `tflite_preprocess` i `vnnx_compile` tworzą zasoby wdrożeniowe `.vnnx`, `.hex` i `.ucomp`; repozytorium zawiera też symulator i sterownik C po stronie celu. Grafy TensorFlow, ONNX i OpenVINO nadal muszą przejść przez udokumentowany proces przygotowania TFLite/INT8.

Aktualna [macierz samouczków](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md) obejmuje YOLOv5n, YOLOv8n do detekcji, klasyfikacji, OBB, estymacji pozy i segmentacji, YOLOv9t, rzadkie/skompresowane warianty YOLO, MobileNet, EfficientNet-Lite0, ResNet18, MiDaS, FFNet i QuickSRNet. Osobny [przewodnik postprocessingu C](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/docs/C_Postprocessing.md) wymienia YOLOv2/3/4/5 oraz dekodery detekcji, estymacji pozy i OBB Ultralytics. SDK 3.1 obsługuje obecnie PolarFire SoC Video Kit, a jawnie nie obsługuje PolarFire Video Kit bez SoC; starszych demonstracji płytek nie należy mylić z aktualnym kontraktem celu.

SDK jest publicznie dostępny na [licencji właściwej dla Microchip](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/LICENSE.md), która ogranicza oprogramowanie i utwory pochodne do produktów Microchip. Microchip udostępnia bezpłatne, ale wymagające wniosku licencje Libero Silver i CoreVectorBlox. Wdrożenie nadal stanowi kontrakt całego systemu FPGA: konfiguracja akceleratora, bitstream, firmware, dane sieci wygenerowane przez SDK i układ pamięci muszą być zgodne. Udana kompilacja VectorBlox nie zmienia modelu w plik binarny, który można skopiować na dowolny projekt PolarFire.

### Syntiant

NDP200 firmy Syntiant jest przeznaczony do inferencji wizji, audio i sensorów stale gotowej do działania, w budżecie mocy niższym niż zwykłe SoC Linux. Aktualna [tabela sprzętu](https://www.syntiant.com/hardware) wymienia NDP200 jako produkowany seryjnie, a NDP250 jako próbkowany. [Opis produktu NDP200](https://www.syntiant.com/ndp200/) dokumentuje obsługę sieci CNN, RNN i w pełni połączonych przy poborze poniżej 1 mW; [ogłoszenie NDP250](https://www.syntiant.com/news/syntiant-unveils-ndp250-neural-decision-processor-with-next-gen-core-3-architecture) mówi natomiast o rozpoznawaniu obrazu stale gotowym do pracy przy poborze poniżej 30 mW. „Poniżej miliwata” nie powinno być uogólniane na oba układy.

Publiczne materiały nie potwierdzają szerokiej zgodności z YOLO, dlatego produkt należy oceniać pod kątem małych klasyfikatorów i detektorów stale gotowych do działania, korzystając z oceny modelu wspieranej przez dostawcę, a nie zakładając ogólną możliwość eksportu ONNX.

### Google Coral: dwie niezwiązane ze sobą generacje

Starsze produkty Edge TPU używają Edge TPU Compiler, `libedgetpu` i PyCoral z w pełni kwantyzowanymi modelami TFLite INT8. Główne repozytoria [Edge TPU](https://github.com/google-coral/edgetpu) i [PyCoral](https://github.com/google-coral/pycoral) są zarchiwizowane. To realne ryzyko utrzymania oprogramowania, ale nie formalne powiadomienie o końcu życia sprzętu.

Google utrzymuje też osobne, aktywne, otwarte [Coral NPU](https://github.com/google-coral/coralnpu): IP akceleratora RISC-V do integracji z energooszczędnymi SoC. Publiczny projekt udostępnia obecnie RTL, symulację sprzętu, toolchain i przykłady ELF. Nie jest to bezpośredni następca Edge TPU w formacie USB/M.2 i nie publikuje aktualnej macierzy wdrożeń YOLO.

## Pułapki przy GPU, NPU klienckich i obliczeniach adaptacyjnych

NVIDIA, AMD, Intel i Apple często pojawiają się w wyszukiwaniach dotyczących NPU, ale nie udostępniają jednej, wymiennej klasy akceleratora.

**NVIDIA Jetson:** Jetson Orin łączy GPU CUDA z dedykowanymi rdzeniami DLA. TensorRT może zbudować serializowany silnik dla DLA, ale nieobsługiwane warstwy działają na GPU tylko wtedy, gdy jawnie włączono GPU fallback. Aktualna [tabela YOLO DeepStream](https://github.com/NVIDIA-AI-IOT/deepstream_tools/blob/main/yolo_deepstream/README.md) obejmuje YOLOv4/v7/v8/v9/11, ale większość wpisów dotyczy GPU; jedynym wymienionym jawnie celem ONNX DLA jest YOLOv8s INT8. Ograniczenia operatorów opisano w [dokumentacji limitów DLA NVIDIA](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html). Thor to odrębny przypadek: [przewodnik migracji DriveOS NVIDIA](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/NVIDIA_DriveOS_7.0.3_Migration_Guide.pdf) podaje, że z Thor usunięto rdzenie DLA na rzecz elastycznego planowania na GPU. Informacja „działa na Jetsonie” nie wskazuje więc użytego akceleratora.

**AMD Vitis AI:** Starsza generacja Kria/DPU kompilowała grafy `.xmodel` wykonywane przez VART. Aktualny [Vitis AI 6.2](https://vitisai.docs.amd.com/en/6.2/) ma status GA dla dwóch odrębnych ścieżek Versal AI Edge: [Gen1](https://vitisai.docs.amd.com/projects/gen1/en/latest/index.html) na VEK280/VE2802 korzysta z ONNX, AMD Quark i przepływu migawek/podgrafów związanych z celem; [Gen2](https://vitisai.docs.amd.com/projects/gen2/en/latest/index.html) na VEK385 ma własny kontrakt kompilacji i środowiska uruchomieniowego. Gen1 publikuje przykłady YOLOv5, YOLOv7, YOLOv8 i YOLOX. Ryzen AI to czwarty, osobny stos NPU klienckiego. Nie należy przenosić artefaktów ani walidacji między starszym DPU, Versal Gen1, Versal Gen2 i Ryzen AI tylko dlatego, że wszystkie używają nazwy Vitis lub AMD.

**Intel OpenVINO:** OpenVINO może kierować wykonanie na CPU, GPU i NPU Core Ultra przez jedno API. [Dokumentacja wtyczki NPU Intel](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) wskazuje obsługiwane generacje NPU, a [macierz zweryfikowanych modeli](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html) pokazuje kolumny zależne od urządzenia. Notebook YOLO potwierdza przepis aplikacyjny, a nie walidację NPU, chyba że potwierdza ją macierz. Obsługiwane operacje, kształty, precyzja i zainstalowany sterownik NPU nadal decydują o tym, czy cały graf zostanie tam wykonany. OpenVINO IR (`.xml` plus `.bin`) jest wielokrotnego użytku na poziomie źródłowym; cache skompilowanego modelu zależy od urządzenia i oprogramowania.

**Apple Core ML:** `coremltools` konwertuje modele do `.mlpackage`, które Xcode kompiluje do `.mlmodelc`. Core ML może planować pracę na CPU, GPU i Apple Neural Engine, ale celowo ukrywa dokładny podział. Apple jest więc świetną platformą wdrożeniową, lecz nie uzasadnia twierdzenia, że „cały graf YOLO działał na NPU”, jeśli nie potwierdzają tego dane profilowania.

## Firmy dostarczające IP NPU dla producentów układów

Niektóre firmy w ogóle nie sprzedają płytek ani gotowych akceleratorów. Licencjonują projekty NPU dostawcom SoC. Są istotne, bo ta sama architektura może pojawić się pod kilkoma markami układów, ale ostateczny SDK i artefakt mogą być zmodyfikowane przez licencjobiorcę.

| Dostawca IP | Rodzina NPU i oprogramowanie | Znaczenie |
|---|---|---|
| [Arm](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) | Ethos-U55/U65/U85 i Vela | Występuje w NXP, Alif, Infineon, Himax i innych produktach wbudowanych; przepływ z kwantyzowanym TFLite i TOSA |
| [Arm China](https://www.armchina.com/mountain?infoId=161&name=) | AIPU Zhouyi i Compass SDK | Publiczny [katalog modeli](https://github.com/Arm-China/Model_zoo) wymienia YOLOv1-tiny/v2/v3/v4/v5, YOLOX i YOLOv8-seg; to modele referencyjne, a nie dowód dla każdego układu licencjobiorcy |
| [VeriSilicon](https://www.verisilicon.com/en/IPPortfolio/VivanteVIP9000Pico) | Rodzina Vivante VIP i Acuity SDK | Pokrewne rodziny NPU Vivante występują w wielu SoC, w tym w osobnych ścieżkach A311D i i.MX 8M Plus; każdy dostawca układu nadal zapewnia własną integrację |
| [Imagination Technologies](https://www.imaginationtech.com/products/open-access/) | PowerVR Series3NX AX3146/AX3386/AX3596; historyczne IMG Series4; Neural Compute SDK/IMG DNN | Licencjonowane IP NNA, a nie płytka detaliczna; Open Access oferuje konfiguracje Series3NX 1/5/10 TOPS, natomiast oficjalny program NC-SDK wymienia PP-YOLOE, EfficientNet i HRNet, ale nie każdą konfigurację IP |
| [CEVA](https://www.ceva-ip.com/product/ceva-neupro-studio/) | NeuPro-Nano/NeuPro-M i NeuPro Studio | Licencjonowane IP NPU z importem, kwantyzacją, kompresją, kompilacją grafu, symulacją i generowaniem C/C++ |
| [MIPS](https://mips.com/processor-solutions/arc-npx-family/) | ARC NPX6 i [MetaWare MX](https://mips.com/processor-solutions/arc-metaware-mx/), dawniej Synopsys ARC | Skalowalne IP NPU i SDK sieci neuronowych przejęte przez GlobalFoundries i włączone do portfolio MIPS w [czerwcu 2026 r.](https://mips.com/press-releases/gfmipsarcclose/); samo w sobie nie jest detalicznym celem wdrożenia |
| [Cadence](https://www.cadence.com/en_US/home/tools/silicon-solutions/ai-ip-platform/neuroweave-sdk.html) | Neo NPU, DSP Tensilica i NeuroWeave SDK | Stos kompilatora/interpretera używany przez licencjobiorców SoC, z obsługą sieci i kwantyzacji zależną od licencjonowanej konfiguracji |
| [Quadric](https://quadric.ai/npu-ip) | IP GPNPU Chimera i SDK | Programowalne IP w stylu NPU/DSP licencjonowane innym firmom; ostatecznym celem jest krzem licencjobiorcy |
| [Expedera](https://www.expedera.com/products-overview/) | IP NPU Origin i stos programowy | Licencjonowana architektura NPU dla urządzeń brzegowych, a nie bezpośrednio dostępna płytka LibreYOLO |

Imagination pokazuje, dlaczego obok nazw architektur trzeba opisywać dostęp i cykl życia. Program [Open Access](https://www.imaginationtech.com/products/open-access/) oferuje trzy sprawdzone na krzemie konfiguracje Series3NX bez początkowej opłaty licencyjnej IP, ale dopiero po ocenie firmy oraz z płatnym wsparciem/utrzymaniem i tantiemami produkcyjnymi. [Neural Compute SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) zapewnia narzędzia kompilacji, optymalizacji, kwantyzacji i środowiska uruchomieniowego; oficjalna [współpraca z Baidu PaddlePaddle](https://www.imaginationtech.com/news/imagination-and-baidu-paddlepaddle-create-open-source-machine-learning-library-for-model-zoo/) wymienia detekcję PP-YOLOE, klasyfikację EfficientNet i segmentację HRNet. Przykłady te nie weryfikują każdej konfiguracji Series3NX lub Series4, a aktualne indywidualne [strony produktów Series4](https://www.imaginationtech.com/product/img-4nx-mc1/) oznaczają je jako niedostępne, więc należy traktować Series4 jako rozwiązania historyczne, chyba że dział sprzedaży potwierdzi inaczej.

Ta warstwa wyjaśnia pozorne podobieństwa rodzin. Nie zapewnia przenośności artefaktów. Dwa SoC zawierające pokrewne IP Vivante lub Ethos mogą mieć inne mapy pamięci, sterowniki, wersje kompilatora, zestawy operatorów i środowiska uruchomieniowe dostawców.

## Co faktycznie się wdraża: tabela zależności artefaktów

Na końcu procesu kompilacji kończy się pozorna przenośność ONNX. Nazwy i rozszerzenia zmieniają się z czasem, ale praktyczna zasada pozostaje taka sama: należy zachować oryginalny model i dane kalibracyjne, ponieważ natywnego artefaktu zazwyczaj nie da się przenieść na inną generację NPU ani niezawodnie odtworzyć innym wydaniem SDK.

| Stos dostawcy | Typowe wejście kompilatora | Co trafia na urządzenie | Od czego zależy |
|---|---|---|---|
| [Amlogic AMLNN](https://github.com/Amlogic-NN/amlnn-toolkit) | ONNX, TFLite, TorchScript lub PT2 | `.adla`; starszy A311D używa `.nb` | ID celu, generacja ADLA, kompilator AMLNN, NNSDK2/runtime, sterownik ADLA i BSP |
| [Rockchip RKNN](https://github.com/airockchip/rknn-toolkit2) | ONNX, TFLite lub eksport frameworka | `.rknn` | Wersja RKNN toolkit/runtime i rodzina NPU Rockchip |
| [Hailo](https://hailo.ai/developer-zone/documentation/) | ONNX lub TensorFlow przez format pośredni HAR | `.hef` | Architektura Hailo i generacja kompilatora/środowiska uruchomieniowego |
| [Axelera Voyager](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) | ONNX lub obsługiwana definicja katalogu modeli | `.axm` w Pipeline Builder alpha, pakiet potoku `.axe`; klasyczny `.axmodel` z manifestem | Generacja API Voyager i konfiguracja celu Metis |
| [DEEPX DXNN](https://github.com/DEEPX-AI/dx-all-suite) | ONNX i obsługiwane ścieżki frameworków | `.dxnn` | Wersja DX-COM/DX-RT i cel DX-M |
| [Qualcomm QAIRT/QNN lub SNPE](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | ONNX, TFLite lub model frameworka | Biblioteka/kontekst QNN lub plik SNPE `.dlc` | Architektura HTP, SoC, backend i wersja środowiska uruchomieniowego |
| [MediaTek NeuroPilot](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) | Przekonwertowany/skwantyzowany TFLite | `.dla` dla offline Neuron Runtime; `.tflite` dla trybu delegowanego | Generacja NP/MDLA, obraz systemu operacyjnego, kompilator i runtime |
| [TI TIDL](https://github.com/TexasInstruments/edgeai-tidl-tools) | ONNX lub TFLite | Katalog zaimportowanych artefaktów i pliki modelu aplikacji | Narzędzia/środowisko uruchomieniowe TIDL i generacja procesora |
| [NXP eIQ](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | Zwykle TFLite lub ONNX | Wynik zależny od backendu: Vela, TIM-VX, Neutron lub Ara | Wybrany akcelerator i.MX/Ara i BSP, a nie ogólnie „NXP” |
| [ST Edge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) | TFLite, ONNX lub obsługiwany model frameworka | Wygenerowana biblioteka/sieć, kod i zasoby firmware | Cel MCU/NPU, konfiguracja pamięci i wersja narzędzia |
| [Infineon PSOC Edge](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) | Całkowitoliczbowy TFLite/Keras/ścieżka PyTorch | TFLite zoptymalizowany przez Vela, ze strumieniem poleceń Ethos-U i firmware aplikacji | Wariant E83/E84, konfiguracja Ethos-U, Vela, ModusToolbox i BSP; NNLite jest osobnym celem |
| [Renesas DRP-AI](https://github.com/renesas-rz/rzv_drp-ai_tvm) | ONNX/TFLite przez TVM lub Translator | Katalog modelu środowiska uruchomieniowego z wieloma plikami danych binarnych | Urządzenie RZ/V i generacja translatora/runtime |
| [Sony IMX500](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) | Sieć skwantyzowana/spakowana w procesie Edge-MDT | Pakiet `.rpk` | Firmware sensora, budżet pamięci i wersja packera |
| [Synaptics](https://developer.synaptics.com/) | TFLite/ONNX i obsługiwane formaty importu | `.synap` w SyNAP; `.vmfb` w Torq | Generacja oprogramowania/sprzętu SL16xx lub SL261x |
| [Ambarella CVflow](https://www.ambarella.com/developer/) | Wejście z toolchainu dla partnerów | Pakiet wdrożeniowy dla partnera; kontrakt artefaktu nie jest publicznie udokumentowany | Należy potwierdzić generację CVflow, SDK/BSP i potok kamery przez Cooper Developer Zone |
| [D-Robotics BPU](https://github.com/D-Robotics/rdk_model_zoo) | ONNX i graf obsługiwany przez toolchain | X5 `.bin`; aktualny `rdk_s` `.hbm`; X3 używa starszego procesu platformy | Generacja BPU, gałąź repozytorium i zgodne wydanie OpenExplorer/runtime |
| [AXERA Pulsar2](https://github.com/AXERA-TECH/ax-samples) | ONNX | `.axmodel` | Cel układu AX, wersje Pulsar2 i AXEngine |
| [SOPHGO TPU-MLIR](https://github.com/sophgo/tpu-mlir) | ONNX, TFLite, TorchScript i inne | `.bmodel` na BM, `.cvimodel` na CV18xx | Cel układu BM/CV i generacja środowiska SOPHON |
| [Huawei Ascend CANN](https://www.hiascend.com/en/software/cann) | ONNX lub obsługiwany graf frameworka | `.om` | Układ Ascend, CANN/ATC i firmware/sterownik urządzenia |
| [Cambricon MagicMind](https://github.com/Cambricon/magicmind_cloud) | Graf frameworka lub model wymiany | Serializowany silnik/model MagicMind | Architektura MLU i wersje Neuware/MagicMind |
| [Canaan nncase](https://github.com/kendryte/nncase) | ONNX lub TFLite | `.kmodel` | Generacja KPU i zgodna wtyczka celu nncase |
| [Mobilint qb](https://www.mobilint.com/sdk-qb) | ONNX i obsługiwane modele frameworków | `.mxq` | Cel REGULUS/ARIES oraz kompilator/runtime qb |
| [Rebellions RBLN](https://docs.rbln.ai/latest/index.html) | PyTorch 2 lub graf obsługiwany przez TensorFlow | `.rbln` | Generacja ATOM i kompilator/runtime RBLN |
| [FuriosaAI](https://developer.furiosa.ai/docs/latest/en/) | ONNX/TFLite i obsługiwana ścieżka frameworka | Warboy `.enf`; RNGD `.fxb` | Całkowicie różne generacje akceleratora i SDK |
| [Sunplus SNNF](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2004353133) | Sieć obsługiwana przez Acuity | `.nb` | Sterownik NPU SP7350, runtime i BSP |
| [ESWIN ENNP](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) | ONNX w udokumentowanym bieżącym procesie EsAAC; inne frameworki wymagają eksportu/konwersji | `.model` | Konkretny cel rodziny EIC7700 i wydanie ENNP/ESSDK |
| [Nuvoton / Ethos-U](https://github.com/OpenNuvoton/NuEdgeWise) | Skwantyzowany TFLite | TFLite zoptymalizowany przez Vela z własnymi operacjami Ethos-U | Plan pamięci M55M1, Vela i firmware |
| [Himax WiseEye2](https://github.com/HimaxWiseEyePlus/YOLOv8_on_WE2) | W pełni całkowitoliczbowy TFLite | `.tflite` zoptymalizowany przez Vela w pamięci modeli wraz z firmware płytki, np. `output.img` | Konfiguracja HX6538/WE2, Vela, TFLite Micro, sterownik Ethos-U i jądra zastępcze |
| [Analog Devices AI8X](https://github.com/analogdevicesinc/ai8x-synthesis) | Checkpoint PyTorch, YAML sieci i próbka wejściowa | Wygenerowany kod C/nagłówki, wagi i firmware skompilowane dla konkretnego urządzenia | Limity pamięci/warstw MAX78000 kontra MAX78002, narzędzie syntezy i wydanie MSDK |
| [Realtek AmebaPro2](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | Wejście/usługa konwersji dostawcy | `.nb` | Firmware RTL8735B i wydanie VoE/NeuralNetwork API |
| [Telechips Enlight](https://docs.topst.ai/product/p/ai) | Ścieżka Darknet, TensorFlow, ONNX lub PyTorch | Pośredni `.enlight` i skompilowany pakiet wdrożeniowy | NPU TCC7500 i wersje TC-NN/Enlight |
| [T-Head HHB](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | ONNX i obsługiwany graf frameworka | `hhb.bm`, wygenerowane parametry/kod i plik wykonywalny | Stos CSI-NN2/SHL TH1520 |
| [SigmaStar MI_IPU](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | Aktualne dokumenty SSU9383CM opisują runtime MI_IPU; starsze materiały używają wejść z rodzin TensorFlow/Caffe | Starsze `.sim` konwertowane do `sgsimg.img`; aktualny publiczny artefakt niezweryfikowany | Konkretny IPU SigmaStar i generacja SDK; zgodność starszego kompilatora z SSU9383CM niepotwierdzona |
| [HiSilicon smart vision](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) | ONNX i graf obsługiwany przez ATC | `.om` | Konkretny SoC do inteligentnej wizji i stos NNN/SVP-NNN; brak ogólnej przenośności Ascend |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | ONNX, TFLite, Keras lub TensorFlow | Pakiet przepływu danych `.dfp` | Konfiguracja sprzętu MX i wydanie kompilatora/runtime |
| [Kneron](https://doc.kneron.com/docs/) | ONNX i konfiguracja toolchainu | `.nef`; KL730 NEFv2 może zawierać `.kne` | Udokumentowany cel kompilatora, generacja układu, firmware i środowisko PLUS; kompilacja dla KL830 nie jest potwierdzona w Toolchain 0.33.1 |
| [SiMa.ai Palette](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) | Obsługiwany framework lub model ONNX | Skompilowane ModelSDK `.tar.gz`, udokumentowany plik wykonywalny ELF lub pakiet MPK Tool `.mpk` | Cel MLSoC/Modalix i wydanie Palette |
| [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/) | ONNX lub definicja sieci | Serializowany silnik, zwykle `.engine` lub `.plan` | Architektura GPU/DLA, TensorRT, CUDA i często konkretne urządzenie |
| [AMD Vitis AI](https://vitisai.docs.amd.com/en/6.2/) | Skwantyzowany graf ONNX/frameworka | Starszy `.xmodel`; podgrafy/migawka NPU Gen1; katalog cache kompilacji Gen2 lub produkcyjny pakiet `.rai` | Dokładna generacja NPU/konfiguracja IP, obraz platformy oraz macierz Vitis/Vivado/PetaLinux |
| [Microchip VectorBlox](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) | W pełni skwantyzowany TFLite INT8 | `.vnnx`, `.hex` i `.ucomp` wraz ze zgodnym firmware/projektem FPGA | Konfiguracja CoreVectorBlox, PolarFire SoC Video Kit, SDK 3.1, bitstream i układ pamięci |
| [Intel OpenVINO](https://docs.openvino.ai/) | Model frameworka lub ONNX | IR `.xml` z `.bin` albo cache skompilowany dla urządzenia | IR można ponownie wykorzystać; cache zależy od urządzenia, sterownika i OpenVINO |
| [Starsze Google Edge TPU](https://coral.ai/docs/edgetpu/models-intro/) | W pełni INT8 TFLite | Skompilowany dla Edge TPU `_edgetpu.tflite` | Kompilator/środowisko Edge TPU i obsługiwane operatory kwantyzacji; niezwiązane z nowym IP Coral NPU |
| [Google Coral NPU IP](https://github.com/google-coral/coralnpu) | Przykłady C/C++ dla aktualnego publicznego projektu IP | Przykłady ELF do symulacji RTL/sprzętu i przyszłej integracji z SoC; brak publicznego artefaktu kompilatora YOLO | Dokładna licencjonowana/zintegrowana implementacja SoC i rozwijający się toolchain open source |
| [Lattice sensAI](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | Opis obsługiwanej sieci | Bitstream FPGA, konfiguracja akceleratora i wagi | Rodzina FPGA, tryb IP sensAI i implementacja pamięci |
| [Imagination NC-SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) | Caffe, TensorFlow lub ONNX | Nazwa artefaktu skompilowanego dla klienta nie jest publicznie podana | Licencjonowana konfiguracja NNA, integracja SoC, NC-SDK/DDK i runtime licencjobiorcy |

Ta tabela to więcej niż ściągawka z rozszerzeń plików. Wersja kompilatora, identyfikator celu, sterownik, firmware i BSP należą do odtwarzalnej tożsamości modelu. Przyszły eksporter LibreYOLO powinien zapisywać je w czytelnym maszynowo manifeście obok każdego artefaktu.

## Dostęp do narzędzi i sygnały dotyczące cyklu życia

Dostępność sprzętu i dostęp do kompilatora to niezależne kwestie. Płytkę może być łatwo kupić, choć aktualny kompilator wymaga umowy z klientem; publiczne SDK może zaś dotyczyć trudno dostępnego krzemu. Poniższe informacje to konkretne sygnały ze źródeł dostawców, a nie uniwersalny ranking trwałości produktów.

| Platforma | Udokumentowany sygnał | Znaczenie w praktyce |
|---|---|---|
| Amlogic ADLA | Publiczne repozytoria Apache-2.0 i pakiety wheel do pobrania; zgodne sterowniki/BSP płytek mogą nadal wymagać Amlogic lub producenta płytki | Łatwa ewaluacja kompilatora, ale należy potwierdzić prawa do redystrybucji plików binarnych i pełną macierz zgodności |
| MediaTek Genio | Publiczny IoT AI Hub i przewodniki Yocto; narzędzia NP8 all-in-one i dokumenty Android oznaczono jako dostępne bezpośrednio dla klienta/NDA | Dowody dotyczące modeli można audytować, ale automatyzacja dla własnych modeli może wymagać partnerstwa |
| Hailo | Publiczny katalog modeli i runtime; Dataflow Compiler przez Developer Zone; publiczne `hailo-camera-apps` zarchiwizowano 3 maja 2026 r. | Szerokie wyszukiwanie modeli jest dostępne, lecz odtwarzalna kompilacja wymaga właściwego konta/wersji; archiwizacja nie dowodzi końca życia Hailo-15, a aktualny pakiet aplikacyjny procesora wizji należy potwierdzić |
| Axelera AI | Publiczna dokumentacja i [publiczny Voyager SDK](https://github.com/axelera-ai-hub/voyager-sdk); wsparcie klienta wymaga konta | Rzadki przypadek akceleratora z SDK do samodzielnego użycia, przy komercyjnym wsparciu i zakupie produktów |
| Qualcomm Dragonwing | [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program) wymienia IQ-9075 jako Sampling z trwałością do 2038 r. i QCS6490 do lipca 2036 r., a strona produktu IQ-9075 podaje Active | Mocny sygnał dla planowania przemysłowego, ale sprzeczny status w źródłach własnych wymaga potwierdzenia SKU i nie gwarantuje stabilności ABI jednego SDK AI przez cały okres |
| Rebellions | Aktualna [macierz obsługi](https://docs.rbln.ai/latest/supports/version_matrix.html) oznacza CA02 i CA12 jako EoL, CA22/CA25 jako Active; brak CA21 | Zakup i zgodność SDK zależą od karty, nawet w rodzinie ATOM |
| NVIDIA Jetson | [Oficjalna tabela cyklu życia modułów](https://developer.nvidia.com/embedded/lifecycle) podaje dostępność modułów Orin do stycznia 2032 r., a AGX Orin Industrial do lipca 2033 r.; zestawy deweloperskie nie mają zobowiązania dotyczącego cyklu życia | Systemy produkcyjne należy projektować wokół modułów, a nie dostępności zestawów deweloperskich |
| Sony IMX500 w Raspberry Pi | [Opis produktu AI Camera](https://datasheets.raspberrypi.com/camera/ai-camera-product-brief.pdf) podaje produkcję co najmniej do stycznia 2028 r. | Konkretne minimum dla tego modułu kamery, a nie każdego produktu IMX500 |
| Amlogic A311Y3 | [Strona premiery z 2026 r.](https://www.amlogic.com/News/index248.html) reklamuje co najmniej dziesięcioletnią gwarancję długowieczności produktu | Obiecujący sygnał dla nowej platformy, który nadal wymaga szczegółów umowy i SKU dla programu produktowego |
| Google Coral | Starsze repozytoria Edge TPU i PyCoral są zarchiwizowane/tylko do odczytu; osobne repozytorium [Coral NPU IP](https://github.com/google-coral/coralnpu) jest aktywne | Ryzyko utrzymania starszego oprogramowania, ale samo w sobie nie stanowi formalnego powiadomienia o końcu życia sprzętu; nie mówi nic o cyklu życia niezwiązanego z nim IP open source |

## Dlaczego TOPS nie jest przewodnikiem zakupowym

Szczytowe TOPS mogą być przydatne w ramach jednego dostawcy i architektury, ale porównania między dostawcami zwykle są nieważne, jeśli nie zgadzają się wszystkie poniższe warunki:

- Precyzja numeryczna, w tym INT8, INT4, FP16 lub tryb mieszany
- Czy jedno multiply-accumulate liczy się jako jedna, czy dwie operacje
- Obliczenia gęste czy strukturalnie rzadkie
- Rozdzielczość wejściowa, rozmiar batcha i graf modelu
- Dokładność po kwantyzacji
- Opóźnienie samego NPU czy całego potoku aplikacji
- Transfery pamięci i awaryjne wykonanie na hoście
- Stan termiczny oraz praca długotrwała, a nie krótkotrwała

Detektor działa w potoku: pobranie obrazu, zmiana rozmiaru/letterbox, normalizacja, transfer do urządzenia, inferencja sieci neuronowej, dekodowanie, NMS, skalowanie współrzędnych i logika aplikacji. Dostawcy często publikują jedynie wyniki dla środkowej części, czyli inferencji sieci. [MLPerf Inference Edge](https://mlcommons.org/benchmarks/inference-edge/) jest wartościowy, ponieważ definiuje scenariusze, progi jakości i reguły pomiaru na poziomie systemu, zamiast porównywać w oderwaniu liczby z materiałów marketingowych.

Minimalny wiarygodny zapis benchmarku wdrożenia YOLO powinien zawierać:

```text
exact checkpoint + checksum
source and compiled model format
compiler, runtime, driver and firmware versions
target board and clock/power mode
input resolution and batch
precision and calibration dataset
accuracy before and after compilation
warmup and sample count
preprocess, inference and postprocess latency
power measurement boundary
```

Bez tych pól dwie wartości FPS zwykle mierzą różne rzeczy.

## Jak wybrać NPU do wizji komputerowej

Wybór należy prowadzić w tej kolejności, a nie według największej liczby na stronie produktu.

1. **Potwierdź zgodność grafu.** Skompiluj dokładnie wyeksportowany model, a nie checkpoint z katalogu o podobnej nazwie.
2. **Zmierz dokładność.** Po kalibracji i kwantyzacji zweryfikuj skompilowany artefakt na zbiorze danych właściwym dla zadania.
3. **Zmierz cały proces.** Uwzględnij konwersję wejścia, transfery, dekodowanie i NMS.
4. **Sprawdź dostęp do oprogramowania.** Ustal, czy kompilator jest publiczny, wymaga rejestracji, czy jest dostępny tylko po zawarciu umowy komercyjnej.
5. **Zamroź macierz wersji.** Zapisz wersje kompilatora, runtime, sterownika, firmware i identyfikatory celów.
6. **Sprawdź rzeczywistą obsługę systemów operacyjnych.** Android, Debian, Yocto, Buildroot, RTOS i Windows nie są wymienne.
7. **Sprawdź dostawy i cykl życia.** Technicznie świetny układ to słaby wybór produkcyjny, jeśli dostępność modułów, sterowników lub długoterminowego wsparcia jest niepewna.
8. **Dopiero potem porównaj koszt, moc i wydajność.** Pomiary mają sens, gdy ten sam model działa poprawnie na obu celach.

### Praktyczne punkty wyjścia zależnie od zastosowania

| Zastosowanie | Platformy warte wstępnej oceny | Uzasadnienie |
|---|---|---|
| Akcelerator przeznaczony do Raspberry Pi | Hailo-8L/8 i Sony IMX500 | Oficjalne produkty Raspberry Pi, obrazy oprogramowania i konkretne procesy deweloperskie |
| Ogólny dodatek PCIe, M.2 lub USB do Linuxa | Produkty DEEPX, MemryX, Hailo i Axelera | Dostępne formaty akceleratorów, zależne od zgodności hosta/sterownika |
| Niedroga płytka SBC lub kamera z Linuxem | Rockchip RK3588/RK3576, Amlogic ADLA przy dostępnej zgodnej płytce/BSP, AXERA, Canaan K230 lub Sunplus SP7350 | Zintegrowane potoki multimedialne i publiczne dowody dla modeli/kompilatorów; dla NPU A311D2 trzeba sprawdzić VIM4 V13A+ |
| Urządzenia mobilne i Android na dużą skalę | Qualcomm QNN/AI Hub, LiteRT, Apple Core ML | Duża baza wdrożeń i utrzymywane środowiska mobilne |
| Przemysłowy Linux i robotyka | TI TDA4/AM6xA, NXP i.MX, Renesas RZ/V, Hailo | Długowieczne produkty wbudowane oraz ekosystemy kamer i wejść/wyjść |
| Małe urządzenie endpoint lub MCU | STM32N6x7, Infineon PSOC Edge, Himax WiseEye2, Analog Devices MAX7800x, Nuvoton M55M1, Alif/Arm Ethos-U, GAP9 i Syntiant | Ścisłe ograniczenia mocy i pamięci, narzędzia projektowane do tych warunków i limity rozmiaru modeli |
| Konfigurowalna wizja FPGA | Microchip VectorBlox, Lattice sensAI i cele AMD Vitis AI | Elastyczne potoki sprzętowe, ale konfiguracja akceleratora, bitstream i kompilator modelu tworzą jeden system zarządzany wersjami |
| Wizja PCIe o dużej przepustowości | Axelera Metis, Hailo, DEEPX, Mobilint, MemryX, Rebellions lub SiMa.ai | Dedykowane akceleratory i pozycjonowanie dla wielu strumieni |
| Ekosystem produktów skupiony na Chinach | Rockchip, AXERA, D-Robotics, SOPHGO, Huawei Ascend, HiSilicon i Cambricon | Silne regionalne płytki, toolchainy i przykłady modeli |

Ta tabela jest krótką listą badawczą, a nie rankingiem wydajności. Zakupy, dostępność regionalna i konkretny model mogą zmienić kolejność.

## Co to oznacza dla LibreYOLO

Skalowalny projekt nie wymaga dziesiątek niezwiązanych ze sobą eksporterów. Powinien opierać się na jednym ścisłym kontrakcie wymiany oraz małych adapterach kompilatorów i środowisk uruchomieniowych dostawców:

```text
LibreYOLO model
    -> deterministic static ONNX
    -> VendorCompiler.compile(model, target, calibration, precision)
    -> native artifact + manifest
    -> VendorRuntime.load() / infer()
    -> shared task-specific decode and Results objects
```

Każdy natywny artefakt powinien mieć towarzyszący manifest zawierający:

- Dostawcę, cel układu i cel płytki
- Wersje kompilatora, runtime, sterownika i firmware
- Sumy kontrolne źródłowego checkpointu i ONNX
- Nazwy wejść, kształty, układ, przestrzeń kolorów, normalizację i typ danych
- Precyzję kwantyzacji, skale jeśli są dostępne, oraz hash danych kalibracyjnych
- Nazwy i kształty tensorów wyjściowych oraz ich znaczenie semantyczne
- Zadanie detekcji, nazwy klas i informację, czy dekodowanie/NMS odbywa się na układzie, czy po stronie hosta
- Zarejestrowane wyniki zgodności numerycznej i dokładności zadania

LibreYOLO udostępnia już przenośny [eksport ONNX](/docs/export/onnx), [narzędzia kwantyzacji](/docs/export/quantization), bezpośredni, ale celowo wąski [eksport RKNN](/docs/export/rknn) oraz uczciwy proces z zewnętrznym kompilatorem [Hailo](/docs/export/hailo). Według kryteriów tego przewodnika mocnym kandydatem do następnej integracji jest Amlogic ADLA: toolkit jest publiczny, obsługa modeli w dużym stopniu pokrywa się z LibreYOLO, a starszą ścieżkę A311D można wyraźnie oddzielić.

Kolejny priorytet po Amlogic powinien zależeć od sprzętu użytkowników i możliwości walidacji dokładności, a nie tylko dostępności kompilatora. DEEPX, Sony IMX500, D-Robotics, AXERA, SOPHGO i Qualcomm są atrakcyjne technicznie; TI, NXP, ST i Renesas stają się szczególnie wartościowe, gdy partnerzy przemysłowi mogą zapewnić płytki i długoterminowy dostęp do CI.

## Lista obserwacyjna: rzeczywisty krzem bez publicznego kontraktu integracji

Całkowite pominięcie firmy może zniekształcić mapę rynku. Oznaczenie jej jako „obsługiwanej” może być jeszcze gorsze. Ci dostawcy sprzedają, próbkują lub zapowiedzieli odpowiedni krzem, ale publiczne dowody nie potwierdzają jeszcze odtwarzalnej, aktualnej ścieżki kompilatora, artefaktu i nazwanego modelu, która nadawałaby się na backend LibreYOLO.

| Firma | Co istnieje | Dlaczego pozostaje na liście obserwacyjnej |
|---|---|---|
| [Samsung](https://semiconductor.samsung.com/processor/automotive-processor/exynos-auto-v920/) | Exynos Auto V920 zawiera dwurdzeniowe NPU reklamowane jako osiągające do 23.1 TOPS | Samsung podaje, że [Neural SDK nie jest już udostępniany zewnętrznym deweloperom](https://developer.samsung.com/neural/overview.html); Samsung ONE/Circle nie potwierdza dostępu do NPU Exynos |
| [Novatek](https://www.novatek.com.tw/en-global/Milestone/aboutus_milestones) | Aktualne kamienie milowe firmy wymieniają SoC Edge AI i Edge Vision/Imaging AI; w kamieniu milowym z 2020 r. wspomniano MobileNet, SSD i YOLOv3 | Brak aktualnej publicznej macierzy numerów części/kompilatora/artefaktów/operatorów/modeli; historyczne YOLOv3 nie potwierdza bieżącego SDK |
| [Nextchip](https://www.nextchip.com/en/adas/adas.php?idx=5) | Procesory samochodowe APACHE5/NVS2900 oraz aktualne [APACHE6/NVS3000](https://www.nextchip.com/en/adas/adas.php?idx=7) z NPU aiMotive aiWare; sama strona APACHE6 podaje obecnie zarówno 12, jak i 8 TOPS | Istnieją aiWare Studio i runtime, ale nie znaleziono publicznego artefaktu, przewodnika kwantyzacji, listy operatorów ani macierzy YOLO; sprzecznych danych dostawcy nie należy po cichu uzgadniać |
| [Black Sesame Technologies](https://bst.ai/en.html) | Układy samochodowe/brzegowe AI Huashan A1000/A2000 i Wudang serii C | Dostępne są publiczne informacje o produkcie, ale brak kompilatora do samodzielnego użycia, kontraktu artefaktu lub odtwarzalnego katalogu modeli |
| [Ingenic](https://en.ingenic.com.cn/products-detail/id-19.html) | SoC kamerowe T41/T40 z deklaracjami NPU o niskiej precyzji; Magik AI opisuje PTQ/QAT i kompilację grafów | Nie znaleziono bieżącego kompilatora do pobrania, kontraktu artefaktu ani dokładnej listy modeli YOLO od dostawcy |
| [Fullhan](https://fullhan.com/en/index.php?a=type&c=article&tid=9) | Wiele SoC IPC reklamuje NPU 0.5–2 TOPS, a [rodzina rejestratorów MC6880 NVR](https://fullhan.com/en/index.php?a=type&c=article&tid=48) reklamuje 4 TOPS | Brak publicznej dokumentacji kompilatora/środowiska/frameworka/modeli NN wystarczającej do niezależnego odtworzenia |
| [Goke Microelectronics](http://www.gokemicro.com/News/info.aspx?itemid=437) | Własna strona układów kamerowych GK7606V1/GK7206V1/GK7203V1 reklamuje wydajność zintegrowanego NPU, lecz w chwili publikacji jest dostępna tylko przez stary HTTP | Brak publicznego konwertera, kontraktu runtime lub macierzy nazwanych modeli; kanał jest zorientowany na OEM |
| [Chengheng Micro](https://en.chenghengmicro.com/) | CH37 reklamuje 64 INT8 TOPS i tryby FP16/FP32/FP64; [oś czasu firmy z 2026 r.](https://chenghengmicro.com/about.html) podaje rozpoczęcie produkcji małoseryjnej | Nie znaleziono publicznego SDK, kontraktu kompilatora ani walidacji nazwanych modeli; należy traktować to jako wczesną platformę komercyjną bez samoobsługowego dostępu, a nie odtwarzalne dowody wdrożenia |
| [Bouffalo Lab](https://github.com/bouffalolab/bouffalo_sdk) | BL808 zawiera NPU BLAI-100, a oficjalne repozytoria SDK są aktualne | Utrzymywany SDK nie udostępnia aktualnego oficjalnego procesu konwersji/przykładów BLAI; dostępne ścieżki są starsze lub pochodzą ze społeczności |

Lista obserwacyjna opiera się celowo na dowodach. Dostawca może trafić do tabel wdrażalnych platform, publikując aktualny toolchain lub ścieżkę ewaluacji, kontrakt artefaktu dla konkretnego celu oraz co najmniej jeden nazwany model z przepisem od początku do końca.

## Czego celowo nie stwierdzono

Ten artykuł nie twierdzi, że każdy wymieniony model działa bez zmian z dowolnego repozytorium źródłowego. Modele z katalogów dostawców są często modyfikowane, dzielone w innych węzłach wyjściowych lub łączone z własnym postprocessingiem.

Poniższych stwierdzeń również nie uznano za deklaracje obsługi:

- Kompilator importuje ONNX.
- Układ reklamuje sterownik Android NNAPI.
- Dystrybutor twierdzi, że płytka jest „zgodna z YOLO”.
- Repozytorium społeczności uruchamia jedną wersję jednego forka modelu.
- Model kompiluje się bez błędu.
- Dostawca podaje FPS samego NPU bez wyniku dokładności.

Akceleratory w telefonach Google Tensor i liczne niestandardowe układy ASIC produkowane wyłącznie dla OEM także istnieją, ale Google nie udostępnia Tensor jako ogólnego celu kompilatora do samodzielnego użycia, porównywalnego z platformami powyżej. Samo istnienie firmy, schemat blokowy NPU lub akceleracja Androida nie wystarczają, aby wymyślić deklarację integracji LibreYOLO.

Podobnie akceleratory chmurowe, takie jak Google TPU, AWS Inferentia/Trainium, Microsoft Maia i karty AI przeznaczone wyłącznie do centrów danych, nie wchodzą w zakres. Ten przewodnik dotyczy sprzętu, który deweloper wizji komputerowej może realnie wdrożyć na brzegu sieci.

Licencjonowanie modelu to osobna kwestia. Opublikowanie przez dostawcę układu przepisu konwersji YOLO nie daje praw do używania ani redystrybucji źródłowych wag, kodu trenowania lub skompilowanej pochodnej. W konkretnym produkcie trzeba osobno sprawdzić obsługę sprzętu, licencję SDK i licencję modelu.

## Metodyka badań i korekty

Dla każdej firmy szukano czterech typów źródeł pierwotnych:

1. Aktualnej strony produktu lub arkusza danych identyfikującego krzem.
2. Dokumentacji kompilatora i środowiska uruchomieniowego opisującej rzeczywistą ścieżkę wdrożenia.
3. Katalogu modeli, tabeli zgodności lub samouczka od początku do końca, które wymieniają modele wizji.
4. Sygnału dotyczącego cyklu życia lub dostępu, pokazującego, czy deweloperzy mogą faktycznie uzyskać narzędzia.

Oficjalne organizacje GitHub liczą się jako źródła dostawców. Dokumentację producentów płytek oznaczono jako dowody ekosystemu, gdy sam dostawca układu nie publikuje tych samych materiałów. Twierdzenia o wydajności ze źródeł zewnętrznych pominięto w tabelach porównawczych.

Następnie gotowy projekt ponownie sprawdzono w osobnych przeglądach grup dostawców i tabel przekrojowych. Ponownie zweryfikowano zakres celów, nazwy artefaktów, precyzję, dowody dla modeli kontra postprocessingu, status zapowiedzi kontra produktów dostępnych, oznaczenia cyklu życia, mianowniki benchmarków i liczbę wszystkich podmiotów. Gdy dwie strony dostawcy są sprzeczne, przewodnik pokazuje tę sprzeczność zamiast po cichu wybierać wygodniejszą wartość.

Rynek szybko się zmienia. Jeśli pracujesz w jednej z tych firm i układ, SDK, lista modeli lub status dostępu są opisane nieprawidłowo, prześlij LibreYOLO dokładny publiczny adres URL dokumentacji i wersję. Korekty poparte źródłami pierwotnymi powinny zastąpić ten tekst; nieudokumentowane twierdzenia marketingowe nie.

## FAQ

### Ile firm oferuje edge AI NPU?

Nie istnieje jedna uniwersalna liczba firm, ponieważ dostawcy produktów, licencjodawcy IP, inteligentne sensory, GPU i prywatne samochodowe układy ASIC nakładają się na siebie. Ten niewyczerpujący przewodnik obejmuje 69 firm i ekosystemów platform z odpowiednimi układami Edge AI, platformami akceleratorów, licencjonowanym IP NPU lub wiarygodnymi układami z listy obserwacyjnej według stanu na sierpień 2026 r.

### Czym jest NPU?

Neural processing unit to wyspecjalizowany układ do operacji sieci neuronowych, takich jak sploty i mnożenie macierzy. Dostawcy używają nazw NPU, AIPU, BPU, KPU, DLA, HTP, TPU i MLA, a skompilowane modele na ogół nie są przenośne między firmami.

### Które firmy oferujące NPU oficjalnie obsługują modele YOLO?

Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Sony IMX500 za pośrednictwem oficjalnego repozytorium AI Camera firmy Raspberry Pi, STMicroelectronics, Renesas, Lattice, Himax, Microchip i kilka innych firm mają własne wpisy w katalogach modeli, samouczki lub zweryfikowane wyniki dla co najmniej jednej generacji YOLO. Nadal istotne są konkretna generacja, zadanie, układ docelowy i wersja SDK.

### Czy dowolny model ONNX można uruchomić na dowolnym NPU?

Nie. ONNX jest formatem wymiany, a nie gwarancją zgodności ze sprzętem. Kompilator NPU musi obsługiwać każdy operator, kształt tensora i typ danych w grafie. Często stosuje się statyczne kształty, podział grafu, operacje niestandardowe i awaryjne wykonanie na CPU.

### Które NPU jest najlepsze do YOLO?

Nie ma uniwersalnego zwycięzcy. Hailo, Rockchip, Axelera AI, DEEPX i Amlogic mają szeroką, udokumentowaną obsługę YOLO, a urządzenia Qualcomm są wyjątkowo rozpowszechnione. Wybór powinien uwzględniać dokładność po kwantyzacji, opóźnienie całego pipeline'u, moc przy długotrwałym obciążeniu, cenę, dostępność, dostęp do SDK i obsługę systemu operacyjnego.

### Dlaczego wartości TOPS dla NPU nie są bezpośrednio porównywalne?

Dostawcy mogą inaczej liczyć precyzję, operacje rzadkie, operacje multiply-accumulate i założenia dotyczące szczytowego wykorzystania. TOPS pomija ruch danych w pamięci, nieobsługiwane operatory, przetwarzanie wstępne i końcowe oraz narzut hosta. Należy porównywać ten sam model, precyzję, rozdzielczość, dokładność i cały pipeline.

### Na czym polega kalibracja NPU?

Podczas kalibracji reprezentatywne, zwykle nieoznaczone obrazy wdrożeniowe są przepuszczane przez model, aby kompilator mógł oszacować zakresy aktywacji dla INT8 lub innego formatu o niskiej precyzji. Losowe lub niereprezentatywne obrazy mogą nadal pozwolić utworzyć skompilowany artefakt, ale pogorszyć dokładność zadania.

### Czy LibreYOLO obsługuje edge NPU?

LibreYOLO eksportuje ONNX i kilka formatów środowisk uruchomieniowych, ma bezpośrednią ścieżkę kompilacji RKNN dla wybranych modeli Rockchip i opisuje zewnętrzny proces kompilacji Hailo. Każdy inny natywny artefakt dostawcy nadal wymaga jego SDK i należy go traktować jako kandydata do integracji, dopóki nie zostanie skompilowany, zweryfikowany i uruchomiony na sprzęcie.
