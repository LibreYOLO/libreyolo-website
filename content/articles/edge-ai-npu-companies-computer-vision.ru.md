---
title: "69 компаний Edge AI с NPU: чипы, SDK и YOLO (2026)"
description: "Обзор 69 компаний и экосистем edge AI с источниками: чипы, SDK, артефакты, квантизация и подтверждённая поддержка компьютерного зрения и YOLO."
date: 2026-08-15
author: Xuban
tags: [edge-ai, npu, computer-vision, yolo, hardware, ai-accelerators, amlogic, hailo, rockchip]
faq:
  - q: "Сколько компаний выпускают edge AI NPU?"
    a: "Универсального числа нет: пересекаются производители продуктов, лицензиары IP, интеллектуальные сенсоры, GPU и закрытые автомобильные ASIC. В этом неисчерпывающем обзоре отслеживаются 69 компаний и экосистем платформ, связанных с edge AI: чипы, платформы ускорителей, лицензируемый NPU IP или заслуживающий внимания кремний по состоянию на август 2026 года."
  - q: "Что такое NPU?"
    a: "Neural Processing Unit — специализированное аппаратное устройство для операций нейронных сетей, например свёрток и умножения матриц. Производители используют термины NPU, AIPU, BPU, KPU, DLA, HTP, TPU и MLA. Скомпилированные модели обычно нельзя переносить между компаниями."
  - q: "Какие компании с NPU официально поддерживают модели YOLO?"
    a: "У Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Sony IMX500 через официальный репозиторий Raspberry Pi AI Camera, STMicroelectronics, Renesas, Lattice, Himax, Microchip и ряда других компаний есть собственные записи в каталогах моделей, руководства или проверенные результаты как минимум для одного поколения YOLO. Важно учитывать конкретное поколение, задачу, целевой чип и версию SDK."
  - q: "Можно ли запустить любую модель ONNX на любом NPU?"
    a: "Нет. ONNX — формат обмена, а не гарантия совместимости с оборудованием. Компилятор NPU должен поддерживать все операции, формы тензоров и типы данных графа. Часто встречаются статические формы, разбиение графа, пользовательские операции и откат на CPU."
  - q: "Какой NPU лучше всего подходит для YOLO?"
    a: "Универсального лидера нет. У Hailo, Rockchip, Axelera AI, DEEPX и Amlogic хорошо документирована поддержка YOLO, а устройства Qualcomm отличаются очень широким распространением. Выбирайте по точности после квантизации, задержке всего пайплайна, длительной мощности, цене, доступности, доступу к SDK и поддержке ОС."
  - q: "Почему показатели NPU в TOPS нельзя напрямую сравнивать?"
    a: "Производители могут учитывать разные точности, разреженные операции, операции умножения с накоплением и допущения о пиковой загрузке. TOPS не учитывает трафик памяти, неподдерживаемые операции, предобработку, постобработку и накладные расходы хоста. Сравнивайте одинаковые модель, точность, разрешение, качество и полный пайплайн."
  - q: "Что такое калибровка NPU?"
    a: "При калибровке через модель пропускают репрезентативные, обычно неразмеченные изображения для развёртывания, чтобы компилятор оценил диапазоны активаций для INT8 или другого формата с низкой точностью. Даже случайные или нерепрезентативные изображения могут дать скомпилированный артефакт, но ухудшить точность на задаче."
  - q: "Поддерживает ли LibreYOLO периферийные NPU?"
    a: "LibreYOLO экспортирует ONNX и несколько форматов среды выполнения, напрямую компилирует выбранные модели Rockchip через RKNN и описывает внешнюю процедуру компиляции для Hailo. Для всех остальных артефактов производителя требуется его SDK; до компиляции, проверки и запуска на оборудовании их следует считать кандидатами на интеграцию."
---

**Единого рынка NPU нет. В этом неисчерпывающем обзоре собраны 69 компаний и экосистем платформ: 51 производитель с готовыми к развёртыванию чипами или платформами, девять поставщиков NPU IP и девять компаний в отдельном списке наблюдения.** Они охватывают несколько несовместимых классов ускорителей и почти столько же компиляторных стеков. Большинство обещает один и тот же путь: экспортировать модель, выполнить квантизацию и компиляцию, скопировать проприетарный артефакт на плату и вызвать среду выполнения производителя. Именно детали определяют, займёт этот путь день или квартал инженерной работы.

В этом руководстве собраны компании, которые в 2026 году предлагают заслуживающее доверия оборудование для компьютерного зрения. Для каждой указаны подходящие чипы, названия SDK и компиляторов, артефакты развёртывания, уровень публичного доступа и модели, документированные самим производителем. Отдельно рассматривается Amlogic: новый стек ADLA существенно отличается от старой экосистемы NPU A311D.

> **Объём исследования:** источники проверены 15 августа 2026 года. Заявления о поддержке конкретных моделей основаны на страницах продуктов, документации, каталогах моделей, репозиториях или руководствах производителей, если не указано иное. Рекламные показатели TOPS приведены со слов производителей и не являются результатами независимых сопоставимых бенчмарков. Это руководство для разработчиков, а не инвестиционная рекомендация или платный рейтинг.

**Быстрая навигация:** [таблицы компаний](#npu-companies-and-platforms-at-a-glance) | [подробно об Amlogic](#amlogic-two-npu-generations-not-one) | [профили производителей](#the-strongest-public-deployment-ecosystems) | [таблица скомпилированных артефактов](#what-you-actually-deploy-the-artifact-lock-in-table) | [доступ и жизненный цикл](#tool-access-and-lifecycle-signals) | [план LibreYOLO](#what-this-means-for-libreyolo)

## Самый важный вывод

Аппаратные решения фрагментированы, но схема развёртывания удивительно единообразна:

```text
PyTorch checkpoint
    -> ONNX or TFLite interchange graph
    -> representative calibration data
    -> vendor quantizer and graph compiler
    -> chip-specific binary or model package
    -> vendor runtime on the target
    -> application preprocessing, decode, NMS and rendering
```

[ONNX явно предусматривает среды выполнения, генераторы кода и аппаратные реализации](https://onnx.ai/onnx/repo-docs/IR.html), но файл ONNX — только точка передачи. Это не означает, что каждый оператор ONNX будет преобразован для любого ускорителя. Реальное выполнение на NPU зависит от статических форм входа, ограничений на поддерживаемые операции, правил квантизации и отката на хост.

Поэтому программный стек — часть чипа. Формально быстрый NPU с закрытым или ненадёжным компилятором может оказаться хуже для развёртывания, чем менее мощное устройство с открытым набором инструментов, каталогом моделей, симулятором и стабильной средой выполнения.

## Что в этом руководстве означает «поддержка»

В документации производителей слово *поддержка* часто используют слишком свободно. В статье выделены четыре уровня доказательств:

| Уровень доказательств | Что подтверждается | Что не подтверждается |
|---|---|---|
| **Проверенная модель** | Производитель публикует запись о модели, результат или таблицу совместимости для конкретного чипа | Что изменённый вами чекпойнт сохранит ту же точность |
| **Официальный пример** | Производитель предоставляет сквозное руководство или демонстрацию для указанной модели | Что скомпилируются другие размеры, задачи или поколения |
| **Возможности компилятора** | SDK импортирует фреймворк или предоставляет список поддерживаемых операций | Что конкретный граф YOLO работает от начала до конца |
| **Маркетинговое заявление или закрытый доступ** | Продукт предназначен для компьютерного зрения и существует закрытый SDK | Что совместимость модели можно публично воспроизвести |

Это различие важно. «Импортирует ONNX» не равно «поддерживает сегментацию YOLO11». Модель может успешно разобрать граф, но выполнить его на CPU, скомпилировать с неверными значениями выходов, переполнить память ускорителя или потерять точность при квантизации.

## NPU-компании и платформы: краткий обзор

В таблицах намеренно смешаны SoC, дискретные ускорители, интеллектуальные сенсоры и смежные платформы GPU/FPGA. Они конкурируют за одно и то же решение о развёртывании, хотя производители называют свои ускорители NPU, AIPU, BPU, KPU, DLA, HTP, TPU или MLA.

### Встраиваемые SoC, интеллектуальные сенсоры и промышленные процессоры

| Компания | Подходящие чипы или платформы | SDK, компилятор и артефакт | Публичные свидетельства поддержки компьютерного зрения | Доступ |
|---|---|---|---|---|
| [Amlogic](https://github.com/Amlogic-NN/amlnn-toolkit) | A311D2, S928X, S905X5/S905D5, A311Y3, C308L/C302X/C302X2, T968D4, C305X2 и A123X | AMLNN Toolkit и `libnnsdk.so`, скомпилированный `.adla`; отдельный устаревший стек Acuity `.nb` для A311D | В [песочнице моделей](https://github.com/Amlogic-NN/amlnn-model-playground) документированы YOLOv5/6/7/8/10/11, YOLOX, YOLOE, YOLO-World, PP-YOLOE, сегментация, поза и OBB на A311D2, S905X5, A311Y3, C305X2 и A123X; остальные перечисленные здесь чипы являются целями компилятора, но не входят в эту матрицу поддержки | Публичные GitHub и пакеты wheels; также нужна совместимая BSP/драйвер |
| [Rockchip](https://github.com/airockchip/rknn-toolkit2) | RK3562/3566/3568, RK3576, RK3588, RV1126B | RKNN-Toolkit2, RKNN Runtime, `.rknn` | YOLOv5/6/7/8/10/11, YOLOX, YOLO-World, PP-YOLOE, сегментация, поза и OBB в [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) | Публичный GitHub; бинарные пакеты wheels |
| [Qualcomm](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | Платформы Snapdragon и Dragonwing, QCS6490, QCS8550 и Dragonwing IQ-9075 | QAIRT/QNN, SNPE и AI Hub; бинарные контексты QNN или DLC | В [коллекции моделей AI Hub](https://github.com/qualcomm/ai-hub-models) есть YOLOv3/5/6/7/8/9/10/11/26, YOLOX, YOLO-World, YOLOR, RF-DETR и модели детекции, сегментации и оценки позы | Документация открыта; требования к SDK и аккаунту различаются |
| [Texas Instruments](https://github.com/TexasInstruments/edgeai-tidl-tools) | AM62A, AM67A, AM68A, AM69A и TDA4x; предварительная версия TDA54-Q1 | Processor SDK Edge AI и TIDL | В [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo) опубликованы оптимизированные модели детекции, сегментации, оценки позы и классификации для текущих AM6xA/TDA4; это пока не матрица целей для предварительной версии TDA54-Q1 | Публичный GitHub и Processor SDK |
| [NXP](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | i.MX 8M Plus, i.MX 93, i.MX 95 и приобретённые Kinara Ara-1/Ara240 | eIQ с TIM-VX, Vela, Neutron Converter и Ara SDK | В [eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) есть ресурсы или рецепты для YOLOv4-tiny/v8, NanoDet, CenterNet, FastestDet, SSD Lite и YOLACT; запись о YOLOv5 содержит только документацию | Часть компонентов общедоступна, часть требует аккаунта |
| [STMicroelectronics](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) | Варианты STM32N6x7, в том числе STM32N657/647 с ускорителем Neural-ART | STM32Cube AI Studio и ST Edge AI Core | В текущем репозитории services перечислены Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26 и ST-YOLOX, а также поза и сегментация | Публичные инструменты и репозитории |
| [Infineon](https://documentation.infineon.com/psocedge/) | PSOC Edge E83/E84 с Cortex-M55 и Ethos-U55; отдельный ускоритель NNLite на стороне M33 | ModusToolbox, [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter), TFLite Micro и Arm Vela | В материалах об архитектуре MobileNetV1/V2 указаны как типовые ядра, а в AI Kit E84 есть камера; подтверждённой производителем матрицы YOLO для PSOC Edge не найдено | Публичная документация, компоненты SDK и доступные комплекты оценки E84 |
| [Renesas](https://github.com/renesas-rz/rzv_drp-ai_tvm) | RZ/V2L, V2M, V2MA, V2H и V2N | DRP-AI Translator, DRP-AI TVM и RUHMI | В [списке моделей RZ/V2H](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) проверены варианты YOLOv5/v8/v11/26, YOLOX, позы и сегментации | Публичный GitHub и SDK платы |
| [Sony](https://www.aitrios.sony-semicon.com/edge-ai-devices/imx500) | Интеллектуальный сенсор зрения IMX500 и Raspberry Pi AI Camera | Edge-MDT, Model Compression Toolkit и упаковщик IMX500; пакет `.rpk` | В [каталоге Raspberry Pi IMX500](https://github.com/raspberrypi/imx500-models) есть YOLOv8n, YOLO11n, EfficientDet Lite, NanoDet+ и SSD | Публичный процесс Raspberry Pi; действуют условия Sony для инструментов |
| [Ambarella](https://www.ambarella.com/developer/model-garden/) | Семейства CV72/CV75, CV5/CV52 и CV3-AD | Cooper Developer Platform, компилятор CVflow и графы среды выполнения | В публичном каталоге моделей названы YOLOX-S, RTMDet-nano, DeepLabV3+, TopFormer, OWL-ViT и LLaVA OneVision | В основном доступ через партнёров |
| [Synaptics](https://developer.synaptics.com/docs/sl/overview) | Astra SL1600/SL1680; SL2611/13/15/17/19; отдельное семейство микроконтроллеров SR100 | SyNAP `.synap` для SL16xx; Torq/IREE `.vmfb` для SL261x; отдельный SR SDK для Ethos-U55 | Официальный бенчмарк SyNAP для YOLOv8n/v8s и пример Torq YOLOv8 для SL261x; результаты SR нужно оценивать отдельно | Портал разработчика; некоторые загрузки закрыты |
| [MediaTek](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) | Genio 360/360P/420/520/720 с NP8 и MDLA 5.3; Genio 510/700/1200 со старыми NP6/MDLA | NeuroPilot Converter, `ncc-tflite`, Neuron Runtime и `.dla`; путь ONNX Runtime на отдельных новых системах | В официальном IoT AI Hub есть страницы моделей и бенчмарки YOLOv5 и YOLOv8, а также классификационных моделей и моделей лиц | Публичная документация Yocto; основные комплекты NP8 и материалы Android требуют доступа прямого заказчика/NDA |
| [Allwinner](https://docs.aw-ol.com/v853/en/npu/dev_npu/) | ВидеосоC V853 с NPU Vivante на 1 TOPS | Преобразование Acuity/Pegasus и среда выполнения `viplite` | В официальных материалах V853 названы YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet и сети для лиц/людей | Публичная документация; загрузка SDK может требовать аккаунта |
| [Canaan/Kendryte](https://github.com/kendryte/nncase) | Современные K230/K230D; устаревшие K210/K510 KPU | Компилятор и среда выполнения `nncase`; `.kmodel` | Руководство nncase для K230 описывает компиляцию/симуляцию/запуск YOLOv5s; официальный каталог демонстраций и актуальный [журнал изменений CanMV](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md) документируют задачи YOLOv8/11/26 | Публичные компилятор и PyPI; модуль чипа распространяется в бинарном виде |
| [Alif Semiconductor](https://alifsemi.com/support/kits/ensemble-e7appkit/) | Ensemble E7 с двумя ML-ускорителями, включая Ethos-U55; [E8](https://alifsemi.com/ensemble-e8-series/) с одним Ethos-U85 и двумя NPU Ethos-U55 | Развёртывание на основе Arm Vela и TFLite Micro | Alif публикует семейный бенчмарк детектора лиц YOLO-Fastest с INT8, но широкой матрицы современных YOLO по отдельным целям нет | Публичная документация и материалы комплектов оценки |
| [Himax](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) | MCU периферийного AI HX6538 WiseEye2 с Cortex-M55 и Ethos-U55 | TFLite Micro и Arm Vela, с CMSIS-NN и эталонными ядрами для отката | В официальных [примерах WiseEye2](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) есть детекция, поза и классификация на YOLOv8n, детекция на YOLO11n, Face Mesh и PeopleNet | Публичные GitHub, документация и доступная плата партнёра |
| [Analog Devices](https://www.analog.com/en/products/max78002.html) | AI MCU MAX78000 и MAX78002 с маломощными CNN-ускорителями | [`ai8x-training`](https://github.com/analogdevicesinc/ai8x-training), `ai8x-synthesis`/`izer` и MSDK; сгенерированные C-код и веса | Официальные материалы посвящены идентификации/детекции лиц, RetinaNet, Visual Wake Words, классификаторам и распознаванию действий; развёртывание YOLO производителем не найдено | Публичные инструменты GitHub, документация и отладочные платы |
| [D-Robotics](https://github.com/D-Robotics/rdk_model_zoo) | Платы RDK X3/X5/Ultra и новых BPU серии S100 | OpenExplorer/Algorithm Toolchain и `hbm_runtime`; X5 `.bin`, текущий `rdk_s` `.hbm` | Текущая ветка `rdk_x5` документирует для RDK X5 YOLOv5/v5u/v8/v9/v10/11/12/13/26, YOLOE, YOLO-World, сегментацию, позу, классификацию, OCR и CLIP; для X3 и серии S используются отдельные ветки | Публичный GitHub; некоторые пакеты инструментов зависят от платформы |
| [AXERA](https://github.com/AXERA-TECH/ax-samples) | AX650, AX637, AX630C, AX620Q и AX615 | Компилятор Pulsar2 и AXEngine; `.axmodel` | В текущих официальных примерах есть YOLOv5/6/7/8/9/10/11/13/26, YOLOX и YOLO-World; задачи и цели различаются по чипам | Публичные примеры и документация; компилятор распространяется отдельно |
| [SOPHGO](https://github.com/sophgo/tpu-mlir) | BM1684/1684X/1688/1690 и CV186X/CV18xx | TPU-MLIR и среда выполнения SOPHON; `.bmodel` или `.cvimodel` | Официальные демонстрации охватывают YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, OBB, сегментацию, лица, SAM и OCR | Компилятор с открытым исходным кодом и среда выполнения производителя |
| [Huawei Ascend](https://www.hiascend.com/en/software/cann) | Ascend 310/310P/310B и периферийные продукты Atlas 200I/300I | CANN, компилятор ATC и AscendCL; `.om` | В официальном [Ascend ModelZoo](https://github.com/Ascend/modelzoo) есть YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN и модели сегментации | Документация открыта; пакеты CANN и ядра должны соответствовать цели |
| [Cambricon](https://github.com/Cambricon/magicmind_cloud) | MLU370-X4/S4 в указанной публичной матрице; MLU270 — устаревшее оборудование, отсутствующее в ней | Neuware и MagicMind | В матрице MagicMind 1.7 из репозитория перечислены YOLOv3/4/5/7/8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN и модели сегментации | Исторические примеры моделей открыты; доступ к актуальным SDK/контейнеру закрыт |
| [Sunplus](https://sunplus.atlassian.net/wiki/spaces/C3/pages/1971126471/SP7350%2BSpecification) | SP7350/C3V, заявлено около 4.1–4.6 TOPS | Docker с Vivante Acuity NPU, среда выполнения SNNF и `.nb` | Официальная документация содержит YOLOv5 и пользовательское сквозное [развёртывание YOLOv8](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350) | Публичные исходники/документация и экосистема плат |
| [ESWIN Computing](https://www.eswincomputing.com/en/news/info/96.html) | EIC7700/EIC7700X и двухкристальные SoC [EIC7702/EIC7702X](https://www.eswincomputing.com/en/news/info/104.html) на RISC-V | ENNP: EsQuant, актуальный компилятор EsAAC, симулятор и среда выполнения ESSDK; `.model`; в старой документации использовался `ennc-compile` | В документации ENNP на сайте Milk-V есть сквозное [руководство YOLOv3](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3), а также примеры MobileNetV2 и ResNet | Сочетание документации производителя и документации плат; региональные загрузки |
| [Nuvoton](https://www.nuvoton.com/products/microcontrollers/arm-cortex-m55-mcus/m55m1-series/index.html) | MCU M55M1 с Ethos-U55-256 | TFLite Micro, Arm Vela и NuEdgeWise/NuML | В официальных примерах [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise) названы YOLOv8-nano, YOLOX-nano, YOLO Fastest, SSD-MobileNet и классификационные модели | В основном публичный набор инструментов MCU |
| [Realtek](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | Камерная платформа AmebaPro2 RTL8735B / AMB82-mini | Arduino/FreeRTOS SDK, VoE и NeuralNetwork API; `.nb` | В готовые пакеты входят YOLOv3-tiny, YOLOv4-tiny, YOLOv7-tiny, SCRFD и MobileFaceNet | Среда выполнения открыта; для доступа к [конвертеру пользовательских моделей](https://ameba-doc-arduino-sdk.readthedocs-hosted.com/en/latest/FAQ/offline_ai_model_conversion_steps.html) нужно обратиться к производителю |
| [Telechips](https://docs.topst.ai/product/p/ai) | Плата TCC7500 / TOPST AI, заявлено 8 TOPS | TC-NN-Toolkit / Enlight SDK; промежуточный `.enlight` и скомпилированный пакет развёртывания | В [официальном проекте TOPST](https://docs.topst.ai/blog/31) развёрнута YOLOv8s; преобразование UFLD v1/v2 не удалось из-за неподдерживаемых слоёв, предложено только обходное решение с разделением и постобработкой. [YOLOv4](https://community.topst.ai/t/segmentation-fault/358) и [другой процесс для YOLOv8](https://community.topst.ai/t/segmentation-fault-error/415) упоминаются в обсуждениях поддержки | Документация платы открыта; загрузки компилятора/операторов часто требуют разрешения |
| [T-Head](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | TH1520, заявлено 4-TOPS INT8, на LicheePi 4A | Компилятор HHB и CSI-NN2/SHL; `hhb.bm` вместе со сгенерированным кодом/параметрами | В официальных примерах производителя платы на NPU запускаются YOLOv5n/s и MobileNetV2 | Публичные примеры; инструментальный стек устаревает, поддержка неясна |
| [SigmaStar](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | Современная SoC для интеллектуальных камер SSU9383CM и прежние семейства IPU | Текущая среда выполнения MI_IPU; [старая цепочка SGS_IPU](https://wx.comake.online/doc/doc/SigmaStarDocs-SSC9381G_9351_Pudding-ULS00V040-20210913/customer/development/dla/tools.html) создавала `.sim`, затем `sgsimg.img` | [Старый официальный постпроцессор SDK](https://wx.comake.online/doc/doc/Sigmastar_SDK_v1.2.2/module/User_Guide/Common/SigmaStar_Post_Processing_Module.html) называет SSD и YOLOv1/2/3; публичная совместимость этих моделей и артефактов с SSU9383CM не подтверждена | Современный чип, но публичные свидетельства компилятора/моделей относятся к разным поколениям |
| [HiSilicon](https://www.hisilicon.com/cn/products/smart-vision/machine-vision/hi3516cv610) | Камерные SoC Hi3516CV610/DV500, Hi3519DV500 и Hi3403V100 | ATC в `.om` с платной средой выполнения NNN/SVP-NNN | Матрица HiSpark для конкретных целей, особенно Hi3403 и Hi3591P, содержит YOLOv3–YOLO11, позу, сегментацию, OBB, OCR и глубину; она не подтверждает все SoC в этой строке | Активное направление, отдельное от Ascend; модели репозитория помечены как некоммерческие |

### Специализированные периферийные ускорители и модули

| Компания | Подходящие микросхемы | SDK и артефакт | Публичные свидетельства о моделях | Доступ |
|---|---|---|---|---|
| [Hailo](https://github.com/hailo-ai/hailo_model_zoo) | Hailo-8, Hailo-8L, Hailo-10H и семейство Hailo-15 | Dataflow Compiler и HailoRT; `.hef` | YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 и YOLO26, а также YOLOX, DAMO-YOLO, SSD, EfficientDet, сегментация, поза и OBB; таблицы зависят от поколения | Model Zoo открыт; компилятор доступен через Developer Zone |
| [Axelera AI](https://docs.axelera.ai/sdk/reference/models/model-zoo/) | Поставляемые продукты Metis; анонсированные Europa и Titania | Публичный Voyager SDK; alpha Pipeline Builder `.axm`/`.axe`, классические пакеты `.axmodel` | Проверены YOLOv3/5/7/8/9/10/11/26, YOLOX, YOLO-NAS, OBB, поза, сегментация и множество моделей вне YOLO | SDK открыт на GitHub; поддержка клиентов требует аккаунта |
| [DEEPX](https://developer.deepx.ai/modelzoo/) | DX-M1 и DX-M1M | DXNN SDK, DX-COM и DX-RT; `.dxnn` | При проверке 15 августа 2026 года каталог содержал 354 записи для DX-COM 2.4.0/DX-RT 3.4.0, включая YOLOv3–YOLO11 и YOLO26, YOLOX, SSD, EfficientDet, NanoDet, сегментацию, позу и OBB | Публичные ресурсы разработчика и пакет инструментов |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | Ускоритель MX3 и многокристальные модули | MemryX SDK, Neural Compiler и среда выполнения; `.dfp` | Официальные примеры и заметки к релизам охватывают детекцию, сегментацию и позу, в том числе YOLOv10, YOLO11 и YOLO26 | Публичные документация и SDK |
| [Kneron](https://doc.kneron.com/docs/) | KL520, KL530, KL630, KL720 и KL730 имеют актуальную документацию компилятора; KL830 встречается в некоторых API PLUS, но отсутствует в текущем списке целей компилятора | Kneron PLUS и Model Toolchain; `.nef` для документированных целей компилятора | Официальное [руководство по YOLO](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/) посвящено Tiny-YOLOv3; в других материалах рассматривается YOLOv5 | Документация открыта; пакеты инструментов/загрузки зависят от чипа |
| [SiMa.ai](https://docs.sima.ai/pages/palette/modelsdk.html) | MLSoC и серийная платформа Modalix на 50 TOPS | Palette, ModelSDK, MLA Compiler и ModelExecutor | Публичные релизы подтверждают YOLOv7/v8, YOLOX, позу/сегментацию, DETR, Mask R-CNN и EfficientDet | Документация открыта; SDK продукта коммерческий |
| [EdgeCortix](https://www.edgecortix.com/en/hardware) | SAKURA-II, заявленный ускоритель на 60 TOPS в модулях M.2 и PCIe | Компилятор MERA и фреймворк | Производитель заявляет о поддержке задач от компьютерного зрения до генеративного AI, но не публикует достаточно точную актуальную матрицу YOLO | Коммерческий запрос на пробную версию/покупку; проверка через партнёра |
| [BrainChip](https://brainchip.com/metatf-dev-tools/) | Поставляемый сопроцессор/модуль AKD1500; прежние платформы AKD1000; IP Akida 2 | Пакеты MetaTF: `akida-models`, `quantizeml`, `cnn2snn` и среда выполнения `akida` | В карточке Akida 2 названы детектор AkidaNet0.5 YOLOv2, CenterNet, AkidaUNet и распознавание лиц; эти результаты автоматически не подтверждают AKD1500 | Основные пакеты Python открыты; соответствие оборудования цели нужно проверять отдельно |
| [Blaize](https://www.blaize.com/products/) | Продукты P1600, Pathfinder и Xplorer | Picasso SDK, NetDeploy и AI Studio | Компьютерное зрение — целевой рынок, но проверяемую публичную матрицу поддержки конкретных моделей найти не удалось | Коммерческий доступ/ограничения |
| [Google Coral](https://github.com/google-coral/edgetpu) | Устаревшие Edge TPU USB, PCIe, M.2 и Dev Board; отдельно активно разрабатываемый IP с открытым кодом [Coral NPU](https://github.com/google-coral/coralnpu) | Устаревшие Edge TPU Compiler/`libedgetpu`/PyCoral; новый RISC-V Coral NPU предоставляет IP, RTL/симуляцию и примеры ELF | Примеры прежней платформы посвящены SSD MobileNet, классификации, DeepLab и MoveNet; у нового IP нет публичной матрицы YOLO, и он не является прямой заменой Edge TPU | Основные репозитории старой платформы архивированы; IP Coral NPU активно развивается |
| [Lattice Semiconductor](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | FPGA ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E и Avant-X | sensAI Studio, Neural Network Compiler и настраиваемый IP ускорителя | В актуальной таблице компилятора указаны YOLOv1, YOLOv5, YOLOv8 и YOLO11 в зависящих от цели режимах, а также SSD, MobileNetV2-SSD, ResNet и ENet | Компилятор и руководства открыты; условия IP различаются, Advanced CNN Accelerator коммерческий |
| [Mobilint](https://www.mobilint.com/sdk-qb) | Ускорители REGULUS на 10 TOPS и ARIES MLA100 на 80 TOPS | SDK qb; компилятор INT8 и `.mxq` | В публичном [каталоге моделей](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md) перечислены детекторы YOLOv3/5/7/8/9/10/11/12/26, а также варианты сегментации, позы и OBB | Публичные документация/модели; SDK и оборудование коммерческие |
| [Rebellions](https://rebellions.ai/developers/) | Действующие ATOM+ CA22 и ATOM-Max CA25; снятые с поддержки ATOM CA02/ATOM+ CA12; статус инструментов ATOM-Lite CA21 неясен | Компилятор/среда выполнения/профилировщик RBLN; `.rbln` | В официальных релизах поддержки указаны семейства YOLOv3, YOLOv5/6/7/8 и актуальные руководства YOLOv8 | Документация открыта; для пакета компилятора нужны учётные данные портала |
| [FuriosaAI](https://furiosa.ai/warboy) | Warboy Gen1 для компьютерного зрения; отдельно поколение RNGD | Furiosa SDK; INT8 `.enf` для Warboy, отдельный стек `.fxb` для RNGD | В каталоге Warboy указаны SSD, YOLOv5M/L и YOLOv7-w6-pose; в дорожной карте RNGD поддержка YOLOv8m отмечена выполненной в 2024 Q4, но подробной текущей публичной матрицы компьютерного зрения нет | IAM/доступ через аккаунт; эти поколения нужно рассматривать отдельно |

### Смежные платформы, которые разработчики сравнивают с NPU

| Компания | Оборудование | Стек развёртывания | Почему входит в сравнение |
|---|---|---|---|
| [NVIDIA](https://github.com/NVIDIA-AI-IOT/deepstream_tools/tree/main/yolo_deepstream) | GPU Jetson Orin и DLA; GPU Jetson Thor без DLA | JetPack, TensorRT и DeepStream; `.engine` | Очень зрелое развёртывание компьютерного зрения; официальные инструменты DeepStream описывают YOLOv4/v7/v8/v9/11, включая конфигурацию YOLO11 OBB. Во многих случаях используется GPU, а неподдерживаемые слои Orin DLA могут выполняться на GPU. |
| [AMD](https://vitisai.docs.amd.com/en/6.2/) | Цели Kria/устаревшие DPU; Vitis AI 6.2 GA для Versal AI Edge Gen1 VEK280/VE2802 и Gen2 VEK385; отдельно клиентские NPU Ryzen AI | Устаревший `.xmodel`; снимок NPU для конкретной цели Gen1; кэш скомпилированной модели или производственный пакет `.rai` для Gen2 | Vitis AI публикует материалы по компьютерному зрению, но устаревший DPU, Versal Gen1, Versal Gen2 и Ryzen AI имеют разные контракты компиляции и развёртывания. |
| [Microchip](https://www.microchip.com/en-us/products/fpgas-and-plds/fpga-and-soc-design-tools/vectorblox) | FPGA PolarFire SoC с настраиваемым IP ускорителя CoreVectorBlox | Публичный VectorBlox SDK 3.1 и Libero; файлы развёртывания `.vnnx`, `.hex` и `.ucomp` | Актуальные [руководства](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md) охватывают YOLOv5n, YOLOv8 для детекции/классификации/OBB/позы/сегментации, YOLOv9t и другие модели компьютерного зрения; SDK 3.1 сейчас рассчитан на PolarFire SoC Video Kit. |
| [Intel](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) | NPU Core Ultra, CPU и встроенный/дискретный GPU | IR OpenVINO или кэш скомпилированной модели | OpenVINO предоставляет открытый общий путь для XPU; проверяйте столбцы NPU в [матрице проверенных моделей](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html), а не считайте любой пример YOLO в OpenVINO подтверждённым для NPU. |
| [Apple](https://developer.apple.com/machine-learning/core-ml/) | Apple Neural Engine в чипах A-серии начиная с A11 и в чипах M-серии | Core ML и `coremltools`; `.mlpackage`/`.mlmodelc` | Доступный потребительский NPU через Core ML, но Apple скрывает точное распределение работы между CPU/GPU/Neural Engine, вместо предоставления компилятора для YOLO. |
| [GreenWaves Technologies](https://github.com/GreenWaves-Technologies/nn_menu_gap9) | GAP9 с нейронным движком NE16 | GAP SDK, NNTool и сгенерированный AutoTiler код | В официальных репозиториях есть MobileNet SSD, детекция лиц, классификация и отдельный [детектор людей YOLOX](https://github.com/GreenWaves-Technologies/yolox_people_detection). Полный SDK доступен только квалифицированным заказчикам. |
| [Syntiant](https://www.syntiant.com/hardware) | Серийный NDP200 и проходящий выборку NDP250 Neural Decision Processor | SDK Syntiant и пакеты развёртывания | Сверхмалопотребляющие процессоры для постоянно активного зрения/сенсоров: для NDP200 заявлено менее 1 mW, для распознавания изображений на NDP250 — менее 30 mW. Широкой публичной матрицы YOLO нет. |

## Amlogic: два поколения NPU, а не одно

Amlogic стоит рассмотреть подробнее: в интернете часто смешивают несовместимые продукты. У компании есть старый путь на базе VeriSilicon и новый проприетарный стек ADLA. У них разные компилятор, артефакт и среда выполнения.

| Поколение | Типичные чипы | NPU и набор инструментов | Скомпилированный артефакт | Практический статус |
|---|---|---|---|---|
| Устаревшее | A311D и S905D3, часто встречаются в Khadas VIM3/VIM3L | VeriSilicon Vivante VIPNano-QI, набор Acuity, KSNN и старый `aml_npu_sdk` | `.nb` в каталоге вывода `nbg_unify` | Существующие платы и демонстрации; отдельная устаревшая интеграция |
| Текущее ADLA2 | C308L/C302X, S928X, A311D2, T968D4, S905X5/S905D5 и C302X2 | Amlogic ADLA2, AMLNN Toolkit и NNSDK2 | `.adla` для конкретной цели; W8A8 или W8A16 | Современный стек с упором на целочисленные типы |
| Текущее ADLA3 | A311Y3, C305X2 и A123X | Amlogic ADLA3, AMLNN Toolkit и NNSDK2 | `.adla` для конкретной цели; W4A8, W8A8, W4A16, W8A16 или W16A16 | Добавлены нативные INT4, FP16 и BF16 |

В [спецификации A311D](https://dl.khadas.com/products/vim3/datasheet/a311d-datasheet.pdf) указан исходный NPU INT8 на 5 TOPS. На старой [странице приложений Khadas VIM3](https://docs.khadas.com/products/sbc/vim3/npu/npu-app) и в [обзоре NPU](https://docs.khadas.com/products/sbc/vim3/npu/start) описаны DenseNet CTC, MTCNN, RetinaFace, YOLOFace, YOLOv2, YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv7-tiny и YOLOv8n; в обзоре также документированы YOLOv8n-pose и пометка об устаревании FaceNet. Это реальные примеры, но они подтверждают старую экосистему Acuity `.nb`, а не актуальные чипы ADLA. A311D — не A311D2, а C305X — не C305X2.

Есть и аппаратная ловушка. В [руководстве по ревизиям Khadas VIM4](https://docs.khadas.com/products/sbc/vim4/configurations/identify-version) сказано, что на исходной плате V12/A311D2 ревизии B нет NPU; заявленный NPU на 3.2 TOPS появился в V13A и более поздних платах с компонентом A311D2-N0D ревизии C. Для выбора тестового устройства недостаточно знать только название продукта.

### Современный стек AMLNN и ADLA

Современный [AMLNN Toolkit](https://github.com/Amlogic-NN/amlnn-toolkit/tree/7d3cc9a36179b756ce79c953c14d43acab1f2af5) охватывает преобразование моделей, квантизацию, компиляцию, инференс и профилирование. В закреплённом [руководстве пользователя за май 2026 года](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/02_Amlogic_NPU_User_Guide_V0.1.pdf) документированы импортёры ONNX, TFLite с плавающей точкой или квантизованный, TorchScript `.pt` и PyTorch 2 ExportedProgram/PT2. Пути для TensorFlow, Paddle и Keras в подробном руководстве помечены как планируемые, хотя обзор репозитория формулирует возможности шире. Компилятор создаёт `.adla`. Для развёртывания на Python используются AMLNN или `amlnn_edge_toolkit_lite`; для нативного C/C++ — `libnnsdk.so` через `nnsdk2.h`. Разработка на хосте может идти через ADB с `nnserver`; среда выполнения также рассчитана на Android, Buildroot, Yocto и отдельные конфигурации Debian/Armbian.

В опубликованные Toolkit идентификаторы платформ входят:

| ID цели | Платформа |
|---:|---|
| `001` | C308L / C302X |
| `002` | S928X |
| `003` | A311D2 |
| `004` | T968D4 |
| `005` | S905X5 / S905D5 |
| `006` | C302X2 |
| `007` | A311Y3 |
| `008` | C305X2 / A123X в подробной документации и матрице моделей |

Это поле цели не для красоты. Для Amlogic должны совпадать `.adla`, `nnsdk2.h`, `libnnsdk.so`, BSP и поколения компилятора/среды выполнения. В закреплённом [кратком руководстве](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/01_Amlogic_NPU_Quick_Start_guide_V0.1.pdf) требуются драйвер ADLA 2.0.2 или новее, NNSDK 3.0.0 или новее и NNSDK2 1.0.0 или новее; [руководство по развёртыванию на Android](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/06_Amlogic_NPU_Android_Deployment_Guide_V0.1.pdf) поясняет взаимосвязь библиотеки, заголовочного файла, драйвера и BSP. `.adla` следует считать артефактом сборки, привязанным к ABI, а не переносимой моделью.

Квантизация также зависит от поколения NPU. В целях ADLA2 001–006 документирована компиляция W8A8 и W8A16. ADLA3 для целей 007 и 008 добавляет сочетания W4A8, W4A16, W8A8, W8A16 и W16A16, а также нативную поддержку INT4, FP16 и BF16 согласно [руководству по операциям](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/05_Amlogic_NPU_Support_Operator_List_V0.1.pdf). Amlogic рекомендует от 200 до 500 репрезентативных примеров для калибровки. Опция случайных данных предназначена именно для проверки производительности, а не для квантизации с сохранением точности.

### Документированный набор моделей Amlogic

Закреплённая [песочница моделей Amlogic](https://github.com/Amlogic-NN/amlnn-model-playground/blob/03ca63ef20c4f1560722de26ab910826884f334a/README.md) даёт более убедительные доказательства, чем общее заявление о поддержке ONNX. В опубликованной матрице поддержки/примеров указаны A311D2, S905X5, A311Y3, C305X2 и A123X. Сам факт принятия компилятором других ID целей не доказывает, что на них проверен весь список моделей.

| Задача | Документированные модели |
|---|---|
| Классификация | MobileNetV2 и ResNet50-v2; DINO на ADLA3 |
| Детекция объектов | PP-YOLOE, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv10, YOLO11, YOLOE, YOLO-World, YOLOX и детекция QR-кодов |
| Лица и жесты | RetinaFace и Gesture Recognition |
| Сегментация | DeepLabV3, PP-LiteSeg, YOLOv5-seg и YOLOv8-seg |
| Детекция с поворотом | YOLOv8 OBB |
| Поза | Детектор/landmark BlazePose и YOLOv8 pose |
| OCR и речь | LPRNet, варианты PaddleOCR, Whisper Tiny и SenseVoice на отдельных новых чипах |
| Новые трансформеры и мультимодальные модели | DETR, MobileSAM, CLIP/MobileCLIP и отдельные примеры LLM/VLM с малой разрядностью на новых платформах |

В том же репозитории опубликованы следующие показатели работы моделей W8A8 в среде выполнения. Это измерения производителя для нейросети на NPU, а не полные показатели обработки изображения с камеры.

| Модель и вход | S905X5 | A311D2 | A311Y3 |
|---|---:|---:|---:|
| YOLOv8n, 640 x 640 | 101.72 FPS | 95.14 FPS | 191.06 FPS |
| YOLOv8s, 640 x 640 | 42.33 FPS | 42.77 FPS | 83.08 FPS |
| YOLOv8m, 640 x 640 | 19.67 FPS | 19.82 FPS | 35.30 FPS |
| YOLOv8l, 640 x 640 | 10.53 FPS | 10.12 FPS | 18.37 FPS |
| YOLO11n, 640 x 640 | 41.14 FPS | 41.48 FPS | 62.24 FPS |

Здесь особенно важны оговорки. В руководстве по операциям Amlogic NMS выполняется программно и для ADLA2, и для ADLA3. В README указано, что это результаты непосредственно работы модели на NPU; предобработка и постобработка не включены. Включены ли все перемещения данных, не сказано. Показатели точности YOLO получены на подвыборках COCO из 300 изображений, а для классификации использован ImageNet `val1000`. Кроме того, таблицы приводят разные значения задержки YOLOv8n на A311Y3 при разных условиях, которые репозиторий не объясняет. Не раскрыты частота, режим мощности, охлаждение, точные версии SDK/BSP и длительность испытаний. Эти показатели помогают понять один стек производителя, но не ранжировать его относительно другого.

### Почему Amlogic стратегически интересна

У Amlogic сочетаются четыре необычных свойства: широкое распространение встроенных SoC, недавно открытый компиляторный стек, актуальная матрица моделей, во многом совпадающая с задачами LibreYOLO, и слабая интеграция в популярные библиотеки обучения. Современные репозитории появились публично в конце 2025-го и начале 2026 года, а содержательные руководства датированы маем–июлем 2026 года. Это создаёт реальную возможность стать одним из первых и одновременно риск нестабильности API.

Надёжная интеграция должна использовать детерминированный экспорт ONNX в LibreYOLO как вход компилятора, требовать репрезентативные калибровочные изображения для сборок с низкой разрядностью, создавать `.adla` вместе с манифестом метаданных и загружать его через адаптер NNSDK2. Для старого A311D нужен явно отдельный артефакт `.nb`: представление `.nb` и `.adla` как одного бэкенда может привести к скрытым ошибкам совместимости. Верхнеуровневые репозитории Amlogic лицензированы по Apache-2.0, но перед размещением в LibreYOLO нужно напрямую проверить права на распространение комплектных wheels, общих библиотек, `nnserver` и Android-бинарников AAR.

## Самые сильные публичные экосистемы развёртывания

У следующих компаний сейчас лучше всего сочетаются доступное оборудование, открытые технические материалы и подтверждения поддержки конкретных моделей. Это не значит, что они всегда быстрее. Их заявления о совместимости проще проверить до покупки оборудования.

### Hailo

Hailo — наглядный пример экосистемы специализированных ускорителей зрения. Модели преобразуются в Hailo Archive, оптимизируются и квантизуются на репрезентативных изображениях, затем компилируются в файл Hailo Executable Format (`.hef`). Приложения загружают этот HEF через HailoRT.

В [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) есть рецепты, предобученные модели, конфигурации постобработки и показатели производительности для детекции, сегментации, классификации, оценки позы и других задач. Среди моделей YOLO — YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 и YOLO26, а также YOLOX, DAMO-YOLO, SSD и EfficientDet. Закреплённая по версии [таблица детекции объектов для Hailo-8](https://github.com/hailo-ai/hailo_model_zoo/blob/v2.19.0/docs/public_models/HAILO8/HAILO8_object_detection.rst) — более надёжный источник совместимости, чем общая страница продукта; в [отдельном приложении](https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/python/standalone_apps/object_detection/README.md) описаны готовые к развёртыванию пайплайны YOLO.

Важно учитывать границу версий. Для Hailo-8 и Hailo-8L используется старая линия Model Zoo 2.x, Dataflow Compiler 3.x и HailoRT 4.x, а для Hailo-10 и Hailo-15 — текущая ветка 5.x. Рецепты, поведение парсера и HEF нельзя считать взаимозаменяемыми только потому, что все устройства называются Hailo.

У Hailo-15 также отдельный уровень приложений. В нём используется Vision Processor Software Package и Hailo Media Library, а не хостовый путь в стиле TAPPAS для Hailo-8/10H. Публичный репозиторий-образец [`hailo-camera-apps`](https://github.com/hailo-ai/hailo-camera-apps) архивирован 3 мая 2026 года, а [`hailo-apps-core`](https://github.com/hailo-ai/hailo-apps-core) остаётся отдельным открытым фреймворком приложений. Архивирование репозитория не доказывает снятие продукта с поддержки, но для проектов на Hailo-15 следует уточнить текущий пакет приложений в Developer Zone.

В [руководстве по развёртыванию Hailo](/docs/export/hailo) LibreYOLO намеренно останавливается на передаче статического ONNX, поскольку проприетарный компилятор нельзя включить как зависимость Python. Так проходит честная граница между подготовкой подходящего графа и заявлением о проверенном HEF.

### Rockchip

Rockchip широко представлен в любительских и коммерческих системах на встроенном Linux, особенно благодаря платам RK3588, RK3576 и RK356x. Публичный [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) преобразует и квантует модели на хосте с x86 Linux, а RKNN Runtime или Toolkit-Lite запускает итоговый `.rknn` на плате.

В [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) необычно подробно описаны чипы, семейства моделей и точность. В текущей таблице указаны пути FP16 и INT8 для YOLOv5/6/7/8/10/11, YOLOX, PP-YOLOE, YOLO-World и вариантов сегментации YOLOv5/v8; для YOLOv8 OBB и YOLOv8 pose указан только INT8. Также есть OCR, лица и другие сети сегментации.

В LibreYOLO уже есть консервативный прямой [экспортёр RKNN](/docs/export/rknn). Он проверяет четыре точных варианта детекции на RK3588, может сравнивать симулятор компилятора с ONNX Runtime и отклоняет неподтверждённые семейства, не считая успешную сборку гарантией корректных предсказаний. Такой ограниченный охват — хороший образец для будущих бэкендов NPU.

### Axelera AI

Axelera AI, название которой часто ошибочно пишут «Accelera», разрабатывает AIPU Metis и продаёт его в устройствах M.2, PCIe и многокристальных продуктах. Семейство Metis можно заказать сейчас; Europa и Titania — анонсированные продукты, поэтому их доступность нельзя представлять одинаково. [Voyager SDK открыт на GitHub](https://github.com/axelera-ai-hub/voyager-sdk) и отвечает за выбор модели, компиляцию, квантизацию, выполнение и пайплайны приложений; поддержка клиентов требует аккаунта.

Публичный [Voyager Model Zoo](https://docs.axelera.ai/sdk/reference/models/model-zoo/) — один из самых информативных в отрасли. В нём перечислены варианты моделей, размеры входа, точность, качество и измеренная производительность для YOLOv3, YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26 n/s/m/l/x, YOLOX, YOLO-NAS, OBB, позы и сегментации, а также множество моделей классификации и плотного предсказания. Указывать рядом с производительностью потерю качества при квантизации полезнее, чем сообщать одно лишь пиковое число TOPS.

Названия артефактов изменились между поколениями API. Альфа-процесс компиляции в [Pipeline Builder](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) создаёт модель `.axm` и может упаковать переносимый пайплайн как `.axe`. В документации [классического пайплайна](https://docs.axelera.ai/sdk/reference/pipeline/model-formats/) описаны `.axmodel`, метаданные модели и манифест. Интеграция должна определять установленное поколение Voyager, а не жёстко задавать одно расширение.

### Qualcomm

Оборудование Qualcomm охватывает огромное число устройств, но за названием «NPU Qualcomm» скрываются разные интерфейсы. Разработчики используют Hexagon HTP через QAIRT/QNN, старый процесс SNPE, сервис компиляции Qualcomm AI Hub, LiteRT или провайдер выполнения QNN в ONNX Runtime — в зависимости от устройства и класса продукта.

Официальный репозиторий [Qualcomm AI Hub Models](https://github.com/qualcomm/ai-hub-models) — сильный источник сведений об отдельных моделях. В нём опубликованы оптимизированные пакеты для YOLOv3, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, детекции/сегментации/позы YOLO26, YOLOX, YOLO-World, YOLOR, RF-DETR и многих моделей классификации, глубины и позы. AI Hub также предоставляет профилирование для конкретных устройств, что важно: модель, скомпилированная для одного поколения HTP, автоматически не переносится на другое.

Основная сложность интеграции — большое число комбинаций: Android или встроенный Linux, заказные компоненты QCS или потребительский Snapdragon, архитектура HTP, версии QNN/QAIRT и схема квантизации. Текущие страницы Qualcomm расходятся в статусе [Dragonwing IQ-9075](https://www.qualcomm.com/internet-of-things/products/iq9-series/iq-9075): на странице продукта указано Active, а оценочный комплект [EVK](https://www.qualcomm.com/developer/hardware/qualcomm-iq-9075-evaluation-kit-evk) доступен для оценки, но в [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program) IQ-9075 всё ещё обозначен как Sampling с поддержкой до 2038 года. Префикс заказа/SKU — QCS9075; Qualcomm заявляет конфигурации 50 и 100 dense INT8 TOPS, а также поддержку Ubuntu/Yocto. Производственный статус следует подтвердить для конкретного SKU заказа, и интеграция LibreYOLO должна записывать именно этот SKU, а не создавать общий каталог «Qualcomm».

### DEEPX

В ускорителях DX-M1/DX-M1M от DEEPX используется SDK DXNN. DX-COM компилирует и квантует модель, DX-RT выполняет её, а развёртываемый артефакт имеет формат `.dxnn`. Компания публикует [DX-AllSuite](https://github.com/DEEPX-AI/dx-all-suite), примеры приложений в [DX-APP](https://github.com/DEEPX-AI/dx_app) и большой [онлайн-каталог моделей](https://developer.deepx.ai/modelzoo/).

На момент проверки 15 августа 2026 года в публичном каталоге было 354 записи для DX-COM 2.4.0 и DX-RT 3.4.0. Документированный набор включает YOLOv3–YOLO11 и YOLO26, YOLOX, SSD, EfficientDet, NanoDet, DAMO-YOLO, сегментацию YOLO, позу и повёрнутые рамки. DEEPX выглядит перспективным кандидатом на интеграцию: граница между компилятором и средой выполнения и формат артефакта чётко определены, а открытый набор моделей компьютерного зрения широк.

## Промышленные SoC — это несколько экосистем, а не одна

Промышленные производители часто предлагают более длительный жизненный цикл продуктов и более тесную интеграцию с камерами, требованиями безопасности и реального времени, чем поставщики плат для энтузиастов. Их AI-стеки могут быть сложнее, поскольку у одного производителя параллельно работают несколько несовместимых архитектур NPU.

### Texas Instruments

К текущим edge-AI-процессорам TI относятся AM62A, AM67A, AM68A, AM69A и связанные устройства TDA4. Программный путь объединяет Processor SDK Linux, компилятор/среду выполнения TIDL, [Edge AI TIDL Tools](https://github.com/TexasInstruments/edgeai-tidl-tools) и [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo). TIDL интегрируется с ONNX Runtime, TensorFlow Lite и другими средами приложений, передавая поддерживаемые подграфы ускорителю.

TI публикует больше, чем просто заявление об импорте во фреймворк: в каталоге есть преобразованные модели и метаданные производительности для детекции объектов, сегментации, позы, классификации, глубины и других задач. TI также предупреждает, что артефакты каталога моделей — это отправные точки разработки, а не автоматически готовые к производству решения. Такое уточнение часто отсутствует в сравнениях производителей.

В качестве предварительной версии TI также указывает [TDA54-Q1](https://www.ti.com/product/TDA54-Q1): до четырёх NPU C7 и до 400 TOPS для этой микросхемы; для более широкого семейства TDA5 заявлено до 1 200 TOPS. Это заявления производителя о новом поколении, но они не дают оснований переносить проверку моделей TIDL на AM6xA/TDA4 на TDA54-Q1, пока TI не опубликует матрицу конкретных целей.

### NXP

Для NXP нужны как минимум четыре отдельных описания бэкендов:

| Цель NXP | Ускоритель | Путь компилятора/среды выполнения |
|---|---|---|
| i.MX 8M Plus | NPU VeriSilicon Vivante на 2.3 TOPS | eIQ с TIM-VX/VX delegate |
| i.MX 93 | Arm Ethos-U65 | Квантизованный TFLite и Arm Vela |
| i.MX 95 | NPU NXP eIQ Neutron | Neutron Converter и среда выполнения eIQ |
| Ara-1 / [Ara240](https://www.nxp.com/products/ARA240) | Дискретный NPU на базе Kinara; для Ara240 заявлено до 40 eTOPS | Ara SDK с интеграцией в общий контур eIQ |

В [eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) есть файлы или рецепты для YOLOv4-tiny, YOLOv8, NanoDet, CenterNet, FastestDet, SSD Lite, YOLACT и других моделей; запись о YOLOv5 добавлена только как документация. Проверка для одного бэкенда не подтверждает все четыре. Руководство NXP по [экспорту YOLO для платформ i.MX](https://community.nxp.com/t5/i-MX-Processors-Knowledge-Base/Exporting-YOLO-Models-for-NXP-i-MX-Platforms/ta-p/2381361) показывает эту зависимость преобразования от целевой платформы. По словам NXP, монолитный [eIQ Toolkit перестал обновляться после версии 1.17 в Q3 2025 года](https://community.nxp.com/t5/eIQ-Machine-Learning-Software/eIQ-FAQ/ta-p/1099741); сейчас используются отдельные пакеты, например eIQ Neutron SDK.

В октябре 2025 года NXP [завершила приобретение Kinara](https://media.nxp.com/news-releases/news-release-details/nxp-completes-acquisitions-aviva-links-and-kinara-advance/). В [спецификации Ara240](https://www.nxp.com/docs/en/fact-sheet/ARA240DNPUFS.pdf) заявлено до 40 eTOPS и 313 изображений в секунду для YOLOv8n. NXP перечисляет Ara SDK и eIQ Toolkit на странице продукта, но это не доказывает взаимозаменяемость артефактов с отдельным [путём Neutron для i.MX 95](https://eiq.nxp.com/learning-hub/convQuant/neutron.html). [Модуль M.2 на 16 GB](https://www.nxp.com/design/design-center/development-boards-and-designs/ARA2-M2-16G-GT) имеет статус Active, тогда как USB-вариант находится в предварительном производстве. NXP расшифровывает «e» в eTOPS как «equivalent», то есть «эквивалентный», а не «эффективный»; это не метрика dense INT8 TOPS другого производителя, поэтому напрямую их сравнивать не следует.

### STMicroelectronics

Устройства [STM32N6x7](https://www.st.com/en/microcontrollers-microprocessors/stm32n6-series.html), включая STM32N657 и STM32N647, добавляют ускоритель Neural-ART на 600 GOPS в MCU-класс; в универсальной линейке N6x5 такого ускорителя нет. [STM32Cube AI Studio](https://www.st.com/en/development-tools/stedgeai-cubeai.html) и ST Edge AI Core анализируют и оптимизируют импортированные модели, а [STM32 AI Model Zoo Services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) предоставляет процессы развёртывания, оптимизации, бенчмаркинга и примеры.

В актуальном [обзоре детекции объектов](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/object_detection/docs/README_OVERVIEW.md) документированы Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26 и ST-YOLOX, а также сервисы классификации, позы и сегментации. Это не тот же класс производительности, что у карты PCIe на 200 TOPS. Платформа интересна тем, что получение изображения с камеры, инференс и управление могут помещаться в жёсткие ограничения по питанию и памяти встраиваемой системы.

### Renesas

В процессорах Renesas RZ/V интегрированы ускорители DRP-AI. Программный стек эволюционировал от DRP-AI Translator и DRP-AI TVM к [RUHMI](https://www.renesas.com/en/software-tool/ruhmi-framework) на базе технологии EdgeCortix MERA. В открытом [репозитории RZ/V DRP-AI TVM](https://github.com/renesas-rz/rzv_drp-ai_tvm) и [списке проверенных моделей RZ/V2H](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) указаны варианты YOLOv5, YOLOv8, YOLO11 и YOLO26 n/s/m, YOLOX, сети для позы и сегментации, а также модели классификации.

Renesas — серьёзный кандидат для робототехники и промышленности, но для воспроизводимости нужно записывать точную плату, поколение DRP-AI, версию Translator и постобработку на CPU.

## Интеллектуальные сенсоры и процессоры для камер

### Sony IMX500

IMX500 от Sony — не обычный прикладной процессор. Сенсор изображения и AI-обработка объединены так, что инференс выполняется прямо в сенсоре и из камеры нужно передавать меньше данных. Raspberry Pi AI Camera делает эту архитектуру необычно доступной.

В официальном [репозитории моделей Raspberry Pi IMX500](https://github.com/raspberrypi/imx500-models) опубликованы пакеты YOLOv8n, YOLO11n, EfficientDet Lite0, NanoDet+ и SSD MobileNetV2 FPN Lite. В [документации AI Camera](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) описаны преобразование, упаковка и постобработка на сенсоре. Результат развёртывания — пакет RPK, а не обычный файл ONNX; основными ограничениями служат память сенсора и поддерживаемые операции. На [странице продукта](https://www.raspberrypi.com/products/ai-camera/) Raspberry Pi обещает выпуск этой камеры как минимум до января 2028 года.

### Ambarella

Процессоры Ambarella CVflow широко используются в камерах, дронах и автомобильных системах компьютерного зрения. К актуальным семействам относятся CV72/CV75 для камер, CV5/CV52 для более производительных систем и автомобильные устройства CV3-AD. Названия программного стека — Cooper developer platform и процесс компиляции/выполнения CVflow.

В [публичном каталоге моделей для разработчиков](https://www.ambarella.com/developer/model-garden/) названы YOLOX, RTMDet, DeepLabV3+, TopFormer, OWL-ViT и мультимодальные модели. Однако подробная документация компилятора и загрузки остаются ориентированными на партнёров. Поэтому интеграцию Ambarella следует начинать с обращения к производителю или OEM-партнёру, а не с предположения, что существует публичный пакет pip.

### Synaptics Astra

У Synaptics есть три значимые цели. В системах Astra SL1600/SL1680 используется [набор SyNAP](https://developer.synaptics.com/docs/synap/introduction), который превращает исходную модель и описание YAML в пакет `model.synap`. В более новых устройствах SL2611/13/15/17/19 применяется [платформа Torq](https://developer.synaptics.com/docs/torq/introduction), основанная на IREE/MLIR и создающая `.vmfb`. Серия [SR100](https://developer.synaptics.com/docs/sr/introduction-sr) — отдельное семейство MCU на Cortex-M55 и Ethos-U55 со своим SDK. Эти артефакты и API несовместимы друг с другом.

В официальном [руководстве по бенчмаркингу YOLO](https://developer.synaptics.com/docs/sl/tutorials/vision/benchmark-yolo) необычно подробно описана процедура: YOLOv8 экспортируется в TFLite, в SyNAP выполняется асимметричная калибровка UINT8, затем модель компилируется для SL1680 и запускается через `synap_cli` и приложение детекции объектов. Руководство подтверждает YOLOv8n/v8s на SL1680 и заявляет поддержку SyNAP вплоть до YOLO11. Отдельное [руководство по детекции объектов SL261x](https://developer.synaptics.com/docs/sl/sl2600/getting-started/object-detection) запускает YOLOv8 из Torq `.vmfb`. Это официальные примеры для конкретных целей, а не заявление, что все графы YOLO поддерживаются во всех трёх семействах.

### MediaTek Genio

MediaTek заслуживает отдельного раздела: в актуальном [IoT AI Hub](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) теперь опубликованы матрица оборудования и ПО, пакеты моделей и таблицы бенчмарков, которых не хватало для проверки в старых обзорах рынка. В Genio 360/360P/420/520/720 используется NeuroPilot 8 с MDLA 5.3; в Genio 510/700 — NP6 с MDLA 3.0; в Genio 1200 — NP6 с MDLA 2.0. MediaTek указывает, что поколение NeuroPilot и MDLA, набор операций, компилятор и среда выполнения привязаны к версии на весь срок жизни каждой SoC.

Аналитический путь AI ориентирован на TFLite:

```text
PyTorch model
    -> NeuroPilot Converter and representative calibration
    -> quantized .tflite
    -> version-matched ncc-tflite compiler
    -> chip-generation-specific .dla
    -> Neuron Runtime on MDLA
```

В новых продуктах Genio также доступны делегирование LiteRT и путь ONNX Runtime CPU/NPU в поддерживаемых ОС. Это другие режимы выполнения, не связанные с офлайн-файлом `.dla`. Различие явно описано в [руководстве по архитектуре ПО](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/software_architecture.html) и [матрице ресурсов](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html).

В официальной [таблице аналитических моделей](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical.html) опубликованы бенчмарки YOLOv5s и YOLOv8s, а также руководства по преобразованию для нескольких поколений MDLA. В ней явно сказано, что готовые преобразованные артефакты YOLO не распространяются из-за ограничений AGPL-3.0. Офлайн-измерения Quant8 при 640 x 640 включают 5.35 ms и 8.04 ms соответственно на Genio 720. На странице модели [YOLOv8s](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical/YOLOv8s.html) опубликованы входные/выходные тензоры, время для разных бэкендов и предупреждения о пользовательских операциях MediaTek. Это бенчмарки производителя, а не результаты полного процесса обработки изображения с камеры.

Ограничивающий фактор — доступ. Публичная документация Yocto подробна, но текущий комплект NP8 «всё в одном» для конвертера/компилятора и многие материалы Android отмечены как ресурсы для прямых заказчиков или NDA. Обычный аккаунт разработчика не даёт того же доступа, что аккаунт клиента MediaTek Online. Поэтому Genio технически проверена, но воспроизводимая интеграция компилятора для собственных моделей зависит от партнёрства.

## Экосистемы edge AI, ориентированные на Китай

В англоязычных обзорах рынка недостаточно представлены несколько мощных и доступных платформ компьютерного зрения с невысокой стоимостью.

### D-Robotics и платформы BPU на базе Horizon

D-Robotics поддерживает экосистему плат разработчика RDK с ускорителями BPU, используемыми в RDK X3, X5, Ultra и новых продуктах серии S100. В текущей ветке `rdk_x5` [каталога моделей RDK](https://github.com/D-Robotics/rdk_model_zoo) описаны пути для RDK X5: YOLOv5/v5u, YOLOv8/9/10/11/12/13/26, YOLOE, YOLO-World, детекция, сегментация, поза, классификация, OCR и CLIP. Для X3 используется отдельная ветка; текущие устройства серии S находятся в основной ветке [`rdk_s`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s); в старом репозитории [`rdk_model_zoo_s`](https://github.com/D-Robotics/rdk_model_zoo_s) лежат исторические демонстрации. Поэтому актуальный список X5 не доказывает, что все модели проверены на X3, Ultra или S100.

OpenExplorer/Algorithm Toolchain импортирует ONNX или Caffe для PTQ и поддерживает потоки QAT для PyTorch. Развёртывание зависит от платформы: в современном RDK X5 применяется `.bin` через `hbm_runtime` поверх `libdnn`; в актуальном основном пути `rdk_s` используется `.hbm` через одноимённый API `hbm_runtime` поверх `libhbucp`; у X3 сохраняются старая ветка и старые интерфейсы инференса. В материалах устаревшего `rdk_model_zoo_s` также описаны прежние пути `.bin` и `.hbm`; такие артефакты нужно связывать с соответствующей веткой и цепочкой инструментов. Неподдерживаемые операции могут выполняться на CPU. Публичные примеры полезны, но совместимость требует правильного сочетания RDK OS, прошивки платы, toolchain и бинарной модели.

### AXERA

Семейства AX650/AX630/AX620 от AXERA используются в компактных AI-камерах и платах вроде линейки MaixCAM от Sipeed. Pulsar2 импортирует ONNX, проводит калибровку и анализ точности, создаёт `.axmodel`; затем артефакт запускается через AXEngine.

В [официальных примерах AXERA](https://github.com/AXERA-TECH/ax-samples) есть YOLOv5/6/7/8/9/10/11/13/26, YOLOX и YOLO-World; набор задач и чипов зависит от платформы. В актуальных примерах YOLO26 для поддерживаемых целей есть детекция, поза, сегментация и OBB. В [примерах преобразования Pulsar2](https://pulsar2-docs.readthedocs.io/en/latest/appendix/model_convert_examples.html) виден объём практической работы: точные имена тензоров, границы выходов, преобразование формата данных, архивы для калибровки и настройки целевого оборудования.

### SOPHGO

TPU-MLIR от SOPHGO — привлекательный стек для независимой интеграции, поскольку сам компилятор имеет открытый исходный код. Он импортирует ONNX, PyTorch, TFLite и Caffe, преобразует и квантует графы, создаёт `.bmodel` для устройств BM или `.cvimodel` для оборудования CV18xx.

Проект [TPU-MLIR](https://github.com/sophgo/tpu-mlir) поддерживает потоки FP32, BF16, FP16 и INT8 в зависимости от цели. В [коллекции демонстраций SOPHON](https://github.com/sophgo/sophon-demo/blob/release/README_EN.md) названы YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, варианты OBB/сегментации, SSD, CenterNet, RetinaFace, SAM/SAM2 и OCR. Демонстрация на BM1684X не подтверждает, что тот же артефакт работает на BM1688 или CV18xx.

### Huawei Ascend

В линейку Huawei для периферийного инференса входят микросхемы Ascend 310/310P/310B в продуктах Atlas 200I и 300I. В CANN есть компилятор ATC, среда выполнения AscendCL и ядра операций для конкретных чипов; ATC преобразует ONNX и другие исходные представления в офлайн-модель `.om`.

В актуальной [документации Atlas 200I DK A2](https://www.hiascend.com/en/hardware/developer-kit-a2/specification) и примерах CANN описан путь запуска YOLOv5 `.om` для Ascend 310B. В более широком, старом [Ascend ModelZoo](https://github.com/Ascend/modelzoo) есть YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN, модели позы и сегментации, но это не значит, что каждая запись заново проверена на 310B. CANN, ядра, прошивка и SoC должны совпадать. `.om` для Ascend310P не является универсальным артефактом Ascend.

### Cambricon

В ускорителях MLU от Cambricon используются Neuware и движок инференса MagicMind. Публичный [репозиторий MagicMind Cloud](https://github.com/Cambricon/magicmind_cloud) закреплён на MagicMind 1.7 и матрице MLU370-X4/S4; это исторические данные о совместимости, а не подтверждение актуального SDK или MLU270. В репозитории описаны пути PyTorch, ONNX, Caffe и Paddle; поддержка фреймворка TensorFlow удалена в 1.7, хотя примеры TensorFlow остались в истории. На текущем [портале разработчика серии 3](https://developer.cambricon.com/index/document/index/classid/3.html) перечислены более поздние общие релизы SDK, поэтому публичные примеры и актуальный коммерческий стек нельзя представлять как одну версию.

Официальный [облачный репозиторий моделей MagicMind](https://github.com/Cambricon/magicmind_cloud) публикует необычно прямую матрицу MLU370-X4/S4. В ней перечислены YOLOv3, Tiny-YOLOv3, YOLOv4, YOLOv5, YOLOv7, YOLOv8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN, DeepLab и UNet, включая наличие примеров C++ или Python. Доказательства убедительны; основная преграда — доступ к SDK и контейнеру через каналы Cambricon.

### Canaan/Kendryte

Современные K230/K230D и устаревшие K210/K510 KPU от Kendryte важны для недорогих плат и интеллектуальных камер. Компилятор [nncase](https://github.com/kendryte/nncase) импортирует ONNX или TFLite, использует калибровочные данные для преобразования в фиксированную точку, позволяет симулировать результат на хосте и создаёт `.kmodel`.

В официальном [руководстве разработки K230 nncase](https://github.com/kendryte/k230_docs/blob/main/en/01_software/board/ai/K230_nncase_Development_Guide.md) пошагово описаны компиляция, симуляция и запуск YOLOv5s. Отдельный [каталог демонстраций AI](https://github.com/kendryte/k230_docs/blob/main/en/02_applications/ai_demos/K230_AI_Demo_Introduction.md) включает готовые артефакты YOLOv8n для детекции, сегментации и позы, а актуальный [журнал изменений CanMV](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md) добавляет оптимизированные примеры классификации, детекции, сегментации, OBB и позы для YOLOv8/11/26. Результат YOLOv5s в [спецификации](https://github.com/kendryte/k230_docs/blob/main/en/00_hardware/K230_datasheet.md) — опубликованный бенчмарк; версии компилятора и бинарного плагина KPU должны соответствовать SDK платы.

### Allwinner

В V853 от Allwinner сочетаются медиаблоки для камер и NPU Vivante на 1 TOPS. В [официальном руководстве NPU на английском языке](https://docs.aw-ol.com/v853/en/npu/dev_npu/) описаны импорт через Acuity/Pegasus, квантизация, проверка и развёртывание через `viplite`. В собственных страницах моделей Allwinner и [руководстве YOLOv5](https://docs.aw-ol.com/v853/npu/npu_yolov5/) перечислены YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet и сети для лиц/людей.

Это реальная поддержка компьютерного зрения, но она привязана к старой цепочке Tina Linux/Vivante, а загрузки SDK требуют аккаунта. Платформа входит в обзор рынка с этой оговоркой.

## Менее заметные платформы Кореи, Тайваня и Китая

Англоязычные обзоры часто перескакивают от Rockchip к NVIDIA, пропуская второй ряд реальных чипов с документацией. В некоторых из этих экосистем матрицы актуальных YOLO шире, чем у более известных западных стартапов.

### Mobilint

Корейская компания Mobilint выпускает маломощный REGULUS и более производительный ARIES MLA100. SDK qb принимает модели PyTorch, TensorFlow, TFLite, ONNX и Keras, квантует их до INT8 и создаёт скомпилированный артефакт `.mxq`. В публичной [матрице компьютерного зрения](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md) перечислены детекторы YOLOv3/v5/v7/v8/v9/v10/11/12/26 и варианты YOLO для сегментации, позы и OBB. На [странице ARIES](https://www.mobilint.com/aries/mla100) опубликованы отдельные результаты для YOLO11s и YOLO26m. Это весомое свидетельство для интеграции, хотя доступ к SDK и оборудованию остаётся коммерческим.

### Sunplus

В SP7350/C3V от Sunplus используется NPU-стек на базе Vivante, но его пакет отличается от Allwinner и старой Amlogic. В открытом процессе сочетаются преобразование Acuity, Docker для NPU, среда выполнения SNNF и выходной файл `.nb`. В официальном [руководстве по YOLOv8](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350) показаны преобразование и развёртывание, а не просто заявлена поддержка фреймворка. Родственное IP не делает `.nb` переносимым на другую SoC с Vivante.

### ESWIN Computing

В линейку RISC-V от ESWIN входят EIC7700/EIC7700X и двухкристальные EIC7702/EIC7702X; EIC7700 установлен в серийные платы, например Milk-V Megrez. В ENNP входят EsQuant, актуальный компилятор EsAAC, создание эталонных данных, симуляция и среда выполнения ESSDK; скомпилированный результат — `.model`, а в старой документации ENNP использовалось название `ennc-compile`. В актуальном [руководстве EsAAC](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) описан ввод ONNX, поэтому для других фреймворков понадобится экспорт или преобразование в поддерживаемое IR. [Обзор ENNP](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/introduction) и [руководство YOLOv3](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3) на сайте Milk-V подтверждают публичный путь от начала до конца, но не дают широкой актуальной матрицы моделей/точности.

### Nuvoton M55M1

M55M1 от Nuvoton — конструкция класса MCU на Cortex-M55 и Ethos-U55-256, а не прикладной процессор Linux. NuEdgeWise/NuML и Arm Vela готовят квантизованные модели TFLite Micro. В публичном [репозитории NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise) названы YOLOv8-nano, YOLOX-nano, YOLO Fastest v1.1, SSD-MobileNet FPNLite и несколько классификаторов. Эти примеры компактных сетей соответствуют ограничениям памяти и мощности устройства, но не подтверждают стандартные варианты YOLO с разрешением 640 пикселей.

### Rebellions и FuriosaAI

В линейке ATOM от Rebellions используются SDK RBLN и скомпилированные артефакты `.rbln`. В [официальном релизе поддержки](https://docs.rbln.ai/v0.8.2/supports/release_note.html) названы YOLOv3 tiny/full/SPP, YOLOv5 от n до x, YOLOv6 от n до l, варианты YOLOv7 и YOLOv8 от n до x. В актуальной [матрице поддержки карт](https://docs.rbln.ai/latest/supports/version_matrix.html) ATOM CA02 и ATOM+ CA12 имеют статус EoL, а ATOM+ CA22 и ATOM-Max CA25 — Active. У ATOM-Lite CA21 есть страница продукта, но его нет в матрице SDK, поэтому текущий статус набора инструментов неясен. Документация открыта, но для загрузки компилятора нужны учётные данные Rebellions Portal.

FuriosaAI нужно рассматривать по поколениям. Warboy — старый ускоритель компьютерного зрения с INT8-артефактами `.enf` и [официальным каталогом моделей](https://developer.furiosa.ai/furiosa-models/latest/), в котором есть SSD, YOLOv5M/L и YOLOv7-w6-pose. Для RNGD используется другой [стек `.fxb`](https://developer.furiosa.ai/latest/en/furiosa_llm/fxb.html), в первую очередь рассчитанный на LLM/VLM. В актуальной [дорожной карте](https://developer.furiosa.ai/latest/en/overview/roadmap.html) поддержка компьютерного зрения YOLOv8m отмечена как завершённая в 2024 Q4, однако текущие публичные сведения о моделях и производительности сосредоточены на LLM/VLM и не содержат подробной матрицы точности/производительности YOLOv8m. Это свидетельство для выпущенного поколения, а не подтверждение совместимости с Warboy.

### Realtek, Telechips, T-Head и SigmaStar

Эти четыре платформы реальны, но возможности для собственных моделей уже, устарели или ограничены доступом:

| Платформа | Воспроизводимые публичные свидетельства | Ограничение, которое нужно учитывать |
|---|---|---|
| Realtek AmebaPro2 | [Arduino/FreeRTOS SDK](https://github.com/Ameba-AIoT/ameba-arduino-pro2) поставляет модели YOLOv3/4/7-tiny, SCRFD и MobileFaceNet `.nb` | Для преобразования пользовательской модели нужно заявить о своём интересе/связаться с Realtek |
| Telechips TOPST TCC7500 | В официальном [проекте 2026 года](https://docs.topst.ai/blog/31) преобразована и запущена YOLOv8s | Преобразование UFLD v1/v2 не удалось из-за неподдерживаемых слоёв; предложено только обходное решение с разделением/постобработкой. Для полных ресурсов компилятора и операторов часто нужно письменное разрешение |
| T-Head TH1520 | В [руководстве по приложениям Sipeed](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) YOLOv5n/s работает с HHB и CSI-NN2/SHL | Стек открыт, но обслуживание и ясность версий отстают от лидеров |
| SigmaStar SSU9383CM | Текущая документация описывает API среды выполнения MI_IPU; старая документация SGS_IPU — преобразование `.sim` в `sgsimg.img`, а старый постпроцессор называет SSD и YOLOv1/2/3 | Нет публичных данных, что старые артефакты компилятора или модели подходят для SSU9383CM |

### Интеллектуальные системы зрения HiSilicon — это не Huawei Ascend

В камерных SoC Hi3516CV610/DV500, Hi3519DV500 и Hi3403V100 от HiSilicon используется процесс ATC в `.om` с средами выполнения NNN/SVP-NNN. В официальном [зоопарке моделей HiSpark](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) для отдельных целей, особенно Hi3403 и Hi3591P, указаны YOLOv3/4/5/6/7/8/9/10/11, сегментация/поза YOLO11, OBB/World/сегментация YOLOv8, OCR, глубина и TinySAM. Это не подтверждает работу всех моделей на всех перечисленных SoC для компьютерного зрения. Некоторые записи относятся к планам развития; выпущенные примеры и планируемую поддержку нужно разделять. В репозитории указано, что предоставленные модели ModelZoo предназначены только для некоммерческого использования. Совпадение названий ATC/`.om` не доказывает совместимость артефактов или среды выполнения с линейкой Ascend на CANN.

## Новые и специализированные компании-разработчики ускорителей

### MemryX

Модули MX3 от MemryX используют архитектуру потоковой обработки данных и могут объединять несколько чипов. [Neural Compiler](https://developer.memryx.com/tools/neural_compiler.html) принимает модели ONNX, TensorFlow Lite, Keras и TensorFlow и создаёт пакет DFP, который обрабатывает среда выполнения ускорителя. [API среды выполнения](https://developer.memryx.com/api/accelerator/accelerator.html) поддерживают несколько моделей и потоковые пайплайны.

В документации и примерах MemryX есть оптимизированная предобработка и постобработка YOLO для детекции, сегментации и позы. В [заметках к релизам SDK](https://developer.memryx.com/release_notes.html) явно указана поддержка YOLOv10, YOLO11 и YOLO26. Публичные материалы полезны технически, хотя в них нет одной удобной таблицы моделей/точности/устройств, сравнимой с каталогом Axelera.

### Kneron

Продукты KL520/KL530/KL630/KL720/KL730 от Kneron охватывают небольшие USB- и встроенные ускорители. Kneron PLUS обеспечивает среду выполнения приложения; актуальная версия Model Toolchain 0.33.1 описывает преобразование, квантизацию, оценку, симуляцию и компиляцию `.nef` для этих целей. KL830 упоминается в некоторых [API среды выполнения PLUS](https://doc.kneron.com/docs/plus_c/introduction/run_examples/), но отсутствует в текущем [списке целей компилятора](https://doc.kneron.com/docs/toolchain/manual_1_overview/). Без подтверждения производителя нельзя переносить на KL830 общее заявление о компиляции `.nef`.

В официальном [примере YOLO](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/) показан процесс на Tiny-YOLOv3; в других материалах Kneron описаны обучение и развёртывание YOLOv5. Свидетельства убедительны, но охватывают меньше моделей, чем широкие актуальные матрицы YOLO у Hailo, Axelera, Rockchip или DEEPX.

### SiMa.ai

В MLSoC и [серийной платформе Modalix на 50 TOPS](https://sima.ai/press-release/sima-ai-next-gen-platform-for-physical-ai-in-production/) от SiMa.ai используется программная среда Palette. ModelSDK, компилятор MLA и ModelExecutor охватывают импорт, квантизацию, компиляцию, профилирование и выполнение. В зависимости от задачи и продукта доступны потоки INT8, INT16 и BF16.

В заметках к релизам названы проверенные пайплайны YOLOv7, YOLOv8 для детекции/позы/сегментации, YOLOX и YOLOX для сегментации, DETR, Mask R-CNN и EfficientDet. Не следует скрывать одно текущее ограничение: в заметках [Palette SDK 2.1](https://docs.sima.ai/v2.1.2/pages/release_notes/2.1.html) сказано, что QAT не работает из-за регрессии квантизации Python/PT2E. Обходной путь — создать ONNX с аннотациями в SDK 2.0 и скомпилировать его в 2.1. Доступ и закупка рассчитаны на корпоративных клиентов.

Граница развёртывания также документирована: [процесс компиляции ModelSDK](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) создаёт скомпилированный `.tar.gz` и может создать исполняемый ELF, а [MPK Tool](https://docs.sima.ai/v2.1.1/pages/palette/mpk_tools.html) упаковывает развёртываемый `.mpk`. Эти выходные файлы привязаны к цели MLSoC/Modalix и релизу Palette.

### EdgeCortix

SAKURA-II — заявленный ускоритель на 60 TOPS для компьютерного зрения и генеративного AI, компилируемый через программный фреймворк MERA. EdgeCortix также предоставляет технологии MERA для более нового процесса RUHMI от Renesas.

В материалах о [SAKURA-II](https://www.edgecortix.com/en/edgecortix-sakura-ii-accelerator-brief) описаны оборудование и компилятор; [оборудование M.2/PCIe](https://www.edgecortix.com/en/hardware) предлагается через запрос на тестирование или заказ. Достаточно точную актуальную публичную матрицу совместимости YOLO найти не удалось. Для принятия решения о покупке это важная информация: до выбора оборудования следует запросить проверку нужной модели.

### BrainChip

Akida от BrainChip — нейроморфный процессор событийной области, а не обычный NPU для плотных тензоров. В среде [MetaTF](https://brainchip.com/metatf-dev-tools/) используются устанавливаемые пакеты Python для создания моделей, квантизации до низкой разрядности, преобразования, симуляции и выполнения на оборудовании. Сейчас BrainChip продаёт [доступный сопроцессор AKD1500 в формате M.2](https://brainchip.com/brainchip-akd1500-now-available-in-compact-m-2-form-factor-enabling-fanless-edge-ai-in-industrial-and-commercial-designs/), а также старое оборудование AKD1000 и IP Akida 2.

В актуальной [карточке модели Akida 2](https://brainchip.com/wp-content/uploads/2025/04/Akida-2-Model-Card-V1.1-Mar.25.pdf) названы детектор AkidaNet0.5 YOLOv2, CenterNet, AkidaUNet и сети распознавания лиц. В других материалах о поддержке упоминаются FOMO и задачи на событиях. В Akida поддерживаются очень низкие разрядности весов и активаций, но для развёртывания может потребоваться преобразование с учётом архитектуры — считать его обычной целью INT8 ONNX нельзя. Результат для AKD1000 или Akida 2 из каталога нельзя приписывать AKD1500, пока явный отчёт map/fit не подтвердит соответствие.

### Blaize

Blaize продаёт продукты Pathfinder и Xplorer на базе P1600; программный путь обеспечивают Picasso SDK, NetDeploy и AI Studio. На [страницах продуктов](https://www.blaize.com/products/) ясно указаны компьютерное зрение и периферийный AI, но актуальной публичной матрицы совместимости по отдельным моделям нет.

Поэтому здесь Blaize отнесена к коммерческим решениям с закрытым доступом, а пробел не заполняется заявлениями сообщества. Перед покупкой следует потребовать от производителя отчёт компиляции и точности для целевой модели.

## TinyML и зрение на конечных устройствах

### Arm Ethos-U и Alif

Arm лицензирует производителям чипов IP NPU Ethos-U55, U65 и U85. Компилятор [Vela](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) принимает квантизованные графы LiteRT/TFLite и переписывает поддерживаемые подграфы в пользовательские операции Ethos-U. Неподдерживаемые операции могут остаться на CPU, поэтому «приложение запускается» и «вся сеть работает на NPU» — разные утверждения.

Среди реальных устройств — Ethos-U55 в Alif Ensemble E7, Ethos-U65 в NXP i.MX 93 и Ethos-U85 в более новых системах. В [E7 AI/ML AppKit](https://alifsemi.com/support/kits/ensemble-e7appkit/) от Alif документированы два ML-ускорителя и показан путь TFLite-to-Vela для E7; в [Ensemble E8](https://alifsemi.com/ensemble-e8-series/) сочетаются один Ethos-U85 и два NPU Ethos-U55. Alif также публикует семейный [бенчмарк детектора лиц YOLO-Fastest INT8](https://alifsemi.com/faster-ai-mcu-inferencing-low-power-consumption/) при разрешении 192 на 192 на Cortex-M55 с Ethos-U55, но не широкую матрицу современных YOLO для отдельных целей. Эти системы с ограниченной памятью подходят для небольших сетей классификации и детекции; не следует считать, что в них поместится любой детектор YOLO на 640 пикселей только потому, что он представлен в TFLite.

### Infineon PSOC Edge, Himax WiseEye2 и Analog Devices MAX7800x

В семействах [PSOC Edge E83](https://documentation.infineon.com/psocedge/docs/xsk1761304020519) и E84 от Infineon Cortex-M55 сочетается с Ethos-U55; на стороне M33 добавлен отдельный блок NNLite. В [руководстве по архитектуре E84](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) описаны веса на 8 бит, активации на 8 или 16 бит и процесс Vela, при котором поддерживаемые операции превращаются в поток команд NPU, а неподдерживаемая работа остаётся на CPU. [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter) принимает входы TFLite, Keras и PyTorch; в материалах об архитектуре MobileNetV1/V2 указаны как типовые ядра, а не бенчмарки цели. В [AI Kit E84](https://documentation.infineon.com/psocedge/docs/cci1762693051052) есть камера, набор инструментов — ModusToolbox, но публичные материалы пока не содержат матрицу совместимости конкретных моделей YOLO. Это реальная программируемая платформа зрения с развивающимися свидетельствами по отдельным моделям, но не проверенная универсальная цель для детекторов.

В [HX6538 WiseEye2](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) от Himax также сочетаются Cortex-M55 и Ethos-U55 для постоянно активного зрения. Публичное описание ПО необычно подробно для такого класса энергопотребления: в официальных [примерах Grove Vision AI Module V2](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) есть детекция объектов, поза и классификация пола на YOLOv8n, детекция на YOLO11n, face mesh и PeopleNet. Единый процесс использует TFLite Micro и Vela, а затем при необходимости — CMSIS-NN и эталонные ядра. Это намеренно компактные модели с небольшими разрешениями, а не доказательство того, что поместится обычный большой граф YOLO.

Analog Devices использует другой подход с [CNN-ускорителями MAX78000/MAX78002](https://www.analog.com/en/products/max78002.html). Публичный инструмент [`ai8x-synthesis`](https://github.com/analogdevicesinc/ai8x-synthesis) квантует обученную сеть и создаёт C-код для конкретной цели, веса и конфигурацию ускорителя, а не переносимый пакет среды выполнения ONNX. К свидетельствам производителя относятся [идентификация лиц](https://www.analog.com/en/resources/app-notes/an-2616.html), детекция QR на TinierSSD в [зоопарке моделей ADI](https://ez.analog.com/dsp/software-and-development-tools/edgebench/a/docs-faqs/DF912/available-models-in-adi-model-zoo) и классификаторы изображений. Устройства подходят для периферийного зрения, но актуальную официальную матрицу современных YOLO найти не удалось.

### GreenWaves GAP9

В GAP9 сочетаются вычисления RISC-V и нейронный движок NE16. NNTool импортирует и квантует сети, AutoTiler и GAP SDK генерируют код для развёртывания. В открытом [меню нейросетей GAP9](https://github.com/GreenWaves-Technologies/nn_menu_gap9) есть MobileNet, EfficientNet, ResNet, MobileNet SSD, детекция лиц и приложения распознавания. GreenWaves также публикует проект [детекции людей YOLOX](https://github.com/GreenWaves-Technologies/yolox_people_detection) с результатами точности и симуляции.

Это необычно прозрачные свидетельства для TinyML, но полный SDK доступен только квалифицированным заказчикам, а модели существенно меньше распространённых вариантов YOLO 640 на 640.

### Lattice sensAI

Lattice sensAI — основанная на FPGA альтернатива фиксированному NPU. sensAI Studio и Neural Network Compiler размещают поддерживаемый граф на настраиваемом IP ускорителя для устройств ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E и Avant-X. Результат привязан к выбранной FPGA, режиму ускорителя, раскладке памяти и битстриму, а не является переносимой моделью среды выполнения.

В актуальном [руководстве Neural Network Compiler](https://www.latticesemi.com/-/media/LatticeSemi/Documents/UserManuals/MQ3/FPGA-UG-02052-8-0-Lattice-Neural-Network-Compiler-Software.ashx?document_id=52343) есть необычно прямая таблица архитектур. В ней YOLOv1 указан для ECP5, YOLOv5 и YOLOv8 — для оптимизированных/расширенных режимов CrossLink-NX и CertusPro-NX, а YOLO11 — только для Advanced mode IP. Также перечислены SSD, MobileNetV2-SSD, ResNet, ENet, SqueezeDet и другие сети компьютерного зрения; неподдерживаемые сочетания показаны отдельно для семейств FPGA. [Advanced CNN Accelerator](https://www.latticesemi.com/en/Products/DesignSoftwareAndIP/IntellectualProperty/IPCore/IPCores05/Advanced-CNN-Accelerator-IP) предназначен для Avant-E/Avant-X/CertusPro-NX и является коммерческим IP. Это свидетельства из таблицы компилятора, а не обещание, что один битстрим одновременно поддерживает все перечисленные архитектуры.

### Microchip VectorBlox

Публичный [VectorBlox SDK 3.1](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) от Microchip выполняет предобработку и компиляцию полностью квантизованных сетей TFLite INT8 для настраиваемого IP CoreVectorBlox в FPGA PolarFire SoC. `tflite_preprocess` и `vnnx_compile` создают файлы для развёртывания `.vnnx`, `.hex` и `.ucomp`; в репозитории также есть симулятор и целевой драйвер C. Графы TensorFlow, ONNX и OpenVINO сначала должны пройти документированный путь подготовки TFLite/INT8.

В актуальной [матрице руководств](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md) указаны YOLOv5n, YOLOv8n для детекции, классификации, OBB, позы и сегментации, YOLOv9t, разреженные/сжатые варианты YOLO, MobileNet, EfficientNet-Lite0, ResNet18, MiDaS, FFNet и QuickSRNet. В [руководстве по постобработке на C](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/docs/C_Postprocessing.md) отдельно названы декодеры детекции, позы и OBB для YOLOv2/3/4/5 и Ultralytics. Сейчас SDK 3.1 поддерживает PolarFire SoC Video Kit и явно не поддерживает PolarFire Video Kit без SoC, поэтому старые демонстрации плат нельзя выдавать за актуальный контракт целей.

SDK можно публично загрузить по [лицензии Microchip](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/LICENSE.md), которая ограничивает использование ПО и производных продуктовами Microchip. Microchip предлагает бесплатные лицензии Libero Silver и CoreVectorBlox по запросу. Для развёртывания всё равно нужен системный контракт FPGA: конфигурация ускорителя, битстрим, прошивка, данные сети, созданные SDK, и раскладка памяти должны совпадать. Успешная компиляция VectorBlox не превращает модель в бинарный файл, переносимый на любую конструкцию PolarFire.

### Syntiant

NDP200 от Syntiant предназначен для постоянно активного инференса зрения, звука и сенсорных данных с энергопотреблением ниже обычных SoC под Linux. В актуальной [таблице оборудования](https://www.syntiant.com/hardware) NDP200 имеет статус серийного производства, NDP250 — проходит выборку. В [описании NDP200](https://www.syntiant.com/ndp200/) указана поддержка CNN, RNN и полносвязных сетей при мощности менее 1 mW; в [анонсе NDP250](https://www.syntiant.com/news/syntiant-unveils-ndp250-neural-decision-processor-with-next-gen-core-3-architecture) говорится о постоянно активном распознавании изображений при мощности менее 30 mW. Значение «менее милливатта» нельзя обобщать на оба чипа.

Публичные материалы не подтверждают широкую совместимость с YOLO, поэтому продукт следует оценивать для небольших постоянно активных классификаторов и детекторов через поддерживаемую производителем оценку модели, а не полагаться на общий экспорт ONNX.

### Google Coral: два не связанных друг с другом поколения

В прежних продуктах Edge TPU используются Edge TPU Compiler, `libedgetpu` и PyCoral с полностью квантизованными моделями TFLite INT8. Основные репозитории [Edge TPU](https://github.com/google-coral/edgetpu) и [PyCoral](https://github.com/google-coral/pycoral) архивированы. Это реальный риск для сопровождения, но не официальное уведомление о снятии оборудования с производства.

Google также поддерживает отдельный активный проект с открытым исходным кодом [Coral NPU](https://github.com/google-coral/coralnpu): IP ускорителя RISC-V для интеграции в маломощные SoC. В публичном проекте доступны RTL, симуляция оборудования, набор инструментов и примеры ELF. Это не прямая замена Edge TPU USB/M.2; актуальная матрица развёртывания YOLO не опубликована.

## Особенности смежных GPU, клиентских NPU и адаптивных вычислений

NVIDIA, AMD, Intel и Apple часто встречаются в поиске NPU, но они не предлагают один взаимозаменяемый тип ускорителя.

**NVIDIA Jetson:** Jetson Orin объединяет GPU CUDA с отдельными ядрами DLA. TensorRT может создать сериализованный движок для DLA, однако неподдерживаемые слои выполняются на GPU только при явном включении GPU fallback. Актуальная [таблица DeepStream для YOLO](https://github.com/NVIDIA-AI-IOT/deepstream_tools/blob/main/yolo_deepstream/README.md) охватывает YOLOv4/v7/v8/v9/11, но большинство записей рассчитаны на GPU; явная конфигурация ONNX DLA — YOLOv8s INT8. В [ограничениях DLA](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html) от NVIDIA объясняются ограничения операций. Thor тоже отличается: в [руководстве по миграции DriveOS](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/NVIDIA_DriveOS_7.0.3_Migration_Guide.pdf) сказано, что в Thor ядра DLA удалены ради гибкого распределения работы между GPU. Формулировка «работает на Jetson» не указывает, какой ускоритель выполняет модель.

**AMD Vitis AI:** В старшем поколении Kria/DPU скомпилированные графы `.xmodel` выполнялись через VART. Текущая версия [Vitis AI 6.2](https://vitisai.docs.amd.com/en/6.2/) имеет статус GA для двух отдельных путей Versal AI Edge: [Gen1](https://vitisai.docs.amd.com/projects/gen1/en/latest/index.html) на VEK280/VE2802 использует ONNX, AMD Quark и привязанный к цели процесс снимков/подграфов; у [Gen2](https://vitisai.docs.amd.com/projects/gen2/en/latest/index.html) на VEK385 свой контракт компиляции и среды выполнения. Для Gen1 опубликованы примеры YOLOv5, YOLOv7, YOLOv8 и YOLOX. Ryzen AI — четвёртый, отдельный клиентский стек NPU. Не переносите артефакты и результаты проверок между устаревшим DPU, Versal Gen1, Versal Gen2 и Ryzen AI только из-за общего названия Vitis или AMD.

**Intel OpenVINO:** OpenVINO может работать с CPU, GPU и NPU Core Ultra через один API. В [документации плагина Intel NPU](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) указаны поддерживаемые поколения NPU, а в [матрице проверенных моделей](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html) есть отдельные столбцы устройств. Ноутбук с YOLO показывает рецепт приложения, а не проверку на NPU, если это не указано в матрице. Полноту выполнения графа определяют поддерживаемые операции, формы, точность и установленный драйвер NPU. IR OpenVINO (`.xml` и `.bin`) — повторно используемое IR исходного уровня; кэш скомпилированной модели зависит от устройства и ПО.

**Apple Core ML:** `coremltools` преобразует модели в `.mlpackage`, который Xcode компилирует в `.mlmodelc`. Core ML может распределять работу между CPU, GPU и Apple Neural Engine, но специально скрывает точное разбиение. Это делает Apple отличной платформой развёртывания, но заявлять, что «весь граф YOLO выполнен на NPU», нельзя без подтверждения профилированием.

## Компании — поставщики NPU IP для производителей чипов

Некоторые компании вообще не выпускают платы или готовые ускорители. Они лицензируют конструкции NPU поставщикам SoC. Это важно, поскольку одна базовая архитектура может встречаться под разными марками чипов, но итоговые SDK и артефакты всё равно могут быть изменены лицензиатом.

| Поставщик IP | Семейство NPU и ПО | Почему это важно |
|---|---|---|
| [Arm](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) | Ethos-U55/U65/U85 и Vela | Используется в NXP, Alif, Infineon, Himax и других встраиваемых продуктах; путь рассчитан на квантизованный TFLite/TOSA |
| [Arm China](https://www.armchina.com/mountain?infoId=161&name=) | AIPU Zhouyi и Compass SDK | В публичном [зоопарке моделей](https://github.com/Arm-China/Model_zoo) названы YOLOv1-tiny/v2/v3/v4/v5, YOLOX и YOLOv8-seg; это эталонные модели, а не доказательство для каждого чипа лицензиата |
| [VeriSilicon](https://www.verisilicon.com/en/IPPortfolio/VivanteVIP9000Pico) | Семейство Vivante VIP и SDK Acuity | Родственные семейства Vivante NPU входят в несколько SoC, в том числе в отдельно закупаемые варианты A311D и i.MX 8M Plus; каждый производитель чипа интегрирует их по-своему |
| [Imagination Technologies](https://www.imaginationtech.com/products/open-access/) | PowerVR Series3NX AX3146/AX3386/AX3596; историческая IMG Series4; Neural Compute SDK/IMG DNN | Лицензируемый IP NNA, а не розничная плата; Open Access предлагает конфигурации Series3NX на 1/5/10 TOPS, а программа NC-SDK от производителя называет PP-YOLOE, EfficientNet и HRNet, но не все конфигурации IP |
| [CEVA](https://www.ceva-ip.com/product/ceva-neupro-studio/) | NeuPro-Nano/NeuPro-M и NeuPro Studio | Лицензируемый NPU IP с импортом, квантизацией, сжатием, компиляцией графа, симуляцией и генерацией C/C++ |
| [MIPS](https://mips.com/processor-solutions/arc-npx-family/) | ARC NPX6 и [MetaWare MX](https://mips.com/processor-solutions/arc-metaware-mx/), ранее Synopsys ARC | Масштабируемый NPU IP и SDK нейросетей, приобретённые GlobalFoundries и переданные в портфель MIPS в [июне 2026 года](https://mips.com/press-releases/gfmipsarcclose/); не розничная цель развёртывания |
| [Cadence](https://www.cadence.com/en_US/home/tools/silicon-solutions/ai-ip-platform/neuroweave-sdk.html) | Neo NPU, DSP Tensilica и NeuroWeave SDK | Стек компилятора/интерпретатора для лицензиатов SoC; сети и квантизация зависят от лицензированной конфигурации |
| [Quadric](https://quadric.ai/npu-ip) | IP GPNPU Chimera и SDK | Программируемый IP в стиле NPU/DSP, лицензируемый производителям других чипов; конечная цель — их кремний |
| [Expedera](https://www.expedera.com/products-overview/) | IP NPU Origin и программный стек | Лицензируемая архитектура периферийного NPU, а не плата для прямой покупки LibreYOLO |

Imagination показывает, почему рядом с названием архитектуры следует указывать доступ и жизненный цикл. Её программа [Open Access](https://www.imaginationtech.com/products/open-access/) предлагает три проверенные на кремнии конфигурации Series3NX без первоначальной платы за лицензию IP, но после оценки компании, с платной поддержкой/обслуживанием и роялти за серийный выпуск. [Neural Compute SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) предоставляет инструменты компиляции, оптимизации, квантизации и среды выполнения; в официальном [сотрудничестве с Baidu PaddlePaddle](https://www.imaginationtech.com/news/imagination-and-baidu-paddlepaddle-create-open-source-machine-learning-library-for-model-zoo/) названы детекция PP-YOLOE, классификация EfficientNet и сегментация HRNet. Эти примеры не подтверждают каждую конфигурацию Series3NX или Series4; текущие страницы отдельных [продуктов Series4](https://www.imaginationtech.com/product/img-4nx-mc1/) помечают их как недоступные, поэтому Series4 следует считать историческим решением, пока продажи не подтвердят обратное.

Этот уровень объясняет внешнее сходство семейств, но не делает артефакты переносимыми. У двух SoC с родственными IP Vivante или Ethos могут отличаться память, драйверы, версии компилятора, наборы операций и среды выполнения производителя.

## Что развёртывается: таблица привязки к артефактам

Последний файл в цепочке компилятора — место, где видимость переносимости ONNX заканчивается. Названия и расширения со временем меняются, но практическое правило остаётся: сохраняйте исходную модель и данные калибровки, поскольку нативный артефакт обычно нельзя перенести на другое поколение NPU или надёжно пересобрать в другой версии SDK.

| Стек производителя | Типичный вход компилятора | Что попадает на устройство | От чего зависит совместимость |
|---|---|---|---|
| [Amlogic AMLNN](https://github.com/Amlogic-NN/amlnn-toolkit) | ONNX, TFLite, TorchScript или PT2 | `.adla`; для старого A311D — `.nb` | ID цели, поколение ADLA, сборка компилятора AMLNN, NNSDK2/среда выполнения, драйвер ADLA и BSP |
| [Rockchip RKNN](https://github.com/airockchip/rknn-toolkit2) | ONNX, TFLite или экспорт фреймворка | `.rknn` | Версии RKNN Toolkit/Runtime и семейство NPU Rockchip |
| [Hailo](https://hailo.ai/developer-zone/documentation/) | ONNX или TensorFlow через промежуточный HAR | `.hef` | Архитектура Hailo и поколения компилятора/среды выполнения |
| [Axelera Voyager](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) | ONNX или поддерживаемое описание каталога | `.axm` в альфа-версии Pipeline Builder, пакет пайплайна `.axe`; классический `.axmodel` с манифестом | Поколение API Voyager и конфигурация цели Metis |
| [DEEPX DXNN](https://github.com/DEEPX-AI/dx-all-suite) | ONNX и поддерживаемые пути фреймворков | `.dxnn` | Версии DX-COM/DX-RT и цель DX-M |
| [Qualcomm QAIRT/QNN или SNPE](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | ONNX, TFLite или модель фреймворка | Библиотека/бинарный контекст QNN или SNPE `.dlc` | Архитектура HTP, SoC, бэкенд и версия среды выполнения |
| [MediaTek NeuroPilot](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) | Преобразованный/квантизованный TFLite | `.dla` для офлайн Neuron Runtime; `.tflite` для режима делегирования | Поколение NP/MDLA, образ ОС, компилятор и среда выполнения |
| [TI TIDL](https://github.com/TexasInstruments/edgeai-tidl-tools) | ONNX или TFLite | Каталог импортированных артефактов и файлы модели приложения | TIDL Tools/Runtime и поколение процессора |
| [NXP eIQ](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | Обычно TFLite или ONNX | Результат для выбранного бэкенда: Vela, TIM-VX, Neutron или Ara | Выбранные ускоритель i.MX/Ara и BSP, а не NXP в целом |
| [ST Edge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) | TFLite, ONNX или поддерживаемая модель фреймворка | Сгенерированная библиотека/код сети и файлы прошивки | Целевая MCU/NPU, конфигурация памяти и версия инструментов |
| [Infineon PSOC Edge](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) | Целочисленно-квантизованный путь TFLite/Keras/PyTorch | Оптимизированный Vela TFLite с командным потоком Ethos-U и прошивкой приложения | Вариант E83/E84, конфигурация Ethos-U, Vela, ModusToolbox и BSP; NNLite — отдельная цель |
| [Renesas DRP-AI](https://github.com/renesas-rz/rzv_drp-ai_tvm) | ONNX/TFLite через TVM или Translator | Каталог модели среды выполнения с несколькими бинарными файлами данных | Устройство RZ/V и поколение Translator/Runtime |
| [Sony IMX500](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) | Квантизованная/упакованная сеть из процесса Edge-MDT | Пакет `.rpk` | Прошивка сенсора, лимит памяти и версия упаковщика |
| [Synaptics](https://developer.synaptics.com/) | TFLite/ONNX и поддерживаемые форматы импорта | `.synap` в SyNAP; `.vmfb` в Torq | Поколение ПО/оборудования SL16xx или SL261x |
| [Ambarella CVflow](https://www.ambarella.com/developer/) | Вход закрытой цепочки инструментов партнёра | Пакет развёртывания для партнёра; точный формат публично не документирован | Уточнить поколение CVflow, SDK/BSP и конвейер камеры через Cooper Developer Zone |
| [D-Robotics BPU](https://github.com/D-Robotics/rdk_model_zoo) | ONNX и граф, поддерживаемый toolchain | X5 `.bin`; текущий `rdk_s` `.hbm`; у X3 старый путь | Поколение BPU, ветка репозитория и соответствующий релиз OpenExplorer/Runtime |
| [AXERA Pulsar2](https://github.com/AXERA-TECH/ax-samples) | ONNX | `.axmodel` | Целевая платформа, версия Pulsar2 и AXEngine |
| [SOPHGO TPU-MLIR](https://github.com/sophgo/tpu-mlir) | ONNX, PyTorch, TFLite или Caffe | `.bmodel` или `.cvimodel` | Целевой чип BM/CV и версия TPU-MLIR/среды выполнения |
| [Huawei Ascend CANN](https://www.hiascend.com/en/software/cann) | ONNX или поддерживаемый граф фреймворка | `.om` | Чип Ascend, CANN/ATC и прошивка/драйвер устройства |
| [Cambricon MagicMind](https://github.com/Cambricon/magicmind_cloud) | Граф фреймворка или формат обмена | Сериализованный движок/модель MagicMind | Архитектура MLU и версии Neuware/MagicMind |
| [Canaan nncase](https://github.com/kendryte/nncase) | ONNX или TFLite | `.kmodel` | Поколение KPU и соответствующий целевой модуль nncase |
| [Mobilint qb](https://www.mobilint.com/sdk-qb) | ONNX и поддерживаемые модели фреймворков | `.mxq` | Цель REGULUS/ARIES и компилятор/среда выполнения qb |
| [Rebellions RBLN](https://docs.rbln.ai/latest/index.html) | PyTorch 2 или граф с поддержкой TensorFlow | `.rbln` | Поколение ATOM и компилятор/среда выполнения RBLN |
| [FuriosaAI](https://developer.furiosa.ai/docs/latest/en/) | ONNX/TFLite и поддерживаемый путь фреймворка | Warboy `.enf`; RNGD `.fxb` | Совершенно разные поколения ускорителя и SDK |
| [Sunplus SNNF](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2004353133) | Сеть, поддерживаемая Acuity | `.nb` | Драйвер NPU SP7350, среда выполнения и BSP |
| [ESWIN ENNP](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) | ONNX в документированном актуальном потоке EsAAC; другие фреймворки требуют экспорта/преобразования | `.model` | Точная цель семейства EIC7700 и релиз ENNP/ESSDK |
| [Nuvoton / Ethos-U](https://github.com/OpenNuvoton/NuEdgeWise) | Квантизованный TFLite | Оптимизированный Vela TFLite с пользовательскими операциями Ethos-U | План памяти M55M1, Vela и прошивка |
| [Himax WiseEye2](https://github.com/HimaxWiseEyePlus/YOLOv8_on_WE2) | Полностью целочисленно-квантизованный TFLite | Оптимизированный Vela `.tflite` во флеш-памяти модели плюс прошивка платы, например `output.img` | Конфигурация HX6538/WE2, Vela, TFLite Micro, драйвер Ethos-U и резервные ядра |
| [Analog Devices AI8X](https://github.com/analogdevicesinc/ai8x-synthesis) | Чекпойнт PyTorch, YAML сети и входной пример | Сгенерированные C-исходники/заголовки для устройства, веса и скомпилированная прошивка | Ограничения памяти/слоёв MAX78000 или MAX78002, synthesis tool и релиз MSDK |
| [Realtek AmebaPro2](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | Запрос/вход для сервиса преобразования производителя | `.nb` | Прошивка RTL8735B и версия VoE/NeuralNetwork API |
| [Telechips Enlight](https://docs.topst.ai/product/p/ai) | Путь Darknet, TensorFlow, ONNX или PyTorch | Промежуточный `.enlight` и пакет развёртывания | NPU TCC7500 и версии TC-NN/Enlight |
| [T-Head HHB](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | ONNX и поддерживаемый граф фреймворка | `hhb.bm`, сгенерированные параметры/код и исполняемый файл | Стек CSI-NN2/SHL для TH1520 |
| [SigmaStar MI_IPU](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | В актуальных документах SSU9383CM описана среда выполнения MI_IPU; в старых материалах используются входы семейств TensorFlow/Caffe | Старый `.sim`, преобразованный в `sgsimg.img`; актуальный артефакт публично не подтверждён | Конкретный IPU SigmaStar и поколение SDK; совместимость старого компилятора с SSU9383CM не проверена |
| [Интеллектуальное зрение HiSilicon](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) | ONNX и граф, поддерживаемый ATC | `.om` | Конкретная SoC для зрения и стек NNN/SVP-NNN; не общий перенос между Ascend |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | ONNX, TFLite, Keras или TensorFlow | Пакет потока данных `.dfp` | Конфигурация оборудования MX и релизы среды выполнения/компилятора |
| [Kneron](https://doc.kneron.com/docs/) | ONNX и настройки toolchain | `.nef`; KL730 NEFv2 может включать `.kne` | Документированная цель компилятора, поколение чипа, прошивка и среда выполнения PLUS; компиляция для KL830 не подтверждена |
| [SiMa.ai Palette](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) | Поддерживаемая модель фреймворка или ONNX | Скомпилированный `.tar.gz` ModelSDK, описанный исполняемый ELF или пакет MPK Tool `.mpk` | Цель MLSoC/Modalix и релиз Palette |
| [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/) | ONNX или описание сети | Сериализованный движок, обычно `.engine` или `.plan` | Архитектура GPU/DLA, TensorRT, CUDA и часто конкретное устройство |
| [AMD Vitis AI](https://vitisai.docs.amd.com/en/6.2/) | Квантизованный граф ONNX/фреймворка | Устаревший `.xmodel`; снимок NPU/подграфы Gen1; каталог кэша компиляции или производственный пакет `.rai` для Gen2 | Точное поколение/конфигурация NPU IP, образ платформы и матрица Vitis/Vivado/PetaLinux |
| [Microchip VectorBlox](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) | Полностью квантизованный TFLite INT8 | `.vnnx`, `.hex`, `.ucomp` с совместимой прошивкой/конструкцией FPGA | Конфигурация CoreVectorBlox, PolarFire SoC Video Kit, SDK 3.1, битстрим и раскладка памяти |
| [Intel OpenVINO](https://docs.openvino.ai/) | Модель фреймворка или ONNX | IR `.xml` и `.bin` или кэш для конкретного устройства | IR можно переиспользовать; скомпилированный кэш зависит от устройства, драйвера и OpenVINO |
| [Устаревший Google Edge TPU](https://coral.ai/docs/edgetpu/models-intro/) | Полностью INT8 TFLite | Скомпилированный для Edge TPU `_edgetpu.tflite` | Компилятор/среда выполнения Edge TPU и поддерживаемые квантизованные операции; не связан с новым IP Coral NPU |
| [IP Google Coral NPU](https://github.com/google-coral/coralnpu) | Примеры C/C++ для актуального публичного IP-проекта | Примеры ELF для RTL/симуляции оборудования и будущей интеграции в SoC; публичного артефакта компилятора YOLO нет | Точная лицензированная/интегрированная реализация SoC и развивающийся набор инструментов с открытым кодом |
| [Lattice sensAI](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | Поддерживаемое описание сети | Битстрим FPGA, конфигурация ускорителя и веса | Семейство FPGA, режим IP sensAI и реализация памяти |
| [Imagination NC-SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) | Caffe, TensorFlow или ONNX | Имя скомпилированного артефакта для заказчика публично не указано | Лицензированная конфигурация NNA, интеграция SoC, NC-SDK/DDK и среда выполнения лицензиата |

Эта таблица — не просто перечень расширений. Версии компилятора, идентификатор цели, драйвер, прошивка и BSP входят в воспроизводимую идентичность модели. В будущих экспортёрах LibreYOLO нужно записывать их в машиночитаемый манифест рядом с каждым артефактом.

## Доступ к инструментам и сигналы жизненного цикла

Доступность оборудования и доступ к компилятору — разные вещи. Плату может быть легко купить, хотя для текущего компилятора нужно соглашение с заказчиком; открытый SDK может предназначаться для кремния, который трудно достать. Далее приведены конкретные сигналы производителей, а не универсальный рейтинг срока службы.

| Платформа | Документированный сигнал | Что это значит на практике |
|---|---|---|
| Amlogic ADLA | Репозитории Apache-2.0 открыты, пакеты wheels можно скачать; совместимые драйверы плат/BSP всё ещё могут потребовать обращения к Amlogic или производителю платы | Компилятор легко оценить, но права на распространение бинарников и полную матрицу совместимости следует уточнить |
| MediaTek Genio | Открыты IoT AI Hub и руководства Yocto; комплект NP8 «всё в одном» и документация Android помечены как ресурсы прямых заказчиков/NDA | Модели можно изучить по открытым данным, но автоматизация для собственных моделей может потребовать партнёрства |
| Hailo | Открыты каталог моделей и среда выполнения; Dataflow Compiler распространяется через Developer Zone; публичный `hailo-camera-apps` архивирован 3 мая 2026 года | Широко изучать модели можно открыто, но воспроизводимая компиляция требует подходящего аккаунта/версии; архивация не доказывает завершение жизненного цикла Hailo-15, нужно уточнить текущий пакет приложений vision processor |
| Axelera AI | Открыты документация и [Voyager SDK](https://github.com/axelera-ai-hub/voyager-sdk); поддержка клиентов требует аккаунта | Редкий ускоритель с SDK для самостоятельной оценки, но поддержка продукта и покупка остаются коммерческими |
| Qualcomm Dragonwing | В [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program) IQ-9075 обозначен Sampling с поддержкой до 2038 года, а QCS6490 — до июля 2036 года; при этом на странице IQ-9075 указан Active | Хороший сигнал для промышленного планирования, но расхождение источников требует проверки конкретного SKU и не гарантирует стабильность ABI одного AI SDK на весь период |
| Rebellions | В актуальной [матрице поддержки](https://docs.rbln.ai/latest/supports/version_matrix.html) CA02 и CA12 помечены EoL, а CA22/CA25 — Active; CA21 отсутствует | Закупка и совместимость с SDK зависят от карты, даже внутри семейства ATOM |
| NVIDIA Jetson | В [официальной таблице жизненного цикла модулей](https://developer.nvidia.com/embedded/lifecycle) для модулей Orin указан срок до января 2032 года, для AGX Orin Industrial — до июля 2033-го; у комплектов разработчика обязательств по сроку нет | Для производственных систем выбирайте модули, а не ориентируйтесь на наличие отладочного комплекта |
| Sony IMX500 в Raspberry Pi | В [описании AI Camera](https://datasheets.raspberrypi.com/camera/ai-camera-product-brief.pdf) указан выпуск как минимум до января 2028 года | Конкретный минимум для этого модуля камеры, не для всех продуктов IMX500 |
| Amlogic A311Y3 | На [странице запуска 2026 года](https://www.amlogic.com/News/index248.html) обещана гарантия жизненного цикла продукта как минимум на десять лет | Перспективный сигнал для новой платформы, но для продуктовой программы нужны условия договора и детали SKU |
| Google Coral | Старые репозитории Edge TPU и PyCoral архивированы/доступны только для чтения; отдельный репозиторий [Coral NPU IP](https://github.com/google-coral/coralnpu) активен | Риск сопровождения старого ПО сам по себе не означает официальное снятие оборудования с производства и ничего не говорит о жизненном цикле отдельного проекта IP |

## Почему TOPS — не руководство к покупке

Пиковые TOPS полезно сравнивать в рамках одного производителя и одной архитектуры, но сравнения между компаниями обычно некорректны, если не совпадает всё перечисленное:

- Числовая точность, включая INT8, INT4, FP16 или смешанный режим
- Считается ли одна операция умножения с накоплением как одна или две операции
- Плотная или структурно-разреженная арифметика
- Разрешение входа, размер батча и граф модели
- Точность после квантизации
- Задержка только NPU или всего пайплайна приложения
- Перемещения данных и откат на хост
- Тепловое состояние и длительная, а не кратковременная работа

Детектор — это пайплайн: получение изображения, изменение размера/letterbox, нормализация, передача на устройство, нейронный инференс, декодирование, NMS, масштабирование координат и логика приложения. Производители часто публикуют показатели только для середины процесса — инференса нейросети. [MLPerf Inference Edge](https://mlcommons.org/benchmarks/inference-edge/) полезен тем, что задаёт сценарии, целевые показатели качества и правила измерения системы вместо сравнения маркетинговой арифметики.

Для развёртываний YOLO минимальная запись о заслуживающем доверия бенчмарке выглядит так:

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

Без этих данных два показателя FPS обычно относятся к разным условиям.

## Как выбрать NPU для компьютерного зрения

Выбирайте в следующем порядке, а не по самому большому числу на странице продукта.

1. **Проверьте совместимость графа.** Скомпилируйте точную экспортированную модель, а не похожий чекпойнт из каталога.
2. **Измерьте точность.** После калибровки и квантизации проверьте скомпилированный артефакт на реальном датасете задачи.
3. **Измерьте полный процесс.** Учтите преобразование входа, передачи данных, декодирование и NMS.
4. **Проверьте доступ к ПО.** Выясните, открыт ли компилятор, требует ли регистраций или доступен только по коммерческому соглашению.
5. **Зафиксируйте матрицу версий.** Запишите компилятор, среду выполнения, драйвер, прошивку и идентификаторы цели.
6. **Проверьте поддержку ОС на практике.** Android, Debian, Yocto, Buildroot, RTOS и Windows не взаимозаменяемы.
7. **Уточните поставки и жизненный цикл.** Технически отличный чип не подойдёт для производства, если неясны наличие модулей, драйверов или долгосрочной поддержки.
8. **Затем сравнивайте цену, мощность и производительность.** Эти измерения имеют смысл, только если одна и та же модель корректно работает на обеих целях.

### С чего начать в зависимости от задачи

| Сценарий | Сначала стоит оценить | Причина |
|---|---|---|
| Ускоритель специально для Raspberry Pi | Hailo-8L/8 и Sony IMX500 | Официальные продукты Raspberry Pi, образы ПО и конкретные процессы для разработчиков |
| Универсальное дополнение для Linux по PCIe, M.2 или USB | Продукты DEEPX, MemryX, Hailo и Axelera | Доступные форм-факторы ускорителей при условии совместимости с хостом/драйвером |
| Недорогая Linux SBC или камера | Rockchip RK3588/RK3576, Amlogic ADLA при доступности поддерживаемой платы/BSP, AXERA, Canaan K230 или Sunplus SP7350 | Встроенные медиапайплайны и открытые свидетельства о моделях/компиляторах; для NPU A311D2 на VIM4 нужна версия V13A+ |
| Мобильные устройства и массовые Android-продукты | Qualcomm QNN/AI Hub, LiteRT, Apple Core ML | Большая установленная база и поддерживаемые мобильные среды выполнения |
| Промышленный Linux и робототехника | TI TDA4/AM6xA, NXP i.MX, Renesas RZ/V, Hailo | Встраиваемые продукты с долгим сроком службы и экосистемами камер/ввода-вывода |
| Очень маломощное конечное устройство или MCU | STM32N6x7, Infineon PSOC Edge, Himax WiseEye2, Analog Devices MAX7800x, Nuvoton M55M1, Alif/Arm Ethos-U, GAP9 и Syntiant | Жёсткие ограничения питания и памяти, специализированные инструменты и ограничения на размер моделей |
| Настраиваемая FPGA для зрения | Microchip VectorBlox, Lattice sensAI и цели AMD Vitis AI | Гибкие аппаратные пайплайны, но конфигурация ускорителя, битстрим и компилятор модели образуют одну систему с версиями |
| Высокопроизводительное зрение по PCIe | Axelera Metis, Hailo, DEEPX, Mobilint, MemryX, Rebellions или SiMa.ai | Специализированные ускорители и расчёт на несколько потоков |
| Экосистема продуктов с фокусом на Китай | Rockchip, AXERA, D-Robotics, SOPHGO, Huawei Ascend, HiSilicon и Cambricon | Сильные региональные платы, toolchain и примеры моделей |

Это исследовательский список для оценки, а не рейтинг производительности. Закупки, доступность в регионе и конкретная модель могут изменить порядок.

## Что это значит для LibreYOLO

Масштабируемая архитектура — не десятки несвязанных экспортёров, а один строгий контракт обмена и небольшие адаптеры компилятора и среды выполнения для производителей:

```text
LibreYOLO model
    -> deterministic static ONNX
    -> VendorCompiler.compile(model, target, calibration, precision)
    -> native artifact + manifest
    -> VendorRuntime.load() / infer()
    -> shared task-specific decode and Results objects
```

К каждому нативному артефакту нужен сопроводительный манифест с такими полями:

- Производитель, цель-чип и целевая плата
- Версии компилятора, среды выполнения, драйвера и прошивки
- Контрольные суммы исходного чекпойнта и ONNX
- Имена входов, формы, раскладка, цветовое пространство, нормализация и тип данных
- Точность квантизации, масштабы, если доступны, и хеш калибровочных данных
- Имена выходных тензоров, формы и семантическое значение
- Задача детекции, названия классов, обработка decode/NMS на чипе или хосте
- Зафиксированные результаты численного соответствия и точности на задаче

В LibreYOLO уже доступны переносимый [экспорт ONNX](/docs/export/onnx), [инструменты квантизации](/docs/export/quantization), прямой, но намеренно ограниченный [экспорт RKNN](/docs/export/rknn) и честный процесс работы с внешним компилятором [Hailo](/docs/export/hailo). По критериям этой статьи следующим сильным кандидатом на интеграцию будет Amlogic ADLA: набор инструментов открыт, поддержка моделей во многом совпадает с LibreYOLO, а старый путь A311D можно оставить явно отдельным.

Следующий приоритет после Amlogic следует определять по оборудованию пользователей и возможности проверить точность, а не только по доступности компилятора. DEEPX, Sony IMX500, D-Robotics, AXERA, SOPHGO и Qualcomm технически привлекательны; TI, NXP, ST и Renesas особенно ценны при наличии у промышленных партнёров плат и долгосрочного доступа к CI.

## Список наблюдения: реальный кремний без открытого контракта интеграции

Полное исключение компании может исказить обзор рынка. Но назвать её «поддерживаемой» ещё хуже. Эти производители продают, поставляют на тестирование или анонсировали подходящее оборудование, однако открытых данных пока недостаточно, чтобы подтвердить воспроизводимую текущую цепочку компилятора, артефакта и именованной модели для бэкенда LibreYOLO.

| Компания | Что существует | Почему остаётся в списке наблюдения |
|---|---|---|
| [Samsung](https://semiconductor.samsung.com/processor/automotive-processor/exynos-auto-v920/) | В Exynos Auto V920 есть двухъядерный NPU с заявленной производительностью до 23.1 TOPS | Samsung сообщает, что [Neural SDK больше не предоставляется сторонним разработчикам](https://developer.samsung.com/neural/overview.html); Samsung ONE/Circle не доказывает доступ к NPU Exynos |
| [Novatek](https://www.novatek.com.tw/en-global/Milestone/aboutus_milestones) | В актуальных корпоративных этапах названы SoC Edge AI и Edge Vision/Imaging AI; в этапе 2020 года упоминались MobileNet, SSD и YOLOv3 | Нет текущей публичной матрицы номеров деталей/компилятора/артефактов/операций/моделей; прежний YOLOv3 не подтверждает актуальный SDK |
| [Nextchip](https://www.nextchip.com/en/adas/adas.php?idx=5) | Автомобильные процессоры APACHE5/NVS2900 и актуальные [APACHE6/NVS3000](https://www.nextchip.com/en/adas/adas.php?idx=7) с NPU aiWare от aiMotive; на странице APACHE6 сейчас указаны и 12, и 8 TOPS | Существуют aiWare Studio и среда выполнения, но публичного формата артефакта, руководства квантизации, списка операций или матрицы YOLO не найдено; расхождение цифр производителя нельзя молча разрешать |
| [Black Sesame Technologies](https://bst.ai/en.html) | Автомобильные/периферийные AI-чипы Huashan A1000/A2000 и Wudang серии C | Есть публичная информация о продуктах, но нет компилятора для самостоятельного использования, контракта артефакта или воспроизводимого каталога моделей |
| [Ingenic](https://en.ingenic.com.cn/products-detail/id-19.html) | Камерные SoC T41/T40 с заявлениями о низкой разрядности NPU; в Magik AI описаны PTQ/QAT и компиляция графа | Не удалось найти текущий загружаемый компилятор, контракт артефакта или точный список моделей YOLO от производителя |
| [Fullhan](https://fullhan.com/en/index.php?a=type&c=article&tid=9) | Для нескольких SoC IPC заявлены NPU на 0.5–2 TOPS, для [семейства NVR MC6880](https://fullhan.com/en/index.php?a=type&c=article&tid=48) — 4 TOPS | Недостаточно публичной документации NN compiler/runtime/framework/model для независимого воспроизведения |
| [Goke Microelectronics](http://www.gokemicro.com/News/info.aspx?itemid=437) | На странице производителя для деталей камер GK7606V1/GK7206V1/GK7203V1 заявлена производительность встроенного NPU; на момент публикации страница доступна только по устаревшему HTTP | Нет открытого конвертера, контракта среды выполнения или матрицы именованных моделей; доступ ориентирован на OEM-канал |
| [Chengheng Micro](https://en.chenghengmicro.com/) | Для CH37 заявлено 64 INT8 TOPS и режимы FP16/FP32/FP64; в [хронологии компании за 2026 год](https://chenghengmicro.com/about.html) говорится о начале мелкосерийного производства | Публичного SDK, контракта компилятора или проверки конкретных моделей не найдено; это ранняя коммерческая платформа без самообслуживания, а не воспроизводимое развёртывание |
| [Bouffalo Lab](https://github.com/bouffalolab/bouffalo_sdk) | В BL808 есть NPU BLAI-100 и актуальные официальные репозитории SDK | В поддерживаемом SDK нет актуального официального пути преобразования/примеров BLAI; сохранившиеся процессы устарели или основаны на материалах сообщества |

Этот список наблюдения опирается на доступные свидетельства. Компания может перейти в таблицы развёртываемых решений, если опубликует актуальный toolchain или способ оценки, контракт артефакта для конкретной цели и хотя бы одну именованную модель со сквозным рецептом.

## Чего эта статья намеренно не утверждает

Статья не заявляет, что каждая указанная модель запускается из любого исходного репозитория. Модели в каталогах производителя часто изменены, разделены на других выходных узлах или связаны с пользовательской постобработкой.

Также здесь не выдаются за подтверждение поддержки следующие сведения:

- Компилятор импортирует ONNX.
- Для чипа заявлен Android-драйвер NNAPI.
- Дистрибьютор говорит, что плата «совместима с YOLO».
- Репозиторий сообщества запускает один форк одной модели.
- Модель компилируется без ошибки.
- Производитель сообщил FPS только для NPU и не указал точность.

Ускорители Google Tensor в телефонах и множество заказных ASIC только для OEM тоже существуют, но Google не предлагает Tensor как общедоступную для самостоятельной работы цель компиляции, сопоставимую с платформами выше. Наличие компании, блок-схемы NPU или ускорения Android недостаточно, чтобы заявлять об интеграции LibreYOLO.

Облачные ускорители Google TPU, AWS Inferentia/Trainium, Microsoft Maia и AI-карты только для центров обработки данных также не входят в сферу этой статьи. Речь идёт об оборудовании, которое разработчик компьютерного зрения может реалистично развернуть на периферии.

Лицензирование модели — отдельный уровень. Публикация рецепта преобразования YOLO производителем чипа не даёт прав использовать или распространять исходные веса, код обучения или производную скомпилированную модель. Для целевого продукта нужно отдельно проверить поддержку оборудования, лицензии SDK и лицензии моделей.

## Методология исследования и исправления

Для каждой компании проверялись четыре вида первичных источников:

1. Актуальная страница продукта или спецификация с описанием кремния.
2. Документация компилятора и среды выполнения с реальным путём развёртывания.
3. Каталог моделей, таблица совместимости или сквозное руководство с названиями моделей зрения.
4. Сведения о жизненном цикле и доступе, показывающие, может ли разработчик получить инструменты.

Официальные организации GitHub считались источниками производителя. Материалы производителя платы помечены как свидетельства экосистемы, если сам производитель чипа не публикует те же сведения. Заявления третьих сторон о производительности исключены из сравнительных таблиц.

После подготовки черновик отдельно проверили по группам поставщиков и по пересекающимся таблицам. Проверки повторно сверили охват целей, имена артефактов, точность, свидетельства о модели и постпроцессоре, статусы анонсированных и выпускаемых продуктов, жизненный цикл, знаменатели бенчмарков и количество каждой сущности. Если первичные страницы расходятся, статья показывает это расхождение, а не выбирает удобное значение без объяснения.

Этот рынок быстро меняется. Если вы работаете в одной из компаний и в статье неверно указаны чип, SDK, список моделей или статус доступа, отправьте LibreYOLO точную ссылку на публичную документацию и её версию. Исправления, подкреплённые первичными свидетельствами, должны заменить этот текст; неподтверждённые маркетинговые заявления — нет.

## FAQ

### Сколько компаний выпускают edge AI NPU?

Универсального числа нет: пересекаются производители продуктов, лицензиары IP, интеллектуальные сенсоры, GPU и закрытые автомобильные ASIC. В этом неисчерпывающем обзоре отслеживаются 69 компаний и экосистем платформ, связанных с edge AI: чипы, платформы ускорителей, лицензируемый NPU IP или заслуживающий внимания кремний по состоянию на август 2026 года.

### Что такое NPU?

Neural Processing Unit — специализированное аппаратное устройство для операций нейронных сетей, например свёрток и умножения матриц. Производители используют термины NPU, AIPU, BPU, KPU, DLA, HTP, TPU и MLA. Скомпилированные модели обычно нельзя переносить между компаниями.

### Какие компании с NPU официально поддерживают модели YOLO?

У Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Sony IMX500 через официальный репозиторий Raspberry Pi AI Camera, STMicroelectronics, Renesas, Lattice, Himax, Microchip и ряда других компаний есть собственные записи в каталогах моделей, руководства или проверенные результаты как минимум для одного поколения YOLO. Важно учитывать конкретное поколение, задачу, целевой чип и версию SDK.

### Можно ли запустить любую модель ONNX на любом NPU?

Нет. ONNX — формат обмена, а не гарантия совместимости с оборудованием. Компилятор NPU должен поддерживать все операции, формы тензоров и типы данных графа. Часто встречаются статические формы, разбиение графа, пользовательские операции и откат на CPU.

### Какой NPU лучше всего подходит для YOLO?

Универсального лидера нет. У Hailo, Rockchip, Axelera AI, DEEPX и Amlogic хорошо документирована поддержка YOLO, а устройства Qualcomm отличаются очень широким распространением. Выбирайте по точности после квантизации, задержке всего пайплайна, длительной мощности, цене, доступности, доступу к SDK и поддержке ОС.

### Почему показатели NPU в TOPS нельзя напрямую сравнивать?

Производители могут учитывать разные точности, разреженные операции, операции умножения с накоплением и допущения о пиковой загрузке. TOPS не учитывает трафик памяти, неподдерживаемые операции, предобработку, постобработку и накладные расходы хоста. Сравнивайте одинаковые модель, точность, разрешение, качество и полный пайплайн.

### Что такое калибровка NPU?

При калибровке через модель пропускают репрезентативные, обычно неразмеченные изображения для развёртывания, чтобы компилятор оценил диапазоны активаций для INT8 или другого формата с низкой точностью. Даже случайные или нерепрезентативные изображения могут дать скомпилированный артефакт, но ухудшить точность на задаче.

### Поддерживает ли LibreYOLO периферийные NPU?

LibreYOLO экспортирует ONNX и несколько форматов среды выполнения, напрямую компилирует выбранные модели Rockchip через RKNN и описывает внешнюю процедуру компиляции для Hailo. Для всех остальных артефактов производителя требуется его SDK; до компиляции, проверки и запуска на оборудовании их следует считать кандидатами на интеграцию.
