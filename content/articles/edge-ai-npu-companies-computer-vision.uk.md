---
title: "69 компаній Edge AI NPU: чипи, SDK і YOLO (2026)"
description: "Посібник із джерелами про 69 компаній Edge AI та екосистеми NPU: чипи, SDK, артефакти, квантування, задокументована підтримка комп'ютерного зору та YOLO."
date: 2026-08-15
author: Xuban
tags: [edge-ai, npu, computer-vision, yolo, hardware, ai-accelerators, amlogic, hailo, rockchip]
faq:
  - q: "Скільки існує компаній Edge AI NPU?"
    a: "Універсальної кількості компаній немає, оскільки постачальники продуктів, ліцензіари IP, розумні сенсори, GPU та приватні автомобільні ASIC перетинаються. Цей невичерпний посібник охоплює 69 компаній і платформних екосистем, пов'язаних із edge AI чипами, платформами прискорення, ліцензованим IP NPU або перспективними чипами зі списку спостереження станом на серпень 2026 року."
  - q: "Що таке NPU?"
    a: "Нейронний процесор (NPU) є спеціалізованим обладнанням для операцій нейронних мереж, як-от згортки та множення матриць. Постачальники використовують назви NPU, AIPU, BPU, KPU, DLA, HTP, TPU та MLA, а скомпільовані моделі зазвичай не можна переносити між компаніями."
  - q: "Які компанії NPU офіційно підтримують моделі YOLO?"
    a: "Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Sony IMX500 через офіційний репозиторій Raspberry Pi AI Camera, STMicroelectronics, Renesas, Lattice, Himax, Microchip та кілька інших мають власні записи в каталогах моделей, посібники або валідовані результати щонайменше для одного покоління YOLO. Конкретне покоління, завдання, цільовий чип і версія SDK мають значення."
  - q: "Чи може будь-яка модель ONNX працювати на будь-якому NPU?"
    a: "Ні. ONNX є форматом обміну, а не гарантією апаратної сумісності. Компілятор NPU має підтримувати кожен оператор, форму тензора й тип даних у графі. Часто трапляються статичні форми, розбиття графа, користувацькі операції та резервне виконання на CPU."
  - q: "Який NPU найкраще підходить для YOLO?"
    a: "Універсального переможця немає. Hailo, Rockchip, Axelera AI, DEEPX і Amlogic мають добре задокументовану підтримку YOLO, а Qualcomm вирізняється надзвичайно широким охопленням пристроїв. Обирайте з огляду на правильність після квантування, затримку всього пайплайна, тривалу потужність, ціну, доступність, доступ до SDK і підтримку операційної системи."
  - q: "Чому показники NPU у TOPS не можна порівнювати безпосередньо?"
    a: "Постачальники можуть рахувати різну точність, розріджені операції, операції множення з накопиченням і припущення щодо пікового завантаження. TOPS не враховує трафік пам'яті, непідтримувані оператори, попередню та післяобробку й накладні витрати хоста. Порівнюйте ту саму модель, точність, роздільну здатність, правильність і весь пайплайн."
  - q: "Що таке калібрування NPU?"
    a: "Під час калібрування репрезентативні, зазвичай неанотовані зображення для розгортання пропускають через модель, щоб компілятор міг оцінити діапазони активацій для INT8 або іншого формату з низькою точністю. На випадкових або нерепрезентативних зображеннях артефакт може скомпілюватися, але точність завдання погіршиться."
  - q: "Чи підтримує LibreYOLO edge NPU?"
    a: "LibreYOLO експортує ONNX і кілька форматів середовищ виконання, має прямий шлях компіляції RKNN для окремих моделей Rockchip і документує зовнішній процес компіляції Hailo. Для кожного іншого артефакту у форматі постачальника все одно потрібен його SDK; до компіляції, валідації та запуску на обладнанні такий варіант слід вважати кандидатом на інтеграцію."
---

**Єдиного ринку NPU немає. Цей невичерпний посібник охоплює 69 компаній і платформних екосистем: 51 постачальника чипів або платформ для розгортання, дев'ять постачальників IP NPU та дев'ять чітко позначених компаній зі списку спостереження.** Вони охоплюють кілька несумісних класів прискорювачів і майже стільки ж стосів компіляторів. Більшість обіцяє той самий шлях: експортувати модель, квантувати її, скомпілювати, скопіювати пропрієтарний артефакт на плату й запустити середовище виконання постачальника. Саме деталі визначають, чи цей шлях займе пів дня або квартал роботи інженерів.

Цей посібник описує компанії, що надають надійне обладнання для комп'ютерного зору у 2026 році. У ньому наведено відповідні чипи, назви SDK і компіляторів, артефакти для розгортання, рівень публічного доступу та моделі, які постачальник справді документує. Окрему увагу приділено Amlogic, чий новіший стек ADLA суттєво відрізняється від старішої екосистеми NPU A311D.

> **Обсяг дослідження:** джерела перевірено 15 серпня 2026 року. Твердження про конкретні моделі спираються на власні сторінки продуктів, документацію, каталоги моделей, репозиторії або посібники постачальників, якщо не зазначено інше. Заявлені TOPS є показниками постачальників, а не незалежно зіставними результатами бенчмарків. Це посібник для розробників, а не інвестиційна порада чи платний рейтинг.

**Швидка навігація:** [таблиці компаній](#npu-companies-and-platforms-at-a-glance) | [детальний огляд Amlogic](#amlogic-two-npu-generations-not-one) | [профілі постачальників](#the-strongest-public-deployment-ecosystems) | [таблиця скомпільованих артефактів](#what-you-actually-deploy-the-artifact-lock-in-table) | [доступ і життєвий цикл](#tool-access-and-lifecycle-signals) | [план LibreYOLO](#what-this-means-for-libreyolo)

## Головний висновок

Апаратне забезпечення фрагментоване, але схема розгортання напрочуд послідовна:

```text
PyTorch checkpoint
    -> ONNX or TFLite interchange graph
    -> representative calibration data
    -> vendor quantizer and graph compiler
    -> chip-specific binary or model package
    -> vendor runtime on the target
    -> application preprocessing, decode, NMS and rendering
```

[ONNX прямо підтримує середовища виконання, генератори коду й апаратні реалізації](https://onnx.ai/onnx/repo-docs/IR.html), але файл ONNX є лише точкою передачі. Це не означає, що кожен оператор ONNX можна перетворити для кожного прискорювача. Те, що справді працюватиме на NPU, визначають статичні форми вхідних даних, обмеження підтримуваних операторів, правила квантування та резервне виконання на хості.

Отже, програмний стек є частиною чипа. Формально швидкий NPU із закритим або нестабільним компілятором може виявитися гіршою ціллю для розгортання, ніж менш потужний пристрій із відкритим набором інструментів, каталогом моделей, симулятором і стабільним середовищем виконання.

## Що в цьому посібнику означає «підтримується»

У документації постачальників слово *підтримка* вживають дуже вільно. У цій статті розрізнено чотири рівні доказів:

| Рівень доказів | Що він підтверджує | Чого він не підтверджує |
|---|---|---|
| **Валідована модель** | Постачальник публікує запис моделі, результат або таблицю сумісності для конкретного чипа | Що змінена вами контрольна точка збереже таку саму точність |
| **Офіційний приклад** | Постачальник надає наскрізний посібник або демонстрацію для названої моделі | Що скомпілюються інші розміри, завдання чи покоління |
| **Можливості компілятора** | SDK імпортує фреймворк або надає підтримувані оператори | Що конкретний граф YOLO працюватиме наскрізно |
| **Маркетингове твердження або закритий доступ** | Продукт призначено для комп'ютерного зору, а приватний SDK існує | Що сумісність моделі можна публічно відтворити |

Це розрізнення має значення. «Імпортує ONNX» не означає «підтримує сегментацію YOLO11». Модель може розібратися, але перейти на CPU, скомпілюватися з неправильними вихідними числовими значеннями, переповнити пам'ять прискорювача або втратити точність під час квантування.

## Компанії та платформи NPU: огляд

У таблицях навмисно поєднано SoC, дискретні прискорювачі, розумні сенсори та суміжні цілі GPU/FPGA. Вони конкурують за те саме рішення щодо розгортання, навіть коли постачальники використовують назви NPU, AIPU, BPU, KPU, DLA, HTP, TPU чи MLA.

### Вбудовані SoC, розумні сенсори та промислові процесори

| Компанія | Відповідні чипи або платформи | SDK, компілятор і артефакт | Публічно задокументовані свідчення для комп'ютерного зору | Доступ |
|---|---|---|---|---|
| [Amlogic](https://github.com/Amlogic-NN/amlnn-toolkit) | A311D2, S928X, S905X5/S905D5, A311Y3, C308L/C302X/C302X2, T968D4, C305X2 та A123X | AMLNN Toolkit і `libnnsdk.so`, скомпільований `.adla`; окремий старий стек Acuity `.nb` для A311D | У [model playground](https://github.com/Amlogic-NN/amlnn-model-playground) задокументовано YOLOv5/6/7/8/10/11, YOLOX, YOLOE, YOLO-World, PP-YOLOE, сегментацію, оцінювання пози та OBB для A311D2, S905X5, A311Y3, C305X2 і A123X; інші перелічені тут чипи є цілями компілятора, а не записами в цій матриці підтримки | Публічні GitHub і колеса; також потрібен сумісний BSP/драйвер |
| [Rockchip](https://github.com/airockchip/rknn-toolkit2) | RK3562/3566/3568, RK3576, RK3588, RV1126B | RKNN-Toolkit2, RKNN Runtime, `.rknn` | YOLOv5/6/7/8/10/11, YOLOX, YOLO-World, PP-YOLOE, сегментація, оцінювання пози й OBB у [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) | Публічний GitHub; бінарні колеса |
| [Qualcomm](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | Платформи Snapdragon і Dragonwing, QCS6490, QCS8550 і Dragonwing IQ-9075 | QAIRT/QNN, SNPE та AI Hub; контекстні бінарні файли QNN або DLC | [Колекція моделей AI Hub](https://github.com/qualcomm/ai-hub-models) містить YOLOv3/5/6/7/8/9/10/11/26, YOLOX, YOLO-World, YOLOR, RF-DETR і моделі виявлення, сегментації та оцінювання пози | Документація публічна; вимоги до SDK і облікового запису різняться |
| [Texas Instruments](https://github.com/TexasInstruments/edgeai-tidl-tools) | AM62A, AM67A, AM68A, AM69A та TDA4x; попередня версія TDA54-Q1 | Processor SDK Edge AI і TIDL | У [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo) опубліковано оптимізовані моделі виявлення, сегментації, оцінювання пози й класифікації для поточних AM6xA/TDA4; це ще не матриця цілей для попередньої версії TDA54-Q1 | Публічний GitHub і Processor SDK |
| [NXP](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | i.MX 8M Plus, i.MX 93, i.MX 95 і придбані Kinara Ara-1/Ara240 | eIQ із TIM-VX, Vela, Neutron Converter та Ara SDK | У [eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) є ресурси або рецепти для YOLOv4-tiny/v8, NanoDet, CenterNet, FastestDet, SSD Lite та YOLACT; запис YOLOv5 містить лише документацію | Деякі компоненти публічні, інші доступні лише за обліковим записом |
| [STMicroelectronics](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) | Варіанти STM32N6x7, зокрема STM32N657/647, з прискорювачем Neural-ART | STM32Cube AI Studio і ST Edge AI Core | У поточному репозиторії services перелічено Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26 та ST-YOLOX, а також моделі оцінювання пози й сегментації | Публічні інструменти й репозиторії |
| [Infineon](https://documentation.infineon.com/psocedge/) | PSOC Edge E83/E84 із Cortex-M55 та Ethos-U55; окремий прискорювач NNLite на боці M33 | ModusToolbox, [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter), TFLite Micro та Arm Vela | В архітектурних матеріалах як типові ядра названо MobileNetV1/V2, а комплект AI Kit E84 має камеру; власної матриці валідації YOLO для PSOC Edge не знайдено | Публічна документація, компоненти SDK і поточні комплекти оцінювання E84 |
| [Renesas](https://github.com/renesas-rz/rzv_drp-ai_tvm) | RZ/V2L, V2M, V2MA, V2H і V2N | DRP-AI Translator, DRP-AI TVM і RUHMI | У [списку моделей RZ/V2H](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) валідовано варіанти розмірів YOLOv5/v8/v11/26, YOLOX, оцінювання пози та сегментації | Публічний GitHub і SDK для плат |
| [Sony](https://www.aitrios.sony-semicon.com/edge-ai-devices/imx500) | Інтелектуальний сенсор зору IMX500 і Raspberry Pi AI Camera | Edge-MDT, Model Compression Toolkit і пакувальник IMX500; пакет `.rpk` | У [каталозі Raspberry Pi IMX500](https://github.com/raspberrypi/imx500-models) є YOLOv8n, YOLO11n, EfficientDet Lite, NanoDet+ і SSD | Публічний процес Raspberry Pi; діють умови Sony для інструментів |
| [Ambarella](https://www.ambarella.com/developer/model-garden/) | Сімейства CV72/CV75, CV5/CV52 та CV3-AD | Cooper Developer Platform, компілятор CVflow і DAG середовища виконання | У публічному model garden названо YOLOX-S, RTMDet-nano, DeepLabV3+, TopFormer, OWL-ViT і LLaVA OneVision | Переважно доступ для партнерів |
| [Synaptics](https://developer.synaptics.com/docs/sl/overview) | Astra SL1600/SL1680; SL2611/13/15/17/19; окреме сімейство MCU SR100 | SyNAP `.synap` для SL16xx; Torq/IREE `.vmfb` для SL261x; окремий SDK SR, орієнтований на Ethos-U55 | Офіційний бенчмарк SyNAP для YOLOv8n/v8s і приклад Torq YOLOv8 для SL261x; свідчення для SR потрібно оцінювати окремо | Портал розробника; деякі завантаження закриті |
| [MediaTek](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) | Genio 360/360P/420/520/720 з NP8 і MDLA 5.3; Genio 510/700/1200 зі старішими NP6/MDLA | NeuroPilot Converter, `ncc-tflite`, Neuron Runtime і `.dla`; шлях ONNX Runtime на окремих новіших системах | В офіційному IoT AI Hub опубліковано сторінки моделей і бенчмарки YOLOv5 та YOLOv8, а також моделі класифікації й розпізнавання облич | Публічна документація Yocto; ключові пакети NP8 та матеріали Android потребують статусу прямого клієнта/NDA |
| [Allwinner](https://docs.aw-ol.com/v853/en/npu/dev_npu/) | SoC для машинного зору V853 із Vivante NPU на 1 TOPS | Перетворення Acuity/Pegasus і середовище виконання `viplite` | В офіційних матеріалах V853 названо YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet та мережі розпізнавання облич і людей | Публічна документація; для завантаження SDK може знадобитися обліковий запис |
| [Canaan/Kendryte](https://github.com/kendryte/nncase) | Поточні K230/K230D; застарілі K210/K510 KPU | Компілятор і середовище виконання `nncase`; `.kmodel` | Посібник nncase для K230 охоплює компіляцію/симуляцію/виконання YOLOv5s; офіційний каталог демо й актуальний [журнал змін CanMV](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md) документують завдання YOLOv8/11/26 | Публічні компілятор і PyPI; плагін чипа бінарний |
| [Alif Semiconductor](https://alifsemi.com/support/kits/ensemble-e7appkit/) | Ensemble E7 із двома прискорювачами ML, зокрема Ethos-U55; [E8](https://alifsemi.com/ensemble-e8-series/) з одним Ethos-U85 і двома NPU Ethos-U55 | Розгортання на основі Arm Vela і TFLite Micro | Alif публікує бенчмарк детектора облич YOLO-Fastest для всього сімейства в INT8, але широкої матриці сучасних YOLO за кожною ціллю немає | Публічна документація та ресурси комплекту оцінювання |
| [Himax](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) | MCU для endpoint AI HX6538 WiseEye2 із Cortex-M55 і Ethos-U55 | TFLite Micro та Arm Vela, резервний шлях через CMSIS-NN і еталонні ядра | В офіційних [прикладах WiseEye2](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) є виявлення, оцінювання пози та класифікація в YOLOv8n, виявлення в YOLO11n, face mesh і PeopleNet | Публічний GitHub, документація й доступна для купівлі плата партнера |
| [Analog Devices](https://www.analog.com/en/products/max78002.html) | MCU для AI MAX78000 і MAX78002 із малопотужними прискорювачами CNN | [`ai8x-training`](https://github.com/analogdevicesinc/ai8x-training), `ai8x-synthesis`/`izer` та MSDK; згенеровані C-код і ваги | В офіційних матеріалах охоплено ідентифікацію/виявлення облич, RetinaNet, Visual Wake Words, класифікатори й розпізнавання дій; розгортання YOLO від першої сторони не знайдено | Публічні інструменти GitHub, документація та плати оцінювання |
| [D-Robotics](https://github.com/D-Robotics/rdk_model_zoo) | Плати RDK X3/X5/Ultra і новіших серій S100 із BPU | OpenExplorer/Algorithm Toolchain і `hbm_runtime`; `.bin` для X5, `.hbm` для поточної `rdk_s` | Поточна гілка `rdk_x5` документує YOLOv5/v5u/v8/v9/v10/11/12/13/26, YOLOE, YOLO-World, сегментацію, оцінювання пози, класифікацію, OCR і CLIP для RDK X5; для X3 і серії S використовують окремі гілки | Публічний GitHub; окремі пакети інструментів залежать від платформи |
| [AXERA](https://github.com/AXERA-TECH/ax-samples) | AX650, AX637, AX630C, AX620Q і AX615 | Компілятор Pulsar2 та AXEngine; `.axmodel` | У поточних офіційних прикладах є YOLOv5/6/7/8/9/10/11/13/26, YOLOX і YOLO-World; завдання та цілі залежать від чипа | Публічні приклади й документація; компілятор поширюється окремо |
| [SOPHGO](https://github.com/sophgo/tpu-mlir) | BM1684/1684X/1688/1690 і CV186X/CV18xx | TPU-MLIR і середовище виконання SOPHON; `.bmodel` або `.cvimodel` | В офіційних демо є YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, OBB, сегментація, розпізнавання облич, SAM та OCR | Компілятор із відкритим кодом і середовище виконання постачальника |
| [Huawei Ascend](https://www.hiascend.com/en/software/cann) | Ascend 310/310P/310B і edge-продукти Atlas 200I/300I | CANN, компілятор ATC та AscendCL; `.om` | В офіційному [Ascend ModelZoo](https://github.com/Ascend/modelzoo) є YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN і моделі сегментації | Публічна документація; пакети CANN і ядра мають відповідати цілі |
| [Cambricon](https://github.com/Cambricon/magicmind_cloud) | MLU370-X4/S4 у наведеній публічній матриці; MLU270 є застарілим обладнанням і до неї не входить | Neuware і MagicMind | У матриці MagicMind 1.7 з репозиторію наведено YOLOv3/4/5/7/8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN і моделі сегментації | Історичні приклади моделей публічні; доступ до поточного SDK/контейнера закритий |
| [Sunplus](https://sunplus.atlassian.net/wiki/spaces/C3/pages/1971126471/SP7350%2BSpecification) | SP7350/C3V, заявлено приблизно 4.1-4.6 TOPS | Docker NPU Vivante Acuity, середовище виконання SNNF і `.nb` | В офіційній документації є YOLOv5 і власне наскрізне [розгортання YOLOv8](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350) | Публічні джерела/документація та екосистема плат |
| [ESWIN Computing](https://www.eswincomputing.com/en/news/info/96.html) | EIC7700/EIC7700X і RISC-V SoC із двома кристалами [EIC7702/EIC7702X](https://www.eswincomputing.com/en/news/info/104.html) | ENNP: EsQuant, поточний компілятор EsAAC, симулятор і середовище виконання ESSDK; `.model`; у старішій документації використовувався `ennc-compile` | Документація ENNP на сайті Milk-V містить наскрізний [посібник YOLOv3](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3), а також приклади MobileNetV2 і ResNet | Документація від виробника та виробника плат; регіональні завантаження |
| [Nuvoton](https://www.nuvoton.com/products/microcontrollers/arm-cortex-m55-mcus/m55m1-series/index.html) | MCU M55M1 з Ethos-U55-256 | TFLite Micro, Arm Vela і NuEdgeWise/NuML | В офіційних прикладах [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise) названо YOLOv8-nano, YOLOX-nano, YOLO Fastest, SSD-MobileNet і моделі класифікації | Переважно публічний набір інструментів MCU |
| [Realtek](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | Камерна платформа AmebaPro2 RTL8735B / AMB82-mini | Arduino/FreeRTOS SDK, VoE і NeuralNetwork API; `.nb` | У пакеті моделей є YOLOv3-tiny, YOLOv4-tiny, YOLOv7-tiny, SCRFD і MobileFaceNet | Середовище виконання публічне; для [доступу до конвертера користувацьких моделей](https://ameba-doc-arduino-sdk.readthedocs-hosted.com/en/latest/FAQ/offline_ai_model_conversion_steps.html) потрібен запит |
| [Telechips](https://docs.topst.ai/product/p/ai) | TCC7500 / плата TOPST AI, заявлено 8 TOPS | TC-NN-Toolkit / Enlight SDK; проміжний `.enlight` і скомпільований пакет для розгортання | В [офіційному проєкті TOPST](https://docs.topst.ai/blog/31) розгорнуто YOLOv8s; перетворення UFLD v1/v2 не вдалося через непідтримувані шари, запропоновано лише обхідний шлях із розділенням/післяобробкою. [YOLOv4](https://community.topst.ai/t/segmentation-fault/358) та [інший процес YOLOv8](https://community.topst.ai/t/segmentation-fault-error/415) фігурують у дописах спільноти | Документація плат публічна; завантаження компілятора/операторів часто потребує дозволу |
| [T-Head](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | TH1520, заявлено INT8 на 4 TOPS, на LicheePi 4A | Компілятор HHB і CSI-NN2/SHL; `hhb.bm` плюс згенеровані код і параметри | В офіційних прикладах виробника плати YOLOv5n/s і MobileNetV2 запускаються на NPU | Публічні приклади; сумніви щодо актуальності й підтримки застарілого стека |
| [SigmaStar](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | Актуальний SoC для розумних камер SSU9383CM і старіші сімейства IPU | Поточне середовище виконання MI_IPU; старіший набір інструментів [SGS_IPU](https://wx.comake.online/doc/doc/SigmaStarDocs-SSC9381G_9351_Pudding-ULS00V040-20210913/customer/development/dla/tools.html) створював `.sim`, що перетворювався на `sgsimg.img` | Старіший [офіційний постпроцесор SDK](https://wx.comake.online/doc/doc/Sigmastar_SDK_v1.2.2/module/User_Guide/Common/SigmaStar_Post_Processing_Module.html) називає SSD і YOLOv1/2/3; сумісність цих моделей і артефактів із SSU9383CM публічно не підтверджено | Актуальний чип, але публічні свідчення щодо компілятора й моделей належать різним поколінням |
| [HiSilicon](https://www.hisilicon.com/cn/products/smart-vision/machine-vision/hi3516cv610) | SoC для розумного машинного зору Hi3516CV610/DV500, Hi3519DV500 і Hi3403V100 | ATC у `.om` із середовищем виконання NNN/SVP-NNN на платі | У цільовій матриці HiSpark, особливо для Hi3403 і Hi3591P, названо YOLOv3-YOLO11, пози, сегментацію, OBB, OCR і глибину; це не валідація для всіх SoC у рядку | Активна лінійка, окрема від Ascend; моделі репозиторію позначено лише для некомерційного використання |

### Спеціалізовані edge-прискорювачі та модулі

| Компанія | Відповідне обладнання | SDK і артефакт | Публічно задокументовані моделі | Доступ |
|---|---|---|---|---|
| [Hailo](https://github.com/hailo-ai/hailo_model_zoo) | Hailo-8, Hailo-8L, Hailo-10H і сімейство Hailo-15 | Dataflow Compiler і HailoRT; `.hef` | YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 і YOLO26, а також YOLOX, DAMO-YOLO, SSD, EfficientDet, сегментація, оцінювання пози та OBB; таблиці моделей залежать від покоління | Model Zoo публічний; компілятор у Developer Zone |
| [Axelera AI](https://docs.axelera.ai/sdk/reference/models/model-zoo/) | Продукти Metis у продажу; анонсовані продукти портфеля Europa і Titania | Публічний Voyager SDK; alpha Pipeline Builder `.axm`/`.axe`, класичні пакети `.axmodel` | Валідовано YOLOv3/5/7/8/9/10/11/26, YOLOX, YOLO-NAS, OBB, оцінювання пози, сегментацію та багато моделей не з YOLO | SDK публічний на GitHub; доступ до підтримки для клієнтів |
| [DEEPX](https://developer.deepx.ai/modelzoo/) | DX-M1 і DX-M1M | DXNN SDK, DX-COM і DX-RT; `.dxnn` | Станом на 15 серпня 2026 року в каталозі Model Zoo було 354 записи для DX-COM 2.4.0/DX-RT 3.4.0, зокрема YOLOv3-YOLO11 і YOLO26, YOLOX, SSD, EfficientDet, NanoDet, сегментація, оцінювання пози та OBB | Публічні ресурси для розробників і пакет SDK |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | Прискорювач MX3 і багаточипові модулі | MemryX SDK, Neural Compiler і середовище виконання; `.dfp` | В офіційних прикладах і нотатках до релізів є виявлення, сегментація й оцінювання пози, зокрема YOLOv10, YOLO11 і YOLO26 | Публічна документація та SDK |
| [Kneron](https://doc.kneron.com/docs/) | KL520, KL530, KL630, KL720 і KL730 мають поточну документацію компілятора; KL830 трапляється в деяких API PLUS, але відсутній у поточному списку цілей компілятора | Kneron PLUS і Model Toolchain; `.nef` для задокументованих цілей компілятора | В офіційному [процесі YOLO](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/) основний приклад стосується Tiny-YOLOv3; інші матеріали охоплюють YOLOv5 | Публічна документація; пакети інструментів/завантаження залежать від чипа |
| [SiMa.ai](https://docs.sima.ai/pages/palette/modelsdk.html) | MLSoC і серійна платформа Modalix на 50 TOPS | Palette, ModelSDK, MLA Compiler і ModelExecutor | У публічних релізах валідовано YOLOv7/v8, YOLOX, пози/сегментацію, DETR, Mask R-CNN і EfficientDet | Документація публічна; SDK продукту комерційний |
| [EdgeCortix](https://www.edgecortix.com/en/hardware) | SAKURA-II, заявлено 60 TOPS, прискорювач у форматах M.2 та PCIe | Компілятор MERA | Постачальник заявляє підтримку від машинного зору до генеративного AI, але не публікує достатньо точну поточну матрицю сумісності YOLO | Комерційне тестування/запит на замовлення; валідація з партнерами |
| [BrainChip](https://brainchip.com/metatf-dev-tools/) | Супроцесор/модуль M.2 AKD1500 у продажу; застарілі платформи AKD1000; IP Akida 2 | Пакети MetaTF: `akida-models`, `quantizeml`, `cnn2snn` і середовище виконання `akida` | У картці моделей Akida 2 названо детектор YOLOv2 AkidaNet0.5, CenterNet, AkidaUNet і розпізнавання облич; ці результати не означають автоматичної валідації AKD1500 | Основні пакети Python публічні; зіставлення з обладнанням залежить від цілі |
| [Blaize](https://www.blaize.com/products/) | Продукти P1600, Pathfinder і Xplorer | Picasso SDK, NetDeploy і AI Studio | Комп'ютерний зір є цільовим ринком, але аудованої публічної матриці сумісності названих моделей не знайдено | Комерційний/закритий доступ |
| [Google Coral](https://github.com/google-coral/edgetpu) | Застарілі Edge TPU USB, PCIe, M.2 і Dev Board; окремий активний IP [Coral NPU](https://github.com/google-coral/coralnpu) із відкритим кодом | Застарілі Edge TPU Compiler/`libedgetpu`/PyCoral; новий RISC-V Coral NPU надає IP, RTL/симуляцію та приклади ELF | У старих прикладах основна увага на SSD MobileNet, класифікації, DeepLab і MoveNet; для нового IP немає публічної матриці розгортання YOLO, і він не є прямим наступником Edge TPU | Основні старі репозиторії заархівовано; IP нового Coral активно розробляється |
| [Lattice Semiconductor](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | FPGA ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E й Avant-X | sensAI Studio, Neural Network Compiler і конфігурований IP прискорювача | У поточній таблиці компілятора перелічено YOLOv1, YOLOv5, YOLOv8 і YOLO11 у режимах для окремих цілей, а також SSD, MobileNetV2-SSD, ResNet і ENet | Компілятор/посібники публічні; умови на IP різняться, а Advanced CNN Accelerator є комерційним |
| [Mobilint](https://www.mobilint.com/sdk-qb) | Прискорювач REGULUS на 10 TOPS і ARIES MLA100 на 80 TOPS | qb SDK; компілятор INT8 і `.mxq` | У публічному [каталозі моделей](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md) перелічено виявлення YOLOv3/5/7/8/9/10/11/12/26, а також варіанти сегментації, оцінювання пози й OBB | Публічна документація/моделі; SDK і обладнання комерційні |
| [Rebellions](https://rebellions.ai/developers/) | Активні ATOM+ CA22 і ATOM-Max CA25; виведені з експлуатації ATOM CA02/ATOM+ CA12; статус інструментів ATOM-Lite CA21 неясний | Компілятор/середовище виконання/профайлер RBLN; `.rbln` | В офіційних релізах підтримки названо YOLOv3, сімейства YOLOv5/6/7/8 і актуальні посібники YOLOv8 | Документація публічна; для колеса компілятора потрібні облікові дані порталу |
| [FuriosaAI](https://furiosa.ai/warboy) | Warboy Gen1, орієнтований на машинний зір; окреме покоління RNGD | Furiosa SDK; INT8 `.enf` для Warboy, окремий `.fxb` для RNGD | У zoo Warboy задокументовано SSD, YOLOv5M/L і YOLOv7-w6-pose; дорожня карта RNGD позначає підтримку YOLOv8m виконаною в 2024 Q4, але не надає докладної поточної публічної матриці моделей для машинного зору | Потрібен IAM/обліковий запис; покоління слід розглядати окремо |

### Суміжні платформи, які розробники порівнюють із NPU

| Компанія | Обладнання | Стек розгортання | Чому її включено до порівняння |
|---|---|---|---|
| [NVIDIA](https://github.com/NVIDIA-AI-IOT/deepstream_tools/tree/main/yolo_deepstream) | GPU Jetson Orin і DLA; GPU Jetson Thor без DLA | JetPack, TensorRT і DeepStream; `.engine` | Дуже зріле розгортання комп'ютерного зору; в офіційних інструментах DeepStream задокументовано YOLOv4/v7/v8/v9/11, зокрема конфігурацію OBB YOLO11. У багатьох результатах використовується GPU, а непідтримувані шари DLA Orin можуть перейти на GPU. |
| [AMD](https://vitisai.docs.amd.com/en/6.2/) | Цілі Kria/застарілого DPU; Vitis AI 6.2 GA для Versal AI Edge Gen1 VEK280/VE2802 і Gen2 VEK385; окремі клієнтські NPU Ryzen AI | Застарілий `.xmodel`; знімок NPU для конкретної цілі Gen1; каталог кешу компіляції або виробничий пакет `.rai` для Gen2 | Vitis AI публікує матеріали з комп'ютерного зору, але застарілий DPU, Versal Gen1, Versal Gen2 і Ryzen AI мають різні контракти компіляції та розгортання. |
| [Microchip](https://www.microchip.com/en-us/products/fpgas-and-plds/fpga-and-soc-design-tools/vectorblox) | FPGA PolarFire SoC із конфігурованим IP прискорювача CoreVectorBlox | Публічний VectorBlox SDK 3.1 і Libero; ресурси для розгортання `.vnnx`, `.hex` і `.ucomp` | Поточні [посібники](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md) охоплюють YOLOv5n, виявлення/класифікацію/OBB/оцінювання пози/сегментацію YOLOv8, YOLOv9t та інші моделі комп'ютерного зору; SDK 3.1 наразі призначено для PolarFire SoC Video Kit. |
| [Intel](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) | NPU Core Ultra, CPU й інтегрований/дискретний GPU | OpenVINO IR або скомпільований кеш моделі | OpenVINO пропонує публічний міжплатформний шлях; звіряйте стовпці NPU у [матриці перевірених моделей](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html), а не припускайте, що кожен приклад YOLO для OpenVINO валідовано для NPU. |
| [Apple](https://developer.apple.com/machine-learning/core-ml/) | Apple Neural Engine у чипах серії A від A11 і новіших та чипах серії M | Core ML і `coremltools`; `.mlpackage`/`.mlmodelc` | Дуже доступний споживчий NPU через Core ML, але Apple абстрагує розподіл роботи між CPU/GPU/Neural Engine, а не надає компілятор для конкретно YOLO. |
| [GreenWaves Technologies](https://github.com/GreenWaves-Technologies/nn_menu_gap9) | GAP9 із нейронним рушієм NE16 | GAP SDK, NNTool і згенерований AutoTiler код | В офіційних репозиторіях є MobileNet SSD, виявлення облич, класифікація та окремий [детектор людей YOLOX](https://github.com/GreenWaves-Technologies/yolox_people_detection). Повний SDK доступний кваліфікованим клієнтам. |
| [Syntiant](https://www.syntiant.com/hardware) | NDP200 Neural Decision Processor у масовому виробництві та NDP250 на етапі зразків | SDK Syntiant і пакети для розгортання | Надмалопотужні процесори для постійного розпізнавання зору/сенсорів: для NDP200 задокументовано потужність нижче 1 mW, а для розпізнавання зображень NDP250 нижче 30 mW. Широкої публічної матриці YOLO не знайдено. |

## Amlogic: два покоління NPU, а не одне

Amlogic заслуговує на окремий розгляд, оскільки вебджерела часто об'єднують несумісні продукти. Компанія має старіший шлях на основі VeriSilicon і новіший власний шлях ADLA. Вони використовують різні компілятори, артефакти й середовища виконання.

| Покоління | Типові чипи | NPU та набір інструментів | Скомпільований артефакт | Практичний стан |
|---|---|---|---|---|
| Застаріле | A311D і S905D3, поширені на Khadas VIM3/VIM3L | VeriSilicon Vivante VIPNano-QI, набір Acuity, KSNN і старіший `aml_npu_sdk` | `.nb` у вихідному каталозі `nbg_unify` | Наявні плати й демо; окрема застаріла інтеграція |
| Поточне ADLA2 | C308L/C302X, S928X, A311D2, T968D4, S905X5/S905D5 та C302X2 | Amlogic ADLA2, AMLNN Toolkit і NNSDK2 | Специфічний для цілі `.adla`; W8A8 або W8A16 | Поточний стек, орієнтований на цілочисельні формати |
| Поточне ADLA3 | A311Y3, C305X2 і A123X | Amlogic ADLA3, AMLNN Toolkit і NNSDK2 | Специфічний для цілі `.adla`; W4A8, W8A8, W4A16, W8A16 або W16A16 | Додає нативні можливості INT4, FP16 і BF16 |

У [даташиті A311D](https://dl.khadas.com/products/vim3/datasheet/a311d-datasheet.pdf) зазначено початковий NPU на 5 TOPS INT8. На старій [сторінці застосунків Khadas VIM3](https://docs.khadas.com/products/sbc/vim3/npu/npu-app) і в [огляді NPU](https://docs.khadas.com/products/sbc/vim3/npu/start) задокументовано DenseNet CTC, MTCNN, RetinaFace, YOLOFace, YOLOv2, YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv7-tiny та YOLOv8n; в огляді також наведено YOLOv8n-pose, а FaceNet позначено застарілою. Це реальні приклади, але вони свідчать про стару екосистему Acuity `.nb`, а не про поточні чипи ADLA. A311D не є A311D2, а C305X не є C305X2.

Є й друга апаратна пастка. У [посібнику з ревізій Khadas VIM4](https://docs.khadas.com/products/sbc/vim4/configurations/identify-version) зазначено, що початкова плата V12/A311D2 ревізії B не має NPU; заявлений NPU на 3.2 TOPS з'явився у V13A та пізніших платах із компонентом A311D2-N0D ревізії C. Самої назви продукту недостатньо, щоб вибрати пристрій для тестування.

### Поточний стек AMLNN і ADLA

Поточний [AMLNN Toolkit](https://github.com/Amlogic-NN/amlnn-toolkit/tree/7d3cc9a36179b756ce79c953c14d43acab1f2af5) охоплює перетворення моделей, квантування, компіляцію, інференс і профілювання. У його зафіксованому [посібнику користувача за травень 2026 року](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/02_Amlogic_NPU_User_Guide_V0.1.pdf) задокументовано імпортери ONNX, TFLite з рухомою комою або квантизований, TorchScript `.pt` і PyTorch 2 ExportedProgram/PT2. У докладному посібнику шляхи TensorFlow, Paddle і Keras позначено як заплановані, хоча в огляді репозиторію формулювання ширше. Компілятор створює `.adla`. Для розгортання Python використовують AMLNN або `amlnn_edge_toolkit_lite`; нативний C/C++ підключає `libnnsdk.so` через `nnsdk2.h`. Для розробки на хості можна використовувати ADB із `nnserver`; середовище виконання також підтримує Android, Buildroot, Yocto та окремі середовища Debian/Armbian.

Поточні ідентифікатори платформ, опубліковані в наборі інструментів:

| Ідентифікатор цілі | Платформа |
|---:|---|
| `001` | C308L / C302X |
| `002` | S928X |
| `003` | A311D2 |
| `004` | T968D4 |
| `005` | S905X5 / S905D5 |
| `006` | C302X2 |
| `007` | A311Y3 |
| `008` | C305X2 / A123X у докладній документації та матриці моделей |

Поле цілі має практичне значення. Amlogic вимагає, щоб `.adla`, `nnsdk2.h`, `libnnsdk.so`, BSP і покоління компілятора/середовища виконання відповідали одне одному. У закріпленому [посібнику зі швидкого старту](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/01_Amlogic_NPU_Quick_Start_guide_V0.1.pdf) вимагається драйвер ADLA версії 2.0.2 або новішої, NNSDK 3.0.0 або новішої та NNSDK2 1.0.0 або новішої; у [посібнику з розгортання Android](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/06_Amlogic_NPU_Android_Deployment_Guide_V0.1.pdf) конкретно пояснено зв'язок бібліотеки, заголовка, драйвера й BSP. Розглядайте `.adla` як артефакт збірки, прив'язаний до ABI, а не як переносну модель.

Квантування також залежить від покоління NPU. Для цілей ADLA2 001-006 задокументовано компіляцію W8A8 і W8A16. Цілі ADLA3 007 і 008 додають комбінації W4A8, W4A16, W8A8, W8A16 і W16A16, а також нативну підтримку INT4, FP16 і BF16 в [посібнику з операторів](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/05_Amlogic_NPU_Support_Operator_List_V0.1.pdf). Amlogic рекомендує 200-500 репрезентативних зразків для калібрування. Опція випадкових даних прямо призначена для тестування продуктивності, а не для квантування зі збереженням точності.

### Задокументоване охоплення моделей Amlogic

Закріплений [каталог моделей Amlogic](https://github.com/Amlogic-NN/amlnn-model-playground/blob/03ca63ef20c4f1560722de26ab910826884f334a/README.md) є набагато сильнішим доказом, ніж загальна заява про підтримку ONNX. У його опублікованій матриці підтримки/прикладів названо A311D2, S905X5, A311Y3, C305X2 та A123X. Те, що компілятор приймає інші ідентифікатори цілей, не доводить валідації на них усього цього списку моделей.

| Завдання | Задокументовані моделі |
|---|---|
| Класифікація | MobileNetV2 і ResNet50-v2; DINO на ADLA3 |
| Виявлення об'єктів | PP-YOLOE, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv10, YOLO11, YOLOE, YOLO-World, YOLOX і виявлення QR-кодів |
| Обличчя та жести | RetinaFace і Gesture Recognition |
| Сегментація | DeepLabV3, PP-LiteSeg, YOLOv5-seg і YOLOv8-seg |
| Орієнтоване виявлення | YOLOv8 OBB |
| Пози | BlazePose detector/landmark і YOLOv8 pose |
| OCR і мовлення | LPRNet, варіанти PaddleOCR, Whisper Tiny і SenseVoice на окремих новіших чипах |
| Новіші трансформери й мультимодальні моделі | DETR, MobileSAM, CLIP/MobileCLIP і окремі приклади LLM/VLM із малою кількістю бітів на новіших платформах |

У цьому ж репозиторії опубліковано такі показники роботи моделі й середовища виконання W8A8. Це вимірювання постачальника для нейронної мережі на NPU, а не для повного пайплайна камери.

| Модель і розмір входу | S905X5 | A311D2 | A311Y3 |
|---|---:|---:|---:|
| YOLOv8n, 640 x 640 | 101.72 FPS | 95.14 FPS | 191.06 FPS |
| YOLOv8s, 640 x 640 | 42.33 FPS | 42.77 FPS | 83.08 FPS |
| YOLOv8m, 640 x 640 | 19.67 FPS | 19.82 FPS | 35.30 FPS |
| YOLOv8l, 640 x 640 | 10.53 FPS | 10.12 FPS | 18.37 FPS |
| YOLO11n, 640 x 640 | 41.14 FPS | 41.48 FPS | 62.24 FPS |

Застереження тут особливо важливі. У посібнику з операторів Amlogic вказано, що NMS виконується програмно і на ADLA2, і на ADLA3. У README ці результати визначено як вимірювання роботи моделі на NPU; попередню та післяобробку виключено, але не уточнено, чи враховано всі передавання даних із пам'яті. У рядках точності для YOLO використовуються підмножини COCO із 300 зображень, а в рядках класифікації набір ImageNet `val1000`. Також у таблицях наведено іншу затримку YOLOv8n на A311Y3, ніж у таблиці FPS, і репозиторій не пояснює різницю умов. Не вказано частоти, режим живлення, охолодження, точні версії SDK/BSP і тривалість тестування. Використовуйте ці цифри для розуміння стека одного постачальника, а не для порівняння з іншим.

### Чому Amlogic має стратегічний інтерес

Amlogic поєднує чотири незвичні переваги: широке поширення вбудованих SoC, нещодавно відкритий компілятор, поточну матрицю моделей із великим перетином із завданнями LibreYOLO та відносно слабку першокласну інтеграцію в поширені бібліотеки навчання. Поточні репозиторії також нові: вони стали публічними наприкінці 2025-го й на початку 2026 року, а змістовні посібники датовано періодом із травня до липня 2026 року. Це створює реальну можливість стати першими, а також ризик нестабільності API.

Чисту інтеграцію варто побудувати так: детермінований експорт ONNX LibreYOLO слугує входом для компілятора, для збірок із низькою точністю використовуються репрезентативні калібрувальні зображення, створюються `.adla` і маніфест метаданих, а завантаження виконує адаптер NNSDK2. Для застарілого A311D слід використовувати чітко окрему ціль `.nb`, оскільки подання `.nb` і `.adla` як одного бекенду спричинило б непомітні помилки сумісності. Репозиторії Amlogic містять ліцензії Apache-2.0 верхнього рівня, але права на поширення комплектних коліс, спільних бібліотек, `nnserver` і двійкових файлів Android AAR слід підтвердити безпосередньо, перш ніж LibreYOLO включатиме їх до своїх матеріалів.

## Найсильніші відкриті екосистеми розгортання

Наведені нижче компанії наразі пропонують найкраще поєднання доступного обладнання, публічних технічних матеріалів і свідчень щодо названих моделей. Це не означає, що вони завжди швидші. Їхні твердження про сумісність легше оцінити до купівлі обладнання.

### Hailo

Hailo є взірцевим прикладом екосистеми спеціалізованого прискорювача для комп'ютерного зору. Моделі розбираються у Hailo Archive, оптимізуються та квантуються на репрезентативних зображеннях, після чого компілюються у файл Hailo Executable Format (`.hef`). Застосунки завантажують цей HEF через HailoRT.

[Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) містить рецепти, попередньо навчені моделі, конфігурації післяобробки й дані продуктивності для виявлення, сегментації, класифікації, оцінювання пози та інших завдань. До переліку YOLO входять YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 і YOLO26, а також YOLOX, DAMO-YOLO, SSD та EfficientDet. Таблиця [виявлення об'єктів Hailo-8 із фіксацією версії](https://github.com/hailo-ai/hailo_model_zoo/blob/v2.19.0/docs/public_models/HAILO8/HAILO8_object_detection.rst) є кращим джерелом сумісності, ніж загальна сторінка продукту, а [окремий застосунок](https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/python/standalone_apps/object_detection/README.md) описує готові до розгортання пайплайни YOLO.

Важливо враховувати межу між версіями. Hailo-8 і Hailo-8L використовують старішу лінійку Model Zoo 2.x, Dataflow Compiler 3.x і HailoRT 4.x, тоді як Hailo-10 і Hailo-15 використовують поточне покоління 5.x. Рецепти, поведінка парсера й HEF не взаємозамінні лише тому, що всі пристрої мають назву Hailo.

Для Hailo-15 також передбачено окремий рівень застосунків. Він використовує Vision Processor Software Package і Hailo Media Library, а не шлях хоста у стилі TAPPAS для Hailo-8/10H. Публічний довідковий репозиторій [`hailo-camera-apps`](https://github.com/hailo-ai/hailo-camera-apps) заархівували 3 травня 2026 року, а [`hailo-apps-core`](https://github.com/hailo-ai/hailo-apps-core) залишається окремим публічним фреймворком застосунків. Архівування репозиторію не доводить завершення життєвого циклу продукту, але для проєктів Hailo-15 варто уточнити поточний пакет застосунків у Developer Zone.

У [посібнику з розгортання Hailo](/docs/export/hailo) LibreYOLO навмисно зупиняється на передачі статичного ONNX, оскільки пропрієтарний компілятор не можна включити як залежність Python. Це чесна межа між створенням відповідного графа й заявою про перевірений HEF.

### Rockchip

Rockchip має одну з найбільших екосистем плат для вбудованого Linux серед ентузіастів і комерційних розробників, особливо на RK3588, RK3576 і RK356x. Публічний [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) перетворює та квантує моделі на хості з x86 Linux, а RKNN Runtime або Toolkit-Lite виконує отриманий файл `.rknn` на платі.

У [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) незвично чітко зазначено чипи, сімейства моделей і точність. У поточній таблиці перелічено шляхи FP16 і INT8 для YOLOv5/6/7/8/10/11, YOLOX, PP-YOLOE, YOLO-World і варіантів сегментації YOLOv5/v8; YOLOv8 OBB і YOLOv8 pose зазначено лише для INT8. Також охоплено OCR, обличчя та інші мережі сегментації.

У LibreYOLO вже є консервативний прямий [експортер RKNN](/docs/export/rknn). Він перевіряє чотири конкретні варіанти виявлення на RK3588, може порівняти симулятор компілятора з ONNX Runtime та відхиляє невалідовані сімейства замість того, щоб прирівнювати успішну збірку до правильних передбачень. Така вузька заява про підтримку є добрим прикладом для кожного майбутнього бекенду NPU.

### Axelera AI

Axelera AI, яку часто помилково пишуть як «Accelera», створює AIPU Metis і продає її у продуктах M.2, PCIe та багаточипових системах. У продажу наразі є сімейство Metis; Europa й Titania є анонсованими продуктами портфеля, тому їх не слід подавати як продукти з однаковим статусом доступності. [Voyager SDK публічно доступний на GitHub](https://github.com/axelera-ai-hub/voyager-sdk) і охоплює вибір моделі, компіляцію, квантування, виконання та пайплайни застосунків; підтримка клієнтів потребує облікового запису.

Публічний [Voyager Model Zoo](https://docs.axelera.ai/sdk/reference/models/model-zoo/) є одним із найінформативніших у галузі. У ньому наведено варіант моделі, розмір входу, точність, правильність і виміряну продуктивність для YOLOv3, YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26 n/s/m/l/x, YOLOX, YOLO-NAS, OBB, оцінювання пози та сегментації, а також багатьох моделей класифікації й щільного передбачення. Публікація втрат точності після квантування поруч зі швидкістю корисніша, ніж саме число пікових TOPS.

Назви артефактів змінилися між поколіннями API. Потік компіляції alpha [Pipeline Builder](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) створює модель `.axm` і може запакувати переносний пайплайн у `.axe`. У [документації класичного пайплайна](https://docs.axelera.ai/sdk/reference/pipeline/model-formats/) описано `.axmodel` разом із метаданими моделі та маніфестом. Інтеграція має визначати встановлене покоління Voyager, а не жорстко задавати одне розширення.

### Qualcomm

Qualcomm охоплює надзвичайно багато пристроїв, але за назвою «NPU Qualcomm» ховаються різні інтерфейси. Розробники працюють із Hexagon HTP через QAIRT/QNN, старішим процесом SNPE, сервісами компіляції Qualcomm AI Hub, LiteRT або провайдером виконання QNN для ONNX Runtime, залежно від пристрою та класу продукту.

Офіційний репозиторій [Qualcomm AI Hub Models](https://github.com/qualcomm/ai-hub-models) є вагомим доказом підтримки на рівні моделей. У ньому публікуються оптимізовані пакети для виявлення/сегментації/оцінювання пози в YOLOv3, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11 і YOLO26, а також для YOLOX, YOLO-World, YOLOR, RF-DETR і багатьох моделей класифікації, оцінювання глибини та пози. AI Hub також надає профілювання для конкретних пристроїв, що важливо, бо модель, скомпільована для одного покоління HTP, не обов'язково переноситься на інше.

Основна складність інтеграції полягає в масштабі матриці: мають значення Android чи вбудований Linux, замовні компоненти QCS чи споживчий Snapdragon, архітектура HTP, версія QNN/QAIRT і схема квантування. Поточні власні сторінки Qualcomm також розходяться щодо статусу [Dragonwing IQ-9075](https://www.qualcomm.com/internet-of-things/products/iq9-series/iq-9075): сторінка продукту вказує Active, а комплект [EVK](https://www.qualcomm.com/developer/hardware/qualcomm-iq-9075-evaluation-kit-evk) доступний для оцінювання, тоді як [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program) досі позначає IQ-9075 як Sampling і вказує довготривалу підтримку до 2038 року. Префікс замовного/SKU-коду: QCS9075; Qualcomm заявляє конфігурації на 50 і 100 щільних INT8 TOPS, а також підтримку Ubuntu/Yocto. Серійний статус слід перевіряти для точного SKU замовлення, і інтеграція LibreYOLO має записувати цей SKU замість створення загальної теки «Qualcomm».

### DEEPX

Прискорювачі DX-M1/DX-M1M від DEEPX використовують SDK DXNN. DX-COM компілює та квантує модель, DX-RT виконує її, а розгорнутий артефакт має формат `.dxnn`. Компанія публікує [DX-AllSuite](https://github.com/DEEPX-AI/dx-all-suite), приклади застосунків у [DX-APP](https://github.com/DEEPX-AI/dx_app) і великий [онлайн-каталог моделей](https://developer.deepx.ai/modelzoo/).

Станом на 15 серпня 2026 року публічний каталог містив 354 записи для DX-COM 2.4.0 і DX-RT 3.4.0. Задокументоване охоплення включає YOLOv3-YOLO11 і YOLO26, YOLOX, SSD, EfficientDet, NanoDet, DAMO-YOLO, сегментацію YOLO, оцінювання пози й орієнтовані рамки. DEEPX є особливо реалістичним кандидатом на інтеграцію, оскільки межу між компілятором і середовищем виконання та артефакт для розгортання чітко визначено, а публічне охоплення моделей комп'ютерного зору широке.

## Промислові SoC: це кілька екосистем, а не одна

Промислові постачальники часто пропонують довший життєвий цикл продуктів і кращу інтеграцію камер, безпеки та систем реального часу, ніж виробники SoC для плат розробників. Їхні стеки AI можуть бути складнішими, бо один постачальник здатен одночасно випускати кілька не пов'язаних архітектур NPU.

### Texas Instruments

До поточних edge-AI процесорів TI належать AM62A, AM67A, AM68A, AM69A та пов'язані пристрої TDA4. Програмний шлях поєднує Processor SDK Linux, компілятор/середовище виконання TIDL, [Edge AI TIDL Tools](https://github.com/TexasInstruments/edgeai-tidl-tools) і [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo). TIDL інтегрується з ONNX Runtime, TensorFlow Lite та іншими середовищами виконання застосунків, передаючи підтримувані підграфи для обчислення.

TI публікує значно більше, ніж заяви про імпорт фреймворку: каталог містить перетворені моделі й метадані продуктивності для виявлення об'єктів, сегментації, оцінювання пози, класифікації, оцінювання глибини та інших завдань. TI також попереджає, що артефакти каталогу моделей є відправною точкою для розробки, а не автоматично готовими до виробництва ресурсами. Це важливе застереження, яке часто відсутнє у порівняннях постачальників.

TI також указує [TDA54-Q1](https://www.ti.com/product/TDA54-Q1) зі статусом Preview, із підтримкою до чотирьох NPU C7 і до 400 TOPS для цієї моделі; для ширшого сімейства TDA5 заявлено до 1,200 TOPS. Це заяви постачальника про нове покоління продуктів, а не підстава переносити валідацію моделей TIDL для AM6xA/TDA4 на TDA54-Q1 до публікації TI матриці для конкретної цілі.

### NXP

Для NXP потрібні щонайменше чотири окремі описи бекендів:

| Ціль NXP | Прискорювач | Шлях компілятора/середовища виконання |
|---|---|---|
| i.MX 8M Plus | NPU VeriSilicon Vivante на 2.3 TOPS | eIQ із TIM-VX/VX delegate |
| i.MX 93 | Arm Ethos-U65 | Квантизований TFLite й Arm Vela |
| i.MX 95 | NPU NXP eIQ Neutron | Neutron Converter і середовище виконання eIQ |
| Ara-1 / [Ara240](https://www.nxp.com/products/ARA240) | Дискретний NPU на основі Kinara; для Ara240 заявлено до 40 eTOPS | Ara SDK з інтеграцією в ширшу пропозицію eIQ |

У [eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) є ресурси або рецепти для YOLOv4-tiny, YOLOv8, NanoDet, CenterNet, FastestDet, SSD Lite, YOLACT та інших моделей; запис YOLOv5 додано лише як документацію. Валідований запис для одного бекенду не є доказом підтримки всіх чотирьох. Власний [посібник NXP з експорту YOLO для платформ i.MX](https://community.nxp.com/t5/i-MX-Processors-Knowledge-Base/Exporting-YOLO-Models-for-NXP-i-MX-Platforms/ta-p/2381361) демонструє цю проблему перетворення для окремих цілей. NXP повідомляє, що монолітний [eIQ Toolkit не оновлювався після версії 1.17 у Q3 2025 року](https://community.nxp.com/t5/eIQ-Machine-Learning-Software/eIQ-FAQ/ta-p/1099741); у поточних процесах використовують окремі пакети, як-от eIQ Neutron SDK.

NXP [завершила придбання Kinara в жовтні 2025 року](https://media.nxp.com/news-releases/news-release-details/nxp-completes-acquisitions-aviva-links-and-kinara-advance/). У [технічному описі Ara240](https://www.nxp.com/docs/en/fact-sheet/ARA240DNPUFS.pdf) вказано до 40 заявлених постачальником eTOPS і 313 зображень за секунду для YOLOv8n. На сторінці продукту NXP згадано Ara SDK та eIQ Toolkit, але це не підтверджує взаємозамінності артефактів з окремим [шляхом Neutron для i.MX 95](https://eiq.nxp.com/learning-hub/convQuant/neutron.html). [Модуль M.2 на 16 GB](https://www.nxp.com/design/design-center/development-boards-and-designs/ARA2-M2-16G-GT) активний, а варіант USB перебуває на передвиробничому етапі. NXP визначає «e» в eTOPS як «equivalent», тобто «еквівалентний», а не «effective», тобто «ефективний»; цей показник не є мірою щільних INT8 TOPS іншого постачальника й не має порівнюватися безпосередньо.

### STMicroelectronics

[Пристрої STM32N6x7](https://www.st.com/en/microcontrollers-microprocessors/stm32n6-series.html), зокрема STM32N657 і STM32N647, додають прискорювач Neural-ART на 600 GOPS у продукт класу MCU; у лінійці загального призначення N6x5 цього прискорювача немає. [STM32Cube AI Studio](https://www.st.com/en/development-tools/stedgeai-cubeai.html) і ST Edge AI Core аналізують та оптимізують імпортовані моделі, а [STM32 AI Model Zoo Services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) надає процеси розгортання, оптимізації, бенчмаркінгу й приклади.

У поточному [огляді виявлення об'єктів](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/object_detection/docs/README_OVERVIEW.md) задокументовано Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26 і ST-YOLOX, а також послуги для класифікації, оцінювання пози й сегментації. Це не той самий клас продуктивності, що й плата PCIe на 200 TOPS. Привабливість у тому, що отримання зображення з камери, інференс і керування можуть вкладатися в жорсткі обмеження енергоспоживання й пам'яті вбудованих систем.

### Renesas

Процесори Renesas RZ/V інтегрують прискорювачі DRP-AI. Програмний стек еволюціонував від DRP-AI Translator і DRP-AI TVM до [RUHMI](https://www.renesas.com/en/software-tool/ruhmi-framework), що працює на технології EdgeCortix MERA. У відкритому [репозиторії RZ/V DRP-AI TVM](https://github.com/renesas-rz/rzv_drp-ai_tvm) та [списку валідації RZ/V2H](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) названо варіанти YOLOv5, YOLOv8, YOLO11 і YOLO26 n/s/m, YOLOX, мережі оцінювання пози й сегментації, а також моделі класифікації.

Renesas є надійною ціллю для робототехніки й промисловості, але для відтворюваності потрібно записувати точну плату, покоління DRP-AI, версію Translator і післяобробку на CPU.

## Розумні сенсори й процесори для камер

### Sony IMX500

Sony IMX500 не є звичайним прикладним процесором. У ньому сенсор зображення поєднано з обробкою AI, щоб виконувати інференс у сенсорі й зменшити обсяг даних зображення, які потрібно передавати з камери. Raspberry Pi AI Camera робить цю архітектуру незвично доступною.

В офіційному [репозиторії моделей Raspberry Pi IMX500](https://github.com/raspberrypi/imx500-models) опубліковано пакети моделей YOLOv8n, YOLO11n, EfficientDet Lite0, NanoDet+ і SSD MobileNetV2 FPN Lite. У [документації AI Camera](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) пояснюються перетворення, пакування та післяобробка на сенсорі. Для розгортання використовують пакет RPK, а не загальний файл ONNX; важливими обмеженнями є пам'ять сенсора й набір операторів. На [сторінці продукту](https://www.raspberrypi.com/products/ai-camera/) Raspberry Pi зазначає, що виробництво цієї камери триватиме щонайменше до січня 2028 року.

### Ambarella

Процесори Ambarella CVflow глибоко вкорінені у камерах, дронах і автомобільних системах комп'ютерного зору. До поточних сімейств продуктів належать CV72/CV75 для камер, CV5/CV52 для потужніших систем машинного зору та автомобільні пристрої CV3-AD. Платформа розробників Cooper і процес компіляції/виконання CVflow утворюють названий у відкритих джерелах програмний стек.

У [публічному модельному каталозі для розробників](https://www.ambarella.com/developer/model-garden/) названо YOLOX, RTMDet, DeepLabV3+, TopFormer, OWL-ViT і мультимодальні моделі. Водночас докладна документація компілятора й завантаження орієнтовані на партнерів. Тому інтеграцію Ambarella варто починати зі зв'язку з постачальником або OEM-виробником, а не з припущення, що існує публічний пакет pip.

### Synaptics Astra

Для Synaptics важливі три цілі. Системи Astra SL1600/SL1680 використовують набір інструментів [SyNAP](https://developer.synaptics.com/docs/synap/introduction), який перетворює вихідну модель разом з описом YAML на пакет `model.synap`. Новіші пристрої SL2611/13/15/17/19 використовують [платформу Torq](https://developer.synaptics.com/docs/torq/introduction), процес на основі IREE/MLIR, що створює `.vmfb`. Сімейство [SR100](https://developer.synaptics.com/docs/sr/introduction-sr) є окремим сімейством MCU на Cortex-M55 та Ethos-U55 із власним SDK. Ці артефакти й API не взаємозамінні.

Офіційний [посібник із бенчмарку YOLO](https://developer.synaptics.com/docs/sl/tutorials/vision/benchmark-yolo) незвично докладний: він експортує YOLOv8 у TFLite, виконує асиметричне калібрування UINT8 у SyNAP, компілює модель для SL1680, а потім запускає `synap_cli` і застосунок виявлення об'єктів. Це підтверджує роботу YOLOv8n/v8s на SL1680 і описує підтримку SyNAP до YOLO11. Окремий [посібник із виявлення об'єктів для SL261x](https://developer.synaptics.com/docs/sl/sl2600/getting-started/object-detection) запускає YOLOv8 із Torq `.vmfb`. Це офіційні приклади для конкретних цілей, а не твердження, що всі графи YOLO підтримуються в усіх трьох сімействах.

### MediaTek Genio

MediaTek заслуговує на окремий розділ, оскільки її поточний [IoT AI Hub](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) тепер показує матрицю обладнання/програмного забезпечення, пакети моделей і таблиці бенчмарків, які раніше не можна було перевірити в оглядах ринку. Genio 360/360P/420/520/720 використовують NeuroPilot 8 з MDLA 5.3; Genio 510/700 використовують NP6 з MDLA 3.0; Genio 1200 використовує NP6 з MDLA 2.0. MediaTek зазначає, що покоління NeuroPilot і MDLA, набір операторів, компілятор і середовище виконання прив'язані до версій протягом життєвого циклу кожного SoC.

Шлях аналітичного AI побудовано навколо TFLite:

```text
PyTorch model
    -> NeuroPilot Converter and representative calibration
    -> quantized .tflite
    -> version-matched ncc-tflite compiler
    -> chip-generation-specific .dla
    -> Neuron Runtime on MDLA
```

У новіших Genio також доступні онлайн-делегування LiteRT і шлях ONNX Runtime CPU/NPU у підтримуваних операційних системах. Це інші режими виконання, ніж офлайн `.dla`. У [посібнику з архітектури програмного забезпечення](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/software_architecture.html) і [матриці ресурсів](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) це розрізнення сформульовано чітко.

В офіційній [таблиці аналітичних моделей](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical.html) опубліковано записи бенчмарків YOLOv5s і YOLOv8s та посібники з перетворення для кількох поколінь MDLA. Там прямо зазначено, що попередньо перетворені артефакти YOLO не поширюються через обмеження AGPL-3.0. Вимірювання Quant8, 640 x 640 в офлайн-режимі включають 5.35 ms і 8.04 ms відповідно на Genio 720. На [сторінці моделі YOLOv8s](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical/YOLOv8s.html) опубліковано вхідні/вихідні тензори, час для конкретних бекендів і попередження щодо користувацьких операцій MediaTek. Це бенчмарки постачальника, а не результати повного процесу роботи камери.

Доступ є обмежувальним чинником. Публічна документація Yocto докладна, але поточний універсальний пакет конвертера/компілятора NP8 і значна частина матеріалів Android позначені як NDA або ресурси для прямих клієнтів. Звичайний обліковий запис розробника не надає такого доступу, як клієнтський обліковий запис MediaTek Online. Тому LibreYOLO варто вважати Genio технічно перевіреним, але залежним від партнерства для відтворюваної інтеграції компілятора з користувацькими моделями.

## Екосистеми Edge AI, орієнтовані на Китай

Кілька найпотужніших і найдоступніших недорогих платформ машинного зору погано представлені в англомовних оглядах ринку.

### D-Robotics і платформи BPU на основі Horizon

D-Robotics підтримує екосистему плат розробників RDK навколо прискорювачів BPU, які використовуються в RDK X3, X5, Ultra та новіших продуктах серії S100. Поточна гілка `rdk_x5` у [RDK Model Zoo](https://github.com/D-Robotics/rdk_model_zoo) документує шляхи YOLOv5/v5u, YOLOv8/9/10/11/12/13/26, YOLOE, YOLO-World, виявлення, сегментації, оцінювання пози, класифікації, OCR і CLIP для RDK X5. Для X3 є окрема гілка, а поточне постачання серії S міститься в гілці [`rdk_s`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s) основного каталогу; старіший репозиторій [`rdk_model_zoo_s`](https://github.com/D-Robotics/rdk_model_zoo_s) містить історичні демо. Отже, сучасний перелік X5 не доводить валідацію всіх моделей для X3, Ultra або S100.

OpenExplorer/Algorithm Toolchain імпортує ONNX або Caffe для PTQ і підтримує процеси QAT, орієнтовані на PyTorch. Розгортання залежить від платформи: поточний RDK X5 використовує `.bin` через `hbm_runtime` поверх `libdnn`, а поточний основний процес `rdk_s` використовує `.hbm` через однойменний API `hbm_runtime` поверх `libhbucp`; для X3 збереглися старіша гілка й інтерфейси інференсу. У старіших матеріалах `rdk_model_zoo_s` також описано історичні шляхи `.bin` і `.hbm`, тому ці артефакти потрібно залишати у відповідності до їхньої гілки й набору інструментів. Непідтримувані оператори можуть перейти на CPU. Публічні приклади корисні, але відповідність версій RDK OS, прошивки плати, набору інструментів і бінарної моделі залишається частиною контракту сумісності.

### AXERA

Сімейства AX650/AX630/AX620 від AXERA використовуються в компактних AI-камерах і платах на зразок лінійки MaixCAM від Sipeed. Pulsar2 імпортує ONNX, виконує калібрування й аналіз точності та створює `.axmodel`; артефакт запускає AXEngine.

В [офіційних прикладах AXERA](https://github.com/AXERA-TECH/ax-samples) є YOLOv5/6/7/8/9/10/11/13/26, YOLOX і YOLO-World; підтримка завдань і чипів різниться залежно від платформи. На підтримуваних цілях поточні приклади YOLO26 включають виявлення, оцінювання пози, сегментацію й OBB. У [прикладах перетворення Pulsar2](https://pulsar2-docs.readthedocs.io/en/latest/appendix/model_convert_examples.html) показано реальну роботу: точні назви тензорів, точки поділу виходів, перетворення розкладки, архіви для калібрування та налаштування цільового обладнання.

### SOPHGO

TPU-MLIR від SOPHGO є одним із найпривабливіших стеків компілятора для незалежної інтеграції, оскільки сам компілятор має відкритий код. Він імпортує ONNX, PyTorch, TFLite і Caffe, перетворює та квантує графи, створюючи `.bmodel` для цілей BM або `.cvimodel` для обладнання CV18xx.

Проєкт [TPU-MLIR](https://github.com/sophgo/tpu-mlir) підтримує процеси FP32, BF16, FP16 і INT8 залежно від цілі. У [колекції демо SOPHON](https://github.com/sophgo/sophon-demo/blob/release/README_EN.md) названо YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, варіанти OBB/сегментації, SSD, CenterNet, RetinaFace, SAM/SAM2 та OCR. Як завжди, демо на BM1684X не валідовує той самий артефакт на BM1688 або CV18xx.

### Huawei Ascend

Лінійка edge-інференсу Huawei охоплює чипи Ascend 310/310P/310B у продуктах Atlas 200I і 300I. CANN надає компілятор ATC, середовище виконання AscendCL і ядра операторів для конкретного чипа; ATC перетворює ONNX та інші вихідні представлення на офлайн-модель `.om`.

У поточній [документації Atlas 200I DK A2](https://www.hiascend.com/en/hardware/developer-kit-a2/specification) і прикладах CANN є шлях `.om` для YOLOv5 на Ascend 310B. Ширший, старіший [Ascend ModelZoo](https://github.com/Ascend/modelzoo) містить YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN, а також моделі оцінювання пози й сегментації, але його не слід подавати так, ніби кожен запис повторно валідовано для 310B. CANN, ядра, прошивка та SoC мають відповідати одне одному. `.om` для Ascend310P не є універсальним артефактом Ascend.

### Cambricon

Прискорювачі MLU від Cambricon використовують Neuware і рушій інференсу MagicMind. Публічний репозиторій [MagicMind Cloud](https://github.com/Cambricon/magicmind_cloud) закріплено на MagicMind 1.7 і матриці MLU370-X4/S4; це свідчення історичної сумісності, а не матриця поточного SDK чи валідація MLU270. У репозиторії задокументовано шляхи PyTorch, ONNX, Caffe і Paddle; підтримку TensorFlow у фреймворку вилучено у версії 1.7, хоча історичні приклади TensorFlow залишилися. У поточному [порталі розробників серії 3](https://developer.cambricon.com/index/document/index/classid/3.html) наведено новіші загальні релізи SDK, тому публічний репозиторій прикладів і поточний комерційний стек не можна подавати як одну версію.

Офіційний [хмарний репозиторій моделей MagicMind](https://github.com/Cambricon/magicmind_cloud) публікує надзвичайно конкретну матрицю MLU370-X4/S4. У ній наведено приклади YOLOv3, Tiny-YOLOv3, YOLOv4, YOLOv5, YOLOv7, YOLOv8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN, DeepLab і UNet, зокрема зазначено наявність прикладів C++ або Python. Докази переконливі; головна перешкода полягає в доступі до SDK і контейнера через канали Cambricon.

### Canaan/Kendryte

Поточні K230/K230D і застарілі чипи KPU K210/K510 від Kendryte важливі для недорогих плат і розумних камер. Компілятор [nncase](https://github.com/kendryte/nncase) імпортує ONNX або TFLite, використовує калібрувальні дані для перетворення з фіксованою точкою, симулює результат на хості й створює `.kmodel`.

В офіційному [посібнику з розробки nncase для K230](https://github.com/kendryte/k230_docs/blob/main/en/01_software/board/ai/K230_nncase_Development_Guide.md) показано компіляцію, симуляцію й запуск YOLOv5s. В окремому [каталозі демо AI](https://github.com/kendryte/k230_docs/blob/main/en/02_applications/ai_demos/K230_AI_Demo_Introduction.md) є артефакти виявлення, сегментації та оцінювання пози YOLOv8n, а в поточному [журналі змін CanMV](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md) додано оптимізовані приклади класифікації, виявлення, сегментації, OBB й оцінювання пози для YOLOv8/11/26. [Результат YOLOv5s](https://github.com/kendryte/k230_docs/blob/main/en/00_hardware/K230_datasheet.md) у даташиті є опублікованим бенчмарком; версії компілятора й двійкового плагіна KPU мають відповідати SDK плати.

### Allwinner

У V853 від Allwinner поєднано мультимедійне обладнання для камер із Vivante NPU на 1 TOPS. В [офіційному посібнику NPU англійською](https://docs.aw-ol.com/v853/en/npu/dev_npu/) описано імпорт Acuity/Pegasus, квантування, перевірку й розгортання через `viplite`. На власних сторінках моделей Allwinner і в [посібнику YOLOv5](https://docs.aw-ol.com/v853/npu/npu_yolov5/) названо YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet та мережі розпізнавання облич і людей.

Це справжня підтримка комп'ютерного зору, але вона залишається прив'язаною до старішого набору інструментів Tina Linux/Vivante, а для завантаження SDK потрібен обліковий запис. Платформа належить до огляду ринку з чітко зазначеним обмеженням.

## Недооцінені платформи Кореї, Тайваню та Китаю

В англомовних оглядах часто переходять від Rockchip одразу до NVIDIA, пропускаючи другий рівень реального й задокументованого кремнію. Деякі з цих екосистем мають ширші поточні матриці YOLO, ніж відоміші західні стартапи.

### Mobilint

Корейський постачальник прискорювачів Mobilint продає малопотужний REGULUS і продуктивніший ARIES MLA100. Його qb SDK приймає дані PyTorch, TensorFlow, TFLite, ONNX і Keras-сумісні входи, квантує до INT8 і створює скомпільований артефакт `.mxq`. У публічній [матриці комп'ютерного зору](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md) названо виявлення YOLOv3/v5/v7/v8/v9/v10/11/12/26, а також варіанти сегментації, оцінювання пози й OBB для YOLO. На [сторінці ARIES](https://www.mobilint.com/aries/mla100) навіть опубліковано окремі результати для YOLO11s і YOLO26m. Це переконливі свідчення можливості інтеграції, хоча доступ до SDK/обладнання залишається комерційним.

### Sunplus

У SP7350/C3V від Sunplus використовується стек NPU на основі Vivante, але його запаковано інакше, ніж в Allwinner або старого Amlogic. Публічний процес поєднує перетворення Acuity, середовище Docker для NPU, середовище виконання SNNF і результат `.nb`. В офіційному [посібнику з власної реалізації YOLOv8](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350) описано перетворення й розгортання, а не лише заявлено підтримку фреймворку. Споріднений IP не робить `.nb` переносним на інший SoC на базі Vivante.

### ESWIN Computing

До сімейства RISC-V від ESWIN належать EIC7700/EIC7700X і EIC7702/EIC7702X із двома кристалами; EIC7700 використовується в платах у продажу, зокрема Milk-V Megrez. ENNP містить EsQuant, поточний компілятор EsAAC, генерування еталонних даних, симуляцію й середовище виконання ESSDK зі скомпільованим результатом `.model`; у старіших матеріалах ENNP використовувалася назва `ennc-compile`. На поточній [сторінці EsAAC](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) задокументовано вхід ONNX, тож для інших фреймворків потрібен експорт або перетворення на підтримуване проміжне представлення. Розміщені на Milk-V [огляд ENNP](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/introduction) і [посібник YOLOv3](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3) підтверджують публічний наскрізний процес, але не широку сучасну матрицю моделей/точності.

### Nuvoton M55M1

M55M1 від Nuvoton є конструкцією MCU-класу Cortex-M55 плюс Ethos-U55-256, а не прикладним процесором Linux. NuEdgeWise/NuML і Arm Vela готують квантизовані моделі TFLite Micro. У публічному репозиторії [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise) названо YOLOv8-nano, YOLOX-nano, YOLO Fastest v1.1, SSD-MobileNet FPNLite та кілька класифікаторів. Ці приклади невеликих мереж є переконливим свідченням відповідності класу пам'яті/потужності пристрою, а не доказом для стандартних варіантів YOLO із роздільною здатністю 640 пікселів.

### Rebellions і FuriosaAI

У лінійці ATOM від Rebellions використовуються SDK RBLN і скомпільовані артефакти `.rbln`. В [офіційному релізі підтримки](https://docs.rbln.ai/v0.8.2/supports/release_note.html) названо YOLOv3 tiny/full/SPP, YOLOv5 від n до x, YOLOv6 від n до l, варіанти YOLOv7 і YOLOv8 від n до x. У поточній [матриці підтримки плат](https://docs.rbln.ai/latest/supports/version_matrix.html) ATOM CA02 і ATOM+ CA12 позначено EoL, а ATOM+ CA22 та ATOM-Max CA25 мають статус Active. Для ATOM-Lite CA21 є сторінка продукту, але його немає в матриці SDK, тому поточний статус набору інструментів неясний. Документація публічна, але для колеса компілятора потрібні облікові дані Rebellions Portal.

FuriosaAI слід розглядати за поколіннями. Warboy є старішим прискорювачем комп'ютерного зору з артефактами INT8 `.enf` та [офіційним каталогом моделей](https://developer.furiosa.ai/furiosa-models/latest/), де є SSD, YOLOv5M/L і YOLOv7-w6-pose. RNGD використовує інший [стек `.fxb`](https://developer.furiosa.ai/latest/en/furiosa_llm/fxb.html), переважно орієнтований на навантаження LLM/VLM. У поточній [дорожній карті](https://developer.furiosa.ai/latest/en/overview/roadmap.html) зазначено, що підтримку машинного зору YOLOv8m завершено в 2024 Q4, але поточні публічні сторінки підтримуваних моделей і продуктивності зосереджені на LLM/VLM і не містять докладної матриці точності/продуктивності YOLOv8m. Це свідчення випущеного покоління, а не заява про сумісність із Warboy.

### Realtek, Telechips, T-Head і SigmaStar

Ці чотири платформи реальні, але мають вужчі, старіші або закриті можливості для користувацьких моделей:

| Платформа | Відтворювані публічні свідчення | Обмеження, яке слід зберегти |
|---|---|---|
| Realtek AmebaPro2 | В [Arduino/FreeRTOS SDK](https://github.com/Ameba-AIoT/ameba-arduino-pro2) у пакеті є моделі YOLOv3/4/7-tiny, SCRFD і MobileFaceNet `.nb` | Щоб перетворювати власні моделі, потрібно зареєструвати зацікавленість/звернутися до Realtek |
| Telechips TOPST TCC7500 | В [офіційному проєкті 2026 року](https://docs.topst.ai/blog/31) перетворено й розгорнуто YOLOv8s | Перетворення UFLD v1/v2 не вдалося через непідтримувані шари; у статті лише запропоновано обхідний шлях із розділенням/післяобробкою. Щоб отримати повні ресурси компілятора/операторів, зазвичай потрібен дозвіл електронною поштою |
| T-Head TH1520 | У [посібнику із застосунків Sipeed](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) запускають YOLOv5n/s за допомогою HHB і CSI-NN2/SHL | Стек публічний, але супровід і ясність версій відстають від поточних лідерів |
| SigmaStar SSU9383CM | У поточній документації наведено API середовища виконання MI_IPU; старіші матеріали SGS_IPU описують `.sim` → `sgsimg.img`, а старий постпроцесор називає SSD і YOLOv1/2/3 | Публічних свідчень, що старі артефакти компілятора або названі моделі працюють на SSU9383CM, немає |

### Розумний зір HiSilicon не є Huawei Ascend

SoC камер Hi3516CV610/DV500, Hi3519DV500 і Hi3403V100 від HiSilicon надають процес ATC у `.om` через середовища виконання NNN/SVP-NNN. В офіційному [каталозі моделей HiSpark](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) у матриці для конкретних цілей, зокрема Hi3403 і Hi3591P, названо YOLOv3/4/5/6/7/8/9/10/11, сегментацію/оцінювання пози YOLO11, OBB/World/сегментацію YOLOv8, OCR, оцінювання глибини й TinySAM. Це не доводить, що кожен запис працює на кожному з перелічених вище SoC для розумного зору. Частина записів є пунктами дорожньої карти, тому випущені приклади й заплановану підтримку потрібно розділяти; крім того, у репозиторії вказано, що надані моделі ModelZoo призначено лише для некомерційного використання. Спільні позначення ATC/`.om` не доводять сумісність артефактів або середовища виконання з лінійкою Ascend на основі CANN.

## Компанії з новими та спеціалізованими прискорювачами

### MemryX

Модулі MX3 від MemryX використовують архітектуру потоку даних і можуть поєднувати кілька чипів. [Neural Compiler](https://developer.memryx.com/tools/neural_compiler.html) приймає моделі ONNX, TensorFlow Lite, Keras і TensorFlow та створює пакет DFP, який використовує середовище виконання прискорювача. Його [API середовища виконання](https://developer.memryx.com/api/accelerator/accelerator.html) підтримують багатомодельні потокові пайплайни.

Документація й приклади MemryX охоплюють оптимізовані попередню та післяобробку YOLO для виявлення, сегментації й оцінювання пози. У [примітках до релізів SDK](https://developer.memryx.com/release_notes.html) прямо названо підтримку YOLOv10, YOLO11 і YOLO26. Публічні матеріали технічно корисні, хоча в них немає єдиної простої таблиці моделей/точності/пристроїв на зразок Axelera.

### Kneron

Продукти Kneron KL520/KL530/KL630/KL720/KL730 охоплюють невеликі прискорювачі USB і вбудовані рішення. Kneron PLUS охоплює середовище виконання застосунків, а поточна версія Model Toolchain 0.33.1 документує для цих цілей перетворення, квантування, оцінювання, симуляцію та компіляцію у `.nef`. KL830 трапляється в окремих [API середовища виконання PLUS](https://doc.kneron.com/docs/plus_c/introduction/run_examples/), але відсутній у поточному [списку цілей компілятора](https://doc.kneron.com/docs/toolchain/manual_1_overview/); не слід поширювати загальну заяву про компіляцію `.nef` на KL830 без підтвердження постачальника.

В офіційному [прикладі YOLO](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/) показано процес на Tiny-YOLOv3; інші матеріали Kneron охоплюють навчання й розгортання YOLOv5. Докази переконливі, але вужчі за широкі сучасні матриці YOLO від Hailo, Axelera, Rockchip або DEEPX.

### SiMa.ai

У MLSoC від SiMa.ai та [серійній платформі Modalix на 50 TOPS](https://sima.ai/press-release/sima-ai-next-gen-platform-for-physical-ai-in-production/) використовується програмне середовище Palette. ModelSDK, компілятор MLA і ModelExecutor охоплюють імпорт, квантування, компіляцію, профілювання й виконання. Залежно від робочого навантаження й продукту інструменти підтримують процеси INT8, INT16 і BF16.

У примітках до релізів компанії названо валідовані пайплайни YOLOv7, виявлення/оцінювання пози/сегментації YOLOv8, YOLOX і сегментації YOLOX, DETR, Mask R-CNN та EfficientDet. Не слід приховувати одну поточну проблему: у [примітках до Palette SDK 2.1](https://docs.sima.ai/v2.1.2/pages/release_notes/2.1.html) сказано, що QAT не працює через регресію квантування Python/PT2E. Задокументований обхідний шлях: створити анотований ONNX у SDK 2.0 і скомпілювати його у 2.1. Доступ і придбання орієнтовані на корпоративних клієнтів.

Межу розгортання також задокументовано: [процес компіляції ModelSDK](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) створює скомпільований `.tar.gz` і може створювати виконуваний ELF, тоді як [MPK Tool](https://docs.sima.ai/v2.1.1/pages/palette/mpk_tools.html) пакує артефакт для розгортання `.mpk`. Ці вихідні файли прив'язані до цілі MLSoC/Modalix і релізу Palette.

### EdgeCortix

SAKURA-II є прискорювачем на 60 заявлених TOPS для машинного зору й генеративного AI, який компілюється через програмний фреймворк MERA. EdgeCortix також надає технологію MERA для новішого процесу RUHMI від Renesas.

У матеріалах про продукт [SAKURA-II](https://www.edgecortix.com/en/edgecortix-sakura-ii-accelerator-brief) описано обладнання й компілятор, а [обладнання M.2/PCIe](https://www.edgecortix.com/en/hardware) пропонують для тестування або замовлення за запитом. Достатньо точної актуальної публічної матриці сумісності YOLO не знайдено. Для закупівлі це важлива інформація: перед придбанням обладнання слід запросити валідацію моделі.

### BrainChip

Akida від BrainChip є нейроморфним процесором подій, а не звичайним NPU для щільних тензорів. [Середовище MetaTF](https://brainchip.com/metatf-dev-tools/) складається з інстальованих пакетів Python для створення моделей, квантування з малою кількістю бітів, перетворення, симуляції та виконання на обладнанні. Тепер BrainChip продає [супроцесор AKD1500 у форматі M.2](https://brainchip.com/brainchip-akd1500-now-available-in-compact-m-2-form-factor-enabling-fanless-edge-ai-in-industrial-and-commercial-designs/), який є в продажу, поряд із застарілим обладнанням AKD1000 та IP Akida 2.

У поточній [картці моделей Akida 2](https://brainchip.com/wp-content/uploads/2025/04/Akida-2-Model-Card-V1.1-Mar.25.pdf) названо детектор YOLOv2 AkidaNet0.5, CenterNet, AkidaUNet і мережі розпізнавання облич. Інші матеріали підтримки охоплюють FOMO і робочі навантаження на основі подій. Akida підтримує незвично низьку розрядність ваг і активацій, але для успішного розгортання може знадобитися перетворення з урахуванням архітектури, а не ставлення до нього як до чергової цілі загального призначення INT8 ONNX. Результат каталогу моделей AKD1000 або Akida 2 не слід приписувати AKD1500, доки його не підтвердить явний звіт зіставлення/відповідності.

### Blaize

Blaize продає продукти Pathfinder і Xplorer на базі P1600; шлях програмного забезпечення забезпечують Picasso SDK, NetDeploy і AI Studio. На [сторінках продуктів](https://www.blaize.com/products/) чітко зазначено цільове призначення для комп'ютерного зору й edge AI, але актуальної публічної матриці сумісності окремих моделей немає.

Тому в цьому посібнику Blaize класифіковано як комерційну платформу із закритим доступом, а не доповнено твердженнями спільноти. До початку робіт слід отримати від постачальника звіт компіляції й точності для потрібної моделі.

## TinyML і машинний зір на кінцевих пристроях

### Arm Ethos-U та Alif

Arm ліцензує компаніям-виробникам чипів IP NPU Ethos-U55, U65 і U85. Компілятор [Vela](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) бере квантизовані графи LiteRT/TFLite й переписує підтримувані підграфи у власні операції Ethos-U. Непідтримувані операції можуть залишитися на CPU, отже «застосунок працює» і «вся мережа працює на NPU» є різними твердженнями.

Реальні реалізації включають Ethos-U55 в Alif Ensemble E7, Ethos-U65 у NXP i.MX 93 і Ethos-U85 у новіших системах. У [E7 AI/ML AppKit](https://alifsemi.com/support/kits/ensemble-e7appkit/) від Alif задокументовано два прискорювачі ML і продемонстровано процес E7 TFLite-Vela; у [Ensemble E8](https://alifsemi.com/ensemble-e8-series/) поєднано один Ethos-U85 із двома NPU Ethos-U55. Alif також публікує бенчмарк детектора облич YOLO-Fastest усього сімейства [INT8](https://alifsemi.com/faster-ai-mcu-inferencing-low-power-consumption/) з роздільною здатністю 192 на 192 на Cortex-M55 плюс Ethos-U55, але широкої матриці сучасних YOLO для конкретних цілей немає. Ці системи з обмеженою пам'яттю підходять для невеликих мереж класифікації й виявлення, а не для довільних детекторів на 640 пікселів лише тому, що їх можна представити у TFLite.

### Infineon PSOC Edge, Himax WiseEye2 і Analog Devices MAX7800x

У сімействах [PSOC Edge E83](https://documentation.infineon.com/psocedge/docs/xsk1761304020519) і E84 від Infineon поєднано Cortex-M55 з Ethos-U55 і додано окремий блок NNLite на малопотужному боці M33. В [архітектурному посібнику E84](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) описано 8-бітні ваги, 8- або 16-бітні активації та процес Vela, у якому підтримувані оператори стають потоком команд NPU, а непідтримувана робота залишається на CPU. [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter) приймає шляхи TFLite, Keras і PyTorch, а в архітектурних матеріалах MobileNetV1/V2 названо типовими ядрами, а не бенчмарками цілі. До [E84 AI Kit](https://documentation.infineon.com/psocedge/docs/cci1762693051052) входить камера; комплект використовує ModusToolbox, але в публічних матеріалах поки немає матриці сумісності з названими моделями YOLO. Це справжня програмована платформа машинного зору з екосистемою моделей, що розвивається, а не валідована ціль для детекторів загального призначення.

У [HX6538 WiseEye2](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) від Himax також поєднано Cortex-M55 та Ethos-U55 для постійного аналізу зору. Публічна історія програмного забезпечення незвично конкретна для такого класу потужності: в [офіційних прикладах Grove Vision AI Module V2](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) є виявлення об'єктів і оцінювання пози YOLOv8n, класифікація статі, виявлення YOLO11n, face mesh і PeopleNet. Уніфікований шлях використовує TFLite Micro і Vela, а за потреби переходить на CMSIS-NN та еталонні ядра. Це навмисно невеликі моделі й роздільні здатності, а не доказ, що звичайний серверний граф YOLO поміститься на пристрої.

Analog Devices використовує інший підхід із [прискорювачами CNN MAX78000/MAX78002](https://www.analog.com/en/products/max78002.html). Публічний інструмент [`ai8x-synthesis`](https://github.com/analogdevicesinc/ai8x-synthesis) квантує навчену мережу й генерує специфічний для цілі код C, ваги й конфігурацію прискорювача замість переносного пакета середовища виконання ONNX. Свідчення від першої сторони включають [ідентифікацію облич](https://www.analog.com/en/resources/app-notes/an-2616.html), виявлення QR-кодів TinierSSD у [каталозі моделей ADI](https://ez.analog.com/dsp/software-and-development-tools/edgebench/a/docs-faqs/DF912/available-models-in-adi-model-zoo) та класифікатори зображень. Ці пристрої доречні для машинного зору на кінцевих пристроях, але актуальної офіційної матриці сучасних YOLO не знайдено.

### GreenWaves GAP9

GAP9 поєднує обчислення RISC-V з нейронним рушієм NE16. NNTool імпортує й квантує мережі, а AutoTiler і GAP SDK створюють код для розгортання. У публічному [меню нейронних мереж GAP9](https://github.com/GreenWaves-Technologies/nn_menu_gap9) є застосунки MobileNet, EfficientNet, ResNet, MobileNet SSD, обробки облич і розпізнавання. GreenWaves також публікує [проєкт виявлення людей YOLOX](https://github.com/GreenWaves-Technologies/yolox_people_detection) із даними точності та симуляції.

Це незвично прозорі свідчення для TinyML, але повний SDK доступний лише кваліфікованим клієнтам, а моделі значно менші за типові варіанти YOLO 640 на 640.

### Lattice sensAI

sensAI від Lattice є альтернативою фіксованому NPU на основі FPGA. sensAI Studio і Neural Network Compiler зіставляють підтримуваний граф із конфігурованим IP прискорювача для пристроїв ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E та Avant-X. Результат прив'язаний до вибраної FPGA, режиму прискорювача, розкладки пам'яті й бітового образу, а не є переносною моделлю для середовища виконання.

У поточному [посібнику Neural Network Compiler](https://www.latticesemi.com/-/media/LatticeSemi/Documents/UserManuals/MQ3/FPGA-UG-02052-8-0-Lattice-Neural-Network-Compiler-Software.ashx?document_id=52343) є незвично пряма таблиця топологій. У ній наведено YOLOv1 для ECP5, YOLOv5 і YOLOv8 для оптимізованих/розширених режимів CrossLink-NX та CertusPro-NX, а YOLO11 лише для IP режиму Advanced. Також наведено SSD, MobileNetV2-SSD, ResNet, ENet, SqueezeDet та інші мережі комп'ютерного зору з видимими непідтримуваними комірками для кожного сімейства FPGA. [Advanced CNN Accelerator](https://www.latticesemi.com/en/Products/DesignSoftwareAndIP/IntellectualProperty/IPCore/IPCores05/Advanced-CNN-Accelerator-IP) призначено для Avant-E/Avant-X/CertusPro-NX і є комерційним IP. Це свідчення з таблиці компілятора, а не обіцянка, що один бітовий образ одночасно підтримує всі перелічені топології.

### Microchip VectorBlox

Публічний [VectorBlox SDK 3.1](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) від Microchip виконує попередню обробку й компілює повністю квантизовані мережі TFLite INT8 для конфігурованого IP CoreVectorBlox у FPGA PolarFire SoC. `tflite_preprocess` і `vnnx_compile` створюють ресурси для розгортання `.vnnx`, `.hex` і `.ucomp`; до репозиторію також входять симулятор і драйвер C для цільового пристрою. Графи TensorFlow, ONNX і OpenVINO все одно мають пройти задокументований процес підготовки TFLite/INT8.

У поточній [матриці посібників](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md) є YOLOv5n, виявлення/класифікація/OBB/оцінювання пози/сегментація YOLOv8n, YOLOv9t, розріджені/стиснені варіанти YOLO, MobileNet, EfficientNet-Lite0, ResNet18, MiDaS, FFNet і QuickSRNet. В [посібнику з післяобробки C](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/docs/C_Postprocessing.md) окремо названо декодери YOLOv2/3/4/5 і Ultralytics для виявлення, оцінювання пози й OBB. Поточна версія SDK 3.1 підтримує PolarFire SoC Video Kit і прямо не підтримує PolarFire Video Kit без SoC, тому старі демо для плат не слід плутати з поточним контрактом цілі.

SDK публічно доступний за [ліцензією Microchip](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/LICENSE.md), яка обмежує використання програмного забезпечення та похідних робіт продуктами Microchip. Microchip пропонує безплатні ліцензії Libero Silver і CoreVectorBlox за запитом. Розгортання все ще є системним контрактом FPGA: конфігурація прискорювача, бітовий образ, прошивка, згенеровані SDK дані мережі й розкладка пам'яті мають відповідати одне одному. Успішна компіляція VectorBlox не перетворює модель на бінарний файл, який можна скопіювати на довільну конструкцію PolarFire.

### Syntiant

NDP200 від Syntiant призначено для постійного інференсу зору, аудіо й сенсорних даних за енергоспоживання, нижчого за звичайні SoC Linux. У поточній [таблиці обладнання](https://www.syntiant.com/hardware) NDP200 позначено як серійне виробництво, а NDP250 як зразок. У [технічному описі NDP200](https://www.syntiant.com/ndp200/) задокументовано підтримку CNN, RNN і повнозв'язних мереж за потужності нижче 1 mW; [анонс NDP250](https://www.syntiant.com/news/syntiant-unveils-ndp250-neural-decision-processor-with-next-gen-core-3-architecture) натомість описує постійне розпізнавання зображень із потужністю нижче 30 mW. Твердження «нижче мілівата» не можна узагальнювати на обидві моделі.

Публічні матеріали не підтверджують широку сумісність із YOLO, тому продукт слід оцінювати для невеликих класифікаторів і детекторів постійної готовності за допомогою оцінювання моделі, яке підтримує постачальник, а не припущення про загальний експорт ONNX.

### Google Coral: два не пов'язані між собою покоління

Для застарілих продуктів Edge TPU використовуються Edge TPU Compiler, `libedgetpu` і PyCoral з повністю INT8-моделями TFLite. Основні репозиторії [Edge TPU](https://github.com/google-coral/edgetpu) і [PyCoral](https://github.com/google-coral/pycoral) заархівовано. Це реальний ризик для супроводу, але не офіційне повідомлення про завершення життєвого циклу обладнання.

Google також підтримує окремий активний IP [Coral NPU](https://github.com/google-coral/coralnpu) із відкритим кодом: прискорювач RISC-V для інтеграції в малопотужні SoC. У поточному публічному проєкті доступні RTL, апаратна симуляція, набір інструментів і приклади ELF. Це не прямий наступник Edge TPU у форматі USB/M.2, і для нього не опубліковано поточної матриці розгортання YOLO.

## Пастки суміжних GPU, клієнтських NPU й адаптивних обчислень

NVIDIA, AMD, Intel і Apple часто з'являються в пошуку NPU, але не надають одного взаємозамінного класу прискорювачів.

**NVIDIA Jetson:** Jetson Orin поєднує GPU CUDA зі спеціальними ядрами DLA. TensorRT може зібрати серіалізований рушій для DLA, але непідтримувані шари виконуються на GPU лише за явно ввімкненого резервного використання GPU. У поточній [таблиці DeepStream YOLO](https://github.com/NVIDIA-AI-IOT/deepstream_tools/blob/main/yolo_deepstream/README.md) є YOLOv4/v7/v8/v9/11, але більшість записів призначено для GPU; явний запис ONNX DLA стосується YOLOv8s INT8. В [обмеженнях DLA від NVIDIA](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html) пояснено обмеження операторів. Thor також має окрему властивість: у [посібнику з переходу DriveOS](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/NVIDIA_DriveOS_7.0.3_Migration_Guide.pdf) NVIDIA зазначає, що в Thor ядра DLA прибрали заради гнучкішого планування GPU. Отже, фраза «працює на Jetson» не вказує, який прискорювач використовується.

**AMD Vitis AI:** у старішому поколінні Kria/DPU скомпільовані графи `.xmodel` виконуються через VART. Поточна версія [Vitis AI 6.2](https://vitisai.docs.amd.com/en/6.2/) має статус GA для двох окремих шляхів Versal AI Edge: [Gen1](https://vitisai.docs.amd.com/projects/gen1/en/latest/index.html) на VEK280/VE2802 використовує ONNX, AMD Quark і цільовий процес зі знімком/підграфами; [Gen2](https://vitisai.docs.amd.com/projects/gen2/en/latest/index.html) на VEK385 має власний контракт компіляції й середовища виконання. У Gen1 наведено приклади YOLOv5, YOLOv7, YOLOv8 і YOLOX. Ryzen AI є четвертим окремим стеком клієнтського NPU. Не переносіть артефакти або валідацію між застарілим DPU, Versal Gen1, Versal Gen2 і Ryzen AI лише тому, що вони мають назву Vitis або AMD.

**Intel OpenVINO:** OpenVINO може спрямовувати обчислення на CPU, GPU й NPU Core Ultra через один API. У [документації плагіна NPU Intel](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) перелічено підтримувані покоління NPU, а в [матриці перевірених моделей](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html) є окремі стовпці для пристроїв. Ноутбук YOLO доводить наявність рецепта застосунку, а не валідацію NPU, якщо цього не підтверджує матриця. Підтримувані операції, форми, точність і встановлений драйвер NPU визначають, чи весь граф працюватиме на цьому пристрої. OpenVINO IR (`.xml` плюс `.bin`) є повторно використовуваним вихідним IR; кеш скомпільованої моделі залежить від пристрою та програмного забезпечення.

**Apple Core ML:** `coremltools` перетворює моделі на `.mlpackage`, який Xcode компілює у `.mlmodelc`. Core ML може планувати обчислення на CPU, GPU й Apple Neural Engine, але навмисно приховує точне розбиття. Через це Apple є чудовою платформою розгортання, але заяву на кшталт «увесь граф YOLO виконувався на NPU» варто робити лише за наявності профілювання, яке це підтверджує.

## Компанії, що створюють IP NPU для виробників чипів

Деякі компанії зовсім не продають плату або готовий прискорювач. Вони ліцензують розробки NPU виробникам SoC. Це важливо, бо та сама базова архітектура може з'являтися під кількома брендами чипів, але кінцеві SDK і артефакт усе одно можуть бути адаптовані ліцензіатом.

| Постачальник IP | Сімейство NPU і програмне забезпечення | Чому це важливо |
|---|---|---|
| [Arm](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) | Ethos-U55/U65/U85 і Vela | Використовується в NXP, Alif, Infineon, Himax та інших вбудованих продуктах; процес орієнтований на квантизований TFLite/TOSA |
| [Arm China](https://www.armchina.com/mountain?infoId=161&name=) | AIPU Zhouyi і Compass SDK | У публічному [каталозі моделей](https://github.com/Arm-China/Model_zoo) названо YOLOv1-tiny/v2/v3/v4/v5, YOLOX і YOLOv8-seg; це еталонні моделі, а не доказ для кожного чипа ліцензіата |
| [VeriSilicon](https://www.verisilicon.com/en/IPPortfolio/VivanteVIP9000Pico) | Сімейство Vivante VIP і SDK Acuity | Споріднені сімейства NPU Vivante є у кількох SoC, зокрема в окремих шляхах A311D і i.MX 8M Plus; кожен виробник чипа надає власну інтеграцію |
| [Imagination Technologies](https://www.imaginationtech.com/products/open-access/) | PowerVR Series3NX AX3146/AX3386/AX3596; історичний IMG Series4; Neural Compute SDK/IMG DNN | Ліцензований IP NNA, а не готова до роздрібного продажу плата; Open Access надає конфігурації Series3NX на 1/5/10 TOPS, а офіційна програма NC-SDK називає PP-YOLOE, EfficientNet і HRNet, але не кожну конфігурацію IP |
| [CEVA](https://www.ceva-ip.com/product/ceva-neupro-studio/) | NeuPro-Nano/NeuPro-M і NeuPro Studio | Ліцензований IP NPU з імпортом, квантуванням, стисненням, компіляцією графів, симуляцією та генеруванням C/C++ |
| [MIPS](https://mips.com/processor-solutions/arc-npx-family/) | ARC NPX6 і [MetaWare MX](https://mips.com/processor-solutions/arc-metaware-mx/), раніше Synopsys ARC | Масштабований IP NPU й SDK нейронних мереж, придбаний GlobalFoundries і включений до портфеля MIPS у [червні 2026 року](https://mips.com/press-releases/gfmipsarcclose/); самостійною ціллю розгортання у роздріб не є |
| [Cadence](https://www.cadence.com/en_US/home/tools/silicon-solutions/ai-ip-platform/neuroweave-sdk.html) | Neo NPU, DSP Tensilica й SDK NeuroWeave | Стек компілятора/інтерпретатора, який використовують ліцензіати SoC; підтримка мереж і квантування залежить від ліцензованої конфігурації |
| [Quadric](https://quadric.ai/npu-ip) | IP Chimera GPNPU і SDK | Програмований IP у стилі NPU/DSP, ліцензований іншим компаніям; кінцевою ціллю є кремній ліцензіата |
| [Expedera](https://www.expedera.com/products-overview/) | IP NPU Origin і програмний стек | Ліцензована архітектура edge NPU, а не готова до придбання плата LibreYOLO |

Imagination показує, чому поруч із назвою архітектури мають бути зазначені доступ і життєвий цикл. Її [програма Open Access](https://www.imaginationtech.com/products/open-access/) пропонує три конфігурації Series3NX, перевірені на кремнії, без попередньої плати за ліцензію IP, але лише після оцінювання компанії, із платною підтримкою/обслуговуванням і роялті за серійне виробництво. [Neural Compute SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) надає інструменти компіляції, оптимізації, квантування й середовище виконання; в офіційній [співпраці з Baidu PaddlePaddle](https://www.imaginationtech.com/news/imagination-and-baidu-paddlepaddle-create-open-source-machine-learning-library-for-model-zoo/) названо виявлення PP-YOLOE, класифікацію EfficientNet і сегментацію HRNet. Ці приклади не валідовують кожну конфігурацію Series3NX або Series4; на поточних окремих [сторінках продуктів Series4](https://www.imaginationtech.com/product/img-4nx-mc1/) зазначено, що продукти недоступні, тож Series4 слід вважати історичним поколінням, доки відділ продажу не підтвердить інше.

Цей рівень пояснює зовнішню схожість сімейств. Він не забезпечує переносності артефактів. Два SoC зі спорідненим IP Vivante або Ethos можуть мати різні карти пам'яті, драйвери, версії компіляторів, набори операторів і середовища виконання постачальників.

## Що саме ви розгортаєте: таблиця прив'язки артефактів

Саме на останньому етапі конвеєра компілятора закінчується позірна переносність ONNX. Назви й розширення змінюються, але практичне правило незмінне: зберігайте вихідну модель і калібрувальні дані, оскільки нативний артефакт зазвичай не можна перенести на інше покоління NPU або надійно перезібрати з іншою версією SDK.

| Стек постачальника | Типовий вхід компілятора | Що потрапляє на пристрій | До чого прив'язано |
|---|---|---|---|
| [Amlogic AMLNN](https://github.com/Amlogic-NN/amlnn-toolkit) | ONNX, TFLite, TorchScript або PT2 | `.adla`; для старого A311D використовується `.nb` | Ідентифікатор цілі, покоління ADLA, збірка компілятора AMLNN, NNSDK2/середовище виконання, драйвер ADLA й BSP |
| [Rockchip RKNN](https://github.com/airockchip/rknn-toolkit2) | ONNX, TFLite або експорт фреймворку | `.rknn` | Версія набору інструментів/середовища виконання RKNN і сімейство NPU Rockchip |
| [Hailo](https://hailo.ai/developer-zone/documentation/) | ONNX або TensorFlow через проміжний HAR | `.hef` | Архітектура Hailo та покоління компілятора/середовища виконання |
| [Axelera Voyager](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) | ONNX або підтримуване визначення zoo | `.axm` у alpha Pipeline Builder, пакет пайплайна `.axe`; класичний `.axmodel` разом із маніфестом | Покоління API Voyager і конфігурація цілі Metis |
| [DEEPX DXNN](https://github.com/DEEPX-AI/dx-all-suite) | ONNX і підтримувані шляхи фреймворків | `.dxnn` | Версії DX-COM/DX-RT і ціль DX-M |
| [Qualcomm QAIRT/QNN або SNPE](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | ONNX, TFLite або модель фреймворку | Бібліотека моделі/контекстний бінарний файл QNN або SNPE `.dlc` | Архітектура HTP, SoC, бекенд і версія середовища виконання |
| [MediaTek NeuroPilot](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) | Перетворений/квантизований TFLite | `.dla` для офлайн-середовища Neuron Runtime; `.tflite` для делегованого режиму | Покоління NP/MDLA, образ ОС, компілятор і середовище виконання |
| [TI TIDL](https://github.com/TexasInstruments/edgeai-tidl-tools) | ONNX або TFLite | Каталог імпортованих артефактів і файли моделей застосунку | Інструменти/середовище виконання TIDL і покоління процесора |
| [NXP eIQ](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | Зазвичай TFLite або ONNX | Вихідний файл конкретного бекенду Vela, TIM-VX, Neutron або Ara | Вибраний прискорювач i.MX/Ara та BSP, а не просто «NXP» |
| [ST Edge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) | TFLite, ONNX або підтримувана модель фреймворку | Згенерована мережева бібліотека/код і ресурси прошивки | Ціль MCU/NPU, конфігурація пам'яті й версія інструмента |
| [Infineon PSOC Edge](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) | TFLite/Keras/PyTorch зі цілочисельним квантуванням | Оптимізований Vela TFLite із потоком команд Ethos-U та прошивкою застосунку | Варіант E83/E84, конфігурація Ethos-U, Vela, ModusToolbox і BSP; NNLite є окремою ціллю |
| [Renesas DRP-AI](https://github.com/renesas-rz/rzv_drp-ai_tvm) | ONNX/TFLite через TVM або Translator | Каталог моделі середовища виконання з кількох двійкових файлів даних | Пристрій RZ/V і покоління Translator/середовища виконання |
| [Sony IMX500](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) | Квантизована/запакована мережа з процесу Edge-MDT | Пакет `.rpk` | Прошивка сенсора, бюджет пам'яті й версія пакувальника |
| [Synaptics](https://developer.synaptics.com/) | TFLite/ONNX та підтримувані імпорти | `.synap` у SyNAP; `.vmfb` у Torq | Покоління програмного забезпечення/обладнання SL16xx чи SL261x |
| [Ambarella CVflow](https://www.ambarella.com/developer/) | Вхід від партнерського набору інструментів | Пакет для розгортання партнерського типу; точний контракт артефакту публічно не задокументовано | Уточніть покоління CVflow, SDK/BSP і пайплайн камери через Cooper Developer Zone |
| [D-Robotics BPU](https://github.com/D-Robotics/rdk_model_zoo) | ONNX і граф, підтримуваний набором інструментів | `.bin` для X5; `.hbm` для поточної `rdk_s`; для X3 використовується старіший процес платформи | Покоління BPU, гілка репозиторію та відповідний реліз OpenExplorer/середовища виконання |
| [AXERA Pulsar2](https://github.com/AXERA-TECH/ax-samples) | ONNX | `.axmodel` | Цільовий чип AX, версії Pulsar2 й AXEngine |
| [SOPHGO TPU-MLIR](https://github.com/sophgo/tpu-mlir) | ONNX, TFLite, TorchScript та інші | `.bmodel` для BM, `.cvimodel` для CV18xx | Цільовий чип BM/CV і покоління середовища виконання SOPHON |
| [Huawei Ascend CANN](https://www.hiascend.com/en/software/cann) | ONNX або підтримуваний граф фреймворку | `.om` | Чип Ascend, CANN/ATC, прошивка/драйвер пристрою |
| [Cambricon MagicMind](https://github.com/Cambricon/magicmind_cloud) | Граф фреймворку або модель обміну | Серіалізований рушій/модель MagicMind | Архітектура MLU та версії Neuware/MagicMind |
| [Canaan nncase](https://github.com/kendryte/nncase) | ONNX або TFLite | `.kmodel` | Покоління KPU і відповідний цільовий плагін nncase |
| [Mobilint qb](https://www.mobilint.com/sdk-qb) | ONNX і підтримувані моделі фреймворків | `.mxq` | Цілі REGULUS/ARIES і компілятор/середовище виконання qb |
| [Rebellions RBLN](https://docs.rbln.ai/latest/index.html) | PyTorch 2 або граф, підтримуваний TensorFlow | `.rbln` | Покоління ATOM і компілятор/середовище виконання RBLN |
| [FuriosaAI](https://developer.furiosa.ai/docs/latest/en/) | ONNX/TFLite і підтримуваний шлях фреймворку | Warboy `.enf`; RNGD `.fxb` | Цілком різні покоління прискорювачів і SDK |
| [Sunplus SNNF](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2004353133) | Мережа, підтримувана Acuity | `.nb` | Драйвер NPU SP7350, середовище виконання й BSP |
| [ESWIN ENNP](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) | ONNX у задокументованому поточному процесі EsAAC; для інших фреймворків потрібен експорт/перетворення | `.model` | Точна ціль сімейства EIC7700 і версія ENNP/ESSDK |
| [Nuvoton / Ethos-U](https://github.com/OpenNuvoton/NuEdgeWise) | Квантизований TFLite | Оптимізований Vela TFLite із користувацькими операціями Ethos-U | План пам'яті M55M1, Vela і прошивка |
| [Himax WiseEye2](https://github.com/HimaxWiseEyePlus/YOLOv8_on_WE2) | Повністю квантизований TFLite | Оптимізований Vela `.tflite` у пам'яті моделей разом із прошивкою плати, наприклад `output.img` | Конфігурація HX6538/WE2, Vela, TFLite Micro, драйвер Ethos-U і резервні ядра |
| [Analog Devices AI8X](https://github.com/analogdevicesinc/ai8x-synthesis) | Контрольна точка PyTorch, YAML мережі й приклад входу | Згенеровані цільові вихідні файли/заголовки C, ваги й скомпільована прошивка | Обмеження пам'яті/шарів MAX78000 або MAX78002, інструмент синтезу й реліз MSDK |
| [Realtek AmebaPro2](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | Сервіс перетворення/вхід постачальника | `.nb` | Прошивка RTL8735B і реліз VoE/NeuralNetwork API |
| [Telechips Enlight](https://docs.topst.ai/product/p/ai) | Шлях Darknet, TensorFlow, ONNX або PyTorch | Проміжний `.enlight` і пакет для розгортання | NPU TCC7500 і версії TC-NN/Enlight |
| [T-Head HHB](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | ONNX і підтримуваний граф фреймворку | `hhb.bm`, згенеровані параметри/код і виконуваний файл | Стек CSI-NN2/SHL для TH1520 |
| [SigmaStar MI_IPU](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | У поточній документації SSU9383CM надано середовище виконання MI_IPU; старіші матеріали використовують вхід TensorFlow/Caffe-сімейства | Старіший `.sim` перетворюється на `sgsimg.img`; поточний публічний артефакт не перевірено | Точне покоління IPU SigmaStar і SDK; сумісність застарілого компілятора з SSU9383CM не підтверджена |
| [Розумний зір HiSilicon](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) | ONNX і граф, сумісний з ATC | `.om` | Точний SoC розумного зору і стек NNN/SVP-NNN; немає загальної переносності на Ascend |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | ONNX, TFLite, Keras або TensorFlow | Пакет потоку даних `.dfp` | Конфігурація обладнання MX і реліз середовища виконання/компілятора |
| [Kneron](https://doc.kneron.com/docs/) | ONNX плюс конфігурація набору інструментів | `.nef`; KL730 NEFv2 може містити `.kne` | Задокументована ціль компілятора, покоління чипа, прошивка й середовище виконання PLUS; компіляцію для KL830 у Toolchain 0.33.1 не підтверджено |
| [SiMa.ai Palette](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) | Підтримуваний фреймворк або модель ONNX | Скомпільований `.tar.gz` ModelSDK, задокументований виконуваний ELF або пакет `.mpk` від MPK Tool | Ціль MLSoC/Modalix і реліз Palette |
| [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/) | ONNX або визначення мережі | Серіалізований рушій, зазвичай `.engine` або `.plan` | Архітектура GPU/DLA, TensorRT, CUDA і часто сам пристрій |
| [AMD Vitis AI](https://vitisai.docs.amd.com/en/6.2/) | Квантизований граф ONNX/фреймворку | Застарілий `.xmodel`; знімок/підграфи NPU Gen1; каталог кешу компіляції Gen2 або серійний пакет `.rai` | Точне покоління NPU/конфігурація IP, образ платформи й матриця Vitis/Vivado/PetaLinux |
| [Microchip VectorBlox](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) | Повністю квантизований TFLite INT8 | `.vnnx`, `.hex` і `.ucomp` разом із відповідною прошивкою/дизайном FPGA | Конфігурація CoreVectorBlox, PolarFire SoC Video Kit, SDK 3.1, бітовий образ і розкладка пам'яті |
| [Intel OpenVINO](https://docs.openvino.ai/) | Модель фреймворку або ONNX | IR `.xml` плюс `.bin` або кеш, скомпільований для пристрою | IR можна повторно використовувати; кеш залежить від пристрою, драйвера й OpenVINO |
| [Застарілий Google Edge TPU](https://coral.ai/docs/edgetpu/models-intro/) | Повністю INT8 TFLite | Файл `_edgetpu.tflite`, скомпільований для Edge TPU | Компілятор/середовище виконання Edge TPU й підтримувані квантизовані оператори; не пов'язано з новим IP Coral NPU |
| [IP Google Coral NPU](https://github.com/google-coral/coralnpu) | Приклади C/C++ для поточного публічного проєкту IP | Приклади ELF для RTL/апаратної симуляції та майбутньої інтеграції в SoC; публічного артефакту компілятора YOLO немає | Точна ліцензована/інтегрована реалізація SoC і набір інструментів із відкритим кодом, що розвивається |
| [Lattice sensAI](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | Опис підтримуваної мережі | Бітовий образ FPGA, конфігурація прискорювача й ваги | Сімейство FPGA, режим IP sensAI і реалізація пам'яті |
| [Imagination NC-SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) | Caffe, TensorFlow або ONNX | Назва артефакту для клієнта публічно не визначена | Ліцензована конфігурація NNA, інтеграція SoC, NC-SDK/DDK і середовище виконання ліцензіата |

Ця таблиця є не просто довідником розширень. Версія компілятора, ідентифікатор цілі, драйвер, прошивка й BSP входять до відтворюваної ідентичності моделі. Майбутній експортер LibreYOLO має записувати їх у машинозчитуваний маніфест поруч із кожним артефактом.

## Доступ до інструментів і сигнали життєвого циклу

Доступність обладнання та доступ до компілятора є незалежними чинниками. Плату може бути легко купити, хоча для поточного компілятора потрібна клієнтська угода; публічний SDK може призначатися для кремнію, який важко дістати. Нижче наведено конкретні сигнали від перших сторін, а не універсальний рейтинг довговічності.

| Платформа | Задокументований сигнал | Практичне значення |
|---|---|---|
| Amlogic ADLA | Публічні репозиторії Apache-2.0 і колеса для завантаження; сумісні драйвери плати/BSP усе одно можуть вимагати участі Amlogic або виробника плати | Компілятор легко оцінити, але поширення бінарних файлів і повну матрицю сумісності слід уточнити |
| MediaTek Genio | Публічні IoT AI Hub і посібники Yocto; універсальні інструменти NP8 і документи Android позначено як ресурси для прямих клієнтів/NDA | Свідчення щодо моделей можна перевірити, але автоматизація власних моделей може потребувати партнерства |
| Hailo | Публічні каталог моделей і середовище виконання; Dataflow Compiler поширюється через Developer Zone; публічний `hailo-camera-apps` заархівовано 3 травня 2026 року | Широкий вибір моделей відкритий, але для відтворюваної компіляції потрібні відповідні обліковий запис і версія; архів не доводить завершення життєвого циклу Hailo-15, потрібно уточнити поточний пакет для процесора зору |
| Axelera AI | Публічна документація й [публічний Voyager SDK](https://github.com/axelera-ai-hub/voyager-sdk); для підтримки клієнтів потрібен обліковий запис | Рідкісний SDK прискорювача для самостійного використання; підтримка продукту й придбання залишаються комерційними |
| Qualcomm Dragonwing | У [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program) для IQ-9075 зазначено Sampling із довготривалою підтримкою до 2038 року, а для QCS6490 до липня 2036 року, тоді як сторінка IQ-9075 зазначає Active | Сильний сигнал для промислового планування, але суперечність у джерелах першої сторони потребує перевірки за SKU й не гарантує стабільності ABI одного SDK AI протягом усього періоду |
| Rebellions | У поточній [матриці підтримки](https://docs.rbln.ai/latest/supports/version_matrix.html) CA02 і CA12 позначено EoL, CA22/CA25 мають статус Active, а CA21 відсутній | Придбання й сумісність SDK залежать від конкретної плати, навіть у сімействі ATOM |
| NVIDIA Jetson | В [офіційній таблиці життєвого циклу модулів](https://developer.nvidia.com/embedded/lifecycle) зазначено для модулів Orin період до січня 2032 року, а для AGX Orin Industrial до липня 2033 року; для комплектів розробника зобов'язань щодо життєвого циклу немає | Виробничі системи слід будувати на модулях, а не на доступності комплектів розробника |
| Sony IMX500 на Raspberry Pi | У [паспорті продукту AI Camera](https://datasheets.raspberrypi.com/camera/ai-camera-product-brief.pdf) зазначено виробництво щонайменше до січня 2028 року | Конкретний мінімальний строк для цього модуля камери, а не для кожного продукту IMX500 |
| Amlogic A311Y3 | На [сторінці запуску 2026 року](https://www.amlogic.com/News/index248.html) заявлено мінімальну гарантію життєвого циклу продукту на десять років | Багатообіцяльний сигнал для нової платформи, для продуктової програми все одно потрібні контракт і деталі SKU |
| Google Coral | Застарілі репозиторії Edge TPU і PyCoral заархівовано/переведено лише для читання; окремий репозиторій [Coral NPU IP](https://github.com/google-coral/coralnpu) активний | Ризик супроводу старого програмного забезпечення, який сам по собі не означає офіційного завершення життєвого циклу обладнання; нічого не говорить про життєвий цикл окремого IP із відкритим кодом |

## Чому TOPS не є критерієм вибору

Пікові TOPS можуть бути корисними в межах одного постачальника й архітектури, але між постачальниками такі порівняння зазвичай некоректні, якщо не збігаються всі наведені нижче умови:

- Числова точність, зокрема INT8, INT4, FP16 або змішаний режим
- Чи рахують одне множення з накопиченням як одну або дві операції
- Щільна чи структурно розріджена арифметика
- Роздільна здатність входу, розмір батча й граф моделі
- Точність після квантування
- Затримка лише на NPU чи в усьому пайплайні застосунку
- Передавання даних із пам'яті й резервне виконання на хості
- Тепловий стан і тривала, а не короткочасна робота

Детектор працює як пайплайн: отримання зображення, зміна розміру/letterbox, нормалізація, передавання на пристрій, інференс нейронної мережі, декодування, NMS, масштабування координат і логіка застосунку. Постачальники часто публікують лише середню частину, тобто інференс нейронної мережі. [MLPerf Inference Edge](https://mlcommons.org/benchmarks/inference-edge/) цінний тим, що визначає сценарії, цільові показники якості й правила вимірювання всієї системи, а не ізольовано порівнює маркетингову арифметику.

Для розгортання YOLO мінімальний достовірний запис бенчмарку має містити:

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

Без цих полів два значення FPS зазвичай вимірюють різні речі.

## Як вибрати NPU для комп'ютерного зору

Обирайте в такому порядку, а не за найбільшим числом на сторінці продукту.

1. **Перевірте сумісність графа.** Скомпілюйте саме експортовану модель, а не модель із подібною назвою з каталогу.
2. **Виміряйте правильність.** Перевірте скомпільований артефакт на реальному датасеті завдання після калібрування й квантування.
3. **Виміряйте весь процес.** Включіть перетворення входу, передавання даних, декодування й NMS.
4. **Перевірте доступ до програмного забезпечення.** З'ясуйте, чи компілятор публічний, потребує реєстрації або доступний лише після комерційної угоди.
5. **Зафіксуйте матрицю версій.** Запишіть версії компілятора, середовища виконання, драйвера, прошивки й ідентифікатори цілей.
6. **Перевірте реальну підтримку ОС.** Підтримка Android, Debian, Yocto, Buildroot, RTOS і Windows не взаємозамінна.
7. **Перевірте постачання й життєвий цикл.** Технічно чудовий чип погано підходить для виробництва, якщо модулі, драйвери або довготривала підтримка ненадійні.
8. **Лише після цього порівнюйте вартість, потужність і продуктивність.** Ці вимірювання мають значення, коли та сама модель коректно працює на обох цілях.

### Практичні відправні точки за сценарієм використання

| Сценарій | Платформи, які варто оцінити першими | Причина |
|---|---|---|
| Прискорювач для Raspberry Pi | Hailo-8L/8 і Sony IMX500 | Офіційні продукти Raspberry Pi, образи програмного забезпечення й конкретні процеси для розробників |
| Універсальне доповнення Linux через PCIe, M.2 або USB | DEEPX, MemryX, Hailo й продукти Axelera | Доступні формфактори прискорювачів за умови сумісності хоста/драйвера |
| Недорога плата Linux SBC або камера | Rockchip RK3588/RK3576, Amlogic ADLA за наявності підтримуваної плати/BSP, AXERA, Canaan K230 або Sunplus SP7350 | Інтегровані мультимедійні пайплайни й публічні свідчення на рівні моделей/компіляторів; для NPU A311D2 перевірте VIM4 V13A+ |
| Мобільні пристрої й Android із великим обсягом продажу | Qualcomm QNN/AI Hub, LiteRT, Apple Core ML | Велика встановлена база й підтримувані мобільні середовища виконання |
| Промисловий Linux і робототехніка | TI TDA4/AM6xA, NXP i.MX, Renesas RZ/V, Hailo | Вбудовані продукти з довгим життєвим циклом та екосистеми камер/вводу-виводу |
| Дуже малий endpoint або MCU | STM32N6x7, Infineon PSOC Edge, Himax WiseEye2, Analog Devices MAX7800x, Nuvoton M55M1, Alif/Arm Ethos-U, GAP9 і Syntiant | Жорсткі обмеження потужності й пам'яті, спеціалізовані інструменти та обмеження розміру моделей |
| Конфігурований машинний зір FPGA | Microchip VectorBlox, Lattice sensAI і цілі AMD Vitis AI | Гнучкі апаратні пайплайни, але конфігурація прискорювача, бітовий образ і компілятор моделі утворюють єдину систему з версіями |
| Машинний зір із високою пропускною здатністю через PCIe | Axelera Metis, Hailo, DEEPX, Mobilint, MemryX, Rebellions або SiMa.ai | Спеціалізовані прискорювачі й позиціювання для кількох потоків |
| Продуктова екосистема, зосереджена в Китаї | Rockchip, AXERA, D-Robotics, SOPHGO, Huawei Ascend, HiSilicon і Cambricon | Сильні регіональні плати, набори інструментів і приклади моделей |

Ця таблиця є коротким списком для дослідження, а не рейтингом продуктивності. Закупівлі, регіональна доступність і конкретна модель можуть змінити порядок.

## Що це означає для LibreYOLO

Масштабований дизайн не потребує десятків не пов'язаних експортерів. Потрібен один суворий контракт обміну та невеликі адаптери компілятора й середовища виконання постачальників:

```text
LibreYOLO model
    -> deterministic static ONNX
    -> VendorCompiler.compile(model, target, calibration, precision)
    -> native artifact + manifest
    -> VendorRuntime.load() / infer()
    -> shared task-specific decode and Results objects
```

Для кожного нативного артефакту має бути супровідний маніфест із такими полями:

- Постачальник, цільовий чип і цільова плата
- Версії компілятора, середовища виконання, драйвера й прошивки
- Контрольна точка джерела та контрольні суми ONNX
- Назви й форми входів, розкладка, колірний простір, нормалізація та тип даних
- Точність квантування, масштаби за наявності та хеш калібрувальних даних
- Назви й форми вихідних тензорів, їхнє семантичне значення
- Завдання виявлення, назви класів і розташування декодування/NMS, на чипі чи на хості
- Результати числової узгодженості й точності завдання

LibreYOLO вже має переносний [експорт ONNX](/docs/export/onnx), [інструменти квантування](/docs/export/quantization), прямий, але навмисно вузький [експорт RKNN](/docs/export/rknn) і чесний процес зовнішньої компіляції [Hailo](/docs/export/hailo). За критеріями цього посібника Amlogic ADLA є сильним наступним кандидатом на інтеграцію: набір інструментів публічний, моделі значною мірою збігаються із завданнями LibreYOLO, а старий шлях A311D можна чітко відокремити.

Наступний пріоритет після Amlogic слід визначати за обладнанням користувачів і можливістю перевіряти точність, а не лише за доступністю компілятора. DEEPX, Sony IMX500, D-Robotics, AXERA, SOPHGO і Qualcomm технічно привабливі; TI, NXP, ST і Renesas стають особливо цінними, коли промислові партнери можуть надати плати й довготривалий доступ до CI.

## Список спостереження: реальний кремній без публічного контракту інтеграції

Повне виключення компанії може зробити огляд ринку оманливим. Назвати її підтримуваною може бути ще гірше. Ці постачальники продають, відвантажують зразки або анонсували відповідні чипи, але публічні свідчення поки не підтверджують відтворюваний актуальний шлях компілятора, артефакту й названої моделі, придатний для бекенду LibreYOLO.

| Компанія | Що реально існує | Чому компанія залишається у списку спостереження |
|---|---|---|
| [Samsung](https://semiconductor.samsung.com/processor/automotive-processor/exynos-auto-v920/) | Exynos Auto V920 має двоядерний NPU із заявленою продуктивністю до 23.1 TOPS | Samsung повідомляє, що [Neural SDK більше не надається стороннім розробникам](https://developer.samsung.com/neural/overview.html); Samsung ONE/Circle не доводить доступу до NPU Exynos |
| [Novatek](https://www.novatek.com.tw/en-global/Milestone/aboutus_milestones) | У поточних корпоративних етапах названо SoC Edge AI і Edge Vision/Imaging AI; у дописі за 2020 рік згадувалися MobileNet, SSD і YOLOv3 | Немає поточної публічної матриці номерів деталей/компілятора/артефактів/операторів/моделей; історичний YOLOv3 не є доказом поточного SDK |
| [Nextchip](https://www.nextchip.com/en/adas/adas.php?idx=5) | Автомобільні процесори APACHE5/NVS2900 і актуальні [APACHE6/NVS3000](https://www.nextchip.com/en/adas/adas.php?idx=7) з NPU aiMotive aiWare; на сторінці APACHE6 одночасно заявлено 12 і 8 TOPS | Існують aiWare Studio і середовище виконання, але артефакт, посібник із квантування, список операторів чи матрицю YOLO публічно не знайдено; суперечливі цифри постачальника не слід мовчки узгоджувати |
| [Black Sesame Technologies](https://bst.ai/en.html) | Автомобільні/edge-чипи AI Huashan A1000/A2000 і Wudang C-series | Публічна інформація про продукти є, але немає компілятора для самостійного використання, контракту артефактів або відтворюваного каталогу моделей |
| [Ingenic](https://en.ingenic.com.cn/products-detail/id-19.html) | SoC камер T41/T40 із заявами про NPU малої розрядності; Magik AI описує PTQ/QAT і компіляцію графів | Не знайдено завантажуваного актуального компілятора, контракту артефактів чи точного переліку моделей YOLO від постачальника |
| [Fullhan](https://fullhan.com/en/index.php?a=type&c=article&tid=9) | Кілька SoC IPC рекламують NPU на 0.5-2 TOPS, а [сімейство NVR MC6880](https://fullhan.com/en/index.php?a=type&c=article&tid=48) заявляє 4 TOPS | Немає публічної документації щодо компілятора NN, середовища виконання, фреймворків або моделей, достатньої для незалежного відтворення |
| [Goke Microelectronics](http://www.gokemicro.com/News/info.aspx?itemid=437) | На сторінці від першої сторони для смарткамер GK7606V1/GK7206V1/GK7203V1 заявлено продуктивність вбудованого NPU, але на момент публікації вона доступна лише через застарілий HTTP | Немає публічного конвертера, контракту середовища виконання чи матриці названих моделей; канал орієнтовано на OEM |
| [Chengheng Micro](https://en.chenghengmicro.com/) | У CH37 заявлено 64 INT8 TOPS і режими FP16/FP32/FP64; у [хронології компанії за 2026 рік](https://chenghengmicro.com/about.html) сказано про початок дрібносерійного виробництва | Публічного SDK, контракту компілятора чи валідації названих моделей не знайдено; це рання комерційна платформа без самостійного доступу, а не відтворюваний шлях розгортання |
| [Bouffalo Lab](https://github.com/bouffalolab/bouffalo_sdk) | BL808 містить NPU BLAI-100, а поточні офіційні репозиторії SDK існують | У підтримуваному SDK немає поточного офіційного шляху конвертації/прикладу BLAI; доступні процеси є застарілими або належать спільноті |

Цей список спостереження навмисно ґрунтується на доказах. Постачальник може перейти до таблиць платформ для розгортання, опублікувавши актуальний набір інструментів або шлях оцінювання, контракт артефактів для конкретної цілі та щонайменше одну названу модель із наскрізним рецептом.

## Що навмисно не стверджувалося

У цій статті не стверджується, що кожна названа модель працює з кожного вихідного репозиторію. Моделі з каталогу постачальника часто змінено, розбито в інших вихідних вузлах або поєднано з користувацькою післяобробкою.

Також наведені твердження не перетворено на заяви про підтримку:

- Компілятор імпортує ONNX.
- Чип рекламує драйвер Android NNAPI.
- Дистриб'ютор каже, що плата «сумісна з YOLO».
- У репозиторії спільноти запускається один варіант однієї моделі.
- Модель компілюється без помилки.
- Постачальник наводить FPS лише для NPU без результату точності.

Прискорювачі телефонів Google Tensor і численні користувацькі ASIC, створені лише для OEM, також існують, але Google не пропонує Tensor як ціль компіляції для самостійного використання, порівнянну з переліченими вище платформами. Самого факту існування компанії, блок-схеми NPU чи прискорення Android недостатньо, щоб вигадувати заяву про інтеграцію LibreYOLO.

Хмарні прискорювачі, як-от Google TPU, AWS Inferentia/Trainium, Microsoft Maia й AI-плати лише для центрів обробки даних, також поза межами теми. Посібник присвячено обладнанню, яке розробник комп'ютерного зору реально може розгорнути на периферії.

Ліцензування моделей є окремим рівнем. Публікація постачальником чипа рецепта перетворення YOLO не надає прав на використання чи поширення вихідних ваг, коду навчання або скомпільованого похідного артефакту. Для цільового продукту потрібно окремо перевіряти підтримку обладнання, ліцензію SDK і ліцензію моделі.

## Методологія дослідження та виправлення

Для кожної компанії шукали чотири типи першоджерел:

1. Поточну сторінку продукту або даташит, де визначено кремній.
2. Документацію компілятора й середовища виконання з поясненням реального процесу розгортання.
3. Каталог моделей, таблицю сумісності або наскрізний посібник із названими моделями комп'ютерного зору.
4. Сигнал щодо життєвого циклу або доступу, який показує, чи можуть розробники отримати інструменти.

Власні організації на GitHub вважаються джерелами постачальника. Документацію виробника плати позначено як свідчення екосистеми, якщо компанія-виробник чипа не публікує аналогічні матеріали. Твердження сторонніх джерел про продуктивність виключено з порівняльних таблиць.

Потім завершену чернетку повторно перевірили окремо за групами постачальників і міжтабличними даними. Під час перевірок повторно звірили обсяг цілей, назви артефактів, точність, докази щодо моделі й постпроцесора, статуси «анонсовано»/«у продажу», позначення життєвого циклу, знаменники бенчмарків і кожну кількість сутностей. Якщо два джерела від першої сторони суперечать одне одному, у посібнику це зазначено, а не обрано мовчки зручніше значення.

Цей ринок швидко змінюється. Якщо ви працюєте в одній із цих компаній і бачите помилку в чипі, SDK, списку моделей чи статусі доступу, надішліть LibreYOLO точну URL-адресу публічної документації та версію. Текст слід оновлювати на підставі виправлень із першоджерел; непідтверджені маркетингові заяви не слід додавати.

## Поширені запитання

### Скільки існує компаній Edge AI NPU?

Універсальної кількості компаній немає, оскільки постачальники продуктів, ліцензіари IP, розумні сенсори, GPU та приватні автомобільні ASIC перетинаються. Цей невичерпний посібник охоплює 69 компаній і платформних екосистем, пов'язаних із edge AI чипами, платформами прискорення, ліцензованим IP NPU або перспективними чипами зі списку спостереження станом на серпень 2026 року.

### Що таке NPU?

Нейронний процесор (NPU) є спеціалізованим обладнанням для операцій нейронних мереж, як-от згортки та множення матриць. Постачальники використовують назви NPU, AIPU, BPU, KPU, DLA, HTP, TPU та MLA, а скомпільовані моделі зазвичай не можна переносити між компаніями.

### Які компанії NPU офіційно підтримують моделі YOLO?

Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Sony IMX500 через офіційний репозиторій Raspberry Pi AI Camera, STMicroelectronics, Renesas, Lattice, Himax, Microchip та кілька інших мають власні записи в каталогах моделей, посібники або валідовані результати щонайменше для одного покоління YOLO. Конкретне покоління, завдання, цільовий чип і версія SDK мають значення.

### Чи може будь-яка модель ONNX працювати на будь-якому NPU?

Ні. ONNX є форматом обміну, а не гарантією апаратної сумісності. Компілятор NPU має підтримувати кожен оператор, форму тензора й тип даних у графі. Часто трапляються статичні форми, розбиття графа, користувацькі операції та резервне виконання на CPU.

### Який NPU найкраще підходить для YOLO?

Універсального переможця немає. Hailo, Rockchip, Axelera AI, DEEPX і Amlogic мають добре задокументовану підтримку YOLO, а Qualcomm вирізняється надзвичайно широким охопленням пристроїв. Обирайте з огляду на правильність після квантування, затримку всього пайплайна, тривалу потужність, ціну, доступність, доступ до SDK і підтримку операційної системи.

### Чому показники NPU у TOPS не можна порівнювати безпосередньо?

Постачальники можуть рахувати різну точність, розріджені операції, операції множення з накопиченням і припущення щодо пікового завантаження. TOPS не враховує трафік пам'яті, непідтримувані оператори, попередню та післяобробку й накладні витрати хоста. Порівнюйте ту саму модель, точність, роздільну здатність, правильність і весь пайплайн.

### Що таке калібрування NPU?

Під час калібрування репрезентативні, зазвичай неанотовані зображення для розгортання пропускають через модель, щоб компілятор міг оцінити діапазони активацій для INT8 або іншого формату з низькою точністю. На випадкових або нерепрезентативних зображеннях артефакт може скомпілюватися, але точність завдання погіршиться.

### Чи підтримує LibreYOLO edge NPU?

LibreYOLO експортує ONNX і кілька форматів середовищ виконання, має прямий шлях компіляції RKNN для окремих моделей Rockchip і документує зовнішній процес компіляції Hailo. Для кожного іншого артефакту у форматі постачальника все одно потрібен його SDK; до компіляції, валідації та запуску на обладнанні такий варіант слід вважати кандидатом на інтеграцію.
