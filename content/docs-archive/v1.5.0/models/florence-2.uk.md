---
title: Florence-2
families:
  - florence2
seo_title: 'Florence-2 у LibreYOLO: виявлення з відкритим словником'
description: >-
  Florence-2 у LibreYOLO: встановлення, налаштування відкритого словника та
  передбачення рамок за допомогою візуальної моделі Microsoft під ліцензією MIT.
lead: >-
  Florence-2 є фундаментальною візуальною моделлю Microsoft, якій передають
  токен завдання замість використання фіксованої голови детектора. LibreYOLO
  надає її як детектор об'єктів із відкритим словником: передайте список класів
  під час передбачення.
keywords:
  - Florence-2
  - візуально-мовна модель
  - виявлення об'єктів із відкритим словником
  - прив'язування тексту до зображення
  - Microsoft
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Відео
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])

        # Будь-яке підтримуване бібліотекою джерело: файл, папка, URL,
        # індекс вебкамери, потік RTSP або список .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
source_hash: ad26d9056465d662
---

## Встановлення

Florence-2 належить до рівня VLM-детекторів LibreYOLO, окремої від сімейств на
основі контрольних точок частини продукту з власною фабрикою. Для неї потрібен
додатковий набір залежностей `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально. LibreYOLO завантажує повторно вивантажену спільнотою florence
контрольну точку, а не початковий репозиторій `microsoft/Florence-2-*`.
Причину пояснено в розділі «Ліцензування».

<code-tabs name="predict" />

Це сімейство завантажується через фабрику `LibreVLM()`, а не `LibreYOLO()`:
сімейства VLM не оголошують завантажувача контрольних точок, тому описана на
інших сторінках моделей маршрутизація за суфіксом файлу тут не застосовується.
`set_classes()` задає словник, об'єкти з якого Florence-2 має знайти на
зображенні. Налаштування зберігається для всіх наступних викликів
`predict()`/`track()`, доки ви знову його не зміните. Повернений об'єкт
`Results` містить `boxes` тієї самої форми, що й для будь-якого іншого
сімейства. Проте кожне виявлення має однакову службову оцінку впевненості,
тому фільтрація за `conf` або пропускає всі результати, або відкидає їх усі,
а не ранжує. Параметр `iou` не діє: обгортка Florence-2 створює список
виявлень безпосередньо з розібраного виходу токена завдання, без етапу усунення
дублікатів. `chat()` спричиняє `NotImplementedError`, оскільки Florence-2
використовує токен завдання `<OPEN_VOCABULARY_DETECTION>`, а не шаблон чату.
CLI LibreYOLO не охоплює цей рівень: форми `libreyolo predict model=...` для
нього немає. Типи джерел, потокове передбачення та обробку результатів описано
в розділі [передбачення](/docs/predict).

## Варіанти

Доступні два розміри: Florence-2-base і Florence-2-large, обидва з роздільною
здатністю 768 px. Вони завантажуються як `LibreVLM("florence-2-base")` або
`LibreVLM("florence-2-large")`. LibreYOLO не публікувала бенчмарк порівняння
їхньої правильності.

LibreYOLO не навчає, не валідує та не експортує Florence-2: `train()`,
`val()` і `export()` спричиняють `NotImplementedError` для кожного
сімейства цього рівня (див. рівень підтримки вище). Якщо потрібен вбудований
власний словник, донавчіть Florence-2 засобами початкового проєкту й завантажте
отримані ваги. Перевіряйте вихід `predict()` візуально замість валідації у
стилі COCO, оскільки кожне виявлення має однакову службову оцінку впевненості.

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />
