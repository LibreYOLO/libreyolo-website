---
title: LocateAnything
families:
  - locateanything
seo_title: 'LocateAnything: виявлення з відкритим словником і вказування'
description: >-
  Використовуйте LocateAnything у LibreYOLO для виявлення з відкритим словником
  і вказування. Виконуйте передбачення з будь-якою текстовою міткою; навчання,
  валідація та експорт не підтримуються.
lead: >-
  LocateAnything є візуально-мовною моделлю прив'язування від NVIDIA, яка
  паралельно декодує обмежувальні рамки й точки, а не по одному токену
  координат. LibreYOLO надає її як детектор і вказівник із відкритим словником:
  будь-який список текстових міток стає набором класів без фіксованої голови та
  потреби в донавчанні.
keywords:
  - LocateAnything
  - NVIDIA
  - візуально-мовна модель
  - виявлення об'єктів із відкритим словником
  - виявлення точок
  - VLM
  - прив'язування
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE


        model = LibreLocateAnything(size="3b")


        # Відкритий словник: працюють будь-які слова, а не фіксована голова
        класів.

        # Зберігається для всіх наступних викликів predict()/track(), доки його
        не змінено.

        model.set_classes(["person", "bicycle", "dog"])

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Запит точкою
      language: python
      code: >
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE


        # task="point" повертає одну точку на кожен зіставлений об'єкт замість
        рамки.

        # Перемикайте завдання вже завантаженої моделі через
        model.set_task("point").

        model = LibreLocateAnything(size="3b", task="point")

        model.set_classes(["the person closest to the camera"])

        result = model(SAMPLE_IMAGE, save=True)


        for pt in result.points:
            print(pt.cls, pt.conf, pt.xy)
    - label: Необроблений чат
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        model = LibreLocateAnything(size="3b")

        # Прямий доступ під зручним інтерфейсом виявлення: довільні запитання,
        # підрахунок або будь-який запит, якого не охоплює обгортка рамок.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 378ea758e507a096
---

## Встановлення

LocateAnything потребує додаткового набору залежностей `vlm`, який установлює
`transformers` разом із пакетами `decord`, `lmdb` і `peft`, імпортованими
її віддаленим кодом Hugging Face під час завантаження.

```bash
pip install "libreyolo[vlm]"
```

## Передбачення

`LibreLocateAnything` є класом Python, а не контрольною точкою `.pt`: він
не завантажується через фабрику `LibreYOLO()`, і CLI `libreyolo` його не
розпізнає. Фабрика `LibreVLM(...)` (`from libreyolo import LibreVLM`)
також відкриває доступ до цього сімейства за псевдонімом, наприклад
`LibreVLM("locate-anything")`; використаний нижче клас є тим, що вона
створює. Під час завантаження модель отримує та виконує власний віддалений код
NVIDIA з Hugging Face, тому LibreYOLO фіксує завантаження на одному коміті
замість змінної гілки `main` і перед першим завантаженням одноразово показує
повідомлення про ліцензію.

<code-tabs name="predict" />

`result.boxes` (завдання `detect`) і `result.points` (завдання `point`)
містять розібраний вихід, як і для будь-якого іншого сімейства. Оцінка
впевненості є службовою: LocateAnything не виводить оцінок для окремих рамок,
тому кожне виявлення отримує однакову сталу впевненість, а `conf=` лише
відкидає рядки нижче цієї сталої, не ранжуючи їх. Якщо пропустити
`set_classes()`, словник типово містить назви COCO-80. Типи джерел, потокове
передбачення та обробку результатів описано в розділі
[передбачення](/docs/predict).

## Варіанти

Опубліковано один розмір 3b. Два завдання використовують ті самі ваги:
`detect` (типове) повертає рамки, а `task="point"` натомість повертає одну
точку на кожен зіставлений об'єкт у `result.points`. Перемикайте їх у вже
завантаженій моделі за допомогою `model.set_task("point")`. Засоби бенчмарків
LibreYOLO не вимірювали це сімейство, тому опублікованих показників
правильності для порівняння немає.

LibreYOLO надає це сімейство лише для передбачення. `train()`, `val()` і
`export()` спричиняють `NotImplementedError`: донавчіть модель засобами
початкового проєкту й завантажте результат; валідація датасету пропускається,
оскільки службова впевненість зробила б mAP COCO оманливою; експорт не
охоплюється для генеративної моделі без словника стану, який можна трасувати.

## Ліцензування

<provenance-box>

NVIDIA License дозволяє використання, відтворення та модифікацію, але для всіх,
крім NVIDIA та її афілійованих осіб, обмежує модель і будь-які похідні
некомерційним використанням, дослідженнями чи оцінюванням: порога доходу або
платного винятку немає. LocateAnything-3B також поєднує два інші ліцензовані
компоненти: мовний бекбон Qwen2.5-3B-Instruct за Qwen Research License і
візуальний енкодер MoonViT-SO-400M за MIT. LibreYOLO нічого з цього не
розміщує, не дзеркалює й не розповсюджує: під час першого запуску
`LibreLocateAnything` завантажує ваги та потрібний віддалений код
безпосередньо з `nvidia/LocateAnything-3B` на Hugging Face у зафіксованому
коміті.

</provenance-box>

## Цитування

<citation-block />
