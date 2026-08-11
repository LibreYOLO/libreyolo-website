---
title: Розпізнавання облич
seo_title: "Розпізнавання облич у LibreYOLO"
description: "Виявляйте обличчя, створюйте ембединги та ідентифікуйте людей у LibreYOLO. Додайте дані до галереї, порівняйте два зображення й зіставляйте за косинусною подібністю з Python або CLI."
lead: "Розпізнавання облич є задачею embed, застосованою до облич. Детектор знаходить і вирівнює кожне обличчя, голова розпізнавання повертає L2-нормалізований вектор для кожного обличчя, а ідентичність визначається косинусною подібністю до доданих еталонів, а не фіксованим списком класів."
keywords: [розпізнавання облич python, ембединг обличчя, верифікація облич, галерея облич, arcface onnx, libreyolo embed, косинусна подібність облич]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Назви librefacerec-* маршрутизуються до сімейства ембедингів облич
        # незалежно від суфікса файла й під час першого використання завантажуються
        # з організації LibreYOLO у Hugging Face разом із типовим детектором облич.
        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)             # (N, 4) рамки облич
        print(result.embeddings.data.shape)  # (N, D), один рядок на обличчя
        print(result.embeddings.dim)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=photo.jpg
    - label: Порівняти два зображення
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        # Виконує виявлення й створення ембедингу для обох зображень і порівнює
        # обличчя з найвищою впевненістю. Косинусна подібність лежить у [-1, 1].
        outcome = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)
        print(outcome["similarity"], outcome["same_person"])
    - label: Додати дані до галереї та ідентифікувати
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("faces.npz")

        result = model("group_photo.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # нижче порога name має значення None
    - label: Додати та ідентифікувати з CLI
      language: bash
      code: |
        libreyolo enroll model=librefacerec-l.onnx source=people/ gallery=faces.npz
        libreyolo predict model=librefacerec-l.onnx source=group_photo.jpg gallery=faces.npz
    - label: Використати власні рамки облич
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")

        # face_boxes повністю пропускає виявлення; face_detector приймає
        # callable, модель виявлення LibreYOLO або екземпляр FaceDetector.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])
        print(result.embeddings.data.shape)
---

## Визначення

Розпізнавання облич повертає вектор для кожного обличчя, а не мітку.
Передбачення виконується у два етапи: детектор облич знаходить кожне обличчя та
його п'ять орієнтирів, кадрована область деформується до канонічного
вирівнювання 112x112, а голова розпізнавання створює L2-нормалізований ембединг.

`result.embeddings` є об'єктом даних `Embeddings` у формі `(N, D)`, вирівняним
за рядками з `result.boxes`, тому рядок `i` описує обличчя в рамці `i`.
Оскільки рядки є одиничними векторами, косинусна подібність дорівнює скалярному
добутку, а `embeddings.similarity()` одним викликом обчислює її для іншого
об'єкта `Embeddings` або цілої матриці.

Присвоєння імені обличчю є окремим етапом. `Gallery` містить іменовані еталонні
вектори; передавання `gallery=` до `predict()` додає `result.identities`,
вирівняний за рядками з ембедингами, що містить ім'я та його найкращу косинусну
оцінку для кожного обличчя. Обличчя нижче порога збігу зберігає `None` як ім'я,
а найближче ім'я нижче порога ніколи не підставляється.

Канонічним ключем задачі бібліотеки є `embed`. `face-recognition`,
`facial-recognition`, `reid` і `face` нормалізуються до нього, тому
`task="face-recognition"` і `task="embed"` вибирають те саме. Обличчя є
регіональною формою ширшої задачі; розділ
[ембедингів](/docs/tasks/embeddings) описує форми для всього зображення й
тексту, спільний API `Embeddings`, `Identities` і `Gallery` та моделі, які
створюють вектори без виявлення об'єктів.

## Моделі

[LibreFaceRec](/docs/models/librefacerec) є сімейством для цієї задачі. За одним
викликом працюють два артефакти ONNX: `librefacerec-l.onnx`, голова
розпізнавання iResNet100, що створює 512-вимірні ембединги, і
`librefacerec-det.onnx`, типовий детектор облич із п'ятьма орієнтирами з
каталогу OpenCV. Обидва завантажуються з організації LibreYOLO у Hugging Face
під час першого використання. Будь-який інший файл ONNX за домовленістю
ArcFace (вирівняний вхід 112x112, вихід `(N, D)`) може замінити голову
розпізнавання, якщо передати його шлях замість назви `librefacerec-*`.

Ключ задачі `embed` охоплює більше, ніж обличчя.
[CLIP](/docs/models/clip), [SigLIP2](/docs/models/siglip2) і
[DINOv2](/docs/models/dinov2) також підтримують `task="embed"` і повертають
один вектор для всього зображення, що відповідає пошуку зображень, а не
ідентичності обличчя. Вони використовують спільний API `Gallery` і
`Embeddings`, тому описаний нижче процес додавання й зіставлення переноситься,
але ці сімейства не виявляють і не вирівнюють обличчя.

Голова розпізнавання працює через `onnxruntime`, якого немає в базовому
встановленні:

```bash
pip install "libreyolo[onnx]"
```

## Передбачення

<code-tabs name="predict" />

Без додаткових налаштувань `predict()` завантажує й підключає типовий детектор.
`face_detector` замінює його на callable, модель виявлення LibreYOLO або
екземпляр `FaceDetector` і може бути заданий у конструкторі або для окремого
виклику. `face_boxes` обходить виявлення за допомогою ваших наявних рамок.
У CLI параметр `face_detector=` приймає шлях до файла `.onnx` детектора облич
або назву детектора LibreYOLO.

`model.verify(image_a, image_b)` є скороченням для двох зображень: він створює
ембединг обличчя з найвищою впевненістю в кожному з них і повертає
`{"similarity", "same_person", "threshold"}`. `model.embed(sources)` повертає
кожен рядок обличчя з одного або кількох зображень, складений в один тензор
форми `(N_total, D)`. Джерела, потокове оброблення й роботу з результатами
описано в розділі [передбачення](/docs/predict).

## Формат датасету

Додавання даних читає окремий каталог для кожної ідентичності. Назва каталогу
стає ідентичністю, а кожне зображення в ньому додає еталони для цього імені:

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

`libreyolo enroll` обходить це дерево й записує галерею `.npz`. Наявний файл
галереї розширюється на місці, а не замінюється, тому ідентичності можна
додавати поступово. Галереї прив'язано до ваг, що їх створили, за розмірністю
ембедингу та відбитком файла; зіставлення з іншою моделлю спричиняє помилку
замість порівняння несумісних просторів векторів.

Типово кожне початкове зображення додає один еталонний рядок, обличчя з
найвищою впевненістю, тому портрет зі сторонніми людьми додає лише основну
людину. Передайте `select="all"` до `Gallery.enroll`, щоб зберегти кожен
повернений рядок.

## Навчання

Жодне сімейство цієї задачі не навчається всередині LibreYOLO.
`LibreFaceEmbedder.train()` спричиняє помилку: навчіть голову розпізнавання в
upstream-проєкті, експортуйте її до ONNX за домовленістю ArcFace і завантажте
файл за шляхом.

## Валідація

Для цієї задачі немає валідатора датасету, а `val()` спричиняє помилку замість
імітації валідації. Правильність верифікації вимірюється на маркованих парах
зображень за допомогою `model.verify()` із перебором `threshold` для вибору
потрібної робочої точки. Правильність ідентифікації вимірюється шляхом додавання
даних до галереї та читання `result.identities.name` і
`result.identities.score` на відкладених зображеннях, де ім'я `None` вважається
відхиленням.

## Експорт

Голова розпізнавання вже є графом ONNX, тому перетворювати нічого:
`LibreFaceEmbedder.export()` спричиняє помилку. Розгортайте файл `.onnx`
безпосередньо або передайте його LibreYOLO, щоб сімейство виконало виявлення,
вирівнювання й нормалізацію.
