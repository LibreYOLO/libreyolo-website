---
title: Ембединги
seo_title: "Ембединги зображень та областей у LibreYOLO"
description: "Задача embed повертає L2-нормалізовані вектори float32 для всього зображення, кожної виявленої області або тексту. Додайте дані до галереї, зіставляйте за косинусною подібністю та шукайте з Python або CLI."
lead: "Одна задача охоплює кожен вектор, який створює LibreYOLO. embed повертає рядки float32 одиничної довжини, скалярний добуток яких є оцінкою подібності, незалежно від того, чи описує рядок усе зображення, одне виявлене обличчя або рядок тексту; та сама Gallery зіставляє їх усі."
keywords: [ембединги зображень python, l2 нормалізований ембединг, пошук за косинусною подібністю, libreyolo embed, пошук схожих зображень, додати до gallery, clip embeddings, dinov2 embeddings, reid embeddings]
last_verified: "1.5.0"
verification: "Ключ задачі та псевдоніми звірено з libreyolo/tasks.py. Об'єкти даних результатів звірено з класами Embeddings та Identities у libreyolo/utils/results.py. API Gallery звірено з libreyolo/utils/gallery.py. embed і _postprocess_embeddings звірено з libreyolo/models/base/model.py. Підтримувані сімейства знайдено пошуком embed у SUPPORTED_TASKS у libreyolo/models/**/model.py. Інтерфейс CLI звірено з libreyolo/cli/__init__.py, libreyolo/cli/commands/special.py і libreyolo/cli/commands/predict.py. Проєктні рішення взято з docs/adr/0015-embed-generalization.md."
meta:
  - label: Ключ задачі
    value: embed
    mono: true
  - label: Псевдоніми
    value: face-recognition, reid, face
    mono: true
  - label: Об'єкти даних результатів
    value: Embeddings, Identities
    mono: true
  - label: Тип даних рядка
    value: float32, одинична довжина
snippets:
  predict:
    - label: Усе зображення
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Типовою задачею CLIP є classify, тому запитайте вектор явно.
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)  # (1, 512), один рядок на зображення
        print(result.boxes)                  # None: нічого не локалізовано
    - label: Для кожної області
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        # Рядок i описує область у рамці i.
        print(result.boxes.xyxy.shape)       # (N, 4)
        print(result.embeddings.data.shape)  # (N, 512)
    - label: Багато зображень одночасно
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # Усі рядки всіх результатів, об'єднані в один тензор.
        vectors = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(vectors.shape)  # (3, 384)
    - label: Текст
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        # Текст передається методу, а не як джерело передбачення. Рядок,
        # переданий до model(...), усе ще є шляхом або URL-адресою.
        text = model.embed_text(["a photo of a cat", "a photo of a dog"])
        print(text.shape)  # (2, 512)
  similarity:
    - label: Порівняти два набори рядків
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        query = model.embed("query.jpg")          # (1, 512)
        pool = model.embed(["a.jpg", "b.jpg"])    # (2, 512)

        # Рядки мають одиничну довжину, тому косинусна подібність є скалярним добутком.
        scores = model("query.jpg").embeddings.similarity(pool)
        print(scores.shape)  # (1, 2)
    - label: Зображення порівняно з текстом
      language: python
      code: |
        import torch

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        image = model.embed("photo.jpg")                       # (1, 512)
        text = model.embed_text(["a cat", "a dog", "a car"])   # (3, 512)

        print(torch.matmul(image, text.T))
  gallery:
    - label: Додати та ідентифікувати
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("refs.npz")

        result = model("group.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # нижче порога name має значення None
    - label: Пошук top-k
      language: python
      code: |
        from libreyolo import Gallery
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        gallery = Gallery.load("refs.npz", model=model)

        result = model("query.jpg")
        matches = gallery.match(result.embeddings, top_k=5, threshold=0.4)
        print(matches[0])   # [(name, score), ...] для першого рядка
    - label: Додати наявний вектор
      language: python
      code: |
        from libreyolo import Gallery

        gallery = Gallery()
        gallery.enroll_embedding("ada", vector)  # нормалізується під час додавання
        print(gallery.identities, gallery.dim, len(gallery))
  cli:
    - label: Додати дерево каталогів
      language: bash
      code: |
        # source/<identity>/*.jpg. Наявна галерея розширюється на місці.
        libreyolo enroll model=librefacerec-l.onnx source=people/ gallery=refs.npz
    - label: Ідентифікувати під час передбачення
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=group.jpg \
          gallery=refs.npz gallery_threshold=0.45
    - label: Порівняти два зображення
      language: bash
      code: |
        libreyolo compare model=librefacerec-l.onnx \
          source=a.jpg source2=b.jpg threshold=0.4

        # verify є тією самою командою під другою назвою.
        libreyolo verify model=librefacerec-l.onnx source=a.jpg source2=b.jpg --json
---

## Визначення

`embed` перетворює зображення, область зображення або рядок на рядок float32
фіксованої ширини з одиничною довжиною. Оскільки кожен рядок є одиничним
вектором, порівняння двох із них зводиться до скалярного добутку, а порівняння
двох наборів до одного матричного множення. Решта задачі не залежить від
моделі: пошук, виявлення дублікатів, повторна ідентифікація й розпізнавання
облич використовують ті самі обчислення для різних рядків.

Вектор є вихідними даними. Списку класів немає, тому ім'я приєднується пізніше
шляхом порівняння з наданими вами еталонами, а не через передбачення, для якого
навчено мережу.

### Три форми

| Форма | `Results.embeddings` | `Results.boxes` | Джерело |
|---|---|---|---|
| Усе зображення | `(1, D)` | `None` | Передавання зображення сімейству для всього зображення |
| Область | `(N, D)` | `(N, 4)`, вирівняно за рядками | Сімейства, що спочатку локалізують, наприклад розпізнавання облич |
| Текст | узагалі не `Results` | | `model.embed_text(texts)`, що повертає `(M, D)` |

Результат для всього зображення залишається двовимірним навіть для одного
зображення. `(D,)` не є дозволеною формою повернення, тому споживачеві не
потрібно окремо обробляти випадок одного рядка. Текст повертає звичайний тензор,
а не `Results`, оскільки рядок не є джерелом зображення: його передавання до
`model(...)` досі означає шлях або URL-адресу, а бібліотека ніколи не вгадує,
що рядок є прозою.

Канонічним ключем задачі є `embed`. `embedding`, `embeddings`,
`face-recognition`, `facial-recognition`, `recognition`, `face`, `faceid` і
`reid` нормалізуються до нього, тому `task="reid"` і `task="embed"` вибирають
те саме.

## Моделі

Задачу виконують чотири сімейства, які чітко поділяються за тим, чи виконують
вони спочатку локалізацію.

| Сімейство | Форма | Розмірність | Також підтримує |
|---|---|---|---|
| [LibreFaceRec](/docs/models/librefacerec) | Область, один рядок на виявлене обличчя | 512 | Нічого; `embed` є єдиною задачею |
| [CLIP](/docs/models/clip) | Усе зображення, зі спареною текстовою вежею | 512 для `b32` і `b16`, 768 для `l14` | `classify`, що залишається типовою задачею |
| [SigLIP 2](/docs/models/siglip2) | Усе зображення, зі спареною текстовою вежею | 768 для `b16`, 1152 для `so400m` | `classify`, що залишається типовою задачею |
| [DINOv2](/docs/models/dinov2) | Усе зображення, лише зображення | 384 | `semantic`, `classify` |

CLIP і SigLIP 2 зберігають `classify` як типову задачу, тому `task="embed"`
потрібно запитувати явно. Їхня наявна контрольна точка `-cls` є спільним
артефактом із двома вежами; окрема контрольна точка `-embed` з ідентичними
вагами не публікується.

`embed_text` існує лише в CLIP і SigLIP 2, двох сімействах із текстовою вежею.
DINOv2 її не має. Ембединг DINOv2 обходить голови семантичної сегментації та
класифікації й читає кінцевий нормалізований токен CLS за розміру 224 пікселі;
варіанти `n`, `s`, `m` і `l` мають спільний кодувальник DINOv2-S, тому всі
чотири повертають `D = 384`.

Бекбони лише для класифікації, додані в цьому релізі,
[ViT](/docs/models/vit), [Swin](/docs/models/swin) і
[DeiT](/docs/models/deit), оголошують лише `classify` і не виконують цю задачу.

<code-tabs name="predict" />

`model.embed(source, **kwargs)` є скороченням для батча: воно запускає
`predict` і об'єднує кожен рядок кожного результату в один тензор CPU float32
у формі `(N_total, D)`, спричиняючи помилку, якщо рядки мають різні
розмірності. Сімейство без `embed` у підтримуваних задачах спричиняє
`NotImplementedError`.

## Об'єкти даних результатів

`result.embeddings` є об'єктом даних `Embeddings`. Його `data` завжди має
форму `(N, D)`, тип float32 і вже L2-нормалізовано шляхом інференсу; вхідні дані
з іншою кількістю вимірів спричиняють помилку замість непомітної зміни форми.

| Член | Значення |
|---|---|
| `.data` | Матриця `(N, D)` |
| `.dim` | `D` |
| `.normalized` | Ті самі рядки після захисної повторної нормалізації |
| `.similarity(other)` | `(N, M)` для іншого набору або `(N,)` для одного вектора `(D,)` |
| `.verify(i, j, threshold=0.4)` | Чи належать рядки `i` та `j` тому самому об'єкту |

`result.identities` є об'єктом даних `Identities`, наявним лише тоді, коли
передано галерею. Це звичайний контейнер, а не тензор, тому переміщення
`Results` між пристроями його не змінює.

| Член | Значення |
|---|---|
| `.name` | Список імен, `None` там, де жодне значення не перевищило поріг |
| `.score` | Найкраща косинусна оцінка float32 у формі `(N,)`, збережена навіть коли ім'я має значення `None` |
| `.data` | Список кортежів `(name, score)` |

<code-tabs name="similarity" />

Типово вектори не включаються до `summary()` і `to_json()`, оскільки рядок із
512 чисел float займає приблизно два кілобайти на об'єкт. Натомість кожен рядок
повідомляє `embedding_dim`, а також `identity` і `identity_score`, якщо
використано галерею. Щоб включити числа, передайте `summary(embeddings=True)`.

## Галереї

`Gallery` є іменованим набором еталонних рядків. Вона зберігає кожен еталон
окремо замість усереднення, тому ім'я оцінюється за його одним найкращим збігом,
а додавання невдалої фотографії не зміщує центроїд ідентичності.

<code-tabs name="gallery" />

`Gallery(model)` прив'язується до ваг, які створюватимуть її вектори.
`enroll(name, sources, select="best")` запускає передбачення для кожного
джерела й зберігає рядок із найвищою впевненістю для кожного результату;
`select="all"` натомість зберігає всі рядки, що потрібно, коли еталонне
зображення справді містить кілька об'єктів. `enroll_embedding(name, vector)`
пропускає інференс і безпосередньо приймає вектор, нормалізуючи його та
відхиляючи повністю нульовий рядок.

`FaceGallery` є постійним псевдонімом того самого класу, а архіви, записані
попередніми релізами лише для облич, досі завантажуються.

### Зіставлення та пороги

Зіставлення виконується щільним матричним множенням з усіма збереженими
еталонами, після чого для кожного імені залишається максимальна оцінка.
Наближеного індексу немає, що зберігає точність чисел і встановлює практичну
верхню межу розміру галереї.

Дві точки входу по-різному поводяться нижче порога. `match()` повертає
`[(name, score), ...]` для кожного рядка, відкидаючи все нижче порога, тому
рядок без збігів отримує порожній список. `identify()` повертає об'єкт даних
`Identities`, який завжди зберігає найкращу оцінку й задає ім'я як `None`,
якщо вона нижча за поріг. Жоден метод ніколи не підставляє найближче ім'я нижче
порога.

Усюди типовим порогом є `0.4`. Це косинусне значення, а не ймовірність;
правильна робоча точка залежить від ваших даних і допустимої кількості хибних
збігів, тому переберіть її на маркованих парах замість безумовного прийняття
типового значення. `libreyolo enroll` і аргумент передбачення `gallery=`
використовують те саме число.

### Збереження

`save(path)` записує стиснений файл `.npz` із векторами, іменами та блоком
метаданих, що містить версію формату, розмірність ембедингу й відбиток ваг,
які створили рядки. `Gallery.load(path, model=...)` перевіряє обидва значення
до будь-якого порівняння, тому використання галереї з іншою моделлю спричиняє
помилку замість непомітного порівняння векторів із двох не пов'язаних просторів.
Збереження порожньої галереї відхиляється.

## Командний рядок

| Команда | Призначення |
|---|---|
| `libreyolo enroll` | Обійти дерево з окремим каталогом для кожної ідентичності та записати чи розширити галерею `.npz` |
| `libreyolo compare` | Створити ембединги основного об'єкта у двох зображеннях і повідомити косинусну подібність |
| `libreyolo verify` | Та сама команда під другою назвою |
| `libreyolo predict gallery=...` | Додати ідентичності до звичайного запуску передбачення |

<code-tabs name="cli" />

Кожна команда LibreYOLO приймає і `key=value`, і `--key value`, тому
`gallery=refs.npz` і `--gallery refs.npz` є тим самим аргументом.

`enroll` приймає `model`, `source` і `gallery`, а також необов'язкові
`face-detector`, `device`, `--json` і `--quiet`. Команда читає по одному
каталогу для кожної ідентичності, де назва каталогу є ідентичністю, а кожне
зображення всередині додає еталони:

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

Зображення без результатів пропускається з рядком у stderr замість переривання
виконання, а підсумок повідомляє кількість збережених еталонів для кожного
імені. Наявний файл галереї розширюється на місці, тому ідентичності можна
додавати поступово.

`compare` і `verify` є однією функцією, зареєстрованою двічі. Вони приймають
`model`, `source`, `source2` і необов'язковий `threshold`, а потім виводять
косинусну подібність, висновок про однаковість або відмінність і поріг, за яким
його отримано. `--json` виводить ті самі три поля як об'єкт.

Для `predict` параметр `gallery` указує на збережений файл `.npz`, а
`gallery_threshold` перевизначає типове значення `0.4`. Передавання галереї до
моделі, задача якої не є `embed`, спричиняє помилку замість непомітної
бездіяльності, а за відсутнього файла галереї пропонується команда
`libreyolo enroll`, що його створить.

## Обличчя

Розпізнавання облич є регіональною формою цієї задачі й єдиною реалізацією цієї
форми, що постачається. Воно додає етап виявлення та вирівнювання перед головою
ембедингу, а також метод `verify()`, аргумент для власних рамок, опубліковані
показники accuracy й настанови з калібрування порога. Усе це наведено в розділі
[розпізнавання облич](/docs/tasks/face-recognition), який слід використовувати
як покроковий посібник для облич. Усе на цій сторінці застосовується до нього
без змін.

## Навчання, валідація та експорт

Ніщо в цій задачі не навчається всередині LibreYOLO. Голова ембедингу облич є
артефактом ONNX, для якого `train()`, `val()` і `export()` спричиняють помилку;
навчіть голову в upstream-проєкті й завантажте файл за шляхом. CLIP, SigLIP 2
і DINOv2 навчаються та експортуються через задачі класифікації й сегментації,
а не через `embed`.

Валідатора пошуку немає. Вимірюйте правильність верифікації на маркованих парах,
перебираючи `threshold`, а правильність ідентифікації вимірюйте, додавши дані
до галереї та прочитавши `identities.name` і `identities.score` для відкладених
зображень, вважаючи ім'я `None` відхиленням.
