---
title: Embedding
seo_title: Embedding ảnh và vùng trong LibreYOLO
description: >-
  Tác vụ embed trả về vector float32 chuẩn hóa L2 cho toàn ảnh, từng vùng phát
  hiện hoặc văn bản. Đăng ký gallery, khớp bằng cosine similarity và tìm kiếm từ
  Python hoặc CLI.
lead: >-
  Một tác vụ bao phủ mọi vector LibreYOLO tạo ra. embed trả về các dòng float32
  có độ dài đơn vị, với dot product là điểm tương đồng, dù dòng mô tả toàn ảnh,
  một khuôn mặt được phát hiện hay một dòng văn bản, và cùng Gallery khớp tất cả
  chúng.
keywords:
  - image embedding python
  - embedding chuẩn hóa l2
  - tìm kiếm cosine similarity
  - tác vụ embed libreyolo
  - truy xuất ảnh
  - đăng ký gallery
  - clip embedding
  - dinov2 embedding
  - reid embedding
last_verified: 1.5.0
verification: >-
  Key tác vụ và alias được đọc từ libreyolo/tasks.py. Payload kết quả lấy từ các
  class Embeddings và Identities trong libreyolo/utils/results.py. API Gallery
  từ libreyolo/utils/gallery.py. embed và _postprocess_embeddings từ
  libreyolo/models/base/model.py. Các family được hỗ trợ được xác định bằng cách
  tìm embed trong SUPPORTED_TASKS ở libreyolo/models/**/model.py. Bề mặt CLI từ
  libreyolo/cli/__init__.py, libreyolo/cli/commands/special.py và
  libreyolo/cli/commands/predict.py. Ý đồ thiết kế từ
  docs/adr/0015-embed-generalization.md.
meta:
  - label: Key tác vụ
    value: embed
    mono: true
  - label: Alias
    value: 'face-recognition, reid, face'
    mono: true
  - label: Payload kết quả
    value: 'Embeddings, Identities'
    mono: true
  - label: Dtype của dòng
    value: 'float32, độ dài đơn vị'
snippets:
  predict:
    - label: Toàn ảnh
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # CLIP mặc định dùng classify, vì vậy hãy yêu cầu vector tường minh.
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)  # (1, 512), mỗi ảnh một dòng
        print(result.boxes)                  # None: không định vị gì
    - label: Theo từng vùng
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        # Dòng i mô tả vùng trong hộp i.
        print(result.boxes.xyxy.shape)       # (N, 4)
        print(result.embeddings.data.shape)  # (N, 512)
    - label: Nhiều ảnh cùng lúc
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # Mọi dòng từ mọi kết quả được nối thành một tensor.
        vectors = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(vectors.shape)  # (3, 384)
    - label: Văn bản
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        # Văn bản là một phương thức, không bao giờ là nguồn dự đoán. Chuỗi được
        # truyền cho model(...) vẫn là đường dẫn hoặc URL.
        text = model.embed_text(["a photo of a cat", "a photo of a dog"])
        print(text.shape)  # (2, 512)
  similarity:
    - label: So sánh hai tập dòng
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        query = model.embed("query.jpg")          # (1, 512)
        pool = model.embed(["a.jpg", "b.jpg"])    # (2, 512)

        # Các dòng có độ dài đơn vị, vì vậy cosine similarity là dot product.
        scores = model("query.jpg").embeddings.similarity(pool)
        print(scores.shape)  # (1, 2)
    - label: Ảnh so với văn bản
      language: python
      code: |
        import torch

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        image = model.embed("photo.jpg")                       # (1, 512)
        text = model.embed_text(["a cat", "a dog", "a car"])   # (3, 512)

        print(torch.matmul(image, text.T))
  gallery:
    - label: Đăng ký và nhận dạng
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
            print(name, score)   # name là None khi dưới ngưỡng
    - label: Tìm kiếm top-k
      language: python
      code: |
        from libreyolo import Gallery
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        gallery = Gallery.load("refs.npz", model=model)

        result = model("query.jpg")
        matches = gallery.match(result.embeddings, top_k=5, threshold=0.4)
        print(matches[0])   # [(name, score), ...] cho dòng đầu tiên
    - label: Đăng ký vector đã có
      language: python
      code: |
        from libreyolo import Gallery

        gallery = Gallery()
        gallery.enroll_embedding("ada", vector)  # được chuẩn hóa khi đưa vào
        print(gallery.identities, gallery.dim, len(gallery))
  cli:
    - label: Đăng ký cây thư mục
      language: bash
      code: >
        # source/<identity>/*.jpg. Gallery hiện có được mở rộng tại chỗ.

        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=refs.npz
    - label: Nhận dạng trong khi dự đoán
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=group.jpg \
          gallery=refs.npz gallery_threshold=0.45
    - label: So sánh hai ảnh
      language: bash
      code: >
        libreyolo compare model=librefacerec-l.onnx \
          source=a.jpg source2=b.jpg threshold=0.4

        # verify là cùng lệnh dưới tên thứ hai.

        libreyolo verify model=librefacerec-l.onnx source=a.jpg source2=b.jpg
        --json
source_hash: ffbaad5599035bc7
---

## Định nghĩa

`embed` chuyển ảnh, vùng ảnh hoặc chuỗi thành một dòng float32 có chiều rộng cố
định và độ dài bằng một. Vì mọi dòng là vector đơn vị, so sánh hai dòng là dot
product, còn so sánh hai tập là một phép nhân ma trận. Không nội dung nào khác
trong tác vụ phụ thuộc vào mô hình: truy xuất, phát hiện trùng lặp, tái định danh
và nhận dạng khuôn mặt đều là cùng phép tính trên các dòng khác nhau.

Vector là đầu ra. Không có danh sách lớp, vì vậy tên được gắn sau bằng cách so
sánh với tham chiếu bạn cung cấp thay vì thứ mà mạng được huấn luyện để dự đoán.

### Ba shape

| Shape | `Results.embeddings` | `Results.boxes` | Được tạo bởi |
|---|---|---|---|
| Toàn ảnh | `(1, D)` | `None` | Truyền ảnh cho family toàn ảnh |
| Vùng | `(N, D)` | `(N, 4)`, căn theo dòng | Family định vị trước, như nhận dạng khuôn mặt |
| Văn bản | hoàn toàn không phải `Results` | | `model.embed_text(texts)`, trả về `(M, D)` |

Kết quả toàn ảnh giữ dạng hai chiều ngay cả với một ảnh. `(D,)` không phải shape
trả về được phép, vì vậy bên sử dụng không cần xử lý riêng trường hợp một dòng.
Văn bản trả về tensor thuần thay vì `Results` vì chuỗi không phải nguồn ảnh:
truyền chuỗi cho `model(...)` vẫn có nghĩa là đường dẫn hoặc URL, và thư viện
không bao giờ đoán chuỗi là văn xuôi.

Key tác vụ chuẩn là `embed`. `embedding`, `embeddings`, `face-recognition`,
`facial-recognition`, `recognition`, `face`, `faceid` và `reid` đều được chuẩn
hóa về key đó, vì vậy `task="reid"` và `task="embed"` chọn đúng cùng một nội dung.

## Mô hình

Bốn family phục vụ tác vụ và được chia rõ theo việc có định vị gì trước hay không.

| Family | Shape | Số chiều | Cũng hỗ trợ |
|---|---|---|---|
| [LibreFaceRec](/docs/models/librefacerec) | Vùng, mỗi khuôn mặt phát hiện một dòng | 512 | Không có; `embed` là tác vụ duy nhất |
| [CLIP](/docs/models/clip) | Toàn ảnh, có text tower ghép cặp | 512 cho `b32` và `b16`, 768 cho `l14` | `classify`, vẫn là mặc định |
| [SigLIP 2](/docs/models/siglip2) | Toàn ảnh, có text tower ghép cặp | 768 cho `b16`, 1152 cho `so400m` | `classify`, vẫn là mặc định |
| [DINOv2](/docs/models/dinov2) | Toàn ảnh, chỉ ảnh | 384 | `semantic`, `classify` |

CLIP và SigLIP 2 giữ `classify` làm tác vụ mặc định, vì vậy phải yêu cầu
`task="embed"`. Checkpoint `-cls` hiện có của chúng là artifact hai tower dùng
chung; không có checkpoint `-embed` trùng lặp cho cùng trọng số.

`embed_text` chỉ tồn tại trên CLIP và SigLIP 2, hai family có text tower. DINOv2
không có. Embedding DINOv2 bỏ qua semantic head và classification head rồi đọc
token CLS cuối đã chuẩn hóa ở 224 pixel; các biến thể `n`, `s`, `m` và `l` đều
dùng chung encoder DINOv2-S, vì vậy cả bốn trả về `D = 384`.

Các backbone chỉ dành cho phân loại được thêm trong bản phát hành này,
[ViT](/docs/models/vit), [Swin](/docs/models/swin) và
[DeiT](/docs/models/deit), chỉ khai báo `classify` và không phục vụ tác vụ này.

<code-tabs name="predict" />

`model.embed(source, **kwargs)` là lối tắt batch: nó chạy `predict` và nối mọi
dòng từ mọi kết quả thành tensor float32 CPU `(N_total, D)`, phát sinh lỗi nếu
các dòng có số chiều khác nhau. Family không có `embed` trong tác vụ được hỗ trợ
sẽ phát sinh `NotImplementedError`.

## Payload kết quả

`result.embeddings` là payload `Embeddings`. `data` luôn là float32 `(N, D)`, đã
được đường dẫn inference chuẩn hóa L2, còn đầu vào không phải hai chiều sẽ phát
sinh lỗi thay vì được âm thầm reshape.

| Thành viên | Ý nghĩa |
|---|---|
| `.data` | Ma trận `(N, D)` |
| `.dim` | `D` |
| `.normalized` | Cùng các dòng được chuẩn hóa lại để phòng vệ |
| `.similarity(other)` | `(N, M)` so với tập khác, hoặc `(N,)` so với vector `(D,)` duy nhất |
| `.verify(i, j, threshold=0.4)` | Dòng `i` và `j` có cùng chủ thể hay không |

`result.identities` là payload `Identities`, chỉ có khi truyền gallery. Đây là
container thuần, không phải tensor, vì vậy di chuyển `Results` giữa các thiết bị
không tác động tới nó.

| Thành viên | Ý nghĩa |
|---|---|
| `.name` | Danh sách tên, `None` ở nơi không có gì vượt ngưỡng |
| `.score` | Điểm cosine tốt nhất float32 `(N,)`, được giữ ngay cả khi tên là `None` |
| `.data` | Danh sách tuple `(name, score)` |

<code-tabs name="similarity" />

Theo mặc định, vector bị loại khỏi `summary()` và `to_json()` vì một dòng 512
float chiếm khoảng hai kilobyte trên mỗi chủ thể. Mỗi dòng báo cáo
`embedding_dim`, cùng `identity` và `identity_score` khi dùng gallery. Truyền
`summary(embeddings=True)` để đưa các số vào.

## Gallery

`Gallery` là tập tham chiếu đã đặt tên. Nó lưu riêng từng tham chiếu thay vì lấy
trung bình, vì vậy tên được tính điểm bằng tham chiếu khớp tốt nhất duy nhất,
còn thêm ảnh xấu không thể kéo centroid của danh tính đi.

<code-tabs name="gallery" />

`Gallery(model)` gắn với trọng số tạo vector. `enroll(name, sources,
select="best")` chạy dự đoán trên mỗi nguồn và giữ dòng có độ tin cậy cao nhất
trên mỗi kết quả; `select="all"` giữ mọi dòng, phù hợp khi ảnh tham chiếu thực sự
có nhiều chủ thể. `enroll_embedding(name, vector)` bỏ qua inference và nhận
vector trực tiếp, chuẩn hóa nó và từ chối dòng toàn 0.

`FaceGallery` là alias lâu dài của cùng class, còn archive được ghi bởi các bản
phát hành cũ chỉ hỗ trợ khuôn mặt vẫn nạp được.

### Khớp và ngưỡng

Phép khớp là phép nhân ma trận dày đặc với mọi tham chiếu đã lưu, sau đó reduce
về một điểm trên mỗi tên bằng giá trị lớn nhất. Không có approximate index, nhờ
vậy số liệu chính xác nhưng tạo giới hạn thực tế cho kích thước gallery.

Hai điểm vào khác nhau ở hành vi dưới ngưỡng. `match()` trả về
`[(name, score), ...]` trên mỗi dòng sau khi loại mọi thứ dưới ngưỡng, nên dòng
không khớp có danh sách rỗng. `identify()` trả về payload `Identities` luôn giữ
điểm tốt nhất và đặt tên thành `None` khi dưới ngưỡng. Không phương thức nào thay
vào tên gần nhất dưới ngưỡng.

Ngưỡng mặc định là `0.4` ở mọi nơi. Đây là giá trị cosine, không phải xác suất,
còn operating point phù hợp là thuộc tính dữ liệu và mức chấp nhận false match,
vì vậy hãy quét trên các cặp có nhãn thay vì chấp nhận mặc định. `libreyolo
enroll` và đối số dự đoán `gallery=` dùng cùng con số.

### Lưu trữ

`save(path)` ghi `.npz` nén chứa vector, tên và khối metadata mang phiên bản
định dạng, số chiều embedding cùng fingerprint của trọng số tạo ra các dòng.
`Gallery.load(path, model=...)` kiểm tra cả hai trước khi so sánh, vì vậy trỏ
gallery tới mô hình khác sẽ phát sinh lỗi thay vì âm thầm tính điểm giữa vector
từ hai không gian không liên quan. Không thể lưu gallery rỗng.

## Dòng lệnh

| Lệnh | Mục đích |
|---|---|
| `libreyolo enroll` | Duyệt cây mỗi danh tính một thư mục và ghi hoặc mở rộng gallery `.npz` |
| `libreyolo compare` | Tạo embedding cho chủ thể chính trong hai ảnh và báo cáo cosine similarity |
| `libreyolo verify` | Cùng lệnh dưới tên thứ hai |
| `libreyolo predict gallery=...` | Gắn danh tính vào lượt dự đoán thông thường |

<code-tabs name="cli" />

Mọi lệnh LibreYOLO nhận cả `key=value` và `--key value`, vì vậy
`gallery=refs.npz` và `--gallery refs.npz` là cùng đối số.

`enroll` nhận `model`, `source` và `gallery`, cùng `face-detector`, `device`,
`--json` và `--quiet` tùy chọn. Nó đọc một thư mục trên mỗi danh tính, trong đó
tên thư mục là danh tính và mọi ảnh bên trong đóng góp tham chiếu:

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

Ảnh không tạo ra gì được bỏ qua kèm một dòng trên stderr thay vì dừng lượt chạy,
còn summary báo cáo số tham chiếu đã lưu cho từng tên. Tệp gallery hiện có được
mở rộng tại chỗ, vì vậy có thể thêm danh tính theo thời gian.

`compare` và `verify` là một hàm được đăng ký hai lần. Chúng nhận `model`,
`source`, `source2` và `threshold` tùy chọn, rồi in cosine similarity, kết luận
giống hay khác và ngưỡng tạo ra kết luận. `--json` in cùng ba trường dưới dạng
đối tượng.

Trên `predict`, `gallery` trỏ tới `.npz` đã lưu, còn `gallery_threshold` ghi đè
mặc định `0.4`. Truyền gallery cho mô hình có tác vụ không phải `embed` là lỗi
thay vì no-op im lặng, còn tệp gallery bị thiếu sẽ gợi ý lệnh `libreyolo enroll`
để tạo nó.

## Khuôn mặt

Nhận dạng khuôn mặt là shape vùng của tác vụ này và là implementation duy nhất
được phân phối cho shape đó. Nó thêm giai đoạn phát hiện và căn chỉnh trước
embedding head, cùng phương thức `verify()`, đối số hộp riêng, số liệu độ chính
xác đã công bố và hướng dẫn hiệu chuẩn ngưỡng. Tất cả nằm tại [nhận dạng khuôn
mặt](/docs/tasks/face-recognition), hướng dẫn cần theo khi chủ thể là khuôn mặt.
Mọi nội dung trên trang này áp dụng nguyên vẹn.

## Huấn luyện, xác thực và xuất

Không nội dung nào trong tác vụ này huấn luyện bên trong LibreYOLO. Face
embedding head là artifact ONNX có `train()`, `val()` và `export()` đều phát
sinh lỗi; hãy huấn luyện head ở upstream rồi nạp tệp bằng đường dẫn. CLIP,
SigLIP 2 và DINOv2 huấn luyện và xuất qua tác vụ phân loại cùng phân đoạn, không
qua `embed`.

Không có validator truy xuất. Hãy đo độ chính xác xác minh trên các cặp có nhãn
bằng cách quét `threshold`, còn độ chính xác nhận dạng bằng cách đăng ký gallery
và đọc `identities.name` cùng `identities.score` trên ảnh giữ lại, trong đó tên
`None` được tính là từ chối.
