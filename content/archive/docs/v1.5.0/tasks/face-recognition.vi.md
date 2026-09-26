---
title: Nhận dạng khuôn mặt
seo_title: Nhận dạng khuôn mặt trong LibreYOLO
description: >-
  Phát hiện, tạo embedding và nhận dạng khuôn mặt trong LibreYOLO. Đăng ký
  gallery, so sánh hai ảnh và khớp bằng cosine similarity từ Python hoặc CLI.
lead: >-
  Nhận dạng khuôn mặt là tác vụ embed áp dụng cho khuôn mặt. Detector định vị và
  căn chỉnh từng khuôn mặt, recognition head trả về một vector được chuẩn hóa L2
  trên mỗi khuôn mặt, còn danh tính được quyết định bằng cosine similarity với
  các tham chiếu đã đăng ký thay vì danh sách lớp cố định.
keywords:
  - nhận dạng khuôn mặt python
  - face embedding
  - xác minh khuôn mặt
  - face gallery
  - arcface onnx
  - tác vụ embed libreyolo
  - cosine similarity khuôn mặt
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Tên librefacerec-* định tuyến tới family face embedding bất kể
        # hậu tố tệp, và tải từ tổ chức Hugging Face LibreYOLO trong lần
        # sử dụng đầu tiên cùng face detector mặc định.
        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)             # hộp khuôn mặt (N, 4)
        print(result.embeddings.data.shape)  # (N, D), mỗi khuôn mặt một dòng
        print(result.embeddings.dim)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=photo.jpg
    - label: So sánh hai ảnh
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        # Chạy phát hiện và embedding trên cả hai ảnh rồi so sánh khuôn mặt
        # có độ tin cậy cao nhất. Cosine similarity nằm trong [-1, 1].
        outcome = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)
        print(outcome["similarity"], outcome["same_person"])
    - label: Đăng ký gallery và nhận dạng
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
            print(name, score)   # name là None khi dưới ngưỡng
    - label: Đăng ký và nhận dạng từ CLI
      language: bash
      code: >
        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=faces.npz

        libreyolo predict model=librefacerec-l.onnx source=group_photo.jpg
        gallery=faces.npz
    - label: Dùng hộp khuôn mặt riêng
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")

        # face_boxes bỏ qua hoàn toàn phát hiện; face_detector nhận callable,
        # mô hình phát hiện LibreYOLO hoặc instance FaceDetector.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])
        print(result.embeddings.data.shape)
source_hash: d7dfcb6f812ebb2d
---

## Định nghĩa

Nhận dạng khuôn mặt trả về một vector trên mỗi khuôn mặt, không phải nhãn. Dự
đoán chạy hai giai đoạn: face detector định vị từng khuôn mặt và năm landmark,
vùng crop được warp về căn chỉnh chuẩn 112x112, rồi recognition head phát ra
embedding được chuẩn hóa L2.

`result.embeddings` là payload `Embeddings` có shape `(N, D)`, các dòng được căn
theo `result.boxes`, vì vậy dòng `i` mô tả khuôn mặt trong hộp `i`. Vì các dòng
là vector đơn vị, cosine similarity là dot product, còn
`embeddings.similarity()` tính giá trị này với một `Embeddings` khác hoặc toàn
bộ ma trận trong một lời gọi.

Đặt tên khuôn mặt là bước riêng. `Gallery` giữ các vector tham chiếu đã đặt tên;
truyền `gallery=` cho `predict()` sẽ gắn `result.identities`, được căn theo các
embedding, chứa tên và điểm cosine tốt nhất trên mỗi khuôn mặt. Khuôn mặt dưới
ngưỡng khớp giữ `None` làm tên, và tên gần nhất nhưng dưới ngưỡng không bao giờ
được thay vào.

Key tác vụ chuẩn của thư viện là `embed`. `face-recognition`,
`facial-recognition`, `reid` và `face` đều được chuẩn hóa về key đó, vì vậy
`task="face-recognition"` và `task="embed"` chọn cùng một nội dung. Khuôn mặt là
shape vùng của tác vụ rộng hơn này; phần [embedding](/docs/tasks/embeddings) bao
gồm shape toàn ảnh và văn bản, API dùng chung `Embeddings`, `Identities` cùng
`Gallery`, và các mô hình tạo vector mà không phát hiện gì.

## Mô hình

[LibreFaceRec](/docs/models/librefacerec) là family cho tác vụ này. Nó gồm hai
artifact ONNX phía sau một lời gọi: `librefacerec-l.onnx`, recognition head
iResNet100 tạo embedding 512 chiều, và `librefacerec-det.onnx`, face detector
mặc định có năm landmark, lấy từ OpenCV zoo. Cả hai được tải từ tổ chức Hugging
Face LibreYOLO trong lần sử dụng đầu tiên. Mọi tệp ONNX khác theo quy ước ArcFace
(đầu vào 112x112 đã căn chỉnh, đầu ra `(N, D)`) có thể thay recognition head
bằng cách truyền đường dẫn thay vì tên `librefacerec-*`.

Key tác vụ `embed` rộng hơn khuôn mặt. [CLIP](/docs/models/clip),
[SigLIP2](/docs/models/siglip2) và [DINOv2](/docs/models/dinov2) cũng hỗ trợ
`task="embed"` và trả về một vector toàn ảnh, tức truy xuất ảnh thay vì danh tính
khuôn mặt. Chúng dùng chung API `Gallery` và `Embeddings`, nên quy trình đăng ký
và khớp bên dưới có thể chuyển sang, nhưng chúng không phát hiện hoặc căn chỉnh
khuôn mặt.

Recognition head chạy qua `onnxruntime`, package không có trong bản cài đặt cơ
sở:

```bash
pip install "libreyolo[onnx]"
```

## Dự đoán

<code-tabs name="predict" />

Khi giữ nguyên, `predict()` tải và ghép detector mặc định. `face_detector` ghi
đè bằng callable, mô hình phát hiện LibreYOLO hoặc instance `FaceDetector`, và
có thể được đặt trên constructor hoặc theo từng lời gọi. `face_boxes` bỏ qua
phát hiện bằng các hộp bạn đã có. Trên CLI, `face_detector=` nhận đường dẫn
`.onnx` của face detector hoặc tên detector LibreYOLO.

`model.verify(image_a, image_b)` là lối tắt hai ảnh: nó tạo embedding cho khuôn
mặt có độ tin cậy cao nhất trong mỗi ảnh và trả về
`{"similarity", "same_person", "threshold"}`. `model.embed(sources)` trả về mọi
dòng khuôn mặt trên một hoặc nhiều ảnh được xếp thành một tensor
`(N_total, D)`. Xem [dự đoán](/docs/predict) để biết về nguồn, stream và cách xử
lý kết quả.

## Định dạng dataset

Quá trình đăng ký đọc một thư mục cho mỗi danh tính. Tên thư mục trở thành danh
tính, và mỗi ảnh bên trong đóng góp tham chiếu cho tên đó:

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

`libreyolo enroll` duyệt cây đó và ghi gallery `.npz`. Tệp gallery hiện có được
mở rộng tại chỗ thay vì thay thế, nên có thể thêm danh tính theo thời gian.
Gallery được gắn với trọng số đã tạo ra nó bằng số chiều embedding và fingerprint
của tệp; khớp bằng mô hình khác sẽ phát sinh lỗi thay vì so sánh các không gian
vector không tương thích.

Theo mặc định, mỗi ảnh nguồn đóng góp một dòng tham chiếu, khuôn mặt có độ tin
cậy cao nhất, vì vậy ảnh chân dung có người đứng cạnh chỉ đăng ký chủ thể chính.
Truyền `select="all"` cho `Gallery.enroll` để lưu mọi dòng trả về.

## Huấn luyện

Không family nào trong tác vụ này huấn luyện bên trong LibreYOLO.
`LibreFaceEmbedder.train()` phát sinh lỗi: hãy huấn luyện recognition head ở
upstream, xuất sang ONNX theo quy ước ArcFace rồi nạp tệp bằng đường dẫn.

## Xác thực

Tác vụ này không có validator dataset và `val()` phát sinh lỗi thay vì giả vờ
khác đi. Độ chính xác xác minh được đo trên các cặp ảnh có nhãn bằng
`model.verify()`, quét `threshold` để chọn operating point mong muốn. Độ chính
xác nhận dạng được đo bằng cách đăng ký gallery rồi đọc
`result.identities.name` và `result.identities.score` trên các ảnh giữ lại,
trong đó tên `None` được tính là từ chối.

## Xuất

Recognition head đã là graph ONNX, vì vậy không có gì để chuyển đổi:
`LibreFaceEmbedder.export()` phát sinh lỗi. Hãy triển khai trực tiếp tệp `.onnx`
hoặc trỏ LibreYOLO tới tệp đó để family xử lý phát hiện, căn chỉnh và chuẩn hóa.
