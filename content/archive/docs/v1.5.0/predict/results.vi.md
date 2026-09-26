---
title: Làm việc với kết quả
seo_title: Đối tượng Results của LibreYOLO
description: >-
  Mỗi ảnh có một đối tượng Results với một slot cho từng loại payload: box,
  mask, keypoint, probs, độ sâu, panoptic, OCR và nhiều loại khác. Vẽ, lưu và
  JSON.
lead: >-
  Mỗi dự đoán trả về một đối tượng Results cho từng ảnh. Đối tượng có một slot
  được đặt tên cho mỗi loại payload; tất cả đều rỗng trừ các slot mô hình tạo
  ra, và artifact đã xuất cũng có cùng các slot.
keywords:
  - đối tượng results yolo python
  - results.boxes xyxy
  - chuyển results sang json
  - lưu ảnh chú thích
  - mask phân đoạn python
  - kết quả keypoint
  - kết quả bản đồ độ sâu
  - tóm tắt results
  - onnx cùng results
last_verified: 1.5.0
verification: >-
  Các lớp payload, slot, ngữ nghĩa di chuyển, summary(), to_json(), plot(),
  save() và cutout() được đọc từ libreyolo/utils/results.py. Hành vi chú thích
  và ghi ra đĩa lấy từ InferenceRunner._save_annotated_image trong
  libreyolo/models/base/inference.py và resolve_save_path trong
  libreyolo/utils/general.py. Cách định tuyến theo hậu tố lấy từ LibreYOLO()
  trong libreyolo/models/__init__.py.
snippets:
  basic:
    - label: Box
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(SAMPLE_IMAGE)


        print(result.orig_shape)   # (chiều cao, chiều rộng) của ảnh nguồn

        print(result.path)         # đường dẫn nguồn, None cho đầu vào trong bộ
        nhớ


        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: Tọa độ chuẩn hóa
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy[:1])    # pixel, x1 y1 x2 y2

        print(result.boxes.xywh[:1])    # pixel, tâm x, tâm y, w, h

        print(result.boxes.xyxyn[:1])   # cùng box chia cho chiều rộng và chiều
        cao

        print(result.boxes.xywhn[:1])
    - label: NumPy và thiết bị
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        # Mỗi thao tác trả về Results mới; bản gốc không đổi.
        as_numpy = result.numpy()
        on_cpu = result.cpu()

        print(type(as_numpy.boxes.xyxy).__name__)
        print(type(on_cpu.boxes.xyxy).__name__)
  json:
    - label: summary và to_json
      language: python
      code: |
        import json

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        rows = result.summary()
        print(json.dumps(rows[:2], indent=2))

        # Cùng nội dung dưới dạng chuỗi, với cùng các đối số từ khóa.
        print(result.to_json(normalize=True, decimals=3)[:200])
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt --json \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  saving:
    - label: Ảnh có chú thích
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # save=True vẽ payload và ghi vào runs/detect/predict*.
        result = model(SAMPLE_IMAGE, save=True)
        print(result.saved_path)
  exported:
    - label: Cài extra xuất
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: Cùng Results từ artifact đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")   # trả về đường dẫn đã ghi

        # LibreYOLO() định tuyến theo hậu tố tệp.
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)

        print(type(result).__name__, len(result.boxes))
source_hash: 548dbc9c7f5552ec
---

## Một đối tượng, một slot cho mỗi payload

Dự đoán trên một ảnh trả về một `Results`. Đối tượng chứa mười tám slot payload,
và mô hình chỉ điền các slot do tác vụ tạo ra. Mọi slot khác là `None`, nên đọc
`result.masks` trên detector sẽ nhận `None` thay vì lỗi.

| Slot | Lớp | Shape | Được tạo bởi |
|---|---|---|---|
| `boxes` | `Boxes` | `(N, 4)` cùng điểm và lớp | Phát hiện và mọi tác vụ định vị trước |
| `masks` | `Masks` | `(N, H, W)` | Phân đoạn instance |
| `keypoints` | `Keypoints` | `(N, K, 2)` hoặc `(N, K, 3)` | Tư thế |
| `probs` | `Probs` | `(C,)` | Phân loại |
| `obb` | `OBB` | `(N, 7)` hoặc `(N, 8)` | Box định hướng |
| `gaze` | `Gaze` | `(N, 2)` pitch và yaw theo radian | Ước lượng hướng nhìn |
| `points` | `Points` | `(N, 4)` gồm x, y, lớp, độ tin cậy | Định vị điểm |
| `semantic_mask` | `SemanticMask` | `(H, W)` class id | Phân đoạn ngữ nghĩa |
| `panoptic` | `PanopticSegmentation` | `(H, W)` segment id cùng `segments_info` | Phân đoạn panoptic |
| `depth_map` | `DepthMap` | `(H, W)` số thực | Ước lượng độ sâu |
| `normal_map` | `NormalMap` | `(H, W, 3)` vector đơn vị | Pháp tuyến bề mặt |
| `edges` | `EdgeMap` | `(H, W)` số thực trong `[0, 1]` | Phát hiện cạnh |
| `restored` | `RestoredImage` | `(H, W, 3)` uint8 RGB | Khôi phục và siêu phân giải |
| `matte` | `Matte` | `(H, W)` số thực trong `[0, 1]` | Alpha matting và xóa nền |
| `ocr` | `OCRRegions` | `(N, 4, 2)` đa giác cùng văn bản | Phát hiện và nhận dạng văn bản |
| `embeddings` | `Embeddings` | `(N, D)` các hàng chuẩn hóa L2 | Tác vụ `embed` |
| `identities` | `Identities` | N tên và điểm | Tác vụ `embed` với gallery |
| `meshes` | `Meshes` | Tham số cơ thể và vertex tùy chọn | Khôi phục mesh cơ thể |

Bên cạnh đó là các trường có trong mọi kết quả: `orig_shape` dưới dạng
`(height, width)`, `path` (đường dẫn nguồn hoặc `None` cho đầu vào trong bộ nhớ),
`names` ánh xạ class id đến tên lớp, `frame_idx` cho video và frame trực tiếp,
`track_id` khi theo dõi và `restore_scale`, hệ số upscale nguyên của kết quả khôi phục.

`result.normals` là bí danh của `result.normal_map`.

`result.speed` tồn tại trên mọi kết quả nhưng chỉ được điền bởi
[ensemble](/docs/predict/ensembling), với các khóa `member_0`, `member_1` và
`fusion` theo mili giây. Với mô hình đơn, nó vẫn là dict rỗng.

## Box

<code-tabs name="basic" />

`Boxes` giữ tọa độ và điểm trong các mảng riêng thay vì một tensor đóng gói.

| Thuộc tính | Nội dung |
|---|---|
| `xyxy` | `(N, 4)` pixel tuyệt đối, x1 y1 x2 y2 |
| `xywh` | `(N, 4)` pixel tuyệt đối, tâm x, tâm y, chiều rộng, chiều cao |
| `xyxyn`, `xywhn` | Các giá trị tương tự chia cho chiều rộng và chiều cao ảnh |
| `conf` | `(N,)` độ tin cậy |
| `cls` | `(N,)` class id dưới dạng số thực |
| `id` | `(N,)` track id hoặc `None` |
| `is_track` | `id` có được đặt hay không |
| `data` | Mọi thứ được nối: box, id tùy chọn, conf, cls |

`cls` là mảng số thực, vì vậy hãy dùng dưới dạng `result.names[int(cls)]`.

`xyxyn` và `xywhn` cần `orig_shape`, được `Results` điền cho bạn.

## Payload dày đặc

Payload bao phủ toàn ảnh hoạt động khác payload theo instance, và sự khác biệt có ý nghĩa khi cắt lát.

`SemanticMask` chứa class id `(H, W)` trên canvas gốc, với `255` dành riêng làm
giá trị bỏ qua và không bao giờ được tính là lớp. `classes` liệt kê các id xuất
hiện và loại giá trị đó; `class_mask(id)` trả về boolean `(H, W)`.

`PanopticSegmentation` chứa segment id `(H, W)`, trong đó `0` là void id, cùng
danh sách dict `segments_info` chứa ít nhất `id` và `category_id`. `segment_ids`
liệt kê các id xuất hiện, còn `segment_mask(id)` chọn một id.

`DepthMap` chứa nghịch đảo độ sâu tương đối `(H, W)`: cao hơn nghĩa là gần hơn,
và giá trị không dùng đơn vị mét. Nó cung cấp `min`, `max`, `mean` trên các giá
trị hữu hạn và `normalized()` để scale lại về `[0, 1]`.

`NormalMap` chứa các vector đơn vị `(H, W, 3)` trong hệ tọa độ camera OpenCV,
với `+x` sang phải, `+y` hướng xuống và `+z` vào cảnh, nên bề mặt hướng về camera
là `(0, 0, -1)`. `assert_normalized()` kiểm tra mọi pixel hữu hạn và có độ dài đơn vị.

`EdgeMap` chứa float32 `(H, W)` trong `[0, 1]`. Bản đồ liên tục được giữ lại thay
vì áp ngưỡng, nên `binary(threshold=0.5)` là nơi bạn chọn ngưỡng cắt.

`Matte` chứa float32 `(H, W)` trong `[0, 1]`, trong đó `1` là hoàn toàn tiền cảnh.
`array` trả về dữ liệu đã cắt giới hạn dưới dạng float32.

`RestoredImage` chứa RGB uint8 `(H, W, 3)`, với `array` cho ndarray thô và
`save(path)` để ghi dữ liệu.

`Probs` chứa một vector xác suất cho ảnh. `top1` và `top5` là chỉ mục lớp,
`top1conf` và `top5conf` là các điểm tương ứng.

`Embeddings` chứa các hàng `(N, D)` đã chuẩn hóa L2, nên độ tương đồng cosine là
một tích vô hướng. `similarity(other)` trả về `(N, M)` khi so với gallery hoặc
`(N,)` khi so với một vector, còn `verify(i, j, threshold=0.4)` so sánh hai hàng.

`OCRRegions` chứa các đa giác `(N, 4, 2)` theo thứ tự đọc, với các góc theo thứ
tự trên-trái, trên-phải, dưới-phải, dưới-trái. Văn bản nằm trong `texts`, điểm
nhận dạng trong `conf`, điểm phát hiện trong `det_conf`. Vì đây là đa giác xoay
thực sự, chúng không điền `boxes`; `ocr.xyxy` cung cấp bao lồi thẳng trục khi bạn
cần hình chữ nhật.

## Cắt lát và di chuyển

`result[i]` trả về `Results` mới chứa một instance. Payload theo instance được
cắt lát; payload toàn ảnh được giữ nguyên, nên cắt lát kết quả phân loại không
thể cắt vector xác suất xuống một lớp, còn cắt lát kết quả độ sâu không thể làm
hỏng bố cục `(H, W)`.

`len(result)` đếm instance: box, điểm, embedding, vùng OCR hoặc mesh. Mọi
payload toàn ảnh dày đặc được tính là `1`. Kết quả không chứa gì có giá trị `0`.

`to()`, `cpu()`, `cuda()` và `numpy()` đều trả về `Results` mới với mọi slot đã
điền được chuyển đổi. Chúng không sửa bản gốc.

`update()` là phương thức duy nhất thay đổi tại chỗ, thay thế các slot được nêu tên và trả về cùng đối tượng.

## JSON

<code-tabs name="json" />

`summary()` trả về danh sách dict thông thường, còn `to_json()` là danh sách đó
được đưa qua `json.dumps`. Cả hai nhận cùng ba đối số: `normalize=False` chuyển
tọa độ sang `[0, 1]`, `decimals=5` đặt mức làm tròn và `embeddings=False` điều
khiển việc đưa vector embedding vào.

Shape hàng tuân theo payload. Hàng phát hiện chứa `name`, `class`, `confidence`
và dict `box`, đồng thời có thêm `segments` khi có mask, `obb` và `corners` cho
box định hướng, góc `gaze` theo cả radian lẫn độ, `track_id` khi theo dõi và tham
số `mesh` khi có mesh.

Khi không có box, một payload quyết định các hàng: OCR phát một hàng cho mỗi vùng
cùng `text`, điểm phát một hàng cho mỗi điểm, panoptic phát một hàng cho mỗi
segment cùng `pixel_count` và `pixel_fraction`, semantic phát một hàng cho mỗi
lớp xuất hiện, còn phân loại phát năm lớp đứng đầu. Độ sâu, pháp tuyến, cạnh, khôi
phục và matting đều phát một hàng tóm tắt mô tả bản đồ thay vì các pixel.

Hai payload được cố ý rút gọn. Vector embedding chỉ được báo cáo dưới dạng
`embedding_dim` vì một hàng 512 số thực chiếm khoảng 2 KB cho mỗi khuôn mặt;
truyền `embeddings=True` để đưa các giá trị vào. Vertex mesh hoàn toàn không
được đưa vào vì có hàng chục nghìn tọa độ cho mỗi người. Hãy đọc
`result.meshes.vertices` hoặc gọi `result.meshes.save_obj(path)` để lấy hình học.

## Vẽ và lưu

<code-tabs name="saving" />

`predict(save=True)` là luồng chú thích và ghi. Phương thức chọn hàm vẽ theo slot
được điền, nên kết quả semantic được ghi dưới dạng mask màu, kết quả độ sâu dưới
dạng trực quan hóa độ sâu, kết quả panoptic với các segment, matte dưới dạng PNG
RGBA nền trong suốt, còn detector dưới dạng box với mask bên dưới. Đường dẫn đã
ghi được gắn vào kết quả dưới dạng `result.saved_path`.

`Results.plot()` có phạm vi hẹp hơn tên gọi. Phương thức chỉ được định nghĩa cho
bản đồ pháp tuyến và bản đồ cạnh, đồng thời phát sinh `NotImplementedError` với
mọi dạng khác. Hãy dùng `save=True` cho các tác vụ khác.

`Results.save(path)` cũng có phạm vi hẹp: phương thức ghi kết quả matte thành ảnh
cắt PNG RGBA nền trong suốt và phát sinh `NotImplementedError` trong trường hợp
khác. `Results.cutout()` trả về cùng mảng RGBA mà không ghi. Cả hai cần ảnh nguồn,
lấy từ `result.path` hoặc được truyền dưới dạng `image=`.

Hai payload có trình ghi riêng: `result.restored.save(path)` cho ảnh đã khôi phục và `result.meshes.save_obj(path, index=0)` cho mesh.

Để biết tệp được đặt ở đâu và `output_path` cùng `output_file_format` hoạt động thế nào, hãy xem [Nguồn dự đoán](/docs/predict/sources).

## Artifact đã xuất trả về cùng đối tượng

<code-tabs name="exported" />

`LibreYOLO()` định tuyến theo hậu tố tệp, nên artifact đã xuất được tải qua cùng
lời gọi như checkpoint `.pt` và trả về cùng `Results`. Các tệp `.onnx`, `.engine`,
`.pte` và `.mnn` được nhận diện theo hậu tố, tương tự các thư mục OpenVINO, Paddle,
ncnn và URL mô hình Triton. Mã đọc `result.boxes.xyxy` không thay đổi khi mô hình
được thay bằng bản đã xuất. Xem [Xuất](/docs/export) để biết toàn bộ định dạng.

Dùng API riêng của runtime đồng nghĩa bạn phải tự đảm nhiệm tiền xử lý, hậu xử lý và tên lớp.

