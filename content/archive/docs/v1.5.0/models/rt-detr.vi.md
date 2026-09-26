---
title: RT-DETR
families:
  - rtdetr
seo_title: 'RT-DETR, RT-DETRv2 và RT-DETRv4 trong LibreYOLO'
description: >-
  Dùng RT-DETR, RT-DETRv2 và RT-DETRv4 trong LibreYOLO để phát hiện đối tượng,
  cùng box định hướng trên RT-DETRv2. Cài đặt, dự đoán, huấn luyện, đánh giá và
  xuất với trọng số Apache-2.0.
lead: >-
  Một detection transformer được xây dựng cho suy luận thời gian thực: mô hình
  giải mã tập query cố định thay vì lưới dày đặc, nên không chạy NMS. LibreYOLO
  cung cấp ba phiên bản được phân biệt bằng checkpoint bạn tải, và phiên bản 2
  còn hỗ trợ box định hướng.
keywords:
  - RT-DETR
  - RT-DETRv2
  - RT-DETRv4
  - detection transformer thời gian thực
  - DETR
  - phát hiện đối tượng
  - phát hiện bounding box định hướng
  - OBB
  - DOTA
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTDETRr18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRTDETRr18.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Video
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Phiên bản nằm trong tên tệp và factory định tuyến theo checkpoint,
        # nên cả ba được tải theo cùng một cách.
        model = LibreYOLO("LibreRTDETRv4s.pt")

        # Mọi nguồn thư viện chấp nhận: tệp, thư mục, URL, chỉ số webcam,
        # luồng RTSP hoặc danh sách .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
    - label: Box định hướng
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Chỉ phiên bản 2. Hậu tố -obb chọn tác vụ và checkpoint được nhận diện

        # là định hướng từ các tensor riêng, nên không cần đối số task. Các
        trọng

        # số này dùng DOTA v1.0, gồm 15 lớp ảnh trên không ở 1024 px.

        model = LibreYOLO("LibreRTDETRv2n-obb.pt")

        result = model("aerial.png", save=True)


        obb = result.obb

        print(obb.xywhr)     # (N, 5): cx, cy, w, h, radian

        print(obb.xyxyxyxy)  # cùng các hàng dưới dạng bốn điểm góc

        print(result.boxes.xyxy)  # các box thẳng trục bao quanh
    - label: 'Box định hướng, CLI'
      language: bash
      code: >
        libreyolo predict model=LibreRTDETRv2n-obb.pt source=aerial.png
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")

        # coco128.yaml tải mẫu 128 ảnh ở lần dùng đầu tiên. Hãy trỏ `data`
        # đến YAML tập dữ liệu của bạn cho lượt chạy thực tế.
        model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 batch=4 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        # Cần extra lora: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")

        # val() trả về dict thông thường, không phải đối tượng
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTDETRr18.pt data=coco128.yaml
    - label: Trên COCO
      language: bash
      code: |
        # coco-val-only.yaml lấy 5.000 ảnh val2017 và bỏ qua tập huấn luyện.
        # Tệp chứa script tải xuống nhúng sẵn, nên cần quyền rõ ràng trừ khi
        # tập dữ liệu đã có cục bộ.
        libreyolo val model=LibreRTDETRr18.pt data=coco-val-only.yaml \
          allow_download_scripts=True
    - label: Oriented boxes
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Đánh giá định hướng ghép bằng IoU xoay, nên dự đoán đúng vị trí nhưng
        # sai góc vẫn được tính là trượt.
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        metrics = model.val(data="my-obb-dataset.yaml")

        print(metrics["metrics/mAP50-95(OBB)"])
        print(metrics["metrics/mAP50(OBB)"])
  export:
    - label: Python
      language: python
      code: |
        # Cần extra onnx: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRTDETRr18.pt format=onnx
    - label: Oriented boxes
      language: bash
      code: >
        # ONNX và TorchScript là các đích đã được đánh giá cho tác vụ định
        hướng,

        # ở FP32, batch 1, trên canvas cố định 1024 x 1024.

        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024

        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript
        imgsz=1024
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, nên artifact đã xuất được tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreRTDETRr18.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 8022a5a591922a90
---

## Cài đặt

RT-DETR không cần extra tùy chọn. Mọi thành phần được import đều có trong bản
cài đặt cơ sở, còn extra `rtdetr` là tên ổn định không bổ sung gì.

```bash
pip install libreyolo
```

Tinh chỉnh bằng adapter với `lora=True` là ngoại lệ và cần extra `lora`.

```bash
pip install "libreyolo[lora]"
```

## Dự đoán

Trọng số được tải từ Hugging Face ở lần dùng đầu tiên và lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về là loại mà mọi họ đều trả về, nên việc đổi sang
detector khác chỉ cần sửa một dòng. `conf` và `max_det` lọc phép giải mã top-k
trên các query và lớp; không có bước NMS cần điều chỉnh, còn `iou` được chấp nhận
nhưng không dùng. Checkpoint định hướng điền trực tiếp `result.obb` và cũng điền
`result.boxes` bằng các hình chữ nhật thẳng trục bao quanh. Xem
[dự đoán](/docs/predict) để biết về nguồn, streaming và xử lý kết quả.

## Các biến thể

Có ba phiên bản với hai tác vụ và mã kích thước không nằm trong một chuỗi duy
nhất. Phiên bản 1 đặt tên kích thước theo backbone ResNet hoặc HGNetv2. Phiên bản
2 chỉ dùng lại các tên ResNet: phiên bản 1 đã phân phối hai kích thước HGNetv2,
và kết quả phiên bản 2 trên các kích thước đó đủ gần để LibreYOLO không công bố
trọng số trùng lặp. Phiên bản 4 dùng chuỗi chữ cái đơn giản, trùng với tên HGNetv2
của phiên bản 1, nên chỉ mã kích thước không thể xác định mô hình. Phiên bản được
ghi trong tên tệp checkpoint.

<benchmark-table task="detect" />

<va-embed />

Phiên bản 2 giữ kiến trúc và bố cục state dict của phiên bản 1, đồng thời thay
đổi cách deformable attention lấy mẫu. Vì vậy, hai phiên bản được phân biệt bằng
metadata trong checkpoint thay vì shape. Phiên bản 4 thuộc một dòng khác: dùng
lại kiến trúc và trình huấn luyện của D-FINE, còn trọng số đến từ việc chưng cất
mô hình giáo viên vision foundation DINOv3 vào mô hình học sinh HGNetv2. Trong
LibreYOLO, `LibreRTDETRv4` là lớp con của `LibreDFINE` với mask head bị tắt cố
định, nên chỉ hỗ trợ phát hiện.

### Box định hướng trên phiên bản 2

Phiên bản 2 là phiên bản duy nhất có tác vụ thứ hai. Các tác vụ được hỗ trợ là
`detect` và `obb`, hai tác vụ không dùng chung đồ thị hay chuỗi kích thước. Phát
hiện dùng các kích thước ResNet ở 640 px; phát hiện định hướng dùng chuỗi HGNetv2
n, s, m, l và x ở 1024 px, còn kích thước đầu vào được phân giải theo tác vụ thay
vì theo họ. Checkpoint được nhận diện là định hướng từ các tensor riêng, qua head
box năm tọa độ và tham số lấy mẫu của phiên bản 2, nên trọng số `-obb` tải vào đồ
thị định hướng mà không cần đối số `task`; trạng thái không khớp giữa hai bên sẽ
là lỗi cứng thay vì được diễn giải lại âm thầm.

Các tệp đã công bố trải từ `LibreRTDETRv2n-obb.pt` đến
`LibreRTDETRv2x-obb.pt`. Đây là checkpoint DOTA v1.0 đơn thang chính thức đã
chuyển sang định dạng LibreYOLO, gồm 15 lớp ảnh trên không từ máy bay và tàu đến
cảng cùng trực thăng; tên lớp được đóng dấu vào checkpoint. Khác với phía phát
hiện, tác vụ định hướng chỉ dành cho suy luận: dự đoán, đánh giá và xuất hoạt
động, còn `train()` trên mô hình định hướng sẽ phát sinh lỗi. Theo dõi và tăng
cường dữ liệu khi kiểm thử cũng không hỗ trợ box định hướng.
[Phát hiện định hướng](/docs/tasks/oriented-detection) trình bày tác vụ, định dạng
nhãn và các chỉ số.

## Huấn luyện

Huấn luyện bắt đầu từ checkpoint đã công bố. `pretrained` được chấp nhận rồi bỏ
qua trên cả ba phiên bản, nên `pretrained=False` không tạo mô hình khởi tạo ngẫu
nhiên. Mọi nội dung trong phần này đều nói về phát hiện: tác vụ định hướng của
phiên bản 2 chỉ dành cho suy luận và không có luồng chuyển trọng số phát hiện sang
nó vì hai tác vụ dùng backbone khác nhau.

<code-tabs name="train" />

Learning rate là đối số cần đặt chính xác, và mỗi phiên bản có giá trị mặc định
riêng thay vì giá trị chung toàn thư viện. Chữ ký `train()` trong Python đọc giá
trị từ cấu hình huấn luyện của phiên bản đó, còn CLI phân giải cùng giá trị khi
không truyền `lr0`. Phiên bản 1 và 2 còn nhận `lr_backbone`, mặc định bằng một
phần hai mươi `lr0` theo công thức gốc; phiên bản 4 chạy qua trình huấn luyện
D-FINE, thay vào đó scale nhóm tham số backbone bằng `backbone_lr_mult`.

Giữ `imgsz` ở kích thước gốc của checkpoint trừ khi có lý do thay đổi. Đánh giá
và dự đoán ở kích thước khác vẫn hoạt động, nhưng còn một hạn chế: kích thước hình
chữ nhật có số token khớp kích thước gốc vẫn dùng lại embedding được xây dựng cho
sai tỉ lệ khung hình.

Xem [huấn luyện](/docs/train) để biết về tập dữ liệu, tăng cường dữ liệu, multi-GPU và logger.

## Đánh giá

`val()` trả về từ điển các khóa `metrics/` bao gồm precision, recall, mAP 50 và
mAP 50-95, được đo trên bất kỳ tập dữ liệu nào theo định dạng bạn đã huấn luyện.

<code-tabs name="val" />

Các hàng trong bảng benchmark trên đến từ hệ thống benchmark LibreYOLO; ghi chú
dưới bảng nêu tập dữ liệu tạo ra chúng và liên kết đến bản ghi lượt chạy.

Đánh giá định hướng chạy qua cùng lời gọi và báo cáo cùng các khóa, cộng thêm bốn
khóa lặp lại với hậu tố `(OBB)`. Việc ghép dùng IoU xoay thay vì IoU của các hình
chữ nhật bao quanh, nên lỗi góc được tính là trượt. `augment=True` bị từ chối trên
tác vụ này.

## Xuất

<export-matrix />

Ma trận bao quát cả dòng mô hình trên một trang: khi ba phiên bản không thống
nhất về một định dạng, ô hiển thị mức hỗ trợ yếu nhất trong ba phiên bản để không
phóng đại khả năng của bất kỳ phiên bản nào bạn tải. Hàng định hướng chỉ thuộc
phiên bản 2. ONNX và TorchScript đã được đánh giá ở FP32, batch 1 và canvas cố
định 1024 x 1024; OpenVINO, TensorRT và ExecuTorch chuyển đổi rồi tải lại được
nhưng chưa đạt mức tương đương đầu ra thô trên toàn bộ tập query, nên các box đầu
khớp đến một phần pixel còn phần đuôi bị lệch.

Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp
`.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng `Results`.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã công bố cho họ này.

<checkpoint-table />

Tên tệp chứa phiên bản, tiếp đến kích thước rồi tác vụ. Trọng số phát hiện là
`LibreRTDETR<size>.pt`, `LibreRTDETRv2<size>.pt` và `LibreRTDETRv4<size>.pt`,
tất cả ở 640 px. Trọng số định hướng chỉ có cho phiên bản 2 và thêm hậu tố tác vụ,
từ `LibreRTDETRv2n-obb.pt` đến `LibreRTDETRv2x-obb.pt`, tất cả ở 1024 px và được
huấn luyện trên DOTA v1.0 thay vì COCO.

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />

Block trên là nội dung tác giả công bố cho tác vụ phát hiện của phiên bản 1 và 2.
Trọng số định hướng phiên bản 2 có nguồn thượng nguồn thứ ba là repo RiO-DETR theo
Apache-2.0 tại
[github.com/RicePasteM/RiO-DETR](https://github.com/RicePasteM/RiO-DETR), nơi
cung cấp các checkpoint DOTA; hãy trích dẫn dự án đó nếu bạn dùng một checkpoint.
Phiên bản 4 là bài báo riêng của một nhóm khác và có block trích dẫn riêng tại
[github.com/RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4#4-citation);
hãy trích dẫn bài đó nếu bạn dùng checkpoint phiên bản 4.

