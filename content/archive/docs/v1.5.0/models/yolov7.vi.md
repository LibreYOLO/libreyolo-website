---
title: YOLOv7
families:
  - yolo7
seo_title: 'YOLOv7 trong LibreYOLO: dự đoán, huấn luyện và xuất theo MIT'
description: >-
  Chạy YOLOv7 trong LibreYOLO để phát hiện đối tượng: cài đặt, dự đoán, huấn
  luyện, đánh giá và xuất mã cùng trọng số theo giấy phép MIT.
lead: >-
  YOLOv7 là detector một giai đoạn dựa trên anchor, có head thêm các độ lệch tri
  thức ngầm đã học trước phép tích chập cuối. LibreYOLO hỗ trợ kích thước duy
  nhất đã công bố cho tác vụ phát hiện.
keywords:
  - YOLOv7
  - phát hiện đối tượng
  - phát hiện dựa trên anchor
  - tri thức ngầm
  - ImplicitA
  - phát hiện đối tượng thời gian thực
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO7b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO7b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO7b.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO7b.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
    - label: Warm start từ mô hình mới
      language: python
      code: |
        from libreyolo import LibreYOLO7

        # pretrained=True luôn tải checkpoint LibreYOLO7b.pt đã công bố, bất kể
        # instance này được khởi tạo với gì. Việc khởi tạo lớp trực tiếp thay vì
        # qua LibreYOLO() sẽ bắt đầu mà chưa tải bất kỳ trọng số nào.
        model = LibreYOLO7(None, size="b")
        model.train(data="my-dataset.yaml", epochs=300, pretrained=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO7b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLO7b.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLO7b.pt format=tensorrt imgsz=640
        half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, nên artifact đã xuất được tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreYOLO7b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 361e81de5614a571
---

## Cài đặt

YOLOv7 không cần extra ngoài gói cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải từ Hugging Face ở lần dùng đầu tiên và lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về là loại mà mọi họ đều trả về, nên việc đổi sang
detector khác chỉ cần sửa một dòng. `conf` đặt ngưỡng độ tin cậy và `iou` đặt
ngưỡng NMS được áp dụng sau khi giải mã head dựa trên anchor. Xem
[dự đoán](/docs/predict) để biết về nguồn, streaming và xử lý kết quả.

## Các biến thể

LibreYOLO phân phối một kích thước là `b`. Thượng nguồn công bố một mô hình
YOLOv7 duy nhất, vì vậy không có nhiều kích thước để lựa chọn.

## Huấn luyện

<code-tabs name="train" />

`pretrained` được đọc, khác với đối số cùng tên không làm gì ở một số họ khác:
truyền `True` để warm-start từ checkpoint `LibreYOLO7b.pt` đã công bố (được tự
động tải), hoặc truyền đường dẫn hay tên cho trường hợp khác. Checkpoint đã công
bố dùng COCO 80 lớp, nên khi yêu cầu nó trên mô hình đã được xây dựng lại cho số
lớp khác, hệ thống trước tiên xây dựng lại về 80 lớp, tải checkpoint, rồi chuyển
mọi tensor có shape khớp sang số lượng head đích sau khi đọc số lớp của tập dữ
liệu. Không thể kết hợp `resume=True` với `pretrained`. Khi giữ mặc định `None`,
quá trình huấn luyện tiếp tục từ trọng số dùng để khởi tạo mô hình hoặc từ khởi
tạo ngẫu nhiên nếu chưa tải gì.

Nếu giữ nguyên các giá trị khác, trình huấn luyện chạy 300 epoch ở `lr0=0.01`
với SGD momentum 0.937, warmup 3 epoch, cùng phép gán SimOTA và giai đoạn 15
epoch cuối không tăng cường dữ liệu như YOLOX, được điều chỉnh cho head dựa trên
anchor. Có một điểm khác biệt: YOLOX thêm bước tinh chỉnh hồi quy box L1 trong
các epoch cuối đó, còn v7 bỏ qua vì loss SimOTA của v7 không có nhánh L1 độ lệch
thô để tinh chỉnh.

Xem [huấn luyện](/docs/train) để biết về tập dữ liệu, tăng cường dữ liệu, multi-GPU và logger.

## Đánh giá

`val()` trả về từ điển các khóa `metrics/` bao gồm precision, recall, mAP 50 và
mAP 50-95, được đo trên bất kỳ tập dữ liệu nào theo định dạng bạn đã huấn luyện.

<code-tabs name="val" />

## Xuất

<export-matrix />

Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp
`.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng `Results`. Cũng
hỗ trợ chạy đồ thị trong runtime độc lập không cài LibreYOLO, nhưng khi đó bạn
phải tự viết bước tiền xử lý và hậu xử lý.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã công bố cho họ này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />

