---
title: YOLOX
families:
  - yolox
seo_title: 'YOLOX: dự đoán, huấn luyện và xuất theo Apache-2.0'
description: >-
  Dùng YOLOX trong LibreYOLO để phát hiện đối tượng: cài đặt, dự đoán, huấn
  luyện, đánh giá và xuất theo Apache-2.0.
lead: >-
  YOLOX là detector một giai đoạn không anchor, có head phân loại-hồi quy tách
  rời và được huấn luyện bằng phép gán nhãn SimOTA. LibreYOLO hỗ trợ mô hình cho
  tác vụ phát hiện.
keywords:
  - YOLOX
  - phát hiện đối tượng
  - phát hiện không anchor
  - head tách rời
  - SimOTA
  - phát hiện đối tượng thời gian thực
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLOXs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLOXs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLOXs.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLOXs.pt data=my-dataset.yaml
    - label: Trên COCO
      language: bash
      code: >
        # YAML COCO đi kèm chứa script tải xuống nhúng sẵn, nên cần quyền rõ
        ràng

        # trừ khi tập dữ liệu đã có cục bộ.

        libreyolo val model=LibreYOLOXn.pt data=coco.yaml imgsz=416 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLOXs.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLOXs.pt format=tensorrt imgsz=640
        half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, nên artifact đã xuất được tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreYOLOXs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: f5ab735a29f85a95
---

## Cài đặt

YOLOX không cần extra ngoài gói cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải từ Hugging Face ở lần dùng đầu tiên và lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về là loại mà mọi họ đều trả về, nên việc đổi sang
detector khác chỉ cần sửa một dòng. `conf` đặt ngưỡng độ tin cậy và `iou` đặt
ngưỡng NMS được áp dụng trên ba thang dự đoán tách rời. Xem
[dự đoán](/docs/predict) để biết về nguồn, streaming và xử lý kết quả.

## Các biến thể

Sáu kích thước dùng chung backbone CSP và neck PAFPN. Hai kích thước nhỏ nhất
`n` và `t` chạy ở độ phân giải đầu vào cố định nhỏ hơn bốn kích thước còn lại;
bảng benchmark bên dưới chứa số liệu chính xác cho từng kích thước.

<benchmark-table task="detect" />

<va-embed />

## Huấn luyện

<code-tabs name="train" />

Nếu giữ nguyên mặc định, trình huấn luyện chạy 300 epoch ở `lr0=0.01` với SGD
momentum 0.9, warmup 5 epoch và tắt tăng cường dữ liệu mosaic cùng mixup trong
15 epoch cuối. `train()` cũng chấp nhận đối số `pretrained`, nhưng giá trị không
bao giờ được đọc bên trong phương thức: quá trình huấn luyện luôn tiếp tục từ
trọng số dùng để khởi tạo mô hình, nên `pretrained=False` không khởi tạo lại mạng.

`imgsz` mặc định là giá trị cố định trong cấu hình huấn luyện cơ sở, không phải
độ phân giải gốc của checkpoint đã tải. Điều này đặc biệt ảnh hưởng đến các
checkpoint `n` và `t`: tiếp tục huấn luyện một trong hai mà không đặt rõ `imgsz`
sẽ chuyển lên giá trị mặc định lớn hơn thay vì kích thước nhỏ hơn khi công bố.

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
phải tự viết bước tiền xử lý và hậu xử lý. Bản xuất CoreML có thể nhúng NMS vào
đồ thị bằng `nms=True`; YOLOX và YOLOv9 là hai họ duy nhất cờ này hiện chấp nhận.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã công bố cho họ này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />

