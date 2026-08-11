---
title: YOLOv3
families:
  - yolo3
seo_title: 'YOLOv3 trong LibreYOLO: dự đoán, xác thực, xuất'
description: >-
  Chạy YOLOv3 trong LibreYOLO: một họ mô hình bảo tàng đóng băng, chỉ dành cho
  inference, với các kích thước tiny, base và SPP. Dự đoán, xác thực và xuất
  theo giấy phép public domain.
lead: >-
  YOLOv3 là detector Darknet-53 đã bổ sung dự đoán đa tỷ lệ và các bộ phân loại
  logistic độc lập vào dòng YOLO. LibreYOLO lưu giữ mô hình như một hiện vật
  đóng băng, chỉ dành cho inference, ở các kích thước tiny, base và SPP.
keywords:
  - YOLOv3
  - Darknet
  - Darknet-53
  - phát hiện đối tượng
  - phát hiện đa tỷ lệ
  - họ mô hình bảo tàng
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO3b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO3b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Kích thước SPP
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Biến thể SPP thêm block spatial pyramid pooling trước các detection
        # head và chạy ở kích thước đầu vào nguyên bản riêng.
        model = LibreYOLO("LibreYOLO3spp.pt")
        result = model(SAMPLE_IMAGE)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO3b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO3b.pt format=onnx
        libreyolo export model=LibreYOLO3b.pt format=tensorrt half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreYOLO3b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: a4c652bb2707fc8f
---

## Cài đặt

YOLOv3 không cần extra ngoài package cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Họ mô hình này chỉ dành cho inference: `train()` phát sinh `NotImplementedError`, vì vậy trang này không có phần Huấn luyện. Dự đoán, xác thực và xuất đều được hỗ trợ. Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về cũng là đối tượng mà mọi họ mô hình đều trả về, vì vậy việc chuyển sang detector khác chỉ cần thay đổi một dòng. `conf` lọc theo ngưỡng độ tin cậy, còn `iou` lọc theo ngưỡng NMS, được áp dụng theo từng tỷ lệ trước khi box từ cả ba head được hợp nhất. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Xác thực

`val()` trả về dictionary gồm các key `metrics/` cho precision, recall, mAP 50 và mAP 50-95, được đo trên mọi dataset có định dạng bạn dùng để xác thực.

<code-tabs name="val" />

## Xuất

<export-matrix />

Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp `.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng đối tượng `Results`. Bạn cũng có thể chạy graph trong runtime thuần túy không cài LibreYOLO, nhưng khi đó phải tự viết bước tiền xử lý và hậu xử lý.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>


