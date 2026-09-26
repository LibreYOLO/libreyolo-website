---
title: YOLOv1
families:
  - yolo1
seo_title: 'YOLOv1 trong LibreYOLO: dự đoán, xác thực, xuất'
description: >-
  Chạy detector YOLOv1 nguyên bản trong LibreYOLO: một họ mô hình bảo tàng đóng
  băng, chỉ dành cho inference. Dự đoán, xác thực và xuất theo giấy phép public
  domain.
lead: >-
  YOLOv1 là detector nguyên bản năm 2016 đã đặt tên cho họ YOLO: một mạng tích
  chập với fully connected head dự đoán mọi box và điểm lớp đối tượng trong một
  lượt duy nhất, không có anchor box. LibreYOLO lưu giữ mô hình như một hiện vật
  đóng băng, chỉ dành cho inference.
keywords:
  - YOLOv1
  - YOLO v1
  - Darknet
  - phát hiện đối tượng
  - Pascal VOC
  - họ mô hình bảo tàng
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO1b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO1b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO1b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO1b.pt format=onnx
        libreyolo export model=LibreYOLO1b.pt format=tensorrt half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreYOLO1b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: a786372dba86f2f8
---

## Cài đặt

YOLOv1 không cần extra ngoài package cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Họ mô hình này chỉ dành cho inference: `train()` phát sinh `NotImplementedError`, vì vậy trang này không có phần Huấn luyện. Dự đoán, xác thực và xuất đều được hỗ trợ. Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về cũng là đối tượng mà mọi họ mô hình đều trả về, vì vậy việc chuyển sang detector khác chỉ cần thay đổi một dòng. Họ mô hình này có hai điểm riêng. Checkpoint đã phát hành được huấn luyện trên Pascal VOC (2007+2012), không phải COCO, vì vậy `box.cls` lập chỉ mục 20 danh mục VOC (aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor) thay vì 80 danh mục COCO. Fully connected detection head cũng chỉ nhận một ảnh mỗi lần, vì vậy danh sách nguồn được lặp thay vì chạy như batch thực. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Xác thực

`val()` trả về dictionary gồm các key `metrics/` cho precision, recall, mAP 50 và mAP 50-95, được đo trên dataset trong cùng không gian nhãn kiểu VOC mà checkpoint đã được huấn luyện.

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


