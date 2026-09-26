---
title: SSD
families:
  - ssd
seo_title: 'SSD (SSD300): phát hiện đối tượng trong LibreYOLO'
description: >-
  Chạy SSD300 trong LibreYOLO: detector VGG16 single-shot để dự đoán, xác thực
  và xuất ONNX theo BSD-3-Clause. Không có tuyến huấn luyện.
lead: >-
  SSD (Single Shot MultiBox Detector) dự đoán mọi box và điểm lớp đối tượng từ
  lưới default box dense trong một forward pass, không có giai đoạn
  region-proposal riêng. LibreYOLO cung cấp checkpoint SSD300 dựa trên VGG16 như
  một detector chỉ dành cho inference.
keywords:
  - SSD
  - SSD300
  - Single Shot MultiBox Detector
  - phát hiện đối tượng
  - VGG16
  - detector dựa trên anchor
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSSD300.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSSD300.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSSD300.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSSD300.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSSD300.pt")


        # Chủ ý bỏ imgsz ở đây: SSD300 trace trên canvas nguyên bản của
        checkpoint,

        # mọi giá trị khác phát sinh lỗi trước khi bắt đầu xuất.

        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSSD300.pt format=onnx
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreSSD300.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 3b3f9ea72291c4fa
---

## Cài đặt

SSD không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về cũng là đối tượng mà mọi họ mô hình đều trả về, vì vậy việc chuyển sang detector khác chỉ cần thay đổi một dòng. SSD giải mã lưới default-box với điểm theo lớp đối tượng rồi chạy non-maximum suppression, vì vậy `conf`, `iou` và `max_det` đều có tác dụng thực sự ở đây, khác với detector dựa trên query trong thư viện. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

SSD cung cấp một checkpoint: mạng SSD300 dựa trên VGG16 ở canvas nguyên bản cố định. Họ mô hình này không có lựa chọn kích thước hoặc scale; dự đoán, xác thực và xuất đều dùng graph duy nhất đó.

Tệp trọng số là `LibreSSD300.pt`, tiền tố họ mô hình theo sau bởi key kích thước duy nhất `"300"`. Lớp phía sau là `LibreSSD`, vì vậy cách khởi tạo trực tiếp là `LibreSSD(size="300")` thay vì lớp được đặt tên theo tệp.

## Xác thực

`val()` trả về dictionary gồm các key `metrics/` cho precision, recall, mAP 50 và mAP 50-95, được đo trên mọi dataset có định dạng giống định dạng bạn đã dùng để huấn luyện.

<code-tabs name="val" />

## Xuất

<export-matrix />

SSD chỉ xuất sang ONNX; mọi định dạng khác hiện bị chặn cho họ mô hình này. Thao tác xuất luôn dùng canvas nguyên bản của checkpoint, còn graph cung cấp packed head thô của SSD thay vì đầu ra non-maximum-suppression hợp nhất, vì vậy `nms=True` không được chấp nhận khi xuất. Các backend riêng của LibreYOLO chạy bước giải mã và loại bỏ sau khi tải lại graph.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box>

Mã SSD300 của LibreYOLO không được port từ bản phát hành Caffe riêng của tác giả bài báo; mã bắt nguồn từ cách triển khai SSD300 BSD-3-Clause của torchvision, cũng là repo được liên kết bên trên làm nguồn upstream. Trọng số VGG16 của backbone có nguồn xa hơn từ fully convolutional reduced VGGNet của Oxford, được Karen Simonyan và Andrew Zisserman phát hành theo CC BY 4.0.

</provenance-box>

## Trích dẫn

<citation-block />


