---
title: DeiT
families:
  - deit
seo_title: 'Bộ phân loại ảnh DeiT: dự đoán, xác thực, xuất'
description: >-
  Chạy các bộ phân loại ảnh DeiT trong LibreYOLO: một họ mô hình bảo tàng đóng
  băng, chỉ dành cho inference, với kích thước tiny, small và base theo
  Apache-2.0.
lead: >-
  DeiT (Data-efficient image Transformer) là bộ phân loại Vision Transformer
  thuần túy được huấn luyện chỉ trên ImageNet-1k, không dùng thêm dữ liệu huấn
  luyện sẵn. LibreYOLO lưu giữ các kích thước patch-16 tiny, small và base như
  một hiện vật đóng băng, chỉ dành cho inference.
keywords:
  - DeiT
  - Vision Transformer
  - ViT
  - phân loại ảnh
  - ImageNet
  - huấn luyện tiết kiệm dữ liệu
  - họ mô hình bảo tàng
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeiTb-cls.pt")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeiTb-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeiTb-cls.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDeiTb-cls.pt format=onnx
        libreyolo export model=LibreDeiTb-cls.pt format=tensorrt half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreDeiTb-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 9c67c8554b2af5c6
---

## Cài đặt

DeiT không cần extra ngoài package cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Họ mô hình này chỉ dành cho inference: `train()` phát sinh `NotImplementedError`, vì vậy trang này không có phần Huấn luyện. Dự đoán, xác thực và xuất đều được hỗ trợ. Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ. Hậu tố `-cls` trong tên tệp là bắt buộc và chọn tác vụ phân loại.

<code-tabs name="predict" />

Đối tượng `Results` trả về chứa tensor `probs` thay cho `boxes`; `top1` và `top5` lập chỉ mục 1.000 lớp đối tượng ImageNet-1k, còn `top1conf` là điểm softmax cho dự đoán hàng đầu. Mỗi kích thước có độ phân giải đầu vào cố định từ positional embedding: bước tiền xử lý đổi kích thước và cắt giữa theo độ phân giải này, việc truyền `imgsz` khác sẽ phát sinh lỗi thay vì âm thầm lấy mẫu lại. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Xác thực

`val()` trả về dictionary có độ chính xác top-1 và top-5, được đo trên dataset bố trí theo cấu trúc thư mục thông thường `train/<class>/` và `val/<class>/`.

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

## Trích dẫn

<citation-block />
