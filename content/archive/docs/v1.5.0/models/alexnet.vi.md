---
title: AlexNet
families:
  - alexnet
seo_title: 'AlexNet: chạy bộ phân loại ImageNet kinh điển trong LibreYOLO'
description: >-
  Dự đoán, đánh giá và xuất AlexNet với LibreYOLO. Trọng số torchvision theo
  giấy phép BSD-3-Clause; tinh chỉnh (fine-tuning) chưa được hỗ trợ.
lead: >-
  AlexNet là mạng tích chập đã thắng ILSVRC 2012 và góp phần khởi đầu kỷ nguyên
  deep learning trong computer vision. LibreYOLO cung cấp bản sửa đổi
  single-tower về sau của kiến trúc này cho bài toán phân loại ảnh.
keywords:
  - AlexNet
  - ImageNet
  - phân loại ảnh python
  - image classification
  - mạng nơ-ron tích chập
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreAlexNetb-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreAlexNetb-cls.pt")


        # data là thư mục gốc chứa các split train/ và val/ theo dạng một thư

        # mục cho mỗi lớp đối tượng (bố cục ImageFolder), không phải YAML
        dataset

        metrics = model.val(data="imagenet-1k/")


        print(metrics["metrics/accuracy_top1"])

        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreAlexNetb-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreAlexNetb-cls.pt format=onnx
        libreyolo export model=LibreAlexNetb-cls.pt format=tensorrt half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo phần đuôi tệp, nên một artifact đã xuất được
        # tải như mọi checkpoint khác và trả về cùng đối tượng Results
        model = LibreYOLO("LibreAlexNetb-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 68c09f080c74bb87
---

## Cài đặt

AlexNet không cần extra tùy chọn nào. Mọi thứ nó import đều nằm trong bản cài
đặt cơ bản.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải về từ Hugging Face ở lần dùng đầu tiên và được cache lại ở máy.

<code-tabs name="predict" />

Một bộ phân loại trả về `result.probs` thay vì `result.boxes`: `top1`
và `top5` cho ra chỉ số lớp đối tượng, còn `top1conf` và `top5conf` cho ra độ
tin cậy tương ứng. Xem [dự đoán](/docs/predict) để biết về nguồn đầu vào, luồng
(stream) và cách xử lý kết quả.

## Các biến thể

Một kích thước duy nhất. Đồ thị được cung cấp là bản sửa đổi single-tower về
sau do torchvision phát hành, với 64 filter ở lớp đầu tiên và không có local
response normalization, chứ không phải kiến trúc hai GPU nguyên bản năm 2012.
LibreYOLO cung cấp họ mô hình này ở chế độ chỉ suy luận (inference): dự đoán,
đánh giá top-1/top-5 kiểu ImageNet và xuất mô hình đều được hỗ trợ, còn tinh
chỉnh thì chưa có phần hiện thực.

## Đánh giá

`val()` chạy trên một split kiểu ImageFolder (một thư mục có các thư mục con
`train/` và `val/`, mỗi lớp đối tượng một thư mục) và trả về độ chính xác top-1
và top-5.

<code-tabs name="val" />

## Xuất mô hình

<export-matrix />

Một artifact đã xuất được tải ngược lại qua `LibreYOLO()` dựa trên phần đuôi tệp
của nó, nên một tệp `.onnx` hay `.engine` hoạt động như một checkpoint và trả về
cùng `Results`. [Xuất mô hình](/docs/export) liệt kê những tham số mà mọi định
dạng đều nhận, cùng các tham số bổ sung mà một vài định dạng thêm vào.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã công bố cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>
