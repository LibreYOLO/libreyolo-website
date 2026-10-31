---
title: SwinIR
families:
  - swinir
seo_title: 'SwinIR: chạy siêu phân giải ảnh 4x trong LibreYOLO'
description: >-
  Dùng SwinIR trong LibreYOLO để siêu phân giải ảnh 4x. Cài đặt, dự đoán, xác
  thực và xuất các checkpoint lightweight, medium và large.
lead: >-
  Mạng Swin Transformer để phục hồi ảnh. LibreYOLO cung cấp inference và xác
  thực cho các checkpoint siêu phân giải 4x: generator lightweight chính thức,
  real-world medium và real-world large.
keywords:
  - SwinIR
  - Swin Transformer
  - siêu phân giải ảnh
  - phục hồi ảnh
  - residual Swin Transformer block
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwinIRm-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Chia tile cho ảnh lớn
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSwinIRl-restore.pt")


        # tile chia forward pass thành các tile chồng lấp rồi hòa trộn đường
        nối;

        # tile_pad là vùng viền thêm quanh mỗi tile trước khi cắt bỏ trở lại.

        # Cả hai là đối số keyword chỉ dành cho Python, không phải cờ CLI.

        result = model("large-photo.jpg", tile=512, tile_pad=16, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwinIRm-restore.pt data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSwinIRm-restore.pt")


        # Khi bỏ qua, imgsz mặc định dùng kích thước patch nội bộ nhỏ, không
        phải

        # độ phân giải làm việc, vì vậy hãy truyền kích thước mà bản triển khai

        # thực sự đưa vào mô hình.

        model.export(format="onnx", imgsz=512)

        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwinIRm-restore.pt format=onnx imgsz=512
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreSwinIRm-restore.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.restored.array.shape)
source_hash: 87fc3d5524480eec
---

## Cài đặt

SwinIR không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Kết quả phục hồi không có box; `result.restored` là ảnh RGB uint8 dense `(H, W, 3)` trên canvas lớn gấp 4x đầu vào theo mỗi chiều. `save=True` ghi trực tiếp ảnh đó thay vì plot có chú thích. Đầu vào được padding thành bội số của 8 thay vì đổi kích thước, vì vậy dự đoán chạy ở độ phân giải riêng của ảnh; nguồn lớn hơn dung lượng bộ nhớ có thể được chia bằng `tile` và `tile_pad`, các tùy chọn này hòa trộn đường nối tile trong đầu ra. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có ba kích thước, tất cả cố định ở upscale 4x. `s` là generator lightweight chính thức, có bốn giai đoạn residual Swin Transformer block (RSTB) và upsampling pixel-shuffle-direct. `m` và `l` là các generator real-world medium và large, có sáu và chín giai đoạn RSTB cùng bộ upsample nearest-neighbor kết hợp phép tích chập được xây dựng cho suy giảm trong thực tế thay vì chỉ downscale bicubic.

## Xác thực

`val()` đo PSNR và SSIM giữa đầu ra đã phục hồi và ảnh đích sạch, cả hai được tính theo RGB trên canvas gốc mà không crop viền hoặc đổi kích thước. SSIM dùng cửa sổ Gaussian 11x11 với sigma 1,5, lấy trung bình trên ba kênh màu.

<code-tabs name="val" />

Đối số tập dữ liệu (dataset) là YAML ghép thư mục ảnh đầu vào bị suy giảm với thư mục ảnh đích sạch có độ phân giải tương ứng; xem [định dạng dataset](/docs/reference/dataset-formats) để biết chính xác các key.

## Xuất

<export-matrix />

Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp `.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng đối tượng `Results`. ExecuTorch và mọi định dạng được ma trận đánh dấu là bị chặn đều không dùng được cho họ mô hình này; ONNX, TorchScript, TensorRT, OpenVINO và TFLite thì dùng được. Trang [Xuất](/docs/export) liệt kê các đối số được mọi định dạng chấp nhận và các extra mà một số định dạng bổ sung.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />


