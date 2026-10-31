---
title: Real-ESRGAN
families:
  - realesrgan
seo_title: 'Real-ESRGAN: siêu phân giải ảnh trong LibreYOLO'
description: >-
  Dùng Real-ESRGAN trong LibreYOLO để siêu phân giải ảnh thực tế ở 4x, 2x và một
  cấp 4x nhanh. Cài đặt, dự đoán, xác thực và xuất.
lead: >-
  Bộ upscale siêu phân giải blind thực tế được huấn luyện trên các dạng suy giảm
  tổng hợp thay vì chỉ downscale bicubic. LibreYOLO cung cấp inference và xác
  thực cho các checkpoint 4x, 2x và 4x nhanh.
keywords:
  - Real-ESRGAN
  - RRDBNet
  - SRVGGNetCompact
  - siêu phân giải ảnh
  - phục hồi ảnh
  - blind super-resolution
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRealESRGANx4-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Chia tile cho ảnh lớn
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRealESRGANx4-restore.pt")


        # tile chia forward pass thành các tile chồng lấp rồi hòa trộn đường
        nối;

        # tile_pad là vùng viền thêm quanh mỗi tile trước khi cắt bỏ trở lại.

        # Cả hai là đối số keyword chỉ dành cho Python, không phải cờ CLI.

        result = model("large-photo.jpg", tile=512, tile_pad=10, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: >
        libreyolo val model=LibreRealESRGANx4-restore.pt
        data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRealESRGANx4-restore.pt")


        # Khi bỏ qua, imgsz mặc định dùng kích thước patch nội bộ nhỏ, không
        phải

        # độ phân giải làm việc, vì vậy hãy truyền kích thước mà bản triển khai

        # thực sự đưa vào mô hình.

        model.export(format="onnx", imgsz=512)

        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRealESRGANx4-restore.pt format=onnx
        imgsz=512
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreRealESRGANx4-restore.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.restored.array.shape)
source_hash: f0efb4f65d38e22d
---

## Cài đặt

Real-ESRGAN không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Kết quả phục hồi không có box; `result.restored` là ảnh RGB uint8 dense `(H, W, 3)` trên canvas lớn gấp `Results.restore_scale` lần đầu vào theo mỗi chiều. `save=True` ghi trực tiếp ảnh đó thay vì plot có chú thích. Đầu vào được chuyển sang RGB và mọi kênh alpha bị bỏ. Nguồn lớn hơn dung lượng bộ nhớ có thể được chia bằng `tile` và `tile_pad`, các tùy chọn này hòa trộn đường nối tile trong đầu ra. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có ba checkpoint được đặt tên theo hệ số upscale. `x4` là RRDBNet (`RealESRGAN_x4plus`) với 23 residual-in-residual dense block, mặc định chất lượng ở 4x. `x2` là cùng kiến trúc RRDBNet ở 2x. `x4t` là SRVGGNetCompact (`realesr-general-x4v3`), generator nhỏ và nhanh hơn được xây dựng cho video cùng trường hợp cần độ trễ thấp hơn ở 4x. Mô hình đa dụng upstream cũng cung cấp mạng denoise-strength theo cặp được hòa trộn khi inference; núm điều chỉnh cường độ đó không thuộc bản port này, bản port chạy generator `x4t` cơ sở.

## Xác thực

`val()` đo PSNR và SSIM giữa đầu ra đã phục hồi và ảnh đích sạch, cả hai được tính theo RGB trên canvas gốc mà không crop viền hoặc đổi kích thước. SSIM dùng cửa sổ Gaussian 11x11 với sigma 1,5, lấy trung bình trên ba kênh màu.

<code-tabs name="val" />

Đối số tập dữ liệu (dataset) là YAML ghép thư mục ảnh đầu vào bị suy giảm với thư mục ảnh đích sạch có độ phân giải tương ứng; xem [định dạng dataset](/docs/reference/dataset-formats) để biết chính xác các key.

## Xuất

<export-matrix />

Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp `.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng đối tượng `Results`. Trang [Xuất](/docs/export) liệt kê các đối số được mọi định dạng chấp nhận và các extra mà một số định dạng bổ sung.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />


