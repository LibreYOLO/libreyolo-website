---
title: NAFNet
families:
  - nafnet
seo_title: 'NAFNet: khử nhiễu, huấn luyện và xuất theo MIT'
description: >-
  Dùng NAFNet trong LibreYOLO để khử nhiễu và khôi phục ảnh. Cài đặt, dự đoán,
  huấn luyện, đánh giá và xuất checkpoint SIDD theo giấy phép MIT.
lead: >-
  NAFNet là mạng tích chập để khôi phục ảnh, loại bỏ các hàm activation phi
  tuyến khỏi block UNet điển hình và thay bằng phép nhân theo phần tử. LibreYOLO
  hỗ trợ mô hình cho một tác vụ là khôi phục, với checkpoint khử nhiễu ảnh thực
  đã công bố và được huấn luyện trên SIDD.
keywords:
  - NAFNet
  - khôi phục ảnh
  - khử nhiễu ảnh
  - khử mờ ảnh
  - mạng không activation phi tuyến
  - SIDD
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model("noisy.jpg", save=True)

        restored = result.restored
        print(restored.array.shape)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreNAFNetl-restore-sidd.pt source=noisy.jpg
        save=True
    - label: Lưu ảnh đã khôi phục
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model.predict("noisy.jpg")

        result.restored.save("denoised.png")
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 imgsz=256 batch=16 lr0=1e-3
    - label: Nguồn gốc checkpoint
      language: python
      code: |
        from libreyolo import LibreYOLO

        # degradation và dataset được ghi vào checkpoint đã lưu; chúng
        # không thay đổi nội dung được huấn luyện.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
    - label: Multi-GPU
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val() trả về dict thông thường, không phải đối tượng
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.export(format="onnx", imgsz=256)
        model.export(format="tensorrt", imgsz=256, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=onnx
        imgsz=256

        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=tensorrt
        imgsz=256 half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Factory định tuyến theo hậu tố tệp, nên artifact đã xuất được tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")
        result = model("noisy.jpg")

        result.restored.save("denoised.png")
source_hash: 9bae9f82bee741bf
---

## Cài đặt

NAFNet không cần extra tùy chọn. Mọi thành phần được import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải từ Hugging Face ở lần dùng đầu tiên và lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về chứa một trường cho họ này là `restored`, một ảnh RGB
uint8 HWC dày đặc trên canvas gốc; không có box để duyệt. `save=True` ghi trực
tiếp ảnh đã khôi phục ra đĩa thay vì vẽ chú thích lên đầu vào. `conf`, `iou` và
`max_det` được chấp nhận để đồng nhất chữ ký với mọi họ khác nhưng không có tác
động vì khôi phục không tạo phát hiện để lọc. Xem [dự đoán](/docs/predict) để
biết về nguồn, streaming và xử lý kết quả.

## Các biến thể

Hai độ rộng dùng chung kiến trúc này: `s` (độ rộng 32) và `l` (độ rộng 64), cả
hai được xây dựng quanh patch huấn luyện 256 px. Dự đoán và đánh giá chạy ở độ
phân giải ảnh gốc bất kể kích thước, chỉ thêm padding đến hệ số downsample của
mạng. Hiện chỉ độ rộng `l` được công bố dưới dạng checkpoint khử nhiễu ảnh thực
được huấn luyện trên SIDD.

## Huấn luyện

NAFNet tinh chỉnh trên các cặp ảnh suy giảm/sạch của bạn: YAML tập dữ liệu trỏ
đến thư mục `inputs/<split>/` chứa ảnh suy giảm và thư mục `targets/<split>/`
chứa đích sạch, được ghép theo phần gốc tên tệp. `degradation` và `dataset` là
các chuỗi tùy chọn được ghi vào checkpoint đã lưu để theo dõi nguồn gốc; chúng
không tham gia vào quá trình huấn luyện.

<code-tabs name="train" />

Nếu giữ nguyên mặc định, trình huấn luyện chạy 100 epoch với AdamW ở `lr0=1e-3`,
batch 16, các crop 256 px và dừng sớm sau 50 epoch không cải thiện PSNR. Họ này
không có luồng LoRA: `lora=True` phát sinh lỗi thay vì chạy vì `NAFNetTrainer`
không bật tinh chỉnh bằng adapter.

Trong khi huấn luyện, mạng chạy với global-average pooling thông thường. Phép
local pooling theo cửa sổ chỉ dành cho suy luận của NAFNet (Test-time Local
Converter) được tách ra trước epoch đầu tiên và gắn lại khi huấn luyện hoàn tất,
vì lan truyền ngược qua local pool cửa sổ cố định sẽ không khớp với cách dùng
checkpoint khi suy luận.

Xem [huấn luyện](/docs/train) để biết về tập dữ liệu, tăng cường dữ liệu, multi-GPU và logger.

## Đánh giá

`val()` trả về từ điển có `metrics/PSNR` và `metrics/SSIM`, được tính trong RGB
trên toàn bộ canvas hợp lệ: SSIM dùng cửa sổ Gaussian 11x11 với sigma 1.5, còn
`fitness` để chọn checkpoint tốt nhất là giá trị PSNR. `data` trỏ đến cùng định
dạng tập dữ liệu ảnh ghép cặp dùng cho huấn luyện.

<code-tabs name="val" />

## Xuất

<export-matrix />

Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp
`.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng `Results`, trong
đó `restored` chứa ảnh đầu ra. NAFNet xuất ở độ phân giải không gian cố định:
`imgsz` phải chia hết cho hệ số downsample của mạng (16 cho cả hai độ rộng kiến
trúc), và chỉ chiều batch là động khi `dynamic=True`; chiều cao và chiều rộng
được cố định tại thời điểm xuất.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã công bố cho họ này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />

