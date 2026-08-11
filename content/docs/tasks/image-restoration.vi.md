---
title: Phục hồi ảnh
seo_title: Phục hồi và nâng độ phân giải ảnh trong LibreYOLO
description: >-
  Khử nhiễu, khử nhòe và nâng độ phân giải ảnh trong LibreYOLO. Dự đoán ảnh RGB
  đã phục hồi, huấn luyện NAFNet trên dữ liệu ghép cặp và đọc các key PSNR cùng
  SSIM.
lead: >-
  Phục hồi ảnh nhận ảnh bị suy giảm và trả về ảnh sạch. LibreYOLO cung cấp dưới
  dạng tác vụ restore, bao gồm khử nhiễu, khử nhòe và super-resolution phía sau
  một hợp đồng đầu ra duy nhất: một ảnh RGB đi vào, một ảnh RGB đi ra.
keywords:
  - phục hồi ảnh python
  - mô hình khử nhiễu ảnh
  - super resolution python
  - mô hình khử nhòe
  - xác thực PSNR SSIM
last_verified: 1.5.0
snippets:
  predict:
    - label: Nâng độ phân giải ảnh
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Generator 4x nhỏ gọn; tile giới hạn bộ nhớ đỉnh trên nguồn lớn.
        model = LibreYOLO("LibreRealESRGANx4t-restore.pt")
        result = model(SAMPLE_IMAGE, tile=512, tile_pad=10)

        result.restored.save("upscaled.png")
        print(result.restored.array.shape)   # gấp 4x đầu vào trên mỗi trục
    - label: Khử nhiễu ảnh
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Được huấn luyện trên nhiễu ảnh thực SIDD; đầu ra giữ kích thước đầu
        vào.

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        result = model(SAMPLE_IMAGE)


        result.restored.save("denoised.png")

        print(result.restore_scale)   # 1: checkpoint này không nâng độ phân
        giải
  train:
    - label: Tinh chỉnh NAFNet trên các cặp ảnh
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: Ghi nguồn gốc vào checkpoint
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # degradation và dataset được ghi vào checkpoint đã lưu để theo dõi
        # nguồn gốc; chúng không tham gia huấn luyện.
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
  val:
    - label: Xác thực và đọc các key metric
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val() trả về dict thuần, không phải đối tượng.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])   # fitness
        print(metrics["metrics/SSIM"])
  export:
    - label: Xuất
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # imgsz được cố định trong graph, vì vậy hãy truyền kích thước mà
        # bản triển khai thực sự đưa vào mô hình.
        model.export(format="onnx", imgsz=256)
    - label: Chạy tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, vì vậy artifact đã xuất được nạp
        # như mọi checkpoint khác và trả về cùng một đối tượng Results.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")
        result = model(SAMPLE_IMAGE)

        result.restored.save("denoised.png")
source_hash: 9dc81cadb3ebf18b
---

## Định nghĩa

Tác vụ `restore` ánh xạ một ảnh sang ảnh khác. Khử nhiễu, khử nhòe và
super-resolution đều là cùng một tác vụ tại đây vì chúng dùng chung hợp đồng:
mô hình nhận ảnh RGB và trả về ảnh RGB, còn loại suy giảm đã được huấn luyện để
khắc phục là thuộc tính checkpoint thay vì API.

Dự đoán điền `result.restored`, một payload `RestoredImage` chứa mảng RGB uint8
`(H, W, 3)`. `.array` trả về dưới dạng NumPy, còn `.save(path)` ghi ra ổ đĩa.
`result.restore_scale` ghi lại hệ số nâng độ phân giải của canvas đầu ra, bằng
`1` cho checkpoint giữ nguyên độ phân giải. `result.boxes` luôn rỗng, vì vậy
`conf`, `iou` và `max_det` được chấp nhận để giữ tương thích signature nhưng
không có tác dụng, còn `save=True` ghi trực tiếp ảnh đã phục hồi thay vì ảnh có
chú thích.

## Mô hình

Ba family phục vụ `restore`, được chia theo loại suy giảm mà chúng khắc phục.

[NAFNet](/docs/models/nafnet) là mô hình khử nhiễu và là restore family duy nhất
LibreYOLO có thể huấn luyện. Kiến trúc thay activation phi tuyến của block UNet
bằng phép nhân theo phần tử, còn checkpoint được công bố đã huấn luyện trên
nhiễu ảnh thực SIDD. Đầu ra giữ nguyên độ phân giải đầu vào.

[Real-ESRGAN](/docs/models/real-esrgan) là mô hình nâng độ phân giải thực dụng:
ba checkpoint được huấn luyện theo suy giảm tổng hợp thay vì chỉ downscale
bicubic, ở mức 4x, 2x và một generator 4x nhỏ hơn, nhanh hơn dành cho độ trễ thấp.

[SwinIR](/docs/models/swinir) nâng độ phân giải 4x bằng backbone Swin Transformer,
với ba kích thước bao gồm generator nhẹ chính thức và hai generator cho ảnh
thực tế.

## Dự đoán

Trọng số được tải từ Hugging Face trong lần sử dụng đầu tiên và lưu vào cache
cục bộ.

<code-tabs name="predict" />

Phục hồi chạy ở độ phân giải riêng của ảnh nguồn thay vì canvas mạng cố định,
chỉ padding theo hệ số downsample của mạng, vì vậy thời gian và bộ nhớ đều tăng
theo số pixel đầu vào. `tile` tách forward pass thành các tile chồng lấn rồi
blend seam lại với nhau, còn `tile_pad` là halo được thêm quanh mỗi tile trước
khi crop lại; cả hai là đối số keyword Python. Xem [dự đoán](/docs/predict) để
biết về nguồn, stream và cách xử lý kết quả.

## Định dạng dataset

Phục hồi ghép mỗi ảnh đầu vào bị suy giảm với ảnh target sạch có đúng cùng độ
phân giải, khớp theo stem tên tệp.

```text
dataset/
  data.yaml
  inputs/
    train/photo.jpg
    val/photo.jpg
  targets/
    train/photo.jpg
    val/photo.jpg
```

```yaml
path: dataset
train: inputs/train
val: inputs/val
input_dir: inputs
target_dir: targets
degradation: denoise
dataset: MyDataset
nc: 1
names: {0: image}
```

`nc` và `names` là placeholder schema; mô hình restore trả về
`Results.restored`, không phải kết quả phát hiện. `degradation` và `dataset` là
nhãn nguồn gốc tùy chọn. `target_stem_suffix` hỗ trợ dataset đặt tên ảnh sạch
khác với cặp ảnh suy giảm. Quá trình xác thực giữ độ phân giải gốc và chỉ
padding đủ để xếp thành batch, vì vậy metric được tính trên canvas gốc. Xem
[định dạng dataset](/docs/reference/dataset-formats) để biết hợp đồng đầy đủ.

## Huấn luyện

NAFNet là restore family duy nhất có implementation huấn luyện.
`Real-ESRGAN.train()` và `SwinIR.train()` đều phát sinh `NotImplementedError`:
các checkpoint đó đến từ quá trình huấn luyện GAN trên pipeline suy giảm tổng
hợp, còn trainer phục hồi ghép cặp sẽ chạy mà không tái tạo recipe đó.

<code-tabs name="train" />

Trainer lấy các vùng crop ghép cặp từ đầu vào và target, vì vậy hai phía luôn
căn chỉnh. Xem [huấn luyện](/docs/train) để biết về dataset, multi-GPU và logger,
cùng [trang NAFNet](/docs/models/nafnet) để biết giá trị mặc định của family và
pooling tại thời điểm inference được tách ra trong khi huấn luyện.

## Xác thực

`val()` so sánh đầu ra đã phục hồi với target sạch, trong RGB, trên canvas gốc,
không crop biên và không đổi kích thước.

<code-tabs name="val" />

`metrics/PSNR` là peak signal-to-noise ratio theo decibel, đồng thời là
`fitness`, con số mà cơ chế chọn checkpoint tốt nhất đọc. `metrics/SSIM` là độ
tương đồng cấu trúc trong `[0, 1]`, được tính bằng cửa sổ Gaussian 11x11 ở sigma
1.5 và lấy trung bình trên ba kênh màu. Cả hai đều cao hơn là tốt hơn.

## Xuất

Mô hình restore đã xuất được nạp lại qua `LibreYOLO()` theo hậu tố tệp, vì vậy
tệp `.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng `Results`,
với `restored` mang ảnh đầu ra.

<code-tabs name="export" />

Việc xuất restore cố định độ phân giải không gian vào graph, vì vậy hãy truyền
`imgsz` mà bản triển khai thực sự đưa vào mô hình. Với NAFNet, kích thước đó phải
chia hết cho hệ số downsample của mạng, và chỉ chiều batch giữ dạng động dưới
`dynamic=True`. Với Real-ESRGAN và SwinIR, bỏ `imgsz` sẽ quay về kích thước patch
nội bộ nhỏ thay vì độ phân giải làm việc. Phạm vi theo định dạng nằm trên từng
trang mô hình và trong [ma trận xuất đầy đủ](/docs/reference/export-matrix).
Phần [Xuất](/docs/export) liệt kê các đối số mà mọi định dạng chấp nhận.
