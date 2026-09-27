---
title: Xóa nền
seo_title: Xóa nền trong LibreYOLO
description: >-
  Tách chủ thể khỏi nền trong LibreYOLO. Dự đoán alpha matte mềm, ghi PNG trong
  suốt và xác thực bằng MAE cùng S-measure.
lead: >-
  Xóa nền tách chủ thể khỏi mọi thứ phía sau. LibreYOLO cung cấp dưới dạng tác
  vụ matte, trả về giá trị alpha mềm trên mỗi pixel thay vì mặt nạ foreground
  cứng.
keywords:
  - xóa nền bằng python
  - mô hình alpha matting
  - phân đoạn ảnh nhị phân
  - tách nền png trong suốt
  - soft alpha matte
last_verified: 1.5.0
snippets:
  predict:
    - label: Dự đoán matte
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        result = model(SAMPLE_IMAGE)


        matte = result.matte

        print(matte.array.shape, matte.array.dtype)   # (H, W) float32 trong [0,
        1]
    - label: Ghi PNG trong suốt
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # save() kết hợp nguồn với matte làm kênh alpha.
        result.save("subject.png")

        rgba = result.cutout()   # cùng mảng uint8 (H, W, 4) trong bộ nhớ
        print(rgba.shape)
    - label: Ghép lên nền mới
      language: python
      code: >
        import numpy as np

        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        result = model(SAMPLE_IMAGE)


        rgba = result.cutout()

        alpha = rgba[..., 3:4].astype(np.float32) / 255.0

        backdrop = np.full_like(rgba[..., :3], 255)          # trắng

        composited = (rgba[..., :3] * alpha + backdrop * (1 -
        alpha)).astype(np.uint8)

        print(composited.shape)
  val:
    - label: Xác thực và đọc các key metric
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # Có thể dùng thư mục chứa images/ và thư mục matte thay cho
        # YAML dataset.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])        # thấp hơn là tốt hơn
        print(metrics["metrics/Smeasure"])   # fitness, cao hơn là tốt hơn
  export:
    - label: Xuất
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="torchscript")
    - label: Chạy tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, vì vậy artifact đã xuất được nạp
        # như mọi checkpoint khác và trả về cùng một đối tượng Results.
        model = LibreYOLO("LibreBiRefNetl-matte.torchscript")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
source_hash: f7d88c74d9729268
---

## Định nghĩa

Tác vụ `matte` dự đoán một giá trị alpha trên mỗi pixel từ một ảnh RGB: `1` là
hoàn toàn foreground và `0` là hoàn toàn background. Giá trị liên tục thay vì
nhị phân, đây chính là mục đích của tác vụ. Mặt nạ cứng chỉ cần một ngưỡng ở
0.5, còn matte mềm bổ sung phần bao phủ một phần tại tóc, lông và cạnh bị nhòe
do chuyển động mà mặt nạ nhị phân loại bỏ.

Dự đoán điền `result.matte`, một payload `Matte` chứa mảng float32 `(H, W)`
trong `[0, 1]` trên canvas ảnh gốc, có thể truy cập dưới dạng NumPy qua `.array`.
`result.cutout()` ghép ảnh nguồn với alpha đó thành mảng RGBA uint8
`(H, W, 4)`, còn `result.save(path)` ghi cùng nội dung vào PNG có nền trong
suốt. `result.boxes` luôn rỗng, nên `conf`, `iou` và `max_det` không có tác dụng.

## Mô hình

Hai family phục vụ `matte` và dùng chung forward path.

[BiRefNet](/docs/models/birefnet) là bilateral-reference network làm nền tảng
cho tác vụ, được công bố tại đây dưới dạng một checkpoint tầng Swin-L.

[FeyNobg](/docs/models/feynobg) là biến thể sâu hơn của Feyn Inc.: kiến trúc
BiRefNet với stage Swin thứ ba tăng từ 18 lên 24 block, sau đó được huấn luyện
lại. LibreYOLO tái sử dụng forward path, preprocessing và đầu ra một logit của
BiRefNet cho mô hình này, vì vậy dự đoán, xác thực và xử lý checkpoint hoạt động
giống hệt; trọng số và danh tính family là của chính FeyNobg.

Hai family dùng giấy phép trọng số khác nhau. Cả hai được nêu trên trang mô hình,
và giấy phép trong repo Hugging Face của checkpoint cụ thể là nguồn có thẩm
quyền.

## Dự đoán

Trọng số được tải từ Hugging Face trong lần sử dụng đầu tiên và lưu vào cache
cục bộ.

<code-tabs name="predict" />

Cả hai family chạy trên canvas gốc cố định 1024x1024 và đổi kích thước matte về
ảnh gốc. Không hỗ trợ độ phân giải khác vì bảng relative-position của backbone
Swin gắn với kích thước đó, còn khi không khớp, việc nội suy làm hỏng bảng thay
vì phát sinh lỗi. `Results.save()` chỉ được định nghĩa cho kết quả matte và cần
ảnh nguồn, được nạp lại từ `Results.path` trừ khi bạn truyền một ảnh. Xem [dự
đoán](/docs/predict) để biết về nguồn, stream và cách xử lý kết quả.

## Định dạng dataset

Xác thực matte ghép mỗi ảnh RGB với alpha matte ground truth một kênh có cùng
stem, trong đó 0 là background và 255 là foreground.

```text
my-matte-dataset/
  images/
    subject.jpg
  mattes/
    subject.png
```

Chỉ cần truyền thư mục gốc đó làm `data=`: thư mục matte được tự động phát hiện
trong số `mattes/`, `matte/`, `gt/`, `masks/`, `mask/` và `alpha/`. YAML dataset
là phương án thay thế, gồm `path` cùng `val_images` và `val_mattes` đặt tên thư
mục tương đối với nó:

```yaml
path: my-matte-dataset
val_images: images
val_mattes: mattes
nc: 1
names: {0: matte}
```

`nc` và `names` là placeholder schema; mô hình matte trả về `Results.matte`,
không phải kết quả phát hiện. Giá trị matte được đọc thành alpha trong `[0, 1]`
bằng cách chia cho 255, còn matte có shape khác canvas dự đoán được đổi kích
thước theo kiểu bilinear cho khớp. Xem [định dạng
dataset](/docs/reference/dataset-formats) để biết hợp đồng đầy đủ.

## Huấn luyện

Không family matte nào có implementation huấn luyện: `train()` phát sinh
`NotImplementedError` trên cả hai, còn hỗ trợ matte chỉ bao gồm dự đoán, xác
thực và xuất. Mỗi trang mô hình nêu tên dự án upstream có mã huấn luyện và
script chuyển checkpoint trở lại.

## Xác thực

`val()` điều khiển `predict` riêng của mô hình, vì vậy quá trình xác thực dùng
đúng preprocessing của family, và cả hai metric được tính trên canvas ảnh gốc.

<code-tabs name="val" />

`metrics/MAE` là sai số tuyệt đối trung bình so với alpha ground truth trong
`[0, 1]`, giá trị thấp hơn là tốt hơn. `metrics/Smeasure` là S-measure của Fan
và cộng sự (ICCV 2017), phép đo độ tương đồng cấu trúc ghi nhận việc tái tạo đúng
hình dạng cùng các lỗ của chủ thể, nội dung mà trung bình theo pixel đơn thuần
bỏ sót; giá trị cao hơn là tốt hơn. S-measure cũng là `fitness`, con số mà cơ
chế chọn checkpoint tốt nhất đọc. Không metric nào phụ thuộc vào độ phân giải.

## Xuất

Mô hình matte đã xuất được nạp lại qua `LibreYOLO()` theo hậu tố tệp, vì vậy
artifact hoạt động như checkpoint và trả về cùng `Results`.

<code-tabs name="export" />

TorchScript là đường dẫn đã xác thực cho tác vụ này. Chuyển đổi ONNX chạy được
nhưng chưa vượt qua cùng tiêu chuẩn độ tương đồng, còn các định dạng khác không
khả dụng. Phạm vi theo định dạng nằm trên các trang
[BiRefNet](/docs/models/birefnet) và [FeyNobg](/docs/models/feynobg), cùng [ma
trận xuất đầy đủ](/docs/reference/export-matrix).
