---
title: Pháp tuyến bề mặt
seo_title: Ước lượng pháp tuyến bề mặt trong LibreYOLO
description: >-
  Dự đoán trường pháp tuyến bề mặt dày đặc từ một ảnh trong LibreYOLO. Đọc quy
  ước hệ tọa độ camera, xác thực sai số góc và xuất mô hình.
lead: >-
  Ước lượng pháp tuyến bề mặt dự đoán hướng mà mỗi bề mặt nhìn thấy đang quay
  về. LibreYOLO cung cấp dưới dạng tác vụ normal, trả về trường vector đơn vị
  dày đặc trên canvas ảnh gốc.
keywords:
  - ước lượng pháp tuyến bề mặt python
  - tạo normal map từ ảnh
  - hình học monocular
  - metric sai số góc
  - dự đoán pháp tuyến dày đặc
last_verified: 1.5.0
snippets:
  predict:
    - label: Dự đoán trường pháp tuyến
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreMoGe2s-normal.pt")

        result = model(SAMPLE_IMAGE, save=True)


        normals = result.normal_map

        print(normals.data.shape)      # (H, W, 3) vector đơn vị float32

        normals.assert_normalized()    # phát sinh lỗi nếu pixel nào không có độ
        dài đơn vị
    - label: Đọc một pixel
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE)

        # Hệ tọa độ camera OpenCV: +x sang phải, +y xuống dưới, +z vào cảnh.
        # Bề mặt quay về camera có giá trị gần (0, 0, -1).
        field = result.normals.data
        h, w = field.shape[:2]
        print(field[h // 2, w // 2])
    - label: Lưu ảnh trực quan hóa
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreMoGe2s-normal.pt")

        result = model(SAMPLE_IMAGE)


        # plot() render trường; phương thức được định nghĩa cho kết quả normal
        và edge.

        result.plot().save("normals.png")
  val:
    - label: Xác thực và đọc các key metric
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])     # độ
        print(metrics["metrics/median_angular_error"])   # độ
        print(metrics["metrics/within_11_25"])           # phần trăm pixel
        print(metrics["metrics/within_22_5"], metrics["metrics/within_30"])
  export:
    - label: Xuất
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
    - label: Chạy tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, vì vậy artifact đã xuất được nạp
        # như mọi checkpoint khác và trả về cùng một đối tượng Results.
        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.normal_map.data.shape)
source_hash: d26d26d894b436ff
---

## Định nghĩa

Tác vụ `normal` dự đoán một vector đơn vị ba thành phần trên mỗi pixel từ một
ảnh RGB: hướng mà bề mặt tại pixel đó đang quay về. Khác với độ sâu, đầu ra
không có tỷ lệ tự do, vì vậy có thể so sánh trực tiếp hai dự đoán mà không cần
căn chỉnh.

Dự đoán điền `result.normal_map`, một payload `NormalMap` chứa mảng float32
`(H, W, 3)` trên canvas ảnh gốc, cũng có thể truy cập dưới dạng
`result.normals`. Vector dùng hệ tọa độ camera OpenCV của LibreYOLO, với `+x`
sang phải, `+y` xuống dưới và `+z` vào cảnh, đồng thời quay về camera, vì vậy bề
mặt song song với mặt phẳng ảnh có giá trị `(0, 0, -1)`. `.assert_normalized()`
kiểm tra mọi pixel hữu hạn và có độ dài đơn vị trong phạm vi dung sai.
`result.boxes` luôn rỗng, nên `conf`, `iou` và `max_det` không có tác dụng, còn
`Results.plot()` hỗ trợ tác vụ này.

## Mô hình

Hai family phục vụ tác vụ `normal`.

[MoGe-2](/docs/models/moge-2) là mô hình chuyên dụng: mô hình hình học monocular
chạy một forward pass với ba kích thước encoder. LibreYOLO không sao chép các
checkpoint này vào tổ chức riêng; việc nạp một checkpoint tải kích thước tương
ứng từ repo chính thức tại revision cố định và xác minh bằng SHA-256 đã ghi lại.

[LibreMODUS](/docs/models/libremodus) tạo pháp tuyến như một target của mô hình
any-to-any và có thể nhận depth map thay vì ảnh RGB làm đầu vào. Nó cần thành
phần bổ sung `modus` và tài khoản Hugging Face đã xác thực riêng của bạn, đồng
thời không cung cấp `val()` lẫn `export()`, vì vậy không tham gia các phần xác
thực và xuất bên dưới.

## Dự đoán

Trọng số MoGe-2 được tải trong lần sử dụng đầu tiên và lưu vào cache cục bộ.

<code-tabs name="predict" />

`imgsz` phải chia hết cho kích thước patch của encoder ViT, điều LibreYOLO kiểm
tra trước khi lượt chạy bắt đầu. Dự đoán danh sách ảnh sẽ chạy một forward pass
cho mỗi ảnh; tác vụ này không có đường dẫn nhanh cho batch xếp chồng. Xem [dự
đoán](/docs/predict) để biết về nguồn, stream và cách xử lý kết quả.

## Định dạng dataset

Xác thực pháp tuyến ghép mỗi ảnh với một PNG 16-bit ba kênh có cùng stem và cùng
độ phân giải, cùng một mặt nạ hợp lệ tùy chọn.

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  normals/
    val/room.png
  masks/
    val/room.png
```

```yaml
path: dataset
train: images/train
val: images/val
normals_dir: normals
masks_dir: masks
nc: 1
names: {0: normal}
```

PNG target phải chính xác là `uint16` ba kênh, với các kênh lưu theo RGB. Công
thức decode là `n = png / 65535 * 2 - 1`, sau đó chuẩn hóa lại từng vector, và
các vector đã decode dùng cùng hệ tọa độ camera OpenCV như dự đoán. Pixel mặt nạ
được tính là hợp lệ khi khác 0; nếu không có tệp mặt nạ, mọi vector đã decode
hữu hạn và khác 0 đều hợp lệ. Pixel target không hợp lệ và được padding được giữ
nội bộ dưới dạng `(0, 0, 0)` và không bao giờ đóng góp vào metric. Xem [định
dạng dataset](/docs/reference/dataset-formats) để biết hợp đồng đầy đủ.

## Huấn luyện

Không family pháp tuyến nào có implementation huấn luyện: `train()` phát sinh
`NotImplementedError` trên cả hai. Trang MoGe-2 trỏ tới các checkpoint chính
thức được cố định để dự đoán, xác thực và xuất.

## Xác thực

`val()` đo góc giữa mỗi vector dự đoán và vector ground truth của nó trên các
pixel được dataset đánh dấu hợp lệ.

<code-tabs name="val" />

`metrics/mean_angular_error` và `metrics/median_angular_error` là góc đó theo độ,
giá trị thấp hơn là tốt hơn. `metrics/within_11_25`, `metrics/within_22_5` và
`metrics/within_30` là tỷ lệ phần trăm pixel hợp lệ có sai số góc nằm trong
11.25, 22.5 và 30 độ, vì vậy giá trị cao hơn là tốt hơn. Lưu ý đơn vị: ba giá
trị này là phần trăm, không phải phân số. `fitness` là
`metrics/within_11_25` chia cho 100, đưa việc chọn checkpoint tốt nhất về cùng
thang `[0, 1]` như mọi tác vụ khác.

## Xuất

Mô hình pháp tuyến đã xuất được nạp lại qua `LibreYOLO()` theo hậu tố tệp, vì vậy
tệp `.onnx` hoạt động như checkpoint và trả về cùng `Results`.

<code-tabs name="export" />

Việc xuất pháp tuyến dùng hợp đồng runtime batch 1, độ phân giải cố định:
`dynamic` và `batch` khác 1 bị từ chối, còn `imgsz` phải chia hết cho kích thước
patch của encoder. Phạm vi theo định dạng nằm trên [trang
MoGe-2](/docs/models/moge-2) và trong [ma trận xuất đầy
đủ](/docs/reference/export-matrix). Phần [Xuất](/docs/export) liệt kê các đối số
mà mọi định dạng chấp nhận.
