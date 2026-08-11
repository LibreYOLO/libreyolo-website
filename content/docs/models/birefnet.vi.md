---
title: BiRefNet
families:
  - birefnet
seo_title: 'BiRefNet: xóa nền và matting trong LibreYOLO'
description: >-
  Dùng BiRefNet trong LibreYOLO để xóa nền và dichotomous image segmentation.
  Cài đặt, dự đoán, kiểm định và xuất checkpoint general.
lead: >-
  Một mạng bilateral-reference dự đoán alpha matte mềm để tách chủ thể khỏi nền.
  LibreYOLO cung cấp suy luận (inference) và kiểm định cho tác vụ matte của
  BiRefNet.
keywords:
  - BiRefNet
  - xóa nền ảnh
  - background removal
  - dichotomous image segmentation
  - alpha matte
  - image matting python
  - tách nền png trong suốt
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreBiRefNetl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Cutout
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8: RGB của ảnh gốc cộng thêm matte làm kênh alpha
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # Một thư mục chứa images/ và một thư mục matte được tự động phát hiện
        # (mattes/, matte/, gt/, masks/, mask/ hoặc alpha/) cũng dùng được thay
        # cho một YAML dataset
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreBiRefNetl-matte.pt format=onnx
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo phần mở rộng tệp, nên một artifact đã xuất
        # được nạp như mọi checkpoint và trả về cùng một đối tượng Results
        model = LibreYOLO("LibreBiRefNetl-matte.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
source_hash: 1af1bd7f4f905081
---

## Cài đặt

BiRefNet không cần extra tùy chọn nào. Mọi thứ nó import đều nằm trong bản cài
đặt cơ bản.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải về từ Hugging Face ở lần chạy đầu tiên và được lưu cache cục
bộ.

<code-tabs name="predict" />

Kết quả matte không mang theo box nào; `result.matte` là một mảng float32 dày
đặc dạng `(H, W)` với giá trị trong `[0, 1]`, 1 là hoàn toàn foreground và 0 là
hoàn toàn background. Khác với mặt nạ (mask) nhị phân, matte mềm giữ được chi
tiết viền đã khử răng cưa như tóc và lông. `result.cutout()` ghép ảnh gốc với
kênh alpha đó thành một mảng RGBA, còn `result.save(path)` (hoặc `save=True`
trong lời gọi predict) ghi thẳng ra một tệp PNG nền trong suốt. Mô hình chạy
trên canvas gốc cố định 1024x1024; độ phân giải khác không được hỗ trợ, vì các
bảng vị trí tương đối của backbone Swin gắn chặt với kích thước đó, và khi
không khớp thì chúng bị nội suy sai lệch thay vì báo lỗi. Xem
[dự đoán](/docs/predict) để biết về nguồn đầu vào, streaming và cách xử lý kết
quả.

## Các biến thể

Một checkpoint được công bố, `l`, là mô hình BiRefNet-general ở tier Swin-L và
cũng là mặc định thiên về chất lượng ở upstream. Mã nguồn của family này cũng
hỗ trợ tier lite Swin-T, `t`, nhưng chưa có bản chuyển đổi LibreYOLO nào cho nó
được công bố.

## Kiểm định

`val()` báo cáo hai chỉ số trên một thư mục chứa các cặp ảnh/matte, cả hai đều
nằm trong `[0, 1]` và không phụ thuộc độ phân giải: MAE, là mean absolute error
so với alpha ground truth (càng thấp càng tốt), và S-measure (Fan và cộng sự,
ICCV 2017), một độ tương đồng cấu trúc ghi nhận việc giữ được hình dáng và các
lỗ thủng của chủ thể, thứ mà chỉ riêng MAE theo pixel bỏ sót (càng cao càng
tốt). Quá trình kiểm định chạy chính `predict` của mô hình, nên nó dùng đúng
phần tiền xử lý của family.

<code-tabs name="val" />

Kiểm định chỉ thực hiện inference; tinh chỉnh (fine-tuning) là hướng tiếp theo
đã được ghi lại chứ chưa phải một tính năng có sẵn (xem Dự đoán để biết ràng
buộc độ phân giải chính xác mà bất kỳ trainer nào trong tương lai cũng sẽ kế
thừa).

## Xuất mô hình

<export-matrix />

Một artifact đã xuất được nạp lại qua `LibreYOLO()` dựa trên phần mở rộng tệp,
nên một tệp `.onnx` hoạt động như một checkpoint và trả về cùng `Results`.
TorchScript là đường đi đã được kiểm chứng; chuyển đổi ONNX chạy được nhưng
chưa đạt cùng mức chuẩn về parity. [Xuất mô hình](/docs/export) liệt kê các
tham số mà mọi format đều nhận và những tham số bổ sung mà một vài format thêm
vào.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã công bố cho family này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />
