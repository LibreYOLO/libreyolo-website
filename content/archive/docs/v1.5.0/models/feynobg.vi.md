---
title: FeyNobg
families:
  - feynobg
seo_title: 'FeyNobg: xóa nền trong LibreYOLO'
description: >-
  Dùng FeyNobg trong LibreYOLO để xóa nền và alpha matting, một biến thể
  BiRefNet sâu hơn từ Feyn Inc. Cài đặt, dự đoán và xác thực.
lead: >-
  Mô hình xóa nền từ Feyn Inc. làm sâu kiến trúc BiRefNet rồi huấn luyện lại.
  LibreYOLO cung cấp inference và xác thực cho tác vụ matte của FeyNobg.
keywords:
  - FeyNobg
  - xóa nền ảnh
  - phân đoạn ảnh nhị phân
  - alpha matte
  - image matting
  - tách chủ thể
  - nobg
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFeyNobgl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Tách chủ thể
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8: RGB nguồn cùng matte làm kênh alpha.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFeyNobgl-matte.pt")

        # Thư mục chứa images/ và thư mục matte được tự động phát hiện
        # (mattes/, matte/, gt/, masks/, mask/ hoặc alpha/) cũng dùng được
        # thay cho YAML dataset.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
source_hash: 45de3b578d7ebbf2
---

## Cài đặt

FeyNobg không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Checkpoint được tải từ tổ chức LibreYOLO trên Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ như mọi họ mô hình khác, dù chưa được liệt kê trong bảng Checkpoint trên trang này.

<code-tabs name="predict" />

Kết quả matte không có box; `result.matte` là mảng float32 dense `(H, W)` trong `[0, 1]`, với 1 là hoàn toàn foreground và 0 là hoàn toàn background. Khác với mặt nạ nhị phân, soft matte giữ chi tiết cạnh anti-alias như tóc và lông. `result.cutout()` ghép ảnh nguồn với kênh alpha đó thành mảng RGBA, còn `result.save(path)` (hoặc `save=True` trong lệnh gọi dự đoán) ghi trực tiếp thành PNG nền trong suốt. Mô hình chạy trên canvas nguyên bản cố định 1024x1024; không hỗ trợ độ phân giải khác vì các bảng relative-position của backbone Swin gắn với độ phân giải này, giá trị không khớp sẽ nội suy sai thay vì phát sinh lỗi. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có một kích thước đã phát hành là `l`, dùng backbone cấp Swin-L. FeyNobg lấy kiến trúc của BiRefNet và tăng độ sâu giai đoạn Swin thứ ba từ 18 lên 24 block trước khi huấn luyện lại, vì vậy bản port LibreYOLO dùng lại forward path, bước tiền xử lý và hợp đồng đầu ra một logit của BiRefNet; dự đoán, xác thực và xử lý checkpoint hoạt động giống họ `birefnet`.

## Xác thực

`val()` báo cáo hai metric trên thư mục ảnh/matte theo cặp, cả hai nằm trong `[0, 1]` và không phụ thuộc độ phân giải: MAE là sai số tuyệt đối trung bình so với alpha ground truth (thấp hơn tốt hơn), còn S-measure (Fan và cộng sự, ICCV 2017) là độ tương đồng cấu trúc ghi nhận việc giữ hình dạng và lỗ của chủ thể mà riêng MAE pixel bỏ sót (cao hơn tốt hơn). Quá trình xác thực điều khiển chính `predict` của mô hình, vì vậy sử dụng chính xác bước tiền xử lý của họ mô hình.

<code-tabs name="val" />

Việc xác thực chỉ phục vụ inference. Thư viện `nobg` upstream cung cấp mã huấn luyện Apache-2.0; hiện tại tinh chỉnh nghĩa là huấn luyện ở đó rồi chuyển đổi kết quả bằng script chuyển đổi riêng của LibreYOLO, không phải gọi `train()` trên họ mô hình này vì hàm sẽ phát sinh lỗi thay vì chạy trainer không đầy đủ.

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />


