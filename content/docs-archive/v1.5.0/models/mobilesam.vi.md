---
title: MobileSAM
families:
  - mobilesam
seo_title: 'MobileSAM: phân đoạn theo prompt nhẹ trong LibreYOLO'
description: >-
  Dùng MobileSAM trong LibreYOLO để phân đoạn theo prompt điểm và box bằng
  encoder TinyViT. Cài đặt và dự đoán checkpoint tiny theo Apache-2.0.
lead: >-
  MobileSAM thay image encoder ViT-H của SAM bằng encoder TinyViT được chưng
  cất, vì vậy cùng quy trình theo prompt điểm và box có thể chạy trên phần cứng
  nhẹ hơn. LibreYOLO cung cấp bản port nguyên bản qua factory LibreSAM chuyên
  dụng, tách biệt với factory detector LibreYOLO().
keywords:
  - MobileSAM
  - Segment Anything
  - TinyViT
  - phân đoạn theo prompt
  - phân đoạn tương tác
  - prompt điểm
  - prompt box
  - phân đoạn nhẹ
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompt điểm và box
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # MobileSAM có một kích thước "tiny", vì vậy không cần bí danh khác.
        model = LibreSAM("mobilesam")

        # Prompt điểm: [x, y] theo tọa độ pixel, nhãn 1 = foreground.
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # polygon cho mỗi mặt nạ
        print(result.boxes.xyxy)    # box khít được suy ra từ mặt nạ

        # Dùng prompt box thay cho điểm.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])

        # Không có prompt sẽ phân đoạn toàn ảnh (trình tạo mặt nạ tự động
        # đơn giản hóa, không phải bản tham chiếu toàn diện).
        result = model.predict(SAMPLE_IMAGE)
    - label: 'Mã hóa một lần, dùng nhiều prompt'
      language: python
      code: |
        from libreyolo import LibreMobileSAM, SAMPLE_IMAGE

        model = LibreMobileSAM()

        # Image encoder là phần tốn kém. set_image() chạy nó một lần;
        # mỗi lần gọi predict() sau đó dùng lại embedding trong bộ nhớ đệm.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
source_hash: f96e885d93f72bdd
---

## Cài đặt

MobileSAM cần extra `sam`: quá trình tải trọng số riêng của LibreYOLO vẫn đi qua công cụ snapshot Hugging Face của `transformers`, dù inference chạy trên decoder nguyên bản không dùng `transformers`.

```bash
pip install "libreyolo[sam]"
```

## Dự đoán

`LibreSAM(...)` (hoặc `LibreMobileSAM(...)` dành riêng cho họ mô hình) là entry point tách biệt với `LibreYOLO(...)`: hàm trả về segmenter theo prompt thay vì detector vì forward pass ở đây không có ý nghĩa nếu thiếu prompt không gian. Không có lệnh CLI `libreyolo predict` cho họ mô hình này; hãy dùng Python API.

<code-tabs name="predict" />

Prompt điểm nhận `[x, y]` cho một đối tượng, `[[x, y], ...]` cho nhiều đối tượng hoặc mảng numpy; `labels` đánh dấu mỗi điểm là `1` (foreground) hoặc `0` (background), mặc định tất cả là foreground. Prompt box nhận `[x1, y1, x2, y2]` hoặc danh sách box, mỗi box một mặt nạ. Bỏ cả hai prompt sẽ phân đoạn toàn ảnh bằng cách đặt prompt trên lưới dense và giữ lại các mặt nạ có độ tin cậy cao, không chồng lấp; chế độ "phân đoạn mọi thứ" này được đơn giản hóa so với trình tạo mặt nạ tự động tham chiếu và có thể phân đoạn thiếu trong cảnh đông, vì vậy prompt điểm hoặc box thực là tuyến chính xác hơn. `conf` lọc theo chất lượng mặt nạ dự đoán (IoU), không phải độ tin cậy phát hiện: truyền `0.0` để giữ mọi candidate. `multimask=True` trả về cả ba mặt nạ biểu thị sự mơ hồ toàn thể so với bộ phận của SAM cho mỗi prompt thay vì chỉ mặt nạ tốt nhất. `device=` chuyển mô hình và nếu đang có phiên `set_image()`, cả embedding trong bộ nhớ đệm. Mọi mặt nạ mang ID lớp đối tượng `0`, tên `"object"`, vì mặt nạ theo prompt không có tập lớp đối tượng cố định. `train()`, `val()`, `export()` và `track()` đều phát sinh `NotImplementedError` cho họ mô hình này: MobileSAM chỉ dành cho dự đoán trong LibreYOLO. Xem [dự đoán](/docs/predict) để biết các loại nguồn.

## Biến thể

Có một kích thước tiny ở đầu vào cố định 1024 px: MobileSAM cung cấp một encoder TinyViT duy nhất thay vì dãy base/large/huge của SAM-1.

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />


