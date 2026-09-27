---
title: EdgeTAM
families:
  - edgetam
seo_title: 'EdgeTAM: phân đoạn theo prompt trên thiết bị trong LibreYOLO'
description: >-
  Dùng EdgeTAM trong LibreYOLO để phân đoạn theo prompt điểm và box, được xây
  dựng cho tốc độ trên thiết bị. Cài đặt và dự đoán checkpoint theo Apache-2.0.
lead: >-
  EdgeTAM là biến thể SAM 2 chạy trên thiết bị, được xây dựng cho tốc độ
  inference di động mà vẫn giữ quy trình theo prompt điểm và box. LibreYOLO hỗ
  trợ tuyến phân đoạn ảnh qua factory LibreSAM chuyên dụng, tách biệt với
  factory detector LibreYOLO().
keywords:
  - EdgeTAM
  - SAM 2
  - phân đoạn theo prompt
  - phân đoạn tương tác
  - phân đoạn trên thiết bị
  - prompt điểm
  - prompt box
  - Meta Reality Labs
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompt điểm và box
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # EdgeTAM có một kích thước "edge". Bí danh: "edgetam", "edge-tam",
        # "edgetam-edge".
        model = LibreSAM("edgetam")

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
        from libreyolo import LibreEdgeTAM, SAMPLE_IMAGE

        model = LibreEdgeTAM()

        # Image encoder là phần tốn kém. set_image() chạy nó một lần;
        # mỗi lần gọi predict() sau đó dùng lại embedding trong bộ nhớ đệm.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
source_hash: e6cce8faad18e73d
---

## Cài đặt

EdgeTAM cần extra `sam`, extra này kéo về `transformers` và `timm`.

```bash
pip install "libreyolo[sam]"
```

## Dự đoán

`LibreSAM(...)` (hoặc `LibreEdgeTAM(...)` dành riêng cho họ mô hình) là entry point tách biệt với `LibreYOLO(...)`: hàm trả về segmenter theo prompt thay vì detector vì forward pass ở đây không có ý nghĩa nếu thiếu prompt không gian. Không có lệnh CLI `libreyolo predict` cho họ mô hình này; hãy dùng Python API. Chỉ hỗ trợ phân đoạn ảnh; tính năng theo dõi video của EdgeTAM nằm ngoài phạm vi ở đây.

<code-tabs name="predict" />

Prompt điểm nhận `[x, y]` cho một đối tượng, `[[x, y], ...]` cho nhiều đối tượng hoặc mảng numpy; `labels` đánh dấu mỗi điểm là `1` (foreground) hoặc `0` (background), mặc định tất cả là foreground. Prompt box nhận `[x1, y1, x2, y2]` hoặc danh sách box, mỗi box một mặt nạ. Bỏ cả hai prompt sẽ phân đoạn toàn ảnh bằng cách đặt prompt trên lưới dense và giữ lại các mặt nạ có độ tin cậy cao, không chồng lấp; chế độ "phân đoạn mọi thứ" này được đơn giản hóa so với trình tạo mặt nạ tự động tham chiếu và có thể phân đoạn thiếu trong cảnh đông, vì vậy prompt điểm hoặc box thực là tuyến chính xác hơn. `conf` lọc theo chất lượng mặt nạ dự đoán (IoU), không phải độ tin cậy phát hiện: truyền `0.0` để giữ mọi candidate. `multimask=True` trả về cả ba mặt nạ biểu thị sự mơ hồ toàn thể so với bộ phận của SAM cho mỗi prompt thay vì chỉ mặt nạ tốt nhất. `device=` chuyển mô hình và nếu đang có phiên `set_image()`, cả embedding trong bộ nhớ đệm. Mọi mặt nạ mang ID lớp đối tượng `0`, tên `"object"`, vì mặt nạ theo prompt không có tập lớp đối tượng cố định. `train()`, `val()`, `export()` và `track()` đều phát sinh `NotImplementedError` cho họ mô hình này: LibreYOLO chỉ hỗ trợ inference ảnh ở đây. Xem [dự đoán](/docs/predict) để biết các loại nguồn.

## Biến thể

Có một kích thước edge ở độ phân giải đầu vào cố định, vì vậy việc chọn họ mô hình này thay vì phần còn lại của cấp SAM là quyết định về phần cứng chứ không phải kích thước: EdgeTAM tồn tại riêng cho inference trên thiết bị có tài nguyên hạn chế.

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />
