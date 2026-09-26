---
title: SAM
families:
  - sam
seo_title: 'SAM (Segment Anything): dự đoán mask trong LibreYOLO'
description: >-
  Dùng SAM trong LibreYOLO để phân đoạn bằng prompt điểm và box. Cài đặt và dự
  đoán với các checkpoint base, large và huge theo Apache-2.0.
lead: >-
  SAM (Segment Anything) biến một lần nhấp vào điểm hoặc box thành mask đối
  tượng. LibreYOLO tải mô hình qua factory LibreSAM chuyên dụng, tách biệt với
  factory detector LibreYOLO(), vì mô hình dùng prompt cần một dạng lời gọi
  khác.
keywords:
  - SAM
  - Segment Anything
  - phân đoạn bằng prompt
  - phân đoạn tương tác
  - prompt điểm
  - prompt box
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompt bằng điểm và box
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # "base" tự động tải facebook/sam-vit-base ở lần dùng đầu tiên.
        # Các kích thước khác: "large", "huge" (cũng có thể dùng "b"/"l"/"h").
        model = LibreSAM("base")

        # Prompt điểm: [x, y] theo tọa độ pixel, nhãn 1 = tiền cảnh.
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # đa giác cho mỗi mask
        print(result.boxes.xyxy)    # box khít suy ra từ mask

        # Dùng prompt box thay cho điểm.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])

        # Không có prompt sẽ phân đoạn toàn bộ ảnh (một trình tạo mask tự động
        # đơn giản hóa, không phải bản tham chiếu vét cạn).
        result = model.predict(SAMPLE_IMAGE)
    - label: 'Mã hóa một lần, dùng nhiều prompt'
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        model = LibreSAM("base")


        # Bộ mã hóa ảnh là phần tốn kém. set_image() chạy nó một lần;

        # mỗi lời gọi predict() sau đó dùng lại embedding đã lưu trong bộ nhớ
        đệm.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: f8904d241ef8a929
---

## Cài đặt

SAM cần extra `sam`, extra này sẽ cài thêm `transformers` và `timm`.

```bash
pip install "libreyolo[sam]"
```

## Dự đoán

`LibreSAM(...)` là một điểm vào riêng, khác với `LibreYOLO(...)`: nó trả về một
mô hình phân đoạn dùng prompt thay vì detector, vì forward pass ở đây không có
ý nghĩa nếu thiếu prompt không gian. Không có lệnh CLI `libreyolo predict` cho
họ này; hãy dùng Python API.

<code-tabs name="predict" />

Prompt điểm nhận `[x, y]` cho một đối tượng, `[[x, y], ...]` cho nhiều đối tượng
hoặc mảng numpy; `labels` đánh dấu từng điểm là `1` (tiền cảnh) hoặc `0` (hậu cảnh)
và mặc định tất cả đều là tiền cảnh. Prompt box nhận `[x1, y1, x2, y2]` hoặc danh
sách các box, mỗi box cho một mask. Nếu bỏ cả hai prompt, mô hình phân đoạn toàn bộ
ảnh bằng cách đặt prompt trên một lưới dày và giữ lại các mask có độ tin cậy cao,
không chồng lấn; chế độ "phân đoạn mọi thứ" này được đơn giản hóa so với trình tạo
mask tự động tham chiếu và có thể phân đoạn thiếu trong cảnh đông đúc, vì vậy prompt
điểm hoặc box thực sự là cách chính xác hơn. `conf` lọc theo chất lượng mask dự đoán
(IoU), không phải độ tin cậy phát hiện: truyền `0.0` để giữ mọi ứng viên.
`multimask=True` trả về cả ba mask biểu diễn sự nhập nhằng toàn thể so với bộ phận
của SAM cho mỗi prompt thay vì chỉ mask tốt nhất. `device=` chuyển mô hình và, nếu
đang có phiên `set_image()`, cả embedding đã lưu trong bộ nhớ đệm. Mỗi mask mang
class id `0`, tên `"object"`, vì mask dùng prompt không có tập lớp cố định.
`train()`, `val()`, `export()` và `track()` đều phát sinh `NotImplementedError`
cho họ này: SAM chỉ hỗ trợ dự đoán trong LibreYOLO và việc theo dõi video nằm ngoài
phạm vi. Xem [dự đoán](/docs/predict) để biết các loại nguồn.

## Các biến thể

Ba kích thước bộ mã hóa ảnh ViT là base, large và huge, tất cả có đầu vào cố định
1024 px. Chưa có benchmark về độ chính xác hoặc độ trễ được công bố cho họ này, vì
vậy việc chọn kích thước là sự đánh đổi trực tiếp giữa trọng lượng bộ mã hóa và chất
lượng mask: base mã hóa nhanh nhất, huge nặng nhất.

## Giấy phép

<provenance-box>

LibreYOLO không lưu trữ bản sao riêng của trọng số SAM-1. `LibreSAM("base")`,
`"large"` và `"huge"` tải trực tiếp từ các repo `facebook/sam-vit-base`,
`facebook/sam-vit-large` và `facebook/sam-vit-huge` của Meta trên Hugging Face;
mỗi repo được gắn giấy phép Apache-2.0 tại đó, độc lập với LibreYOLO.

</provenance-box>

## Trích dẫn

<citation-block />

