---
title: SAM 3
families:
  - sam3
seo_title: 'SAM 3: phân đoạn bằng prompt và khái niệm trong LibreYOLO'
description: >-
  Dùng SAM 3 trong LibreYOLO để phân đoạn bằng điểm, box và khái niệm văn bản.
  Cài đặt và dự đoán với checkpoint large bị giới hạn truy cập theo SAM License
  của Meta.
lead: >-
  SAM 3 mở rộng SAM bằng prompt khái niệm văn bản bên cạnh các điểm và box thông
  thường, vì vậy một cụm từ như "xe buýt trường học màu vàng" sẽ trả về mọi
  instance khớp. LibreYOLO hỗ trợ luồng ảnh của mô hình qua factory LibreSAM
  chuyên dụng, tách biệt với factory detector LibreYOLO().
keywords:
  - SAM 3
  - Segment Anything
  - phân đoạn bằng prompt
  - phân đoạn theo khái niệm
  - prompt văn bản
  - prompt điểm
  - prompt box
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompt bằng điểm và box
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # "sam3" là kích thước duy nhất ("large"); bí danh: "sam3", "sam-3",
        "sam3-large".

        model = LibreSAM("sam3")


        # Prompt điểm: [x, y] theo tọa độ pixel, nhãn 1 = tiền cảnh.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # đa giác cho mỗi mask

        print(result.boxes.xyxy)    # box khít suy ra từ mask


        # Dùng prompt box thay cho điểm.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: Prompt văn bản (khái niệm)
      language: python
      code: |
        from libreyolo import LibreSAM3, SAMPLE_IMAGE

        model = LibreSAM3("large")

        # Tìm mọi instance khớp với cụm từ, không chỉ một đối tượng.
        # text= loại trừ lẫn nhau với points, bboxes, labels và masks.
        result = model.predict(SAMPLE_IMAGE, text="a person")
        print(result.names)         # {0: "a person"}
        print(result.boxes.conf)    # điểm phát hiện PCS cho mỗi instance
    - label: 'Mã hóa một lần, dùng nhiều prompt'
      language: python
      code: >
        from libreyolo import LibreSAM3, SAMPLE_IMAGE


        model = LibreSAM3("large")


        # Bộ mã hóa ảnh là phần tốn kém. set_image() chạy nó một lần;

        # mỗi lời gọi predict() sau đó dùng lại embedding đã lưu trong bộ nhớ
        đệm.

        # Lời gọi text= mã hóa lại ở bên trong vì tracker và bộ mã hóa phân đoạn

        # theo khái niệm không dùng chung bộ nhớ đệm.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: c4fb6d5a622f99ff
---

## Cài đặt

SAM 3 cần extra `sam`, extra này sẽ cài thêm `transformers` và `timm`.

```bash
pip install "libreyolo[sam]"
```

Trọng số bị giới hạn truy cập: hãy truy cập
[huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), chấp nhận
SAM License của Meta, sau đó chạy `hf auth login` (hoặc đặt `HF_TOKEN`) trước
lần tải đầu tiên. LibreYOLO ghi thông báo giấy phép ở lần đầu tải họ này.

## Dự đoán

`LibreSAM(...)` (hoặc `LibreSAM3(...)` dành riêng cho họ) là một điểm vào riêng,
khác với `LibreYOLO(...)`: nó trả về mô hình phân đoạn dùng prompt thay vì detector,
vì forward pass ở đây không có ý nghĩa nếu thiếu prompt. Không có lệnh CLI
`libreyolo predict` cho họ này; hãy dùng Python API. Chỉ hỗ trợ suy luận ảnh;
các mô hình video của SAM 3 nằm ngoài phạm vi ở đây.

<code-tabs name="predict" />

Luồng điểm và box khớp với phần còn lại của họ SAM: prompt điểm nhận `[x, y]`
cho một đối tượng hoặc `[[x, y], ...]` cho nhiều đối tượng, `labels` đánh dấu
mỗi điểm là `1` (tiền cảnh) hoặc `0` (hậu cảnh), còn prompt box nhận
`[x1, y1, x2, y2]` hoặc danh sách các box. Trên luồng này, `conf` lọc theo chất
lượng mask dự đoán (IoU), không phải độ tin cậy phát hiện.

Luồng `text=` là phần bổ sung của SAM 3: chuỗi khái niệm trả về mọi instance khớp
trong ảnh qua Promptable Concept Segmentation và không thể kết hợp với điểm, box,
nhãn hoặc mask. Ở đây, `conf` là điểm phát hiện PCS thay vì IoU của mask; giữ giá
trị mặc định sẽ áp dụng ngưỡng 0.3 của chính mô hình, còn `conf=0.0` giữ mọi ứng
viên. `names` trả về ánh xạ class id `0` đến chuỗi khái niệm được yêu cầu, vì mask
dùng prompt không có tập lớp cố định. `device=` chuyển mô hình và, nếu đang có
phiên `set_image()`, cả embedding đã lưu trong bộ nhớ đệm. `train()`, `val()`,
`export()` và `track()` đều phát sinh `NotImplementedError` cho họ này: SAM 3 chỉ
hỗ trợ dự đoán trong LibreYOLO và theo dõi video nằm ngoài phạm vi. Xem
[dự đoán](/docs/predict) để biết các loại nguồn.

## Các biến thể

Có một kích thước là large, với đầu vào cố định 1008 px. SAM 3.1 không được hỗ
trợ: bản triển khai của nó mang giấy phép tùy chỉnh không thể được đưa vào repo
MIT này, và phiên bản Transformers mà LibreYOLO phụ thuộc chưa tải được định
dạng checkpoint của nó.

## Giấy phép

<provenance-box>

LibreYOLO không lưu trữ bản sao riêng của trọng số SAM 3 và không phân phối lại
chúng. `LibreSAM("sam3")` tải trực tiếp từ repo `facebook/sam3` bị giới hạn truy
cập của Meta trên Hugging Face, nơi yêu cầu chấp nhận SAM License của Meta và
xác thực trước lần tải đầu tiên.

</provenance-box>

## Trích dẫn

<citation-block />

