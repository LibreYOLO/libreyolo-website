---
title: LFM2-VL
families:
  - lfm2vl
seo_title: 'LFM2-VL: phát hiện với từ vựng mở trong LibreYOLO'
description: >-
  Dùng LFM2-VL trong LibreYOLO để phát hiện đối tượng với từ vựng mở trên thiết
  bị. Dự đoán bằng bất kỳ nhãn văn bản nào; không hỗ trợ huấn luyện, xác thực và
  xuất.
lead: >-
  LFM2-VL là vision-language model nhỏ gọn chạy trên thiết bị do Liquid AI phát
  hành. LibreYOLO bọc mô hình thành detector với từ vựng mở: mọi danh sách nhãn
  văn bản đều trở thành tập lớp đối tượng, không có head cố định và không cần
  tinh chỉnh.
keywords:
  - LFM2-VL
  - LFM2
  - Liquid AI
  - vision-language model
  - phát hiện với từ vựng mở
  - VLM
  - VLM thiết bị biên
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE


        model = LibreLFM2VL(size="450m")


        # Từ vựng mở: dùng được mọi từ, không phải class head cố định. Thiết lập

        # được giữ qua mọi lệnh gọi predict()/track() sau đó cho đến khi đặt
        lại.

        model.set_classes(["person", "bicycle", "dog"])

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat thô
      language: python
      code: |
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE

        model = LibreLFM2VL(size="450m")

        # Lối truy cập mức thấp bên dưới tiện ích phát hiện: câu hỏi tự do,
        # đếm hoặc mọi prompt mà wrapper box không bao quát.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 40237f0ecc0d2cd5
---

## Cài đặt

LFM2-VL cần extra `vlm`, extra này kéo về `transformers` cho backbone dùng chat template.

```bash
pip install "libreyolo[vlm]"
```

## Dự đoán

`LibreLFM2VL` là lớp Python, không phải checkpoint `.pt`: lớp không được tải qua factory `LibreYOLO()`, còn CLI `libreyolo` không phân giải nó. Factory `LibreVLM(...)` (`from libreyolo import LibreVLM`) cũng truy cập họ mô hình này qua bí danh, ví dụ `LibreVLM("lfm2-vl-450m")`; lớp dùng bên dưới là lớp mà factory khởi tạo. Trọng số đến từ repo Hugging Face riêng của Liquid AI, không phải mirror LibreYOLO; lệnh gọi đầu tiên tải về và lưu chúng vào bộ nhớ đệm cục bộ, đồng thời ghi log một thông báo giấy phép duy nhất trước khi tải.

<code-tabs name="predict" />

`result.boxes` chứa các detection đã parse như mọi họ mô hình khác. Độ tin cậy là placeholder: LFM2-VL không phát ra điểm theo từng box, vì vậy mọi detection nhận cùng độ tin cậy hằng số, còn `conf=` chỉ loại các hàng dưới hằng số đó chứ không xếp hạng. `iou` loại các box gần trùng của cùng lớp đối tượng khi vượt độ chồng lấp cho trước, một tác dụng phụ khi giải mã greedy lặp lại đối tượng; đây không phải lượt NMS theo lớp đối tượng. Bỏ qua `set_classes()` thì từ vựng mặc định dùng tên COCO-80. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có hai kích thước 450m và 1.6b, cả hai đến từ bản phát hành LFM2.5-VL của Liquid AI và được xây dựng để triển khai trên thiết bị. Bộ công cụ benchmark của LibreYOLO chưa đo họ mô hình này, vì vậy không có số liệu độ chính xác đã công bố để so sánh; hãy chọn kích thước theo ngân sách tính toán của bạn.

LibreYOLO chỉ cung cấp họ mô hình này để dự đoán. `train()`, `val()` và `export()` đều phát sinh `NotImplementedError`: thay vào đó hãy tinh chỉnh ở upstream rồi tải kết quả, việc xác thực tập dữ liệu (dataset) bị bỏ qua vì độ tin cậy placeholder sẽ khiến COCO mAP gây hiểu nhầm, còn xuất nằm ngoài phạm vi của mô hình sinh không có state dict để trace.

## Giấy phép

<provenance-box>

LFM Open License v1.0 cho phép sử dụng thương mại, sao chép và sửa đổi, nhưng chỉ dưới ngưỡng doanh thu hằng năm 10 triệu USD; pháp nhân đạt hoặc vượt ngưỡng đó hoàn toàn không được cấp phép sử dụng thương mại theo thỏa thuận này và phải liên hệ trực tiếp Liquid AI. Các tổ chức phi lợi nhuận đủ điều kiện được miễn ngưỡng cho mục đích phi thương mại hoặc nghiên cứu. LibreYOLO không cung cấp mã nguồn LiquidAI vì mô hình tải qua thư viện `transformers` dùng Apache-2.0, đồng thời không host hoặc phân phối lại trọng số: `LibreLFM2VL` tải trực tiếp kích thước tương ứng từ repo Hugging Face riêng của Liquid AI trong lần chạy đầu tiên và ghi log thông báo một lần trước khi tải.

</provenance-box>

## Trích dẫn

<citation-block />


