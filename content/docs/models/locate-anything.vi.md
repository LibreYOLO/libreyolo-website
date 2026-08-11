---
title: LocateAnything
families:
  - locateanything
seo_title: 'LocateAnything: phát hiện open-vocabulary và xác định điểm'
description: >-
  Dùng LocateAnything trong LibreYOLO để phát hiện open-vocabulary và xác định
  điểm. Dự đoán với bất kỳ nhãn văn bản nào; không hỗ trợ huấn luyện, đánh giá
  và xuất.
lead: >-
  LocateAnything là mô hình grounding thị giác-ngôn ngữ do NVIDIA phát hành,
  giải mã song song các bounding box và điểm thay vì lần lượt từng token tọa độ.
  LibreYOLO bọc mô hình này thành detector và bộ xác định điểm open-vocabulary:
  bất kỳ danh sách nhãn văn bản nào cũng trở thành tập lớp, không có head cố
  định và không cần tinh chỉnh.
keywords:
  - LocateAnything
  - NVIDIA
  - mô hình thị giác ngôn ngữ
  - phát hiện open-vocabulary
  - phát hiện điểm
  - VLM
  - grounding
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE


        model = LibreLocateAnything(size="3b")


        # Open vocabulary: dùng được mọi từ, không phải class head cố định.
        Thiết lập

        # được giữ cho mọi lời gọi predict()/track() sau đó cho đến khi đặt lại.

        model.set_classes(["person", "bicycle", "dog"])

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Prompt bằng điểm
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        # task="point" trả về một điểm cho mỗi đối tượng khớp thay vì một box.
        # Chuyển tác vụ trên mô hình đã tải bằng model.set_task("point").
        model = LibreLocateAnything(size="3b", task="point")
        model.set_classes(["the person closest to the camera"])
        result = model(SAMPLE_IMAGE, save=True)

        for pt in result.points:
            print(pt.cls, pt.conf, pt.xy)
    - label: Trò chuyện thô
      language: python
      code: >
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE


        model = LibreLocateAnything(size="3b")


        # Lối truy cập linh hoạt bên dưới tiện ích phát hiện: câu hỏi dạng tự
        do,

        # đếm hoặc bất kỳ prompt nào mà wrapper box không bao quát.

        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")

        print(text)
source_hash: 378ea758e507a096
---

## Cài đặt

LocateAnything cần extra `vlm`, extra này cài thêm `transformers` cùng các gói
`decord`, `lmdb` và `peft` mà mã từ xa trên Hugging Face nhập khi tải.

```bash
pip install "libreyolo[vlm]"
```

## Dự đoán

`LibreLocateAnything` là một lớp Python, không phải checkpoint `.pt`: lớp này
không được tải qua factory `LibreYOLO()` và CLI `libreyolo` không phân giải nó.
Factory `LibreVLM(...)` (`from libreyolo import LibreVLM`) cũng truy cập họ này
qua bí danh, chẳng hạn `LibreVLM("locate-anything")`; lớp dùng bên dưới chính là
thứ factory đó khởi tạo. Khi tải, LibreYOLO tải xuống và thực thi mã mô hình từ
xa của NVIDIA trên Hugging Face, vì vậy bản tải được ghim vào một commit revision
cố định thay vì nhánh `main` có thể thay đổi, đồng thời ghi thông báo giấy phép
một lần trước lần tải đầu tiên.

<code-tabs name="predict" />

`result.boxes` (tác vụ `detect`) và `result.points` (tác vụ `point`) chứa
đầu ra đã phân tích như mọi họ khác. Độ tin cậy chỉ là giá trị giữ chỗ:
LocateAnything không phát ra điểm cho từng box, nên mọi phát hiện nhận cùng
một độ tin cậy hằng số và `conf=` chỉ loại các hàng thấp hơn hằng số đó, không
xếp hạng chúng. Nếu bỏ qua `set_classes()`, từ vựng mặc định là các tên COCO-80.
Xem [dự đoán](/docs/predict) để biết về nguồn, streaming và xử lý kết quả.

## Các biến thể

Có một kích thước được công bố là 3b. Hai tác vụ dùng chung trọng số: `detect`
(mặc định) trả về các box, còn `task="point"` trả về một điểm cho mỗi đối tượng
khớp trong `result.points`; chuyển đổi giữa chúng trên mô hình đã tải bằng
`model.set_task("point")`. Hệ thống benchmark của LibreYOLO chưa đo họ này, vì
vậy không có số liệu độ chính xác đã công bố để so sánh.

LibreYOLO chỉ cung cấp họ này để dự đoán. `train()`, `val()` và `export()` đều
phát sinh `NotImplementedError`: thay vào đó, hãy tinh chỉnh ở thượng nguồn rồi
tải kết quả; việc đánh giá tập dữ liệu bị bỏ qua vì độ tin cậy giữ chỗ sẽ khiến
COCO mAP gây hiểu lầm; xuất nằm ngoài phạm vi của mô hình sinh không có state
dict để truy vết.

## Giấy phép

<provenance-box>

NVIDIA License cho phép sử dụng, sao chép và sửa đổi, nhưng giới hạn mô hình và
mọi sản phẩm phái sinh chỉ cho mục đích phi thương mại, nghiên cứu hoặc đánh giá
đối với bất kỳ ai ngoài NVIDIA và các đơn vị liên kết: không có ngưỡng doanh thu
hay ngoại lệ trả phí. LocateAnything-3B còn kết hợp hai thành phần theo giấy phép
khác: backbone ngôn ngữ Qwen2.5-3B-Instruct theo Qwen Research License và bộ mã
hóa thị giác MoonViT-SO-400M theo MIT. LibreYOLO không lưu trữ, tạo bản sao hay
phân phối lại bất kỳ thành phần nào: ở lần chạy đầu tiên, `LibreLocateAnything`
tải trực tiếp trọng số và mã từ xa cần thiết từ `nvidia/LocateAnything-3B` trên
Hugging Face, được ghim vào một commit cố định.

</provenance-box>

## Trích dẫn

<citation-block />

