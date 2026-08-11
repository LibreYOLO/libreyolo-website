---
title: Phát hiện với từ vựng mở
seo_title: Phát hiện với từ vựng mở trong LibreYOLO
description: >-
  Phát hiện đối tượng từ từ vựng văn bản trong LibreYOLO. Nạp Grounding DINO,
  OWLv2, OMDet-Turbo hoặc OV-DEIM qua LibreOpenVocab và đặt lớp đối tượng tại
  runtime.
lead: >-
  Phát hiện với từ vựng mở thay danh sách lớp cố định của checkpoint bằng các từ
  bạn chọn tại thời điểm gọi. Trong LibreYOLO, đây không phải tác vụ riêng: đó
  là tác vụ detect do một tầng mô hình riêng phục vụ, được nạp qua factory
  LibreOpenVocab thay vì LibreYOLO.
keywords:
  - phát hiện từ vựng mở
  - zero shot object detection
  - open set detection
  - grounding dino python
  - owlv2
  - omdet turbo
  - phát hiện bằng text prompt
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-t")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        print(result.names)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Thay từ vựng
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("owlv2-b16")

        # set_classes giữ nguyên: giá trị tồn tại đến lần gọi tiếp theo.
        # Nhãn phải là duy nhất sau khi chuyển chữ thường và bỏ mạo từ.
        model.set_classes(["a red backpack", "traffic cone"])
        result = model.predict(SAMPLE_IMAGE)

        model.set_classes(["bicycle wheel"])
        result = model.predict(SAMPLE_IMAGE)
    - label: Ngưỡng văn bản Grounding DINO
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-b")
        model.set_classes(["remote control", "school bus"])

        # conf lọc theo điểm hộp, text_threshold theo điểm token của cụm từ đã
        # decode. Cả hai mặc định là 0.25 khi không đặt. Chỉ Grounding DINO
        # chấp nhận text_threshold; các mô hình khác phát sinh lỗi.
        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)
source_hash: 17197cf4d80f3d6f
---

## Định nghĩa

Phát hiện với từ vựng mở trả về `Results` phát hiện thông thường: hộp, độ tin
cậy và index lớp đối tượng, với `result.names` ánh xạ các index đó về chuỗi bạn
yêu cầu. Điểm thay đổi là nguồn của danh sách lớp đối tượng. Detector thông
thường được huấn luyện theo tập category cố định và không bao giờ có thể phát ra
category bên ngoài tập đó. Các mô hình này nhận từ vựng dưới dạng văn bản tại
thời điểm inference, vì vậy chỉ cần `set_classes(["forklift", "safety cone"])`
để biến chúng thành các lớp cần phát hiện.

LibreYOLO không có key tác vụ `open-vocabulary`. Các mô hình này khai báo
`SUPPORTED_TASKS = ("detect",)` như mọi detector khác. Điểm khác biệt là đường
dẫn nạp: chúng là snapshot Hugging Face thay vì checkpoint state-dict
LibreYOLO, vì vậy không nằm trong factory `LibreYOLO()` mà được dựng qua
`LibreOpenVocab()`. Factory đó ngang hàng với `LibreSAM()` và `LibreVLM()`,
không thay thế `LibreYOLO()`.

Điểm số là điểm phát hiện thực, không phải caption được tạo rồi parse sau đó.
Mỗi family tính điểm vùng ảnh so với text embedding của từng prompt.

## Mô hình

Bốn family tạo thành tầng này, tất cả chỉ dự đoán. Nạp family bất kỳ bằng alias
qua `LibreOpenVocab`.

[Grounding DINO](/docs/models/grounding-dino) từ IDEA Research, có kích thước
`t` và `b`. Đây là mặc định của tầng và là family duy nhất nhận
`text_threshold`, ngưỡng thứ hai trên điểm token của cụm từ đã decode.

[OWLv2](/docs/models/owlv2) từ Google Research, có kích thước `b16` và `l14`. Nó
tính điểm vùng ảnh so với text embedding từ encoder kiểu CLIP.

[OMDet-Turbo](/docs/models/omdet-turbo) từ Om AI Lab, có một kích thước `t`. Nó
tách embedding lớp khỏi prompt tác vụ ngôn ngữ, và là family duy nhất tại đây
suppression hộp chồng lấn trong post-processing riêng, vì vậy `iou=` có tác dụng.

[OV-DEIM](/docs/models/ov-deim) có kích thước `s`, `m` và `l`, là detector kiểu
DETR khớp query decoder với text embedding từ MobileCLIP text tower đi kèm. Nó
dùng matching một-một với lựa chọn top-K, vì vậy không có NMS ở bất kỳ đâu.

Trọng số OV-DEIM là trường hợp bị hạn chế trong tầng này. Trọng số detector dùng
CC BY-NC 4.0, chỉ cho mục đích phi thương mại. Text tower đi kèm dùng Apple
Machine Learning Research Model license, chỉ cho nghiên cứu. Checkpoint `l` bổ
sung bản tinh chỉnh backbone DINOv3-S theo DINOv3 License của Meta. Cả ba văn
bản giấy phép nằm trong repo trọng số, và thư viện ghi cùng bản tóm tắt khi phân
giải trọng số, trước khi dựng mô hình. Đọc [OV-DEIM](/docs/models/ov-deim) trước
khi triển khai.

Tầng này cần một thành phần bổ sung:

```bash
pip install "libreyolo[openvocab]"
```

Nó bao gồm `transformers` và `timm` cho ba family wrapper, cùng các package
`huggingface_hub`, `safetensors`, `regex` và `ftfy` mà bản port gốc OV-DEIM cần.

Một tầng thứ hai cũng nhận từ vựng văn bản: `LibreVLM()` nạp mô hình thị giác
ngôn ngữ có khả năng sinh, như [Qwen3-VL](/docs/models/qwen3-vl) và
[Florence-2](/docs/models/florence-2), rồi chuyển đầu ra thành cùng `Results`.
Nó dùng chung bề mặt `set_classes()`. Điểm khác biệt nằm ở nguồn tạo hộp: các
family trên trang này là detector phân biệt phát điểm trực tiếp, còn tầng VLM
tạo ra chúng.

## Dự đoán

<code-tabs name="predict" />

`set_classes()` nhận danh sách chuỗi nhãn không rỗng và giữ nguyên cho đến lần
gọi tiếp theo. Nhãn phải là duy nhất sau khi chuyển chữ thường và bỏ mạo từ đầu,
vì vậy `"a bus"` và `"bus"` không thể cùng tồn tại trong một từ vựng. Cụm nhiều
từ là nhãn như mọi nhãn khác, và mỗi family chuyển danh sách thành đầu vào văn
bản riêng trước khi tokenize, vì vậy `"traffic cone"` là query khác với
`"cone"`.

Ba đối số dự đoán hoạt động khác so với detector gốc. `imgsz=` bị từ chối vì
processor quản lý việc đổi kích thước cho các family này. `augment=True` bị từ
chối vì test-time augmentation nằm ngoài phạm vi của tầng. `iou=` chỉ áp dụng
cho family có processor chạy suppression riêng; khi không có gì được
suppression, việc truyền đối số sẽ cảnh báo và bị bỏ qua.

Khi không đặt, `conf` nhận giá trị mặc định riêng của family đã nạp thay vì mức
0.25 thông thường của `predict()`, và giá trị mặc định đó không giống nhau trên
toàn tầng. Hãy đặt tường minh khi so sánh hai family trên cùng ảnh.

`track()` phát sinh lỗi trên toàn tầng. Thay vào đó, hãy chạy `predict()` theo
từng frame. Xem [dự đoán](/docs/predict) để biết về nguồn, stream và cách xử lý
kết quả.

## Huấn luyện

Không family nào trong tầng này huấn luyện bên trong LibreYOLO. `train()` phát
sinh lỗi: hãy tinh chỉnh ở upstream rồi nạp trọng số kết quả. Từ vựng được
truyền cho `set_classes()` là cài đặt duy nhất thay đổi nội dung mô hình đã nạp
phát hiện.

## Xác thực

Tầng này không có validator và `val()` phát sinh lỗi. Xác thực từ vựng mở cần
validator riêng vì validator phát hiện chuẩn truyền tensor ảnh thẳng tới mô
hình, trong khi các family này cần đầu vào có điều kiện văn bản được dựng cùng.

## Xuất

Xuất nằm ngoài phạm vi của tầng và `export()` phát sinh lỗi. Các mô hình này
chạy qua `predict()` trong PyTorch.
