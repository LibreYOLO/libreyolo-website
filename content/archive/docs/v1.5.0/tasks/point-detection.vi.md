---
title: Phát hiện điểm
seo_title: Phát hiện điểm và đếm trong LibreYOLO
description: >-
  Định vị đối tượng bằng một điểm thay vì hộp trong LibreYOLO. Dự đoán tâm, đếm
  đối tượng, huấn luyện FOMO và đọc các point metric.
lead: >-
  Phát hiện điểm trả về một vị trí x, y trên mỗi đối tượng thay vì bounding box.
  LibreYOLO cung cấp dưới dạng tác vụ point, và một dự đoán mang một dòng x, y,
  lớp đối tượng cùng độ tin cậy cho mỗi đối tượng.
keywords:
  - phát hiện điểm python
  - đếm đối tượng python
  - phát hiện tâm đối tượng
  - định vị điểm FOMO
  - đếm vật thể trong ảnh
  - point localization
last_verified: 1.5.0
snippets:
  predict:
    - label: Dự đoán điểm và đếm
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Trọng số LibreFOMO không được tự động tải. Trước hết hãy tải
        checkpoint từ

        # https://huggingface.co/LibreYOLO rồi nạp bằng đường dẫn cục bộ.

        model = LibreYOLO("./LibreFOMOs-point.pt")

        result = model(SAMPLE_IMAGE, save=True)


        points = result.points

        print(len(points))     # số lượng đối tượng

        print(points.xy)       # tâm (N, 2) theo pixel ảnh gốc

        print(points.cls, points.conf)
    - label: Tọa độ chuẩn hóa và số lượng theo lớp
      language: python
      code: >
        from collections import Counter


        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("./LibreFOMOs-point.pt")

        result = model(SAMPLE_IMAGE)


        points = result.points.numpy()

        print(points.xyn)                          # cùng các tâm đó trong [0,
        1]

        print(Counter(points.cls.astype(int).tolist()))
  train:
    - label: Huấn luyện FOMO trên dataset YOLO
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(data="my-dataset.yaml", epochs=40, batch=32, lr0=3e-4)
    - label: Dự đoán bằng checkpoint đã huấn luyện
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("./LibreFOMOs-point.pt")
        results = model.train(data="my-dataset.yaml", epochs=40)

        # train() nạp lại checkpoint tốt nhất vào cùng đối tượng, vì vậy
        # mô hình dự đoán bằng trọng số đã huấn luyện khi lời gọi trả về.
        print(results["best_checkpoint"])
        print(model(SAMPLE_IMAGE).points.xy)
  val:
    - label: Xác thực và đọc các key metric
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("./LibreFOMOs-point.pt")

        metrics = model.val(data="my-dataset.yaml")


        print(metrics["metrics/precision"], metrics["metrics/recall"])

        print(metrics["metrics/f1"])

        print(metrics["metrics/mAP@[0.01:0.10]"])   # fitness

        print(metrics["metrics/MLE"])               # sai số định vị trung bình

        print(metrics["metrics/MAE"], metrics["metrics/RMSE"])   # sai số số
        lượng
    - label: Thay đổi các ngưỡng khoảng cách
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("./LibreFOMOs-point.pt")


        # Biên lượt quét là một phần văn bản key, vì vậy lượt quét tùy chỉnh

        # đổi tên các key mAP được tạo ra.

        metrics = model.val(data="my-dataset.yaml", dist_thresholds=[0.02,
        0.05])


        print(metrics["metrics/mAP@0.02"])

        print(metrics["metrics/mAP@[0.02:0.05]"])
  export:
    - label: Xuất
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
    - label: Chạy tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, vì vậy artifact đã xuất được nạp
        # như mọi checkpoint khác và trả về cùng một đối tượng Results.
        model = LibreYOLO("./LibreFOMOs-point.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.points.xy)
source_hash: 932153c8870d1c7c
---

## Định nghĩa

Tác vụ `point` định vị mỗi đối tượng bằng một tọa độ x, y và một lớp đối tượng,
không có chiều rộng, chiều cao hoặc mặt nạ. Vì dự đoán là danh sách phẳng các
đối tượng, số dòng chính là số đối tượng, đây là lý do nó trở thành tác vụ đếm.

Dự đoán điền `result.points`, một payload `Points` bọc mảng `(N, 4)` gồm các
dòng `x, y, class, confidence` theo pixel ảnh gốc. `.xy` trả về tọa độ, `.xyn`
trả về cùng tọa độ chia cho kích thước ảnh, `.cls` là index lớp đối tượng và
`.conf` là điểm số; `len()` trả về số điểm. `result.boxes` luôn rỗng, vì vậy
`iou` và `max_det` không có gì để tác động.

## Mô hình

Ba family phục vụ `point` và không thể thay thế lẫn nhau.

[FOMO](/docs/models/fomo) là lựa chọn có từ vựng cố định: grid classifier gán
nhãn từng cell của grid độ phân giải thấp thành background hoặc tâm đối tượng.
Đây là point family duy nhất LibreYOLO có thể huấn luyện và cũng là family duy
nhất có thể xuất.

[LocateAnything](/docs/models/locate-anything) nhận văn bản thay vì index lớp,
vì vậy từ vựng là bất kỳ cụm từ nào bạn viết. Nó cần thành phần bổ sung `vlm`,
được dựng dưới dạng `LibreLocateAnything` thay vì qua factory `LibreYOLO()`, còn
trọng số bị giới hạn cho mục đích phi thương mại. Điều khoản chính xác cùng hai
giấy phép bổ sung mà checkpoint kết hợp nằm trên trang của mô hình.

[SenseNova-Vision](/docs/models/sensenova-vision) thực hiện `point` qua cùng
checkpoint sinh theo prompt dùng cho sáu tác vụ khác, được nạp bằng
`LibreVLM("sensenova-vision", task="point")`. Nó cần thành phần bổ sung
`sensenova`, và mỗi dự đoán là một generation pass trên mô hình 7B, vì vậy độ
trễ trên mỗi ảnh cao hơn đáng kể so với detector chuyên dụng. Trọng số chỉ dùng
cho mục đích phi thương mại; giấy phép nằm trên trang của mô hình.

## Dự đoán

Trọng số LibreFOMO là ngoại lệ duy nhất của cơ chế tải tự động trên trang này.
`LibreYOLO("LibreFOMOs-point.pt")` tìm tệp đó trên ổ đĩa và phát sinh
`ValueError` nêu tên tệp thay vì tải. Trước hết hãy tải checkpoint từ [tổ chức
LibreYOLO](https://huggingface.co/LibreYOLO) trên Hugging Face rồi nạp bằng
đường dẫn cục bộ, hoặc tự huấn luyện.

<code-tabs name="predict" />

Tên tệp phải mang hậu tố tác vụ `-point` để loader nhận dạng.
`predict(..., nms_radius=1)` điều khiển khoảng cách tối thiểu theo cell grid để
hai kết quả phát hiện FOMO cùng được giữ lại. Xem [dự đoán](/docs/predict) để
biết về nguồn, stream và cách xử lý kết quả.

## Định dạng dataset

`point` không có định dạng nhãn riêng. Các point family đọc bố cục phát hiện YOLO
chuẩn và suy ra một tâm từ từng dòng hộp, vì vậy `cx cy` là điểm, còn `w h` chỉ
quyết định dòng có hợp lệ hay không.

```text
dataset/
  data.yaml
  images/
    train/scene.jpg
    val/scene.jpg
  labels/
    train/scene.txt
    val/scene.txt
```

Mỗi tệp nhãn chứa một dòng trên mỗi đối tượng, với tọa độ chuẩn hóa:

```text
<class_id> <cx> <cy> <w> <h>
```

```yaml
path: dataset
train: images/train
val: images/val
nc: 1
names: {0: seedling}
```

Tệp nhãn bị thiếu hoặc rỗng nghĩa là không có đối tượng. Xem [định dạng
dataset](/docs/reference/dataset-formats) để biết hợp đồng đầy đủ.

## Huấn luyện

FOMO là point family duy nhất có implementation huấn luyện. `train()` trên
LocateAnything và SenseNova-Vision phát sinh `NotImplementedError`; hãy tinh
chỉnh chúng ở upstream rồi nạp kết quả.

<code-tabs name="train" />

`imgsz` không phải lựa chọn tự do với FOMO: mặc định theo độ phân giải gốc của
checkpoint đã nạp, còn truyền giá trị khác sẽ phát sinh `ValueError` nêu kích
thước dự kiến. Xem [huấn luyện](/docs/train) để biết về dataset, logger và
multi-GPU, cùng [trang FOMO](/docs/models/fomo) để biết giá trị mặc định của
family.

## Xác thực

`val()` khớp điểm dự đoán với điểm ground truth theo kiểu một-một bằng thuật
toán Hungarian, trên một lượt quét ngưỡng khoảng cách. Ngưỡng là khoảng cách
Euclid trong tọa độ ảnh chuẩn hóa, còn lượt quét mặc định gồm mười giá trị từ
0.01 đến 0.10.

<code-tabs name="val" />

`metrics/precision`, `metrics/recall` và `metrics/f1` được lấy macro-average
trên các lớp tại ngưỡng nghiêm ngặt nhất của lượt quét, mặc định là 0.01.
`metrics/mAP@0.01` là average precision tại cùng ngưỡng, còn
`metrics/mAP@[0.01:0.10]` là trung bình trên toàn lượt quét. Giá trị lượt quét
cũng là `fitness`, con số mà cơ chế chọn checkpoint tốt nhất đọc. Cả hai key mAP
được dựng từ ngưỡng đang dùng, vì vậy truyền `dist_thresholds=` sẽ đổi tên chúng.

`metrics/MLE` là khoảng cách trung bình giữa các cặp khớp tại ngưỡng nghiêm ngặt
nhất, theo cùng đơn vị chuẩn hóa. `metrics/MAE` và `metrics/RMSE` là metric đếm
thay vì định vị: chúng đo chênh lệch theo từng ảnh giữa số điểm dự đoán và ground
truth.

FOMO bổ sung nhóm thứ hai ở cấp grid phía trên các metric này. Nó quét độ tin
cậy cùng `nms_radius` và công bố tổ hợp F1 tốt nhất dưới dạng `metrics/grid_F1`,
`metrics/grid_precision`, `metrics/grid_recall`, `metrics/grid_mean_distance`,
`metrics/grid_TP`, `metrics/grid_FP` và `metrics/grid_FN`, với các cài đặt tạo ra
kết quả nằm dưới `decode/threshold` và `decode/nms_radius`.

## Xuất

FOMO xuất qua đường dẫn dùng chung, còn artifact đã xuất được nạp lại qua
`LibreYOLO()` theo hậu tố tệp, vì vậy tệp `.onnx` hoặc `.engine` hoạt động như
checkpoint và trả về cùng `Results`.

<code-tabs name="export" />

Phạm vi theo định dạng nằm trên [trang FOMO](/docs/models/fomo) và trong [ma
trận xuất đầy đủ](/docs/reference/export-matrix). LocateAnything và
SenseNova-Vision không xuất được: `export()` phát sinh lỗi trên cả hai vì mô
hình sinh không có detection graph có thể trace.
