---
title: Ngưỡng và lọc
seo_title: 'conf, iou và max_det trong LibreYOLO'
description: >-
  conf, iou, max_det và classes thực sự làm gì khi dự đoán, những họ nào bỏ qua
  iou vì không chạy NMS và vì sao agnostic_nms không có tác dụng.
lead: >-
  Bốn đối số quyết định dự đoán nào được giữ lại: conf, iou, max_det và classes.
  Chỉ hai trong số đó áp dụng cho mọi họ vì set predictor giải mã tập query cố
  định và không bao giờ chạy NMS.
keywords:
  - ngưỡng conf yolo
  - ngưỡng iou nms
  - max_det
  - lọc lớp phát hiện python
  - nms không phân biệt lớp
  - detr không nms
  - ngưỡng độ tin cậy phát hiện
  - lọc lớp khi suy luận
last_verified: 1.5.0
verification: >-
  Giá trị mặc định được trích từ InferenceRunner.__call__ trong
  libreyolo/models/base/inference.py. Hành vi NMS theo họ được đọc từ mọi module
  trong libreyolo/postprocess/ và đối chiếu với _is_nms_free_family trong
  libreyolo/backends/base.py. Cách lọc lớp lấy từ
  InferenceRunner._apply_classes_filter và _wrap_results. Trạng thái
  agnostic_nms lấy từ NOOP_PREDICT_KWARGS trong libreyolo/utils/predict_args.py.
  Cách xử lý open-vocabulary lấy từ NMS_THRESHOLD trong
  libreyolo/models/openvocab/base.py. Giá trị mặc định khi đánh giá lấy từ
  BaseModel.val.
snippets:
  basic:
    - label: Bốn đối số
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(
            SAMPLE_IMAGE,
            conf=0.25,      # giữ dự đoán có điểm bằng hoặc cao hơn mức này
            iou=0.45,       # ngưỡng chồng lấn NMS tại nơi NMS chạy
            max_det=300,    # giới hạn cho mỗi ảnh
            classes=None,   # hoặc danh sách class id
        )
        print(len(result.boxes))
    - label: Quét conf
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        for conf in (0.1, 0.25, 0.5, 0.75):
            result = model(SAMPLE_IMAGE, conf=conf)
            print(conf, len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt conf=0.4 iou=0.5 max_det=100 \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  classes:
    - label: Lọc theo lớp cụ thể
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # Class id lập chỉ mục model.names. Trên COCO, 0 là person.
        result = model(SAMPLE_IMAGE, classes=[0])

        print({result.names[int(c)] for c in result.boxes.cls.tolist()})
    - label: Tìm id theo tên
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        wanted = {"person", "backpack"}
        ids = [i for i, name in result.names.items() if name in wanted]
        print(ids)

        filtered = model(SAMPLE_IMAGE, classes=ids)
        print(len(filtered.boxes))
  nmsfree:
    - label: iou trên họ không chạy NMS
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # RF-DETR giải mã tập query cố định, nên iou không thay đổi gì ở đây.

        model = LibreYOLO("LibreRFDETRs.pt")


        loose = model(SAMPLE_IMAGE, iou=0.9)

        tight = model(SAMPLE_IMAGE, iou=0.1)


        # Số lượng giống nhau trong cả hai trường hợp. conf và max_det là các
        điều khiển có tác dụng.

        print(len(loose.boxes), len(tight.boxes))
source_hash: 0b978963c356027d
---

## Bốn đối số

| Đối số | Mặc định | Áp dụng cho |
|---|---|---|
| `conf` | `0.25` | Mọi họ |
| `iou` | `0.45` | Các họ chạy non-maximum suppression |
| `max_det` | `300` | Mọi họ |
| `classes` | `None` | Mọi họ |

<code-tabs name="basic" />

Hai đối số trong số này áp dụng toàn cục còn hai đối số thì không. Đây là điều
hữu ích nhất cần biết trước khi điều chỉnh bất kỳ thứ gì.

Việc đánh giá cố ý dùng giá trị mặc định khác: `val()` chạy ở `conf=0.001` và
`iou=0.6` vì average precision được tính trên toàn bộ đường cong precision-recall,
còn ngưỡng 0.25 sẽ cắt cụt đường cong.

## conf

`conf` là điểm mà dưới mức đó dự đoán sẽ bị loại. Nó áp dụng cho mọi họ, kể cả
các họ không bao giờ chạy NMS, và là điều khiển đầu tiên cần chỉnh khi có quá
nhiều hoặc quá ít phát hiện.

Giá trị mặc định `0.25` phù hợp để xem ảnh. Khi cung cấp dữ liệu cho hệ thống hạ
nguồn, bạn thường cần giá trị cao hơn; khi đo độ chính xác, cần giá trị thấp hơn nhiều.

## iou

`iou` là mức chồng lấn mà trên mức đó non-maximum suppression loại box có điểm
thấp hơn trong hai box cùng lớp. Đối số chỉ có ý nghĩa nếu họ thực sự chạy bước
loại bỏ.

Set predictor giải mã số query cố định và lấy các query có điểm cao nhất. Bản
trùng lặp bị loại bên trong kiến trúc khi huấn luyện, không phải bằng bước hậu xử
lý, nên không có ngưỡng để điều chỉnh. Các họ sau chấp nhận `iou` để đồng nhất
API rồi bỏ qua nó:

CenterNet, DEIM, DETR, Deformable DETR, D-FINE, DINO-DETR, EdgeCrafter,
Faster R-CNN, LW-DETR, Mask R-CNN, RF-DETR, RT-DETR và head YOLOv9 đầu cuối.
Các biến thể xây dựng trên những decoder đó kế thừa hành vi này.

<code-tabs name="nmsfree" />

Hầu hết các họ nêu điều này trong docstring hậu xử lý, nhưng runtime không đưa
ra cảnh báo, nên quét `iou` trên RF-DETR tạo ra đường phẳng thay vì lỗi. Faster
R-CNN và Mask R-CNN là trường hợp hơi khác: cả hai đã chạy NMS bên trong mô hình
ở ngưỡng thượng nguồn cố định mà `iou` không có cách được hỗ trợ để thay đổi.

Các họ sau có dùng đối số này: YOLOv1 đến YOLOv4, YOLOv7, YOLOv9, YOLOX,
YOLO-NAS, RTMDet, PicoDet, EfficientDet, FCOS, RetinaNet và SSD.

Hai tùy chọn khi dự đoán khiến `iou` có ý nghĩa ngay cả với set predictor vì cả
hai đều hợp nhất box sau khi mô hình hoàn tất:

- `tiling=True` điều hòa các tile chồng lấn bằng NMS theo lớp ở `iou`
- `augment=True` hợp nhất các góc nhìn đã lật bằng NMS theo lớp ở `iou`

Cả hai được trình bày trong [Hiệu năng suy luận](/docs/predict/performance).

Detector open-vocabulary có quy tắc riêng. Họ có processor chạy NMS sẽ khai báo
ngưỡng mặc định riêng và tuân theo `iou`, như trường hợp OMDet-Turbo. Các họ
không loại gì gồm Grounding DINO, OWLv2 và OV-DEIM sẽ phát cảnh báo khi truyền
`iou`. Đây là cảnh báo duy nhất thuộc loại này trong thư viện.

## max_det

`max_det` giới hạn số dự đoán trả về cho một ảnh. Nó áp dụng ở mọi nơi nhưng qua
các cơ chế khác nhau: họ NMS cắt bớt sau bước loại bỏ, còn set predictor dùng nó
làm kích thước lựa chọn top-k.

Một số họ giới hạn thấp hơn giá trị bạn yêu cầu vì cấu hình tham chiếu thượng
nguồn làm như vậy. SSD giới hạn ở 200, phân đoạn instance RTMDet ở 100, còn FCOS
ở giới hạn phát hiện riêng cho mỗi ảnh. Tăng `max_det` vượt các mức đó không có
tác dụng.

Nơi duy nhất `max_det` được áp dụng tập trung thay vì theo từng họ là suy luận
chia tile, nơi danh sách đã hợp nhất được cắt bớt sau khi điều hòa các tile.

## Lọc lớp

<code-tabs name="classes" />

`classes` nhận danh sách class id và chỉ giữ các dự đoán có lớp nằm trong danh
sách. ID lập chỉ mục `result.names`, và cách chắc chắn nhất để lấy ID là đọc
`names` từ kết quả thay vì giả định thứ tự tập dữ liệu.

Việc lọc diễn ra tập trung sau hậu xử lý của từng họ, trong một luồng duy nhất mà
mọi đường dự đoán đều đi qua. Điều này có hai hệ quả đáng biết. Cách lọc hoạt
động trên mọi họ, kể cả họ không có NMS. Nó cũng lọc các payload căn chỉnh với
box, nên mask, keypoint và box định hướng được cắt giảm cùng nhau thay vì bị lệch.

Trên dòng lệnh, `classes` chấp nhận một số nguyên đơn, danh sách hoặc chuỗi phân
tách bằng dấu phẩy:

```bash
libreyolo predict model=LibreYOLO9s.pt classes=0 source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
libreyolo predict model=LibreYOLO9s.pt classes="[0,2,5]" source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
```

Lọc không tự tạo thêm độ chính xác. Mô hình vẫn tiêu tốn ngân sách để dự đoán các
lớp mà sau đó bạn loại, còn `max_det` được họ áp dụng trước bộ lọc, nên ảnh chứa
nhiều lớp không mong muốn có thể chạm giới hạn trước khi đến lớp của bạn. Hãy
giảm `conf` hoặc tăng `max_det` nếu điều đó xảy ra.

## agnostic_nms

`agnostic_nms` được chấp nhận nhưng không làm gì. Việc truyền nó phát cảnh báo
rằng đây là no-op để tương thích dòng lệnh, rồi đối số bị loại bỏ.

Không có chế độ loại bỏ không phân biệt lớp. Mọi lời gọi NMS trong thư viện đều
nhận biết lớp, nên hai box chồng lấn thuộc hai lớp khác nhau đều được giữ ở mọi
`iou`. Khi điều này gây vấn đề, trước tiên hãy lọc bằng `classes` hoặc tự loại bỏ
xuyên lớp trên `result.boxes`.

## Những gì predict từ chối

Hai đối số phát sinh lỗi thay vì cảnh báo: `visualize` và `embed` đều phát sinh
`NotImplementedError`. Để tạo embedding, hãy tải mô hình với `task="embed"` rồi
gọi `predict` hoặc `embed` như bình thường.

Bất kỳ tùy chọn nào không được nhận diện đều phát sinh `TypeError` nêu các tùy
chọn được hỗ trợ, nên lỗi chính tả thất bại ngay thay vì bị âm thầm bỏ qua.

Các tùy chọn sau được chấp nhận, cảnh báo rồi loại bỏ: `agnostic_nms`, `boxes`,
`dnn`, `half`, `line_width`, `retina_masks`, `show_conf`, `show_labels` và
`verbose`.

