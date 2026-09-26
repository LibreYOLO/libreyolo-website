---
title: Ensemble API
seo_title: API LibreEnsemble và các thao tác hợp nhất
description: >-
  LibreEnsemble, ExternalDetector và ba thao tác hợp nhất trong libreyolo.ops:
  weighted boxes fusion, biến thể seeded và hợp nhất NMS nhận biết lớp.
lead: >-
  LibreEnsemble chạy nhiều detector trên cùng một ảnh và hợp nhất các phát hiện
  vào một Results. Quá trình hợp nhất diễn ra sau hậu xử lý riêng của từng thành
  viên, nên các thành viên giữ kích thước đầu vào, chuẩn hóa và bước loại bỏ
  riêng.
keywords:
  - LibreEnsemble
  - weighted boxes fusion
  - wbf
  - ExternalDetector
  - libreyolo.ops.fusion
  - đồng thuận min_votes
last_verified: 1.5.0
verification: >-
  Chữ ký và giá trị mặc định được đọc từ libreyolo/ensemble/model.py và
  libreyolo/ops/fusion.py ở v1.5.0. Mục đích thiết kế lấy từ
  docs/adr/0004-model-ensembling.md.
snippets:
  usage:
    - label: 'Hai thành viên, hợp nhất mặc định'
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])

        # Nguồn một ảnh trả về một Results, không phải danh sách.
        result = ens(SAMPLE_IMAGE, conf=0.25)

        print(result.boxes.xyxy)
        print(result.speed)
    - label: Đồng thuận và ngưỡng theo thành viên
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(
            ["LibreYOLO9t.pt", "LibreYOLO9s.pt"],
            weights=[1.0, 2.0],
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,
        )
        result = ens(SAMPLE_IMAGE, conf=[0.25, 0.4])
        print(len(result))
  ops:
    - label: 'Thao tác hợp nhất, không liên quan mô hình'
      language: python
      code: >
        import torch

        from libreyolo.ops import weighted_boxes_fusion


        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [12.0, 11.0, 51.0,
        49.0]])

        scores = torch.tensor([0.9, 0.8])

        labels = torch.tensor([0, 0])

        model_ids = torch.tensor([0, 1])


        fused = weighted_boxes_fusion(
            boxes, scores, labels, model_ids, num_models=2, iou_thr=0.55
        )

        print(fused)
source_hash: 3834f628efb1193d
---

## LibreEnsemble

```python
LibreEnsemble(
    members,
    *,
    weights=None,
    fusion="wbf",
    fusion_iou=0.55,
    min_votes=1,
)
```

| Đối số | Mặc định | Ý nghĩa |
|---|---|---|
| `members` | | Hai detector trở lên |
| `weights` | `None` | Hệ số tin cậy theo thành viên; tất cả là `1.0` khi bỏ qua |
| `fusion` | `"wbf"` | `"wbf"`, `"wbf_seeded"`, `"nms"` hoặc callable |
| `fusion_iou` | `0.55` | Ngưỡng IoU để gom cụm hợp nhất |
| `min_votes` | `1` | Chỉ giữ box được ít nhất số thành viên này xác nhận |

Thành viên là đường dẫn trọng số được phân giải qua factory `LibreYOLO()`, mô
hình đã khởi tạo, backend đã xuất hoặc `ExternalDetector`. Mọi thành viên phải
là mô hình tác vụ detect.

<code-tabs name="usage" />

Quá trình khởi tạo từ chối khi có ít hơn hai thành viên, danh sách `weights` sai
độ dài, trọng số không dương, `min_votes` không phải số nguyên dương hoặc
`min_votes` lớn hơn số thành viên. `fusion="nms"` với `min_votes > 1` cũng phát
sinh lỗi vì NMS loại thông tin thành viên cụm và không thể đếm phiếu.

`weights` scale mức tin cậy đặt vào từng thành viên. Trọng số cao hơn kéo tọa độ
và điểm đã hợp nhất về phía thành viên đó. Quy ước là đặt chúng tỉ lệ với mAP đánh giá.

## Không gian lớp

Các thành viên có `names` giống nhau đi thẳng qua. Nếu không, không gian lớp được
hợp theo tên, class ID của thành viên được ánh xạ lại qua bảng tra cứu và
`Results.names` sau hợp nhất là tập hợp. Quá trình chỉ hợp nhất box trong cùng lớp
đã hợp, nên lớp chỉ một thành viên biết đi qua mà không hợp nhất. Trạng thái không
khớp ghi cảnh báo khi khởi tạo.

`min_votes` được giới hạn theo từng lớp bằng số không gian nhãn thành viên có
chứa lớp đó, nên đồng thuận vẫn có ý nghĩa trên từ vựng chỉ dùng chung một phần.

## Gọi ensemble

```python
ens(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    output_path=None,
    color_format="auto",
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    **kwargs,
)
```

`predict` là bí danh của `__call__`. Giá trị trả về là `Results` thông thường,
trong đó `speed` phân tách chi phí theo thành viên và thêm mục `fusion`. Nguồn
một ảnh trả về một đối tượng, danh sách hoặc thư mục trả về danh sách, còn
`stream=True` trả về generator.

`conf`, `iou` và `device` được phát rộng đến mọi thành viên, đồng thời chấp nhận
một giá trị cho mỗi thành viên, nên `conf=[0.25, 0.4]` đặt ngưỡng 0.25 cho thành
viên 0 và 0.4 cho thành viên 1. `imgsz` được phát rộng khi là int hoặc tuple và
chỉ áp dụng theo thành viên khi là danh sách, nên `imgsz=(480, 640)` là một kích
thước hình chữ nhật cho mọi thành viên, còn `imgsz=[480, 640]` là 480 cho thành
viên 0 và 640 cho thành viên 1. Mỗi mục phải hợp lệ với họ của thành viên đó.

`augment` được phát rộng đến các thành viên hỗ trợ tăng cường dữ liệu khi kiểm
thử, còn backend đã xuất bỏ qua. `classes` nhận class ID của tập hợp và `max_det`
áp dụng cho kết quả đã hợp nhất, nên thành viên chạy rộng rãi rồi ensemble cắt
bớt một lần. `batch` được chấp nhận để đồng nhất API; ảnh được xử lý tuần tự.

`val()` và `export()` phát sinh `NotImplementedError`. Hãy đánh giá và xuất từng thành viên riêng.

## ExternalDetector

```python
ExternalDetector(fn: Callable, names: dict[int, str])
```

Chuyển bất kỳ callable phát hiện nào thành thành viên. `fn` nhận ảnh PIL và trả
về `(boxes, scores, labels)`, trong đó box là xyxy theo pixel ảnh gốc và nhãn là
class ID hợp lệ trong `names`. Tensor, mảng và danh sách lồng nhau đều hoạt động.
LibreYOLO không import gì từ mã bên ngoài.

Adapter kiểm tra giá trị trả về: phải là tuple 3 phần tử, box phải có shape
`(N, 4)`, ba mảng phải cùng độ dài và mọi class ID phải xuất hiện trong `names`.
Phát hiện bằng hoặc thấp hơn `conf` bị loại trước khi hợp nhất.

## Thao tác hợp nhất

Các primitive hợp nhất là thao tác torch độc lập trong `libreyolo.ops`. Chúng
không phụ thuộc mô hình và có thể được import riêng, vì vậy được xuất tách khỏi ensemble.

<code-tabs name="ops" />

Cả ba nhận cùng đối số vị trí `boxes, scores, labels, model_ids` và trả về `(boxes, scores, labels)`.

| Thao tác | Khóa registry | Hành vi |
|---|---|---|
| `weighted_boxes_fusion` | `wbf` | Weighted boxes fusion tuần tự, trung thành với bài báo |
| `wbf_seeded` | `wbf_seeded` | Biến thể song song một lượt của cùng phép rút gọn |
| `nms_fusion` | `nms` | Nối mọi thứ và áp dụng NMS nhận biết lớp |

`FUSIONS` ánh xạ ba khóa registry đến callable, còn `LibreEnsemble` tra cứu `fusion=` tại đó.

```python
weighted_boxes_fusion(
    boxes, scores, labels, model_ids,
    *,
    weights=None,
    num_models=None,
    iou_thr=0.55,
    skip_box_thr=0.0,
    conf_type="avg",
    min_votes=1,
    models_per_label=None,
    label_weights=None,
)
```

`wbf_seeded` nhận chữ ký giống hệt. `nms_fusion` nhận cùng đối số trừ `conf_type`
và phát sinh `ValueError` khi `min_votes > 1`.

Trong `weighted_boxes_fusion`, các phát hiện được duyệt theo thứ tự độ tin cậy đã
scale theo trọng số giảm dần. Mỗi phát hiện hoặc tham gia cụm hiện có có box hợp
nhất đang chạy chồng lấn tốt nhất ở IoU trên `iou_thr` và cùng nhãn, hoặc bắt đầu
cụm mới. Box hợp nhất của cụm là trung bình tọa độ thành viên theo trọng số độ tin
cậy; điểm là trung bình có trọng số hoặc cực đại của độ tin cậy, được scale lại
để box do ít mô hình xác nhận có điểm thấp hơn.

`wbf_seeded` chọn tâm cụm bằng NMS nhận biết lớp ở `iou_thr`, gán mỗi phát hiện
vào tâm cùng nhãn có IoU tốt nhất rồi rút gọn từng cụm theo cùng cách. Shape cụm
không dịch chuyển giữa lượt, nên toàn bộ thao tác là phép toán tensor shape cố
định. Hai biến thể đồng nhất khi cụm không nhập nhằng và có thể khác nhẹ trên
chuỗi cụm chồng lấn.

`nms_fusion` giữ nguyên box có độ tin cậy cao nhất của mỗi nhóm chồng lấn.
`weights` theo mô hình chỉ scale độ tin cậy cho thứ hạng loại bỏ, còn box sống
sót giữ điểm gốc.

## Hợp nhất tùy chỉnh

`fusion=` cũng chấp nhận callable có cùng chữ ký như các thao tác trên. Tên được
ghi vào `ens.fusion`, hoặc `"custom"` khi không có tên. Giá trị trả về được kiểm
tra: phải là bộ ba `(boxes, scores, labels)` có shape nhất quán.

