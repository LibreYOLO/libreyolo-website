---
title: Khái niệm cốt lõi
seo_title: Các khái niệm cốt lõi của LibreYOLO
description: >-
  Cách tác vụ, họ mô hình, kích thước và tên file checkpoint liên hệ với nhau
  trong LibreYOLO, cùng cam kết của từng cấp hỗ trợ.
lead: >-
  Bốn khái niệm mô tả mọi mô hình trong LibreYOLO: tác vụ nó thực hiện, họ mà nó
  thuộc về, kích thước trong họ đó và cấp hỗ trợ của họ. Tên file checkpoint mã
  hóa ba khái niệm đầu.
keywords:
  - khái niệm libreyolo
  - tác vụ libreyolo
  - họ mô hình libreyolo
  - quy tắc đặt tên checkpoint libreyolo
  - cấp hỗ trợ libreyolo
last_verified: 1.5.0
meta:
  - label: Schema tên file
    value: 'Libre<FAMILY><size>[-<task>].pt'
    mono: true
  - label: Tác vụ chuẩn
    value: 17
  - label: Cấp hỗ trợ
    value: 'Flagship, Core, Supported, Chỉ inference, Museum, Cấp sibling'
snippets:
  inspect:
    - label: Liệt kê các họ
      language: bash
      code: |
        # Tác vụ, kích thước và độ phân giải đầu vào của mọi họ đã đăng ký.
        libreyolo models
    - label: Một mô hình
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        print(model.family, model.size, model.task)
        print(model.input_size)
        print(model.nb_classes, model.names[0])
    - label: Chọn tác vụ
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Alias được chuẩn hóa ở biên API: "keypoints" phân giải thành

        # "pose", "det" thành "detect", "semantic-segmentation" thành
        "semantic".

        model = LibreYOLO("LibreYOLO9t.pt", task="det")

        print(model.task)
source_hash: 23d045463a6a8411
---

## Tác vụ

Tác vụ là thứ mô hình trả về. LibreYOLO có mười bảy tên tác vụ chuẩn, và mỗi
tên tương ứng với trường trên object `Results` chứa đầu ra của tác vụ đó.

| Tác vụ | Trả về |
|---|---|
| `detect` | Box thẳng trục kèm lớp và độ tin cậy |
| `segment` | Mask theo instance, một mask cho mỗi object được phát hiện |
| `semantic` | Một nhãn lớp cho mỗi pixel, không tách instance |
| `panoptic` | Một nhãn không chồng lấp cho mỗi pixel, kết hợp vật thể đếm được với vùng vô định hình |
| `pose` | Keypoint theo instance, các hàng căn theo box |
| `classify` | Xác suất trên một tập nhãn cho toàn ảnh |
| `obb` | Box có hướng, kèm góc xoay |
| `point` | Một tọa độ ảnh cho mỗi detection thay vì box |
| `depth` | Bản đồ độ sâu nghịch đảo tương đối dày đặc |
| `normal` | Trường vector đơn vị pháp tuyến bề mặt dày đặc |
| `edge` | Bản đồ xác suất cạnh dày đặc |
| `restore` | Ảnh RGB được khôi phục để khử nhòe, khử nhiễu hoặc tăng độ phân giải |
| `matte` | Bản đồ tiền cảnh mềm từ 0 đến 1 để xóa nền |
| `ocr` | Tứ giác văn bản kèm nội dung nhận dạng, theo thứ tự đọc |
| `embed` | Vector chuẩn hóa L2 có tích vô hướng đo mức tương đồng |
| `gaze` | Hướng nhìn cho mỗi khuôn mặt được phát hiện |
| `mesh` | Cơ thể 3D đã tạo dáng cho mỗi người được phát hiện |

Đây là các tên xuất hiện trong metadata checkpoint và tên file. Các alias quen
thuộc được chấp nhận ở mọi nơi có thể truyền tác vụ và được chuẩn hóa trước mọi
xử lý khác: `detection` và `det` thành `detect`, `keypoints` thành `pose`, `cls`
thành `classify`, `deblur`, `denoise` và `super-resolution` đều thành `restore`,
`face-recognition` và `reid` thành `embed`. Tên không được nhận diện sẽ báo lỗi
thay vì âm thầm dùng giá trị mặc định.

`segment`, `semantic` và `panoptic` là ba tác vụ khác nhau, không phải ba cách
gọi của một tác vụ. Mask instance, nhãn theo pixel và bản đồ kết hợp
thing-plus-stuff có ground truth, metric và trường kết quả khác nhau.

## Họ mô hình

Một họ là một dòng kiến trúc có mã nạp, tiền xử lý và hậu xử lý riêng. Mỗi họ
khai báo định danh `FAMILY` như `yolo9`, `rfdetr` hoặc `dfine`, các tác vụ được
hỗ trợ và độ phân giải đầu vào cho từng kích thước được phát hành.

`LibreYOLO()` là factory chứ không phải lớp. Khi nhận đường dẫn, nó nạp file,
xác định họ từ metadata checkpoint hoặc nếu không có thì từ chính các khóa
tensor, rồi trả về instance mô hình của họ đó. Vì vậy đổi detector chỉ cần thay
một dòng: object trả về cung cấp cùng giao diện `predict`, `train`, `val` và
`export`, đồng thời trả về cùng kiểu `Results`.

<code-tabs name="inspect" />

Một họ phục vụ nhiều tác vụ thường công bố checkpoint riêng cho từng tác vụ,
thường với tập kích thước khác nhau; một số ít dùng chung một artifact cho hai
tác vụ runtime. Dù theo cách nào, danh sách tác vụ hỗ trợ vẫn cố định, và yêu
cầu tác vụ nằm ngoài danh sách sẽ báo lỗi kèm danh sách hỗ trợ thay vì nạp một
phương án gần đúng.

Danh sách đầy đủ cùng benchmark theo họ và trọng số đã công bố nằm tại [tất cả
mô hình](/docs/models).

## Kích thước

Kích thước là một biến thể trong họ, được viết bằng mã chữ thường gắn trực tiếp
vào prefix của họ. Các chữ thường gặp là `n` cho nano, `t` cho tiny, `s` cho
small, `m` cho medium, `l` cho large và `x` cho xlarge, nhưng mã tùy theo họ và
nhiều họ dùng cách hoàn toàn khác: mã theo tên backbone như `r50` hoặc `r101`
khi kích thước là độ sâu ResNet, mã compound scaling như `b0` đến `b3`, hoặc
một tên xác định checkpoint duy nhất được phát hành. YOLOv9 dùng `c` cho compact
trong khi các họ khác dùng `l`.

Kích thước cũng cố định độ phân giải đầu vào, và với các họ có nhiều tác vụ,
độ phân giải có thể khác nhau theo tác vụ. Cả hai đều được đọc từ họ, không bao
giờ được giả định; `libreyolo models` in chúng ra.

## Tên file checkpoint

Mọi file trọng số được công bố tuân theo một schema:

```text
Libre<FAMILY><size>[-<task>].pt
```

Prefix họ là chuỗi cố định theo từng họ, kích thước viết thường và gắn liền
không có dấu phân cách, còn hậu tố tác vụ có dấu gạch nối phía trước. Detection
không có hậu tố theo quy ước lâu đời của checkpoint YOLO, nên
`LibreYOLO9t.pt` là detector và `LibreRFDETRn-seg.pt` là mô hình segmentation
của cùng họ.

| Tác vụ | Hậu tố |
|---|---|
| `detect` | |
| `segment` | `-seg` |
| `semantic` | `-sem` |
| `panoptic` | `-panoptic` |
| `pose` | `-pose` |
| `classify` | `-cls` |
| `gaze` | `-gaze` |
| `obb` | `-obb` |
| `point` | `-point` |
| `depth` | `-depth` |
| `edge` | `-edge` |
| `normal` | `-normal` |
| `restore` | `-restore` |
| `matte` | `-matte` |
| `ocr` | `-ocr` |
| `embed` | `-embed` |
| `mesh` | `-mesh` |

Một họ không có tác vụ thiếu hậu tố có thể yêu cầu hậu tố, vì vậy tên không có
hậu tố không được chấp nhận là checkpoint hợp lệ của họ. Nếu một họ công bố
trọng số được huấn luyện trên dataset khác mặc định, tên dataset được nối thêm
làm hậu tố và biến thể đó vẫn thuộc tên repo nơi file được tải xuống.

Ba cấp nằm ngoài schema này. Các họ promptable segmentation, họ
vision-language và detector open-vocabulary không được đăng ký vào factory
checkpoint và không tạo file `Libre<FAMILY><size>.pt`. Thay vào đó, prefix của
chúng chỉ snapshot Hugging Face đã tải hoặc checkpoint promptable, và cách viết
hoa thương hiệu upstream được cố ý giữ nguyên.

## Cách xác định tác vụ

Khi nhiều tín hiệu có thể chỉ định tác vụ, chúng được xét theo thứ tự cố định
và tín hiệu đầu tiên có mặt sẽ thắng: đối số `task` bạn truyền, tiếp theo là tác
vụ ghi trong metadata checkpoint, hậu tố tác vụ trong tên file, rồi tác vụ mặc
định của họ. Kết quả được kiểm tra với các tác vụ họ hỗ trợ trước khi dựng mô
hình, nên sự không khớp thất bại lúc nạp thay vì tạo đầu ra sai về sau.

## Cấp hỗ trợ

Mỗi họ được xếp vào đúng một cấp. Cấp thể hiện mức độ ưu tiên kỹ thuật, không
phải độ chính xác: nó cho biết tính năng mới xuất hiện ở đâu trước và phần nào
được duy trì ổn định.

| Cấp | Ý nghĩa |
|---|---|
| Flagship | Tính năng được thiết kế và xác thực GPU đầy đủ tại đây trước |
| Core | Detector cốt lõi có thể huấn luyện. Tính năng theo sau flagship trong cùng đợt phát hành |
| Supported | Các họ hỗ trợ có thể huấn luyện. Được duy trì ổn định trong CI, tính năng bổ sung khi thích hợp |
| Chỉ inference | Predict, validate và export. Tính năng huấn luyện không áp dụng |
| Museum | Hiện vật đóng băng. Chỉ sửa bug |
| Cấp sibling | Bề mặt sản phẩm riêng với factory và contract riêng |

Mỗi trang mô hình hiển thị cấp của họ ở phần đầu. Hai họ flagship là
[YOLOv9](/docs/models/yolov9) cho detector CNN và
[RF-DETR](/docs/models/rf-detr) cho detector transformer; hãy bắt đầu từ đó
trừ khi bạn có lý do chọn khác.

Chỉ inference cho biết phần còn thiếu là vòng lặp huấn luyện trong LibreYOLO.
Predict, validate và export ở nơi họ hỗ trợ đều hoạt động. Gọi `train()` trên
họ như vậy sẽ phát `NotImplementedError` kèm lý do.
