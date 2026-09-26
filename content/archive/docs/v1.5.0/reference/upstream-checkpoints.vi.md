---
title: Checkpoint thượng nguồn
seo_title: Tải checkpoint thượng nguồn trong LibreYOLO
description: >-
  Cách tự động chuyển đổi biến checkpoint thượng nguồn đã phát hành thành
  checkpoint LibreYOLO v1.0: các bố cục được mở, họ nào nhận diện dạng nào và
  giới hạn của quá trình.
lead: >-
  Các họ LibreYOLO được chuyển từ dự án thượng nguồn có checkpoint phát hành gần
  như tải được nhưng không chứa metadata LibreYOLO. Tự động chuyển đổi nhận diện
  các tệp đó, bọc chúng trong schema v1.0 và ghi kết quả cạnh nguồn.
keywords:
  - libreyolo autoconvert
  - tải checkpoint thượng nguồn
  - convert_upstream_state_dict
  - trọng số thượng nguồn libreyolo
  - chuyển đổi checkpoint
last_verified: 1.5.0
verification: >-
  Hành vi được đọc từ libreyolo/models/autoconvert.py và
  BaseModel.convert_upstream_state_dict; bộ nhận diện theo họ được kiểm tra bằng
  cách đọc phần ghi đè convert_upstream_state_dict của từng họ, tất cả ở v1.5.0.
  Quy tắc RF-DETR COCO lấy từ docs/checkpoint_schema.md.
snippets:
  usage:
    - label: Chỉ cần truyền tệp vào factory
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Tệp thượng nguồn được nhận diện sẽ được chuyển đổi khi tải, và
        # checkpoint đã chuyển đổi được ghi cạnh tệp đó.
        # model = LibreYOLO("yolov9-t-converted.pt")

        # Mọi checkpoint LibreYOLO đều được tải không thay đổi.
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model.family, model.size, model.task, model.nb_classes)
source_hash: c6022771a2a207a1
---

## Điều xảy ra khi tải

Khi `LibreYOLO()` gặp tệp `.pt` chưa phải checkpoint v1.0 hoàn chỉnh, nó gọi
trình tự động chuyển đổi, trình này:

1. mở dict tensor khỏi các bố cục thượng nguồn phổ biến;
2. hỏi mọi họ đã đăng ký xem có nhận diện bố cục không, ánh xạ lại khóa khi cách
   đặt tên thượng nguồn khác bản chuyển đổi native;
3. bọc kết quả thắng trong checkpoint metadata v1.0 nghiêm ngặt, đọc kích thước,
   tác vụ và số lớp từ chính tensor để checkpoint đã tinh chỉnh được chuyển đổi đúng;
4. ghi cạnh nguồn dưới dạng `<source>-<Prefix><size>[-task].pt` và trả về đường
   dẫn để factory tải bình thường.

Người gọi không cần cung cấp gì thêm. Tệp không được họ nào nhận sẽ không trả về
gì và factory báo không thể tải.

<code-tabs name="usage" />

## Các bố cục được mở

Dict tensor được tìm theo thứ tự ưu tiên sau, EMA trước, và từng ứng viên được thử
đến khi một mục thực sự chứa tensor. Vì vậy, block EMA rỗng hoặc chỉ có metadata
không che trọng số hợp lệ bên dưới.

| Khóa | Ghi chú |
|---|---|
| `ema.module` | Wrapper EMA phổ biến |
| `ema` | Wrapper EMA phẳng cũ lưu tensor trực tiếp |
| `ema_state_dict` | Mục có tiền tố `module.` được loại tiền tố |
| `params_ema` | |
| `params` | |
| `ema_net` | |
| `net` | |
| `model` | |
| `state_dict` | |
| Chính tệp | State dict thông thường |

Sau đó, mỗi ứng viên được thu hẹp còn các mục có giá trị tensor và chuẩn hóa:
tiền tố đầu `module.` hoặc `_orig_mod.` bị loại, còn dict có mọi khóa bắt đầu
bằng `model.model.` được loại tiền tố đó.

## Họ nào nhận diện dạng nào

Nhận diện là classmethod theo từng họ. Bản triển khai mặc định nhận bố cục có các
khóa đã khớp bản chuyển đổi native. Họ có cách đặt tên khóa thượng nguồn khác sẽ
ghi đè bằng phép ánh xạ lại và không trả về gì cho bố cục không nhận diện.

Các họ phân phối bộ nhận diện ánh xạ lại: `centernet`, `deeplabv3`,
`deformable_detr`, `dexined`, `moge2`, `picodet`, `rtdetr`, `rtdetrv2`,
`rtdetrv4`, `rtmdet`, `segformer`, `swin`, `teed`, `yolo7`, `yolo9`,
`yolo9_e2e`, `yolo9_p2`.

Các họ từ chối hoàn toàn tự động chuyển đổi: `efficientdet`, `eomt` và `pidnet`
không trả về gì từ bộ nhận diện, nên tệp thượng nguồn của chúng đi qua script
chuyển đổi thay thế. `l2cs` bị loại khỏi bộ nhận diện chung vì chỉ dành cho suy
luận với trọng số bị hạn chế phân phối lại.

RF-DETR giữ bộ nhận diện riêng vì cần toàn bộ checkpoint thay vì chỉ dict tensor
để phát hiện kích thước và ánh xạ lại lớp COCO. Nó chỉ được đăng ký khi cài các
dependency tùy chọn.

Mọi họ đã đăng ký khác dùng mặc định: họ nhận tệp khi trình tải riêng đã nhận diện các khóa đó.

## Họ nào thắng

Nhiều họ có thể nhận cùng tệp, nên cách phân giải phản ánh quy tắc định tuyến của factory.

Yêu cầu nhận của lớp con thắng lớp cơ sở. Thứ tự đăng ký theo thứ tự tạo lớp, nên
họ dẫn xuất đăng ký sau lớp cơ sở mà nó tinh chỉnh; các dấu hiệu dương của nó
không được thua luồng passthrough rộng hơn của lớp cơ sở.

Sau đó thứ tự registry quyết định vì mã hóa mức độ cụ thể: yêu cầu nhận sớm nhất là kết quả khớp cụ thể nhất.

Trường hợp hòa duy nhất thứ tự registry không thể phá là DEIM với D-FINE, có các
khóa kiến trúc giống nhau. Chỉ tại đó, tên tệp là tín hiệu quyết định và tệp có
tên không gợi ý sẽ bị từ chối thay vì đoán. Tên tệp được cố ý không dùng ở nơi
khác, nên yêu cầu nhận dương tính giả rộng không bao giờ được ưu tiên hơn yêu cầu
cụ thể chỉ vì tên tệp.

## Tải an toàn

Tệp thượng nguồn được tải qua unpickler chỉ dành cho trọng số. Một số checkpoint
huấn luyện thượng nguồn nhúng đối tượng thư viện bị unpickler từ chối. Các đối
tượng đó là metadata huấn luyện chứ không phải trọng số, nên mỗi global bị chặn
được thử lại với lớp thay thế bất hoạt, đáp ứng unpickler mà không thực thi gì.
Tên được ghi lại chỉ dùng làm nhãn chuỗi, không bao giờ được import, đánh giá hay gọi.

Tên module nhạy cảm bị từ chối hoàn toàn và không bao giờ được stub: `builtins`,
`os`, `sys`, `posix`, `nt` và `subprocess`. Vòng lặp thử lại giới hạn ở 32 lần,
nên tệp được tạo để đưa vào chuỗi global riêng biệt vô hạn sẽ thất bại an toàn
thay vì quay mãi. Chỉ tensor tồn tại trong checkpoint đã chuyển đổi.

## Nơi ghi tệp đã chuyển đổi

Đầu ra được ghi cạnh nguồn với tên `<source>-<Prefix><size>[-task].pt`. Tệp luôn
được ghi lại thay vì dùng lại, giúp các lần tải lặp lại cùng nguồn luôn mới và
tránh xung đột với trọng số chính thức hoặc bản tinh chỉnh khác cùng họ, kích
thước và tác vụ trong cùng thư mục.

Khi thư mục nguồn chỉ đọc, quá trình chuyển đổi dùng thư mục tạm riêng mới được
tạo cho mỗi lời gọi, và dòng log nêu đường dẫn đã dùng. Chỉ khi cách đó cũng thất
bại thì chuyển đổi mới bị hủy kèm cảnh báo.

## Checkpoint LibreYOLO hiện có

Tệp mang dấu hiệu riêng của LibreYOLO là `libreyolo_version` hoặc `model_family`
thuộc luồng tải bình thường và không được chuyển đổi lại. Việc bỏ qua chỉ áp dụng
cho yêu cầu passthrough, tức keyset không đổi. Yêu cầu có quá trình chuyển đổi làm
thay đổi keyset là bằng chứng về bố cục thượng nguồn bên ngoài và được chấp nhận
ngay cả trên tệp đã đánh dấu.

`schema_version` cố ý không được coi là dấu hiệu vì các công cụ huấn luyện và xuất
khác dùng tên chung đó; `names`, `nc`, `size`, `task` và `imgsz` cũng vậy vì bản
tinh chỉnh thượng nguồn có thể chứa chúng. Do đó, bản tinh chỉnh bên ngoài chỉ
mang khóa `names` chung không bị đánh dấu, nên yêu cầu nhận theo khóa native được
chuyển đổi bình thường và suy ra số lớp từ head tensor thay vì bị tải sai thành 80 lớp.

## Metadata trình chuyển đổi đọc

Tên lớp được lấy từ khóa `names` cấp cao nhất hoặc `class_names` trong block
`args` hay `hyper_parameters`. Bản đồ names có khóa là nhãn thay vì chỉ mục lớp
không dùng được và được thay bằng giá trị mặc định đã sinh. Danh sách names dài
hơn số lớp phát hiện được sẽ bị cắt vì chỉ mục ngoài phạm vi làm bộ đánh giá
nghiêm ngặt thất bại và âm thầm hủy chuyển đổi.

`args` thượng nguồn được chuyển sang dưới dạng metadata thông thường; mọi giá trị
không phải chuỗi, số, boolean, danh sách hoặc dict đều bị loại để không có gì
không an toàn đến tệp đã lưu.

## Chuẩn hóa RF-DETR COCO

Checkpoint RF-DETR thượng nguồn cung cấp head phân loại 91 đầu ra, gồm 90 lớp
COCO cộng hậu cảnh. Tự động chuyển đổi chuẩn hóa RF-DETR COCO theo quy ước COCO-80,
với phép ánh xạ lại được áp dụng khi hậu xử lý.

Checkpoint được coi là COCO khi chứa đúng 80 tên, khai báo số lớp 80, có gợi ý
tập dữ liệu `coco`, hoặc hoàn toàn không có metadata lớp hay tập dữ liệu. Trường
hợp cuối rất quan trọng: state dict thượng nguồn trần là checkpoint tiền huấn
luyện COCO chuẩn và là RF-DETR 91 đầu ra không metadata duy nhất được phân phối.

RF-DETR tùy chỉnh 90 lớp thực sự được giữ nguyên 90 lớp. Nó được nhận diện bằng
danh sách names, số lớp khác 80 được khai báo rõ hoặc gợi ý tập dữ liệu không
phải COCO, nên luồng dự phòng checkpoint trần không kích hoạt. Giá trị giữ chỗ
rỗng bị bỏ qua khi quyết định có gợi ý tập dữ liệu hay không.

## Giới hạn

Tự động chuyển đổi nhận diện bố cục thượng nguồn đã phát hành. Nó không viết lại
kiến trúc và không làm mô hình chưa được chuyển đổi trở nên tải được. Khi không
họ nào nhận tệp, giải pháp là script chuyển đổi thay vì đối số factory: repo phân
phối `weights/convert_*.py` cho các họ cần, gồm EoMT, PIDNet và EfficientDet.

Quá trình chuyển đổi cũng không tự tạo metadata không thể đọc. Kích thước, tác vụ
và số lớp đến từ tensor; tên lấy từ tệp khi có và được sinh dưới dạng `class_i`
khi không có.

