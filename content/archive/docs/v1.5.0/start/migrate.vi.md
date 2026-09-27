---
title: Nhập trọng số có sẵn
seo_title: Nạp trọng số upstream trong LibreYOLO
description: >-
  Trỏ LibreYOLO đến checkpoint từ một dự án upstream. Tự động chuyển đổi đóng
  gói lại file lúc nạp, giữ nguyên số lớp và tên.
lead: >-
  LibreYOLO port các họ mô hình từ dự án upstream, vì vậy checkpoint được phát
  hành của chúng gần như đã có thể nạp. Phần còn thiếu là metadata. Tự động
  chuyển đổi bổ sung metadata lúc nạp.
keywords:
  - chuyển đổi trọng số libreyolo
  - nạp checkpoint upstream
  - di chuyển sang libreyolo
  - đổi pth sang libreyolo
  - tự động chuyển đổi checkpoint
last_verified: 1.5.0
meta:
  - label: Điểm vào
    value: LibreYOLO("path/to/upstream.pth")
    mono: true
  - label: Ghi cạnh nguồn với tên
    value: '<source>-<Prefix><size>[-task].pt'
    mono: true
  - label: Script chuyển đổi
    value: weights/ trong repo
    mono: true
snippets:
  convert:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Thay bằng đường dẫn đến checkpoint bạn đã có. Layout upstream được
        # nhận diện sẽ được chuyển đổi tức thời, ghi cạnh file nguồn rồi nạp.
        model = LibreYOLO("path/to/upstream-checkpoint.pth")

        # Số lớp và tên đến từ tensor cùng metadata của file, nên bản tinh chỉnh
        # giữ tập nhãn của nó thay vì tập nhãn COCO.
        print(model.family, model.size, model.task, model.nb_classes)
        print(model.names)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=path/to/upstream-checkpoint.pth \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Kiểm tra kết quả
      language: bash
      code: |
        # File đã chuyển đổi đáp ứng cùng schema như file được công bố.
        libreyolo metadata path=path/to/upstream-checkpoint-LibreYOLO9t.pt
source_hash: bf9d7c7d168fd2c0
---

Trang này nói về checkpoint từ dự án khác. Nếu bạn đang chuyển mã của mình từ
LibreYOLO cũ, xem [nâng cấp lên 1.5.0](/docs/upgrade).

## Điều gì xảy ra khi bạn nạp một file bên ngoài

`LibreYOLO()` trước tiên nạp mọi file trọng số qua pipeline hạn chế chỉ cho
trọng số. Nếu kết quả có metadata LibreYOLO đầy đủ, nó được dùng trực tiếp. Nếu
không, file được chuyển cho bộ tự động chuyển đổi trước mọi thao tác khác. Nếu
quá trình nạp hạn chế thất bại hoàn toàn, thường do checkpoint chứa object bên
thứ ba được pickle, bộ tự động chuyển đổi sẽ thử với loader vô hiệu hóa các
object đó.

Tự động chuyển đổi thực hiện bốn việc. Nó lấy dictionary tensor ra khỏi layout
mà dự án upstream dùng. Nó hỏi mọi họ đã đăng ký xem họ có nhận diện các khóa
kết quả không, đồng thời ánh xạ lại tên khi quy tắc đặt tên upstream khác bản
port LibreYOLO. Nó bọc họ phù hợp trong checkpoint đáp ứng metadata schema
v1.0, đọc kích thước, tác vụ và số lớp từ chính tensor. Sau đó nó ghi kết quả
cạnh file nguồn và nạp file đó.

<code-tabs name="convert" />

Việc chuyển đổi không diễn ra âm thầm. File đã chuyển đổi được log cùng họ, tên
nguồn, tên đầu ra và số lớp kết quả, để log của lần chạy ghi chính xác những gì
được nạp.

## Các layout được mở gói

Checkpoint upstream lồng trọng số ở một số vị trí quy ước, và bộ chuyển đổi thử
theo thứ tự đến khi tìm được tensor: block EMA dưới `ema.module` hoặc `ema`
phẳng, `ema_state_dict` sau khi bỏ prefix `module.`, rồi `params_ema`, `params`,
`ema_net`, `net`, `model`, `state_dict`, cuối cùng là chính object. Việc thử
nhiều vị trí thay vì chỉ vị trí đầu giúp block `ema` chỉ chứa bộ đếm không che
mất trọng số thực phía dưới.

Các prefix wrapper cũng được bỏ: `module.` từ huấn luyện phân tán, `_orig_mod.`
từ mô hình đã compile và lớp lồng `model.model.` do một số bản phân phối lại
thêm vào.

## Nội dung được đọc và nguồn của chúng

Kích thước, tác vụ và số lớp đến từ tensor chứ không phải tên file, vì vậy
checkpoint tinh chỉnh được chuyển đổi với số lớp riêng thay vì mặc định của
kiến trúc. Tên lớp lấy từ metadata của checkpoint nếu có, từ block `args` hoặc
`hyper_parameters` nếu nằm ở đó, và được cắt về số lớp phát hiện được để bản
tinh chỉnh giữ tập nhãn cơ sở không mang theo chỉ mục mà head không còn hỗ trợ.

Tác vụ dense được xử lý tường minh thay vì gán nhãn giả. Checkpoint depth nhận
một lớp tên `depth`, checkpoint restore nhận một lớp tên `image`. Checkpoint
pose phải cung cấp số keypoint từ tensor hoặc từ họ; nếu cả hai đều không có,
quá trình chuyển đổi bị từ chối thay vì ghi file không hoàn chỉnh.

RF-DETR có bộ nhận diện riêng vì việc xác định kích thước cần toàn bộ checkpoint
và vì head có 91 đầu ra trong khi LibreYOLO dùng quy ước COCO 80 lớp. Một
checkpoint được chuẩn hóa về 80 lớp khi nó có đúng 80 tên, khai báo số lớp 80,
nêu COCO là dataset, hoặc hoàn toàn không có metadata lớp hay dataset. Mô hình
90 lớp thực, được xác định bằng tên, số lớp khác 80 tường minh hoặc gợi ý
dataset không phải COCO, được giữ nguyên.

## File đã chuyển đổi được lưu ở đâu

Đầu ra được ghi cạnh nguồn với tên theo nguồn:

```text
<source stem>-<FilenamePrefix><size>[-<task suffix>].pt
```

Detector YOLOv9 tiny lưu dưới `upstream-checkpoint.pth` vì vậy trở thành
`upstream-checkpoint-LibreYOLO9t.pt`. Đặt tên theo nguồn thay vì họ giúp hai bản
tinh chỉnh cùng họ và kích thước trong một thư mục không ghi đè nhau, đồng thời
không xung đột với checkpoint chính thức. File được ghi lại mỗi lần nạp nên
không bao giờ lỗi thời so với nguồn. Nếu thư mục chỉ đọc, file đã chuyển đổi
được ghi vào thư mục tạm riêng mới và log cho biết vị trí.

Từ đó, nó là checkpoint LibreYOLO thông thường: nạp qua pipeline metadata và
`libreyolo metadata` báo file hợp lệ.

## Các trường hợp cần xử lý thủ công

Hai họ nằm ngoài bộ nhận diện chung. Họ gaze bị loại hoàn toàn: nó chỉ cung cấp
inference và trọng số phát hành có hạn chế phân phối lại. RF-DETR bị loại vì có
bộ nhận diện chuyên dụng mô tả ở trên, và bộ đó xử lý thay thế.

Checkpoint PIDNet upstream thô bị từ chối với lỗi trỏ đến
`weights/convert_pidnet_weights.py`. Script này ghi metadata semantic
Cityscapes mà checkpoint cần.

D-FINE và DEIM dùng chung khóa kiến trúc nên chỉ tensor không thể phân biệt.
Khi cả hai cùng nhận một file và không có họ sibling mang marker phân biệt,
tên file quyết định: tên dạng `dfine_hgnetv2_n_coco.pth` hoặc
`deim_hgnetv2_n_coco.pth` giải quyết được, còn tên không cung cấp thông tin sẽ
bị từ chối kèm giải thích thay vì phỏng đoán. Khởi tạo trực tiếp `LibreDFINE`
hoặc `LibreDEIM` cũng giải quyết được.

Khi nhiều họ cùng nhận hợp lệ một file, subclass thắng lớp cơ sở mà nó tinh
chỉnh, còn thứ tự registry quyết định phần còn lại vì thứ tự mã hóa mức độ cụ
thể của phép kiểm tra từng họ. Tên file chỉ được xét khi D-FINE và DEIM hòa,
nên tên không bao giờ đẩy một kết quả khớp rộng lên trên kết quả chính xác.

## Các script chuyển đổi

Repo chứa script chuyển đổi theo từng họ dưới `weights/`, cùng helper dùng
chung cho phần xử lý lặp lại. Đây là hướng dành cho file mà pipeline runtime từ
chối, để tạo checkpoint trước thay vì lúc nạp, và cho các họ có metadata phải
được cung cấp chứ không thể suy ra từ tensor.

Các script này thuộc repo, không nằm trong package đã cài, nên muốn dùng phải
clone:

```bash
git clone https://github.com/LibreYOLO/libreyolo.git
cd libreyolo
python weights/convert_pidnet_weights.py --help
```

Mỗi script ghi checkpoint đáp ứng schema v1.0, cùng tiêu chuẩn với tự động
chuyển đổi và trọng số công bố. Xem [checkpoint và trọng số](/docs/weights) để
biết schema chứa gì.
