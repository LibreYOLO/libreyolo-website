---
title: Giấy phép
seo_title: 'Giấy phép LibreYOLO: mã nguồn và trọng số'
description: >-
  Mã nguồn riêng của LibreYOLO dùng MIT. Mã upstream được đưa vào và checkpoint
  đã công bố có giấy phép riêng, trong đó một số chỉ cho phép phi thương mại.
lead: >-
  LibreYOLO chứa ba thành phần được cấp phép riêng: mã nguồn của chính nó, mã
  upstream được đưa vào một họ mô hình và checkpoint huấn luyện sẵn. Chúng
  thường không dùng cùng giấy phép.
keywords:
  - giấy phép libreyolo
  - thư viện thị giác máy tính mit
  - trọng số mô hình phi thương mại
  - giấy phép checkpoint mô hình
  - apache 2.0 detection
last_verified: 1.5.0
source_hash: 83536fea4dc4eaec
---

## Mã nguồn riêng của LibreYOLO

Thư viện dùng MIT. Giấy phép này bao gồm Python API, CLI, trainer, validator và
exporter, loader dataset cùng script chuyển đổi dưới `weights/`. Bạn có thể
dùng trong sản phẩm thương mại hoặc nguồn đóng; hãy giữ dòng bản quyền và nội
dung giấy phép cùng mọi bản sao được phân phối lại, và nghĩa vụ dừng ở đó.

Quyền cấp phép chỉ áp dụng cho mã nguồn. File
[`LICENSE`](https://github.com/LibreYOLO/libreyolo/blob/release/LICENSE) nói rõ:

> Các giấy phép đó khác nhau và không phải tất cả đều thông thoáng: một số
> trọng số công bố chỉ dành cho mục đích phi thương mại hoặc bị hạn chế theo
> cách khác, và Giấy phép MIT này không mở rộng đến chúng. Chọn mô hình cũng có
> nghĩa là chọn giấy phép của nó.

## Mã upstream theo từng họ

Phần lớn các họ là bản port của nghiên cứu đã công bố, và một số đưa trực tiếp
mã nguồn upstream vào. File được đưa vào giữ header bản quyền và giấy phép gốc.
MIT không ghi đè giấy phép đó, và LibreYOLO không cấp lại giấy phép cho công
trình của bất kỳ ai. Apache-2.0 và BSD-3-Clause là hai giấy phép thường gặp nhất.

Apache-2.0 bao phủ dòng DETR và phần lớn công trình transformer: DETR từ Meta AI
(FAIR), Deformable DETR từ SenseTime, LW-DETR từ Baidu, OV-DEIM của Leilei Wang
và các đồng tác giả, bản triển khai SegFormer mà LibreYOLO port từ Hugging Face
Transformers, PP-OCRv5 từ PaddlePaddle Authors, SwinIR từ Computer Vision Lab
tại ETH Zurich và Depth Anything 3 từ ByteDance Seed. Nó cũng bao phủ các
classifier bắt nguồn từ timm của Ross Wightman và cộng đồng timm, gồm ResNet,
DeiT, EfficientNetV2, MobileNetV4 và Swin, với tên module phản chiếu timm để
tensor ImageNet được nạp không thay đổi.

BSD-3-Clause bao phủ mọi thứ bắt nguồn từ torchvision: Faster R-CNN, Mask
R-CNN, FCOS, RetinaNet, SSD300, AlexNet, VGG, FCN và DeepLabv3.

MIT bao phủ một nhóm nhỏ hơn, gồm NAFNet từ Megvii, CenterNet từ Xingyi Zhou và
YOLOv7 được chính các tác giả Kin-Yiu Wong cùng Hao-Tang Tsui tại
MultimediaTechLab phát hành lại. Các họ YOLOv1 đến YOLOv4 tái tạo kiến trúc từ
dự án Darknet của Joseph Redmon và với YOLOv4 là Alexey Bochkovskiy. Darknet
thuộc phạm vi công cộng nên các họ đó không có nghĩa vụ nào.

Một subtree được bundle không dùng giấy phép nguồn mở. Họ DEIMv2 cung cấp mã
backbone DINOv3 từ Meta Platforms theo DINOv3 License Agreement, một giấy phép
tùy chỉnh không thuộc OSI. Phân phối lại mã này đòi hỏi kèm bản sao thỏa thuận,
và thỏa thuận cấm dùng cho hoạt động thuộc ITAR, mục đích quân sự hoặc chiến
tranh, ngành hạt nhân, gián điệp và phát triển vũ khí. Các điều khoản đó chỉ
ràng buộc subtree này.

Hai file trong repo chứa bức tranh đầy đủ.
[`NOTICE`](https://github.com/LibreYOLO/libreyolo/blob/release/NOTICE) liệt kê
mọi subtree bên thứ ba được bundle cùng đường dẫn, file giấy phép và nguồn
upstream.
[`THIRD_PARTY_NOTICES.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/THIRD_PARTY_NOTICES.txt)
liệt kê các dự án upstream LibreYOLO bắt nguồn từ và chép toàn bộ nội dung từng
giấy phép.

## Trọng số theo từng checkpoint

Không file trọng số huấn luyện sẵn nào nằm trong package. Checkpoint công bố
nằm trên Hugging Face dưới [tổ chức
LibreYOLO](https://huggingface.co/LibreYOLO), và mỗi repo có `LICENSE` cùng
thông tin ghi công riêng phản ánh dự án cung cấp trọng số.

Repo đó là nguồn có thẩm quyền cho các điều khoản. Không phải trang này, trang
mô hình hay phần tóm tắt trong cây nguồn. Xem [checkpoint và trọng
số](/docs/weights) để biết cách đặt tên file và nơi tải.

Giấy phép khác nhau giữa các họ và giữa các file trong cùng một họ. Hai ví dụ
cho trường hợp thứ hai:

- Các checkpoint YOLO9 COCO dùng MIT. `LibreYOLO9P2s-visdrone.pt`, được huấn
  luyện trên VisDrone2019-DET, dùng CC BY-NC-SA 3.0 và chỉ cho phép phi thương
  mại.
- Checkpoint detection RF-DETR dùng Apache-2.0. Checkpoint oriented-box dùng
  CC BY 4.0 vì được tinh chỉnh trên dataset Roboflow Universe công bố theo CC
  BY 4.0, và trọng số tiếp tục mang yêu cầu ghi công của dataset đó.

Giữa các họ, phạm vi còn rộng hơn và một số checkpoint công bố không thể dùng
trong sản phẩm thương mại:

- SegFormer cho thấy rõ nhất sự phân tách giữa hai lớp. Bản triển khai là bản
  port Apache-2.0 từ mã Hugging Face Transformers. Checkpoint ADE20K công bố
  được chuyển đổi từ bản phát hành NVIDIA theo NVIDIA Source Code License, cho
  phép phân phối lại nhưng giới hạn sử dụng vào nghiên cứu hoặc đánh giá phi
  thương mại, đồng thời truyền giới hạn đó sang tác phẩm phái sinh. Các
  checkpoint này không nằm dưới điều khoản thông thoáng của LibreYOLO.
- Checkpoint OV-DEIM dùng CC BY-NC 4.0, được tác giả upstream xác nhận. Mỗi dự
  đoán còn nạp text tower MobileCLIP-B(LT) của Apple, có giấy phép giới hạn sử
  dụng cho nghiên cứu, nghiêm ngặt hơn giấy phép riêng của checkpoint.
- Mã SenseNova-Vision dùng Apache-2.0 và trọng số dùng CC BY-NC 4.0. Loader in
  thông báo phi thương mại trước mỗi lần tự động tải.

Một số họ hoàn toàn không có checkpoint do LibreYOLO lưu trữ, và trang của họ
nói rõ ở hàng Trọng số. SAM 3 bị giới hạn truy cập trên Hugging Face theo SAM
License tùy chỉnh của Meta và được tải trực tiếp từ Meta. Asset phát hành MiDaS
được lấy từ URL chính thức và xác minh hash thay vì lưu trữ lại. Dome-DETR được
liên kết về upstream vì metadata model card không nêu giấy phép, trong khi nội
dung lại tuyên bố Apache-2.0 đồng thời giới hạn dùng cho nghiên cứu học thuật,
hai điều này không nhất quán. Kiến trúc TEED và DexiNed dùng MIT, nhưng
checkpoint của tác giả được huấn luyện trên BIPED có điều khoản dataset phi
thương mại, nên LibreYOLO không bundle hoặc tự động tải chúng.

Một số checkpoint torchvision không có file giấy phép riêng. LibreYOLO mirror
chúng theo giấy phép mà dự án phát hành sử dụng, nêu trên từng model card rằng
căn cứ này là ngụ ý chứ không phải được cấp riêng cho checkpoint, và lặp lại
cảnh báo của torchvision rằng điều khoản mô hình huấn luyện sẵn có thể bắt
nguồn từ dữ liệu huấn luyện.

## Tìm điều khoản cho một mô hình

Trang mô hình có hàng **Giấy phép** ở phần đầu, dạng `Mã nguồn X, trọng số Y`,
liên kết xuống phần Giấy phép trên trang. Phần này liệt kê công trình gốc và
tác giả, giấy phép upstream, nguồn upstream, giấy phép mã LibreYOLO, trọng số
và diễn giải điều khoản cho phép gì. Bảng Checkpoint trên cùng trang có cột
**Giấy phép trọng số**, mỗi file công bố một hàng, nên họ có điều khoản hỗn hợp
sẽ hiển thị theo từng file.

Tất cả được render từ cùng dữ liệu mà thư viện được kiểm tra đối chiếu, vì vậy
trang này không lặp lại dưới dạng bảng. Ma trận giấy phép nhập tay sẽ sai ngay
trong một bản phát hành, và sai ở đây gây hậu quả lớn.

Trong cây nguồn, các file tương ứng là `NOTICE` cho mã được bundle,
`THIRD_PARTY_NOTICES.txt` cho dự án upstream cùng nội dung giấy phép, và
[`weights/LICENSE_NOTICE.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/weights/LICENSE_NOTICE.txt)
cho bản tóm tắt theo họ của checkpoint công bố.

Sau đó hãy kiểm tra repo Hugging Face chứa chính xác file bạn sắp tải. Đây là
nguồn có thẩm quyền và có thể thay đổi mà trang tài liệu không đổi theo.

## Sử dụng thương mại

Mã nguồn hiếm khi là vấn đề. MIT, Apache-2.0 và BSD-3-Clause đều cho phép sử
dụng thương mại và nguồn đóng. Mỗi giấy phép yêu cầu giữ nội dung giấy phép và
thông báo ghi công cùng bản sao phân phối lại, Apache-2.0 còn cấp quyền bằng
sáng chế, và không giấy phép nào đặt điều kiện lên mã ứng dụng của riêng bạn.

Checkpoint là nơi sản phẩm thường mắc kẹt. Checkpoint phi thương mại vẫn là
phi thương mại dù mã xung quanh thông thoáng đến đâu, và chuyển đổi file không
thay đổi điều khoản áp dụng, đúng như `weights/LICENSE_NOTICE.txt` nói trực
tiếp. Artifact ONNX hoặc TensorRT dựng từ checkpoint bị hạn chế sẽ kế thừa hạn
chế đó.

Khi giấy phép truyền hạn chế sang tác phẩm phái sinh như NVIDIA Source Code
License, tinh chỉnh cũng không loại bỏ được. Huấn luyện cùng kiến trúc từ đầu
trên dữ liệu bạn có quyền dùng thì có thể: mã nguồn thông thoáng, nên mô hình
bạn tự huấn luyện thuộc về bạn và điều khoản checkpoint huấn luyện sẵn không
tham gia. Trang SegFormer nêu rõ điều này với trọng số của họ; hãy đọc hàng
Diễn giải trên trang của bất kỳ họ nào bạn định phát hành.

Hãy quyết định vấn đề giấy phép khi chọn mô hình thay vì lúc phát hành, và đọc
điều khoản trên chính file đã tải vì họ có một checkpoint thông thoáng vẫn có
thể có checkpoint bị hạn chế bên cạnh.

## Không phải tư vấn pháp lý

Trang này mô tả các giấy phép liên quan. Đây là nội dung mô tả, không phải tư
vấn pháp lý và không tạo ra bất kỳ bảo đảm nào. Nếu câu trả lời có ý nghĩa
thương mại, hãy tự đọc giấy phép và tham khảo cố vấn riêng.
