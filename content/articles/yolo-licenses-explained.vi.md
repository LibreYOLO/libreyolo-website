---
title: "Giấy phép YOLO: từ v1 đến YOLO26, phiên bản nào dùng thương mại được? (2026)"
description: "Bản đồ giấy phép YOLO đầy đủ năm 2026: từ YOLOv1 đến YOLOv13, YOLO26, YOLO-World và YOLOE, repo gốc và các bản viết lại có giấy phép permissive, kèm liên kết bài báo, GitHub và thông tin bạn thực sự có thể phát hành trong sản phẩm thương mại."
date: 2026-07-11
author: Xuban
tags: [LibreYOLO, yolo-license, agpl, gpl, apache, mit-license, yolov9, yolov10, yolo11, yolov12, yolov13, yolo26]
faq:
  - q: "YOLOv9 có miễn phí để sử dụng thương mại không?"
    a: "Còn tùy bạn dùng repo nào. Repo bài báo (WongKinYiu/yolov9) dùng GPL-3.0. Cùng phòng thí nghiệm đó cũng phát hành một triển khai YOLOv9 và YOLOv7 riêng, dùng MIT, tại MultimediaTechLab/YOLO, được viết sau khi người dùng thương mại yêu cầu một lựa chọn permissive. Đây là bản viết lại, không phải đổi giấy phép cho các tệp GPL. Giấy phép mã nguồn đã rõ; cả hai repo đều không có tuyên bố giấy phép riêng cho trọng số pretrained, vì vậy hãy xem nguồn gốc trọng số là một vấn đề riêng."
  - q: "Những phiên bản YOLO nào miễn phí cho sản phẩm thương mại mã nguồn đóng?"
    a: "Có ít nhất một repo permissive cho các phiên bản sau: YOLOv1 đến YOLOv4 (Darknet gốc, thuộc public domain), YOLOv7 và YOLOv9 (bản viết lại MIT của chính phòng thí nghiệm), YOLOX và PP-YOLOE (Apache-2.0), cùng các bản triển khai lại MIT trong LibreYOLO. Tính đến tháng 7 năm 2026, YOLOv5, v6, v8, v10, YOLO11, v12, v13, YOLO26, YOLO-World và YOLOE không có triển khai permissive. Hãy tự kiểm tra tệp LICENSE hiện tại trước khi dựa vào thông tin này: giấy phép có thể thay đổi, và đây là thông tin tổng quát, không phải tư vấn pháp lý."
  - q: "YOLOv10 dùng giấy phép nào?"
    a: "AGPL-3.0. YOLOv10 là bản phát hành học thuật của Đại học Thanh Hoa, nhưng mã nguồn được xây dựng trên codebase Ultralytics nên kế thừa AGPL-3.0. Không có bản triển khai lại permissive."
  - q: "YOLOv12 và YOLOv13 dùng giấy phép nào?"
    a: "Cả hai đều dùng AGPL-3.0, được xác nhận trong tệp LICENSE của chúng. Giống YOLOv10, đây là các bài báo học thuật có mã tham chiếu được xây dựng trên repo Ultralytics, vì vậy AGPL được áp dụng bất kể ai viết bài báo. Các trang bên thứ ba liệt kê YOLOv13 là Apache-2.0 đều sai."
  - q: "Ultralytics chuyển YOLO sang AGPL-3.0 khi nào?"
    a: "Ngày 14 tháng 4 năm 2023, không phải năm 2022 như nhiều nơi thường nhắc lại. Tệp LICENSE trong ultralytics/yolov5 chỉ được thay đổi đúng ba lần, và commit AGPL (34cf749, PR #11359) có ngày 2023-04-14. Bản phát hành cuối cùng năm 2022, YOLOv5 v7.0, vẫn dùng GPL-3.0. Repo ultralytics/ultralytics cũng được đổi giấy phép cùng ngày. Vì vậy YOLOv8 ra mắt vào tháng 1 năm 2023 dưới GPL-3.0: các bản phát hành PyPI từ 8.0.0 đến 8.0.76 khai báo GPL-3.0, còn 8.0.80 (ngày 16 tháng 4 năm 2023) là bản đầu tiên khai báo AGPL-3.0."
  - q: "YOLO26 có miễn phí để sử dụng thương mại không?"
    a: "Không dành cho sản phẩm mã nguồn đóng. YOLO26 được phát hành theo AGPL-3.0, với Enterprise License trả phí làm lựa chọn thay thế, cùng điều khoản với YOLOv8 và YOLO11."
  - q: "YOLO Darknet gốc thực sự thuộc public domain chứ?"
    a: "Đúng. LICENSE của Darknet do Joseph Redmon viết nêu rằng 'Darknet is public domain. Do whatever you want with it.' Fork YOLOv4 của AlexeyAB có cùng nội dung public domain. Điểm cần chú ý là các bản PyTorch mà mọi người thực sự dùng có giấy phép riêng, từ Apache-2.0 đến AGPL-3.0 hoặc hoàn toàn không có giấy phép."
  - q: "Tôi có thể dùng trọng số YOLO-NAS pretrained cho mục đích thương mại không?"
    a: "Không. Mã super-gradients dùng Apache-2.0, nhưng trọng số YOLO-NAS chính thức được phát hành theo giấy phép riêng, trong đó nêu rằng bạn 'may not use the Software for any commercial use, including in connection with any models used in a production environment.'"
  - q: "Trọng số mạng nơ-ron có kế thừa giấy phép của mã huấn luyện không?"
    a: "Vấn đề này chưa ngã ngũ. Ultralytics nêu rằng AGPL-3.0 áp dụng cho 'the training code and the models produced by that training code', và với tư cách chủ sở hữu bản quyền, họ đặt điều khoản cho nội dung họ phát hành. Chưa có vụ việc nào kiểm nghiệm việc tòa án có đồng ý rằng trọng số bạn tự huấn luyện là tác phẩm phái sinh của mã huấn luyện hay không. Hãy xem trọng số AGPL đã phát hành là có ràng buộc và thận trọng khi tải chúng vào một bản triển khai lại permissive."
  - q: "Vì sao các triển khai khác nhau của cùng một YOLO có thể có giấy phép khác nhau?"
    a: "Bản quyền bảo vệ cách thể hiện, không bảo vệ ý tưởng (17 U.S.C. 102(b)). Một kiến trúc được mô tả trong bài báo có thể được triển khai độc lập và tác giả có thể chọn giấy phép tùy ý. Bạn không thể sao chép các tệp mã nguồn GPL hoặc AGPL rồi đổi giấy phép cho chúng. Lưu ý, đây chỉ là lập luận về bản quyền: triển khai độc lập không phải là biện pháp bảo vệ trước bằng sáng chế."
---

Mỗi năm lại có thêm nhiều YOLO, và năm nào câu hỏi về giấy phép cũng được đặt ra, thường là chỉ năm phút trước khi quyết định sản phẩm. Phần gây nhầm lẫn là "YOLO" không phải một dự án. Đây là tên thương hiệu được dùng chung cho khoảng hai mươi dòng mô hình của nhiều tác giả khác nhau, và đây là chi tiết mà phần lớn hướng dẫn về giấy phép bỏ sót: **giấy phép gắn với repo, không gắn với mô hình.** Cùng một phiên bản YOLO có thể đồng thời tồn tại trong repo GPL và repo MIT, từ cùng một nhóm tác giả. YOLOv9 là trường hợp như vậy, và điều đó làm thay đổi hoàn toàn quyết định.

Hướng dẫn này lập bản đồ toàn cảnh theo từng repo, kèm liên kết tới từng bài báo và codebase, cập nhật đến tháng 7 năm 2026. Mỗi giấy phép dưới đây đều được đối chiếu với tệp LICENSE thực tế trong repo thực tế, vì có khá nhiều thông tin về giấy phép YOLO (kể cả trong các hướng dẫn xếp hạng cao nhất cho chủ đề này) là sai.

Nếu bạn chỉ cần câu trả lời về AGPL của Ultralytics, chúng tôi giải thích chi tiết trong bài [YOLO có miễn phí để sử dụng thương mại không?](/articles/yolo-commercial-license). Bài này là bản đồ đầy đủ.

**Hai điều cần công khai trước khi bắt đầu.** Thứ nhất, chúng tôi duy trì LibreYOLO, một thư viện dùng MIT cạnh tranh với một số dự án bên dưới, và phần cuối bài viết nói về thư viện này. Đó là lý do để kiểm tra nội dung chúng tôi viết, chứ không phải tin theo vô điều kiện, vì vậy mọi khẳng định về giấy phép ở đây đều liên kết tới nguồn chính. Thứ hai, bài viết này cung cấp thông tin tổng quát về văn bản giấy phép đã công bố và tuyên bố công khai, cập nhật theo ngày nêu trên. Đây không phải tư vấn pháp lý và không thay thế luật sư tại khu vực pháp lý của bạn. Giấy phép và lập trường của người duy trì có thể thay đổi. Hãy đọc tệp LICENSE hiện tại của bất kỳ repo nào trước khi quyết định phát hành sản phẩm.

## Bảng giấy phép

"Lựa chọn permissive" có nghĩa là: một repo triển khai mô hình này mà bạn có thể dùng trong sản phẩm thương mại mã nguồn đóng, không cần mở mã nguồn ứng dụng và không phải trả tiền mua giấy phép.

| Mô hình | Năm | Mã nguồn gốc | Lựa chọn permissive | Bài báo |
| --- | --- | --- | --- | --- |
| YOLOv1 | 2015 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (public domain) | Chính bản gốc | [arXiv](https://arxiv.org/abs/1506.02640) |
| YOLOv2 | 2016 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (public domain) | Chính bản gốc | [arXiv](https://arxiv.org/abs/1612.08242) |
| YOLOv3 | 2018 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (public domain) | Bản gốc; lưu ý bản port PyTorch dùng AGPL | [arXiv](https://arxiv.org/abs/1804.02767) |
| YOLOv4 | 2020 | [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) (public domain) | Bản gốc; [pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) (Apache-2.0) | [arXiv](https://arxiv.org/abs/2004.10934) |
| YOLOv5 | 2020 | [ultralytics/yolov5](https://github.com/ultralytics/yolov5) (AGPL-3.0) | Không có | không có bài báo |
| YOLOv6 | 2022 | [meituan/YOLOv6](https://github.com/meituan/YOLOv6) (GPL-3.0) | Không có | [arXiv](https://arxiv.org/abs/2209.02976) |
| YOLOv7 | 2022 | [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) (GPL-3.0) | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (MIT, cùng phòng thí nghiệm) | [arXiv](https://arxiv.org/abs/2207.02696) |
| YOLOv8 | 2023 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0; GPL-3.0 khi ra mắt, xem bên dưới) | Không có (bản port Keras đã bị loại bỏ, xem bên dưới) | không có bài báo |
| YOLOv9 | 2024 | [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) (GPL-3.0) | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (MIT, cùng phòng thí nghiệm) | [arXiv](https://arxiv.org/abs/2402.13616) |
| YOLOv10 | 2024 | [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10) (AGPL-3.0) | Không có | [arXiv](https://arxiv.org/abs/2405.14458) |
| YOLO11 | 2024 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | Không có | không có bài báo |
| YOLOv12 | 2025 | [sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12) (AGPL-3.0) | Không có | [arXiv](https://arxiv.org/abs/2502.12524) |
| YOLOv13 | 2025 | [iMoonLab/yolov13](https://github.com/iMoonLab/yolov13) (AGPL-3.0) | Không có | [arXiv](https://arxiv.org/abs/2506.17733) |
| YOLO26 | 2026 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | Không có | [arXiv](https://arxiv.org/abs/2606.03748) |
| YOLOX | 2021 | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0) | Chính bản gốc | [arXiv](https://arxiv.org/abs/2107.08430) |
| PP-YOLOE | 2022 | [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) (Apache-2.0) | Bản gốc từ PaddleDetection (không phải PaddleYOLO) | [arXiv](https://arxiv.org/abs/2203.16250) |
| YOLO-NAS | 2023 | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) (mã Apache-2.0, trọng số phi thương mại) | Mã nguồn dùng được, trọng số chính thức thì không | không có bài báo |
| YOLO-World | 2024 | [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World) (GPL-3.0) | Không có | [arXiv](https://arxiv.org/abs/2401.17270) |
| YOLOE | 2025 | [THU-MIG/yoloe](https://github.com/THU-MIG/yoloe) (AGPL-3.0) | Không có | [arXiv](https://arxiv.org/abs/2503.07465) |
| MMYOLO | 2022 | [open-mmlab/mmyolo](https://github.com/open-mmlab/mmyolo) (GPL-3.0, ngừng hoạt động từ năm 2024) | Không có | không có bài báo |

Đọc từng cột, bạn sẽ thấy rõ một điều: số phiên bản không cho biết gì về giấy phép. Repo mới là yếu tố quyết định.

## Cùng mô hình, hai giấy phép: trường hợp YOLOv9

YOLOv9 là minh chứng rõ nhất cho thấy câu hỏi "YOLOvN dùng giấy phép nào?" không đúng trọng tâm, và timeline này đáng được kể chính xác vì đây là lần duy nhất trong lịch sử YOLO mà sức ép cộng đồng thực sự làm thay đổi kết quả.

- **Ngày 18 tháng 2 năm 2024:** [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) được công khai cùng với [bài báo](https://arxiv.org/abs/2402.13616).
- **Ngày 22 tháng 2 năm 2024:** một người dùng mở issue #10 để hỏi giấy phép là gì. Chien-Yao Wang (WongKinYiu) trả lời "I think it should be GPL3" và thêm tệp GPL-3.0 bốn ngày sau đó.
- **Ngày 26 tháng 2 năm 2024:** người dùng thứ hai mở [issue #82, "An Apache/MIT rewrite"](https://github.com/WongKinYiu/yolov9/issues/82). Trong hai tuần rưỡi tiếp theo, một tá người dùng thương mại tham gia thảo luận.
- **Ngày 14 tháng 3 năm 2024:** Wang đăng trong thread: *"Okay I create a new repo for mit rewrite... I think I can handle most of implementation of architectures and loss functions. And need someone to give great help about dataloader and ddp training."*

Repo đó, ban đầu có tên `yolov9mit`, hiện là **[MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO)**: dùng MIT, bản quyền thuộc về "Kin-Yiu, Wong and Hao-Tang, Tsui", và tự mô tả là *"the official implementation of YOLOv7 and YOLOv9, YOLO-RD."* Phần lớn bản viết lại do Hao-Tang Tsui, đồng nghiệp cùng phòng thí nghiệm với Wang, thực hiện chứ không phải các tình nguyện viên trong thread. Tsui là tác giả của khoảng 90 phần trăm số commit.

Vì vậy YOLOv9 vừa "không thể phát hành" vừa "có thể phát hành", tùy repo bạn clone. Cùng kiến trúc, cùng phòng thí nghiệm, hai giấy phép.

**Trước khi đặt cược sản phẩm vào repo này, có ba điểm cần lưu ý mà trang chủ repo không nói rõ:**

1. **Dự án vẫn đang hoàn thiện.** Một [issue đang mở](https://github.com/MultimediaTechLab/YOLO/issues/231) từ tháng 2 năm 2026 báo cáo rằng các checkpoint đã phát hành không tái tạo được các chỉ số COCO trong bài báo và có số lượng tham số cao hơn đáng kể so với mức bài báo công bố. Việc huấn luyện segmentation và keypoint [vẫn chưa được triển khai](https://github.com/MultimediaTechLab/YOLO/issues/232). Không có thay đổi nào được merge vào main kể từ tag v1.0 tháng 12 năm 2025.
2. **Trọng số không có tuyên bố giấy phép riêng.** Theo thông lệ, LICENSE MIT của repo bao gồm các tài sản phát hành, và bằng chứng kỹ thuật cho thấy checkpoint do dự án này huấn luyện chứ không sao chép từ repo GPL: các tệp khác nhau về kích thước và số lượng tham số so với checkpoint của repo GPL. Tuy nhiên, không tài liệu nào nói các tệp .pt dùng MIT, và [yêu cầu kiểm toán chính thức về nhiễm AGPL](https://github.com/MultimediaTechLab/YOLO/issues/51) của cộng đồng vẫn đang mở. Đây là thiếu sót về giấy tờ: chưa ai tuyên bố "đã xác minh là không bị nhiễm", và cũng chưa ai cáo buộc có vấn đề.
3. **Đây là codebase khác**, không phải bản thay thế dùng ngay cho các script của repo GPL.

Những điều đó không khiến dự án không thể dùng được. Chúng có nghĩa đây là dự án cần được đánh giá, chứ không phải một ô để đánh dấu.

## Kỷ nguyên 1: những năm public domain (YOLOv1 đến YOLOv4)

Joseph Redmon phát hành [YOLOv1](https://arxiv.org/abs/1506.02640), [YOLOv2](https://arxiv.org/abs/1612.08242) (được xuất bản dưới tên bài báo YOLO9000) và [YOLOv3](https://arxiv.org/abs/1804.02767) trong framework Darknet của ông, còn tệp [LICENSE](https://github.com/pjreddie/darknet/blob/master/LICENSE) nổi tiếng vì chỉ dài ba dòng:

> 0. Darknet is public domain.
> 1. Do whatever you want with it.
> 2. Stop emailing me about it!

Đây là tuyên bố đưa tác phẩm vào public domain thực sự. [YOLOv4](https://arxiv.org/abs/2004.10934), do Alexey Bochkovskiy duy trì trong một [fork Darknet](https://github.com/AlexeyAB/darknet), có cùng tệp đó, chứ không dùng Unlicense như một số hướng dẫn giấy phép khẳng định. Bộ phân loại của chính GitHub cũng bó tay và báo "Other". Điều này đáng lưu ý nếu công cụ tuân thủ của bạn đọc trường đó: trình quét sẽ đánh dấu giấy phép là chưa xác định thay vì nhận diện đây là giấy phép permissive rõ ràng, dù nội dung khó có thể permissive hơn.

Bản thân Darknet cũng chưa hẳn đã chết. Repo của Redmon không có hoạt động từ năm 2022, nhưng README của AlexeyAB hiện trỏ tới [hank-ai/darknet](https://github.com/hank-ai/darknet) là bản kế nhiệm C/C++ được khuyến nghị và vẫn đang được duy trì tích cực.

**Cái bẫy nằm ở các bản port.** Hầu như không ai huấn luyện bằng Darknet trong năm 2026; mọi người thường chọn một bản triển khai lại PyTorch, và các bản đó có giấy phép riêng:

- [ultralytics/yolov3](https://github.com/ultralytics/yolov3), bản port YOLOv3 phổ biến nhất, dùng **AGPL-3.0**. Một kiến trúc thuộc public domain được phân phối lại theo giấy phép copyleft nghiêm ngặt nhất thường gặp trong lĩnh vực này. Giấy phép bạn nhận được hoàn toàn phụ thuộc vào bản port bạn cài bằng pip.
- [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) dùng **GPL-3.0**.
- [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) dùng **Apache-2.0**, vì vậy YOLOv4 vẫn có lựa chọn permissive khi chuyển sang PyTorch.
- [WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4), do đồng tác giả YOLOv4 phát hành, hoàn toàn **không có tệp LICENSE**. Điều này còn tệ hơn copyleft: khi không có giấy phép thì không có quyền nào được cấp, và mặc định mọi quyền đều được bảo lưu.

Hãy kiểm tra giấy phép của repo bạn đã clone, không phải số phiên bản trên bài báo.

## Kỷ nguyên 2: GPL-3.0 (YOLOv6, v7, v9)

[YOLOv6](https://github.com/meituan/YOLOv6) (Meituan), [YOLOv7](https://github.com/WongKinYiu/yolov7) và [YOLOv9](https://github.com/WongKinYiu/yolov9) được phát hành theo GPL-3.0.

GPL-3.0 là copyleft: nếu bạn phân phối phiên bản đã sửa đổi hoặc kết hợp mã GPL với mã của mình thành một chương trình duy nhất, bạn phải cung cấp toàn bộ mã nguồn tương ứng của tác phẩm kết hợp đó theo GPL-3.0. Có hai ranh giới cần lưu ý. **Chỉ tập hợp (mere aggregation)** có nghĩa là các chương trình độc lập, riêng biệt được phát hành cùng nhau không tự động trở thành một chương trình kết hợp. Và điều kiện kích hoạt là **phân phối**: khác với AGPL, nếu bạn không phát hành tác phẩm kết hợp thì GPL thông thường không buộc bạn công khai mã nguồn. Vì vậy một số công ty chỉ để mô hình GPL chạy phía sau API của riêng họ.

Con đường đó hẹp hơn vẻ ngoài và có ba điểm thường bị bỏ qua. Ngay khi mô hình được cài trên thiết bị khách hàng hoặc trong hệ thống on-prem, bạn đang phân phối nó. Tính chất "chỉ dùng nội bộ" không còn đúng khi một bản sao rời khỏi pháp nhân của bạn: giao một bản build cho nhà thầu, nhóm phát triển thuê ngoài hoặc công ty liên kết được thành lập riêng là phân phối, dù việc sao chép giữa các bộ phận trong cùng một công ty thì không. Và biện pháp bảo vệ này chỉ có hiệu lực nếu *mọi thứ* mà API đó truy cập đều dùng GPL hoặc giấy phép permissive hơn, vì chỉ cần một thành phần AGPL ở bất kỳ đâu trong tác phẩm được phục vụ là toàn bộ tác phẩm thuộc Section 13, bất kể tệp GPL ghi gì.

**YOLOv7 và YOLOv9 có lối thoát MIT đã mô tả ở trên.** YOLOv6 thì không, và bằng chứng mạnh nhất cho điều đó đến từ Baidu: PaddleDetection dùng Apache-2.0, còn người duy trì đã chủ động tách YOLOv5, YOLOv6, YOLOv7 và YOLOv8 vào repo GPL-3.0 riêng, [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO), đồng thời nêu rằng mã nguồn của các mô hình đó "will not be merged into PaddleDetection." Khi một công ty lớn như Baidu dựng ranh giới để giữ YOLOv6 khỏi framework permissive của họ, điều đó cho bạn biết giấy phép này có ý nghĩa thế nào.

## Kỷ nguyên 3: AGPL-3.0 (YOLOv5, v8, v10, 11, v12, v13, 26, YOLOE)

Ultralytics chuyển sang AGPL-3.0 vào **ngày 14 tháng 4 năm 2023**, không phải năm 2022 như nhiều nơi thường nhắc lại (kể cả các hướng dẫn xuất hiện ở trang đầu kết quả tìm kiếm cho câu hỏi này). Mọi phiên bản sau đó ([YOLOv8](https://github.com/ultralytics/ultralytics), YOLO11 và [YOLO26](https://arxiv.org/abs/2606.03748)) đều dùng AGPL-3.0, với Enterprise License trả phí làm lựa chọn thay thế. [Trang giấy phép](https://ultralytics.com/license) của họ cho biết gói Enterprise áp dụng cho "YOLO26, earlier YOLO versions, and any future YOLO models", vì vậy đây là chính sách có chủ đích và hướng tới tương lai. Giá không được công bố.

Có thể kiểm tra ngày này, và đáng để kiểm tra, vì cách kể phổ biến về sự kiện này sai ở một điểm quan trọng:

- Tệp `LICENSE` trong `ultralytics/yolov5` chỉ được sửa đúng ba lần trong lịch sử repo. Lần thứ ba là commit [`34cf749`](https://github.com/ultralytics/yolov5/commit/34cf749958d2dd3ed1205f6bb07e0f20f6e2372d), "Update LICENSE to AGPL-3.0 (#11359)", ngày **2023-04-14**, thay `GNU GENERAL PUBLIC LICENSE` bằng `GNU AFFERO GENERAL PUBLIC LICENSE` trong 101 tệp.
- Bản phát hành YOLOv5 cuối cùng của năm 2022, **v7.0 (ngày 22 tháng 11 năm 2022), vẫn dùng GPL-3.0**, cũng như v6.2, v6.1 và v6.0 trước đó. Hàng nghìn fork ngừng đồng bộ trước tháng 4 năm 2023 cũng vậy: chọn bất kỳ fork nào trong số đó và GitHub vẫn báo GPL-3.0 hôm nay.
- Cùng ngày, 41 phút sau, `ultralytics/ultralytics` cũng được [xử lý tương tự](https://github.com/ultralytics/ultralytics/commit/2c6fc0a4443b9cf805ef17b1cfdd71a98693b4d4) (PR #2031).

Từ đó có một sự thật hầu như không ai biết: **YOLOv8 không ra mắt dưới AGPL.** Mô hình được phát hành vào tháng 1 năm 2023 dưới **GPL-3.0**, rồi được đổi giấy phép ba tháng sau đó. Bản ghi PyPI không mơ hồ: các phiên bản `ultralytics` từ 8.0.0 (ngày 10 tháng 1 năm 2023) đến 8.0.76 (ngày 13 tháng 4 năm 2023) đều khai báo `GPL-3.0`. Phiên bản 8.0.80, tải lên ngày 16 tháng 4 năm 2023, là phiên bản đầu tiên khai báo `AGPL-3.0`.

Đây không phải kẽ hở. Giấy phép đã cấp không bị thu hồi hồi tố, vì vậy các phiên bản cũ vẫn có thể được dùng theo điều khoản tại thời điểm phát hành. Nhưng chúng đã cũ ba năm, không được duy trì và không có bản vá; GPL-3.0 vẫn là copyleft (bạn chỉ đổi điều khoản mạng lấy điều khoản phân phối, chứ không thoát copyleft), và lập trường thương mại của Ultralytics vốn rộng hơn nội dung giấy phép. Kết luận hữu ích hẹp hơn và khái quát hơn: **giấy phép gắn với phiên bản bạn nhận được, không gắn với tên dự án.** Hãy pin các dependency và ghi lại nội dung giấy phép vào ngày bạn lấy bản sao, vì dự án có thể thay đổi nó vào ngày mai, còn hồ sơ tuân thủ của bạn phụ thuộc vào chính phiên bản bạn đã nhận.

AGPL-3.0 là GPL cộng Section 13, điều khoản mạng: *nếu bạn sửa đổi Program*, người dùng tương tác với phiên bản đã sửa đổi của bạn qua mạng phải được cung cấp mã nguồn. Điều này đóng con đường SaaS mà GPL thông thường để ngỏ.

Hãy chú ý điều kiện này, vì phần lớn bài viết bỏ qua. Section 0 của AGPL định nghĩa "modify" là sao chép hoặc điều chỉnh tác phẩm *theo cách đòi hỏi sự cho phép về bản quyền*. Chạy mã huấn luyện không sửa đổi trên dataset của riêng bạn là thực thi chương trình, không phải điều chỉnh mã nguồn, cũng giống như biên dịch mã của bạn bằng một compiler GPL không sửa đổi không làm thay đổi compiler. Vì vậy chỉ riêng việc huấn luyện không tự động kích hoạt điều khoản, và ai nói như vậy là đã bỏ qua định nghĩa.

Điều đưa bạn vào phạm vi của giấy phép là kết hợp hoặc nhúng package vào codebase của bạn sao cho cả hai được phát hành hoặc phục vụ như một chương trình. Cách diễn giải của chính FSF cho rằng liên kết một tác phẩm GPL hoặc AGPL với các module khác tạo thành tác phẩm kết hợp; chưa có vụ việc nào kiểm nghiệm việc tòa án có mở rộng cách hiểu này tới mức import Python hay không. Cũng cần lưu ý rằng [điều khoản thương mại](https://ultralytics.com/license) riêng của Ultralytics yêu cầu Enterprise License đối với nền tảng SaaS, API và hệ thống cloud dùng YOLO ở phía sau, vì vậy lập trường của nhà cung cấp rộng hơn nội dung giấy phép thuần túy. Hãy xem việc tích hợp vào sản phẩm bạn cung cấp hoặc phát hành là rủi ro; đừng giả định rằng huấn luyện riêng lẻ cũng vậy.

Riêng với YOLO26, các mốc thời gian thường bị nhầm lẫn ở khắp nơi, vì vậy cần làm rõ: mô hình được **giới thiệu trước vào tháng 9 năm 2025** tại YOLO Vision, **thực sự phát hành vào tháng 1 năm 2026** (bản phát hành `ultralytics` 8.4.0, ghi chú nêu "Ultralytics YOLO26 has arrived"), và **bài báo được công bố vào tháng 6 năm 2026**. Mô hình nằm trong repo chính `ultralytics/ultralytics`; `ultralytics/yolo26` chỉ là trang giới thiệu.

**Điều khiến nhiều người bất ngờ:** các YOLO học thuật trong hai năm gần đây cũng dùng AGPL, và không phải do Ultralytics viết.

- [YOLOv10](https://arxiv.org/abs/2405.14458) của Đại học Thanh Hoa. README ghi *"The code base is built with ultralytics."*
- [YOLOv12](https://arxiv.org/abs/2502.12524): *"The code is based on ultralytics."*
- [YOLOv13](https://arxiv.org/abs/2506.17733): *"The code is based on Ultralytics."*
- [YOLOE](https://arxiv.org/abs/2503.07465), mô hình "see anything" open-vocabulary của nhóm YOLOv10, cũng dùng AGPL-3.0 và được xây dựng trên Ultralytics.

Copyleft được kế thừa: tác phẩm phái sinh từ mã AGPL phải được cung cấp theo AGPL, vì vậy ngay khi một fork trong số này được phát hành hoặc phục vụ, giấy phép đó cũng đi kèm. (Sửa đổi nó rồi chỉ giữ cho riêng mình thì không phát sinh nghĩa vụ nào, nên việc dùng trong nghiên cứu không bị ảnh hưởng.) Nghiên cứu là độc lập, giấy phép thì không. Từ năm 2024, xuất bản "YOLO tiếp theo" dưới dạng dẫn xuất của Ultralytics đã trở thành quy trình mặc định, đồng nghĩa AGPL giờ tự động lan truyền theo các số phiên bản.

Nếu bạn thấy trang nào cho rằng YOLOv13 dùng Apache-2.0 thì trang đó sai. Tệp LICENSE, GitHub API và model card đều ghi AGPL-3.0.

### Lựa chọn permissive cho YOLOv8 đã không còn tồn tại

Nhiều hướng dẫn vẫn còn trực tuyến, kể cả bản nháp trước của bài viết này, chỉ tới YOLOv8 của KerasCV dùng Apache-2.0 như một lựa chọn sạch để tránh AGPL YOLOv8. **Hiện nay bạn không thể dựa vào lựa chọn đó.** Hồ sơ công khai như sau:

- Trong phần thảo luận KerasCV [Clarifications regarding YOLOv8 licensing](https://github.com/keras-team/keras-cv/discussions/2032), một contributor viết rằng "many parts are derived form ultralytics." Không maintainer nào trả lời thread, nên vấn đề mức độ độc lập của triển khai chưa bao giờ được giải quyết công khai theo hướng nào.
- Issue KerasHub [add YOLOV8](https://github.com/keras-team/keras-hub/issues/1760) được một maintainer Keras đóng vào tháng 7 năm 2025 với nội dung: "Hey all!! Because of license issues, we have decided to drop this." Sau đó hai người hỏi câu này có ý nghĩa gì với người dùng thương mại hiện tại của phiên bản KerasCV. Cả hai đều không nhận được phản hồi.
- KerasCV hiện đã được lưu trữ, còn KerasHub, dự án kế nhiệm đang được duy trì, **không phát hành YOLOv8**.

Chúng tôi không biết liệu hai việc đó có liên quan với nhau hay không, và không khẳng định đã có điều gì sai trái xảy ra. Nhưng đó không phải nền tảng vững chắc để xây dựng sản phẩm.

## Các ngoại lệ permissive: YOLOX, PP-YOLOE, YOLO-NAS

Không phải YOLO nào cũng đi theo hướng copyleft.

- **[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)** (Megvii, 2021) dùng Apache-2.0 không có ngoại lệ, áp dụng cho cả mã nguồn lẫn trọng số. Đây vẫn là YOLO cổ điển phù hợp nhất cho mục đích thương mại, dù repo không có commit nào kể từ giữa năm 2025 và một [câu hỏi về việc đóng gói trọng số pretrained trong ứng dụng thương mại](https://github.com/Megvii-BaseDetection/YOLOX/issues/1865) đã mở từ tháng 3 năm 2026 mà chưa có phản hồi. Nội dung giấy phép rất rõ ràng. Hiện chỉ không còn ai ở đó để trả lời câu hỏi. Chúng tôi đã viết về [chạy YOLOX với LibreYOLO](/articles/yolox-with-libreyolo).
- **[PP-YOLOE](https://arxiv.org/abs/2203.16250)** (Baidu) dùng Apache-2.0 **trong PaddleDetection**. Hãy lấy mô hình từ đó, không phải PaddleYOLO, nơi chứa thư mục config gần như giống hệt nhưng dùng GPL-3.0. Cùng mô hình, cùng công ty, hai repo, hai giấy phép. Đây là tình huống YOLOv9 theo chiều ngược lại.
- **[YOLO-NAS](https://github.com/Deci-AI/super-gradients)** (Deci, 2023) là trường hợp dễ gây nhầm lẫn. Mã nguồn dùng Apache-2.0, còn trọng số chính thức thì không. [Giấy phép riêng](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md) của họ ghi rằng bạn *"may not use the Software for any commercial use, including in connection with any models used in a production environment."* Kể từ khi NVIDIA mua lại Deci vào năm 2024, chúng tôi không thấy repo có tính năng mới nào, còn URL trọng số ban đầu không còn hoạt động (checkpoint hiện nằm trên một máy chủ khác). Một cách xử lý đôi khi được đề xuất là huấn luyện kiến trúc từ đầu để giữ tình trạng Apache sạch, điều này có vẻ hợp lý nhưng chưa được Deci hay NVIDIA xác nhận, đồng thời không dễ dung hòa với nội dung giấy phép áp dụng tới "any components comprising the model." Đọc thêm về YOLO-NAS [tại đây](/articles/yolo-nas-with-libreyolo).
- **[MMYOLO](https://github.com/open-mmlab/mmyolo)** đôi khi được các hướng dẫn cũ liệt kê là Apache-2.0. Thực tế không phải vậy: MMYOLO dùng GPL-3.0 (dự án cùng hệ sinh thái là MMDetection mới dùng Apache), và repo không có commit nào kể từ tháng 7 năm 2024.
- **[YOLO-World](https://github.com/AILab-CVC/YOLO-World)** (Tencent) dùng **GPL-3.0**, không phải AGPL và cũng không permissive. Điều này đáng nói vì người ta thường mặc định YOLO open-vocabulary là permissive do liên hệ với hệ sinh thái kiểu CLIP mà chúng sử dụng.

Các trường hợp này bao quát ba lỗi phổ biến nhất về giấy phép trong lĩnh vực này: cho rằng trọng số theo giấy phép của mã nguồn, cho rằng mọi thứ từ cùng một tổ chức đều dùng chung giấy phép, và suy đoán giấy phép của mô hình từ tên gọi.

## Giấy phép bảo vệ mã nguồn, không bảo vệ ý tưởng (nhưng cần chú ý bằng sáng chế)

Một nguyên tắc là cơ sở pháp lý cho mọi lựa chọn permissive ở trên: **bản quyền bảo vệ cách thể hiện, không bảo vệ ý tưởng.** Đây không phải khẩu hiệu, mà là quy định pháp luật rõ ràng ở cả hai bờ Đại Tây Dương.

Tại Mỹ, đó là [17 U.S.C. 102(b)](https://www.law.cornell.edu/uscode/text/17/102), quy định loại trừ mọi "idea, procedure, process, system, method of operation, concept, principle, or discovery" khỏi phạm vi bảo hộ bản quyền, bắt nguồn từ vụ *Baker v. Selden* (1879). Tại EU, nơi luật áp dụng cho chúng tôi và có lẽ cho nhiều độc giả, Điều 1(2) của [Chỉ thị Phần mềm (2009/24/EC)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32009L0024) nêu rằng "ideas and principles which underlie any element of a computer program" không được bảo vệ, và Tòa án Công lý đã xác nhận trong vụ *SAS Institute v World Programming* ([C-406/10](https://curia.europa.eu/juris/liste.jsf?num=C-406/10), năm 2012) rằng chức năng của chương trình, ngôn ngữ lập trình và định dạng dữ liệu tự chúng không phải là cách thể hiện được bảo vệ. Chỉ mã nguồn mới được bảo vệ.

Kiến trúc được mô tả trong bài báo arXiv là một ý tưởng đã công bố. Bất kỳ ai cũng có thể triển khai nó. Điều bạn không thể làm là sao chép hoặc điều chỉnh các tệp mã nguồn GPL/AGPL của người khác rồi đổi giấy phép cho kết quả. Một bản triển khai độc lập, do những người dựa trên bài báo chứ không dựa trên mã nguồn viết ra, là tác phẩm mới và tác giả có thể chọn giấy phép.

Hai lưu ý thẳng thắn mà phần lớn nhà cung cấp bỏ qua:

- **Tính "độc lập" phải là thật.** Cách làm có thể bảo vệ được là quy trình clean-room: người viết mã làm việc từ bài báo và đặc tả, không dựa trên mã nguồn copyleft. Đọc kỹ repo GPL rồi "viết lại" không giống như vậy, và khác biệt này rất quan trọng nếu có người xem xét.
- **Đây là lập luận về bản quyền, không phải bằng sáng chế.** Sáng chế độc lập *không* phải biện pháp bảo vệ trước bằng sáng chế. Triển khai lại theo quy trình clean-room giải quyết câu hỏi bản quyền và để nguyên câu hỏi bằng sáng chế. Hiện không có bằng sáng chế nào được khẳng định rộng rãi là bao trùm các kiến trúc YOLO, nhưng "chưa ai kiện" không đồng nghĩa với "không có gì để kiện".

## Trọng số là một giấy phép thứ hai

Sai lầm tốn kém nhất trong lĩnh vực này là kiểm tra mã nguồn rồi quên checkpoint. Có ba lớp giấy phép và chúng có thể khác nhau:

**1. Mã nguồn.** Tất cả nội dung bên trên.

**2. Trọng số.** Đôi khi được nêu rõ, như YOLO-NAS. Đôi khi chỉ được khẳng định: [trang giấy phép](https://ultralytics.com/license) của Ultralytics nêu rằng AGPL-3.0 "covers the training code and the models produced by that training code", đồng thời áp dụng cho các mô hình bạn tự huấn luyện bằng mã của họ. Với tư cách chủ sở hữu bản quyền của những gì họ phát hành, họ có thể đặt điều khoản cho checkpoint của mình. Việc trọng số *do bạn* tạo ra có phải là tác phẩm phái sinh hợp pháp từ mã huấn luyện hay không thực sự vẫn là câu hỏi mở: chưa có án lệ, và điều này đi ngược quy tắc chung của chính FSF rằng đầu ra của chương trình không tự động được bản quyền của chương trình bao phủ. Tuy chưa ngã ngũ không có nghĩa là an toàn. Trên thực tế, hãy xem trọng số AGPL đã phát hành là có ràng buộc.

Điểm rủi ro trực tiếp với bất kỳ ai làm theo lời khuyên trong bài này: **các repo YOLOv7 và YOLOv9 dùng GPL hoàn toàn không nói gì về trọng số của chúng.** Checkpoint của họ là tài sản phát hành trong một repo GPL-3.0 mà không có cấp phép riêng. Tải các tệp .pt đó vào bản triển khai lại MIT sẽ giúp mã nguồn sạch, nhưng checkpoint vẫn chưa rõ tình trạng pháp lý. Nếu bạn đổi sang repo khác vì lý do giấy phép, hãy dùng trọng số do chính dự án permissive huấn luyện hoặc tự huấn luyện.

**3. Dữ liệu huấn luyện.** Các *annotation* của COCO dùng CC BY 4.0, vì vậy nhãn ổn. *Ảnh* không thuộc quyền cấp phép của COCO: đó là ảnh Flickr với nhiều điều khoản riêng lẻ khác nhau, và liên minh COCO nói rõ họ không sở hữu chúng. Phần lớn ngành xem đây là rủi ro thấp rồi tiếp tục, nhưng "COCO ổn" chỉ đúng với annotation. Ngoài COCO, yêu cầu thường nghiêm ngặt hơn nhiều: không ít dataset ảnh hàng không, y tế và lái xe không cho mục đích thương mại, và giấy phép permissive của mã mô hình không xóa được các điều khoản đó.

Mỗi lần, hãy kiểm tra cả ba lớp.

## Ghi chú về MIT và Apache-2.0

Chúng tôi phát hành thư viện dùng MIT, vì vậy sẽ tiện nếu nói MIT đơn giản là giấy phép tốt nhất. So sánh trung thực sẽ thú vị hơn.

MIT không phải không có nghĩa vụ: bạn phải giữ thông báo bản quyền và nội dung giấy phép trong các bản sao mình phân phối. Đó là một điều kiện thực sự, chỉ có chi phí tuân thủ thấp. Apache-2.0 có một điều mà MIT không có: **cấp phép bằng sáng chế rõ ràng** từ mọi contributor, cộng với điều khoản trả đũa chấm dứt quyền cấp phép của bất kỳ ai kiện liên quan tới tác phẩm. MIT không đề cập đến bằng sáng chế. Với công ty lo ngại nhất về rủi ro bằng sáng chế, có thể nói Apache-2.0 *an toàn hơn* trong hai lựa chọn, và phần lớn hệ sinh thái phát hiện permissive (YOLOX, RT-DETR, D-FINE, DEIM) dùng Apache-2.0 chính vì lý do đó.

Điều MIT mang lại là sự đơn giản và tính tương thích. Không giấy phép nào trong hai loại này buộc bạn công bố mã nguồn, đây mới là khía cạnh mà bài viết thực sự bàn tới. Nếu ai đó nói MIT so với Apache là quyết định quan trọng, họ đang bán thứ gì đó. Quyết định quan trọng là permissive hay copyleft.

## Vậy thực sự có thể dùng YOLO nào?

- **Sản phẩm thương mại mã nguồn đóng:** YOLOv7 hoặc YOLOv9 qua [repo MIT](https://github.com/MultimediaTechLab/YOLO) của phòng thí nghiệm, YOLOX hoặc PP-YOLOE (từ PaddleDetection) trong số các bản gốc, hoặc framework MIT đang được duy trì như LibreYOLO nếu bạn không muốn tự lắp ghép và kiểm tra mọi thứ. Tránh mọi thứ dùng GPL hoặc AGPL, trừ khi bạn sẽ mở mã nguồn toàn bộ ứng dụng hoặc mua Enterprise License.
- **Dự án mã nguồn mở:** có thể dùng bất kỳ phiên bản nào miễn là giấy phép của bạn tương thích. Nếu dự án dùng AGPL thì dòng Ultralytics là lựa chọn phù hợp, và các nghiên cứu mới nhất (YOLOv13, YOLO26, YOLOE) xuất hiện ở đó trước tiên.
- **Nghiên cứu và benchmark:** giấy phép hầu như không hạn chế bạn. Hãy dùng những gì bài báo bạn đang so sánh đã dùng.
- **Kế hoạch "để sau tính":** không hiệu quả. Gỡ dependency AGPL sau khi xây dựng sản phẩm đồng nghĩa phải huấn luyện lại, xác thực lại và xuất lại mọi thứ, kể cả trọng số. Hãy chọn repo, và do đó chọn giấy phép, trước tiên.

Để so sánh các framework chứ không phải giấy phép của chúng, xem [Các lựa chọn thay thế Ultralytics tốt nhất năm 2026](/articles/best-ultralytics-alternatives).

## Lựa chọn MIT: LibreYOLO phát hành những gì

Công khai: LibreYOLO là dự án của chúng tôi, và giấy phép của dự án là lý do bài viết này ra đời.

**[LibreYOLO](https://github.com/LibreYOLO/libreyolo) là một codebase dùng MIT** bao phủ các bộ phát hiện bên dưới thông qua một API duy nhất, tích hợp sẵn huấn luyện và xuất sang ONNX, TensorRT, OpenVINO và NCNN. Không copyleft, không điều khoản mạng, không gói doanh nghiệp.

Dưới đây là danh sách đầy đủ các mô hình phát hiện đối tượng cùng upstream và giấy phép của từng mô hình, để bạn thấy chính xác mỗi lựa chọn permissive đến từ đâu:

| Trong LibreYOLO | Mô hình | Upstream | Giấy phép upstream |
| --- | --- | --- | --- |
| `LibreYOLO2`, `LibreYOLO3`, `LibreYOLO4` | Các YOLO thời Darknet, chạy bằng PyTorch | [pjreddie/darknet](https://github.com/pjreddie/darknet), [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) | Public domain |
| `LibreYOLO7` | YOLOv7 | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT (bản viết lại của chính phòng thí nghiệm, không phải repo GPL) |
| `LibreYOLO9`, `LibreYOLO9E2E` | YOLOv9, cùng biến thể end-to-end không cần NMS | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT (bản viết lại của chính phòng thí nghiệm, không phải repo GPL) |
| `LibreYOLO9P2` | YOLOv9 với head stride-4 cho vật thể nhỏ | Bản gốc LibreYOLO | MIT |
| `LibreYOLOX` | YOLOX | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache-2.0 |
| `LibreYOLONAS` | YOLO-NAS, phát hiện và pose | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) | Mã nguồn Apache-2.0. Không phân phối lại trọng số upstream bị hạn chế |
| `LibreRTDETR`, `LibreRTDETRv2` | RT-DETR và v2 | [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | Apache-2.0 |
| `LibreRTDETRv4` | RT-DETRv4 | [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | Apache-2.0 |
| `LibreRFDETR` | RF-DETR, phát hiện, phân đoạn, pose, OBB | [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | Apache-2.0 cho N/S/M/L. XL/2XL dùng Platform Model License của Roboflow, vì vậy chúng tôi chỉ phát hành các kích thước Apache |
| `LibreDFINE` | D-FINE | [Peterande/D-FINE](https://github.com/Peterande/D-FINE) | Apache-2.0 |
| `LibreDEIM` | DEIM | [Intellindust-AI-Lab/DEIM](https://github.com/Intellindust-AI-Lab/DEIM) | Apache-2.0 |
| `LibreDEIMv2` | DEIMv2 | [Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) | Mã nguồn Apache-2.0. Các kích thước lớn hơn dùng backbone DINOv3 của Meta theo giấy phép phi OSI riêng của Meta. Giấy phép cho phép phân phối lại nhưng **cấm sử dụng cho mục đích quân sự, hạt nhân, gián điệp và vũ khí**. Các kích thước đó được công bố là `other`, không phải Apache. Hãy đọc kỹ trước khi phát hành sản phẩm lưỡng dụng |
| `LibreRTMDet` | RTMDet ([không cần MMDetection](/articles/rtmdet-without-mmdetection)) | [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) | Apache-2.0 (nhánh Apache, không phải mmyolo dùng GPL-3.0) |
| `LibrePICODET` | PP-PicoDet | Kiến trúc từ [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), checkpoint qua bản port lại [Picodet_Pytorch](https://github.com/Bo396543018/Picodet_Pytorch) | Apache-2.0 (cả hai) |
| `LibreEC` | EdgeCrafter, phát hiện, pose, phân đoạn | [Intellindust-AI-Lab/EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter) | Apache-2.0 |
| `LibreFOMO` | Bộ phát hiện centroid cho phần cứng cấp vi điều khiển | Bản gốc LibreYOLO | MIT |
| `LibreGroundingDINO` | Grounding DINO, open-vocabulary (inference) | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | Apache-2.0 |
| `LibreOWLv2` | OWLv2, open-vocabulary (inference) | Google, qua `transformers` | Apache-2.0 |

Có hai điểm trong cột này được trình bày một cách thẳng thắn có chủ ý.

**Không sao chép mã nguồn GPL hoặc AGPL vào codebase.** Mỗi dòng mô hình ở trên được xây dựng từ upstream public domain, MIT hoặc Apache-2.0, đây là điều giúp framework giữ giấy phép MIT. Khi kiến trúc có cả phiên bản permissive lẫn copyleft, chúng tôi chọn phiên bản permissive: YOLOv9 và YOLOv7 của chúng tôi bắt nguồn từ repo MIT của phòng thí nghiệm, không phải các bản GPL gốc; RTMDet của chúng tôi dựa trên MMDetection dùng Apache thay vì MMYOLO dùng GPL. YOLO-World không có trong danh sách dù thường được hỏi đến, vì nó dùng GPL-3.0 và không có nguồn permissive nào để lấy.

**Permissive không đồng nghĩa không bị hạn chế, và chúng tôi muốn nói rõ điều đó để bạn không phát hiện ra trong một cuộc kiểm toán.** Các kích thước lớn hơn của DEIMv2 kế thừa điều khoản DINOv3 của Meta, gồm lệnh cấm sử dụng cho mục đích quân sự, hạt nhân, gián điệp và vũ khí, đây là hạn chế thực sự nếu bạn làm việc trong lĩnh vực lưỡng dụng. Một số checkpoint xem trước cho nghiên cứu được huấn luyện trên dataset phi thương mại (DOTA, VisDrone) và được gắn nhãn `cc-by-nc` trên Hugging Face thay vì âm thầm phát hành như thể chúng miễn phí. Cùng sự thận trọng áp dụng với trọng số của người khác cũng áp dụng với trọng số của chúng tôi: checkpoint YOLO9 được chuyển đổi từ checkpoint của repo MIT thuộc phòng thí nghiệm, vì vậy chúng kế thừa tình trạng của các checkpoint đó, như đã nói ở trên, là "được khẳng định, chưa được kiểm toán chính thức." Giấy phép MIT bao phủ mã chúng tôi viết. Từng checkpoint có điều khoản riêng dựa trên dữ liệu và nguồn gốc huấn luyện, được công bố theo từng trọng số.

Ngoài phát hiện, cùng API còn hỗ trợ phân đoạn thực thể và ngữ nghĩa, pose, bounding box xoay, phân loại (MobileNetV4, ConvNeXt, EfficientNetV2, ResNet, CLIP), ước lượng độ sâu đơn ảnh, phục hồi ảnh và ước lượng hướng nhìn.

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
```

Cùng quy trình YOLO, không phải lo phần giấy phép.

Gắn sao trên GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Tài liệu: [libreyolo.com/docs](https://libreyolo.com/docs)

---

*Thương hiệu và mối liên kết: YOLO, YOLOv8, YOLO11, YOLO26 và Ultralytics là thương hiệu của chủ sở hữu tương ứng, bao gồm Ultralytics Inc. LibreYOLO là dự án độc lập, không liên kết, không được tài trợ hay xác nhận bởi Ultralytics hoặc bất kỳ tổ chức nào khác được nêu trong bài viết. Tên sản phẩm và repo chỉ được dùng để nhận diện và so sánh phần mềm được đề cập.*

*Bài viết này cung cấp thông tin tổng quát, không phải tư vấn pháp lý, cập nhật đến ngày 11 tháng 7 năm 2026 và do một trong các nhà cung cấp được so sánh viết. Hãy xác minh giấy phép hiện tại trước khi phát hành.*
