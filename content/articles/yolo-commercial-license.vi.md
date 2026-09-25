---
title: YOLO có miễn phí cho mục đích thương mại không? Giấy phép YOLOv8, YOLO11 và YOLO26
description: "YOLOv5, YOLOv8, YOLO11 và YOLO26 được phát hành theo AGPL-3.0, nên không miễn phí cho mục đích thương mại với mã nguồn đóng. Đây là những gì giấy phép thực sự yêu cầu theo từng phiên bản, cùng lựa chọn thay thế dùng giấy phép MIT."
date: 2026-07-04
author: Xuban
tags: [LibreYOLO, yolo-license, agpl, commercial-use, yolov8, yolo11, yolo26, mit-license]
faq:
  - q: "YOLOv8 có miễn phí cho mục đích thương mại không?"
    a: "Không dành cho sản phẩm mã nguồn đóng. YOLOv8 dùng AGPL-3.0, vì vậy bạn phải phát hành toàn bộ mã nguồn ứng dụng theo AGPL-3.0 hoặc mua Enterprise License có phí của nhà cung cấp. Để sử dụng thương mại miễn phí mà không bị hạn chế, hãy dùng thư viện có giấy phép MIT như LibreYOLO."
  - q: "YOLO dùng giấy phép nào?"
    a: "Các bản phát hành phổ biến, YOLOv5, YOLOv8, YOLO11 và YOLO26, mặc định dùng AGPL-3.0. AGPL-3.0 là giấy phép copyleft mạnh: đây là giấy phép mã nguồn mở, nhưng không cho phép sử dụng miễn phí bên trong một sản phẩm độc quyền."
  - q: "Tôi có thể dùng YOLO trong sản phẩm thương mại mã nguồn đóng không?"
    a: "Không thể dùng theo AGPL-3.0 nếu không tuân thủ, nghĩa là phải mở mã nguồn toàn bộ sản phẩm theo AGPL-3.0. Để giữ mã nguồn riêng tư, bạn phải mua Enterprise License của nhà cung cấp hoặc chuyển sang framework dùng giấy phép ít hạn chế hơn như LibreYOLO (MIT)."
  - q: "Có phiên bản YOLO nào không dùng AGPL không?"
    a: "Có. LibreYOLO dùng giấy phép MIT, nên không có copyleft AGPL hay điều khoản sử dụng qua mạng. Bạn có thể tự do dùng thư viện này trong phần mềm thương mại và mã nguồn đóng mà không mất phí."
  - q: "Giấy phép thương mại YOLO có giá bao nhiêu?"
    a: "Enterprise License của nhà cung cấp là thỏa thuận có phí, áp dụng theo từng công ty và có mức giá không công khai. LibreYOLO không mất phí: giấy phép MIT cho phép sử dụng thương mại miễn phí."
  - q: "Chạy YOLO chỉ trên máy chủ của tôi có tránh được AGPL không?"
    a: "Không. Mục 13 của AGPL-3.0 áp dụng cho tương tác qua mạng. Nếu người dùng truy cập phiên bản đã sửa đổi của bạn qua mạng, bạn phải đề nghị cung cấp mã nguồn tương ứng cho họ. Chạy phần mềm chỉ trên máy chủ của bạn không phải là trường hợp được miễn trừ."
---

**Không. Các mô hình YOLO phổ biến, YOLOv5, YOLOv8, YOLO11 và YOLO26 mới, không miễn phí cho mục đích thương mại với mã nguồn đóng.** Chúng được phát hành theo **AGPL-3.0**, một giấy phép copyleft mạnh. Nếu xây dựng sản phẩm dựa trên một trong số đó, bạn phải mở mã nguồn *toàn bộ* ứng dụng theo AGPL-3.0 hoặc mua Enterprise License có phí của nhà cung cấp. Nếu cả hai lựa chọn đều không phù hợp, có một lựa chọn thay thế dùng giấy phép ít hạn chế hơn: **[LibreYOLO](/) dùng giấy phép MIT và miễn phí cho mọi mục đích thương mại, không kèm điều kiện ràng buộc.**

Nếu bạn tìm kiếm "is YOLOv8 free for commercial use," "YOLO license" hoặc "YOLO without AGPL," trang này trả lời chính xác câu hỏi đó theo từng phiên bản.

## Tổng quan nhanh về giấy phép YOLO

| Mô hình | Giấy phép mặc định | Miễn phí cho mục đích thương mại **mã nguồn đóng**? | Bạn cần làm gì |
| --- | --- | --- | --- |
| YOLOv5 | AGPL-3.0 | Không | Mở mã nguồn toàn bộ ứng dụng hoặc mua Enterprise License |
| YOLOv8 | AGPL-3.0 | Không | Mở mã nguồn toàn bộ ứng dụng hoặc mua Enterprise License |
| YOLO11 | AGPL-3.0 | Không | Mở mã nguồn toàn bộ ứng dụng hoặc mua Enterprise License |
| YOLO26 | AGPL-3.0 | Không | Mở mã nguồn toàn bộ ứng dụng hoặc mua Enterprise License |
| **LibreYOLO** | **MIT** | **Có** | **Không cần làm gì. Đưa vào sản phẩm độc quyền, miễn phí.** |

Tóm tắt trong một câu: các mô hình YOLO phổ biến là mã nguồn mở, nhưng dùng **AGPL-3.0**, một giấy phép copyleft mà hầu hết công ty không thể tuân thủ trong sản phẩm độc quyền. LibreYOLO cung cấp workflow YOLO tương tự theo **giấy phép MIT**, không đặt ra nghĩa vụ như vậy cho bạn.

## YOLO dùng giấy phép nào?

Các mô hình YOLO được dùng rộng rãi, **YOLOv5, YOLOv8, YOLO11 và YOLO26 mới nhất**, được nhà cung cấp phân phối theo **GNU Affero General Public License v3.0 (AGPL-3.0)**.

AGPL-3.0 là giấy phép *copyleft mạnh*. Đây không phải loại giấy phép mã nguồn mở dễ dãi theo kiểu "muốn làm gì thì làm" như MIT hay Apache-2.0. Giấy phép đặt ra một nghĩa vụ cụ thể, phạm vi rộng, và chính nghĩa vụ đó gây khó khăn cho người dùng thương mại.

## YOLOv8 có miễn phí cho mục đích thương mại không?

Không theo cách mà đa số mọi người hiểu. Bạn *có thể* tải YOLOv8 và sử dụng cho mục đích thương mại, nhưng chỉ khi tuân thủ AGPL-3.0. Trên thực tế, điều đó có nghĩa là:

- Nếu bạn phân phối, phát hành hoặc lưu trữ một sản phẩm có tích hợp YOLOv8, bạn phải **công bố mã nguồn đầy đủ của sản phẩm đó**, gồm ứng dụng, mã tích hợp, script và cấu hình, theo AGPL-3.0.
- Nghĩa vụ này phát sinh không chỉ khi phát hành phần mềm cho người dùng, mà còn khi cho phép người dùng tương tác với phần mềm **qua mạng**, theo điều khoản Mục 13 đặc trưng của AGPL. Backend SaaS hoặc API cũng không được miễn trừ.

Với dự án cá nhân hoặc nghiên cứu học thuật, điều này thường không thành vấn đề. Với sản phẩm thương mại độc quyền, công bố toàn bộ codebase hầu như luôn là điều không thể chấp nhận. Tất cả nội dung này cũng áp dụng tương tự cho **YOLO11** và **YOLO26**: cùng giấy phép, cùng nghĩa vụ.

## YOLOv5, YOLO11 và YOLO26 thì sao?

Tình hình cũng vậy. Tất cả đều mặc định dùng **AGPL-3.0**. Enterprise License của nhà cung cấp nêu rõ phạm vi bao gồm "YOLO26, các phiên bản YOLO trước đó và mọi mô hình YOLO trong tương lai", cho thấy cách cấp phép được áp dụng có chủ đích và nhất quán trong toàn bộ dòng sản phẩm. Chọn phiên bản mới hơn không giúp bạn có giấy phép dễ chịu hơn.

Để đầy đủ: không phải *mọi* mô hình có tên "YOLO" đều dùng AGPL. Một số biến thể cộng đồng dùng GPL-3.0 hoặc Apache-2.0, và giấy phép tùy theo tác giả. Nhưng các bản phát hành mà mọi người muốn nói đến khi gõ "YOLOv8" hay "YOLO11" đều dùng AGPL-3.0.

## AGPL-3.0 thực sự yêu cầu gì, nói đơn giản

AGPL-3.0 mở rộng GPL bằng một bổ sung quan trọng: **điều khoản sử dụng qua mạng**. Danh sách thực tế cần nhớ:

1. **Phân phối mã nguồn.** Nếu bạn chuyển giao phần mềm có chứa mã AGPL, bạn phải cung cấp *mã nguồn tương ứng đầy đủ* của toàn bộ tác phẩm kết hợp theo AGPL-3.0.
2. **Sử dụng qua mạng cũng được tính.** Nếu người dùng tương tác với phiên bản bạn đã sửa đổi qua mạng, chẳng hạn ứng dụng web, API hoặc SaaS, bạn phải đề nghị cung cấp chính mã nguồn đó cho họ. Không có ngoại lệ kiểu "chúng tôi chỉ chạy trên máy chủ của mình".
3. **Copyleft có phạm vi lan rộng.** Nghĩa vụ áp dụng cho *toàn bộ tác phẩm phái sinh*, không chỉ các tệp YOLO. Logic nghiệp vụ của bạn cũng nằm trong phạm vi AGPL.

Đây là thông tin chung, không phải tư vấn pháp lý, vì vậy hãy trao đổi với luật sư về trường hợp cụ thể của bạn. Nhưng kết luận rất đơn giản: **AGPL-3.0 và phần mềm thương mại mã nguồn đóng không tương thích với nhau.**

## Lựa chọn Enterprise License và điểm cần lưu ý

Nhà cung cấp có Enterprise License trả phí để loại bỏ các nghĩa vụ AGPL, cho phép bạn tích hợp YOLO vào sản phẩm độc quyền mà không cần mở mã nguồn. Đây là một hướng đi hợp lệ và phù hợp với một số công ty.

Các điểm cần lưu ý:

- **Tốn phí**, là khoản phí thương mại định kỳ, được thương lượng theo từng công ty và không công khai.
- **Phụ thuộc vào điều khoản của một nhà cung cấp**, các điều khoản này có thể thay đổi và chỉ bao gồm mã của nhà cung cấp đó.
- **Bạn trả tiền để được phép sử dụng** một kiến trúc mô hình mà về cốt lõi đã được công bố trong nghiên cứu.

Nếu bạn chấp nhận khoản phí định kỳ và sự phụ thuộc vào nhà cung cấp, Enterprise License sẽ phù hợp. Nếu bạn không muốn trả phí hoặc không muốn gánh nghĩa vụ này, hãy đọc tiếp.

## Lựa chọn MIT: LibreYOLO

Minh bạch: LibreYOLO là sản phẩm của chúng tôi. Đây cũng là lý do trang này có thể đưa ra câu trả lời rõ ràng thay vì chỉ nhún vai.

**LibreYOLO là framework YOLO được phát hành theo giấy phép MIT ít hạn chế.** Chỉ riêng điều đó đã thay đổi hoàn toàn cách sử dụng thương mại:

- **Miễn phí cho mục đích thương mại**, bao gồm cả sản phẩm độc quyền, mã nguồn đóng.
- **Không copyleft, không điều khoản sử dụng qua mạng.** Bạn không bao giờ phải mở mã nguồn ứng dụng.
- **Không phí theo người dùng hay phí doanh nghiệp.** MIT nghĩa là MIT.
- **Workflow quen thuộc**, nên việc chuyển đổi khá đơn giản.

LibreYOLO là framework được duy trì thực sự, không phải wrapper: phát hiện đối tượng, phân đoạn, tư thế, phân loại, độ sâu và nhiều tính năng khác, trong cùng một API quen thuộc, tích hợp sẵn huấn luyện và xuất sang ONNX/TensorRT/OpenVINO/NCNN. Workflow không khác biệt. Giấy phép mới là điểm khác biệt.

Nếu bạn đến đây vì câu hỏi rộng hơn về "rời bỏ YOLO dùng AGPL", chúng tôi đã viết bài so sánh đầy đủ hơn về hệ sinh thái trong [Các lựa chọn thay thế Ultralytics tốt nhất năm 2026](/articles/best-ultralytics-alternatives).

Để xem giấy phép của mọi phiên bản YOLO khác, từ các bản Darknet gốc thuộc phạm vi công cộng đến YOLOv9, YOLOv10, YOLOv12, YOLOv13 và YOLO26, hãy xem [Giải thích giấy phép của mọi phiên bản YOLO](/articles/yolo-licenses-explained).

### Vì sao MIT quan trọng với doanh nghiệp

Giấy phép MIT cho phép bạn sử dụng, sửa đổi, tích hợp và bán phần mềm được xây dựng trên LibreYOLO mà **không có nghĩa vụ công bố mã nguồn** và **không mất phí**. Đây là giấy phép được dùng trong phần lớn nền tảng phần mềm hiện đại chính vì an toàn cho việc áp dụng trong thương mại. Sản phẩm thuộc về bạn, bạn không phải chịu nghĩa vụ nào.

## Dùng thử

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

Bạn vẫn giữ trải nghiệm phát triển với YOLO: huấn luyện trên dữ liệu của mình, dự đoán và xuất mô hình để triển khai. Bạn loại bỏ nghĩa vụ AGPL và hóa đơn Enterprise.

LibreYOLO dùng giấy phép MIT, chạy trên Linux, Mac và Windows, đồng thời hoạt động trên GPU, Apple Silicon và CPU thông thường mà không cần sửa code. Một API hỗ trợ YOLO9, RF-DETR, RTMDet, YOLOX, D-FINE, dòng RT-DETR, phân đoạn, tư thế, độ sâu và nhiều tính năng khác, tất cả đều dùng giấy phép dễ dãi.

Đánh sao trên GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Tài liệu: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
