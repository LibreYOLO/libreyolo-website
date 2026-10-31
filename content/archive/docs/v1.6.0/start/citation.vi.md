---
title: Trích dẫn
seo_title: Trích dẫn LibreYOLO và các tác giả upstream
description: >-
  Cách trích dẫn LibreYOLO trong bài báo và cách trích dẫn tác giả của họ mô
  hình bạn đã chạy. Cả hai đều thuộc cùng một phần phương pháp.
lead: >-
  Một trích dẫn LibreYOLO đầy đủ gồm hai phần: thư viện và công trình đã công bố
  đứng sau họ mô hình tạo ra kết quả.
keywords:
  - cách trích dẫn libreyolo
  - bibtex libreyolo
  - citation cff libreyolo
  - trích dẫn mô hình
  - trích dẫn thị giác máy tính
last_verified: 1.5.0
source_hash: 0f3f23e4e85e38be
---

## Trích dẫn LibreYOLO

Repo công bố metadata trích dẫn trong
[`CITATION.cff`](https://github.com/LibreYOLO/libreyolo/blob/release/CITATION.cff),
không phải dưới dạng một block BibTeX. GitHub đọc file này và cung cấp nút Cite
this repository trên trang repo, từ đó tạo định dạng APA và BibTeX. Hãy lấy mục
trích dẫn từ đó thay vì tự nhập.

Nội dung đầy đủ của file:

```yaml
cff-version: 1.2.0
message: "If you use LibreYOLO in your research or software, please cite it as below."
title: "LibreYOLO"
type: software
authors:
  - family-names: Ceccon
    given-names: Xuban
  - name: "The LibreYOLO contributors"
license: MIT
url: "https://github.com/LibreYOLO/libreyolo"
repository-code: "https://github.com/LibreYOLO/libreyolo"
```

File này cố ý không chứa phiên bản và ngày phát hành.
[`RELEASING.md`](https://github.com/LibreYOLO/libreyolo/blob/release/RELEASING.md)
yêu cầu maintainer không bao giờ tăng phiên bản, thêm ngày hoặc đổi tiêu đề
`CITATION.cff` hay `.zenodo.json` trong lúc phát hành, để mọi trích dẫn đều trỏ
về một bản ghi thay vì phân tán theo phiên bản. Hãy ghi phiên bản bạn đã chạy
trong nội dung của mình và giữ nguyên trích dẫn.

## Trích dẫn họ mô hình

LibreYOLO là một bản port. Chạy `LibreRFDETRm.pt` có nghĩa là chạy RF-DETR, và
người viết RF-DETR mới là những người mà reviewer mong đợi được ghi công. Chỉ
trích dẫn thư viện sẽ gán công trình của họ cho sai dự án.

Mọi thông tin cần thiết đều nằm trên trang của họ mô hình. Hàng Upstream trong
phần đầu trang nêu tên công trình gốc và tổ chức đứng sau, đồng thời liên kết
đến bài báo và repo mã nguồn. Phần Trích dẫn ở bên dưới chứa BibTeX.

BibTeX đó được sao chép nguyên văn từ block trích dẫn của chính tác giả, thường
là phần Citation trong README upstream hoặc một `CITATION.cff`, và được hiển
thị kèm liên kết trở lại block nguồn để bạn có thể đối chiếu. Nó không bao giờ
được ghép từ metadata của bài báo. Một mục được dựng lại thủ công có thể sai
một cách khó nhận biết và gây hậu quả lớn: thiếu đồng tác giả, sai hội nghị,
sai loại mục hoặc dùng năm của bản preprint. Preprint cũng có thể được chấp
nhận sau đó, nên một mục có thể là `@inproceedings` dù phiên bản bạn đọc nằm
trên arXiv.

Hãy sao chép nguyên block. Nếu kiểu tài liệu tham khảo của bạn cần loại mục
khác, hãy chuyển đổi mục đó thay vì nhập lại và giữ nguyên thứ tự tác giả.

## Phần phương pháp cần những gì

Ba yếu tố giúp kết quả LibreYOLO có thể tái lập và được ghi công chính xác:

- Thư viện, được trích dẫn từ `CITATION.cff`, cùng phiên bản bạn đã chạy.
  `libreyolo version` in ra thông tin này cùng các phiên bản Python, torch và
  CUDA đang được sử dụng.
- Công trình upstream, được trích dẫn từ phần Citation trên trang của họ mô hình.
- Tên file checkpoint chính xác, chẳng hạn `LibreRFDETRm.pt`. Các kích thước
  trong cùng một họ hoạt động khác nhau, và một số họ công bố checkpoint được
  huấn luyện trên các dataset khác nhau với cùng prefix, nên chỉ tên họ không
  đủ để xác định mô hình đã chạy.

Ghi công cũng là một điều khoản giấy phép đối với phần lớn nội dung LibreYOLO
công bố. Apache-2.0 và họ CC BY đều yêu cầu thông báo phải đi cùng trọng số bạn
phân phối lại, đây là nghĩa vụ riêng biệt với việc trích dẫn bài báo. Xem
[giấy phép](/docs/licensing) để biết điều khoản áp dụng cho từng checkpoint.
