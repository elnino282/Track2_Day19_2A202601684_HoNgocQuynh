# Reflection — Lab 19

**Tên:** Hồ Ngọc Quỳnh
**Cohort:** A20
**Path:** Lite
**Bonus:** Đã thực hiện — Hybrid AI Memory (`bonus/`)

Trên 50 golden queries, Hybrid đạt Precision@10 cao nhất (78,6%), hơn BM25
(77,8%) và Vector (73,2%). Với `mixed`, Hybrid thắng ở 100% nhờ RRF kết hợp
từ khóa và ngữ nghĩa. Với `exact`, BM25 và Hybrid cùng đạt 96,7%; tên riêng,
mã lỗi hoặc thuật ngữ cần khớp chính xác không cần thêm chi phí vector.

Ở `paraphrase`, Path Lite dùng `bge-small-en-v1.5` thiên về tiếng Anh nên
Vector chỉ đạt 24,0%; BM25 đạt 33,3% và Hybrid 32,0%. Điều này cho thấy phải
chọn model theo ngôn ngữ và đo trên dữ liệu thật. Tôi chỉ dùng pure vector cho
truy vấn tiếng Việt diễn đạt lại sau khi thay bằng model đa ngôn ngữ. Tôi
không dùng Hybrid khi BM25 đã đủ cho lookup chính xác, vector đã đủ cho
semantic search thuần, hoặc ngân sách latency/tài nguyên không cho phép hai
index.

Điều ngạc nhiên nhất: warm-up và cache vector query giúp Hybrid P99 steady-state
giảm còn 14,3 ms; không nên trộn cold-start của model vào phép đo này.
