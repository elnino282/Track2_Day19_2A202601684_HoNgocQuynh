# Reflection — Lab 19

**Tên:** Hồ Ngọc Quỳnh  
**MSSV:** 202601684  
**Cohort:** A20-K4
**Path:** Lite  
**Bonus:** Hybrid AI Memory (`bonus/`)

Trên 50 golden queries, Hybrid đạt Precision@10 cao nhất (78,6%), nhỉnh hơn
BM25 (77,8%) và Vector (73,2%). Với `mixed`, Hybrid thắng 100% vì RRF kết hợp
khớp từ khóa với tín hiệu ngữ nghĩa. Với `exact`, BM25 và Hybrid cùng đạt
96,7%; tên riêng, mã lỗi và thuật ngữ kỹ thuật được lợi từ khớp chính xác.

Ở `paraphrase`, BM25 đạt 33,3%, Hybrid 32,0% và Vector 24,0%, trái với kỳ vọng
vector thường thắng truy vấn diễn đạt lại. Nguyên nhân hợp lý là model Lite
`bge-small-en-v1.5` thiên về tiếng Anh, còn dữ liệu chủ yếu là tiếng Việt.
Vì vậy, embedding phải được kiểm chứng trên golden set đúng ngôn ngữ; trong
production tôi sẽ benchmark `bge-m3` hoặc multilingual E5 trước khi re-index.

Tôi không dùng Hybrid khi BM25 đã đủ cho lookup chính xác, vector đã đủ cho
semantic search thuần, hoặc ngân sách latency và bộ nhớ không cho phép duy trì
hai index. Sau warm-up, Hybrid P99 đạt 14,3 ms; cần tách cold-start khỏi phép
đo steady-state.
