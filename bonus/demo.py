"""Run the five required HybridMemoryAgent recall scenarios."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bonus.agent import HybridMemoryAgent  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    agent = HybridMemoryAgent(top_k=3)
    memories = [
        "Tôi đã đọc tài liệu Kubernetes về Deployment, Service và cách dùng HPA để tự động mở rộng hạ tầng theo CPU.",
        "Ghi chú cloud security: áp dụng least privilege IAM, mã hóa dữ liệu, xoay vòng secret và bật audit log.",
        "Bài viết về autoscaling infrastructure giải thích Kubernetes HPA, cluster autoscaler và capacity planning.",
        "Tôi thích tài liệu cloud bằng tiếng Việt, có ví dụ ngắn và checklist thực hành.",
        "Zero Trust yêu cầu xác minh liên tục, phân đoạn mạng và không mặc định tin cậy workload nội bộ.",
        "Kubernetes NetworkPolicy giới hạn luồng mạng giữa pod và giảm blast radius khi có sự cố bảo mật.",
    ]
    for memory in memories:
        agent.remember(memory)
    agent.remember("Hồ sơ y tế riêng của người dùng khác.", user_id="u_999")

    queries = [
        "Tôi đã đọc gì về Kubernetes?",
        "Recommend đọc gì tiếp",
        "Tôi đang quan tâm gì gần đây?",
        "Tài liệu về tự động mở rộng hạ tầng?",
        "Cho tôi summary cloud security",
    ]
    for number, query in enumerate(queries, start=1):
        print(f"\n{'=' * 72}\nQUERY {number}: {query}\n{'-' * 72}")
        print(agent.recall(query))


if __name__ == "__main__":
    main()
