import json

# Đọc file JSON
with open("experiment_logs/aime2025/2026-02-27_18-50-21/judger_outputs_summary.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# data là một mảng (list)
for idx, item in enumerate(data):
    judger_output = item.get("judger_output")
    correct = item.get("correct")

    print(f"[{idx}] judger_output:")
    print(judger_output)

    print("Correct? ", correct == 'true')

    user_input = input("Nhập 'y' để tiếp tục, ký tự khác để dừng: ").strip().lower()
    if user_input != "y":
        break