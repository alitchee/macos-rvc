import torch
import os
import argparse

a = "/Users/ty/Documents/Retrieval-based-Voice-Conversion-WebUI-main/assets/weights/"
input_path = a + "wendi.pth"
output_path = a + "wendimodify.pth"
# model_data = torch.load(output_path, map_location="cpu")

# print(model_data.keys())
# # 你可能看到 'config' 或者 'model_info' 这类键
# print(model_data.get("config", {}))

# print(model_data.get("sr", {}))


def add_tgt_sr(data):
    if isinstance(data, dict):
        if "sr" in data and "tgt_sr" not in data:
            data["tgt_sr"] = data["sr"]
        for v in data.values():
            add_tgt_sr(v)


def main():
    if not os.path.isfile(input_path):
        print(f"❌ 输入模型不存在: {input_path}")
        return

    print(f"📥 正在加载模型: {input_path}")
    data = torch.load(input_path, map_location="cpu")
    add_tgt_sr(data)
    torch.save(data, output_path)
    print(f"✅ 已保存新模型: {output_path}")


if __name__ == "__main__":
    main()


# 修改 sr 为 tgt_sr
# def fix_sr_to_tgt_sr(model_path, output_path):
#     data = torch.load(model_path, map_location="cpu")
#     if "tgt_sr" not in data and "sr" in data:
#         data["tgt_sr"] = data["sr"]
#         print(f"已将 'sr' 字段复制到 'tgt_sr'，值为 {data['tgt_sr']}")
#     else:
#         print("模型已有 'tgt_sr' 字段或没有 'sr' 字段，未做修改。")

#     torch.save(data, output_path)
#     print(f"保存修改后的模型到 {output_path}")

# if __name__ == "__main__":
#     fix_sr_to_tgt_sr(model_path, output_path)
