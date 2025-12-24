import os
import json
from datasets import load_dataset


# AI-ModelScope/LaTeX_OCR
data_files = {
    "train": ["/data/cheyujie/datasets/human_handwrite/train-00000-of-00001.parquet"],
    "validation": "/data/cheyujie/datasets/human_handwrite/validation-00000-of-00001.parquet",
    "test": "/data/cheyujie/datasets/human_handwrite/test-00000-of-00001.parquet"
}

dataset = load_dataset("parquet", data_files=data_files)
print("train dataset length: ", len(dataset["train"]))
print("validation dataset length: ", len(dataset["validation"]))
print("test dataset length: ", len(dataset["test"]))

output_dir = "/data/cheyujie/datasets/latex_ocr"
os.makedirs(output_dir, exist_ok=True)

res = []

for split in dataset.keys():
    for i, item in enumerate(dataset[split]):
        img = item['image']
        filename = f"{i}.png"
        save_path = os.path.abspath(os.path.join(output_dir, 'imgs', filename))
        
        text = item['text']
        
        messages = {
            "messages": [
                {"role": "user", "content": "Using LaTeX to perform OCR on the image."}, 
                {"role": "assistant", "content": text}
            ], 
            "images": [f"/datasets/latex_ocr/imgs/{filename}"]
        }
        res.append(messages)
        
        # 保存图片
        # img.save(save_path)
        
        if i % 100 == 0:
            print(f"Saved {i} images...")

# 保存文本数据
with open(os.path.join(output_dir, "human_handwrite.jsonl"), "w") as f:
    for data in res:
        json.dump(data, f)
        f.write("\n")
