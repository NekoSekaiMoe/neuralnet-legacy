import argparse
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Convert CSV pixel data to PNG image.")
    parser.add_argument("--input", default="./build/digit.csv", help="Input CSV file")
    parser.add_argument("--output", default="./build/output.png", help="Output PNG file")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = f.read().strip().split(",")

    pixels = np.array(data, dtype=np.float32)

    if pixels.size != 784:
        raise ValueError(f"像素数量错误，期望 784，实际 {pixels.size}")

    img_array = pixels.reshape(28, 28)
    img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)

    img = Image.fromarray(img_array, mode="L")
    img.save(args.output)

    print(f"图片已保存为 {args.output}")

if __name__ == "__main__":
    main()
