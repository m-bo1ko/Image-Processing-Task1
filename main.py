import sys
import cv2
import numpy as np

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    return img

def save_image(path, img):
    cv2.imwrite(path, img)

def brightness(img, value):
    h, w, c = img.shape
    result = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                new_val = int(img[y, x, ch]) + int(value)
                if new_val < 0:
                    new_val = 0
                elif new_val > 255:
                    new_val = 255
                result[y, x, ch] = new_val
    return result

def contrast(img, value):
    h, w, c = img.shape
    result = np.zeros_like(img)
    mean = np.mean(img)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                new_val = (img[y, x, ch] - mean) * value + mean
                if new_val < 0:
                    new_val = 0
                elif new_val > 255:
                    new_val = 255
                result[y, x, ch] = int(new_val)
    return result

def hflip(img):
    h, w, c = img.shape
    result = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            result[y, x] = img[y, w - 1 - x]
    return result

def dflip(img):
    h, w, c = img.shape
    result = np.zeros((w, h, c), dtype=img.dtype)
    for y in range(h):
        for x in range(w):
            result[x, y] = img[y, x]
    return result

def shrink(img, factor):
    h, w, c = img.shape
    new_h = int(h / factor)
    new_w = int(w / factor)
    result = np.zeros((new_h, new_w, c), dtype=img.dtype)
    for y in range(new_h):
        for x in range(new_w):
            src_y = int(y * factor)
            src_x = int(x * factor)
            result[y, x] = img[src_y, src_x]
    return result

def median_filter(img, kernel_size):
    pad = kernel_size // 2
    h, w, c = img.shape
    result = np.zeros_like(img)
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                neighbors = []
                for ky in range(-pad, pad + 1):
                    for kx in range(-pad, pad + 1):
                        neighbors.append(padded[y + pad + ky, x + pad + kx, ch])
                neighbors.sort()
                median = neighbors[len(neighbors) // 2]
                result[y, x, ch] = median
    return result

def mse(img1, img2):
    h, w, c = img1.shape
    total = 0
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                diff = int(img1[y, x, ch]) - int(img2[y, x, ch])
                total += diff * diff
    return total / (h * w * c)

def snr(img1, img2):
    h, w, c = img1.shape
    signal_sum = 0
    noise_sum = 0
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                signal = int(img1[y, x, ch])
                noise = int(img1[y, x, ch]) - int(img2[y, x, ch])
                signal_sum += signal * signal
                noise_sum += noise * noise
    if noise_sum == 0:
        return float('inf')
    return 10 * np.log10(signal_sum / noise_sum)

def md(img1, img2):
    h, w, c = img1.shape
    max_diff = 0
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                diff = abs(int(img1[y, x, ch]) - int(img2[y, x, ch]))
                if diff > max_diff:
                    max_diff = diff
    return max_diff

def main():
    if len(sys.argv) == 1 or "--help" in sys.argv:
        print("""
Command-line image processing application

Usage:
    python imgproc.py --command [-argument=value ...]

Example:
    python imgproc.py --brightness -value=40 -input=input.png -output=bright.png
""")
        sys.exit(0)

    cmd = None
    args_dict = {}
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            cmd = arg
        elif "=" in arg and arg.startswith("-"):
            key, val = arg[1:].split("=", 1)
            args_dict[key] = val

    if "input" not in args_dict:
        print("Error: input image is required (-input=...)")
        sys.exit(1)

    img = load_image(args_dict["input"])

    if cmd == "--brightness":
        value = float(args_dict.get("value", 0))
        result = brightness(img, value)
    elif cmd == "--contrast":
        value = float(args_dict.get("value", 1))
        result = contrast(img, value)
    elif cmd == "--hflip":
        result = hflip(img)
    elif cmd == "--dflip":
        result = dflip(img)
    elif cmd == "--shrink":
        factor = float(args_dict.get("factor", 2))
        result = shrink(img, factor)
    elif cmd == "--median":
        kernel = int(args_dict.get("kernel", 3))
        result = median_filter(img, kernel)
    elif cmd == "--mse":
        img2 = load_image(args_dict["ref"])
        print("MSE:", mse(img, img2))
        return
    elif cmd == "--snr":
        img2 = load_image(args_dict["ref"])
        print("SNR:", snr(img, img2))
        return
    elif cmd == "--md":
        img2 = load_image(args_dict["ref"])
        print("MD:", md(img, img2))
        return
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

    if "output" in args_dict:
        save_image(args_dict["output"], result)
        print(f"Saved result to {args_dict['output']}")
    else:
        print("No output file specified (-output=...)")

if __name__ == "__main__":
    main()
