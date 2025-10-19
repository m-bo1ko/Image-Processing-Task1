import sys
import cv2

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    return img

def save_image(path, img):
    cv2.imwrite(path, img)

def brightness(img, value):
    print("[TODO] brightness() called")
    return img

def contrast(img, value):
    print("[TODO] contrast() called")
    return img

def negative(img):
    print("[TODO] negative() called")
    return img

def hflip(img):
    print("[TODO] hflip() called")
    return img

def vflip(img):
    print("[TODO] vflip() called")
    return img

def dflip(img):
    print("[TODO] dflip() called")
    return img

def shrink(img, factor):
    print("[TODO] shrink() called")
    return img

def enlarge(img, factor):
    print("[TODO] enlarge() called")
    return img

def median_filter(img, kernel_size):
    print("[TODO] median_filter() called")
    return img

def geometric_mean_filter(img, kernel_size):
    print("[TODO] geometric_mean_filter() called")
    return img

def mse(img1, img2):
    print("[TODO] mse() called")
    return 0

def pmse(img1, img2):
    print("[TODO] pmse() called")
    return 0

def snr(img1, img2):
    print("[TODO] snr() called")
    return 0

def psnr(img1, img2):
    print("[TODO] psnr() called")
    return 0

def md(img1, img2):
    print("[TODO] md() called")
    return 0

def main():
    if len(sys.argv) == 1 or "--help" in sys.argv:
        print("""
Command-line image processing application

Usage:
    python imgproc.py --command [-argument=value ...]

Commands:
  Elementary operations:
    --brightness         Modify brightness (-value)
    --contrast           Modify contrast (-value)
    --negative           Negative image

  Geometric operations:
    --hflip              Horizontal flip
    --vflip              Vertical flip
    --dflip              Diagonal flip
    --shrink             Shrink image (-factor)
    --enlarge            Enlarge image (-factor)

  Noise removal (variant N1):
    --median             Median filter (-kernel)
    --gmean              Geometric mean filter (-kernel)

  Evaluation metrics:
    --mse, --pmse, --snr, --psnr, --md

Example:
    python imgproc.py --brightness -value=30 -input=input.png -output=output.png
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
    elif cmd == "--negative":
        result = negative(img)
    elif cmd == "--hflip":
        result = hflip(img)
    elif cmd == "--vflip":
        result = vflip(img)
    elif cmd == "--dflip":
        result = dflip(img)
    elif cmd == "--shrink":
        factor = float(args_dict.get("factor", 1))
        result = shrink(img, factor)
    elif cmd == "--enlarge":
        factor = float(args_dict.get("factor", 1))
        result = enlarge(img, factor)
    elif cmd == "--median":
        kernel = int(args_dict.get("kernel", 3))
        result = median_filter(img, kernel)
    elif cmd == "--gmean":
        kernel = int(args_dict.get("kernel", 3))
        result = geometric_mean_filter(img, kernel)
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
