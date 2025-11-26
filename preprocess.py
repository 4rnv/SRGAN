from PIL import Image
import os
import random
from pathlib import Path
from typing import Tuple, Dict

HR_PATCH = 384
LR_PATCH = 96
UPSCALE = HR_PATCH // LR_PATCH
PATCHES_PER_IMAGE = 6
MODE = "generate"
RANDOM_CROP = True
SEED = 201
MIN_HR_DIM = HR_PATCH

random.seed(SEED)

def get_all_image_files(folder: str) -> Dict[str, str]:
    p = Path(folder)
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tiff","*.JPG","*.JPEG","*.PNG","*.BMP","*.TIFF")
    files = {}
    for pattern in exts:
        for fp in p.glob(pattern):
            files[fp.stem] = str(fp)
    return files

def ensure_min_size(img: Image.Image, min_size: int) -> Image.Image:
    w, h = img.size
    if min(w, h) >= min_size:
        return img
    scale = min_size / min(w, h)
    new_size = (int(round(w*scale)), int(round(h*scale)))
    return img.resize(new_size, resample=Image.Resampling.LANCZOS)

def center_crop(img: Image.Image, crop_w: int, crop_h: int) -> Tuple[Image.Image, Tuple[int,int]]:
    w, h = img.size
    left = max(0, (w - crop_w)//2)
    top = max(0, (h - crop_h)//2)
    return img.crop((left, top, left+crop_w, top+crop_h)), (left, top)

def random_crop(img: Image.Image, crop_w: int, crop_h: int) -> Tuple[Image.Image, Tuple[int,int]]:
    w, h = img.size
    if w == crop_w and h == crop_h:
        return img.copy(), (0,0)
    max_left = max(0, w - crop_w)
    max_top = max(0, h - crop_h)
    left = random.randint(0, max_left)
    top = random.randint(0, max_top)
    return img.crop((left, top, left+crop_w, top+crop_h)), (left, top)

def downsample(img: Image.Image, out_size: Tuple[int,int]) -> Image.Image:
    return img.resize(out_size, resample=Image.Resampling.BICUBIC)

def process_generate_mode(hr_path: str, hr_out_dir: str, lr_out_dir: str,
                          patches_per_image: int = PATCHES_PER_IMAGE,
                          random_crop_enabled: bool = RANDOM_CROP):
    img = Image.open(hr_path).convert("RGB")
    img = ensure_min_size(img, MIN_HR_DIM)
    w, h = img.size
    for i in range(patches_per_image):
        if random_crop_enabled:
            hr_patch, (left, top) = random_crop(img, HR_PATCH, HR_PATCH) if (w > HR_PATCH and h > HR_PATCH) else center_crop(img, HR_PATCH, HR_PATCH)
        else:
            hr_patch, (left, top) = center_crop(img, HR_PATCH, HR_PATCH)
        lr_patch = downsample(hr_patch, (LR_PATCH, LR_PATCH))
        base = Path(hr_path).stem
        out_hr = os.path.join(hr_out_dir, f"{base}_p{i:03d}.png")
        out_lr = os.path.join(lr_out_dir, f"{base}_p{i:03d}.png")
        hr_patch.save(out_hr, "PNG")
        lr_patch.save(out_lr, "PNG")

def process_paired_mode(hr_path: str, lr_path: str, hr_out_dir: str, lr_out_dir: str,
                        random_crop_enabled: bool = RANDOM_CROP):
    hr_img = Image.open(hr_path).convert("RGB")
    lr_img = Image.open(lr_path).convert("RGB")

    lr_img = ensure_min_size(lr_img, LR_PATCH)
    hr_img = ensure_min_size(hr_img, HR_PATCH)

    hr_w, hr_h = hr_img.size
    lr_w, lr_h = lr_img.size

    scale_w = hr_w / lr_w
    scale_h = hr_h / lr_h
    if abs(scale_w - scale_h) > 0.1:
        print(f"[WARN] HR/LR aspect/scale mismatch for {Path(hr_path).stem} -> falling back to generate mode.")
        process_generate_mode(hr_path, hr_out_dir, lr_out_dir, patches_per_image=1, random_crop_enabled=random_crop_enabled)
        return

    scale = (scale_w + scale_h) / 2.0
    if abs(scale - UPSCALE) > 0.5:
        print(f"[WARN] Expected scale ~{UPSCALE}, got {scale:.2f} for {Path(hr_path).stem}. Falling back to generate mode.")
        process_generate_mode(hr_path, hr_out_dir, lr_out_dir, patches_per_image=1, random_crop_enabled=random_crop_enabled)
        return

    lr_img = lr_img
    lr_w, lr_h = lr_img.size
    patches = 3 if random_crop_enabled else 1
    for i in range(patches):
        if random_crop_enabled and lr_w > LR_PATCH and lr_h > LR_PATCH:
            lr_patch, (l_left, l_top) = random_crop(lr_img, LR_PATCH, LR_PATCH)
        else:
            lr_patch, (l_left, l_top) = center_crop(lr_img, LR_PATCH, LR_PATCH)

        hr_left = int(round(l_left * scale))
        hr_top = int(round(l_top * scale))

        hr_left = min(max(0, hr_left), hr_w - HR_PATCH)
        hr_top = min(max(0, hr_top), hr_h - HR_PATCH)

        hr_patch = hr_img.crop((hr_left, hr_top, hr_left + HR_PATCH, hr_top + HR_PATCH))
        lr_patch_resized = lr_patch.resize((LR_PATCH, LR_PATCH), resample=Image.Resampling.BICUBIC)
        base = Path(hr_path).stem
        out_hr = os.path.join(hr_out_dir, f"{base}_paired_p{i:03d}.png")
        out_lr = os.path.join(lr_out_dir, f"{base}_paired_p{i:03d}.png")
        hr_patch.save(out_hr, "PNG")
        lr_patch_resized.save(out_lr, "PNG")

def batch_process(hr_folder: str, output_folder: str, mode: str = MODE):
    hr_out = os.path.join(output_folder, "hr")
    lr_out = os.path.join(output_folder, "lr")
    os.makedirs(hr_out, exist_ok=True)
    os.makedirs(lr_out, exist_ok=True)

    hr_files = get_all_image_files(hr_folder)

    print(f"Found {len(hr_files)} HR images")
    processed = 0

    if mode == "generate":
        print("Mode: generate LR patches by downsampling HR.")
        for hr_name, hr_path in hr_files.items():
            try:
                process_generate_mode(hr_path, hr_out, lr_out)
                processed += 1
                if processed % 20 == 0:
                    print(f"  Processed {processed}/{len(hr_files)} HR images...")
            except Exception as e:
                print(f"[ERROR] Processing {hr_path}: {e}")
    else:
        raise ValueError("Unknown mode. Choose 'generate'.")

    print(f"\nFinished. Processed {processed} HR images.")
    print(f"Output HR dir: {hr_out}")
    print(f"Output LR dir: {lr_out}")

if __name__ == "__main__":
    hr_folder = "data/hr"
    output_folder = "processed"

    batch_process(hr_folder, output_folder, mode=MODE)
