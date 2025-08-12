import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np

OUT = "dataset"
IM_DIR = os.path.join(OUT, "images")
ANN_DIR = os.path.join(OUT, "annotations")
os.makedirs(IM_DIR, exist_ok=True)
os.makedirs(ANN_DIR, exist_ok=True)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = None

W = 640
H = 480

def random_bg(w, h):
    arr = np.uint8(np.random.normal(200, 30, (h, w, 3)))
    img = Image.fromarray(arr, mode='RGB')
    
    return img

def draw_letter_image(letter, size, color, angle, font_path):
    
    if font_path:
        font = ImageFont.truetype(font_path, size)
    else:
        font = ImageFont.load_default()
        
    dummy = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dummy)
    bbox = dd.textbbox((0, 0), letter, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    im = Image.new("RGBA", (w + 4, h + 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.text((2, 2), letter, font=font, fill=color)
    im = im.rotate(angle, expand=True)
    
    return im

def place_letter(base, letter_im, center):
    bx, by = center
    w, h = letter_im.size
    x0 = int(bx - w / 2)
    y0 = int(by - h / 2)
    base.paste(letter_im, (x0, y0), letter_im)
    
    return (x0, y0, x0 + w, y0 + h)

def clamp_bbox(bbox, W, H):
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(W - 1, x0))
    x1 = max(0, min(W - 1, x1))
    y0 = max(0, min(H - 1, y0))
    y1 = max(0, min(H - 1, y1))
    
    return x0, y0, x1, y1

def gen_one(idx, max_letters=8):
    bg = random_bg(W, H)
    n = random.randint(1, max_letters)
    ann_lines = []
    tries = 0
    placed = 0
    attempts_limit = 200
    existing_boxes = []

    while placed < n and tries < attempts_limit:
        tries += 1
        ch = chr(ord('A') + random.randint(0, 25))
        size = random.randint(30, 120)
        angle = random.uniform(0, 360)
        color = tuple(random.randint(0, 255) for _ in range(3))
        letter_im = draw_letter_image(ch, size, color, angle, FONT_PATH)
        lw, lh = letter_im.size

        cx = random.randint(lw // 2, W - lw // 2 - 1)
        cy = random.randint(lh // 2, H - lh // 2 - 1)
        bbox = (int(cx - lw / 2), int(cy - lh / 2), int(cx + lw / 2), int(cy + lh / 2))

        flag = True
        for eb in existing_boxes:
            xa = max(bbox[0], eb[0])
            ya = max(bbox[1], eb[1])
            xb = min(bbox[2], eb[2])
            yb = min(bbox[3], eb[3])
            inter = max(0, xb - xa) * max(0, yb - ya)
            
            if inter > 0.5 * (lw * lh):
                flag = False
                break
        if not flag:
            continue

        bbox = place_letter(bg, letter_im, (cx, cy))
        bbox = clamp_bbox(bbox, W, H)
        existing_boxes.append(bbox)
        cls = ord(ch) - ord('A')
        ann_lines.append(f"{cls} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")
        placed += 1

    if random.random() < 0.3:
        arr = np.array(bg).astype(np.int16)
        arr = arr + np.random.normal(0, 8, arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        bg = Image.fromarray(arr)

    name = f"{idx:06d}.png"
    bg.save(os.path.join(IM_DIR, name))
    with open(os.path.join(ANN_DIR, f"{idx:06d}.txt"), "w") as f:
        f.write("\n".join(ann_lines))

NUM = 2000
for i in range(NUM):
    gen_one(i, max_letters=8)
    if (i + 1) % 100 == 0:
        print("generated", i + 1)