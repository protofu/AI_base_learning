import cv2, numpy as np, os
from pathlib import Path

src = Path("c:/prac/ai_prac/datasets/images/train")
dst = Path("c:/prac/ai_prac/datasets/images/train_aug")
dst.mkdir(parents=True, exist_ok=True)

for f in src.glob("*.jpg"):
    img = cv2.imread(str(f))

    # 밝기 조정
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[...,2] = np.clip(hsv[...,2] * np.random.uniform(0.6, 1.4), 0, 255)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 가우시안 블러
    if np.random.rand() > 0.5:
        img = cv2.GaussianBlur(img, (5,5), 0)

    # 회전
    angle = np.random.uniform(-10, 10)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))

    cv2.imwrite(str(dst / f.name), img)
