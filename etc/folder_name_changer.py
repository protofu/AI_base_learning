import os

def rename_dataset(image_dir, label_dir, prefix="fire"):
    # 이미지 목록 가져오기 (확장자 jpg)
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")])

    # 라벨 목록 가져오기 (확장자 txt)
    label_files = sorted([f for f in os.listdir(label_dir) if f.lower().endswith(".txt")])

    # 개수 체크
    if len(image_files) != len(label_files):
        raise ValueError("이미지 개수와 라벨 개수가 일치하지 않습니다.")

    # 파일명만 비교(확장자 제외)
    images_no_ext = sorted([os.path.splitext(f)[0] for f in image_files])
    labels_no_ext = sorted([os.path.splitext(f)[0] for f in label_files])

    if images_no_ext != labels_no_ext:
        raise ValueError("이미지와 라벨 파일명이 1:1로 매칭되지 않습니다.")

    # 새로운 이름으로 리네이밍
    for idx, old_name in enumerate(images_no_ext, start=1):
        new_base = f"{prefix}_{idx:04d}"

        old_img_path = os.path.join(image_dir, old_name + ".jpg")
        old_lbl_path = os.path.join(label_dir, old_name + ".txt")

        new_img_path = os.path.join(image_dir, new_base + ".jpg")
        new_lbl_path = os.path.join(label_dir, new_base + ".txt")

        os.rename(old_img_path, new_img_path)
        os.rename(old_lbl_path, new_lbl_path)

    print("리네이밍 완료!")


# 예시 실행
rename_dataset(rf"C:\dataset\dataset_fire\C_confusing_light\sunset", rf"C:\dataset\dataset_fire\C_confusing_light\sunset_labels", prefix="confusing_light_sunset_train")
