import os

VOC_ROOT = '~/jetson-inference/python/training/detection/ssd/data/fire_smoke_voc'
IMG_DIR = os.path.join(VOC_ROOT, 'JPEGImages')
ANN_DIR = os.path.join(VOC_ROOT, 'Annotations')
SET_DIR = os.path.join(VOC_ROOT, 'ImageSets/Main')

REMOVE_LIST_FILE = 'remove_list.txt'  #one ID per line, no extension


def delete_file_safe(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted {path}")
    else:
        print(f"Could not find {path}")


def remove_from_dataset(img_ids):
    print(f"\nDeleting {len(img_ids)} img/data pairs")

    for img_id in img_ids:
        img_path = os.path.join(IMG_DIR, f"{img_id}.jpg")
        ann_path = os.path.join(ANN_DIR, f"{img_id}.xml")

        delete_file_safe(img_path)
        delete_file_safe(ann_path)

    print("\nUpdating spliting files")
    for split_file in os.listdir(SET_DIR):
        if not split_file.endswith('.txt'):
            continue
        split_path = os.path.join(SET_DIR, split_file)
        with open(split_path, 'r') as f:
            lines = [line.strip() for line in f]

        new_lines = [line for line in lines if line not in img_ids]

        with open(split_path, 'w') as f:
            f.write('\n'.join(new_lines) + '\n')

        print(f"Updated {split_file} ({len(lines)} → {len(new_lines)})")

    print("\nfinished")


if __name__ == "__main__":
    if not os.path.isfile(REMOVE_LIST_FILE):
        print(f"File {REMOVE_LIST_FILE} is missing")
        exit(1)

    with open(REMOVE_LIST_FILE) as f:
        to_remove = [line.strip() for line in f if line.strip()]

    if not to_remove:
        print("remove_list.txt is empty.")
    else:
        remove_from_dataset(to_remove)
