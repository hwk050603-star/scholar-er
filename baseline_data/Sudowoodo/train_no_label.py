from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def generate_train_no_label(
    train_path=SCRIPT_DIR / "train.txt",
    output_path=SCRIPT_DIR / "train_no_label.txt"
):
    train_path = Path(train_path)
    output_path = Path(output_path)

    entities = []
    seen = set()

    bad_lines = 0
    total_lines = 0

    with open(train_path, "r", encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, start=1):
            total_lines += 1
            line = line.rstrip("\n")

            if not line.strip():
                continue

            parts = line.split("\t")

            if len(parts) != 3:
                bad_lines += 1
                print(
                    f"[Warning] line {line_num}: expected 3 columns, "
                    f"got {len(parts)}. Skipped."
                )
                continue

            left_entity, right_entity, label = parts

            left_entity = left_entity.strip()
            right_entity = right_entity.strip()

            if left_entity and left_entity not in seen:
                seen.add(left_entity)
                entities.append(left_entity)

            if right_entity and right_entity not in seen:
                seen.add(right_entity)
                entities.append(right_entity)

    with open(output_path, "w", encoding="utf-8") as f:
        for entity in entities:
            f.write(entity + "\n")

    print("Generation finished.")
    print(f"train.txt rows read: {total_lines}")
    print(f"invalid rows skipped: {bad_lines}")
    print(f"unique entities: {len(entities)}")
    print(f"output file: {output_path}")


if __name__ == "__main__":
    generate_train_no_label()
