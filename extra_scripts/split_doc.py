import os

mdetr_text_file_path = "/data/output/mdetr_train_doc_dict.txt"
output_dir = "/data/output/mdetr_train_doc_dict"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def main():
    with open(mdetr_text_file_path, "r") as f:
        text = f.read()
    text = text.split("\n")

    captions = []
    for i, caption in enumerate(text):
        if i > 0 and i % 10000 == 0:
            print(f"On caption {i}.")
            with open(f"{output_dir}/mdetr_data_train_{i}.txt", "a") as f:
                for c in captions:
                    f.write(f"{c}\n")
            captions = []
        captions.append(caption)

    with open(f"{output_dir}/mdetr_data_train_{i}.txt", "a") as f:
        for c in captions:
            f.write(f"{c}\n")


if __name__ == "__main__":
    main()
