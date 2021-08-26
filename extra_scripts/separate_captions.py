import os
import time


mdetr_text_dir_path = "/data/output/mdetr_train_doc"
output_dir = "/data/output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
search_word = ["revealing"]
output_file_path = f"{output_dir}/revealing.txt"


def main():
    unique_occurrences = 0
    files = os.listdir(mdetr_text_dir_path)
    for file in files:
        print(f"On file {file}")
        mdetr_text_file_path = f"{mdetr_text_dir_path}/{file}"
        print(f"Reading text from file.")
        with open(mdetr_text_file_path, "r") as f:
            mdetr_train_text = f.read()
        captions = mdetr_train_text.split('\n')
        print(f"There are total {len(captions)} captions in the file.")
        filtered_captions = []
        start = time.time()
        for i, cap in enumerate(captions):
            if i > 0 and i % 10000 == 0:
                print(f"On caption: {i}. Time: {time.time() - start}")
                with open(f"{output_file_path}", "a") as f:
                    for c in filtered_captions:
                        f.write(f"{c}\n")
                filtered_captions = []
            for w in search_word:
                ca = cap.split('.')
                for c in ca:
                    c = c.split(' ')
                    for e in c:
                        if w == e:
                            filtered_captions.append(cap)
                            unique_occurrences += 1
                            break

        with open(f"{output_file_path}", "a") as f:
            for c in filtered_captions:
                f.write(f"{c}\n")
    print(f"Unique occurrences of {[a for a in search_word]}: {unique_occurrences}.")


if __name__ == "__main__":
    main()
