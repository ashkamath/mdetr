import os
import time
import spacy


mdetr_text_dir_path = "/data/output/mdetr_train_doc"
output_dir = "/data/output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

nlp = spacy.load("en_core_web_md")  # make sure to use larger package!


def main():
    files = os.listdir(mdetr_text_dir_path)
    for file in files:
        print(f"On file {file}")
        mdetr_text_file_path = f"{mdetr_text_dir_path}/{file}"
        print(f"Reading text from file.")
        with open(mdetr_text_file_path, "r") as f:
            mdetr_train_text = f.read()
        nlp.max_length = len(mdetr_train_text)
        print(f"Tokenizing text.")
        mdetr_train_doc = nlp(mdetr_train_text)
        print(f"There are total {len(mdetr_train_doc)} words in the document.")
        query_text = nlp("object")

        similarites = {}
        start = time.time()
        for i, token in enumerate(mdetr_train_doc):
            if i > 0 and i % 100000 == 0:
                print(f"On word: {i}. Time: {time.time() - start}")
                with open(f"{output_dir}/similar_to_object.txt", "a") as f:
                    for key in similarites.keys():
                        f.write(f"{key},{similarites[key]}\n")
            similarity = query_text.similarity(token)
            if similarity > 0.5:
                similarites[token.text] = similarity
        with open(f"{output_dir}/similar_to_object.txt", "a") as f:
            for key in similarites.keys():
                f.write(f"{key},{similarites[key]}\n")


if __name__ == "__main__":
    main()
    print("")
