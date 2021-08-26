import os

ann_dir_path = "/home/maaz/PycharmProjects/VOC_EVAL/dets_from_diff_methods/gpv_1"
# text_queries = ['all_objects', 'all_entities', 'all_visible_entities_and_objects',
# 'all_obscure_entities_and_objects', 'all_entities_and_objects', 'all_object_entities', 'all_visible_entities',
# 'all_visible_objects', 'all_visible_entities_and_objects', 'all_visible_objects_and_entities',
# 'visible_entities_and_objects', 'all_revealing_entities_and_objects', 'combine_7']
# text_queries = ['all_objects', 'all_entities', 'all_visible_entities_and_objects', 'all_obscure_entities_and_objects',
#                 'combine_6']
text_queries = ['find_all_objects']


def parse_det_txt(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.readlines()
        boxes = []
        scores = []
        for line in lines:
            content = line.rstrip().split(' ')
            bbox = content[2:]
            boxes.append([int(b) for b in bbox])
            scores.append(content[1])
        return boxes, scores
    else:
        return [], []


def main():
    stats = {}
    det_files = os.listdir(f"{ann_dir_path}/{text_queries[0]}")
    for q in text_queries:
        stats[q] = {}
        for i, file in enumerate(det_files):
            if i % 1000 == 0:
                print(f"On file no. {i}")
            file_path = f"{ann_dir_path}/{q}/{file}"
            boxes, scores = parse_det_txt(file_path)
            stats[q][file] = len(boxes)
    for tq in stats.keys():
        tq_stats = stats[tq]
        avg_boxes = sum(tq_stats.values()) / len(tq_stats)
        print(f"{tq}: {avg_boxes}")


if __name__ == "__main__":
    main()
