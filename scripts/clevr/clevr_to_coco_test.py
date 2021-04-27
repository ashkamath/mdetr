# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
import json
from pathlib import Path

from clevr_to_coco import Item, load_templates, parse_args, parse_prog


def test_suite(templates, val_questions, val_scenes):
    def _do_test(qid, target_caption, target_objs):
        print("testing", qid)
        scid = val_questions["questions"][qid]["image_index"]
        objs, caption = parse_prog(val_scenes["scenes"][scid], val_questions["questions"][qid], templates)
        assert caption == target_caption, "wrong caption"

        assert len(objs) == len(target_objs), "wrong number of objs"
        for o1, o2 in zip(objs, target_objs):
            assert o1[0].is_same(o2[0]), f"object {o1[0]} not matching {o2[0]}"
            assert sorted(o1[1]) == sorted(o2[1]), "tokens not matching"

    _do_test(
        1940,
        "Are there the same number of tiny blocks and matte objects? no",
        [
            (Item("purple", "small", "sphere", "rubber"), [(45, 50), (51, 58)]),
            (Item("yellow", "large", "cube", "rubber"), [(45, 50), (51, 58)]),
        ],
    )

    _do_test(
        3074,
        "Are there an equal number of red rubber objects and tiny gray metallic things? no",
        [
            (Item("red", "small", "cylinder", "rubber"), [(29, 32), (33, 39), (40, 47)]),
            (Item("gray", "small", "cylinder", "metal"), [(52, 56), (57, 61), (62, 70), (71, 77)]),
            (Item("gray", "small", "cylinder", "metal"), [(52, 56), (57, 61), (62, 70), (71, 77)]),
        ],
    )

    _do_test(
        1176,
        "Is the number of matte spheres less than the number of cubes? yes",
        [
            (Item("yellow", "large", "sphere", "rubber"), [(17, 22), (23, 30)]),
            (Item("gray", "small", "cube", "rubber"), [(55, 60)]),
            (Item("gray", "small", "cube", "rubber"), [(55, 60)]),
            (Item("brown", "small", "cube", "metal"), [(55, 60)]),
        ],
    )

    _do_test(
        101,
        "Are there more green objects than tiny rubber cylinders? yes",
        [
            (Item("green", "large", "cube", "metal"), [(15, 20), (21, 28)]),
            (Item("green", "large", "cube", "metal"), [(15, 20), (21, 28)]),
            (Item("green", "large", "cube", "metal"), [(15, 20), (21, 28)]),
            (Item("purple", "small", "cylinder", "rubber"), [(34, 38), (39, 45), (46, 55)]),
        ],
    )

    _do_test(
        516,
        "Is the number of red matte cylinders that are behind the big cyan rubber thing the same as the number of tiny cylinders? no",
        [
            (Item("cyan", "large", "cylinder", "rubber"), [(57, 60), (61, 65), (66, 72), (73, 78)]),
            (Item("red", "small", "cylinder", "rubber"), [(105, 109), (110, 119)]),
            (Item("cyan", "small", "cylinder", "metal"), [(105, 109), (110, 119)]),
            (Item("yellow", "small", "cylinder", "rubber"), [(105, 109), (110, 119)]),
        ],
    )

    _do_test(
        35,
        "Is the number of brown cylinders in front of the brown matte cylinder less than the number of brown rubber cylinders? no",
        [
            (
                Item("brown", "small", "cylinder", "rubber"),
                [(100, 106), (49, 54), (107, 116), (94, 99), (61, 69), (55, 60)],
            ),
            (Item("brown", "small", "cylinder", "metal"), [(17, 22), (23, 32), (33, 44)]),
        ],
    )

    _do_test(
        14,
        "Are there more shiny things that are to the left of the yellow rubber thing than small red spheres? yes",
        [
            (Item("yellow", "small", "cylinder", "rubber"), [(56, 62), (63, 69), (70, 75)]),
            (Item("gray", "large", "cylinder", "metal"), [(15, 20), (21, 27), (37, 51)]),
            (Item("green", "small", "cylinder", "metal"), [(15, 20), (21, 27), (37, 51)]),
        ],
    )

    _do_test(
        114,
        "Is the number of big brown objects behind the tiny red shiny cylinder the same as the number of metallic blocks that are on the right side of the gray ball? no",
        [
            (Item("red", "small", "cylinder", "metal"), [(46, 50), (51, 54), (55, 60), (61, 69)]),
            (Item("gray", "small", "sphere", "rubber"), [(146, 150), (151, 155)]),
            (Item("red", "small", "cube", "metal"), [(96, 104), (105, 111), (121, 141)]),
        ],
    )

    _do_test(
        80,
        "Do the purple cylinder and the yellow rubber thing have the same size? no",
        [
            (Item("purple", "small", "cylinder", "rubber"), [(7, 13), (14, 22)]),
            (Item("yellow", "large", "cylinder", "rubber"), [(31, 37), (38, 44), (45, 50)]),
        ],
    )

    _do_test(
        13,
        "Is the color of the big matte object the same as the large metal cube? yes",
        [
            (Item("blue", "large", "cylinder", "rubber"), [(20, 23), (24, 29), (30, 36)]),
            (Item("blue", "large", "cube", "metal"), [(53, 58), (59, 64), (65, 69)]),
        ],
    )

    _do_test(
        17,
        "Is the material of the yellow block the same as the yellow cylinder? no",
        [
            (Item("yellow", "small", "cube", "metal"), [(23, 29), (30, 35)]),
            (Item("yellow", "small", "cylinder", "rubber"), [(52, 58), (59, 67)]),
        ],
    )

    _do_test(
        3,
        "There is a small gray block; are there any spheres to the left of it? yes",
        [
            (Item("gray", "small", "cube", "metal"), [(11, 16), (17, 21), (22, 27)]),
            (Item("purple", "large", "sphere", "metal"), [(43, 50), (51, 65), (70, 73)]),
        ],
    )

    _do_test(
        4,
        "Is the purple thing the same shape as the large gray rubber thing? no",
        [
            (Item("purple", "large", "sphere", "metal"), [(7, 13), (14, 19)]),
            (Item("gray", "large", "cube", "rubber"), [(42, 47), (48, 52), (53, 59), (60, 65)]),
        ],
    )

    _do_test(
        51,
        "There is a rubber object that is left of the yellow block; is it the same size as the tiny rubber block? no",
        [
            (Item("yellow", "small", "cube", "rubber"), [(45, 51), (91, 97), (98, 103), (86, 90), (52, 57)]),
            (Item("yellow", "large", "sphere", "rubber"), [(11, 17), (18, 24), (33, 40)]),
        ],
    )

    _do_test(
        429,
        "There is a matte sphere; is it the same size as the yellow ball behind the tiny metal cylinder? no",
        [
            (Item("yellow", "large", "sphere", "rubber"), [(11, 16), (17, 23)]),
            (Item("blue", "small", "cylinder", "metal"), [(75, 79), (80, 85), (86, 94)]),
            (Item("yellow", "small", "sphere", "metal"), [(52, 58), (59, 63), (64, 70)]),
        ],
    )

    # skipping comparison 6-14

    _do_test(
        131,
        "Do the small red object that is right of the large purple shiny object and the large object that is in front of the big sphere have the same shape? no",
        [
            (Item("purple", "large", "cube", "metal"), [(45, 50), (51, 57), (58, 63), (64, 70)]),
            (Item("red", "small", "sphere", "rubber"), [(7, 12), (13, 16), (17, 23), (32, 40)]),
            (Item("blue", "large", "sphere", "metal"), [(116, 119), (120, 126)]),
            (Item("gray", "large", "cylinder", "metal"), [(79, 84), (85, 91), (100, 111)]),
        ],
    )

    _do_test(
        55,
        "How many tiny matte objects are to the left of the small yellow thing? 0",
        [(Item("yellow", "small", "cube", "rubber"), [(51, 56), (57, 63), (64, 69)])],
    )

    _do_test(
        109,
        "There is a small purple shiny block; how many large metallic blocks are behind it? 1",
        [
            (Item("purple", "small", "cube", "metal"), [(11, 16), (17, 23), (24, 29), (30, 35)]),
            (Item("green", "large", "cube", "metal"), [(46, 51), (52, 60), (61, 67), (72, 78), (83, 84)]),
        ],
    )

    _do_test(
        280,
        "What size is the metallic thing that is left of the tiny blue thing? large",
        [
            (Item("blue", "small", "cube", "rubber"), [(52, 56), (57, 61), (62, 67)]),
            (Item("purple", "large", "sphere", "metal"), [(17, 25), (26, 31), (40, 47), (69, 74)]),
        ],
    )

    _do_test(
        73,
        "What color is the matte thing in front of the large cube? cyan",
        [
            (Item("brown", "large", "cube", "rubber"), [(46, 51), (52, 56)]),
            (Item("cyan", "small", "cylinder", "rubber"), [(18, 23), (24, 29), (30, 41), (58, 62)]),
        ],
    )

    _do_test(
        8,
        "What is the thing in front of the small metallic object made of? rubber",
        [
            (Item("gray", "small", "cube", "metal"), [(34, 39), (40, 48), (49, 55)]),
            (Item("gray", "large", "cube", "rubber"), [(12, 17), (18, 29), (65, 71)]),
        ],
    )

    _do_test(
        27,
        "There is a big metallic thing left of the tiny green object; what is its shape? sphere",
        [
            (Item("green", "small", "sphere", "metal"), [(42, 46), (47, 52), (53, 59)]),
            (Item("yellow", "large", "sphere", "metal"), [(11, 14), (15, 23), (24, 29), (30, 37), (80, 86)]),
        ],
    )

    _do_test(
        269,
        "Are there any other things that are the same size as the purple matte cylinder? yes",
        [
            (Item("purple", "small", "cylinder", "rubber"), [(57, 63), (64, 69), (70, 78)]),
            (Item("yellow", "small", "cylinder", "rubber"), [(40, 49), (10, 26), (80, 83)]),
            (Item("brown", "small", "cube", "rubber"), [(40, 49), (10, 26), (80, 83)]),
            (Item("yellow", "small", "cylinder", "rubber"), [(40, 49), (10, 26), (80, 83)]),
            (Item("cyan", "small", "sphere", "metal"), [(40, 49), (10, 26), (80, 83)]),
        ],
    )

    _do_test(
        11,
        "Is there anything else that has the same color as the large shiny cube? yes",
        [
            (Item("blue", "large", "cube", "metal"), [(54, 59), (60, 65), (66, 70)]),
            (Item("blue", "small", "cylinder", "metal"), [(36, 46), (9, 22), (72, 75)]),
            (Item("blue", "large", "cylinder", "rubber"), [(36, 46), (9, 22), (72, 75)]),
        ],
    )

    _do_test(
        123,
        "Are there any other things that are the same material as the large brown object? yes",
        [
            (Item("brown", "large", "cylinder", "metal"), [(61, 66), (67, 72), (73, 79)]),
            (Item("purple", "large", "cylinder", "metal"), [(40, 53), (10, 26), (81, 84)]),
        ],
    )

    _do_test(
        365,
        "Is there anything else that is the same shape as the small red object? yes",
        [
            (Item("red", "small", "cylinder", "metal"), [(53, 58), (59, 62), (63, 69)]),
            (Item("yellow", "small", "cylinder", "metal"), [(35, 45), (9, 22), (71, 74)]),
        ],
    )

    _do_test(
        5,
        "What number of other objects are the same size as the purple shiny object? 2",
        [
            (Item("purple", "large", "sphere", "metal"), [(54, 60), (61, 66), (67, 73)]),
            (Item("brown", "large", "cylinder", "rubber"), [(37, 46), (15, 28), (75, 76)]),
            (Item("gray", "large", "cube", "rubber"), [(37, 46), (15, 28), (75, 76)]),
        ],
    )

    _do_test(
        34,
        "How many other objects are there of the same color as the matte cylinder? 1",
        [
            (Item("brown", "small", "cylinder", "rubber"), [(58, 63), (64, 72)]),
            (Item("brown", "small", "cylinder", "metal"), [(40, 50), (9, 22), (74, 75)]),
        ],
    )

    _do_test(
        15,
        "What number of other things are the same material as the big gray cylinder? 6",
        [
            (Item("gray", "large", "cylinder", "metal"), [(57, 60), (61, 65), (66, 74)]),
            (Item("blue", "small", "cylinder", "metal"), [(36, 49), (15, 27), (76, 77)]),
            (Item("purple", "large", "sphere", "metal"), [(36, 49), (15, 27), (76, 77)]),
            (Item("yellow", "small", "cube", "metal"), [(36, 49), (15, 27), (76, 77)]),
            (Item("cyan", "small", "cube", "metal"), [(36, 49), (15, 27), (76, 77)]),
            (Item("blue", "large", "cube", "metal"), [(36, 49), (15, 27), (76, 77)]),
            (Item("green", "small", "cylinder", "metal"), [(36, 49), (15, 27), (76, 77)]),
        ],
    )

    _do_test(
        93,
        "Is there a blue metal object that has the same size as the gray metal object? yes",
        [
            (Item("gray", "small", "sphere", "metal"), [(59, 63), (64, 69), (70, 76)]),
            (Item("blue", "small", "cube", "metal"), [(42, 51), (11, 15), (16, 21), (22, 28), (78, 81)]),
        ],
    )

    _do_test(
        116,
        "Is there a block that has the same material as the large brown cylinder? yes",
        [
            (Item("brown", "large", "cylinder", "rubber"), [(51, 56), (57, 62), (63, 71)]),
            (Item("red", "large", "cube", "rubber"), [(30, 43), (11, 16), (73, 76)]),
        ],
    )

    _do_test(
        1367,
        "Is there another rubber cube of the same color as the tiny block? yes",
        [
            (Item("blue", "small", "cube", "metal"), [(54, 58), (59, 64)]),
            (Item("blue", "large", "cube", "rubber"), [(36, 46), (17, 23), (24, 28), (66, 69)]),
        ],
    )

    _do_test(
        1,
        "Is there a big brown object of the same shape as the green thing? yes",
        [
            (Item("green", "small", "cylinder", "rubber"), [(53, 58), (59, 64)]),
            (Item("brown", "large", "cylinder", "rubber"), [(35, 45), (11, 14), (15, 20), (21, 27), (66, 69)]),
        ],
    )

    _do_test(
        6682,
        "How many other metallic things have the same size as the yellow thing? 4",
        [
            (Item("yellow", "large", "cylinder", "rubber"), [(57, 63), (64, 69)]),
            (Item("gray", "large", "cylinder", "metal"), [(40, 49), (15, 23), (24, 30), (71, 72)]),
            (Item("gray", "large", "cube", "metal"), [(40, 49), (15, 23), (24, 30), (71, 72)]),
            (Item("gray", "large", "cube", "metal"), [(40, 49), (15, 23), (24, 30), (71, 72)]),
            (Item("gray", "large", "cylinder", "metal"), [(40, 49), (15, 23), (24, 30), (71, 72)]),
        ],
    )

    _do_test(
        2406,
        "How many other metallic objects have the same color as the tiny metal object? 1",
        [
            (Item("purple", "small", "sphere", "metal"), [(59, 63), (64, 69), (70, 76)]),
            (Item("purple", "large", "cylinder", "metal"), [(41, 51), (15, 23), (24, 31), (78, 79)]),
        ],
    )

    _do_test(
        1094,
        "How many other small spheres have the same material as the cyan sphere? 2",
        [
            (Item("cyan", "large", "sphere", "rubber"), [(59, 63), (64, 70)]),
            (Item("blue", "small", "sphere", "rubber"), [(38, 51), (15, 20), (21, 28), (72, 73)]),
            (Item("blue", "small", "sphere", "rubber"), [(38, 51), (15, 20), (21, 28), (72, 73)]),
        ],
    )

    _do_test(
        795,
        "How many other small shiny objects are the same shape as the red object? 1",
        [
            (Item("red", "large", "sphere", "metal"), [(61, 64), (65, 71)]),
            (Item("purple", "small", "sphere", "metal"), [(43, 53), (15, 20), (21, 26), (27, 34), (73, 74)]),
        ],
    )

    _do_test(
        771,
        "There is a shiny cylinder that is the same size as the blue metal sphere; what color is it? brown",
        [
            (Item("blue", "large", "sphere", "metal"), [(55, 59), (60, 65), (66, 72)]),
            (Item("brown", "large", "cylinder", "metal"), [(38, 47), (11, 16), (17, 25), (92, 97)]),
        ],
    )

    _do_test(
        104,
        "What is the material of the other object that is the same size as the matte thing? metal",
        [
            (Item("purple", "small", "cylinder", "rubber"), [(70, 75), (76, 81)]),
            (Item("purple", "small", "cube", "metal"), [(53, 62), (34, 40), (83, 88)]),
        ],
    )

    _do_test(
        96,
        "There is a gray rubber thing that is the same size as the gray sphere; what shape is it? cube",
        [
            (Item("gray", "small", "sphere", "metal"), [(58, 62), (63, 69)]),
            (Item("gray", "small", "cube", "rubber"), [(41, 50), (11, 15), (16, 22), (23, 28), (89, 93)]),
        ],
    )

    _do_test(
        253,
        "There is a metallic thing that is the same color as the big matte cylinder; what size is it? small",
        [
            (Item("cyan", "large", "cylinder", "rubber"), [(56, 59), (60, 65), (66, 74)]),
            (Item("cyan", "small", "cylinder", "metal"), [(38, 48), (11, 19), (20, 25), (93, 98)]),
        ],
    )

    _do_test(
        231,
        "There is a tiny ball that is the same color as the small metal cylinder; what is its material? metal",
        [
            (Item("yellow", "small", "cylinder", "metal"), [(51, 56), (57, 62), (63, 71)]),
            (Item("yellow", "small", "sphere", "metal"), [(33, 43), (11, 15), (16, 20), (95, 100)]),
        ],
    )

    _do_test(
        49,
        "The other object that is the same color as the large shiny thing is what shape? cylinder",
        [
            (Item("gray", "large", "sphere", "metal"), [(47, 52), (53, 58), (59, 64)]),
            (Item("gray", "large", "cylinder", "rubber"), [(29, 39), (10, 16), (80, 88)]),
        ],
    )

    _do_test(
        316,
        "There is a blue cube that is the same material as the tiny yellow thing; what size is it? small",
        [
            (Item("yellow", "small", "sphere", "rubber"), [(54, 58), (59, 65), (66, 71)]),
            (Item("blue", "small", "cube", "rubber"), [(33, 46), (11, 15), (16, 20), (90, 95)]),
        ],
    )

    _do_test(
        29,
        "There is another thing that is the same material as the gray object; what is its color? yellow",
        [
            (Item("gray", "large", "cylinder", "rubber"), [(56, 60), (61, 67)]),
            (Item("yellow", "large", "cube", "rubber"), [(35, 48), (17, 22), (88, 94)]),
        ],
    )

    _do_test(
        431,
        "There is a green thing that is made of the same material as the tiny cyan cylinder; what shape is it? cylinder",
        [
            (Item("cyan", "small", "cylinder", "rubber"), [(64, 68), (69, 73), (74, 82)]),
            (Item("green", "large", "cylinder", "rubber"), [(43, 56), (11, 16), (17, 22), (102, 110)]),
        ],
    )

    _do_test(
        211,
        "There is another thing that is the same shape as the brown metallic object; what is its size? small",
        [
            (Item("brown", "large", "cube", "metal"), [(53, 58), (59, 67), (68, 74)]),
            (Item("yellow", "small", "cube", "metal"), [(35, 45), (17, 22), (94, 99)]),
        ],
    )

    _do_test(
        10,
        "The other small shiny thing that is the same shape as the tiny yellow shiny object is what color? cyan",
        [
            (Item("yellow", "small", "cube", "metal"), [(58, 62), (63, 69), (70, 75), (76, 82)]),
            (Item("cyan", "small", "cube", "metal"), [(40, 50), (10, 15), (16, 21), (22, 27), (98, 102)]),
        ],
    )

    _do_test(
        40,
        "There is a big gray object that is the same shape as the purple rubber object; what is it made of? rubber",
        [
            (Item("purple", "small", "cylinder", "rubber"), [(57, 63), (64, 70), (71, 77)]),
            (Item("gray", "large", "cylinder", "rubber"), [(39, 49), (11, 14), (15, 19), (20, 26), (99, 105)]),
        ],
    )

    _do_test(
        89,
        "What number of tiny things are both on the left side of the gray shiny sphere and to the right of the brown rubber cube? 1",
        [
            (Item("brown", "small", "cube", "rubber"), [(102, 107), (108, 114), (115, 119)]),
            (Item("gray", "small", "sphere", "metal"), [(60, 64), (65, 70), (71, 77)]),
            (Item("purple", "small", "cube", "metal"), [(36, 55), (82, 97), (15, 19), (20, 26), (121, 122)]),
        ],
    )

    _do_test(
        99,
        "There is a metallic object that is left of the brown ball and in front of the tiny blue block; what is its size? small",
        [
            (Item("blue", "small", "cube", "metal"), [(78, 82), (83, 87), (88, 93)]),
            (Item("brown", "small", "sphere", "metal"), [(47, 52), (53, 57)]),
            (Item("gray", "small", "sphere", "metal"), [(35, 42), (62, 73), (11, 19), (20, 26), (113, 118)]),
        ],
    )

    _do_test(
        95,
        "The large thing that is both on the left side of the purple shiny object and behind the tiny gray metallic ball is what color? brown",
        [
            (Item("gray", "small", "sphere", "metal"), [(88, 92), (93, 97), (98, 106), (107, 111)]),
            (Item("purple", "small", "cube", "metal"), [(53, 59), (60, 65), (66, 72)]),
            (Item("brown", "large", "cylinder", "rubber"), [(29, 48), (77, 83), (4, 9), (10, 15), (127, 132)]),
        ],
    )

    _do_test(
        60,
        "What is the material of the thing that is left of the blue block and on the right side of the big green matte block? rubber",
        [
            (Item("green", "large", "cube", "rubber"), [(94, 97), (98, 103), (104, 109), (110, 115)]),
            (Item("blue", "large", "cube", "metal"), [(54, 58), (59, 64)]),
            (Item("gray", "small", "sphere", "rubber"), [(42, 49), (69, 89), (28, 33), (117, 123)]),
        ],
    )

    _do_test(
        75,
        "There is a thing that is both to the left of the gray sphere and to the right of the small cylinder; what shape is it? cube",
        [
            (Item("cyan", "small", "cylinder", "rubber"), [(85, 90), (91, 99)]),
            (Item("gray", "small", "sphere", "rubber"), [(49, 53), (54, 60)]),
            (Item("brown", "small", "cube", "rubber"), [(30, 44), (65, 80), (11, 16), (119, 123)]),
        ],
    )

    _do_test(
        851,
        "What number of objects are cubes or purple things? 3",
        [
            (Item("brown", "large", "cube", "rubber"), [(27, 32), (51, 52)]),
            (Item("green", "small", "cube", "rubber"), [(27, 32), (51, 52)]),
            (Item("purple", "small", "sphere", "rubber"), [(36, 42), (43, 49), (51, 52)]),
        ],
    )

    _do_test(
        3274,
        "How many things are big things or purple rubber spheres? 5",
        [
            (Item("green", "large", "cube", "metal"), [(24, 30), (57, 58), (20, 23)]),
            (Item("red", "large", "sphere", "metal"), [(24, 30), (57, 58), (20, 23)]),
            (Item("purple", "large", "sphere", "rubber"), [(24, 30), (57, 58), (20, 23), (48, 55), (41, 47), (34, 40)]),
            (Item("blue", "large", "cylinder", "metal"), [(24, 30), (57, 58), (20, 23)]),
            (Item("brown", "large", "cube", "metal"), [(24, 30), (57, 58), (20, 23)]),
        ],
    )

    _do_test(
        718,
        "How many tiny objects are either rubber things or brown rubber objects? 4",
        [
            (Item("green", "small", "sphere", "rubber"), [(72, 73), (9, 13), (40, 46), (33, 39)]),
            (Item("purple", "small", "sphere", "rubber"), [(72, 73), (9, 13), (40, 46), (33, 39)]),
            (Item("purple", "small", "cylinder", "rubber"), [(72, 73), (9, 13), (40, 46), (33, 39)]),
            (Item("purple", "small", "sphere", "rubber"), [(72, 73), (9, 13), (40, 46), (33, 39)]),
        ],
    )

    _do_test(
        1240,
        "What number of small objects are either spheres or red matte blocks? 3",
        [
            (Item("blue", "small", "sphere", "metal"), [(69, 70), (15, 20), (40, 47)]),
            (Item("red", "small", "cube", "rubber"), [(51, 54), (15, 20), (61, 67), (69, 70), (55, 60)]),
            (Item("blue", "small", "sphere", "rubber"), [(69, 70), (15, 20), (40, 47)]),
        ],
    )

    _do_test(
        767,
        "What number of big objects are either cyan matte objects or things? 6",
        [
            (Item("yellow", "large", "cylinder", "metal"), [(68, 69), (15, 18), (60, 66)]),
            (Item("gray", "large", "cube", "rubber"), [(68, 69), (15, 18), (60, 66)]),
            (Item("red", "large", "cube", "rubber"), [(68, 69), (15, 18), (60, 66)]),
            (Item("cyan", "large", "sphere", "rubber"), [(68, 69), (38, 42), (60, 66), (49, 56), (43, 48), (15, 18)]),
            (Item("cyan", "large", "sphere", "rubber"), [(68, 69), (38, 42), (60, 66), (49, 56), (43, 48), (15, 18)]),
            (Item("blue", "large", "cube", "rubber"), [(68, 69), (15, 18), (60, 66)]),
        ],
    )

    _do_test(
        177,
        "How many brown things are big rubber balls or metallic cylinders? 3",
        [
            (Item("brown", "large", "cylinder", "metal"), [(66, 67), (55, 64), (46, 54), (9, 14)]),
            (Item("brown", "small", "cylinder", "metal"), [(66, 67), (55, 64), (46, 54), (9, 14)]),
            (Item("brown", "large", "sphere", "rubber"), [(37, 42), (30, 36), (66, 67), (26, 29), (9, 14)]),
        ],
    )

    _do_test(
        69,
        "What number of things are objects behind the big green matte cube or things that are in front of the big shiny thing? 3",
        [
            (
                Item("green", "large", "cube", "rubber"),
                [(69, 75), (61, 65), (118, 119), (49, 54), (45, 48), (85, 96), (55, 60)],
            ),
            (
                Item("blue", "large", "cube", "metal"),
                [(105, 110), (101, 104), (26, 33), (111, 116), (118, 119), (34, 40)],
            ),
            (Item("gray", "small", "sphere", "rubber"), [(85, 96), (118, 119), (69, 75)]),
        ],
    )

    _do_test(
        239,
        "What number of things are large objects on the left side of the red matte thing or objects that are left of the small red rubber ball? 4",
        [
            (
                Item("red", "small", "sphere", "rubber"),
                [(129, 133), (74, 79), (68, 73), (64, 67), (122, 128), (118, 121), (112, 117)],
            ),
            (Item("yellow", "small", "sphere", "metal"), [(135, 136), (83, 90), (100, 107)]),
            (
                Item("brown", "large", "sphere", "metal"),
                [(40, 59), (135, 136), (100, 107), (83, 90), (32, 39), (26, 31)],
            ),
            (Item("yellow", "small", "cylinder", "metal"), [(135, 136), (83, 90), (100, 107)]),
            (
                Item("gray", "large", "cylinder", "metal"),
                [(40, 59), (135, 136), (100, 107), (83, 90), (32, 39), (26, 31)],
            ),
        ],
    )

    _do_test(
        12,
        "How many metallic objects are big blue cubes or blue objects? 2",
        [
            (Item("blue", "small", "cylinder", "metal"), [(62, 63), (9, 17), (48, 52), (53, 60)]),
            (
                Item("blue", "large", "cube", "metal"),
                [(30, 33), (53, 60), (48, 52), (34, 38), (62, 63), (9, 17), (39, 44)],
            ),
        ],
    )

    _do_test(
        262,
        "How many matte objects are purple objects or tiny brown things? 2",
        [
            (Item("purple", "small", "cylinder", "rubber"), [(27, 33), (34, 41), (9, 14), (64, 65)]),
            (Item("brown", "small", "cube", "rubber"), [(45, 49), (56, 62), (64, 65), (50, 55), (9, 14)]),
        ],
    )

    _do_test(
        84,
        "What number of cylinders are small purple things or yellow rubber things? 2",
        [
            (Item("yellow", "large", "cylinder", "rubber"), [(59, 65), (74, 75), (15, 24), (52, 58), (66, 72)]),
            (Item("purple", "small", "cylinder", "rubber"), [(74, 75), (15, 24), (42, 48), (35, 41), (29, 34)]),
        ],
    )

    _do_test(
        6,
        "How many objects are either metal things behind the small green rubber cylinder or small green rubber objects? 2",
        [
            (
                Item("green", "small", "cylinder", "rubber"),
                [(83, 88), (89, 94), (71, 79), (111, 112), (102, 109), (64, 70), (52, 57), (58, 63), (95, 101)],
            ),
            (Item("purple", "large", "sphere", "metal"), [(28, 33), (41, 47), (34, 40), (111, 112)]),
        ],
    )

    _do_test(
        71,
        "What number of objects are tiny spheres or brown blocks behind the gray matte object? 2",
        [
            (Item("gray", "small", "sphere", "rubber"), [(27, 31), (78, 84), (67, 71), (32, 39), (86, 87), (72, 77)]),
            (Item("brown", "small", "cube", "rubber"), [(43, 48), (56, 62), (49, 55), (86, 87)]),
        ],
    )

    _do_test(
        65,
        "There is a cube to the left of the rubber thing that is on the right side of the large green matte block; how many big blue metallic objects are right of it? 1",
        [
            (Item("green", "large", "cube", "rubber"), [(93, 98), (87, 92), (11, 15), (81, 86), (16, 30), (99, 104)]),
            (Item("gray", "small", "sphere", "rubber"), [(35, 41), (42, 47), (56, 76)]),
            (
                Item("blue", "large", "cube", "metal"),
                [(115, 118), (119, 123), (124, 132), (133, 140), (145, 153), (158, 159)],
            ),
        ],
    )

    _do_test(
        113,
        "What number of red shiny cubes are to the right of the thing that is to the left of the red metal object that is behind the gray matte sphere? 1",
        [
            (Item("gray", "small", "sphere", "rubber"), [(124, 128), (129, 134), (135, 141)]),
            (Item("red", "small", "cylinder", "metal"), [(88, 91), (92, 97), (98, 104), (113, 119)]),
            (Item("brown", "large", "cylinder", "rubber"), [(55, 60), (69, 83)]),
            (Item("red", "small", "cube", "metal"), [(15, 18), (19, 24), (25, 30), (35, 50), (143, 144)]),
        ],
    )

    _do_test(
        125,
        "Are there any cylinders in front of the shiny object that is right of the metallic thing in front of the large red object? yes",
        [
            (Item("red", "large", "cube", "rubber"), [(105, 110), (111, 114), (115, 121)]),
            (
                Item("brown", "large", "cylinder", "metal"),
                [(83, 88), (123, 126), (14, 23), (89, 100), (24, 35), (74, 82)],
            ),
            (Item("purple", "large", "cylinder", "metal"), [(40, 45), (46, 52), (61, 69)]),
        ],
    )

    _do_test(
        59,
        "What size is the yellow ball behind the sphere that is on the right side of the object that is behind the tiny yellow matte thing? large",
        [
            (Item("yellow", "small", "cube", "rubber"), [(106, 110), (111, 117), (118, 123), (124, 129)]),
            (
                Item("yellow", "large", "sphere", "metal"),
                [(17, 23), (131, 136), (29, 35), (80, 86), (24, 28), (95, 101)],
            ),
            (Item("yellow", "large", "sphere", "metal"), [(40, 46), (55, 75)]),
        ],
    )

    _do_test(
        78,
        "What shape is the brown rubber object that is in front of the brown rubber block on the right side of the matte object that is on the left side of the tiny rubber cylinder? cube",
        [
            (Item("cyan", "small", "cylinder", "rubber"), [(151, 155), (156, 162), (163, 171)]),
            (
                Item("brown", "large", "cube", "rubber"),
                [(24, 30), (18, 23), (173, 177), (106, 111), (46, 57), (112, 118), (127, 146), (31, 37)],
            ),
            (Item("brown", "small", "cube", "rubber"), [(62, 67), (68, 74), (75, 80), (81, 101)]),
        ],
    )

    _do_test(
        285,
        "There is a tiny shiny object that is behind the big ball that is to the right of the big metallic thing behind the big brown cube; what is its color? brown",
        [
            (Item("brown", "large", "cube", "rubber"), [(115, 118), (119, 124), (125, 129)]),
            (Item("purple", "large", "sphere", "metal"), [(85, 88), (89, 97), (98, 103), (104, 110)]),
            (Item("yellow", "large", "sphere", "rubber"), [(48, 51), (52, 56), (65, 80)]),
            (Item("brown", "small", "sphere", "metal"), [(11, 15), (16, 21), (22, 28), (37, 43), (150, 155)]),
        ],
    )

    _do_test(
        62,
        "What is the big thing that is in front of the block that is behind the block that is in front of the large shiny block made of? rubber",
        [
            (Item("blue", "large", "cube", "metal"), [(60, 66), (113, 118), (101, 106), (46, 51), (107, 112)]),
            (Item("green", "large", "cube", "rubber"), [(85, 96), (16, 21), (71, 76), (128, 134), (12, 15), (30, 41)]),
        ],
    )

    _do_test(
        305,
        "There is a small shiny object that is behind the tiny green ball; what number of cyan balls are in front of it? 1",
        [
            (Item("green", "small", "sphere", "rubber"), [(49, 53), (54, 59), (60, 64)]),
            (Item("purple", "small", "cube", "metal"), [(11, 16), (17, 22), (23, 29), (38, 44)]),
            (Item("cyan", "small", "sphere", "rubber"), [(81, 85), (86, 91), (96, 107), (112, 113)]),
        ],
    )

    _do_test(
        198,
        "There is a block in front of the big yellow metallic cube; are there any yellow things behind it? yes",
        [
            (
                Item("yellow", "large", "cube", "metal"),
                [(44, 52), (33, 36), (98, 101), (87, 93), (80, 86), (37, 43), (53, 57), (73, 79)],
            ),
            (Item("brown", "small", "cube", "metal"), [(11, 16), (17, 28)]),
        ],
    )

    _do_test(
        244,
        "There is a object in front of the metal cube that is to the right of the large cylinder; how big is it? small",
        [
            (Item("purple", "large", "cylinder", "metal"), [(73, 78), (79, 87)]),
            (Item("gray", "large", "cube", "metal"), [(34, 39), (40, 44), (53, 68)]),
            (Item("brown", "small", "cube", "rubber"), [(11, 17), (18, 29), (104, 109)]),
        ],
    )

    _do_test(
        162,
        "The cylinder that is to the right of the small object behind the tiny rubber cylinder is what color? red",
        [
            (Item("red", "small", "cylinder", "rubber"), [(21, 36), (101, 104), (65, 69), (77, 85), (4, 12), (70, 76)]),
            (Item("brown", "small", "cylinder", "metal"), [(41, 46), (47, 53), (54, 60)]),
        ],
    )

    _do_test(
        20,
        "There is a yellow thing to the right of the rubber thing on the left side of the gray rubber cylinder; what is its material? metal",
        [
            (Item("gray", "large", "cylinder", "rubber"), [(81, 85), (86, 92), (93, 101)]),
            (Item("yellow", "large", "cube", "rubber"), [(44, 50), (51, 56), (57, 76)]),
            (Item("yellow", "large", "sphere", "metal"), [(11, 17), (18, 23), (24, 39), (125, 130)]),
        ],
    )

    _do_test(
        50,
        "What is the shape of the small yellow rubber thing that is in front of the large yellow metal ball that is behind the small matte object? cube",
        [
            (
                Item("yellow", "small", "cube", "rubber"),
                [(138, 142), (118, 123), (59, 70), (124, 129), (38, 44), (130, 136), (45, 50), (25, 30), (31, 37)],
            ),
            (Item("yellow", "large", "sphere", "metal"), [(75, 80), (81, 87), (88, 93), (94, 98), (107, 113)]),
        ],
    )

    _do_test(
        1930,
        "What number of gray objects are there? 2",
        [
            (Item("gray", "large", "cube", "metal"), [(15, 19), (20, 27), (39, 40)]),
            (Item("gray", "small", "cylinder", "rubber"), [(15, 19), (20, 27), (39, 40)]),
        ],
    )

    _do_test(
        410,
        "Is the shape of the shiny thing that is right of the big gray sphere the same as  the large cyan matte thing? no",
        [
            (Item("gray", "large", "sphere", "rubber"), [(53, 56), (57, 61), (62, 68)]),
            (Item("gray", "small", "sphere", "metal"), [(20, 25), (26, 31), (40, 48)]),
            (Item("cyan", "large", "cylinder", "rubber"), [(86, 91), (92, 96), (97, 102), (103, 108)]),
        ],
    )
    return True


def main(args):
    clevr_path = Path(args.clevr_path)
    clevr_box_path = Path(args.clevr_box_path)

    with open(clevr_path / "questions/CLEVR_val_questions.json") as f:
        print("loading questions...")
        questions = json.load(f)
    templates = load_templates()

    with open(clevr_path / "scenes/CLEVR_val_scenes.json") as f:
        scenes = json.load(f)
    test_suite(templates, questions, scenes)

    print("tests ok")


if __name__ == "__main__":
    main(parse_args())
