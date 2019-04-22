import csv
from collections import OrderedDict
import itertools


def read_csv(path):
    cat = 4
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # skip header
        num_cols = len(next(reader))

        # init result
        result = []
        for i in range(num_cols - 1):
            result.append([])
            for j in range(cat):
                result[-1].append(set())

        # assign file into sets
        for row in reader:
            filename = row[0]
            for i in range(num_cols - 1):
                index = int(row[i + 1]) - 1
                if index >= 0:
                    result[i][index].add(filename)

        if len(result) == 1:
            return result[0]
        else:
            return result


def write_roman(number):
    roman = OrderedDict()
    roman[1000] = "M"
    roman[900] = "CM"
    roman[500] = "D"
    roman[400] = "CD"
    roman[100] = "C"
    roman[90] = "XC"
    roman[50] = "L"
    roman[40] = "XL"
    roman[10] = "X"
    roman[9] = "IX"
    roman[5] = "V"
    roman[4] = "IV"
    roman[1] = "I"

    def roman_num(num):
        for r in roman.keys():
            x, y = divmod(num, r)
            yield roman[r] * x
            num -= (r * x)
            if num > 0:
                roman_num(num)
            else:
                break

    return "".join([a for a in roman_num(number)])


def count_match(ref_set, match_set):
    intersect = set(ref_set).intersection(set(match_set))
    return len(intersect)


def most_frequent(_list):
    return max(set(_list), key=_list.count)


def find_best_match_cats(ref_sets, match_sets, repeat=True):
    ref_cat_count = len(ref_sets)
    match_cat_count = len(match_sets)
    best_match = []
    best_match_count = 0
    for index_list in itertools.product(range(ref_cat_count), repeat=match_cat_count):
        index_set = set(index_list)
        if repeat or len(index_set) == ref_cat_count:
            match_count = 0
            for k in range(match_cat_count):
                match_count += count_match(ref_sets[index_list[k]], match_sets[k])
            if match_count >= best_match_count:
                best_match_count = match_count
                best_match = index_list
            # print(index_list, match_count)
    max_match_count = count_match([item for ref_set in ref_sets for item in ref_set],
                                  [item for match_set in match_sets for item in match_set])
    print(best_match, '{0} / {1}'.format(best_match_count, max_match_count))
    return best_match


def find_best_match_cats2(ref_sets, match_sets, repeat=True):
    ref_cat_count = len(ref_sets)
    match_cat_count = len(match_sets)
    best_match = []
    best_match_count = 0
    if repeat:
        for i in range(match_cat_count):
            max_match_groups = [count_match(ref_set, match_sets[i]) for ref_set in ref_sets]
            best_match.append(max_match_groups.index(max(max_match_groups)))
            best_match_count += count_match(ref_sets[best_match[-1]], match_sets[i])
    else:
        assert ref_cat_count == match_cat_count
        for index_list in itertools.product(range(ref_cat_count), repeat=match_cat_count):
            index_set = set(index_list)
            if len(index_set) == ref_cat_count:
                match_count = 0
                for k in range(match_cat_count):
                    match_count += count_match(ref_sets[index_list[k]], match_sets[k])
                if match_count >= best_match_count:
                    best_match_count = match_count
                    best_match = index_list
                # print(index_list, match_count)
    max_match_count = count_match([item for ref_set in ref_sets for item in ref_set],
                                  [item for match_set in match_sets for item in match_set])
    print(best_match, '{0} / {1}'.format(best_match_count, max_match_count))
    return best_match


def find_route(ref_sets, match_sets, match_seq, ref_set_title='ref', matched_set_title='matched',
               sankeymatic_output_format=False):
    # http://sankeymatic.com/build/
    # output format
    matched_sets = []
    for index in match_seq:
        matched_sets.append(match_sets[index])
    ref_length_list = [len(ref_set) for ref_set in ref_sets]
    matched_length_list = [len(matched_set) for matched_set in matched_sets]
    print(ref_set_title, "_length_list", ref_length_list)
    print(matched_set_title, "_length_list", matched_length_list)
    print('from {0} => {1}:'.format(ref_set_title, matched_set_title))
    for i in range(len(ref_sets)):
        ref_set = ref_sets[i]
        for j in range(len(matched_sets)):
            matched_set = matched_sets[j]
            if sankeymatic_output_format:
                print('{0}-{1} [{4}] {2}-{3}'.format(ref_set_title, write_roman(i + 1),
                                                     matched_set_title, write_roman(j + 1),
                                                     count_match(ref_set, matched_set))) \
                    if count_match(ref_set, matched_set) > 0 else None
            else:
                print('{0}:{1} => {2}:{3}'.format(ref_set_title, i, matched_set_title, j),
                      '{0} / {1}'.format(count_match(ref_set, matched_set), ref_length_list[i]))
    print('from {0} => {1}:'.format(matched_set_title, ref_set_title))
    for i in range(len(matched_sets)):
        matched_set = matched_sets[i]
        for j in range(len(ref_sets)):
            ref_set = ref_sets[j]
            if sankeymatic_output_format:
                print('{0}-{1} [{4}] {2}-{3}'.format(matched_set_title, write_roman(i + 1),
                                                     ref_set_title, write_roman(j + 1),
                                                     count_match(ref_set, matched_set))) \
                    if count_match(ref_set, matched_set) > 0 else None
            else:
                print('{0}:{1} => {2}:{3}'.format(matched_set_title, i, ref_set_title, j),
                      '{0} / {1}'.format(count_match(ref_set, matched_set), matched_length_list[i]))

    # matching matrix printing
    head = ''
    for j in range(len(matched_sets)):
        head += '\t{0}-{1}'.format(matched_set_title, write_roman(j + 1))
    print(head)
    for i in range(len(ref_sets)):
        ref_set = ref_sets[i]
        line = '{0}-{1}'.format(ref_set_title, write_roman(i + 1))
        for j in range(len(matched_sets)):
            matched_set = matched_sets[j]
            line += '\t{0}'.format(count_match(ref_set, matched_set))
        print(line)


def find_route2(ref_sets, match_sets, match_seq, ref_set_title='ref', matched_set_title='matched',
                sankeymatic_output_format=False):
    # http://sankeymatic.com/build/
    # output format
    ref_length_list = [len(ref_set) for ref_set in ref_sets]
    matched_length_list = [len(match_set) for match_set in match_sets]
    print(ref_set_title, "_length_list", ref_length_list)
    print(matched_set_title, "_length_list", matched_length_list)
    print('from {0} => {1}:'.format(ref_set_title, matched_set_title))
    for i in range(len(ref_sets)):
        ref_set = ref_sets[i]
        for j in range(len(match_sets)):
            matched_set = match_sets[j]
            if sankeymatic_output_format:
                print('{0}-{1} [{4}] {2}-{3}'.format(ref_set_title, write_roman(i + 1),
                                                     matched_set_title, write_roman(match_seq[j] + 1),
                                                     count_match(ref_set, matched_set))) \
                    if count_match(ref_set, matched_set) > 0 else None
            else:
                print('{0}:{1} => {2}:{3}'.format(ref_set_title, i, matched_set_title, j),
                      '{0} / {1}'.format(count_match(ref_set, matched_set), ref_length_list[i]))
    print('from {0} => {1}:'.format(matched_set_title, ref_set_title))
    for i in range(len(match_sets)):
        matched_set = match_sets[i]
        for j in range(len(ref_sets)):
            ref_set = ref_sets[j]
            if sankeymatic_output_format:
                print('{0}-{1} [{4}] {2}-{3}'.format(matched_set_title, write_roman(match_seq[j] + 1),
                                                     ref_set_title, write_roman(j + 1),
                                                     count_match(ref_set, matched_set))) \
                    if count_match(ref_set, matched_set) > 0 else None
            else:
                print('{0}:{1} => {2}:{3}'.format(matched_set_title, i, ref_set_title, j),
                      '{0} / {1}'.format(count_match(ref_set, matched_set), matched_length_list[i]))

    # matching matrix printing
    head = ''
    for j in range(len(match_sets)):
        head += '\t{0}-{1}'.format(matched_set_title, write_roman(j + 1))
    print(head)
    for i in range(len(ref_sets)):
        ref_set = ref_sets[i]
        line = '{0}-{1}'.format(ref_set_title, write_roman(i + 1))
        for j in range(len(match_sets)):
            matched_set = match_sets[j]
            line += '\t{0}'.format(count_match(ref_set, matched_set))
        print(line)


if __name__ == '__main__':
    # read scv
    csv_path = '../csv/1st_4.csv'

    # assign sets and lists
    # human_cat, kmeans_cat, ae_cat = read_csv(csv_path)
    human_cat = read_csv('../csv/human_4.csv')
    kmeans_cat = read_csv('../csv/kmeans_4.csv')
    ae_cat = read_csv('../csv/ae_4.csv')
    hierarchical_cat = read_csv('../csv/hierarchical_4.csv')

    # find best match
    human_kmeans_match = find_best_match_cats(human_cat, kmeans_cat)
    human_hierarchical_match = find_best_match_cats(human_cat, hierarchical_cat)
    human_ae_match = find_best_match_cats(human_cat, ae_cat)
    kmeans_hierarchical_match = find_best_match_cats(kmeans_cat, hierarchical_cat)

    # find route from best match
    print('****************')
    print('human & kmeans')
    find_route(human_cat, kmeans_cat, human_kmeans_match, "Human", "K-Means", sankeymatic_output_format=True)
    print('****************')
    print('human & ae')
    find_route(human_cat, ae_cat, human_ae_match, "Human", "AutoEncoder", sankeymatic_output_format=True)
    print('****************')
    print('kmeans & hierarchical')
    find_route(human_cat, hierarchical_cat, human_hierarchical_match, "Human", "Hierarchical",
               sankeymatic_output_format=True)
    print('****************')
    print('kmeans & hierarchical')
    find_route(kmeans_cat, hierarchical_cat, kmeans_hierarchical_match, "Kmeans", "Hierarchical",
               sankeymatic_output_format=True)
    print('****************')

    # loop test
    # import os
    #
    # folder_path = 'csv/'
    # file_path_list = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if
    #                   os.path.isfile(os.path.join(folder_path, file_name)) and (
    #                           file_name.startswith('kmeans_4_') or file_name.startswith('hierarchical_4_'))]
    # for file_path in file_path_list:
    #     kmeans_cat = read_csv(file_path)
    #     print('r{0} '.format(file_path), end='')
    #     human_kmeans_match = find_best_match_cat4(human_cat, kmeans_cat)
