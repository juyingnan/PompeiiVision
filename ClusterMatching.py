import csv
from collections import OrderedDict


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


def find_best_match_cat4(ref_sets, match_sets):
    best_match = []
    best_match_count = 0
    for i in range(4):
        for j in range(4):
            for m in range(4):
                for n in range(4):
                    index_list = [i, j, m, n]
                    index_set = set(index_list)
                    if len(index_set) == 4:
                        match_count = 0
                        for k in range(4):
                            match_count += count_match(ref_sets[k], match_sets[index_list[k]])
                        if match_count >= best_match_count:
                            best_match_count = match_count
                            best_match = index_list
                        # print(index_list, match_count)
    print(best_match, '{0} / {1}'.format(best_match_count, sum([len(set) for set in ref_sets])))
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


# read scv
csv_path = 'csv/1st_4.csv'

# assign sets and lists
human_cat, kmeans_cat, ae_cat = read_csv(csv_path)

# find best match
human_kmeans_match = find_best_match_cat4(human_cat, kmeans_cat)
human_ae_match = find_best_match_cat4(human_cat, ae_cat)

# find route from best match
print('****************')
print('human & kmeans')
find_route(human_cat, kmeans_cat, human_kmeans_match, "Human", "K-Means", sankeymatic_output_format=True)
print('****************')
print('human & ae')
find_route(human_cat, ae_cat, human_ae_match, "Human", "AutoEncoder", sankeymatic_output_format=True)
print('****************')
