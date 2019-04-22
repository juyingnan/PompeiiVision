from scipy import io as sio
import sys
import ClusterMatching
from sklearn import cluster


def hierarchical_clustering(data, log=False):
    print("Compute unstructured hierarchical clustering...")
    ward = cluster.AgglomerativeClustering(n_clusters=cluster_number,  # connectivity=connectivity,
                                           linkage='ward').fit(data)
    correct_count = 0
    if log:
        for k in range(len(ward.labels_)):
            print(str(ward.labels_[k] + 1) + '\t' + str(y[k] + 1))
            if ward.labels_[k] == y[k]:
                correct_count += 1
    # if path == '':
    #     write_csv(y, ward.labels_ + 1, path='csv/hierarchical_{0}.csv'.format(cluster_number))
    # else:
    #     write_csv(y, ward.labels_ + 1, path=path)
    return ward.labels_


def k_means_clustering(data, log=False):
    print("Compute K-means clustering...")
    k_means = cluster.KMeans(n_clusters=cluster_number)
    k_means.fit(data)
    if log:
        # print(k_means.labels_)
        # print(train_label)
        for k in range(len(k_means.labels_)):
            print(str(k_means.labels_[k] + 1) + '\t' + str(y[k] + 1))
    # if path == '':
    #     write_csv(train_label, k_means.labels_ + 1, path='csv/kmeans_{0}.csv'.format(cluster_number))
    # else:
    #     write_csv(train_label, k_means.labels_ + 1, path=path)
    return k_means.labels_


def assign_set(cat_list):
    # init result
    result = list()
    for i in range(len(set(cat_list))):
        result.append(set())

    # assign file into sets
    for i in range(len(cat_list)):
        result[cat_list[i]].add(i)

    return result


input_file_name = 'raw_50_g'

if len(sys.argv) >= 2:
    input_file_name = sys.argv[1]

mat_path = '../mat/' + input_file_name + '.mat'
digits = sio.loadmat(mat_path)
X, y = digits.get('feature_matrix'), digits.get('label')[0]  # X: nxm: n=67//sample, m=12,10,71,400//feature
file_names, indexes = digits.get('file_name'), digits.get('index')[0]
n_samples, n_features = X.shape
cluster_number = 8

# cluster_result = hierarchical_clustering(X, log=False, path=csv_path)
cluster_result = k_means_clustering(X, log=False)

reference = assign_set(y)
actual = assign_set(cluster_result)
print(reference)
print(actual)

best = ClusterMatching.find_best_match_cats2(reference, actual, repeat=True)
ClusterMatching.find_route2(reference, actual, best, "Human", "K-Means", sankeymatic_output_format=True)
