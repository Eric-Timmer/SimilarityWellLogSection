import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from mca_algorithm import load, standardize
from numba import jit
from skimage import exposure
from sklearn.cluster import MiniBatchKMeans


@jit(nopython=True)
def non_nan(x):
    temp = [0.]
    for i in x:
        if np.isnan(i):
            continue
        temp.append(i)
    temp.pop(0)
    x = np.array(temp)
    return x


@jit(nopython=True)
def pearson(x, y):
    xm = np.mean(x)
    ym = np.mean(y)

    nominator = np.sum((x-xm) * (y-ym))
    denominator = np.sqrt(np.sum(x-xm)**2) * np.sqrt(np.sum(y-ym)**2)

    if denominator == 0:
        return 1
    else:
        r = 1 - (nominator / denominator)
        return r


@jit(nopython=True)
def pearson_similarity(df):
    similarity = np.zeros(shape=(df.shape[1], df.shape[1]))
    for i in range(df.shape[1]):
        a_full = df[:, i]
        a = non_nan(a_full)
        if len(a) < 0.5 * len(a_full):
            continue
        for j in range(i + 1, df.shape[1]):
            b_full = df[:, j]
            b = non_nan(b_full)
            if len(b) < 0.5 * len(b_full):
                continue
            if len(a) > len(b):
                a = a[0:len(b)]
            elif len(b) > len(a):
                b = b[0:len(a)]
            dist = pearson(a, b)
            similarity[i, j] = dist
            similarity[j, i] = dist
    return similarity


@jit(nopython=True)
def compare_histograms(df):
    similarity = np.zeros(shape=(df.shape[1], df.shape[1]))
    for i in range(df.shape[1]):
        a_full = df[:, i]
        a = non_nan(a_full)
        if len(a) < 0.5*len(a_full):
            continue

        hist1, _ = np.histogram(a)
        for j in range(i, df.shape[1]):
            b = non_nan(df[:, j])
            if len(b) < 0.5*len(a_full):
                continue
            hist2, _ = np.histogram(b)
            diff = np.mean(np.abs(hist1 - hist2))
            similarity[i, j] = diff
    return similarity


# @jit(nopython=True)
def compare_frequency(df):
    similarity = np.zeros(shape=(df.shape[1], df.shape[1]))
    for i in range(df.shape[1]):
        a_full = df[:, i]
        a = non_nan(a_full)
        if len(a) < 0.5 * len(a_full):
            continue

        fft1 = np.real(np.fft.fft(a))[1:20] / float(len(a))
        for j in range(i, df.shape[1]):
            b = non_nan(df[:, j])
            if len(b) < 0.5 * len(a_full):
                continue
            fft2 = np.real(np.fft.fft(b))[1:20] / float(len(b))

            diff = np.mean(np.abs(fft1 - fft2))
            similarity[i, j] = diff
    return similarity


@jit(nopython=True)
def mean_similarity(df, mean):
    similarity = np.zeros(shape=(df.shape[1], df.shape[1]))

    for i in range(df.shape[1]):
        if np.isnan(mean[i]):
            continue
        for j in range(i + 1,  df.shape[1]):
            if np.isnan(mean[i]):
                continue
            diff = abs(mean[i] - mean[j])
            similarity[i, j] = diff
            similarity[j, i] = diff
    return similarity


@jit(nopython=True)
def z_norm(a):
    if np.max(a) == np.min(a):
        return np.zeros_like(a)
    else:
        a = (a - np.min(a)) / (np.max(a) - np.min(a))
        return a


@jit(nopython=True)
def z_norm_euclidean_similarity(df):
    similarity = np.zeros(shape=(df.shape[1], df.shape[1]))
    for i in range(df.shape[1]):
        a_full = df[:, i]
        a = z_norm(non_nan(a_full))
        if np.max(a) == 0:
            continue
        if len(a) < 0.5 * len(a_full):
            continue
        for j in range(i + 1, df.shape[1]):
            b_full = df[:, j]
            b = z_norm(non_nan(b_full))
            if np.max(b) == 0:
                continue
            if len(b) < 0.5 * len(b_full):
                continue

            n = len(a)
            if len(a) > len(b):
                a = a[0:len(b)]
                n = len(b)
            elif len(b) > len(a):
                b = b[0:len(a)]
            dist = np.sum(np.abs(a - b)) / float(n)
            similarity[i, j] = dist
            similarity[j, i] = dist
    return similarity


@jit(nopython=True)
def sorted_euclidean_similarity(df):
    similarity = np.zeros(shape=(df.shape[1], df.shape[1]))
    for i in range(df.shape[1]):
        a_full = df[:, i]
        a = np.sort(non_nan(a_full))
        if np.max(a) == 0:
            continue
        if len(a) < 0.5 * len(a_full):
            continue
        for j in range(i + 1, df.shape[1]):
            b_full = df[:, j]
            b = np.sort(non_nan(b_full))
            if np.max(b) == 0:
                continue
            if len(b) < 0.5 * len(b_full):
                continue

            n = len(a)
            if len(a) > len(b):
                a = a[0:len(b)]
                n = len(b)
            elif len(b) > len(a):
                b = b[0:len(a)]
            dist = np.sum(np.abs(a - b)) / float(n)
            similarity[i, j] = dist
            similarity[j, i] = dist
    return similarity


@jit(nopython=True)
def euclidean_similarity(df):
    similarity = np.zeros(shape=(df.shape[1], df.shape[1]))
    for i in range(df.shape[1]):
        a_full = df[:, i]
        a = non_nan(a_full)
        if len(a) < 0.5 * len(a_full):
            continue
        for j in range(i + 1, df.shape[1]):
            b_full = df[:, j]
            b = non_nan(b_full)
            if len(b) < 0.5 * len(b_full):
                continue

            n = len(a)
            if len(a) > len(b):
                a = a[0:len(b)]
                n = len(b)
            elif len(b) > len(a):
                b = b[0:len(a)]
            dist = np.sum(np.abs(a - b)) / float(n)
            similarity[i, j] = dist
            similarity[j, i] = dist
    return similarity


@jit(nopython=True)
def rescale_similarity(similarity):
    nan_max = np.max(similarity)
    nan_min = np.min(similarity)

    if nan_max != 0 and not np.isnan(nan_max):
        if nan_min != 0 and not np.isnan(nan_min):
            similarity = (similarity - nan_min) / (nan_max - nan_min)

        else:
            similarity = similarity  / nan_max
    else:
        similarity = np.zeros_like(similarity)

    return similarity


def convert_dict_to_similarity(similarity_dict):
    similarity = None
    for i in similarity_dict:
        sub_sim = None
        for j in similarity_dict[i]:
            if sub_sim is None:
                sub_sim = rescale_similarity(j)
            else:
                sub_sim += rescale_similarity(j)
        sub_sim = rescale_similarity(sub_sim)
        if similarity is None:
            similarity = sub_sim
        else:
            similarity += sub_sim

    return similarity


def assign_to_dict(similarity_dict, key, sub):
    if key == "mean":
        mean = np.nanmean(sub, axis=0)
        out = mean_similarity(sub, mean)
    elif key == "euclidean":
        out = euclidean_similarity(sub)
    elif key == "pearson":
        out = pearson_similarity(sub)
    elif key == "fft":
        out = compare_frequency(sub)
    elif key == "hist":
        out = compare_histograms(sub)
    elif key == "sorted_euclidean":
        out = sorted_euclidean_similarity(sub)
    elif key == "z_norm_euclidean":
        out = z_norm_euclidean_similarity(sub)
    else:
        raise UserWarning("Error: The similarity measure you specified wasn't found in the 'assign_to_dict' function")

    try:
        similarity_dict[key].append(out)
    except KeyError:
        similarity_dict[key] = [out]
    return similarity_dict


def zone_align(df, zones):
    increment = int(df.shape[0] / float(zones))
    overlap = int(.33*increment)

    start = 0
    end = increment + overlap

    similarity_dict = dict()
    mean_dict = dict()

    for z in tqdm(range(zones)):
        sub = df[start:end, :]
        start += increment
        end += increment
        if end > df.shape[0]:
            break

        # comparisons = ["mean", "euclidean", "z_norm_euclidean", "sorted_euclidean", "pearson", "fft", "hist"]
        comparisons = ["sorted_euclidean"]
        for k in tqdm(comparisons, desc="Zone %i, acquiring similarity matrices" % z):
            similarity_dict = assign_to_dict(similarity_dict, k, sub)
        mean_dict = assign_to_dict(mean_dict, "mean", sub)

    mean = convert_dict_to_similarity(mean_dict)
    similarity = convert_dict_to_similarity(similarity_dict)
    return similarity, mean


def traverse_similarity(similarity, mean):
    c_sim = np.copy(similarity)
    initial, min_index = np.unravel_index(np.nanargmin(similarity), similarity.shape)
    similarity_list = [np.NaN]
    path = [initial, min_index]
    for i in range(similarity.shape[0]):
        min_index = path[-1]
        min_list = np.argsort(c_sim[:, min_index])

        for j in min_list:
            if j in path:
                continue
            elif not np.isfinite(c_sim[min_index, j]):
                continue
            elif c_sim[min_index, j] == 0:
                continue
            else:
                path.append(j)
                similarity_list.append(mean[min_index, j])
                break
    return path, similarity_list


def find_breaks(similarity_list):
    m = np.nanmean(np.abs(np.roll(similarity_list, 1)[1:] - similarity_list[1:]))
    s = np.nanstd(np.abs(np.roll(similarity_list, 1)[1:] - similarity_list[1:]))
    c = m + 1.96*s
    breaks = list()
    labels = list()
    lab = 0
    for i, v in enumerate(similarity_list):
        if i < 2:
            continue
        if abs(v - similarity_list[i-1]) > c:
            lab += 1
            breaks.append(i)
        labels.append(lab)
    return breaks, labels


def find_depth_differences(path, df):
    diff_list = list()
    for i, p in enumerate(tqdm(path, desc="Finding height differences")):
        if i == 0:
            diff_list.append(0)
            continue
        a = len(non_nan(df[df.columns[p]].values))
        b = len(non_nan(df[df.columns[path[i-1]]].values))
        diff_list.append(abs(a-b))

    return diff_list


def plot_im(path, df, similarity_path, breaks, labels):
    aligned = None
    for i in tqdm(path, desc="Aligning Plot"):
        col = df[df.columns[i]].values
        if aligned is None:
            aligned = col
        else:
            aligned = np.vstack((aligned, col))

    im = exposure.equalize_adapthist(aligned.T)
    fig, axes = plt.subplots(nrows=3, gridspec_kw={"height_ratios": [1, 1, 10]}, sharex=True)

    axes[0].plot(similarity_path)
    labels = np.vstack((labels, labels))
    axes[1].imshow(labels, aspect="auto", interpolation="nearest")
    # axes[1].plot(diff)
    for i in breaks:
        axes[0].plot([i, i], [0, 1], c="black")
        # axes[2].plot([i, i], [0, aligned.shape[1]], c="black")
    axes[2].imshow(im, aspect="auto", interpolation="nearest")
    plt.show()


def rescale(a):
    a = (a- np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a))
    return a


def generate_label_indices(labels, path):
    groups_dict = dict()
    for i in range(0, max(labels) + 1):
        indices = np.where(np.array(labels) == i)[0]
        temp = list()
        for it in indices:
            temp.append(path[it])
        groups_dict[i] = temp
    return groups_dict


@jit(nopython=True)
def get_sub(a, indices):
    sub = a[:, indices]
    sub = sub[indices, :]
    return sub


@jit(nopython=True)
def calc_mean_score(mean_similarity, sub_select, a, b, shift_range):

    # select sub matrix
    sub_mean = get_sub(mean_similarity, sub_select)
    # only use entries that are greater than 10% of the similarity shape
    if shift_range[0] * mean_similarity.shape[0] > sub_mean.shape[0]:
        return np.inf
    elif sub_mean.shape[0] > shift_range[1] * mean_similarity.shape[0]:
        return np.inf
    else:
        a = get_sub(mean_similarity, a)
        b = get_sub(mean_similarity, b)
        score = np.abs(np.nanmean(a) - np.nanmean(b))
        return score


def assign_to_quarantine(groups_dict, quarantine):
    temp_dict = dict()
    for i in groups_dict:
        if len(groups_dict[i]) < 2:
            for entry in groups_dict[i]:
                quarantine.append(entry)
        else:
            temp_dict[i] = groups_dict[i]
    return temp_dict, quarantine


def trim_path(path, mean_similarity, quarantine):
    # USE LOCAL OUTLIER FACTOR OR SOMETHIGN LIKE THAT
    from sklearn.ensemble import IsolationForest
    outliers = IsolationForest(contamination="auto", behaviour="new").fit_predict(mean_similarity)
    outlier_indices = list(path[np.where(outliers == -1)])
    keep = list(path[np.where(outliers == 1)])
    quarantine += outlier_indices

    return keep, quarantine


def realign_thick_quarantine(mean_similarity, labels, path, shift_range):
    groups_dict = generate_label_indices(labels, path)
    quarantine = list()
    for lab in labels:
        try:
            indices = groups_dict[lab]
        except KeyError:
            continue

        # find sub matrix with best mean similarity values
        best_score = np.inf
        best_new_path = []
        best_lab = None
        best_sub = None
        for ind in groups_dict:
            if ind == lab:
                continue
            # list of indices for sub matrix
            if len(groups_dict[ind]) < 2:
                continue
            sub_select = groups_dict[ind] + indices
            sub_select = np.array(sub_select)
            score = calc_mean_score(mean_similarity, sub_select, shift_range)
            if score < best_score:
                best_score = score
                best_new_path = sub_select
                best_lab = ind
                sub_mean = mean_similarity[:, sub_select]
                sub_mean = sub_mean[sub_select, :]
                best_sub = sub_mean
        if not np.isfinite(best_score):
            continue

        groups_dict.pop(best_lab)
        best_new_path, quarantine = trim_path(best_new_path, best_sub, quarantine)
        groups_dict[lab] = list(best_new_path)

        groups_dict, quarantine = assign_to_quarantine(groups_dict, quarantine)

    out_path = list()
    for lab in groups_dict:
        out_path += list(groups_dict[lab])

    # add quarantine at the end of the path
    print(len(out_path), len(quarantine))
    out_path += quarantine
    similarity_path = list()
    for i, p in enumerate(out_path):
        if i == 0:
            similarity_path.append(np.NaN)
        else:
            similarity_path.append(mean_similarity[p, out_path[i - 1]])

    return out_path, similarity_path


def realign_thick(mean_similarity, labels, path, shift_range):

    groups_dict = generate_label_indices(labels, path)
    for lab in labels:
        try:
            indices = groups_dict[lab]
            if len(indices) < 2:
                continue
        except KeyError:
            continue

        # find sub matrix with best mean similarity values
        best_score = np.inf
        best_new_path = []
        best_lab = None
        for ind in groups_dict:
            if ind == lab:
                continue
            # list of indices for sub matrix
            if len(groups_dict[ind]) < 2:
                continue
            sub_select = groups_dict[ind] + indices
            sub_select = np.array(sub_select)
            a = np.array(indices)
            b = np.array(groups_dict[ind])
            score = calc_mean_score(mean_similarity, sub_select,a,b, shift_range)
            if score < best_score:
                best_score = score
                best_new_path = sub_select
                best_lab = ind
        if not np.isfinite(best_score):
            continue
        groups_dict.pop(best_lab)
        groups_dict[lab] = list(best_new_path)

    out_path = list()
    for lab in groups_dict:
        out_path += groups_dict[lab]

    similarity_path = list()
    for i, p in enumerate(out_path):
        if i == 0:
            similarity_path.append(np.NaN)
        else:
            similarity_path.append(mean_similarity[p, out_path[i - 1]])

    return out_path, similarity_path


def peak_detection(similarity_list):
    plt.plot(similarity_list)
    plt.show()


    return



def main():
    # print("Loading log data")
    # df = pd.read_csv(r"S:\Timmer\dump\Python_Projects\StratPickQCML\Paper\extracted_las_data\AGS\GR_MONTNEY.csv",
    #                  index_col=0)
    # df = load(df)
    #
    # print("Standardizing data")
    # df = standardize(df)
    # # print("Generating similarity matrices")
    # # similarity, m = zone_align(df.values, zones=5)
    # # path, similarity_list = traverse_similarity(similarity, m)
    # # pickle.dump([similarity, m, path, similarity_list], open("dump.pkl", "wb"))
    similarity, m, path, similarity_list = pickle.load(open("dump.pkl", "rb"))
    peak_detection(similarity_list)
    # breaks, _ = find_breaks(similarity_list)


    # path.pop(0)
    # print(path)
    # print(len(path), len(set(path)))
    # for i in tqdm(range(0, 10), desc="smoothing 10x"):
    #     breaks, labels = find_breaks(similarity_list)
    #     if len(set(labels)) < 5:
    #         break
    #     path, similarity_list = realign_thick(m, labels, path, shift_range=[0.0, 1.0])
    #
    # plot_im(path, df, similarity_list, breaks, labels)


if __name__ == "__main__":
    main()
