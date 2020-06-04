import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm
# from dtw import dtw
from sklearn import cluster
import pickle
from skimage import exposure

from sklearn.neighbors import LocalOutlierFactor



def load(amalgam):
    cols = list()
    for i in amalgam.columns.values:
        if "." in i:
            continue
        else:
            cols.append(i)
    amalgam = amalgam[cols]
    return amalgam


def standardize(df, by_column=True, min_max=False):
    if by_column is True:
        temp = pd.DataFrame()
        for col in df.columns.values:
            data = df[col]
            stand = (data - np.nanmean(data)) / np.nanstd(data)
            if min_max is True:
                stand = (stand - np.nanmin(stand)) / (np.nanmax(stand) - np.nanmin(stand))
            temp[col] = stand
        df = temp
    else:
        df = (df - df.mean()) / df.std()
    return df


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
def dist(x, y):
    d = (x-y) * (x-y)
    return d

@jit(nopython=True)
def prunedDTW(vector_a, vector_b, w):

    m = len(vector_a)
    r = np.floor(m*w)

    # variables to implement the pruning
    sc = 0
    ec = 0
    lp = None  # last pruning

    # instead of using matrix of size )(m^2) or O(mr), we will reuse two arrays of size O(m)
    cost = np.empty_like(vector_a)
    cost_prev = np.empty_like(vector_a)
    ub_partials = np.empty_like(vector_a)

    for j in range(0, m-1):
        cost[j] = np.inf
        cost_prev[j] = np.inf
        if j == 0:
            ub_partials[m-j-1] = dist(vector_a[m-j-1], vector_b[m-j-1])
        else:
            ub_partials[m-j-1] = ub_partials[m-j] + dist(vector_a[m-j-1], vector_b[m-j-1])
    ub = ub_partials[0]

    for i in range(0, m-1):
        found_sc = False
        pruned_ec = False
        next_ec = i+r+1

        ini_j = int(max(0, max(i-r, sc)))
        for j in range(ini_j, int(min(m-1, i+r))):
            # initialize all rows and columns
            if i == 0 and j == 0:
                cost[j] = dist(vector_a[0], vector_b[0])
                found_sc = True
                continue
            if j == ini_j:
                y = np.inf
            else:
                y = cost[j-1]
            if i == 0 or j == i+r or j >= lp:
                x = np.inf
            else:
                x = cost_prev[j]
            if i == 0 or j == 0 or j > lp:
                z = np.inf
            else:
                z = cost_prev[j-1]

            # classic DTW calculation
            cost[j] = min(min(x, y), z + dist(vector_a[i], vector_b[i]))

            # pruning criteria
            if found_sc is False and cost[j] <= ub:
                sc = j
                found_sc = True
            if cost[j] > ub:
                if j > ec:
                    lp = j
                    pruned_ec = True
                    break
            else:
                next_ec = j+1
        ub = ub_partials[i+1] + cost[i]

        # move current array to previous array
        cost_tmp = cost
        cost = cost_prev
        cost_prev = cost_tmp
        # pruning statistics update
        if sc > 0:
            cost_prev[sc-1] = np.inf
        if pruned_ec is False:
            lp = i+r+1
        ec = next_ec
    # the DTW distance is in the last cell in the matrix of size )(m^2) or at the middle of our array
    final_dtw = cost_prev[-1]
    return final_dtw


@jit(nopython=True)
def freq_dist(a, b):
    d = np.sum(np.abs(a-b))
    return d


def compute_freq_sim(freq_list):
    distances = np.zeros((len(freq_list), len(freq_list)))
    for i, f1 in enumerate(tqdm(freq_list, desc="comparing logs")):
        for j, f2 in enumerate(freq_list):
            if j == i:
                continue
            f2 = freq_list[j]
            d = freq_dist(f1, f2)
            distances[i, j] = d
            # distances[j, i] = d
    return distances


def compute_frequency_space(df):
    shape = df.shape
    freq_list = list()
    for i in tqdm(range(0, shape[1]), desc="Converting logs to frequency space"):
        a = non_nan(df[:, i])
        # mod = MiniBatchKMeans(n_clusters=100).fit(a.reshape(-1, 1))
        # a = mod.labels_
        w = np.real(np.fft.fft(a))[1:100] / float(len(a))
        if len(w) != 99:
            w = np.array([np.NaN]*99)
        freq_list.append(w)

    distances = compute_freq_sim(freq_list)
    return distances



def mca(similarity_matrix, df):
    model = cluster.MiniBatchKMeans(n_clusters=20).fit(similarity_matrix)
    labels = model.labels_

    sorted_df = None
    for i in set(labels):
        j = np.where(labels == i)
        sorted_df = df[df.columns[j]]


        # TODO re-assemble matrix
        sub_sim = similarity_matrix[:, j]
        sub_sim = sub_sim[j, :]



        plt.imshow(sub_sim)
        # from skimage import exposure
        # im = exposure.equalize_adapthist(sorted_df)
        # plt.imshow(sorted_df, aspect="auto", interpolation="nearest")
        plt.show()


    #
    #     if sorted_df is None:
    #         sorted_df = df[df.columns.values[j]].values
    #     else:
    #         sorted_df = np.vstack((sorted_df, df[df.columns.values[j]]))
    # from skimage import exposure
    # im = exposure.equalize_adapthist(sorted_df.T)
    # plt.imshow(im, aspect="auto", interpolation="nearest")
    # plt.show()
    # return



@jit(nopython=True, parallel=True)
def compute_similarities_symmetric(df):
    shape = df.shape
    distances = np.zeros((shape[1], shape[1]))
    count = 0
    for i in range(0, shape[1] - 1):
        early_exit = np.inf  # set up early exit to speed up process after 100 iterations
        last_100 = 0.  # generate mean of last 100 measurements to determine which pass through eventually.
        for j in range(i+1, shape[1]):
            a = non_nan(df[:, i])
            b = non_nan(df[:, j])
            d = mjc_dxy(a, b, early_exit)

            if j < int(0.1*shape[1]):
                last_100 += d
            elif j == int(0.1*shape[1]):
                early_exit = last_100 / float(0.1*shape[1])
            distances[i, j] = d
            distances[j, i] = d
        count += 1
        # if count % 100 == 0:
        print(count, "of", shape[1])
    return distances


@jit(nopython=True)
def mjc_dxy(x, y, early_exit):
    t_x = 0
    t_y = 0
    d_xy = 0
    x = non_nan(x)
    y = non_nan(y)
    phi_x = 4 * np.std(x) / float(len(x))
    phi_y = 4 * np.std(y) / float(len(y))
    while t_x < len(x) and t_y < len(y):
        c, t_x, t_y = cmin(x, t_x, y, t_y, phi_y)
        d_xy += c
        if d_xy > early_exit:
            return np.inf
        if t_x > len(x) - 1 or t_y > len(y) - 1:
            break
        c, t_y, t_x = cmin(y, t_y, x, t_x, phi_x)
        d_xy += c
        if d_xy > early_exit:
            return np.inf
    return d_xy


@jit(nopython=True)
def cmin(x, t_x, y, t_y, phi):
    c_min = np.inf
    delta, delta_min = 0, 0
    while t_y + delta < len(y):
        c = (phi * delta) ** 2
        if c >= c_min:
            if t_y + delta > t_x:
                break
        else:
            c = c + (x[t_x] - y[t_y] + delta) ** 2
            if c < c_min:
                c_min = c
                delta_min = delta
        delta = delta + 1
    t_x = t_x + 1
    t_y = t_y + delta_min + 1
    return c_min, t_x, t_y


def loop_mat(well_indices, temp_sim, similarity_matrix):
    for i in well_indices:
        for j in well_indices:
            if i == j:
                continue
            temp_sim[i, j] = similarity_matrix[i, j]
    return temp_sim


def split_similarity_by_cluster(cluster_df, df, similarity_matrix, n):
    similarity_matrix = similarity_matrix.values
    sim_matrix_list = list()
    for i in tqdm(range(0, n)):
        wells = cluster_df["Well"][cluster_df["Label"] == i].values

        well_indices = [df.columns.get_loc(c) for c in df.columns if c in wells]
        temp_sim = np.zeros_like(similarity_matrix)
        temp_sim[temp_sim == 0] = np.NaN

        temp_sim = loop_mat(well_indices, temp_sim, similarity_matrix)
        sim_matrix_list.append(temp_sim)
    return sim_matrix_list


def mca_by_cluster(sim_list, df):
    n = 0
    for similarity_matrix in sim_list:
        for i in range(0, similarity_matrix.shape[0]):
            similarity_matrix[i, i] = np.inf

        # from similarity matrix, choose pair with best similarity score and do alignment on those two
        i, j = np.unravel_index(np.nanargmin(similarity_matrix), similarity_matrix.shape)

        # set i and j as 'nan' values now
        similarity_matrix[i, :] = np.NaN
        similarity_matrix[:, i] = np.NaN

        # extract the well log values from those indices
        a = df[df.columns[i]].values
        b = df[df.columns[j]].values

        # align a and b
        # temp = pairwise_align(a, b)
        aligned = np.vstack((a, b))

        # now, loop through wells until matrix is entirely non nan and all sequences have been aligned!
        for step in tqdm(range(0, len(df.columns) - 1), "Multiple Sequence Alignment, cluster %i" % n):
            i = np.copy(j)
            try:
                j = np.nanargmin(similarity_matrix[:, j])
            except ValueError:  # all nan slice occurred!
                break
            a = np.copy(b)
            b = df[df.columns[j]].values
            similarity_matrix[i, :] = np.NaN
            similarity_matrix[:, i] = np.NaN
            # temp = pairwise_align(a, b)

            aligned = np.vstack((aligned, b))

        plt.imshow(aligned.T, interpolation="nearest", aspect="auto")
        plt.show()
        n += 1


def multiple_sequence_alignment(similarity_matrix, df):
    """
    The easiest way to align multiple sequences is to do a number of pairwise alignments.
    First get pairwise similarity scores for each pair and store those scores.
    This is the most expensive part of the process. Choose the pair that has the best similarity score
    and do that alignment.
    Now pick the sequence which aligned best to one of the sequences in the set of aligned sequences,
    and align it to the aligned set, based on that pairwise alignment. Repeat until all sequences are in.

    https://stackoverflow.com/questions/5813859/how-to-compute-multiple-sequence-alignment-for-text-strings
    :param similarity_df:
    :return:
    """
    print("Finding best matched pair")
    aligned_well_list = list()
    for i in range(0, similarity_matrix.shape[0]):
        similarity_matrix[i, i] = np.inf

    # from similarity matrix, choose pair with best similarity score and do alignment on those two
    i, j = np.unravel_index(np.nanargmin(similarity_matrix), similarity_matrix.shape)

    # set i and j as 'nan' values now
    similarity_matrix[i, :] = np.NaN
    similarity_matrix[:, i] = np.NaN

    # extract the well log values from those indices
    a = df[df.columns[i]].values
    b = df[df.columns[j]].values

    # align a and b
    aligned = np.vstack((a, b))

    well = df.columns[i]
    aligned_well_list.append(well)

    well = df.columns[j]
    aligned_well_list.append(well)

    # now, loop through wells until matrix is entirely non nan and all sequences have been aligned!
    for _ in tqdm(range(0, len(df.columns) - 1), "Multiple Sequence Alignment"):
        i = np.copy(j)
        try:
            j = np.nanargmin(similarity_matrix[:, j])
        except ValueError:
            break
        a = np.copy(b)
        b = df[df.columns[j]].values
        similarity_matrix[i, :] = np.NaN
        similarity_matrix[:, i] = np.NaN
        # temp = pairwise_align(a, b)

        aligned = np.vstack((aligned, b))
        well = df.columns[j]
        aligned_well_list.append(well)

    im = exposure.equalize_adapthist(aligned.T)
    plt.imshow(im, aspect="auto", interpolation="nearest")

    plt.show()
    return


def clustering(distances, wells, n):
    model = cluster.KMeans(n_clusters=n, n_jobs=-1, verbose=1)
    model.fit(distances)
    labels = model.labels_
    # model = KNeighborsClassifier(n_jobs=-1).fit(distances)
    # labels = model.predict(distances)
    temp = pd.DataFrame()
    temp["Label"] = labels
    temp["Well"] = wells
    return temp


def main(plot_by_cluster=False, show_random=False):
    # print("Loading data")
    df = pd.read_csv(r"S:\Timmer\dump\Python_Projects\StratPickQCML\Paper\extracted_las_data\AGS\GR_Montney.csv", index_col=0)
    df = load(df)
    print("Standardizing data")
    df = standardize(df)

    if show_random is True:
        plt.imshow(df, interpolation="nearest", aspect="auto")
        plt.show()
        plt.clf()
        plt.close()
        exit()
    print("Computing similarities")
    # similarities = compute_similarities_symmetric(df.values)
    similarities = compute_frequency_space(df.values)

    pd.DataFrame(similarities).to_csv("similarities.csv", index=False)
    # mca(similarities, df)
    #
    if plot_by_cluster is True:
        similarities_list = pickle.load(open("similarities_list.pkl", "rb"))
        mca_by_cluster(similarities_list, df)
    else:
        # cluster_df = clustering(similarities, df.columns.values, n=10)
        multiple_sequence_alignment(similarities, df)


if __name__ == "__main__":
    main()
