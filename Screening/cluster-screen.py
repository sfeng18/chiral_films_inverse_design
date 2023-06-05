#!/usr/bin/env python
import numpy as np

import uv
import sys
# import time

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib.colors import ListedColormap

n_clusters = 3
task_sfx = ''
# task_sfx = '_rand'
ColorMap = ListedColormap(uv.ColorList[:3])


def feature_normalize(data):
    # return data
    mu = np.mean(data, axis=0)  # 均值
    std = np.std(data, axis=0)  # 标准差
    return (data - mu) / std


def feature_unnormalize(data, arr):
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return arr * std + mu


def norm_all(arr):
    max_all = np.max(arr, axis=0)
    min_all = np.min(arr, axis=0)
    return (arr - min_all) / (max_all - min_all), max_all, min_all


def onehot_input(arr):
    para_num = arr.shape[1]
    for i in range(para_num):
        unq_value = set(arr[:, i].tolist())
        print(i, unq_value)
    max_all = np.max(arr, axis=0)
    min_all = np.min(arr, axis=0)
    return (arr - min_all) / (max_all - min_all), max_all, min_all


def count_num(arr):
    para_num = arr.shape[1]
    counts = [{} for _ in range(para_num)]
    for i in range(para_num):
        for v in arr[:, i]:
            counts[i][v] = counts[i].get(v, 0) + 1
    return counts


if __name__ == "__main__":
    Num = {'.': 0, 't': 1, 'test': 0}
    Para_Dict = uv.get_para(sys.argv[1:], Num_Dict=Num)
    DataBasePath = './Data/uvdb.msgz'  # Your Database

    with uv.timer('Read data'):
        sdb = uv.UVSpectrumDataBase(DataBasePath)
        sdb.load()

    T_min = 0.1
    rc2label = {0: 'L', 1: '', 2: 'H'}
    TickSize = 18
    LabelSize = 24
    TitleSize = 16
    LegendSize = 16

    Colors, Films = [], []
    ExpParas = []
    Ratio = []
    MaxRatio = []
    avg_T0 = []
    Names = []
    for sd in sdb.FullData:
        # print(f'{sd.Name}: {len(sd.T0)} {len(sd.T90)}')
        Films.append(sd.Film)
        Colors.append(sd.Color)
        ExpParas.append((sd.Thickness, sd.Strain, sd.DyeTime))
        # ExpParas.append((sd.Strain, sd.DyeTime))
        my_x = np.array(sd.X)
        idx = np.where((my_x >= 400) & (my_x <= 800))[0]
        T0 = np.array(sd.T0)[idx]
        T90 = np.array(sd.T90)[idx]
        T0[T0 < T_min] = T_min
        T90[T90 < T_min] = T_min
        if len(T0) != len(T90):
            print(sd.Name)
            exit()
        my_ratio = np.log10(T0 / T90)
        max_ratio = np.max(my_ratio)
        idx = np.where(my_ratio > 0.3 * max_ratio)[0]
        Ratio.append(my_ratio)
        MaxRatio.append(max_ratio)
        mean_T0 = np.mean(T0[idx]) if len(idx) else np.mean(T0)
        avg_T0.append(mean_T0)
        Names.append(sd.Name)
        print(f'{sd.Name:28s} {max_ratio:6.2f} {mean_T0:6.2f}')

    Color_list = sorted(list(set(Colors)))
    Film_list = sorted(list(set(Films)))
    print('color:\n', Color_list)
    print('film:\n', Film_list)
    NColors = len(Color_list)
    NFilms = len(Film_list)
    Color_names = [uv.int2roman(_ + 1) for _ in range(NColors)]
    Color2num = {c: i for i, c in enumerate(Color_list)}
    Film2num = {c: i for i, c in enumerate(Film_list)}
    # ExpParas = np.array([(Color2num[c],Film2num[f],*p) for c,f,p in zip(Colors,Films,ExpParas)])
    ExpParas = np.array([(Color2num[c], *p) for c, p in zip(Colors, ExpParas)])

    # for normalized input
    paras_norm, paras_max, paras_min = norm_all(ExpParas)
    print('max:', paras_max)
    print('min:', paras_min)
    
    ### Set thresholds for high & low ld & T0

    Ratio = np.array(Ratio)
    MaxRatio = np.array(MaxRatio)
    avg_T0 = np.array(avg_T0)

    low_criterion_ld = np.quantile(MaxRatio, 0.4)
    high_criterion_ld = np.quantile(MaxRatio, 0.6)
    # high_criterion = 1.5
    Ratio_class = np.ones_like(MaxRatio, dtype=int)
    Ratio_class[MaxRatio < low_criterion_ld] = 0
    Ratio_class[MaxRatio >= high_criterion_ld] = 2
    print(f'MaxRatio: low: {low_criterion_ld:.3f}  high: {high_criterion_ld:.3f}')

    low_criterion_T0 = max(min(np.quantile(avg_T0, 0.4), 50), 30)
    high_criterion_T0 = np.quantile(avg_T0, 0.6)
    T0_class = np.ones_like(avg_T0, dtype=int)
    T0_class[avg_T0 < low_criterion_T0] = 0
    T0_class[avg_T0 >= high_criterion_T0] = 2
    print(f'avg_T0: low: {low_criterion_T0:.3f}  high: {high_criterion_T0:.3f}')

    ### Classification by thresholds & plot histogram

    real_class = np.ones_like(avg_T0, dtype=int)
    real_class[(T0_class == 0) & (Ratio_class == 0)] = 0
    real_class[(T0_class == 2) & (Ratio_class == 2)] = 2

    hist_figsize = (9.6, 4.8)
    hist_axissize = (0.11, 0.15, 0.87, 0.83)
    hist_alpha = 0.3

    Color_H2 = 'xkcd:dark red'
    Color_L1 = 'xkcd:bright blue'
    Color_L2 = 'xkcd:dark blue'

    counts, bins = np.histogram(MaxRatio, bins=50)
    x = 0.5 * (bins[1:] + bins[:-1])
    dx = bins[1] - bins[0]
    fig = plt.figure(figsize=hist_figsize)
    ax = fig.add_axes(hist_axissize)
    plt.hist(x, bins, weights=counts, ec="black", fc="xkcd:light grey", alpha=0.7)
    Xst, Xed = (x[0] - dx, x[-1] + dx)
    # Yst, Yed = ax.get_ylim()
    Yst, Yed = (0, 10.5)
    plt.fill_between((Xst, low_criterion_ld), Yst, Yed, color=Color_L2, alpha=hist_alpha)
    plt.fill_between((high_criterion_ld, Xed), Yst, Yed, color=Color_H2, alpha=hist_alpha)
    ax.set_xlim(Xst, Xed)
    ax.set_ylim(Yst, Yed)
    plt.xlabel('LD', fontsize=LabelSize)
    plt.ylabel('Count', fontsize=LabelSize)
    plt.xticks(fontsize=TickSize)
    plt.yticks(fontsize=TickSize)
    # plt.title("hist_max_ratio", fontsize=TitleSize)
    plt.savefig('hist_max_ratio.png', dpi=300)
    # plt.savefig('hist_max_ratio.pdf', dpi=300)

    counts, bins = np.histogram(avg_T0, bins=50)
    x = 0.5 * (bins[1:] + bins[:-1])
    dx = bins[1] - bins[0]
    fig = plt.figure(figsize=hist_figsize)
    ax = fig.add_axes(hist_axissize)
    plt.hist(x, bins, weights=counts, ec="black", fc="xkcd:light grey", alpha=0.7)
    Xst, Xed = (x[0] - dx, x[-1] + dx)
    # Yst, Yed = ax.get_ylim()
    Yst, Yed = (0, 10.5)
    plt.fill_between((Xst, low_criterion_T0), Yst, Yed, color=Color_L2, alpha=hist_alpha)
    plt.fill_between((high_criterion_T0, Xed), Yst, Yed, color=Color_H2, alpha=hist_alpha)
    ax.set_xlim(Xst, Xed)
    ax.set_ylim(Yst, Yed)
    plt.xlabel(r'T$_0$ (%)', fontsize=LabelSize)
    plt.ylabel('Count', fontsize=LabelSize)
    plt.xticks(fontsize=TickSize)
    plt.yticks(fontsize=TickSize)
    # plt.title("hist_avg_T0", fontsize=TitleSize)
    plt.savefig('hist_avg_T0.png', dpi=300)
    # plt.savefig('hist_avg_T0.pdf', dpi=300)

    ### Hierarchy clustering

    scale_MaxRatio = np.max(MaxRatio)
    scale_T0 = 100
    print(f'scale: MaxRatio: {scale_MaxRatio} T0: {scale_T0}')
    inputs = np.array([MaxRatio / scale_MaxRatio, avg_T0 / scale_T0]).T
    print('DataSize:', inputs.shape)

    label2color = {'L': Color_L2, 'H': Color_H2, '': 'xkcd:pale grey'}
    with uv.timer('Run hierarchy'):
        Z = linkage(inputs, 'ward')
        Z_thres = 0.4 * np.max(Z[:, 2])
        fig = plt.figure(figsize=(10.8, 6.0))
        ax = fig.add_axes((0.05, 0.11, 0.9, 0.87))
        # f=fcluster(Z,t=n_clusters,criterion='distance')
        f = fcluster(Z, t=Z_thres, criterion='distance')
        # dn = dendrogram(Z, labels=Names)
        dn = dendrogram(Z, color_threshold=Z_thres, labels=[rc2label[_] for _ in real_class], ax=ax)
        locs, labels = plt.xticks(fontsize=12, rotation=0)
        plt.savefig("scipy_hierarchy.png", dpi=300)
        # plt.savefig("scipy_hierarchy.pdf", dpi=300)
        for i in set(f):
            print(f'Class {i}:')
            idx = np.where(np.array(f) == i)[0]
            for j in idx:
                print(f'{real_class[j]} {Ratio_class[j]} {MaxRatio[j]:8.2f} {T0_class[j]} {avg_T0[j]:8.2f} {Names[j]}')
        hl_x, hl_c = [], []
        for loc, lab in zip(locs, labels):
            lab_key = lab._text
            if lab_key:
                hl_x.append(loc)
                hl_c.append(label2color[lab_key])
        fig = plt.figure(figsize=(10.8, 2.4))
        ax = fig.add_axes((0.05, 0.11, 0.9, 0.87))
        plt.scatter(hl_x, np.ones_like(hl_x), c=hl_c, s=20)
        plt.xlim(min(locs) - 5, max(locs) + 5)
        plt.savefig("scipy_hierarchy_colordots.png", dpi=300)
        # plt.savefig("scipy_hierarchy_colordots.pdf", dpi=300)

    ### Calculate good & bad probability

    f_good, f_bad = [], []
    idx_f_good, idx_f_bad = [], []
    for i in set(f):
        idx = np.where(np.array(f) == i)[0]
        my_rc = set(real_class[idx].tolist())
        if 2 in my_rc and 0 not in my_rc:
            f_good.append(i)
            idx_f_good.extend(idx.tolist())
        elif 0 in my_rc and 2 not in my_rc:
            f_bad.append(i)
            idx_f_bad.extend(idx.tolist())
    good_f = ','.join(['%d' % _ for _ in f_good])
    bad_f = ','.join(['%d' % _ for _ in f_bad])

    count_f_good = count_num(ExpParas[idx_f_good])
    count_f_bad = count_num(ExpParas[idx_f_bad])
    idx = np.where(np.array(real_class) == 2)[0]
    count_rc2 = count_num(ExpParas[idx])
    idx = np.where(np.array(real_class) == 0)[0]
    count_rc0 = count_num(ExpParas[idx])
    count_all = count_num(ExpParas)

    FigSize = (9.6, 7.2)
    AxisSize = (0.10, 0.18, 0.88, 0.80)
    TickSize = 18
    LabelSize = 28
    TitleSize = 16
    LegendSize = 16
    bar_alpha = 0.7
    edge_width = 1
    cname2title = {
        'Color': 'Dye',
        'Thickness': r'Thickness ($\mu$m)',
        'Thick': r'Thickness ($\mu$m)',
        'Strain': 'Strain (%)',
        'DyeTime': 'Dyeing Time (s)',
        'NormStrain': 'NormStrain',
    }
    TickSize = 14
    LabelSize = 20
    my_figsizes = {
        'Color': (14.4, 4.2),
        'Thickness': (3.8, 4.2),
        'Thick': (3.8, 4.2),
        'Strain': (10.0, 4.2),
        'DyeTime': (8.6, 4.2),
        'NormStrain': (10.0, 4.2),
    }
    my_axissizes = {
        'Color': (0.08, 0.16, 0.90, 0.80),
        'Thickness': (0.28, 0.16, 0.68, 0.80),
        'Thick': (0.28, 0.16, 0.68, 0.80),
        'Strain': (0.08, 0.16, 0.90, 0.80),
        'DyeTime': (0.10, 0.16, 0.88, 0.80),
        'NormStrain': (0.08, 0.16, 0.90, 0.80),
    }
    print('Probability:')
    col_names = ['Color', 'Thickness', 'Strain', 'DyeTime']
    for i, cname in enumerate(col_names):
        print(cname)
        col_count_all = count_all[i]
        col_count_f_good = count_f_good[i]
        col_count_f_bad = count_f_bad[i]
        col_count_rc2 = count_rc2[i]
        col_count_rc0 = count_rc0[i]
        col_p = []
        for k, v in col_count_all.items():
            col_p.append((k, col_count_f_good.get(k, 0) / v, col_count_f_bad.get(k, 0) / v, col_count_rc2.get(k, 0) / v, col_count_rc0.get(k, 0) / v))
        col_p.sort(key=lambda x: x[1], reverse=True)
        if cname == 'Color':
            print('%16s%6s%6s%6s%6s' % ('value', f'f={good_f}', f'f={bad_f}', 'rc=2', 'rc=0'))
            for k, p1, p2, p3, p4 in col_p:
                print(f'{Color_list[int(k)]:16s}{p1:6.2f}{p2:6.2f}{p3:6.2f}{p4:6.2f}')
        else:
            print('%6s%6s%6s%6s%6s' % ('value', f'f={good_f}', f'f={bad_f}', 'rc=2', 'rc=0'))
            for k, p1, p2, p3, p4 in col_p:
                print(f'{k:6.2f}{p1:6.2f}{p2:6.2f}{p3:6.2f}{p4:6.2f}')
        # For plot
        col_p.sort()
        x = np.arange(len(col_p))
        labels = []
        ps = []
        for kp in col_p:
            labels.append(f'{int(kp[0])}' if cname != 'Color' else '')
            ps.append(kp[1:])
        ps = np.array(ps) * 100


        fig = plt.figure(figsize=my_figsizes[cname])
        ax = fig.add_axes(my_axissizes[cname])
        width = 0.3 if cname != 'Thickness' else 0.2
        # width = 0.3
        ax.bar(x - 0.5 * width, ps[:, 0], width, color=Color_H2, alpha=bar_alpha, label='High Ld & T')
        ax.bar(x + 0.5 * width, ps[:, 1], width, color=Color_L2, alpha=bar_alpha, label='Low Ld & T')
        ax.set_ylabel('Probability (%)', fontsize=LabelSize)
        ax.set_xlabel(cname2title[cname], fontsize=LabelSize)
        ax.set_xticks(x)
        if cname == 'Color':
            # ax.set_xticklabels(Color_list, rotation=70, fontsize=TickSize)
            ax.set_xticklabels(Color_names, fontsize=TickSize)
        else:
            ax.set_xticklabels(labels, fontsize=TickSize)
        plt.yticks(fontsize=TickSize)
        plt.ylim(0, 100)
        plt.xlim(np.min(x) - 1.5 * width, np.max(x) + 1.5 * width)
        # ax.legend(fontsize=LegendSize)
        ax.spines['top'].set_linewidth(edge_width)
        ax.spines['bottom'].set_linewidth(edge_width)
        ax.spines['left'].set_linewidth(edge_width)
        ax.spines['right'].set_linewidth(edge_width)
        plt.savefig(f'Prob2_{cname}.png', dpi=120)
        # plt.savefig(f'Prob2_{cname}.pdf', dpi=120)
        plt.close()
        # exit()

    X_tsne = inputs
    
    s_size = 30
    unq_y = np.unique(f)
    ColorMap2 = ListedColormap(uv.ColorList[:len(unq_y)])
    plt.figure(figsize=(14.4, 9.6))
    ax1 = plt.subplot(121)
    scatter1 = ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], c=f, s=s_size, cmap=ColorMap2)
    legend1 = ax1.legend(*scatter1.legend_elements(), title="F classes")
    ax1.add_artist(legend1)
    ax2 = plt.subplot(122)
    scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=real_class, s=s_size, cmap=ColorMap)
    legend2 = ax2.legend(*scatter2.legend_elements(), title="Real classes")
    ax2.add_artist(legend2)
    plt.suptitle('hierarchy')
    plt.savefig('points_hierarchy.png', dpi=120)
    # exit()

    ### Select good parameters (color, thickness, strain, dyetime) for 2nd round screening

    good_colors = ('direct yellow', 'methylorange', 'congored', 'red', 'purple', 'tai', 'blue2', 'blue6', 'blue71', 'green')

    all_color2fine = {}
    fine_idx = []
    norm_strain = np.copy(ExpParas[:, -2])
    fine_colors = []
    fine_color_names = []
    for i in range(NColors):
        idx = np.where((ExpParas[:, 0] == i))[0]
        max_strain = np.max(norm_strain[idx])
        norm_strain[idx] /= max_strain
        if Color_list[i] not in good_colors:
            continue
        idx = np.where((ExpParas[:, 0] == i) & (ExpParas[:, -3] == 80) & (ExpParas[:, -2] >110)& (ExpParas[:, -2] <550))[0]
        fine_idx.extend(idx.tolist())
        all_color2fine[i] = len(all_color2fine)
        fine_colors.append(Color_list[i])
        fine_color_names.append(Color_names[i])
    print(f'fine: {len(fine_idx)}/{len(ExpParas)}')

    fine_idx.sort()
    fine_paras = ExpParas[fine_idx]
    fine_ld = MaxRatio[fine_idx]
    fine_T0 = avg_T0[fine_idx]
    fine_paras[:, 0] = [all_color2fine[_] for _ in fine_paras[:, 0]]
    fine_norm_strain = norm_strain[fine_idx]

    ### 2nd round hierarchy clustering

    FigSize = (9.6, 7.2)
    AxisSize = (0.14, 0.12, 0.83, 0.83)
    ColorMap = ListedColormap(uv.ColorList)

    with uv.timer('Run hierarchy'):
        Z = linkage(fine_ld.reshape((-1, 1)), 'ward')
        fig = plt.figure(figsize=(25, 5))
        # f=fcluster(Z,t=n_clusters,criterion='distance')
        f = fcluster(Z, t=1.0, criterion='distance')
        # dn = dendrogram(Z, labels=Names)
        # dn = dendrogram(Z, color_threshold=1.0, labels=['%.2f' % _ for _ in fine_ld])
        dn = dendrogram(Z, color_threshold=1.0, labels=[Names[_] for _ in fine_idx])
        plt.savefig("fine_hierarchy.png", dpi=300)
        for i in set(f):
            print(f'Class {i}:')
            idx = np.where(np.array(f) == i)[0]
            my_ld = fine_ld[idx]
            print(f'avg/min/max: {np.mean(my_ld):8.2f} {np.min(my_ld):8.2f} {np.max(my_ld):8.2f}')

    sort_idx = np.argsort(fine_ld)[::-1]
    print_info = np.hstack([fine_paras, fine_norm_strain.reshape((-1, 1)), fine_ld.reshape((-1, 1))])[sort_idx]
    print('%6s%6s%6s%6s%8s%8s' % ('Color', 'Thick', 'Strain', 'DyeTime', 'NormStr', 'MaxRatio'))
    for i in set(fine_paras[:, 0].tolist()):
        idx = np.where(print_info[:, 0] == i)[0]
        for info in print_info[idx, :]:
            print('%6d%6d%6d%6d%8.2f%8.2f' % (info[0], *info[1:]))

    ### 2nd round probability calculation

    low_criterion = np.quantile(fine_ld, 0.1)
    high_criterion = np.quantile(fine_ld, 0.8)
    print(f'fine_ld: low: {low_criterion:.3f}  high: {high_criterion:.3f}')
    
    low_criterion = min(low_criterion_ld * 0.8, np.quantile(fine_ld, 0.1))
    high_criterion = min(high_criterion_ld * 1.1, np.quantile(fine_ld, 0.8))
    Ratio_class = np.ones_like(fine_ld, dtype=int)
    Ratio_class[fine_ld < low_criterion] = 0
    Ratio_class[fine_ld >= high_criterion] = 2
    print(f'fine_ld: low: {low_criterion:.3f}  high: {high_criterion:.3f}')

    low_criterion = np.quantile(fine_T0, 0.1)
    high_criterion = np.quantile(fine_T0, 0.8)
    print(f'fine_T0: low: {low_criterion:.3f}  high: {high_criterion:.3f}')
    
    low_criterion = min(low_criterion_T0 * 0.8, np.quantile(fine_T0, 0.1))
    high_criterion = min(high_criterion_T0 * 1.1, np.quantile(fine_T0, 0.8))
    T0_class = np.ones_like(fine_T0, dtype=int)
    T0_class[fine_T0 < low_criterion] = 0
    T0_class[fine_T0 >= high_criterion] = 2
    print(f'fine_T0: low: {low_criterion:.3f}  high: {high_criterion:.3f}')

    fine_class = np.ones_like(fine_ld, dtype=int)
    fine_class[(T0_class == 0) | (Ratio_class == 0)] = 0
    fine_class[(T0_class == 2) & (Ratio_class == 2)] = 2
    idx_f_high = np.where(fine_class == 2)[0]
    idx_f_low = np.where(fine_class == 0)[0]
    good_f = 'high'
    bad_f = '1low'

    interest_infos = np.hstack([fine_paras[:, [0, 1, 2, 3]], (10 * fine_norm_strain).astype(int).reshape((-1, 1))])
    count_f_high = count_num(interest_infos[idx_f_high])
    count_f_low = count_num(interest_infos[idx_f_low])
    count_fine = count_num(interest_infos)

    s_size = 30
    unq_y = np.unique(f)
    ColorMap2 = ListedColormap(uv.ColorList[:len(unq_y)])
    ColorMap = ListedColormap(uv.ColorList[:3])
    plt.figure(figsize=(14.4, 9.6))
    ax1 = plt.subplot(121)
    scatter1 = ax1.scatter(fine_ld, fine_T0, c=f, s=s_size, cmap=ColorMap2)
    legend1 = ax1.legend(*scatter1.legend_elements(), title="F classes")
    ax1.add_artist(legend1)
    ax2 = plt.subplot(122)
    scatter2 = ax2.scatter(fine_ld, fine_T0, c=fine_class, s=s_size, cmap=ColorMap)
    legend2 = ax2.legend(*scatter2.legend_elements(), title="Real classes")
    ax2.add_artist(legend2)
    plt.suptitle('fine hierarchy')
    plt.savefig('fine_points_hierarchy.png', dpi=120)

    # FigSize = (9.6, 7.2)
    # AxisSize = (0.10, 0.18, 0.88, 0.80)
    # TickSize = 18
    # LabelSize = 24
    # TitleSize = 16
    # LegendSize = 16
    # bar_alpha = 0.7
    col_names = ['Color', 'Thickness', 'Strain', 'DyeTime', 'NormStrain']
    print('Probability:')
    for i, cname in enumerate(col_names):
        print(cname)
        col_count_fine = count_fine[i]
        col_count_f_high = count_f_high[i]
        col_count_f_low = count_f_low[i]
        col_p = []
        for k, v in col_count_fine.items():
            col_p.append((k, col_count_f_high.get(k, 0) / v, col_count_f_low.get(k, 0) / v))
        col_p.sort(key=lambda x: x[1], reverse=True)
        if cname == 'Color':
            print('%16s%6s%6s' % ('value', f'f={good_f}', f'f={bad_f}'))
            for k, p1, p2 in col_p:
                print(f'{fine_colors[int(k)]:16s}{p1:6.2f}{p2:6.2f}')
        else:
            print('%6s%6s%6s' % ('value', f'f={good_f}', f'f={bad_f}'))
            for k, p1, p2 in col_p:
                print(f'{k:6.2f}{p1:6.2f}{p2:6.2f}')
        # For plot
        col_p.sort()
        x = np.arange(len(col_p))
        labels = []
        ps = []
        for kp in col_p:
            labels.append(f'{int(kp[0])}' if cname != 'Color' else '')
            ps.append(kp[1:])
        ps = np.array(ps) * 100

        fig = plt.figure(figsize=my_figsizes[cname])
        ax = fig.add_axes(my_axissizes[cname])
        width = 0.3
        ax.bar(x - 0.5 * width, ps[:, 0], width, color=Color_H2, alpha=bar_alpha, label='High Ld')
        ax.bar(x + 0.5 * width, ps[:, 1], width, color=Color_L1, alpha=bar_alpha, label='Low Ld')
        ax.set_ylabel('Probability (%)', fontsize=LabelSize)
        ax.set_xlabel(cname2title[cname], fontsize=LabelSize)
        ax.set_xticks(x)
        if cname == 'Color':
            # ax.set_xticklabels(fine_colors, rotation=70, fontsize=TickSize)
            ax.set_xticklabels(fine_color_names, fontsize=TickSize)
        else:
            ax.set_xticklabels(labels, fontsize=TickSize)
        plt.yticks(fontsize=TickSize)
        plt.ylim(0, 100)
        plt.xlim(np.min(x) - 1.5 * width, np.max(x) + 1.5 * width)
        # ax.legend(fontsize=LegendSize)
        ax.spines['top'].set_linewidth(edge_width)
        ax.spines['bottom'].set_linewidth(edge_width)
        ax.spines['left'].set_linewidth(edge_width)
        ax.spines['right'].set_linewidth(edge_width)
        plt.savefig(f'FineProb2_{cname}.png', dpi=120)
        # plt.savefig(f'FineProb2_{cname}.pdf', dpi=120)
        plt.close()

    exit()

