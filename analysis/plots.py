# %%
import numpy
from matplotlib import pyplot as plt
import glob
import os
import pandas

%matplotlib inline

# %%
folder = "/Users/corey.adams/data/cosmic_tagging_downsample/"
files = folder + "*.npy"
files =glob.glob(files)
print(files)

# %%
def get_flat_dtype():
    flat_dtypes = numpy.dtype([
        ('name'       , 'S32'),
        ('entry'      , numpy.uint32),
        ('neut'       , numpy.uint32),
        ('n_neut_true', numpy.uint32),
        ('n_neut_true_0', numpy.uint32),
        ('n_neut_true_1', numpy.uint32),
        ('n_neut_true_2', numpy.uint32),
        ('n_neut_pred', numpy.uint32),
        ('n_neut_pred_0', numpy.uint32),
        ('n_neut_pred_1', numpy.uint32),
        ('n_neut_pred_2', numpy.uint32),
        ('neut_x_mean', numpy.float32),
        ('neut_x_mean_0', numpy.float32),
        ('neut_x_mean_1', numpy.float32),
        ('neut_x_mean_2', numpy.float32),
        ('neut_y_mean', numpy.float32),
        ('neut_y_mean_0', numpy.float32),
        ('neut_y_mean_1', numpy.float32),
        ('neut_y_mean_2', numpy.float32),
        ('neut_x_std' , numpy.float32),
        ('neut_x_std_0' , numpy.float32),
        ('neut_x_std_1' , numpy.float32),
        ('neut_x_std_2' , numpy.float32),
        ('neut_y_std' , numpy.float32),
        ('neut_y_std_0', numpy.float32),
        ('neut_y_std_1', numpy.float32),
        ('neut_y_std_2', numpy.float32),
        ('accuracy'   , numpy.float32),
        ('accuracy_0'   , numpy.float32),
        ('accuracy_1'   , numpy.float32),
        ('accuracy_2'   , numpy.float32),
        ('acc_neut'   , numpy.float32),
        ('acc_neut_0'   , numpy.float32),
        ('acc_neut_1'   , numpy.float32),
        ('acc_neut_2'   , numpy.float32),
        ('acc_cosm'   , numpy.float32),
        ('acc_cosm_0'   , numpy.float32),
        ('acc_cosm_1'   , numpy.float32),
        ('acc_cosm_2'   , numpy.float32),
        ('iou_neut'   , numpy.float32),
        ('iou_neut_0'   , numpy.float32),
        ('iou_neut_1'   , numpy.float32),
        ('iou_neut_2'   , numpy.float32),
        ('iou_cosm'   , numpy.float32),
        ('iou_cosm_0'   , numpy.float32),
        ('iou_cosm_1'   , numpy.float32),
        ('iou_cosm_2'   , numpy.float32),
        ('acc_non0'   , numpy.float32),
        ('acc_non0_0'   , numpy.float32),
        ('acc_non0_1'   , numpy.float32),
        ('acc_non0_2'   , numpy.float32),
        ('energy'     , numpy.float32),
    ])

    return flat_dtypes

# %%
def convert_array(input_arr):
    # print(input_arr.dtype)

    out_dtype = get_flat_dtype()
    new_data = numpy.ndarray(shape=input_arr.shape, dtype=out_dtype)
    # print(out_dtype)
    for in_type in input_arr.dtype.fields:
        # print(in_type)
        # print(input_arr.dtype.fields[in_type])
        this_fields = input_arr.dtype.fields[in_type]
        if this_fields[0].shape == (3,):
            # print("Vector")
            new_data[in_type] = numpy.mean(input_arr[in_type], axis=-1)
            for i in [0,1,2]:
                new_data[in_type+f"_{i}"] = input_arr[in_type][:,i]
        else:
            # print("scalar")
            new_data[in_type] = input_arr[in_type]

    return new_data

# %%
df_list = []
names = []
for i, _f in enumerate(files):
    data = numpy.load(files[i])
    name = os.path.basename(files[i]).replace("cosmic_tagging_downsample_test_sparse_output_","").replace(".npy","")
    name = name.replace("_2", "")
    new_data = convert_array(data)
    new_data['name'] = name
    names.append(name)
    df_list.append(pandas.DataFrame(new_data))
    if name == 'biggerbatch':
        df_real = df_list[-1]
        break

df = pandas.concat(df_list)
df['name'] = df['name'].str.decode("utf-8")


# %%
df_real.info()

#%%

df_real.query('n_neut_true == 0')

# %%

numpy.max(df_real['neut_x_mean_2'])


# %%
# df = df.query('n_neut_pred > 150')
def plot_ious(_df, filename):
    fig = plt.figure(figsize=(16,9))
    ebins = numpy.arange(0.0, 4.01, 0.25)
    print(ebins)
    neuts = []
    labels = []
    for neut in [0,1,2]:

        if neut == 0:
            label=r"$\nu_e$ CC"
            color='blue'
        elif neut == 1:
            label=r"$\nu_{\mu}$ CC"
            color='black'
        else:
            label="NC"
            color='green'

        sub_df = _df.query(f'neut == {neut}')
        e = sub_df['energy']
        acc = sub_df['iou_neut']
        y_iou_neut = []
        y_iou_cosm = []
        y_acc_non0 = []
        for i_e in range(len(ebins)-1):
            locs_low = e > ebins[i_e]
            locs_high = e < ebins[i_e + 1]
            locs= numpy.logical_and(locs_low, locs_high)
            y_iou_neut.append(numpy.mean(sub_df['iou_neut'][locs]))
            y_iou_cosm.append(numpy.mean(sub_df['iou_cosm'][locs]))
            y_acc_non0.append(numpy.mean(sub_df['acc_non0'][locs]))
        l1, = plt.plot(ebins[:-1], y_iou_neut, color, lw=3,ls="-")
        l2, = plt.plot(ebins[:-1], y_iou_cosm, color, lw=3,ls="--")
        # plt.plot(ebins[:-1], y_acc_non0, color, lw=3,ls=":")

        neuts.append(l1)
        labels.append(label)

        if neut == 0:
            legend = plt.legend([l1, l2], ["Neutrino", "Cosmic"], loc=3, fontsize=20)
    # legend = plt.legend([l1, l2], ["Neutrino", "Cosmic"], loc=1, fontsize=20)

    legend2 = plt.legend(neuts, labels, loc=4, fontsize=20 )
    plt.gca().add_artist(legend)
    plt.ylim([0.0,1.0])
    plt.grid(True)
    # plt.legend(fontsize=20)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    plt.xlabel("Neutrino Energy [GeV]", fontsize = 25)
    plt.ylabel("Intersection / Union", fontsize = 25 )
    plt.savefig(f"/Users/corey.adams/Desktop/CosmicTaggingWriteUp/analysis/{filename}.pdf")
    plt.show()

#%%
plot_ious(df_real, "biggerbatch_iou")
#%%
df_selected = select(df_real, n_cut=100, match_criteria=50)
# hit_cut=100
# match_criteria=50
plot_ious(df_selected, "biggerbatch_iou")

#%%

# Make a selection across events:


df_real['match_0_1'] = abs(df_real['neut_x_mean_0'] - df_real['neut_x_mean_1'])
df_real['match_1_2'] = abs(df_real['neut_x_mean_1'] - df_real['neut_x_mean_2'])
df_real['match_2_1'] = abs(df_real['neut_x_mean_2'] - df_real['neut_x_mean_0'])


# df_real['match_0_1'] = abs(df_real['neut_x_mean_0'] - df_real['neut_x_mean_1']) / (df_real[['neut_x_std_0', 'neut_x_std_1']].max(axis=1) + 0.1)
# df_real['match_1_2'] = abs(df_real['neut_x_mean_1'] - df_real['neut_x_mean_2']) / (df_real[['neut_x_std_2', 'neut_x_std_1']].max(axis=1) + 0.1)
# df_real['match_2_1'] = abs(df_real['neut_x_mean_2'] - df_real['neut_x_mean_0']) / (df_real[['neut_x_std_0', 'neut_x_std_2']].max(axis=1) + 0.1)

# df_real.query("n_neut_true == 0")['match_1_2']


#%%

df_nue = df_real.query('neut==0 & n_neut_true > 0')
df_numu = df_real.query('neut==1 & n_neut_true > 0')
df_nc = df_real.query('neut==2 & n_neut_true > 0')
df_cosmic = df_real.query('n_neut_true==0')

# %%

len(df_cosmic.index)


def select(df, n_cut, match_criteria):
    sub_df =  df.query(f"n_neut_pred_0 > {n_cut} & n_neut_pred_1 > {n_cut} & n_neut_pred_2 > {n_cut}")
    sub_df = sub_df.query(f"match_0_1 <  {match_criteria} & match_1_2 < {match_criteria} & match_2_1 < {match_criteria}")
    return sub_df

def efficiency(df, n_cut, match_criteria):
    n_start = len(df.index)
    selected = select(df, n_cut, match_criteria)
    return len(selected.index) / n_start

def fom(df_signal, df_background, n_cut, match_criteria):
    return efficiency(df_signal, n_cut, match_criteria) / numpy.sqrt(efficiency(df_background, n_cut, match_criteria))
#%%


hit_cut=100
match_criteria=50
print(efficiency(df_nue, hit_cut, match_criteria))
print(efficiency(df_numu, hit_cut, match_criteria))
print(efficiency(df_nc, hit_cut, match_criteria))
print(efficiency(df_cosmic, hit_cut, match_criteria))

#%%

# eff_cosmic = []
hit_cut=50
match_steps = numpy.arange(0,100,1)
eff_nue = [fom(df_nue, df_cosmic, hit_cut, i) for i in match_steps]
# eff_cosmic = [fom(, hit_cut, i) for i in steps]
plt.plot(match_steps, eff_nue)
# plt.plot(steps, eff_cosmic)
#%%
match_criteria= 50
hits = numpy.arange(0,500,10)
eff_nue = [fom(df_numu, df_cosmic, i, match_criteria) for i in hits]

plt.plot(hits, eff_nue)

#%%

df_nue['match_0_1'].hist(bins=numpy.arange(0,2,0.25))
df_cosmic['match_0_1'].hist(bins=numpy.arange(0,2,0.25))
