#%%

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


#%%
def construct_dataframe(filename):

    event_acc = EventAccumulator(filename)
    event_acc.Reload()

    values = []

    w_times, step_nums, total_accuracy = zip(*event_acc.Scalars('accuracy/All_Plane_Total_Accuracy'))
    _, _, acc_non_bkg  = zip(*event_acc.Scalars('accuracy/All_Plane_Non_Background_Accuracy'))
    _, _, neutrino_iou = zip(*event_acc.Scalars('accuracy/All_Plane_Neutrino_IoU'))
    _, _, cosmic_iou   = zip(*event_acc.Scalars('accuracy/All_Plane_Cosmic_IoU'))
    _, _, total_loss   = zip(*event_acc.Scalars('cross_entropy/Total_Loss'))

    df = pd.DataFrame(columns=['step',
                               'wall_time',
                               'loss',
                               'total_accuracy', 'acc_non_bkg', 'neutrino_iou', 'cosmic_iou'],
                      data = np.stack([step_nums,
                              w_times,
                              total_loss,
                              total_accuracy,
                              acc_non_bkg,
                              neutrino_iou,
                              cosmic_iou,
                              ], axis=-1),
                      index=np.arange(len(step_nums))
                      )

    return df


#%%

def construct_dataframe_dict():

    df_train = {}
    df_test = {}


    runs = glob.glob("/Users/corey.adams/DeepLearnPhysics/CosmicTagger/runs/*")
    print(runs[0])
    for run in runs:
        name = os.path.basename(run)
        df_train[name] = construct_dataframe(run+'/train/')
        df_test[name]  = construct_dataframe(run+'/test/')

    return df_train, df_test


 # %%

df_train, df_test =  construct_dataframe_dict()



#%%
df_train['concat']['step']


#%%

for name in df_train.keys():
    print(name)
    for metric_name in ['acc_non_bkg', 'cosmic_iou', 'neutrino_iou']:
        #Use the last 10 measurements for an average:
        steps = np.asarray(df_test[name]['step'])
        average_step = np.mean(df_test[name]['step'][-10:])
        metric_value = np.mean(df_test[name][metric_name][-10:])
        print(f"{name}, test, {metric_name} at {average_step}: {metric_value}")

#%%
fig = plt.figure(figsize=(16,9))
plt.grid(True)
steps = df_train['baseline']['step']
metric =  df_train['baseline']['loss']

smooth_metric = df_train['baseline']['loss'].rolling(35).mean()
smooth_steps  = df_train['baseline']['step'].rolling(35).mean()
smooth_test   = df_test['baseline']['loss'].rolling(35).mean()
test_steps    = df_test['baseline']['step'].rolling(35).mean()

plt.plot(steps, metric, 'lightblue', label="Loss")
plt.plot(smooth_steps, smooth_metric, label="Smoothed")
plt.plot(test_steps, smooth_test, 'black', label="Test Data")
plt.legend(fontsize=20)

plt.xlabel("Step", fontsize=25)
plt.ylabel("Loss", fontsize=25)
plt.yscale("log")
# We change the fontsize of minor ticks label
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)

plt.savefig("/Users/corey.adams/Desktop/CosmicTaggingWriteUp/training_figures/loss.pdf")
plt.savefig("/Users/corey.adams/Desktop/CosmicTaggingWriteUp/training_figures/loss.png")
plt.show()
#%%
fig = plt.figure(figsize=(16,9))
plt.grid(True)
steps = df_train['baseline']['step']
metric =  df_train['baseline']['acc_non_bkg']

smooth_metric = df_train['baseline']['acc_non_bkg'].rolling(35).mean()
smooth_steps  = df_train['baseline']['step'].rolling(35).mean()
smooth_test   = df_test['baseline']['acc_non_bkg'].rolling(35).mean()
test_steps    = df_test['baseline']['step'].rolling(35).mean()

plt.plot(steps, metric, 'lightblue', label="Accuracy on Non-Zero Pixels")
plt.plot(smooth_steps, smooth_metric, label="Smoothed")
plt.plot(test_steps, smooth_test, 'black', label="Test Data")
plt.legend(fontsize=20)

plt.xlabel("Step", fontsize=25)
plt.ylabel("Accuracy", fontsize=25)
# plt.yscale("log")
# We change the fontsize of minor ticks label
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)

plt.savefig("/Users/corey.adams/Desktop/CosmicTaggingWriteUp/training_figures/acc_non_bkg.pdf")
plt.savefig("/Users/corey.adams/Desktop/CosmicTaggingWriteUp/training_figures/acc_non_bkg.png")
plt.show()


#%%
fig = plt.figure(figsize=(16,9))
plt.grid(which='both')
steps = df_train['baseline']['step']
metric =  df_train['baseline']['cosmic_iou']

smooth_metric = df_train['baseline']['cosmic_iou'].rolling(35).mean()
smooth_steps  = df_train['baseline']['step'].rolling(35).mean()
smooth_test   = df_test['baseline']['cosmic_iou'].rolling(35).mean()
test_steps    = df_test['baseline']['step'].rolling(35).mean()

plt.plot(steps, metric, 'lightblue', label="IoU for Cosmic Pixels")
plt.plot(smooth_steps, smooth_metric, label="Smoothed")
plt.plot(test_steps, smooth_test, 'black', label="Test Data")
plt.legend(fontsize=20)

plt.xlabel("Step", fontsize=25)
plt.ylabel("Intersection / Union ", fontsize=25)
# plt.yscale("log")
# We change the fontsize of minor ticks label
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)

plt.savefig("/Users/corey.adams/Desktop/CosmicTaggingWriteUp/training_figures/cosmic_iou.pdf")
plt.savefig("/Users/corey.adams/Desktop/CosmicTaggingWriteUp/training_figures/cosmic_iou.png")
plt.show()

#%%
fig = plt.figure(figsize=(16,9))
plt.grid(which='both')
steps = df_train['baseline']['step']
metric =  df_train['baseline']['neutrino_iou']

smooth_metric = df_train['baseline']['neutrino_iou'].rolling(35).mean()
smooth_steps  = df_train['baseline']['step'].rolling(35).mean()
smooth_test   = df_test['baseline']['neutrino_iou'].rolling(35).mean()
test_steps    = df_test['baseline']['step'].rolling(35).mean()
plt.plot(steps, metric, 'lightblue', label="IoU for Neutrino Pixels")
plt.plot(smooth_steps, smooth_metric, label="Smoothed")
plt.plot(test_steps, smooth_test, 'black', label="Test Data")

plt.legend(fontsize=20)

plt.xlabel("Step", fontsize=25)
plt.ylabel("Intersection / Union ", fontsize=25)
# plt.yscale("log")
# We change the fontsize of minor ticks label
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)

plt.savefig("/Users/corey.adams/Desktop/CosmicTaggingWriteUp/training_figures/neutrino_iou.pdf")
plt.savefig("/Users/corey.adams/Desktop/CosmicTaggingWriteUp/training_figures/neutrino_iou.png")
plt.show()
