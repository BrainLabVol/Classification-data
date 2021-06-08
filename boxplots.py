import pandas as pd
import matplotlib.pyplot as plt
df0_fd = pd.read_csv("/home/stylianos/eclipse-workspace/thesis/final_datasets/cdr1_vs_cdr0/cdr0_fd.csv")
df1_fd = pd.read_csv("/home/stylianos/eclipse-workspace/thesis/final_datasets/cdr1_vs_cdr0/cdr1_fd.csv")
df0_vol = pd.read_csv("/home/stylianos/eclipse-workspace/thesis/final_datasets/cdr1_vs_cdr0/cdr0_volume.csv")
df1_vol = pd.read_csv("/home/stylianos/eclipse-workspace/thesis/final_datasets/cdr1_vs_cdr0/cdr1_volume.csv")


my_dict = {'0': df0_fd['Right_Putamen'],'1 or greater': df1_fd['Right_Putamen']}

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Right Putamen')
ax1.boxplot(my_dict.values())
ax1.set_xticklabels(my_dict.keys())
ax1.set_ylabel('Fractal Dimension')

my_dict = {'0': df0_vol['Right_Putamen'],'1 or greater': df1_vol['Right_Putamen']}

ax2.boxplot(my_dict.values())
ax2.set_xticklabels(my_dict.keys())
ax2.set_xlabel('CDR')
ax2.set_ylabel('Volume (mm^3)')
plt.show()