import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns


sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8

columns = ['user','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
df = pd.read_csv('data/WISDM_ar_v1.1_raw.txt', header = None, names = columns)
df = df.dropna()

#df['activity'].value_counts().plot(kind='bar', title='activity')
#plt.xlabel("Activity", labelpad=14)
#plt.ylabel("Count of Data", labelpad=14)
#df['user'].value_counts().plot(kind='bar', title='user')
#plt.xlabel("user", labelpad=14)
#plt.ylabel("Count of Data", labelpad=14)
#plt.show()


def plot_activity(activity, df):
    data = df[df['activity'] == activity][['x-axis', 'y-axis', 'z-axis']][:200]
    axis = data.plot(subplots=True, figsize=(16, 12), title=activity)
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))


#plot_activity("Sitting", df)
#plot_activity("Standing", df)
#plot_activity("Walking", df)
#plot_activity("Jogging", df)
#plot_activity("Upstairs", df)
#plot_activity("Downstairs", df)
plt.show()

