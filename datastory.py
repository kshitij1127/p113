import statistics
import pandas as pd
import plotly.express as px 
import csv 
import plotly.graph_objects as go 
import numpy as np 
import seaborn as sns
import plotly.figure_factory as ff
import random 

# initialising the csv
df = pd.read_csv("savings_data.csv")

# displaying a scatter plot to find the nature of the data 
fig = px.scatter(df, y="quant_saved", color="female")
# fig.show()

# finding mean, median and mode of the big data 
avg = statistics.mean(df["quant_saved"])
med = statistics.median(df["quant_saved"])
mod = statistics.mode(df['quant_saved'])
print(avg, med, mod)
# stating the findings in the data 
# mean median and mode are too far apart while mode is zero 

# finding if the people who saved are wealthy or not 
with open("savings_data.csv", newline="") as f:
    reader = csv.reader(f)
    savings_data = list(reader)

savings_data.pop(0)

total_entries = len(savings_data)
total_people_wealthy = 0 
male_saved = 0 

wealthy = []
not_wealthy = []
male = []
female = []

for data in savings_data:
    if int(data[3]) == 1:
        total_people_wealthy += 1
        wealthy.append(float(data[0]))
    else:
        not_wealthy.append(float(data[0]))

    if int(data[1]) == 0:
        male_saved += 1
        male.append(float(data[0]))
    else:
        female.append(float(data[0]))

print(male_saved)
print(total_people_wealthy)
print(f"wealthy people savings mean: {statistics.mean(wealthy)}")
print(f"wealthy people savings median: {statistics.median(wealthy)}")
print(f"wealthy people savings mode: {statistics.mode(wealthy)}")

print("\n\n")
print(f"not wealthy people savings mean: {statistics.mean(not_wealthy)}")
print(f"not wealthy people savings median: {statistics.median(not_wealthy)}")
print(f"not wealthy people savings mode: {statistics.mode(not_wealthy)}")

print("\n\n")
print(f"male people savings mean: {statistics.mean(male)}")
print(f"male people savings median: {statistics.median(male)}")
print(f"male people savings mode: {statistics.mode(male)}")

print("\n\n")

print(f"female people savings mean: {statistics.mean(female)}")
print(f"female people savings median: {statistics.median(female)}")
print(f"female people savings mode: {statistics.mode(female)}")

# state findings 
# the mean median and mode are still very far apart from each other 

figure = go.Figure(go.Bar(x = ["wealthy", "not wealthy"], y = [total_people_wealthy, (total_entries - total_people_wealthy)]))
# figure.show()

malefig = go.Figure(go.Bar(x = ["male, female"], y = [male_saved, (total_entries - male_saved)]))
# malefig.show()

# find the standard deviation 
standard_deviation = statistics.stdev(df["quant_saved"])
print( "standard deviation of population data: " ,standard_deviation)

# finding correlation between weatlhy and quantity saved

quant_saved = [] 
wealth = []
for data in savings_data:
    if data[3] != 0:
        wealth.append(float(data[3]))
        quant_saved.append(float(data[0]))

correlation = np.corrcoef(wealth, quant_saved)
print(f"correlation between age of a person and quantity saved is: {correlation[0, 1]}")


# finding IQR

sns.boxplot(data=df, x = df['quant_saved'])

q1 = df['quant_saved'].quantile(0.25)
q3 = df["quant_saved"].quantile(0.75)

iqr = q3 - q1
print(f"q1 = {q1}")
print(f"q3 = {q3}")
print(f"iqr = {iqr}")

lower_whisker = q1 - 1.5 * iqr
upper_whisker = q3 + 1.5 * iqr

print(f"lower whisker = {lower_whisker}")
print(f"upper whisker = {upper_whisker}")

new_df = df[df["quant_saved"] < upper_whisker]

new_savings = new_df['quant_saved'].tolist()

print(f"mean of new savings - {statistics.mean(new_savings)}")
print(f"median of new savings - {statistics.median(new_savings)}")
print(f"mode of savings - {statistics.mode(new_savings)}")

fij = ff.create_distplot([new_savings], ["new savings"], show_hist=False)
# fij.show()

print(f"standard deviation of new savings = {statistics.stdev(new_savings)}")

sampling_meanlist = []

for data in range(1000):
    templist = []
    for i in range(100):
        templist.append(random.choice(new_savings))
    sampling_meanlist.append(templist)

meansampling = statistics.mean(sampling_meanlist)
figu = ff.create_distplot([sampling_meanlist], ["savings for the sampling"], show_hist=False)
figu.add_trace(go.Scatter(x = [meansampling, meansampling], y = [0, 0.17], mode="lines", name="mean"))
# figu.show()

print(f"standard deviation of sample data = {statistics.stdev(sampling_meanlist)}")
print(f"mean of population = {statistics.mean(new_savings)}")
print(f"sampling mean = {meansampling}")

wealthydf = new_df.loc[new_df['wealthy'] == 1]
notwealthydf = new_df.loc[new_df['wealthy'] == 0]

print(wealthydf.head())
print(notwealthydf.head())

notwealthysavings = notwealthydf['wealthy'].tolist()

sm = []

for i in range(1000):
    newlist = []
    for j in range(100):
        newlist.append(random.choice(notwealthysavings))

    sm.append(newlist)

mean_sampling_new_savings = statistics.mean(sm)
stdev_sampling_new_savings = statistics.stdev(sm)

print(mean_sampling_new_savings)
print(stdev_sampling_new_savings)

not_wealthy_savings_figure = ff.create_distplot([sm], ['savings for sampling'], show_hist=False)
not_wealthy_savings_figure.show()