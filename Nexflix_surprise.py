import pandas as pd
import numpy as np
from surprise import Reader, Dataset
from surprise import CoClustering


def mem_usage(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:  # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


def processDataFile(data, counter):
    df_nan = pd.DataFrame(pd.isnull(data.Rating))
    df_nan = df_nan[df_nan['Rating'] == True]
    df_nan = df_nan.reset_index()
    # print df_na
    movie_np = []
    movie_id = 1
    # print df_nan['index'][1:],df_nan['index'][:-1]
    a = zip(df_nan['index'][1:], df_nan['index'][:-1])
    store_df = pd.DataFrame([])
    temp_total_array = np.zeros(((0), 1))
    for i, j in a:
        temp_np_array = np.full((i - j, 1), counter)
        temp_total_array = np.concatenate((temp_total_array, temp_np_array))
        counter = counter + 1
        # print counter
    # print temp_total_array.shape
    remaining = data.shape[0] - temp_total_array.shape[0]
    temp_np_array = np.full((remaining, 1), counter)
    temp_total_array = np.concatenate((temp_total_array, temp_np_array))
    data['movie_id'] = temp_total_array
    data = data.dropna(thresh=3)
    return (data, counter)


total_data = pd.DataFrame([])

data = pd.read_csv("./netflix-prize-data/combined_data_1.txt", names=['Cust_Id', 'Rating', 'date'])
temp_data = processDataFile(data, 1)
total_data = pd.concat([total_data, temp_data[0]], ignore_index=True)
data = pd.read_csv("./netflix-prize-data/combined_data_2.txt", names=['Cust_Id', 'Rating', 'date'])
temp_data = processDataFile(data, temp_data[1])
total_data = pd.concat([total_data, temp_data[0]], ignore_index=True)
data = pd.read_csv("./netflix-prize-data/combined_data_3.txt", names=['Cust_Id', 'Rating', 'date'])
temp_data = processDataFile(data, temp_data[1])
total_data = pd.concat([total_data, temp_data[0]], ignore_index=True)
data = pd.read_csv("./netflix-prize-data/combined_data_4.txt", names=['Cust_Id', 'Rating', 'date'])
temp_data = processDataFile(data, temp_data[1])
total_data = pd.concat([total_data, temp_data[0]], ignore_index=True)

sample_df = total_data[:4000]
temp = pd.pivot_table(sample_df, index='Cust_Id', columns='movie_id')

probe = pd.read_csv("./netflix-prize-data/probe.txt", names=['Cust_Id'])
probe_movie = pd.DataFrame([])
Cust_Id = []
movie_id = []
currentMovie = 0
index = 0

for row in probe['Cust_Id']:
    if row[len(row) - 1] == ":":
        currentMovie = int(row[:-1])
        # print currentMovie
        continue
    Cust_Id.append(row)
    movie_id.append(currentMovie)
    # print index,row
    index += 1
probe_movie['Cust_Id'] = Cust_Id
probe_movie['movie_id'] = movie_id
merged_probe_set = pd.merge(probe_movie, total_data, on=['Cust_Id', 'movie_id'])

reader = Reader()
# get just top 100K rows for faster run time
trainData = Dataset.load_from_df(total_data[['Cust_Id', 'movie_id', 'Rating']].sample(n=10000000), reader)
trainData.split(n_folds=3)
trainset = trainData.build_full_trainset()


def evaluateTrainAndTestModel(testData, predictions):
    coclustering = CoClustering()
    coclustering.fit(trainset)

    # Predict Movie ratings and compute error
    mse_sum = 0
    n = len(merged_probe_set)
    for index, row in testData.iterrows():
        user = row['Cust_Id']
        currentMovie = row['movie_id']
        pred = coclustering.predict(user, currentMovie, r_ui=row['Rating'], verbose=False)
        predictions[coclustering][index] = pred.est
        diff = row['Rating'] - pred.est
        mse_sum += diff ** 2
    rmse = (mse_sum / n) ** 0.5
    print(" RMSE: ", rmse)
    return predictions

predictions = pd.DataFrame(0, index=np.arange(len(merged_probe_set)), columns='coclustering')
evaluateTrainAndTestModel(merged_probe_set, predictions)

