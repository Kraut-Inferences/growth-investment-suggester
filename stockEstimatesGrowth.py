import yfinance as yf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from yahoo_fin import stock_info as si
from sklearn.utils import shuffle
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from keras.layers import MaxPooling1D, Conv1D
from keras.layers import Flatten

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
import numpy as np
import keras
import time
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from numpy import array
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import TimeDistributed

from sklearn.svm import LinearSVR

buggedTicks = ['CABO','MTD','MKC','CNMD','MCB','AMK','KAI','HUBB','GHC','GCP','MMS','MTX','MCY','MUSA']

#########################################################
def create_dataset(dataset, look_back=1):
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:i+look_back]
    dataX.append(a)
    dataY.append([dataset[i + look_back - 1, :]])
  dataX         =   numpy.reshape(dataX,(len(dataX),1,look_back))
  return np.array(dataX), np.array(dataY)

def build_model(look_back=1):
    model = Sequential()
    model.add(LSTM(look_back,return_sequences=True,input_shape=(1,look_back)))
    model.add(Dense(1,activation='tanh'))
    model.compile(loss='mean_squared_error',optimizer='SGD')
    return model

def perceptronModel():
  model = Sequential()
  print(input_dim)
  model.add(Dense(13,input_dim=13,kernel_initializer='normal',activation='tanh'))
  model.add(Dense(6,kernel_initializer='normal',activation='tanh'))
  model.add(Dense(1,kernel_initializer='normal'))
  model.compile(loss='mean_squared_error',optimizer='adam')
  return model

def series(ticker,dayOffSet):
    #print(ticker)
    data                =   si.get_data(ticker)
    if(len(data.iloc[:,4])<=253*5):
      return 0,0
    scaler              =   MinMaxScaler(feature_range=(0,1))
    neuralInput         =   numpy.reshape(data.iloc[:,4].values,(len(data.iloc[:,4].values),1))
    #neuralInput         =   numpy.reshape(data.iloc[:-dayOffSet,4].values,(len(data.iloc[:-dayOffSet,4].values),1))
    neuralInput         =   scaler.fit_transform(neuralInput)    
    x,y                 =   create_dataset(neuralInput,look_back=dayOffSet)
    model = build_model(dayOffSet)
    model.fit(x, y, epochs=100,verbose=0)
    predictions         =   model.predict(x)
    predictions = scaler.inverse_transform(numpy.reshape(predictions,(len(predictions),1)))
    price = scaler.inverse_transform(numpy.reshape(y,(len(y),1)))    
    #print("prediction:"+str(predictions[-1][0])+", price:"+str(price[-1][0])+" ratio"+str(())
    return (predictions[-1][0]/price[-1][0])-1,price[-1][0]

########################################################

table = pd.read_csv("tableGrowth.csv")
table = table.drop('Unnamed: 0',axis=1)
print("before survey")
table = table[table.price != 0.0]

table = table.drop(["dividendRate"],axis=1)
table = table.drop(["pegRatio"],axis=1)
table = table.drop(["dividendYield"],axis=1)
table = table.drop(["fiveYearAvgDividendYield"],axis=1)
table = table.drop(["trailingAnnualDividendYield"],axis=1)
table = table.drop(["trailingAnnualDividendRate"],axis=1)
table = table.drop(["earningsQuarterlyGrowth"],axis=1)
print(table.isnull().mean())

table = table[table['priceToSalesTrailing12Months'].notna()]
table['trailingSales'] = table['price']/table['priceToSalesTrailing12Months']
table = table[table['forwardPE'].notna()]
table = table[table['forwardEps'].notna()]
table = table.fillna(0)

table['Tsales2Teps']= table['trailingSales']/table['trailingEps']
table['Tsales2Feps']= table['trailingSales']/table['forwardEps']
table['FPE2FEPS']   = table['forwardPE']/table['forwardEps']
table['TPE2TEPS']   = table['trailingPE']/table['trailingEps']
table['F2Teps']     = table['forwardEps']/table['trailingEps']
table['PF2T']       = table['forwardPE']/table['trailingPE']
table['PB2book']    = table['priceToBook']/table['bookValue']
table['Feps2book']  = table['forwardEps']/table['bookValue']
table['Teps2book']  = table['trailingEps']/table['bookValue']

table['Feps2book-'] = table['forwardEps']-table['bookValue']
table['Teps2book-'] = table['trailingEps']-table['bookValue']
table['F2Teps-']    = table['forwardEps']-table['trailingEps']

table  = table.drop(columns='forwardPE')
table  = table.drop(columns='Tsales2Teps')
table  = table.drop(columns='trailingPE')
table  = table.drop(columns='TPE2TEPS')
table  = table.drop(columns='F2Teps')
table  = table.drop(columns='F2Teps-')
table  = table.drop(columns='priceToSalesTrailing12Months')
table  = table.drop(columns='priceToBook')

table = table[table['price'] > 5]
print(table.isnull().mean())
print(table.info())
#time.sleep(30)

table = table.fillna(0)
stdPrice    = 0
meanPrice   = 0
for i in buggedTicks:
  table = table[table.ticker != i]
for column in table.columns.values.tolist():
    if(column == "ticker"):
        continue
    if(column == "price"):
        stdPrice  = table["price"].std()
        meanPrice = table["price"].mean()
        table["price"] = (table["price"]-meanPrice)/stdPrice
        continue
    curStd = table[column].std()
    curMean= table[column].mean()
    table[column] = (table[column]-curMean)/curStd


database = pd.read_csv("growthDatabase.csv")
database = database.drop('Unnamed: 0',1)
print("before")

database = database[database.price != 0.0]
database = database.drop(["dividendRate"],axis=1)
database = database.drop(["pegRatio"],axis=1)
database = database.drop(["dividendYield"],axis=1)
database = database.drop(["fiveYearAvgDividendYield"],axis=1)
database = database.drop(["trailingAnnualDividendYield"],axis=1)
database = database.drop(["trailingAnnualDividendRate"],axis=1)
database = database.drop(["earningsQuarterlyGrowth"],axis=1)
print(database.isnull().mean())

database = database[database['priceToSalesTrailing12Months'].notna()]
database['trailingSales'] = database['price']/database['priceToSalesTrailing12Months']
database = database[database['forwardPE'].notna()]
database = database[database['forwardEps'].notna()]
database = database.fillna(0)

database['Tsales2Teps']= database['trailingSales']/database['trailingEps']
database['Tsales2Feps']= database['trailingSales']/database['forwardEps']
database['FPE2FEPS']   = database['forwardPE']/database['forwardEps']
database['TPE2TEPS']   = database['trailingPE']/database['trailingEps']
database['F2Teps']     = database['forwardEps']/database['trailingEps']
database['PF2T']       = database['forwardPE']/database['trailingPE']
database['PB2book']    = database['priceToBook']/database['bookValue']
database['Feps2book']  = database['forwardEps']/database['bookValue']
database['Teps2book']  = database['trailingEps']/database['bookValue']
database['Feps2book-'] = database['forwardEps']-database['bookValue']
database['Teps2book-'] = database['trailingEps']-database['bookValue']
database['F2Teps-']    = database['forwardEps']-database['trailingEps']
#database  = database.drop(columns='payoutRatio')
database  = database.drop(columns='forwardPE')
database  = database.drop(columns='Tsales2Teps')
database  = database.drop(columns='trailingPE')
database  = database.drop(columns='TPE2TEPS')
database  = database.drop(columns='F2Teps')
database  = database.drop(columns='F2Teps-')
database  = database.drop(columns='priceToSalesTrailing12Months')
database  = database.drop(columns='priceToBook')

#database = database[database['price'] > 5]
print(database.isnull().mean())
print(database.info())
#time.sleep(30)

database = database.fillna(0)
databaseStdPrice    = 0
databaseMeanPrice   = 0
for i in buggedTicks:
  database = database[database.ticker != i]
for column in database.columns.values.tolist():
    if(column == "ticker"):
        continue
    if(column == "price"):
        databaseStdPrice  = database["price"].std()
        databaseMeanPrice = database["price"].mean()
        database["price"] = (database["price"]-databaseMeanPrice)/databaseStdPrice
        continue
    curStd = database[column].std()
    curMean= database[column].mean()
    database[column] = (database[column]-curMean)/curStd



inputs     = keras.layers.Input(shape=(len(table.columns.values.tolist())-3,))
layer1     = keras.layers.Dense(5,activation="tanh")(inputs)
#layer2     = keras.layers.Dense(3,activation="tanh")(layer1)
#layer3     = keras.layers.Dense(3,activation="tanh")(layer1)
outputs    = keras.layers.Dense(1,activation="tanh")(layer1)
sgd        = keras.optimizers.SGD(lr=0.05,decay=0.0005,momentum=0.9,nesterov=True)
model      = keras.models.Model(inputs=inputs,outputs=outputs)
model.compile(optimizer=sgd,loss='mean_absolute_error')

database = shuffle(database)
trainInputs     = database.iloc[:,3:]
trainOutputs    = database.iloc[:,1]

def build_regressor():
  regressor = Sequential()
  regressor.add(Dense(units=16,activation='tanh',input_dim=len(table.columns.values.tolist())-3))
  #regressor.add(Dense(units=8,activation='tanh'))
  regressor.add(Dense(units=4,activation='sigmoid'))
  regressor.add(Dense(units=1,activation='linear'))
  sgd        = keras.optimizers.SGD(lr=0.005,decay=0.00005,momentum=0.9,nesterov=True)
  regressor.compile(optimizer=sgd,loss='mean_squared_error',metrics=['mse'])
  return regressor

#database = database[database['price'] < (1000-databaseMeanPrice)/databaseStdPrice]
#table = table[table['price'] < (1000-databaseMeanPrice)/databaseStdPrice]
#database = database[database['price'] > (5-databaseMeanPrice)/databaseStdPrice]
#table = table[table['price'] > (5-databaseMeanPrice)/databaseStdPrice]


regressor = KerasRegressor(build_fn=build_regressor,batch_size=5,epochs=100)
#print(len(database.iloc[:,1]))
#svr = LinearSVR(random_state=None,tol=1e-9,C=1000.0,max_iter=100000,loss='squared_epsilon_insensitive')
#svm = SVR(kernel='linear',tol=0.00000001,epsilon=0.1,C=100000,cache_size=1000)
#knn = KNeighborsRegressor(n_neighbors=int(math.sqrt(len(database.iloc[:,1]))), weights='distance', algorithm='brute')
#clf = tree.DecisionTreeRegressor()
#reg = LinearRegression()
#lasLars = linear_model.LassoLars(alpha=0.1)
#p = perceptronModel(len((database.iloc[:,2:])[0]))

results=regressor.fit(database.iloc[:,3:],database.iloc[:,1],verbose=1,shuffle=True)
#model.fit(trainInputs,trainOutputs,epochs=100,batch_size=10,verbose=1,shuffle=True)
#svr.fit(trainInputs,trainOutputs)
#svm.fit(trainInputs,trainOutputs)
#knn.fit(database.iloc[:,3:],database.iloc[:,1])
#clf.fit(database.iloc[:,3:],database.iloc[:,1])
#reg.fit(database.iloc[:,3:],database.iloc[:,1])
#lasLars.fit(database.iloc[:,3:],database.iloc[:,1])
#p.fit(database.iloc[:,3:],database.iloc[:,1])

table["change"]     = regressor.predict(table.iloc[:,3:])
#table["change"]     = model.predict(table.iloc[:,3:])
#table["change"]     = svr.predict(table.iloc[:,3:])
#table["change"]     = svm.predict(table.iloc[:,3:])
#table["change"]     = knn.predict(table.iloc[:,3:])
#table["change"]     = clf.predict(table.iloc[:,3:])
#table["change"]     = reg.predict(table.iloc[:,3:])
#table["change"]     = lasLars.predict(table.iloc[:,3:])
#table["change"]     = p.predict(table.iloc[:,3:])

#n_features = len(table.columns.values.tolist())-2

#table["price"]          = (table["price"]*stdPrice)+meanPrice
#table["priceTarget"]    = (table["priceTarget"]*stdPrice)+meanPrice
#table["delta"]          = (table["priceTarget"]/table["price"])
#table = table[table.price < 200]
#table = table[table.price > 20]

#table = table[table.price > table.price.quantile(0.5)]
table1 = table[table.change > table.change.quantile(0.975)]
table2 = table[table.change < table.change.quantile(0.025)]
print(database.columns)
print(table.columns)

seriesTable1 = pd.DataFrame(data={"ticker":[numpy.nan],"priceRatio":[numpy.nan]})
for i in table1.ticker.values:
  try:
    print(i)
    ratio, price = series(i,14)
    entry = pd.DataFrame(data={"ticker":[i],"priceRatio":[ratio],"price":[price]})
    seriesTable1 = seriesTable1.append(entry)
  except:
    pass
  
seriesTable2 = pd.DataFrame(data={"ticker":[numpy.nan],"priceRatio":[numpy.nan]})
for i in table2.ticker.values:
  try:
    print(i)
    ratio, price = series(i,14)
    entry = pd.DataFrame(data={"ticker":[i],"priceRatio":[ratio],"price":[price]})
    seriesTable2 = seriesTable2.append(entry)
  except:
    pass

print("BUY")
seriesTable1 = seriesTable1[seriesTable1.priceRatio > 0]
print(seriesTable1.sort_values("priceRatio"))
print(table1.sort_values("change"))

print("SELL")
seriesTable2 = seriesTable2[seriesTable2.priceRatio < 0]
print(seriesTable2.sort_values("priceRatio"))
print(table2.sort_values("change"))
