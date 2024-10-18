
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

import pandas as pd
import numpy as np
#import Regression Modules - ML
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor

# import tuing model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import optuna
from sklearn.preprocessing import StandardScaler

# split data
from sklearn.model_selection import train_test_split
C:\Users\Rithvika\AppData\Roaming\Python\Python311\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
train_df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1460 entries, 0 to 1459
Data columns (total 81 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Id             1460 non-null   int64  
 1   MSSubClass     1460 non-null   int64  
 2   MSZoning       1460 non-null   object 
 3   LotFrontage    1201 non-null   float64
 4   LotArea        1460 non-null   int64  
 5   Street         1460 non-null   object 
 6   Alley          91 non-null     object 
 7   LotShape       1460 non-null   object 
 8   LandContour    1460 non-null   object 
 9   Utilities      1460 non-null   object 
 10  LotConfig      1460 non-null   object 
 11  LandSlope      1460 non-null   object 
 12  Neighborhood   1460 non-null   object 
 13  Condition1     1460 non-null   object 
 14  Condition2     1460 non-null   object 
 15  BldgType       1460 non-null   object 
 16  HouseStyle     1460 non-null   object 
 17  OverallQual    1460 non-null   int64  
 18  OverallCond    1460 non-null   int64  
 19  YearBuilt      1460 non-null   int64  
 20  YearRemodAdd   1460 non-null   int64  
 21  RoofStyle      1460 non-null   object 
 22  RoofMatl       1460 non-null   object 
 23  Exterior1st    1460 non-null   object 
 24  Exterior2nd    1460 non-null   object 
 25  MasVnrType     588 non-null    object 
 26  MasVnrArea     1452 non-null   float64
 27  ExterQual      1460 non-null   object 
 28  ExterCond      1460 non-null   object 
 29  Foundation     1460 non-null   object 
 30  BsmtQual       1423 non-null   object 
 31  BsmtCond       1423 non-null   object 
 32  BsmtExposure   1422 non-null   object 
 33  BsmtFinType1   1423 non-null   object 
 34  BsmtFinSF1     1460 non-null   int64  
 35  BsmtFinType2   1422 non-null   object 
 36  BsmtFinSF2     1460 non-null   int64  
 37  BsmtUnfSF      1460 non-null   int64  
 38  TotalBsmtSF    1460 non-null   int64  
 39  Heating        1460 non-null   object 
 40  HeatingQC      1460 non-null   object 
 41  CentralAir     1460 non-null   object 
 42  Electrical     1459 non-null   object 
 43  1stFlrSF       1460 non-null   int64  
 44  2ndFlrSF       1460 non-null   int64  
 45  LowQualFinSF   1460 non-null   int64  
 46  GrLivArea      1460 non-null   int64  
 47  BsmtFullBath   1460 non-null   int64  
 48  BsmtHalfBath   1460 non-null   int64  
 49  FullBath       1460 non-null   int64  
 50  HalfBath       1460 non-null   int64  
 51  BedroomAbvGr   1460 non-null   int64  
 52  KitchenAbvGr   1460 non-null   int64  
 53  KitchenQual    1460 non-null   object 
 54  TotRmsAbvGrd   1460 non-null   int64  
 55  Functional     1460 non-null   object 
 56  Fireplaces     1460 non-null   int64  
 57  FireplaceQu    770 non-null    object 
 58  GarageType     1379 non-null   object 
 59  GarageYrBlt    1379 non-null   float64
 60  GarageFinish   1379 non-null   object 
 61  GarageCars     1460 non-null   int64  
 62  GarageArea     1460 non-null   int64  
 63  GarageQual     1379 non-null   object 
 64  GarageCond     1379 non-null   object 
 65  PavedDrive     1460 non-null   object 
 66  WoodDeckSF     1460 non-null   int64  
 67  OpenPorchSF    1460 non-null   int64  
 68  EnclosedPorch  1460 non-null   int64  
 69  3SsnPorch      1460 non-null   int64  
 70  ScreenPorch    1460 non-null   int64  
 71  PoolArea       1460 non-null   int64  
 72  PoolQC         7 non-null      object 
 73  Fence          281 non-null    object 
 74  MiscFeature    54 non-null     object 
 75  MiscVal        1460 non-null   int64  
 76  MoSold         1460 non-null   int64  
 77  YrSold         1460 non-null   int64  
 78  SaleType       1460 non-null   object 
 79  SaleCondition  1460 non-null   object 
 80  SalePrice      1460 non-null   int64  
dtypes: float64(3), int64(35), object(43)
memory usage: 924.0+ KB
msno.matrix(train_df)
<Axes: >

msno.matrix(test_df)
<Axes: >

remove_list1=["Id"]
for i in train_df.columns.tolist():
    if train_df[i].isnull().sum() >= 500:
        print(i,train_df[i].isnull().sum())
        remove_list1.append(i)
Alley 1369
MasVnrType 872
FireplaceQu 690
PoolQC 1453
Fence 1179
MiscFeature 1406
train_df = train_df.drop(columns=remove_list1)
test_df = test_df.drop(columns=remove_list1)
object_columns = train_df.select_dtypes(include=['object']).columns.tolist()
print(object_columns)
['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
for i in object_columns:
    frequency = train_df[i].value_counts(normalize=True)  # Normalize to get frequency ratio
    train_df[str(i)+'_encoded'] = train_df[i].map(frequency)
for i in object_columns:
    train_df = train_df.drop(columns = [i])
object_columns = test_df.select_dtypes(include=['object']).columns.tolist()
for i in object_columns:
    frequency = test_df[i].value_counts(normalize=True)  # Normalize to get frequency ratio
    test_df[str(i)+'_encoded'] = test_df[i].map(frequency)
for i in object_columns:
    test_df = test_df.drop(columns = [i])
per_df = pd.DataFrame()
for i in range(len(train_df)):
    if train_df.iloc[i].isnull().sum() == 0:
        per_df = per_df._append(train_df.iloc[i], ignore_index=True)
per_df.head()
MSSubClass	LotFrontage	LotArea	OverallQual	OverallCond	YearBuilt	YearRemodAdd	MasVnrArea	BsmtFinSF1	BsmtFinSF2	...	Electrical_encoded	KitchenQual_encoded	Functional_encoded	GarageType_encoded	GarageFinish_encoded	GarageQual_encoded	GarageCond_encoded	PavedDrive_encoded	SaleType_encoded	SaleCondition_encoded
0	60.0	65.0	8450.0	7.0	5.0	2003.0	2003.0	196.0	706.0	0.0	...	0.914325	0.401370	0.931507	0.630892	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548
1	20.0	80.0	9600.0	6.0	8.0	1976.0	1976.0	0.0	978.0	0.0	...	0.914325	0.503425	0.931507	0.630892	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548
2	60.0	68.0	11250.0	7.0	5.0	2001.0	2002.0	162.0	486.0	0.0	...	0.914325	0.401370	0.931507	0.630892	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548
3	70.0	60.0	9550.0	7.0	5.0	1915.0	1970.0	0.0	216.0	0.0	...	0.914325	0.401370	0.931507	0.280638	0.438724	0.950689	0.961566	0.917808	0.867808	0.069178
4	60.0	84.0	14260.0	8.0	5.0	2000.0	2000.0	350.0	655.0	0.0	...	0.914325	0.401370	0.931507	0.630892	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548
5 rows × 74 columns

per_test_df = per_df.drop(columns=['SalePrice'])
per_test_df.head()
MSSubClass	LotFrontage	LotArea	OverallQual	OverallCond	YearBuilt	YearRemodAdd	MasVnrArea	BsmtFinSF1	BsmtFinSF2	...	Electrical_encoded	KitchenQual_encoded	Functional_encoded	GarageType_encoded	GarageFinish_encoded	GarageQual_encoded	GarageCond_encoded	PavedDrive_encoded	SaleType_encoded	SaleCondition_encoded
0	60.0	65.0	8450.0	7.0	5.0	2003.0	2003.0	196.0	706.0	0.0	...	0.914325	0.401370	0.931507	0.630892	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548
1	20.0	80.0	9600.0	6.0	8.0	1976.0	1976.0	0.0	978.0	0.0	...	0.914325	0.503425	0.931507	0.630892	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548
2	60.0	68.0	11250.0	7.0	5.0	2001.0	2002.0	162.0	486.0	0.0	...	0.914325	0.401370	0.931507	0.630892	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548
3	70.0	60.0	9550.0	7.0	5.0	1915.0	1970.0	0.0	216.0	0.0	...	0.914325	0.401370	0.931507	0.280638	0.438724	0.950689	0.961566	0.917808	0.867808	0.069178
4	60.0	84.0	14260.0	8.0	5.0	2000.0	2000.0	350.0	655.0	0.0	...	0.914325	0.401370	0.931507	0.630892	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548
5 rows × 73 columns

train_null_columns = []
train_null_sum = []
for i in range(len(train_df)):
    if train_df.iloc[i].isnull().sum() > 0:
        train_null_columns.append(i)
        train_null_sum.append(train_df.iloc[i].isnull().sum())
per_df
MSSubClass	LotFrontage	LotArea	OverallQual	OverallCond	YearBuilt	YearRemodAdd	MasVnrArea	BsmtFinSF1	BsmtFinSF2	...	Electrical_encoded	KitchenQual_encoded	Functional_encoded	GarageType_encoded	GarageFinish_encoded	GarageQual_encoded	GarageCond_encoded	PavedDrive_encoded	SaleType_encoded	SaleCondition_encoded
0	60.0	65.0	8450.0	7.0	5.0	2003.0	2003.0	196.0	706.0	0.0	...	0.914325	0.401370	0.931507	0.630892	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548
1	20.0	80.0	9600.0	6.0	8.0	1976.0	1976.0	0.0	978.0	0.0	...	0.914325	0.503425	0.931507	0.630892	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548
2	60.0	68.0	11250.0	7.0	5.0	2001.0	2002.0	162.0	486.0	0.0	...	0.914325	0.401370	0.931507	0.630892	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548
3	70.0	60.0	9550.0	7.0	5.0	1915.0	1970.0	0.0	216.0	0.0	...	0.914325	0.401370	0.931507	0.280638	0.438724	0.950689	0.961566	0.917808	0.867808	0.069178
4	60.0	84.0	14260.0	8.0	5.0	2000.0	2000.0	350.0	655.0	0.0	...	0.914325	0.401370	0.931507	0.630892	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
1089	60.0	62.0	7917.0	6.0	5.0	1999.0	2000.0	0.0	0.0	0.0	...	0.914325	0.503425	0.931507	0.630892	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548
1090	20.0	85.0	13175.0	6.0	6.0	1978.0	1988.0	119.0	790.0	163.0	...	0.914325	0.503425	0.021233	0.630892	0.438724	0.950689	0.961566	0.917808	0.867808	0.820548
1091	70.0	66.0	9042.0	7.0	9.0	1941.0	2006.0	0.0	275.0	0.0	...	0.914325	0.401370	0.931507	0.630892	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548
1092	20.0	68.0	9717.0	5.0	6.0	1950.0	1996.0	0.0	49.0	1029.0	...	0.064428	0.401370	0.931507	0.630892	0.438724	0.950689	0.961566	0.917808	0.867808	0.820548
1093	20.0	75.0	9937.0	5.0	6.0	1965.0	1965.0	0.0	830.0	290.0	...	0.914325	0.503425	0.931507	0.630892	0.255257	0.950689	0.961566	0.917808	0.867808	0.820548
1094 rows × 74 columns

print("Row with null value",train_null_columns,end=" ")
print("\n\n","the number of null values",train_null_sum,end=" ")
Row with null value [7, 12, 14, 16, 17, 24, 31, 39, 42, 43, 48, 50, 64, 66, 76, 78, 84, 88, 89, 90, 95, 99, 100, 102, 104, 108, 111, 113, 116, 120, 125, 126, 127, 131, 133, 136, 140, 147, 148, 149, 152, 153, 155, 156, 160, 163, 165, 166, 169, 170, 177, 180, 182, 186, 191, 198, 203, 207, 208, 210, 214, 218, 221, 234, 237, 241, 244, 249, 250, 259, 269, 287, 288, 291, 293, 307, 308, 310, 319, 328, 330, 332, 335, 342, 346, 347, 351, 356, 360, 361, 362, 364, 366, 369, 370, 371, 375, 384, 386, 392, 393, 404, 405, 412, 421, 426, 431, 434, 441, 447, 452, 457, 458, 459, 464, 465, 470, 484, 490, 495, 496, 516, 518, 520, 528, 529, 532, 533, 535, 537, 538, 539, 541, 545, 553, 559, 560, 562, 564, 569, 580, 582, 593, 610, 611, 612, 613, 614, 616, 620, 623, 626, 635, 636, 638, 641, 645, 646, 649, 650, 660, 666, 668, 672, 679, 682, 685, 687, 690, 705, 706, 709, 710, 714, 720, 721, 726, 734, 736, 738, 745, 746, 749, 750, 751, 757, 770, 778, 783, 784, 785, 789, 791, 794, 811, 816, 817, 822, 826, 828, 840, 843, 845, 851, 853, 855, 856, 859, 865, 868, 879, 882, 893, 894, 897, 900, 904, 908, 911, 917, 921, 925, 927, 928, 929, 936, 939, 941, 942, 944, 948, 953, 954, 960, 961, 967, 968, 970, 973, 975, 976, 977, 980, 983, 984, 988, 996, 997, 1000, 1003, 1006, 1009, 1011, 1017, 1018, 1024, 1030, 1032, 1033, 1035, 1037, 1038, 1041, 1045, 1048, 1049, 1057, 1059, 1064, 1077, 1084, 1086, 1090, 1096, 1097, 1108, 1110, 1116, 1122, 1123, 1124, 1131, 1137, 1138, 1141, 1143, 1146, 1148, 1153, 1154, 1161, 1164, 1173, 1177, 1179, 1180, 1190, 1193, 1206, 1213, 1216, 1218, 1219, 1230, 1232, 1233, 1234, 1243, 1244, 1247, 1251, 1253, 1257, 1260, 1262, 1268, 1270, 1271, 1272, 1276, 1277, 1278, 1283, 1286, 1287, 1290, 1300, 1301, 1309, 1312, 1318, 1321, 1323, 1325, 1326, 1337, 1342, 1346, 1348, 1349, 1354, 1356, 1357, 1358, 1362, 1365, 1368, 1373, 1379, 1381, 1383, 1396, 1407, 1412, 1417, 1419, 1423, 1424, 1429, 1431, 1441, 1443, 1446, 1449, 1450, 1453] 

 the number of null values [1, 1, 1, 1, 5, 1, 1, 10, 1, 1, 5, 1, 1, 1, 1, 5, 1, 5, 5, 5, 1, 5, 1, 5, 1, 5, 1, 1, 1, 1, 5, 1, 5, 1, 1, 1, 5, 1, 5, 1, 1, 1, 5, 5, 1, 5, 5, 1, 1, 1, 1, 1, 5, 1, 1, 5, 1, 1, 1, 5, 1, 1, 1, 2, 1, 5, 1, 1, 5, 5, 1, 6, 1, 5, 1, 6, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 6, 1, 5, 6, 6, 1, 1, 1, 1, 1, 5, 5, 5, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1, 1, 1, 10, 5, 2, 5, 10, 5, 1, 1, 1, 1, 1, 5, 1, 1, 5, 1, 1, 1, 5, 1, 1, 1, 1, 5, 5, 1, 5, 1, 1, 5, 5, 5, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 1, 1, 5, 1, 1, 1, 1, 1, 5, 5, 1, 1, 5, 5, 1, 1, 1, 5, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 5, 5, 1, 1, 5, 5, 1, 1, 5, 1, 1, 1, 5, 1, 1, 1, 5, 1, 1, 5, 10, 1, 1, 1, 6, 1, 1, 6, 1, 5, 1, 6, 5, 5, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1, 5, 1, 5, 5, 1, 1, 6, 1, 1, 1, 1, 1, 1, 5, 1, 10, 1, 1, 1, 1, 1, 5, 10, 5, 1, 5, 1, 5, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 6, 5, 5, 5, 5, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5] 
def fill_null_val(per,df):
    filled_df = df.copy()

    null_columns = []
    null_sum = []
    for i in range(len(filled_df)):
        if filled_df.iloc[i].isnull().sum() > 0:
            null_columns.append(i)
            null_sum.append(filled_df.iloc[i].isnull().sum())
    for t in null_columns:
        d_f = filled_df.copy()
        result = d_f.iloc[t].isnull()
        null_columns_name = result[result].index.tolist()

        # train model - use per_df data
        y = pd.DataFrame(per[null_columns_name])
        X = pd.DataFrame(per.drop(columns=null_columns_name))
        model = MultiOutputRegressor(LinearRegression())

        # create model
        model.fit(X, y)
        # insert to actual data - from train/test data
        X_test = pd.DataFrame(d_f.loc[t])

        # generate to dataframe
        result_df = pd.DataFrame(X_test, columns=X_test.columns)

        # exchange column - index
        X_test = result_df.T
        X_test = X_test.drop(columns=null_columns_name)

        y_pred = model.predict(X_test)

        for i in range(len(null_columns_name)):
            d_f.loc[t, null_columns_name[i]] = y_pred.tolist()[0][i]
            filled_df = d_f.copy()
    
    return filled_df
filled_train_df = fill_null_val(per_df,train_df)
msno.matrix(filled_train_df)
<Axes: >

per_test_df = per_df.copy()
column_to_move = 'SalePrice'

new_order = [col for col in per_test_df.columns if col != column_to_move] + [column_to_move]
per_test_df = per_test_df[new_order]
per_test_df = per_test_df.iloc[:,:-1]
per_test_df
MSSubClass	LotFrontage	LotArea	OverallQual	OverallCond	YearBuilt	YearRemodAdd	MasVnrArea	BsmtFinSF1	BsmtFinSF2	...	Electrical_encoded	KitchenQual_encoded	Functional_encoded	GarageType_encoded	GarageFinish_encoded	GarageQual_encoded	GarageCond_encoded	PavedDrive_encoded	SaleType_encoded	SaleCondition_encoded
0	60.0	65.0	8450.0	7.0	5.0	2003.0	2003.0	196.0	706.0	0.0	...	0.914325	0.401370	0.931507	0.630892	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548
1	20.0	80.0	9600.0	6.0	8.0	1976.0	1976.0	0.0	978.0	0.0	...	0.914325	0.503425	0.931507	0.630892	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548
2	60.0	68.0	11250.0	7.0	5.0	2001.0	2002.0	162.0	486.0	0.0	...	0.914325	0.401370	0.931507	0.630892	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548
3	70.0	60.0	9550.0	7.0	5.0	1915.0	1970.0	0.0	216.0	0.0	...	0.914325	0.401370	0.931507	0.280638	0.438724	0.950689	0.961566	0.917808	0.867808	0.069178
4	60.0	84.0	14260.0	8.0	5.0	2000.0	2000.0	350.0	655.0	0.0	...	0.914325	0.401370	0.931507	0.630892	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
1089	60.0	62.0	7917.0	6.0	5.0	1999.0	2000.0	0.0	0.0	0.0	...	0.914325	0.503425	0.931507	0.630892	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548
1090	20.0	85.0	13175.0	6.0	6.0	1978.0	1988.0	119.0	790.0	163.0	...	0.914325	0.503425	0.021233	0.630892	0.438724	0.950689	0.961566	0.917808	0.867808	0.820548
1091	70.0	66.0	9042.0	7.0	9.0	1941.0	2006.0	0.0	275.0	0.0	...	0.914325	0.401370	0.931507	0.630892	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548
1092	20.0	68.0	9717.0	5.0	6.0	1950.0	1996.0	0.0	49.0	1029.0	...	0.064428	0.401370	0.931507	0.630892	0.438724	0.950689	0.961566	0.917808	0.867808	0.820548
1093	20.0	75.0	9937.0	5.0	6.0	1965.0	1965.0	0.0	830.0	290.0	...	0.914325	0.503425	0.931507	0.630892	0.255257	0.950689	0.961566	0.917808	0.867808	0.820548
1094 rows × 73 columns

filled_test_df = fill_null_val(per_test_df,test_df)
msno.matrix(filled_test_df)
<Axes: >

def features_(filled_train_df):
    filled_train_df['TotalBathArea'] = filled_train_df['FullBath'] + 0.5 * filled_train_df['HalfBath']
    filled_train_df['TotalBathroomArea'] = filled_train_df['FullBath'] + 0.5 * filled_train_df['HalfBath']
    outdoor_features = ['WoodDeckSF', 'OpenPorchSF', '3SsnPorch', 'ScreenPorch']
    filled_train_df['TotalOutdoorArea'] = filled_train_df[outdoor_features].sum(axis=1)
    basement_features = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
    filled_train_df['TotalBasementArea'] = filled_train_df[basement_features].sum(axis=1)
for i in [filled_train_df,filled_test_df,per_df]:
    features_(i)
filled_train_df.head()
MSSubClass	LotFrontage	LotArea	OverallQual	OverallCond	YearBuilt	YearRemodAdd	MasVnrArea	BsmtFinSF1	BsmtFinSF2	...	GarageFinish_encoded	GarageQual_encoded	GarageCond_encoded	PavedDrive_encoded	SaleType_encoded	SaleCondition_encoded	TotalBathArea	TotalBathroomArea	TotalOutdoorArea	TotalBasementArea
0	60	65.0	8450	7	5	2003	2003	196.0	706	0	...	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548	2.5	2.5	61	1712
1	20	80.0	9600	6	8	1976	1976	0.0	978	0	...	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548	2.0	2.0	298	2524
2	60	68.0	11250	7	5	2001	2002	162.0	486	0	...	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548	2.5	2.5	42	1840
3	70	60.0	9550	7	5	1915	1970	0.0	216	0	...	0.438724	0.950689	0.961566	0.917808	0.867808	0.069178	1.0	1.0	35	1512
4	60	84.0	14260	8	5	2000	2000	350.0	655	0	...	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548	2.5	2.5	276	2290
5 rows × 78 columns

filled_test_df.head()
MSSubClass	LotFrontage	LotArea	OverallQual	OverallCond	YearBuilt	YearRemodAdd	MasVnrArea	BsmtFinSF1	BsmtFinSF2	...	GarageFinish_encoded	GarageQual_encoded	GarageCond_encoded	PavedDrive_encoded	SaleType_encoded	SaleCondition_encoded	TotalBathArea	TotalBathroomArea	TotalOutdoorArea	TotalBasementArea
0	20	80.0	11622	5	6	1961	1961	0.0	468.0	144.0	...	0.452571	0.936278	0.961622	0.891707	0.862826	0.825223	1.0	1.0	260	1764.0
1	20	81.0	14267	6	6	1958	1958	108.0	923.0	0.0	...	0.452571	0.936278	0.961622	0.891707	0.862826	0.825223	1.5	1.5	429	2658.0
2	60	74.0	13830	5	5	1997	1998	0.0	791.0	0.0	...	0.265749	0.936278	0.961622	0.891707	0.862826	0.825223	2.5	2.5	246	1856.0
3	60	78.0	9978	6	6	1998	1998	20.0	602.0	0.0	...	0.265749	0.936278	0.961622	0.891707	0.862826	0.825223	2.5	2.5	396	1852.0
4	120	43.0	5005	8	5	1992	1992	0.0	263.0	0.0	...	0.281680	0.936278	0.961622	0.891707	0.862826	0.825223	2.0	2.0	226	2560.0
5 rows × 77 columns

per_df.head()
MSSubClass	LotFrontage	LotArea	OverallQual	OverallCond	YearBuilt	YearRemodAdd	MasVnrArea	BsmtFinSF1	BsmtFinSF2	...	GarageFinish_encoded	GarageQual_encoded	GarageCond_encoded	PavedDrive_encoded	SaleType_encoded	SaleCondition_encoded	TotalBathArea	TotalBathroomArea	TotalOutdoorArea	TotalBasementArea
0	60.0	65.0	8450.0	7.0	5.0	2003.0	2003.0	196.0	706.0	0.0	...	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548	2.5	2.5	61.0	1712.0
1	20.0	80.0	9600.0	6.0	8.0	1976.0	1976.0	0.0	978.0	0.0	...	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548	2.0	2.0	298.0	2524.0
2	60.0	68.0	11250.0	7.0	5.0	2001.0	2002.0	162.0	486.0	0.0	...	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548	2.5	2.5	42.0	1840.0
3	70.0	60.0	9550.0	7.0	5.0	1915.0	1970.0	0.0	216.0	0.0	...	0.438724	0.950689	0.961566	0.917808	0.867808	0.069178	1.0	1.0	35.0	1512.0
4	60.0	84.0	14260.0	8.0	5.0	2000.0	2000.0	350.0	655.0	0.0	...	0.306019	0.950689	0.961566	0.917808	0.867808	0.820548	2.5	2.5	276.0	2290.0
5 rows × 78 columns

colormap = plt.cm.PuBu
plt.figure(figsize=(18, 18))
plt.title("Person Correlation of Features", y = 1.05, size = 15)
sns.heatmap(filled_train_df.astype(float).corr(), linewidths = 0.1, vmax = 1.0,
           square = True, cmap = colormap, linecolor = "white", annot = True, annot_kws = {"size" : 0})
<Axes: title={'center': 'Person Correlation of Features'}>

corr = per_df.corr(method ="pearson")
sp_corr = corr.iloc[corr.index.tolist().index("SalePrice")].values.tolist()
left_col = []
for i in range(len(sp_corr)):
    if type(sp_corr[i]) == float:
        if sp_corr[i] >= 0.25:
            left_col.append(corr.index[i])
print(left_col,end = " ")
['LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'SalePrice', 'Exterior1st_encoded', 'Exterior2nd_encoded', 'BsmtFinType1_encoded', 'HeatingQC_encoded', 'TotalBathArea', 'TotalBathroomArea', 'TotalOutdoorArea', 'TotalBasementArea'] 
corr = per_df.corr(method ="pearson")
sp_corr = corr.iloc[corr.index.tolist().index("SalePrice")].values.tolist()
left_col = []
for i in range(len(sp_corr)):
    if type(sp_corr[i]) == float:
        if sp_corr[i] >= 0.25:
            left_col.append(corr.index[i])
print(left_col,end = " ")
['LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'SalePrice', 'Exterior1st_encoded', 'Exterior2nd_encoded', 'BsmtFinType1_encoded', 'HeatingQC_encoded', 'TotalBathArea', 'TotalBathroomArea', 'TotalOutdoorArea', 'TotalBasementArea'] 
new_df = filled_train_df[left_col]

column_to_move = 'SalePrice'

new_order = [col for col in new_df.columns if col != column_to_move] + [column_to_move]
new_df = new_df[new_order]

(new_df)
LotFrontage	LotArea	OverallQual	YearBuilt	YearRemodAdd	MasVnrArea	BsmtFinSF1	TotalBsmtSF	1stFlrSF	2ndFlrSF	...	OpenPorchSF	Exterior1st_encoded	Exterior2nd_encoded	BsmtFinType1_encoded	HeatingQC_encoded	TotalBathArea	TotalBathroomArea	TotalOutdoorArea	TotalBasementArea	SalePrice
0	65.0	8450	7	2003	2003	196.0	706	856	856	854	...	61	0.352740	0.345205	0.293746	0.507534	2.5	2.5	61	1712	208500
1	80.0	9600	6	1976	1976	0.0	978	1262	1262	0	...	0	0.150685	0.146575	0.154603	0.507534	2.0	2.0	298	2524	181500
2	68.0	11250	7	2001	2002	162.0	486	920	920	866	...	42	0.352740	0.345205	0.293746	0.507534	2.5	2.5	42	1840	223500
3	60.0	9550	7	1915	1970	0.0	216	756	961	756	...	35	0.141096	0.026027	0.154603	0.165068	1.0	1.0	35	1512	140000
4	84.0	14260	8	2000	2000	350.0	655	1145	1145	1053	...	84	0.352740	0.345205	0.293746	0.507534	2.5	2.5	276	2290	250000
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
1455	62.0	7917	6	1999	2000	0.0	0	953	953	694	...	40	0.352740	0.345205	0.302178	0.507534	2.5	2.5	40	1906	175000
1456	85.0	13175	6	1978	1988	119.0	790	1542	2073	0	...	0	0.073973	0.097260	0.154603	0.293151	2.0	2.0	349	3084	210000
1457	66.0	9042	7	1941	2006	0.0	275	1152	1188	1152	...	60	0.041781	0.041096	0.293746	0.507534	2.0	2.0	60	2304	266500
1458	68.0	9717	5	1950	1996	0.0	49	1078	1078	0	...	0	0.150685	0.146575	0.293746	0.165068	1.0	1.0	366	2156	142125
1459	75.0	9937	5	1965	1965	0.0	830	1256	1256	0	...	68	0.152055	0.141781	0.104006	0.165068	1.5	1.5	804	2512	147500
1460 rows × 29 columns

column_to_move = 'SalePrice'

new_order = [col for col in filled_train_df.columns if col != column_to_move] + [column_to_move]
filled_train_df = filled_train_df[new_order]

(filled_train_df)
MSSubClass	LotFrontage	LotArea	OverallQual	OverallCond	YearBuilt	YearRemodAdd	MasVnrArea	BsmtFinSF1	BsmtFinSF2	...	GarageQual_encoded	GarageCond_encoded	PavedDrive_encoded	SaleType_encoded	SaleCondition_encoded	TotalBathArea	TotalBathroomArea	TotalOutdoorArea	TotalBasementArea	SalePrice
0	60	65.0	8450	7	5	2003	2003	196.0	706	0	...	0.950689	0.961566	0.917808	0.867808	0.820548	2.5	2.5	61	1712	208500
1	20	80.0	9600	6	8	1976	1976	0.0	978	0	...	0.950689	0.961566	0.917808	0.867808	0.820548	2.0	2.0	298	2524	181500
2	60	68.0	11250	7	5	2001	2002	162.0	486	0	...	0.950689	0.961566	0.917808	0.867808	0.820548	2.5	2.5	42	1840	223500
3	70	60.0	9550	7	5	1915	1970	0.0	216	0	...	0.950689	0.961566	0.917808	0.867808	0.069178	1.0	1.0	35	1512	140000
4	60	84.0	14260	8	5	2000	2000	350.0	655	0	...	0.950689	0.961566	0.917808	0.867808	0.820548	2.5	2.5	276	2290	250000
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
1455	60	62.0	7917	6	5	1999	2000	0.0	0	0	...	0.950689	0.961566	0.917808	0.867808	0.820548	2.5	2.5	40	1906	175000
1456	20	85.0	13175	6	6	1978	1988	119.0	790	163	...	0.950689	0.961566	0.917808	0.867808	0.820548	2.0	2.0	349	3084	210000
1457	70	66.0	9042	7	9	1941	2006	0.0	275	0	...	0.950689	0.961566	0.917808	0.867808	0.820548	2.0	2.0	60	2304	266500
1458	20	68.0	9717	5	6	1950	1996	0.0	49	1029	...	0.950689	0.961566	0.917808	0.867808	0.820548	1.0	1.0	366	2156	142125
1459	20	75.0	9937	5	6	1965	1965	0.0	830	290	...	0.950689	0.961566	0.917808	0.867808	0.820548	1.5	1.5	804	2512	147500
1460 rows × 78 columns

column_to_move = 'SalePrice'

new_order = [col for col in per_df.columns if col != column_to_move] + [column_to_move]
per_df = per_df[new_order]

per_df
MSSubClass	LotFrontage	LotArea	OverallQual	OverallCond	YearBuilt	YearRemodAdd	MasVnrArea	BsmtFinSF1	BsmtFinSF2	...	GarageQual_encoded	GarageCond_encoded	PavedDrive_encoded	SaleType_encoded	SaleCondition_encoded	TotalBathArea	TotalBathroomArea	TotalOutdoorArea	TotalBasementArea	SalePrice
0	60.0	65.0	8450.0	7.0	5.0	2003.0	2003.0	196.0	706.0	0.0	...	0.950689	0.961566	0.917808	0.867808	0.820548	2.5	2.5	61.0	1712.0	208500.0
1	20.0	80.0	9600.0	6.0	8.0	1976.0	1976.0	0.0	978.0	0.0	...	0.950689	0.961566	0.917808	0.867808	0.820548	2.0	2.0	298.0	2524.0	181500.0
2	60.0	68.0	11250.0	7.0	5.0	2001.0	2002.0	162.0	486.0	0.0	...	0.950689	0.961566	0.917808	0.867808	0.820548	2.5	2.5	42.0	1840.0	223500.0
3	70.0	60.0	9550.0	7.0	5.0	1915.0	1970.0	0.0	216.0	0.0	...	0.950689	0.961566	0.917808	0.867808	0.069178	1.0	1.0	35.0	1512.0	140000.0
4	60.0	84.0	14260.0	8.0	5.0	2000.0	2000.0	350.0	655.0	0.0	...	0.950689	0.961566	0.917808	0.867808	0.820548	2.5	2.5	276.0	2290.0	250000.0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
1089	60.0	62.0	7917.0	6.0	5.0	1999.0	2000.0	0.0	0.0	0.0	...	0.950689	0.961566	0.917808	0.867808	0.820548	2.5	2.5	40.0	1906.0	175000.0
1090	20.0	85.0	13175.0	6.0	6.0	1978.0	1988.0	119.0	790.0	163.0	...	0.950689	0.961566	0.917808	0.867808	0.820548	2.0	2.0	349.0	3084.0	210000.0
1091	70.0	66.0	9042.0	7.0	9.0	1941.0	2006.0	0.0	275.0	0.0	...	0.950689	0.961566	0.917808	0.867808	0.820548	2.0	2.0	60.0	2304.0	266500.0
1092	20.0	68.0	9717.0	5.0	6.0	1950.0	1996.0	0.0	49.0	1029.0	...	0.950689	0.961566	0.917808	0.867808	0.820548	1.0	1.0	366.0	2156.0	142125.0
1093	20.0	75.0	9937.0	5.0	6.0	1965.0	1965.0	0.0	830.0	290.0	...	0.950689	0.961566	0.917808	0.867808	0.820548	1.5	1.5	804.0	2512.0	147500.0
1094 rows × 78 columns

data_frames = [new_df, per_df, filled_train_df]
scaler = StandardScaler()
for i in range(len(data_frames)):
  data_frames[i] = scaler.fit_transform(data_frames[i])
new_df
LotFrontage	LotArea	OverallQual	YearBuilt	YearRemodAdd	MasVnrArea	BsmtFinSF1	TotalBsmtSF	1stFlrSF	2ndFlrSF	...	OpenPorchSF	Exterior1st_encoded	Exterior2nd_encoded	BsmtFinType1_encoded	HeatingQC_encoded	TotalBathArea	TotalBathroomArea	TotalOutdoorArea	TotalBasementArea	SalePrice
0	65.0	8450	7	2003	2003	196.0	706	856	856	854	...	61	0.352740	0.345205	0.293746	0.507534	2.5	2.5	61	1712	208500
1	80.0	9600	6	1976	1976	0.0	978	1262	1262	0	...	0	0.150685	0.146575	0.154603	0.507534	2.0	2.0	298	2524	181500
2	68.0	11250	7	2001	2002	162.0	486	920	920	866	...	42	0.352740	0.345205	0.293746	0.507534	2.5	2.5	42	1840	223500
3	60.0	9550	7	1915	1970	0.0	216	756	961	756	...	35	0.141096	0.026027	0.154603	0.165068	1.0	1.0	35	1512	140000
4	84.0	14260	8	2000	2000	350.0	655	1145	1145	1053	...	84	0.352740	0.345205	0.293746	0.507534	2.5	2.5	276	2290	250000
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
1455	62.0	7917	6	1999	2000	0.0	0	953	953	694	...	40	0.352740	0.345205	0.302178	0.507534	2.5	2.5	40	1906	175000
1456	85.0	13175	6	1978	1988	119.0	790	1542	2073	0	...	0	0.073973	0.097260	0.154603	0.293151	2.0	2.0	349	3084	210000
1457	66.0	9042	7	1941	2006	0.0	275	1152	1188	1152	...	60	0.041781	0.041096	0.293746	0.507534	2.0	2.0	60	2304	266500
1458	68.0	9717	5	1950	1996	0.0	49	1078	1078	0	...	0	0.150685	0.146575	0.293746	0.165068	1.0	1.0	366	2156	142125
1459	75.0	9937	5	1965	1965	0.0	830	1256	1256	0	...	68	0.152055	0.141781	0.104006	0.165068	1.5	1.5	804	2512	147500
1460 rows × 29 columns

new_df = new_df.loc[:, ['TotalBathArea', 'TotalBathroomArea','TotalOutdoorArea','TotalBasementArea','SalePrice']]
print(new_df)
      TotalBathArea  TotalBathroomArea  TotalOutdoorArea  TotalBasementArea  \
0               2.5                2.5                61               1712   
1               2.0                2.0               298               2524   
2               2.5                2.5                42               1840   
3               1.0                1.0                35               1512   
4               2.5                2.5               276               2290   
...             ...                ...               ...                ...   
1455            2.5                2.5                40               1906   
1456            2.0                2.0               349               3084   
1457            2.0                2.0                60               2304   
1458            1.0                1.0               366               2156   
1459            1.5                1.5               804               2512   

      SalePrice  
0        208500  
1        181500  
2        223500  
3        140000  
4        250000  
...         ...  
1455     175000  
1456     210000  
1457     266500  
1458     142125  
1459     147500  

[1460 rows x 5 columns]
per_df = per_df.loc[:, ['TotalBathArea', 'TotalBathroomArea','TotalOutdoorArea','TotalBasementArea','SalePrice']]
print(per_df)
      TotalBathArea  TotalBathroomArea  TotalOutdoorArea  TotalBasementArea  \
0               2.5                2.5                61               1712   
1               2.0                2.0               298               2524   
2               2.5                2.5                42               1840   
3               1.0                1.0                35               1512   
4               2.5                2.5               276               2290   
...             ...                ...               ...                ...   
1455            2.5                2.5                40               1906   
1456            2.0                2.0               349               3084   
1457            2.0                2.0                60               2304   
1458            1.0                1.0               366               2156   
1459            1.5                1.5               804               2512   

      SalePrice  
0        208500  
1        181500  
2        223500  
3        140000  
4        250000  
...         ...  
1455     175000  
1456     210000  
1457     266500  
1458     142125  
1459     147500  

[1460 rows x 5 columns]
filled_train_df = filled_train_df.loc[:, ['TotalBathArea', 'TotalBathroomArea','TotalOutdoorArea','TotalBasementArea','SalePrice']]
print(filled_train_df)
      TotalBathArea  TotalBathroomArea  TotalOutdoorArea  TotalBasementArea  \
0               2.5                2.5                61               1712   
1               2.0                2.0               298               2524   
2               2.5                2.5                42               1840   
3               1.0                1.0                35               1512   
4               2.5                2.5               276               2290   
...             ...                ...               ...                ...   
1455            2.5                2.5                40               1906   
1456            2.0                2.0               349               3084   
1457            2.0                2.0                60               2304   
1458            1.0                1.0               366               2156   
1459            1.5                1.5               804               2512   

      SalePrice  
0        208500  
1        181500  
2        223500  
3        140000  
4        250000  
...         ...  
1455     175000  
1456     210000  
1457     266500  
1458     142125  
1459     147500  

[1460 rows x 5 columns]
filled_test_df
MSSubClass	LotFrontage	LotArea	OverallQual	OverallCond	YearBuilt	YearRemodAdd	MasVnrArea	BsmtFinSF1	BsmtFinSF2	...	GarageFinish_encoded	GarageQual_encoded	GarageCond_encoded	PavedDrive_encoded	SaleType_encoded	SaleCondition_encoded	TotalBathArea	TotalBathroomArea	TotalOutdoorArea	TotalBasementArea
0	20	80.0	11622	5	6	1961	1961	0.0	468.0	144.0	...	0.452571	0.936278	0.961622	0.891707	0.862826	0.825223	1.0	1.0	260	1764.0
1	20	81.0	14267	6	6	1958	1958	108.0	923.0	0.0	...	0.452571	0.936278	0.961622	0.891707	0.862826	0.825223	1.5	1.5	429	2658.0
2	60	74.0	13830	5	5	1997	1998	0.0	791.0	0.0	...	0.265749	0.936278	0.961622	0.891707	0.862826	0.825223	2.5	2.5	246	1856.0
3	60	78.0	9978	6	6	1998	1998	20.0	602.0	0.0	...	0.265749	0.936278	0.961622	0.891707	0.862826	0.825223	2.5	2.5	396	1852.0
4	120	43.0	5005	8	5	1992	1992	0.0	263.0	0.0	...	0.281680	0.936278	0.961622	0.891707	0.862826	0.825223	2.0	2.0	226	2560.0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
1454	160	21.0	1936	4	7	1970	1970	0.0	0.0	0.0	...	0.433491	0.865288	0.922705	0.891707	0.862826	0.825223	1.5	1.5	0	1092.0
1455	160	21.0	1894	4	5	1970	1970	0.0	252.0	0.0	...	0.452571	0.936278	0.961622	0.891707	0.862826	0.061001	1.5	1.5	24	1092.0
1456	20	160.0	20000	5	7	1960	1996	0.0	1224.0	0.0	...	0.452571	0.936278	0.961622	0.891707	0.862826	0.061001	1.0	1.0	474	2448.0
1457	85	62.0	10441	5	5	1992	1992	0.0	337.0	0.0	...	0.370231	0.941486	0.901317	0.891707	0.862826	0.825223	1.0	1.0	112	1824.0
1458	60	74.0	9627	7	5	1993	1994	94.0	758.0	0.0	...	0.265749	0.936278	0.961622	0.891707	0.862826	0.825223	2.5	2.5	238	1992.0
1459 rows × 77 columns

filled_test_df = filled_test_df.loc[:, ['TotalBathArea', 'TotalBathroomArea','TotalOutdoorArea','TotalBasementArea']]
print(filled_test_df)
      TotalBathArea  TotalBathroomArea  TotalOutdoorArea  TotalBasementArea
0               1.0                1.0               260             1764.0
1               1.5                1.5               429             2658.0
2               2.5                2.5               246             1856.0
3               2.5                2.5               396             1852.0
4               2.0                2.0               226             2560.0
...             ...                ...               ...                ...
1454            1.5                1.5                 0             1092.0
1455            1.5                1.5                24             1092.0
1456            1.0                1.0               474             2448.0
1457            1.0                1.0               112             1824.0
1458            2.5                2.5               238             1992.0

[1459 rows x 4 columns]
def differ_df_ml(df_type):
    import pandas as pd
    import numpy as np
    X = df_type.iloc[:,:-1]
    y = df_type.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    
    k_fold = KFold(n_splits = 9, shuffle=True, random_state = 0)

    lr_model = LinearRegression()
    rf_regressor = RandomForestRegressor(n_estimators=1000, random_state=42)
    cb_model = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.1, loss_function='RMSE')

    lr_model.fit(X_train, y_train)
    rf_regressor.fit(X_train, y_train)
    cb_model.fit(X_train, y_train, verbose=100)

    lr_pred = lr_model.predict(X_test)
    rf_pred = rf_regressor.predict(X_test)
    cb_pred = cb_model.predict(X_test)
        
    lr_mse = mean_squared_error(y_test, lr_pred)
    rf_mse = mean_squared_error(y_test, rf_pred)
    cb_mse = mean_squared_error(y_test, cb_pred)
    print(f'CB - RMSE: {cb_mse**0.5:.2f}')
    print(f'RF - RMSE: {rf_mse**0.5:.2f}')
    print(f'LR - RMSE: {lr_mse**0.5:.2f}')
differ_df_ml(filled_train_df)
0:	learn: 73689.5934678	total: 3.44ms	remaining: 3.44s
100:	learn: 33834.4348869	total: 121ms	remaining: 1.08s
200:	learn: 29575.6460715	total: 234ms	remaining: 930ms
300:	learn: 26537.8464292	total: 357ms	remaining: 830ms
400:	learn: 24203.3978912	total: 472ms	remaining: 704ms
500:	learn: 22353.7762103	total: 586ms	remaining: 584ms
600:	learn: 20981.5711169	total: 744ms	remaining: 494ms
700:	learn: 19742.7319990	total: 887ms	remaining: 378ms
800:	learn: 18690.6662202	total: 1.05s	remaining: 262ms
900:	learn: 17795.4600301	total: 1.18s	remaining: 129ms
999:	learn: 17012.1390735	total: 1.3s	remaining: 0us
CB - RMSE: 45679.37
RF - RMSE: 42800.79
LR - RMSE: 50626.54
differ_df_ml(per_df)
0:	learn: 73689.5934678	total: 1.93ms	remaining: 1.93s
100:	learn: 33834.4348869	total: 116ms	remaining: 1.03s
200:	learn: 29575.6460715	total: 228ms	remaining: 907ms
300:	learn: 26537.8464292	total: 374ms	remaining: 868ms
400:	learn: 24203.3978912	total: 493ms	remaining: 736ms
500:	learn: 22353.7762103	total: 607ms	remaining: 604ms
600:	learn: 20981.5711169	total: 736ms	remaining: 488ms
700:	learn: 19742.7319990	total: 855ms	remaining: 365ms
800:	learn: 18690.6662202	total: 981ms	remaining: 244ms
900:	learn: 17795.4600301	total: 1.1s	remaining: 121ms
999:	learn: 17012.1390735	total: 1.24s	remaining: 0us
CB - RMSE: 45679.37
RF - RMSE: 42800.79
LR - RMSE: 50626.54
differ_df_ml(new_df)
0:	learn: 73689.5934678	total: 1.39ms	remaining: 1.39s
100:	learn: 33834.4348869	total: 109ms	remaining: 974ms
200:	learn: 29575.6460715	total: 222ms	remaining: 882ms
300:	learn: 26537.8464292	total: 386ms	remaining: 897ms
400:	learn: 24203.3978912	total: 523ms	remaining: 782ms
500:	learn: 22353.7762103	total: 633ms	remaining: 630ms
600:	learn: 20981.5711169	total: 751ms	remaining: 499ms
700:	learn: 19742.7319990	total: 894ms	remaining: 381ms
800:	learn: 18690.6662202	total: 1.02s	remaining: 253ms
900:	learn: 17795.4600301	total: 1.14s	remaining: 125ms
999:	learn: 17012.1390735	total: 1.25s	remaining: 0us
CB - RMSE: 45679.37
RF - RMSE: 42800.79
LR - RMSE: 50626.54
new_X = new_df.iloc[:,:-1]
new_y = new_df.iloc[:,-1]

per_X = per_df.iloc[:,:-1]
per_y = per_df.iloc[:,-1]

ft_X = filled_train_df.iloc[:,:-1]
ft_y = filled_train_df.iloc[:,-1]
new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(new_X, new_y, test_size=0.2, random_state=42)
per_X_train, per_X_test, per_y_train, per_y_test = train_test_split(per_X, per_y, test_size=0.2, random_state=42)
ft_X_train, ft_X_test, ft_y_train, ft_y_test = train_test_split(ft_X, ft_y, test_size=0.2, random_state=42)
new_rf_regressor = RandomForestRegressor(n_estimators=1000, random_state=42)
ft_rf_regressor = RandomForestRegressor(n_estimators=1000, random_state=42)
per_cb_model = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.1, loss_function='RMSE')

new_rf_regressor.fit(new_X_train, new_y_train)
ft_rf_regressor.fit(ft_X_train, ft_y_train)
per_cb_model.fit(per_X_train, per_y_train, verbose=100)

new_rf_pred = new_rf_regressor.predict(new_X_test)
ft_rf_pred = ft_rf_regressor.predict(ft_X_test)
per_cb_pred = per_cb_model.predict(per_X_test)
0:	learn: 72901.7636123	total: 1.67ms	remaining: 1.67s
100:	learn: 34138.8587279	total: 115ms	remaining: 1.02s
200:	learn: 29703.2323108	total: 232ms	remaining: 922ms
300:	learn: 26688.6594687	total: 361ms	remaining: 839ms
400:	learn: 24611.9466432	total: 494ms	remaining: 738ms
500:	learn: 22740.2732663	total: 635ms	remaining: 632ms
600:	learn: 21356.1683972	total: 758ms	remaining: 503ms
700:	learn: 20232.5333008	total: 872ms	remaining: 372ms
800:	learn: 19207.8046707	total: 998ms	remaining: 248ms
900:	learn: 18307.8429462	total: 1.12s	remaining: 123ms
999:	learn: 17604.2621796	total: 1.23s	remaining: 0us
name_list = ["new_rf","ft_rf","per_cb"]
ml_list = [new_rf_pred,ft_rf_pred,per_cb_pred]
test_list = [new_y_test,ft_y_test,per_y_test]
for i in range(len(ml_list)):
    x = list(range(len(ml_list[i])))
    y1 = ml_list[i]
    y2 = test_list[i]

    plt.figure(figsize=(20,8))
    plt.scatter(x, y1, label='ML pred Price',color="red")
    plt.plot(x, y2, label='Real Price')

    plt.title(name_list[i])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    plt.show()



 
