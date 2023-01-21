import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Task 1 : Exploratory Data Analysis

df= pd.read_csv("WEEK_6/ÖDEVLER/TELCO/Telco-Customer-Churn.csv")
df.head()

# Step 1: General Analysis

def check_df(dataframe, head=5):
    print("############### shape #############")
    print(dataframe.shape)
    print("############### types #############")
    print(dataframe.dtypes)
    print("############### head #############")
    print(dataframe.head())
    print("############### tail #############")
    print(dataframe.tail())
    print("############### NA #############")
    print(dataframe.isnull().sum())
    print("############### Quantiles #############")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


#Step 2: Finding numerical and categorical variables

#df["TotalCharges"] = df["TotalCharges"].astype(float)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat


cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

print(f"cat_cols: {cat_cols}")
print(f"num_cols: {num_cols}")
print(f"cat_but_car: {cat_but_car}")
print(f"num_but_cat: {num_but_cat}")


# Step 3:  Analyzing of the numeric and categorical variables.

def cat_summary(dataframe, col_name, plot=False):  # create plot graph
    if df[col_name].dtypes == "bool":
        df[col_name] = df[col_name].astype(int)

        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        if plot:  # meaning that plot is true
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)  # birden fazla döngüye girdiğimizde görsel olacağından dolayı block argümanına true dedik.

    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col)
    
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:  # meaning that plot is true
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=False)


# Step 4: Analyzing target variable. (meaning of the target variable by categorical variables, meaning of numerical variables by target variables)

# churn is converting into 0-1 binary.

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

label_encoder(df, "Churn")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN":dataframe.groupby(categorical_col)[target].mean()}))

for col in cat_cols:
    target_summary_with_cat(df,"Churn",col)
    

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}),end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)
    
    
# Step 5: Outlier Analyzing

#How to find ?
#1. industry knowledge
#2. standard deviation approach
#3.z-score approach
#4.boxplot(interquantile range -IQR) Method =>
#5. lof yöntemi => multiple variables method

###################
# Outlier values by graphical methods.
###################

sns.boxplot(x=df["tenure"])
plt.show()


def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

# if it was outlier and we would use below code to access it.

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] >10: #eğer 10 dan fazla aykırı değer varse, head yapıyoruz.
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

#lof
df_float_num = df.select_dtypes(include=['float64', 'int64'])
df_float_num.dropna(inplace=True)
clf = LocalOutlierFactor(n_neighbors=20) # the number of neighbours is 20


clf.fit_predict(df_float_num)
df_scores = clf.negative_outlier_factor_

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()


th = np.sort(df_scores)[4] #rakamlar değişebilir.
df_float_num[df_scores < th].shape
df_float_num[df_scores < th]

lof_index = df_float_num[df_scores < th].index
df.loc[lof_index] # find local outlier

# Step 6: Missing observation analysis.
df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0] 
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False) 
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio']) #birleştirme
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

#msno.matrix(df)
#plt.show() # değelerin birbiri arasında ilişkisini verir, örneğin bir değer boşken diğeride boş mu? Çizgiler ile takip ederiz.

#msno.heatmap(df)
#plt.show()

###################
# Examining the Relationship of Missing Values ​​with the Dependent Variable
###################


na_cols = missing_values_table(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0) # boşluk var ise 1 yok ise 0 yaz.

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Churn", na_cols)


# Steo 7: Correlation Analysis.

corr = df[num_cols].corr()

f, ax = plt.subplots(figsize=[10, 10])
sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# Task 2: Feature Engineering

# Steo 1:  Take necessary actions for missing and outlier values.

df["TotalCharges"] = df["TotalCharges"].fillna(df.groupby("gender")["TotalCharges"].transform("mean"))

#eksik değerleri mod veya medyan ile, kategorik kırınımına göre medyan veya knn ile doldurabilirdik veya direkt silebilirdik.

#aykırı değerleri silebilirdik, baskılayabilirdik.

#local-outlier olsaydı

#belki baskılama yapılabilir, ucundan kırpılabilir. eğer 500 tane de 2 tane varsa silin gitsin,
#500 tanede 50-100 tane varsa baskılayın.

#ağaç yöntemi kullanıyorsak aykırı değerlere hiç dokunmuyoruz, illa dokuncasan ucundan dokuncaz tıraşlıcaz.
# doğrusal kullanıyorsak ya sileceğiz yada tek değişken kullanıp baskılayacağız.


# Step 2: Creating new variables.


def totalCharges_seperated (total_charges):
    if total_charges <= 3000:
        return 'Total Charges Beetween 0 and 3000'
    elif total_charges > 3000 and total_charges <=6000:
        return 'Total Charges Beetween 3000 and 6000'
    else:
        return 'Total Charges greater than 6000'
    
df['seperated_total_charges'] = df['TotalCharges'].apply(totalCharges_seperated)

def yearly_tenure(tenure):
  if tenure <= 12:
     return 1
  elif tenure >12 and tenure <=24:
      return 2
  elif tenure >24 and tenure <=36:
     return 3
  elif tenure >36 and tenure <=48:
     return 4
  elif tenure >48 and tenure <=60:
     return 5
  elif tenure > 60 and tenure <=72:
     return 6

df['yearly_tenure'] = df['tenure'].apply(yearly_tenure)


def Monthly_plans(monthly_charge):
    if monthly_charge <= 30:
        return 'Basic Plan'
    elif monthly_charge >30 and monthly_charge <= 60:
        return 'Advanced Plan'
    elif monthly_charge >60 and monthly_charge <= 100:
        return 'Premium Plan'
    elif monthly_charge >100:
        return 'Executive Plan'
    
df['PlanType'] = df['MonthlyCharges'].apply(Monthly_plans)


# Step 3:  Encoding operations

#binary encoder

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)
    
#rare encoder

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız.

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "Churn", cat_cols)


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

df = rare_encoder(df, 0.01)


#one-hot encoder
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]


df = one_hot_encoder(df, ohe_cols)

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df) # yeni değişken türettik.

print(f"cat_cols: {cat_cols}")
print(f"num_cols: {num_cols}")
print(f"cat_but_car: {cat_but_car}")

# Step 4: Standardize for numeric variables.

scaler = MinMaxScaler() # değerleri 1 ile 0 'a dönüştür yapmayı sağlıyor.
df[num_cols]= pd.DataFrame(scaler.fit_transform(df[num_cols]), columns = df[num_cols].columns)
df.head()

# Step 5: Creating a model

y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=20)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# imprtance of variables.

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)
