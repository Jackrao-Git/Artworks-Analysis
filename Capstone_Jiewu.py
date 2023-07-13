import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import scipy.stats as stats
from sklearn import tree # this is needed to run the decisiontree regression model



warnings.filterwarnings('ignore')
N = random.seed('N18043416')

# load data
Art_data = pd.read_csv('theArt.csv')
data = pd.read_csv('theData.csv', header=None)

# after observing the data, I decide to fill NaN data with its median.
data.fillna(data.median(), inplace=True)

# 1. Is classical art more well liked than modern art?
print('Q1: Is classical art more well liked than modern art?')
classical_index = Art_data[Art_data.loc[:, 'Source (1 = classical, 2 = modern, 3 = nonhuman)'] == 1].index.tolist()
modern_index = Art_data[Art_data.loc[:, 'Source (1 = classical, 2 = modern, 3 = nonhuman)'] == 2].index.tolist()
print(f'There are {len(classical_index)} classical arts.')
print(f'There are {len(modern_index)} modern arts.')

# integrate all preference ratings for both types of art into two lists
classical_lik, modern_lik = [], []
for ind in classical_index:
    classical_lik += data.loc[:, ind].values.tolist()
for ind in modern_index:
    modern_lik += data.loc[:, ind].values.tolist()
print(f'There are {len(classical_lik)} preference ratings for classical art.')
print(f'There are {len(modern_lik)} preference ratings for modern art.\n')

# implement Mann-Whitney U test for both liking lists
# null hypothesis: mean(classical_lik) <= mean(modern_lik)
# alternative hypothesis: mean(classical_lik) > mean(modern_lik)
print(f"p-value of U-test = {stats.mannwhitneyu(classical_lik, modern_lik, alternative='greater').pvalue} < 0.05,")
print(f"meaning that under the confidence level of 95%, ")
print(f"we can reject the null hypothesis: mean(classical_lik) <= mean(modern_lik).")
print('From U-test result, a conclusion can be drawn that classical art is more well liked than modern art.\n')

# check the mean and median difference of two types of art
print(f'The mean difference of likings between classical art and modern art = {sum(classical_lik)/len(classical_lik) - sum(modern_lik)/len(modern_lik)}')
print(f'The median difference of likings between classical art and modern art = {np.median(np.array(classical_lik)) - np.median(np.array(modern_lik))}')
print('From the mean and median difference above, the conclusion drawn from U-test can be reconfirmed.\n')

# visualize the results
name = ["classical art","modern art"]
classical_mean = sum(classical_lik)/len(classical_lik)
modern_mean = sum(modern_lik)/len(modern_lik)
ratings_list =[classical_mean,modern_mean]
plt.bar(name, ratings_list)
plt.ylabel("mean ratings")
plt.title("classical art vs modern art (rating)")
plt.show()





# 2. Is there a difference in the preference ratings for modern art vs. non-human (animals and computers) generated art?
print('Q2: Is there a difference in the preference ratings for modern art vs. non-human (animals and computers) generated art?')
nonhuman_index = Art_data[Art_data.loc[:, 'Source (1 = classical, 2 = modern, 3 = nonhuman)'] == 3].index.tolist()
print(f'There are {len(modern_index)} modern arts.')
print(f'There are {len(nonhuman_index)} nonhuman arts.')

nonhuman_lik = []
for ind in nonhuman_index:
    nonhuman_lik += data.loc[:, ind].values.tolist()
print(f'There are {len(modern_lik)} preference ratings for modern art.')
print(f'There are {len(nonhuman_lik)} preference ratings for nonhuman art.\n')

# implement Mann-Whitney U test for both liking lists
# null hypothesis: mean(nonhuman_lik) = mean(modern_lik)
# alternative hypothesis: mean(nonhuman_lik) ≠ mean(modern_lik)
print(f"p-value of U-test = {stats.mannwhitneyu(nonhuman_lik, modern_lik, alternative='two-sided').pvalue} < 0.05,")
print(f"meaning that under the confidence level of 95%, ")
print(f"we can reject the null hypothesis: mean(nonhuman_lik) = mean(modern_lik).")
print('From U-test result, a conclusion can be drawn that nonhuman art is not similarly liked as modern art.\n')

# check the mean and median difference of two types of art
print(f'The mean difference of likings between nonhuman art and modern art = {sum(nonhuman_lik)/len(nonhuman_lik) - sum(modern_lik)/len(modern_lik)}')
print(f'The median difference of likings between nonhuman art and modern art = {np.median(np.array(nonhuman_lik)) - np.median(np.array(modern_lik))}')
print('From the mean and median difference above, the conclusion drawn from U-test can be reconfirmed.\n')

# visualize the results
name = ["non human art","modern art"]
non_human_mean = sum(nonhuman_lik)/len(nonhuman_lik)
ratings_list =[non_human_mean,modern_mean]
plt.bar(name, ratings_list)
plt.ylabel("mean ratings")
plt.title("non human art vs modern art (rating)")
plt.show()










# 3. Do women give higher art preference ratings than men?
print('Q3: Do women give higher art preference ratings than men?')
# theData.csv Column 217: User gender (1 = male, 2 = female, 3 = non-binary)
male_lik_df = data.iloc[:, :91][data.loc[:, 216] == 1]
female_lik_df = data.iloc[:, :91][data.loc[:, 216] == 2]
print(f'There are {male_lik_df.shape[0]} male raters.')
print(f'There are {female_lik_df.shape[0]} female raters.\n')

# get male and female preference ratings list
male_lik, female_lik = [], []
for cols in male_lik_df.columns:
    male_lik += male_lik_df.loc[:, cols].values.tolist()
for cols in female_lik_df.columns:
    female_lik += female_lik_df.loc[:, cols].values.tolist()

# implement Mann-Whitney U test for both liking lists
# null hypothesis: mean(female_lik) <= mean(male_lik)
# alternative hypothesis: mean(female_lik) > mean(male_lik)
print(f"p-value of U-test = {stats.mannwhitneyu(female_lik, male_lik, alternative='greater').pvalue} > 0.05,")
print(f"meaning that under the confidence level of 95%, ")
print(f"we fail to reject the null hypothesis: mean(female_lik) <= mean(male_lik).")

# Since p-value is greater than 0.05, a U-test with alternative hypothesis mean(female_lik) ≠ mean(male_lik) should be
# conducted to further check whether women give higher ratings than men.
# null hypothesis: mean(female_lik) = mean(male_lik)
# alternative hypothesis: mean(female_lik) ≠ mean(male_lik)
print(f"p-value of U-test = {stats.mannwhitneyu(female_lik, male_lik, alternative='two-sided').pvalue} > 0.05,")
print(f"meaning that under the confidence level of 95%,")
print(f"we fail to reject the null hypothesis: mean(female_lik) = mean(male_lik).")
print('From U-test result, it can be concluded that women DO NOT give higher art preference ratings than men.\n')

# check the mean and median difference of preference ratings of women and men
print(f'The mean difference of likings between women and men = {sum(female_lik)/len(female_lik) - sum(male_lik)/len(male_lik)}')
print(f'The median difference of likings between women and men = {np.median(np.array(female_lik)) - np.median(np.array(male_lik))}')
print('From the mean and median difference above, the conclusion drawn from U-test can be reconfirmed.\n')

# visualize the results
men_mean = sum(male_lik)/len(male_lik)
women_mean = sum(female_lik)/len(female_lik)
name = ["men","women"]
ratings_list =[men_mean,women_mean]
plt.bar(name, ratings_list)
plt.ylabel("mean ratings")
plt.title("men vs women")
plt.show()







# 4. Is there a difference in the preference ratings of users with some art background (some art education) vs. none?
print('Q4: Is there a difference in the preference ratings of users with some art background (some art education) vs. none?')
# theData.csv Column 219: Art education (The higher the number, the more: 0 = none, 3 = years of art education)
artedu_df = data.iloc[:, :91][data.loc[:, 218] != 0]
non_artedu_df = data.iloc[:, :91][data.loc[:, 218] == 0]
print(f'There are {artedu_df.shape[0]} raters with some art education.')
print(f'There are {non_artedu_df.shape[0]} raters with no art education.\n')

# get art-educated and non-art-educated preference ratings list
artedu_lik, non_artedu_lik = [], []
for cols in artedu_df.columns:
    artedu_lik += artedu_df.loc[:, cols].values.tolist()
for cols in non_artedu_df.columns:
    non_artedu_lik += non_artedu_df.loc[:, cols].values.tolist()

# implement Mann-Whitney U test for both liking lists
# null hypothesis: mean(artedu_lik) = mean(non_artedu_lik)
# alternative hypothesis: mean(artedu_lik) ≠ mean(non_artedu_lik)
print(f"p-value of U-test = {stats.mannwhitneyu(artedu_lik, non_artedu_lik, alternative='two-sided').pvalue} < 0.05,")
print(f"meaning that under the confidence level of 95%, ")
print(f"we can reject the null hypothesis: mean(artedu_lik) = mean(non_artedu_lik).")
print('From U-test result, a conclusion can be drawn that people with art education rate differently from the ones without.\n')

# check the mean and median difference of two types of art
print(f'The mean difference of likings between people with art education and without = {np.mean(np.array(artedu_lik)) - np.mean(np.array(non_artedu_lik))}')
print(f'The median difference of likings between people with art education and without = {np.median(np.array(artedu_lik)) - np.median(np.array(non_artedu_lik))}')
print('From the mean and median difference above, the conclusion drawn from U-test can be reconfirmed.\n')

# visualize the results
some_education_mean = np.mean(np.array(artedu_lik))
no_education_mean = np.mean(np.array(non_artedu_lik))
name = ["with art background","no art background"]
ratings_list =[some_education_mean,no_education_mean]
plt.bar(name, ratings_list)
plt.ylabel("mean ratings")
plt.title("art background vs no art background")
plt.show()









## 5.Build a regression model to predict art preference ratings from energy ratings only. Make sure to use cross-validation methods to avoid overfitting and characterize how well your model predicts art preference ratings.

print('Q5: Build a regression model to predict art preference ratings from energy ratings only. Make sure to use cross-validation methods to avoid overfitting and characterize how well your model predicts art preference ratings.')

preference_ratings_data = data.iloc[:,:91].mean(axis=1).values
energy_ratings_data = data.iloc[:,91:182]

# Build a linear regression model and fit the model, the ratio between the train set and test set is 5:1 since we have 300 in total and 250 for training.
model_LinearRegression_5 = linear_model.LinearRegression()
scores_5 = cross_val_score(model_LinearRegression_5,
                           energy_ratings_data[:250],
                           preference_ratings_data[:250], 
                           scoring='neg_mean_squared_error',
                           cv=5)
print("Q5 cross validation mse result:",np.mean(scores_5))

model_LinearRegression_5.fit(energy_ratings_data[:250],preference_ratings_data[:250])
score_5 =model_LinearRegression_5.score(energy_ratings_data[:250],preference_ratings_data[:250])
print('The score of the linear regression model is ', score_5)
# This is the rSq, showing how much of the outcome the predictor can account for.

# visualize the results
plt.figure()
train_5=model_LinearRegression_5.predict(energy_ratings_data[:250])
pred_5 = model_LinearRegression_5.predict(energy_ratings_data[250:])
plt.plot(np.arange(len(preference_ratings_data[:250])),preference_ratings_data[:250],'go-',label='true value')
plt.plot(np.arange(len(preference_ratings_data[:250])),train_5,'yo-',label='train value')
plt.plot(np.arange(250,300),pred_5 ,'ro-',label='predict value')

plt.title('Q5 Linear regression score: %f'%score_5)
plt.legend()
plt.show()



## 6.Build a regression model to predict art preference ratings from energy ratings and demographic information. Make sure to use cross-validation methods to avoid overfitting and comment on how well your model predicts relative to the “energy ratings only” model.
print('Q6: Build a regression model to predict art preference ratings from energy ratings and demographic information. Make sure to use cross-validation methods to avoid overfitting and comment on how well your model predicts relative to the “energy ratings only” model.')

# create data x,y
y_6 = data.iloc[:,:91].mean(axis=1).values
x_6 = pd.concat([energy_ratings_data,data.iloc[:,215:217]],axis=1)

# Build a linear regression model and fit the model, the same ratio of 5:1. 
lr_6 = linear_model.LinearRegression()

scores_6 = cross_val_score(lr_6,
                           x_6[:250],
                           y_6[:250], 
                           scoring='neg_mean_squared_error', 
                           cv=5)
print("Q6 cross validation mse result:",np.mean(scores_6))

lr_6.fit(x_6[:250],y_6[:250])
score_6 = lr_6.score(x_6[:250],y_6[:250])
print('The score of the linear regression model is ', score_6)


# visualize the results
plt.figure()
pred_6 = lr_6.predict(x_6[250:])
train_6=lr_6.predict(x_6[:250])
plt.plot(np.arange(len(y_6[:250])),y_6[:250],'go-',label='true value')
plt.plot(np.arange(len(y_6[:250])),train_6,'yo-',label='train value')
plt.plot(np.arange(250,300),pred_6 ,'ro-',label='predict value')
plt.title('Q6 Linear regression score: %f'%score_6)
plt.legend()
plt.show()













## 7. Considering the 2D space of average preference ratings vs. average energy rating (that contains the 91 art pieces as elements), how many clusters can you – algorithmically - identify in this space? Make sure to comment on the identity of the clusters – do they correspond to particular types of art?
print('Q7: Considering the 2D space of average preference ratings vs. average energy rating (that contains the 91 art pieces as elements), how many clusters can you – algorithmically - identify in this space? Make sure to comment on the identity of the clusters – do they correspond to particular types of art?')
# create data——2D
preference_ratings_data = data.iloc[:,:91].mean().values
energy_ratings_data = data.iloc[:,91:182].mean().values

data_2D = {'average preference ratings':preference_ratings_data,
           'average energy rating':energy_ratings_data}
data_2D = pd.DataFrame(data_2D)
print('2D space of average preference ratings vs. average energy rating head 5 item')
print(data_2D.head())
# contour coefficient method to find n_clusters
inertia = []
for n in range(2 , 11):
    algorithm = (KMeans(n_clusters = n ,init='k-means++',max_iter=100, tol=0.001) )
    algorithm.fit(data_2D)
    inertia.append(algorithm.inertia_)
plt.figure(1 , figsize = (10 ,6))
plt.plot(np.arange(2 , 11) , inertia , 'o')
plt.plot(np.arange(2 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()


# KMeans model
algorithm = KMeans(n_clusters = 3 ,init='k-means++',max_iter=300, tol=0.001)
algorithm.fit(data_2D)
labels = algorithm.labels_
data_2D["pred label"] = labels
data_2D["real label-1"] = Art_data["Source (1 = classical, 2 = modern, 3 = nonhuman)"]
data_2D["real label-2"] = Art_data["computerOrAnimal (0 = human, 1 = computer, 2 = animal)"] 

# visualize the results
plt.figure(figsize=(12, 9), facecolor='w')
plt.subplot(311)
plt.scatter(data_2D["average preference ratings"].values,data_2D["average energy rating"].values, c=data_2D["pred label"].values, s=30)
plt.title(u'pred label')

plt.subplot(312)
plt.scatter(data_2D["average preference ratings"].values,data_2D["average energy rating"].values, c=data_2D["real label-1"].values, s=30)
plt.title(u'real label-1 (1 = classical, 2 = modern, 3 = nonhuman)')

plt.subplot(313)
plt.scatter(data_2D["average preference ratings"].values,data_2D["average energy rating"].values, c=data_2D["real label-2"].values, s=30)
plt.title(u'real label-2 computerOrAnimal (0 = human, 1 = computer, 2 = animal)')
plt.tight_layout()
plt.show()


# Each clustering predict results are therefore considered to correspond to a particular type of art.












# 8.Considering only the first principal component of the self-image ratings as inputs to a regression model – how well can you predict art preference ratings from that factor alone?
print('Q8: Considering only the first principal component of the self-image ratings as inputs to a regression model – how well can you predict art preference ratings from that factor alone?')

# create dataset
x8 = np.array(data.iloc[:, 205:215])
y8 = np.array(data.iloc[:, :91].mean(axis=1))

# Eliminate outliers
left = y8.mean()-3*y8.std()
right = y8.mean()+3*y8.std()
y_8 = y8[(left < y8) & (y8 < right)]
x_8 = x8[(left < y8) & (y8 < right)]
plt.boxplot(y_8)
plt.show()
plt.boxplot(y8)
plt.show()

zscoredata = stats.zscore(x_8)

pca=PCA(n_components=1).fit(zscoredata)
eigVals = pca.explained_variance_
loadindgs = pca.components_
rotatedata = pca.fit_transform(zscoredata)

varExplained = eigVals/sum(eigVals)*100
#This is going to be 100 since there is only n_component = 1 

# get the first principal component of the self-image ratings
x_train_8, x_test_8, y_train_8, y_test_8 = train_test_split(rotatedata, y_8, test_size=N,random_state=10)
# linear model
lr_8 = linear_model.LinearRegression()
lr_8.fit(x_train_8,y_train_8)
score_8 =lr_8.score(x_test_8, y_test_8)
print('The score of the linear regression model is ', score_8)


# visualize the results
plt.figure()
pred_8 = lr_8.predict(x_test_8)
plt.plot(np.arange(len(y_test_8)), y_test_8,'go-',label='true value')
plt.plot(np.arange(len(pred_8 )),pred_8 ,'ro-',label='predict value')
plt.title('Q8 Linear regression score: %f'%score_8)
plt.legend()
plt.show()






# 9.Consider the first 3 principal components of the “dark personality” traits – use these as inputs to a regression model to predict art preference ratings. Which of these components significantly predict art preference ratings? Comment on the likely identity of these factors (e.g. narcissism, manipulativeness, callousness, etc.).
print('Q9: Consider the first 3 principal components of the “dark personality” traits – use these as inputs to a regression model to predict art preference ratings. Which of these components significantly predict art preference ratings? Comment on the likely identity of these factors (e.g. narcissism, manipulativeness, callousness, etc.)')
# create dataset
y_9 = y_8

zscoredata = stats.zscore(data)
x9 = zscoredata.iloc[:,182:194]

# Eliminate outliers
x_9 = x9[(left < y8) & (y8 < right)]

pca_2=PCA(n_components=3)

pca_2.fit(x_9)
newData_dark=pca_2.fit_transform(x_9)

eigVals_2 = pca_2.explained_variance_
loadings_2 = pca_2.components_
varExplained_2 = eigVals_2/sum(eigVals_2)*100
# get the first principal component of the self-image ratings
x_train_9, x_test_9, y_train_9, y_test_9 = train_test_split(newData_dark, y_9, test_size=N, random_state=random.randint(0, 2 ^ 32))
# linear regression model
lr_9 = linear_model.LinearRegression()
lr_9.fit(x_train_9,y_train_9)
score_9 =lr_9.score(x_test_9, y_test_9)
print('The score of the linear regression model is ', score_9)
'''
#decision tree regression model

lr_9 = tree.DecisionTreeRegressor(min_samples_leaf=int(0.01*len(x_6)), max_depth=100)
lr_9.fit(x_train_9,y_train_9)
score_9 =lr_9.score(x_test_9, y_test_9)
print('The score of the decisiontree regression model is ', score_9)
'''

# visualize the results
plt.figure()
pred_9 = lr_9.predict(x_test_9)
plt.plot(np.arange(len(y_test_9)), y_test_9,'go-',label='true value')
plt.plot(np.arange(len(pred_9 )),pred_9 ,'ro-',label='predict value')
plt.title('Q9 Linear regression score: %f'%score_9)
plt.legend()
plt.show()



for i in 0,1,2:
    print(np.corrcoef(newData_dark[:,i], y_9)[0,1])








# 10.Can you determine the political orientation of the users (to simplify things and avoid gross class imbalance issues, you can consider just 2 classes: “left” (progressive & liberal) vs. “non- left” (everyone else)) from all the other information available, using any classification model of your choice? Make sure to comment on the classification quality of this model.
print('Q10: Can you determine the political orientation of the users (to simplify things and avoid gross class imbalance issues, you can consider just 2 classes: “left” (progressive & liberal) vs. “non- left” (everyone else)) from all the other information available, using any classification model of your choice? Make sure to comment on the classification quality of this model.')
# 1 is non- left , 0 is left 
def left_or_not_map(x):
    political = 1 if x > 2  else 0
    return political
y_10 = data.iloc[:,217].map(left_or_not_map)
x_10 = data.iloc[:,182:217]
X_train_10,X_test_10,y_train_10,y_test_10=train_test_split(x_10,y_10,test_size=N, random_state=random.randint(0, 2 ^ 32))

# Random Forest Algorithm
random_forest = RandomForestClassifier(random_state=10)
random_forest.fit(X_train_10,y_train_10)
Y_pred = random_forest.predict(X_test_10)
print("Test set accuracy :",accuracy_score(y_test_10,Y_pred))
print(classification_report(y_test_10,Y_pred))














"""## Extra credit:

### something interesting
1. Intentionally created art works rating higher than unintentionally created art works
"""

print('Extra credit: The intentionally created art works rated higher than unintentionally created art works.')
no_Intent_index = Art_data[Art_data["Intent (0 = no, 1 = yes)"]==0].index.tolist()
Intent_index = Art_data[Art_data["Intent (0 = no, 1 = yes)"]==1].index.tolist()
print(f'There are {len(no_Intent_index)} unintentionally created arts.')
print(f'There are {len(Intent_index)} intentionally created arts.')


no_Intent_lik, Intent_lik = [], []
for ind in no_Intent_index:
    no_Intent_lik += data.loc[:, ind].values.tolist()
for ind in Intent_index:
    Intent_lik += data.loc[:, ind].values.tolist()
print(f'There are {len(classical_lik)} preference ratings for unintentionally created art.')
print(f'There are {len(modern_lik)} preference ratings for intentionally created art.\n')

# implement Mann-Whitney U test for both liking lists
# null hypothesis: mean(no_Intent_lik) <= mean(Intent_lik)
# alternative hypothesis: mean(no_Intent_lik) > mean(Intent_lik)
print(f"p-value of U-test = {stats.mannwhitneyu(no_Intent_lik, Intent_lik, alternative='greater').pvalue} > 0.05,")
print(f"meaning that under the confidence level of 95%, ")
print(f"we fail to reject the null hypothesis: mean(no_Intent_lik) <= mean(Intent_lik).")
print('From U-test result, a conclusion can be drawn that intentionally created art has a higher raing than unintentionally created art.\n')

# check the mean and median difference of two types of art
print(f'The mean difference of likings between intentionally created art and unintentionally created art = {sum(Intent_lik)/len(Intent_lik) - sum(no_Intent_lik)/len(no_Intent_lik)}')
print(f'The median difference of likings between intentionally created art and unintentionally created art = {np.median(np.array(Intent_lik)) - np.median(np.array(no_Intent_lik))}')
print('From the mean and median difference above, the conclusion drawn from U-test can be reconfirmed.\n')

# visualize the results
name = ["unintentionally created arts","intentionally created arts"]
no_Intent_mean = sum(no_Intent_lik)/len(no_Intent_lik)
Intent_mean = sum(Intent_lik)/len(Intent_lik)
ratings_list =[no_Intent_mean,Intent_mean]
plt.bar(name, ratings_list)
plt.ylabel("mean ratings")
plt.title("unintentionally created arts vs intentionally created arts (rating)")
plt.show()









