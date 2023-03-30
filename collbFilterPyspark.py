### Collaborative Filtering Based Recommender System using Pyspark ###

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
import pandas as pd

spark = SparkSession.builder.appName('movielens').getOrCreate()

df = spark.read.csv(r'C:\Users\MSI\Desktop\ds project\ratings.csv',header=True,inferSchema=True)

# Descriptive Stats
df.head(3)
df.describe().show()
df.na.drop(how='any')
(train,test) = df.randomSplit([0.75,0.25])

als = ALS(maxIter=10,regParam=0.16,userCol="userId",ratingCol="rating",itemCol="movieId",rank=16,
          numUserBlocks=30,numItemBlocks=30)

model = als.fit(train)

pred = model.transform(test)

eva = RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
rmse = eva.evaluate(pred.where("prediction != 'NaN'"))

print("RMSE score is : {}".format(rmse))

# Recommending movies to a given user based on collabarative filtering :
    
user_1 = test.filter(test["userId"]==1).select(["movieId","userId"])
recommend = model.transform(user_1)
pref = recommend.orderBy("prediction",ascending=False)

# Top 5 recommendations for USER 1
pref_pandas = pref.limit(5).toPandas()
movie_ids = list(pref_pandas.movieId)
pred_ratings = list(pref_pandas.movieId)

df_movies = spark.read.csv(r'C:\Users\MSI\Desktop\ds project\movies.csv',header=True,inferSchema=True)

movies = []
for i in movie_ids:
    movies.append(df_movies.filter("movieId = {}".format(i)).toPandas().title[0])
    
dic = {'movieID':movie_ids,'movieName':movies, 'predictedRating':pref_pandas.prediction}
output = pd.DataFrame(dic)
print(output)
