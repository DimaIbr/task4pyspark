from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, split, avg, explode, when
from pyspark.sql.types import StringType, FloatType, BooleanType
from pyspark import SparkConf, SparkContext, SQLContext

import re



def upperCase(str):
    text = re.findall(pattern='[A-Za-zА-Яа-я]+', string=str)
    text = ' '.join(text)
    return text

def latin(str):
    text = re.findall(pattern='[A-Za-z]+', string=str)
    text = ' '.join(text)
    return text

# Создание SparkSession
spark = SparkSession.builder.appName("LongestWord").getOrCreate()


# Определение функции для нахождения самого длинного слова
def find_longest_word(text):
    text=text.split('\t')[-1]
    words = text.split(" ")
    longest_word = ""
    for word in words:
        if len(word) > len(longest_word):
            longest_word = word
    return longest_word

def word_length(text):
    words = text.split(" ")
    summ=0
    for i in words:
        summ+=len(i)
    return summ/len(words)

# Регистрация UDF
find_longest_word_udf = udf(find_longest_word, StringType())

# Загрузка текстового файла в DataFrame
text_df = spark.read.text("wiki.txt")

upperCaseUDF = udf(lambda z:upperCase(z),StringType())
text_df = text_df.withColumn("split_col", split(text_df["value"], "\t"))
text_df = text_df.withColumn("col1", text_df["split_col"][0])
text_df = text_df.withColumn("col2", text_df["split_col"][1])
text_df = text_df.withColumn("col3", text_df["split_col"][2])
text_df=text_df.withColumn("refactor_col", upperCaseUDF(col("col3")))
text_df = text_df.drop("value", "split_col")
# text_df.show()
# 1. Применение UDF для нахождения самого длинного слова
result = text_df.select(find_longest_word_udf("refactor_col").alias("longest_word")).agg({"longest_word": "max"}).collect()[0][0]
print("Самое длинное слово:", result)

# 2. Средняя длинна слова
word_length_udf = udf(word_length, StringType())
text_df = text_df.withColumn("word_length", word_length_udf("refactor_col"))
#text_df.show()
result=text_df.select(avg("word_length").cast(FloatType())).collect()[0][0]
print("Средняя длина слова:", result)


# 3. Нахождение самого частоупотребляемого латинского слова
latinUDF = udf(lambda z: latin(z), StringType())
text_df = text_df.withColumn("latin", latinUDF(col("col3")))

df = text_df.select(explode(split(col("latin"), " ")).alias("word"))

word_counts = df.groupBy("word").count()
word_counts = word_counts.filter(col('word') != "")

most_frequent_word = word_counts.orderBy(col("count").desc())
#most_frequent_word.show()

print("Самое частоупотребляемое слово:", most_frequent_word.select("word").first()[0])


# 4.Все слова, которые более чем в половине случаев начинаются с большой буквы и встречаются больше 10 раз.
def lowerCase(str):
    return str.lower()

def IsUpper(str):
    return str[0].isupper()

df = text_df.select(explode(split(col("refactor_col"), " ")).alias("word"))

lowerCaseUDF = udf(lambda z: lowerCase(z), StringType())

IsUpperUDF = udf(lambda z: IsUpper(z), BooleanType())

df = df.groupBy("word").count()
df = df.withColumn("IsUpper", IsUpperUDF(col("word")))
df = df.withColumn("UpperCount", when(col("IsUpper"), col("count")).otherwise(0))
df = df.withColumn("Lower", lowerCaseUDF(col("word")))
df = df.groupBy("Lower").sum("count","UpperCount")
df = df.filter(((col("sum(count)") >= 10) & (col("sum(UpperCount)")/col("sum(count)")>0.5)))
df.show()

# 5.Напишите программу, которая с помощью статистики определяет устойчивые сокращения вида пр., др., ...
def cutsOneCase(str):
    text = re.findall(pattern='[ ][A-Za-zА-Яа-я][a-zа-я][\.]', string=str)
    text = ' '.join(text)
    return text

cutsOneCaseUDF=udf(lambda z:cutsOneCase(z),StringType())
cutsOne_df = text_df.withColumn("Cuts", cutsOneCaseUDF(col("col3")))
#cutsOne_df.show()
cutsOne_df = cutsOne_df.select(explode(split(col("Cuts"), " ")).alias("word"))
cutsOne_df = cutsOne_df.groupBy("word").count()
cutsOne_df = cutsOne_df.orderBy(col("count"))
cutsOne_df = cutsOne_df.filter(col('word') != "")
cutsOne_df = cutsOne_df.filter(col("count") > (cutsOne_df.select(avg("count").cast(FloatType())).collect()[0][0]))
cutsOne_df.show()


# 6.Напишите программу, которая с помощью статистики определяет устойчивые сокращения вида т.п., н.э., ...
def cutsCase(str):
    text = re.findall(pattern='[A-Za-zА-Яа-я][\.][a-zа-я][\.]', string=str)
    text = ' '.join(text)
    return text

cutsCaseUDF = udf(lambda z: cutsCase(z), StringType())
cuts_df = text_df.withColumn("Cuts", cutsCaseUDF(col("col3")))
#cuts_df.show()
cuts_df = cuts_df.select(explode(split(col("Cuts"), " ")).alias("word"))
cuts_df = cuts_df.groupBy("word").count()
cuts_df = cuts_df.orderBy(col("count"))
cuts_df = cuts_df.filter(col('word') != "")
cuts_df = cuts_df.filter(col("count") > (cuts_df.select(avg("count").cast(FloatType())).collect()[0][0]))
cuts_df.show()