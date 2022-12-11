from pyspark.sql.session import SparkSession
from pyspark.sql.functions import *
from math import log
import sys
import re

class PopularTopic:
    def run(self, inputPath, outputPath, stopwords, k):
        spark = SparkSession.builder.master("local").appName("PopularTopic").getOrCreate()
        file = spark.sparkContext.textFile(inputPath)
        k = int(k)
        
        # make dataframe
        y_n = file.map(lambda line: line.split(","))
        pairs = y_n.map(lambda x: (x[0][0:4], x[1].split(" ")))
        pairs = pairs.map(lambda x: (x[0], list(set(x[1]))))
        pairs = pairs.flatMap(lambda x: map(lambda e: (x[0], e, ), x[1])).toDF()
        pairs = pairs.select(col("_1").alias("year"), col("_2").alias("word"))
        
        # convert stopwords to a list
        sw = spark.sparkContext.textFile(stopwords)
        sw = sw.collect()
        
        # filter out stopwords
        pairs = pairs[~ pairs.word.isin(sw)].toDF("year", "word")
        
        # count and join in one df
        # count headlines with term in a year
        res = pairs.groupBy("year","word").agg(count("word").alias("num_headlines"))
        # count num of years having term
        y_t = pairs.groupBy("word").agg(countDistinct("year").alias("years_having_t"))
        res = res.join(y_t, res.word == y_t.word, 'outer').select(res.year, res.word, res.num_headlines, y_t.years_having_t)
        # count num of years in dataset
        years = pairs.agg(countDistinct("year").alias("years_in_d"))
        res = res.join(years)
        
        # count result
        res = res.withColumn("ratio", round(res.num_headlines*log10(res.years_in_d/res.years_having_t), 6)).select("year","word","ratio")
        
        # sort
        res = res.orderBy(col("year"), -col("ratio"), col("word"))
        res = res.select("year", concat_ws(',', res.word, res.ratio).alias("popularity"))
        res = res.groupBy("year").agg(collect_list("popularity").alias("popularity"))
        # convert to rdd to map and convert back
        res = res.rdd.map(lambda x: (x[0], x[1][0:k]))
        res = res.map(lambda x: (f"{x[0]}"+"\t"+f"{';'.join([item for item in x[1]])}", ))
        res = res.toDF()
        res = res.orderBy(col("_1"))
        
        res.write.format("text").save(outputPath)
        spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Wrong inputs")
        sys.exit(-1)
    PopularTopic().run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
