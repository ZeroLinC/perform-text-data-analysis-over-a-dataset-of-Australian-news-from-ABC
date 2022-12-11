from pyspark import SparkContext, SparkConf
from math import log
import sys
import re

class PopularTopic:
    def run(self, inputPath, outputPath, stopwords, k):
        conf = SparkConf().setAppName("PopularTopic")
        sc = SparkContext(conf=conf)

        fp = sc.textFile(inputPath)
        
        # split the sencence into years and list of words
        y_n = fp.map(lambda line: line.split(","))
        pairs = y_n.map(lambda x: (x[0][0:4], x[1].split(" ")))
        pairs = pairs.map(lambda x: (x[0], list(set(x[1]))))
        
        # set stopwords as broadcast varibale
        sw = sc.textFile(stopwords)
        stopwords_local = set(sw.collect())
        stopwords_bc = sc.broadcast(stopwords_local)
        
        # map word in list of words to their year
        # like (("2003", "council"),1)
        pairs = pairs.flatMap(lambda x: map(lambda e: ((x[0], e),1), x[1]))
        # filter word in stopwords
        pairs = pairs.filter(lambda x: True if x[0][1] not in stopwords_bc.value else False)
        
        # count num of years in dataset by map the origin split with only years
        years = y_n.map(lambda x: x[0][0:4])
        num_of_years_in_d = years.distinct().count()
        
        # count num of years having term by counting distinct year-word pair
        y_t = pairs.map(lambda x: (x[0][1],x[0][0])).distinct()
        y_t = y_t.map(lambda x: (x[0], 1))
        y_t = y_t.reduceByKey(lambda x,y: x+y)
        
        # count headlines with term in a year
        res = pairs.reduceByKey(lambda x,y: x+y)
        res = res.map(lambda x: (x[0][1], (x[0][0], x[1])))
        
        # join res and y_t, not join num_of_years_in_d because this keeps the same
        res = res.leftOuterJoin(y_t)
        
        # count result
        res = res.map(lambda x: ((x[1][0][0], x[0]), round(x[1][0][1]*log(num_of_years_in_d/x[1][1],10), 6))).coalesce(1)
        res = res.map(lambda x: (x[0][0], (x[0][1],x[1])))
        
        # make the result into one year row
        res_y = res.map(lambda x: (x[0], set([x[1]])))
        res_y = res_y.reduceByKey(lambda x, y: x | y)
        
        # multiple levels sort and choose top k results
        res_y = res_y.map(lambda x: (x[0], sorted(x[1], key = lambda e: (-e[1], e[0]))))
        res_y = res_y.map(lambda x: (x[0], x[1][0:int(k)]))
        res_y = res_y.sortByKey()
        
        res_y = res_y.map(lambda x: f"{x[0]}"+"\t"+f"{';'.join([f'{item[0]},{item[1]}' for item in x[1]])}")
        
        res_y.saveAsTextFile(outputPath)
        sc.stop()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Wrong inputs")
        sys.exit(-1)
    PopularTopic().run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
