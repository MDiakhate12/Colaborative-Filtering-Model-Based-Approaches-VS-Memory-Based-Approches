<div class="cell markdown" data-colab_type="text" id="view-in-github">

<a href="https://colab.research.google.com/github/MDiakhate12/Colaborative-Filtering-Model-Based-Approaches-VS-Memory-Based-Approches/blob/main/Projet_Transversal_Matrix_Factorization.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

</div>

<div class="cell code" id="_Wp2p0AGr_Nr">

``` python
!curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
!unzip awscliv2.zip
!sudo ./aws/install

!aws --version 
```

</div>

<div class="cell code" data-colab="{&quot;height&quot;:85,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="M9Y6uzS1r_Nv" data-outputId="6e9ddd17-b6bc-49d5-a606-8228fa33933d">

``` python
#AKIAQBKLXKEALPR7EXOO
#J5Rh5E1Y5galwVym3mYuHPySrp/wU39pIRMW1hOz
!aws configure --output json --region us-west-2

!aws s3 cp s3://amazon-reviews-pds/tsv/amazon_reviews_us_Electronics_v1_00.tsv.gz ./
```

<div class="output stream stdout">

``` 
AWS Access Key ID [None]: AKIAQBKLXKEALPR7EXOO
AWS Secret Access Key [None]: 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
Default region name [None]: 
Default output format [None]: 
```

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:258,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="xst0KUykr_Ny" data-outputId="e022a58b-6c32-4cf1-b894-308d2b6e2aa3">

``` python
!pip install pyspark
```

<div class="output stream stdout">

    Collecting pyspark
    e=pyspark-3.0.0-py2.py3-none-any.whl size=205044182 sha256=533e7070666011b7ff5daed33a64efe9b07d8cbf3360d1439893d3b88cf4bcb7
      Stored in directory: /root/.cache/pip/wheels/57/27/4d/ddacf7143f8d5b76c45c61ee2e43d9f8492fc5a8e78ebd7d37
    Successfully built pyspark
    Installing collected packages: py4j, pyspark
    Successfully installed py4j-0.10.9 pyspark-3.0.0

</div>

</div>

<div class="cell markdown" id="Ea7ceBhpGy0o">

# Introduction

</div>

<div class="cell markdown" id="H1oJzVwVGy0p">

Recommendation systems are in tons of things you interact with every
day. Amazon, Spotify, and Facebook are some of the biggest players, and
they're using all the data they can to suggest products that they think
you'll love.

<img src ="./trends.png" />

Some companies have teams of people collection, cleaning, and building
models around this data. However, with a few useful Python packages and
some great data from Amazon's customer review dataset, Were're going to
build a recommendation system with different collaborative filetring
methods.

</div>

<div class="cell markdown" id="3VP6RZq3Gy0r">

<hr>

# Data Preprocessing

</div>

<div class="cell markdown" id="lM3Ig067Gy0s">

## Loading Data

</div>

<div class="cell code" id="ZgxvRXu6Gy0t">

``` python
import pandas as pd
import sys
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
```

</div>

<div class="cell code" data-colab="{&quot;height&quot;:68,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="ujm1ZbjaGy0y" data-outputId="2ca5dadd-0a84-4d34-e244-f3a5bd07353d">

``` python
# create a spark session
# using local[*] to run Spark locally with as many worker threads as logical cores has your machine.
conf = SparkConf().setAppName("Projet Transversal Matrix Factorisation").setMaster("local[*]")
sc = SparkContext.getOrCreate(conf)
spark = SparkSession(sc)

# spark and python version
print(spark.version)
print(sys.version)
```

<div class="output stream stdout">

    3.0.0
    3.6.9 (default, Jul 17 2020, 12:50:27) 
    [GCC 8.4.0]

</div>

</div>

<div class="cell markdown" id="zzOiCBmBGy01">

### [DATA COLUMNS](https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt)

<pre>marketplace       - 2 letter country code of the marketplace where the review was written.
customer_id       - Random identifier that can be used to aggregate reviews written by a single author.
review_id         - The unique ID of the review.
product_id        - The unique Product ID the review pertains to. In the multilingual dataset the reviews
                    for the same product in different countries can be grouped by the same product_id.
product_parent    - Random identifier that can be used to aggregate reviews for the same product.
product_title     - Title of the product.
product_category  - Broad product category that can be used to group reviews 
                    (also used to group the dataset into coherent parts).
star_rating       - The 1-5 star rating of the review.
helpful_votes     - Number of helpful votes.
total_votes       - Number of total votes the review received.
vine              - Review was written as part of the Vine program.
verified_purchase - The review is on a verified purchase.
review_headline   - The title of the review.
review_body       - The review text.
review_date       - The date the review was written.
</pre>

\#\#\# DATA FORMAT Tab ('\\t') separated text file, without quote or
escape characters. First line in each file is header; 1 line corresponds
to 1 record.

</div>

<div class="cell code" id="b_xiktSYGy02">

``` python
path = 'amazon_reviews_us_Electronics_v1_00.tsv.gz'

df_plain = spark.read.csv(path, sep='\t', header=True)
df = df_plain.select("customer_id", "product_id", "star_rating","product_title","product_parent","review_date",)
```

</div>

<div class="cell code" data-colab="{&quot;height&quot;:241,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="qrYT6lGSGy05" data-outputId="e764ac22-cba2-4e25-a6ad-48d4d51f4ee7">

``` python
df_plain.show(5)
df_plain.count()
```

<div class="output stream stdout">

``` 
+-----------+-----------+--------------+----------+--------------+--------------------+----------------+-----------+-------------+-----------+----+-----------------+--------------------+--------------------+-----------+
|marketplace|customer_id|     review_id|product_id|product_parent|       product_title|product_category|star_rating|helpful_votes|total_votes|vine|verified_purchase|     review_headline|         review_body|review_date|
+-----------+-----------+--------------+----------+--------------+--------------------+----------------+-----------+-------------+-----------+----+-----------------+--------------------+--------------------+-----------+
|         US|   41409413|R2MTG1GCZLR2DK|B00428R89M|     112201306|yoomall 5M Antenn...|     Electronics|          5|            0|          0|   N|                Y|          Five Stars|       As described.| 2015-08-31|
|         US|   49668221|R2HBOEM8LE9928|B000068O48|     734576678|Hosa GPM-103 3.5m...|     Electronics|          5|            0|          0|   N|                Y|It works as adver...|It works as adver...| 2015-08-31|
|         US|   12338275|R1P4RW1R9FDPEE|B000GGKOG8|     614448099|Channel Master Ti...|     Electronics|          5|            1|          1|   N|                Y|          Five Stars|         Works pissa| 2015-08-31|
|         US|   38487968|R1EBPM82ENI67M|B000NU4OTA|      72265257|LIMTECH Wall char...|     Electronics|          1|            0|          0|   N|                Y|            One Star|Did not work at all.| 2015-08-31|
|         US|   23732619|R372S58V6D11AT|B00JOQIO6S|     308169188|Skullcandy Air Ra...|     Electronics|          5|            1|          1|   N|                Y|Overall pleased w...|Works well. Bass ...| 2015-08-31|
+-----------+-----------+--------------+----------+--------------+--------------------+----------------+-----------+-------------+-----------+----+-----------------+--------------------+--------------------+-----------+
only showing top 5 rows

```

</div>

<div class="output execute_result" data-execution_count="8">

    3093869

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:340,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="1ND6LbEkGy07" data-outputId="04298e8a-edeb-480f-825b-acdf0c479d66">

``` python
df.show(5)

# display data types and check if they are correct, e.g. rating should be double
df.printSchema()
```

<div class="output stream stdout">

``` 
+-----------+----------+-----------+--------------------+--------------+-----------+
|customer_id|product_id|star_rating|       product_title|product_parent|review_date|
+-----------+----------+-----------+--------------------+--------------+-----------+
|   41409413|B00428R89M|          5|yoomall 5M Antenn...|     112201306| 2015-08-31|
|   49668221|B000068O48|          5|Hosa GPM-103 3.5m...|     734576678| 2015-08-31|
|   12338275|B000GGKOG8|          5|Channel Master Ti...|     614448099| 2015-08-31|
|   38487968|B000NU4OTA|          1|LIMTECH Wall char...|      72265257| 2015-08-31|
|   23732619|B00JOQIO6S|          5|Skullcandy Air Ra...|     308169188| 2015-08-31|
+-----------+----------+-----------+--------------------+--------------+-----------+
only showing top 5 rows

root
 |-- customer_id: string (nullable = true)
 |-- product_id: string (nullable = true)
 |-- star_rating: string (nullable = true)
 |-- product_title: string (nullable = true)
 |-- product_parent: string (nullable = true)
 |-- review_date: string (nullable = true)

```

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:187,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="rMYUnFaCGy0-" data-outputId="2b92d557-e751-432a-8c91-72a3eb891db0">

``` python
# show summary
df.describe().show()
```

<div class="output stream stdout">

``` 
+-------+--------------------+--------------------+------------------+--------------------+--------------------+-----------+
|summary|         customer_id|          product_id|       star_rating|       product_title|      product_parent|review_date|
+-------+--------------------+--------------------+------------------+--------------------+--------------------+-----------+
|  count|             3093869|             3093869|           3093861|             3093869|             3093869|    3093750|
|   mean|2.8789345281890087E7|3.1505156710778136E9| 4.035506443243571|   2247.254098360656|5.1020043971820265E8|       null|
| stddev|1.5430609004342766E7| 3.551824947508875E9|1.3874382233284288|  1721.9748059774513| 2.868316246944571E8|       null|
|    min|            10000013|          0141186178|                 1|" Burst " Variabl...|           100007543| 1999-06-09|
|    max|             9999987|          BT008V9J9U|                 5|☆ Power Adapter ◘...|           999998189| 2015-08-31|
+-------+--------------------+--------------------+------------------+--------------------+--------------------+-----------+

```

</div>

</div>

<div class="cell markdown" id="DCPt1unYGy1C">

## Removing missing values

As the summary shown there are some rows with empty rating and they've
to be removed. In fact the count for `customer_id` and `product_id` is
the same (`3093869`) but it's less than the `star_rating` column's count
(`3093861`)

</div>

<div class="cell code" data-colab="{&quot;height&quot;:187,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="4nMQdsA_Gy1D" data-outputId="30d05096-a91a-40c8-b610-915888744372">

``` python
df = df.na.drop(subset=["star_rating"])

df.describe().show()
```

<div class="output stream stdout">

``` 
+-------+--------------------+--------------------+------------------+--------------------+-------------------+-----------+
|summary|         customer_id|          product_id|       star_rating|       product_title|     product_parent|review_date|
+-------+--------------------+--------------------+------------------+--------------------+-------------------+-----------+
|  count|             3093861|             3093861|           3093861|             3093861|            3093861|    3093750|
|   mean|2.8789345585748035E7|3.1505156710778136E9| 4.035506443243571|   2247.254098360656|5.102008536329082E8|       null|
| stddev|1.5430618553487921E7| 3.551824947508875E9|1.3874382233284288|  1721.9748059774513|2.868316559225427E8|       null|
|    min|            10000013|          0141186178|                 1|" Burst " Variabl...|          100007543| 1999-06-09|
|    max|             9999987|          BT008V9J9U|                 5|☆ Power Adapter ◘...|          999998189| 2015-08-31|
+-------+--------------------+--------------------+------------------+--------------------+-------------------+-----------+

```

</div>

</div>

<div class="cell markdown" id="tqvFFrfCGy1G">

## Removing duplicates

There could be some duplicated rows in here. To check, I'll see if there
are any row duplicated in the data.

</div>

<div class="cell code" data-colab="{&quot;height&quot;:34,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="LWKwVOq_Gy1G" data-outputId="077d2b24-ca3c-411f-abbe-7817ed0f81bf" data-scrolled="true">

``` python
num_duplicates = df.count() - df.dropDuplicates(['customer_id', 'product_id']).count()

if num_duplicates > 0:
    print('Data has {} duplicates'.format(num_duplicates))
    df_clean = df.dropDuplicates()
else : 
    print("Data hasn't duplicates")
```

<div class="output stream stdout">

    Data has 1705 duplicates

</div>

</div>

<div class="cell markdown" id="54n5JboXGy1L">

## Checking ratings distribution

</div>

<div class="cell code" data-colab="{&quot;height&quot;:68,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="yqHR4eNz_Wrm" data-outputId="d94eb902-1577-4b30-f89f-a3c2bf240c9c">

``` python
print("There is {} products\n".format(df_clean.select("product_id").distinct().count()))
print("There is {} customers".format(df_clean.select("customer_id").distinct().count()))
```

<div class="output stream stdout">

    There is 185848 products
    
    There is 2154352 customers

</div>

</div>

<div class="cell code" id="7OScgEDUGy1M" data-scrolled="true">

``` python
reviews_count = df_clean.groupBy('customer_id').count().withColumnRenamed("count","reviews")
#reviews_count.show()
```

</div>

<div class="cell code" data-colab="{&quot;height&quot;:459,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="qiKdNX4ZGy1O" data-outputId="fce4b56f-ca43-4a46-8447-11afc1d6d8d6" data-scrolled="true">

``` python
reviews_freq = reviews_count.groupBy('reviews').count().sort("reviews").withColumnRenamed("count","frequency")
reviews_freq.show()
#reviews_freq.select("frequency").groupBy().sum().show()
```

<div class="output stream stdout">

``` 
+-------+---------+
|reviews|frequency|
+-------+---------+
|      1|  1688517|
|      2|   281981|
|      3|    91371|
|      4|    38768|
|      5|    19638|
|      6|    11086|
|      7|     6702|
|      8|     4215|
|      9|     2970|
|     10|     2122|
|     11|     1399|
|     12|     1059|
|     13|      817|
|     14|      649|
|     15|      509|
|     16|      423|
|     17|      308|
|     18|      246|
|     19|      201|
|     20|      182|
+-------+---------+
only showing top 20 rows

```

</div>

</div>

<div class="cell markdown" id="oBKvAModGy1Q">

The last table shows us that `1688102` users have been rating only once
.In other word `50%` of users have rated only one item. Hence, **this
shows how much the the interation matrix is sparse**

</div>

<div class="cell markdown" id="x2xzE7tPGy1T">

## Drop unused columns

</div>

<div class="cell code" id="W27W9Z29Gy1T">

``` python
df = df_clean.drop('product_parent', 'review_date',  'product_title')
```

</div>

<div class="cell markdown" id="KotlALtyGy1V">

## Train Test Split

</div>

<div class="cell code" id="v7vyPYJ0Gy1V">

``` python

seed=42
df_train, df_test = df.randomSplit([0.8, 0.2], seed)

```

</div>

<div class="cell markdown" id="JH3f8m3iGy1X">

<hr>

# Memory based Collabortive filtering

</div>

<div class="cell code" data-colab="{&quot;height&quot;:204,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="uVE8as4nGy1X" data-outputId="5a868f68-f617-43dc-c4c3-15755667cea5">

``` python
df_train.show(5)
```

<div class="output stream stdout">

``` 
+-----------+----------+-----------+
|customer_id|product_id|star_rating|
+-----------+----------+-----------+
|   10005635|B0077V88V8|          3|
|   10006647|B002S53LJ2|          2|
|   10011241|B003SVHYAC|          5|
|   10011614|B0083C8AWW|          2|
|   10012362|B001JTQUYG|          2|
+-----------+----------+-----------+
only showing top 5 rows

```

</div>

</div>

<div class="cell markdown" id="smHhDsBuGy1Z">

## Item based approach

</div>

<div class="cell code" id="sMpO0iM2Gy1Z">

``` python
# Item-based Collaborative Filtering on pySpark 

from collections import defaultdict
from itertools import combinations
import numpy as np
import random
import csv
from scipy.stats import pearsonr

def parseRowOnUser(row):
    '''
    Parse each row of the specified list.
    Converts each rating to a float
    '''
    return row.customer_id,(row.product_id,float(row.star_rating))

def sampleInteractions(user_id,items_with_rating,n):
    '''
    For users with # interactions > n, replace their interaction history
    with a sample of n items_with_rating
    '''
    if len(items_with_rating) > n:
        return user_id, random.sample(items_with_rating,n)
    else:
        return user_id, items_with_rating

def calcSim(item_pair,rating_pairs):
    ''' 
    Inputs : 
        item_pair : (item1,item2)   
        rating_pairs : [(item1_rating1,item2_rating1),(item1_rating2,item2_rating2)...]
    Output : 
        (item1,item2) (pearson_corr,co_raters_count )
    For each item-item pair, return the specified similarity measure,
    along with co_raters_count
    '''
    rating_pairs = [rating_pair for rating_pair in rating_pairs]
    '''
    with open ('test','w') as f :
        f.write(str(rating_pairs))
        f.write('-')
    '''
    item1_ratings, item2_ratings = np.array(rating_pairs).T #decoupling item1_ratings & item2_ratings
    
    corr, _ = pearsonr(item1_ratings, item2_ratings)
    #corr = 0 if np.isnan(corr) else corr
    
    return item_pair, (corr,len(item1_ratings))

def keyOnFirstFoo(foo_pair,foo_sim_data):
    '''
    For each foo-foo pair, make the first item's id the key 
    foo can be either item or user 
    '''
    (foo1_id,foo2_id) = foo_pair
    return foo1_id,(foo2_id,foo_sim_data)

def nearestNeighbors(item_id,items_and_sims,n,strategy="top-N",threshold=0.5):
    '''
    Sort the predictions list by similarity and select the top-N neighbors 
    '''
    '''
    REMEMBER : 
    A cosine value of 0 means that the two vectors are at 90 degrees to each other (orthogonal) and have no match. 
    The closer the cosine value to 1, the smaller the angle and the greater the match between vectors.
    
    The Pearson correlation coefficient, r, can take a range of values from +1 to -1.
    A value of 0 indicates that there is no association between the two variables. 
    A value greater than 0 indicates a positive association
    '''
    if strategy =="threshold" :
        items_and_sims = [number for number in test if number[1][0] > threshold]
    else : 
        items_and_sims.sort(key=lambda x: x[1][0],reverse=True)
    
    return item_id, items_and_sims[:n]



```

</div>

<div class="cell code" id="aO4Pp2xOGy1e">

``` python
#df_train1,_ = df_train.randomSplit([0.4,0.6],seed)
```

</div>

<div class="cell markdown" id="Lzg-1RQscf1s">

### 1- Obtain the sparse user-item matrix:

    user_id -> [(item_id_1, rating_1),
               [(item_id_2, rating_2),
                ...]

</div>

<div class="cell code" id="DfRN-sl1Gy1f" data-scrolled="false">

``` python
''' 
Obtain the sparse user-item matrix:
    user_id -> [(item_id_1, rating_1),
               [(item_id_2, rating_2),
                ...]
'''

#user_item_pairs = df_train.rdd.filter(lambda r : r[0]=="10000297")

user_item_pairs = df_train1.rdd.map(parseRowOnUser).groupByKey().map(
    lambda p: sampleInteractions(p[0],list(p[1]),500)).cache()

#user_item_pairs.take(5)
```

</div>

<div class="cell markdown" id="ueTBaqZhc8zg">

### 2- Get all item-item pair combos:

    (item1,item2) ->    [(item1_rating,item2_rating),
                         (item1_rating,item2_rating),
                         ...]

</div>

<div class="cell code" data-colab="{&quot;height&quot;:51,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="KEgeXMGUGy1i" data-outputId="16617752-308b-43a0-8983-1af89582ee95">

``` python
%%time
'''
Get all item-item pair combos:
    (item1,item2) ->    [(item1_rating,item2_rating),
                         (item1_rating,item2_rating),
                         ...]
'''

def findItemPairs(user_id,items_with_rating):
    '''
    For each user, find all item-item pairs combos. (i.e. items rated by the same user) 
    '''
    itemPairsList =[((item1[0],item2[0]),(item1[1],item2[1])) for item1,item2 in combinations(items_with_rating,2)]
    return itemPairsList

pairwise_items = user_item_pairs.map(
    lambda p: findItemPairs(p[0],p[1])).flatMap(
    lambda p: (items_with_rating_pair for items_with_rating_pair in p)).groupByKey().filter(
    lambda p: len(p[1]) > 1) #filtering out item_pair with only 1 rating_pair (not enough data to compute sim for them #SG)



def iterate(iterable):
    return [v1_iterable for v1_iterable in iterable ]

##pairwise_items.take(1).foreach(lambda row: print(row[0],row[1].collect))
#[print(row[0],iterate(row[1])) for row in pairwise_items.take(2)] 
#pairwise_items.toDF(["item1_item2","rating1_rating"]).show(7)
#pairwise_items.take(2)
```

<div class="output stream stdout">

    CPU times: user 11.9 ms, sys: 2.02 ms, total: 14 ms
    Wall time: 28.1 ms

</div>

</div>

<div class="cell markdown" id="rgN2CKF9dVJr">

### 3- Calculate the similarity for each item pair and select the top-N nearest neighbors:

    (item1,item2) ->    (similarity,co_raters_count)

</div>

<div class="cell code" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="HvTmUwg2Gy1j" data-outputId="e8583d3d-cb5f-4375-c1a4-d00bda3e5361" data-scrolled="true">

``` python
%%time
'''
Calculate the similarity for each item pair and select the top-N nearest neighbors:
    (item1,item2) ->    (similarity,co_raters_count)
'''
#Calculate the cosine similarity for each item pair 
# (item1,item2) ->    (similarity,co_raters_count)
print("Computing the similarity for each item pair...")
top_n = 30
item_sims = pairwise_items.map( #Computing sim for each pair of item
        lambda p: calcSim(p[0],p[1])).filter(
        lambda p: np.isnan(p[1][0])==False).map(    #(item1,item2) (pearson_corr,co_raters_count )
        lambda p: keyOnFirstFoo(p[0],p[1])).groupByKey().map(  # (item1) [(item2,(pearson_corr,co_raters_count )),... (itemM,(pearson_corr,co_raters_count ))] m item
        lambda p: nearestNeighbors(p[0],list(p[1]),top_n)).collect() # (item1) [(item2,(pearson_corr,co_raters_count )),... (itemN,(pearson_corr,co_raters_count ))] top n item

# collecting it into a list 
#item_sims =item_sims.collect()
```

<div class="output stream stdout">

    Computing the similarity for each item pair...
    CPU times: user 121 ms, sys: 22.9 ms, total: 144 ms
    Wall time: 49.8 s

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:51,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="MbULlb7pGy1l" data-outputId="6199ec30-9a59-4dba-e6d5-40cb19cc15b7">

``` python
%%time
'''
Preprocess the item similarity matrix into a dictionary and store it as a broadcast variable:
'''

item_sim_dict = {}
for (item,data) in item_sims: 
    item_sim_dict[item] = data

isb = sc.broadcast(item_sim_dict)

```

<div class="output stream stdout">

    CPU times: user 94.6 ms, sys: 16 ms, total: 111 ms
    Wall time: 127 ms

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:51,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="GuC0wK20Gy1m" data-outputId="76c451a8-cc66-4cd0-aecd-dbaedc6bc7c5">

``` python
%%time
'''
    Obtain the sparse item-user matrix:
        item_id -> ((user_1,rating),(user2,rating))
'''

def parseRowOnItem(row):
    return row.product_id,(row.customer_id,float(row.star_rating))

item_user_pairs = df_train1.rdd.map(parseRowOnItem).groupByKey().map(
    lambda p: sampleInteractions(p[0],list(p[1]),500)).cache()
```

<div class="output stream stdout">

    CPU times: user 16.7 ms, sys: 979 µs, total: 17.7 ms
    Wall time: 49.8 ms

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:51,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="gEUT7WKHGy1n" data-outputId="9497a6ea-aeb5-4ec4-aa4e-f9043b503550" data-scrolled="true">

``` python
%%time
def fooAverageRating(foos_pairs):
    """
        Obtain foo_rating_average matrix containing average rating for each foo:
        dict : (foo_id) -> (avg_rating)
        where foo can be items or users and foos is either item_user or user_item
    """
    
    def averageRating(row):
        '''
        Compute the average rating of an item
        '''
        _, ratings =np.array(row[1],dtype = object).T
        return row[0],np.mean(ratings)

    return dict(foos_pairs.map(averageRating).collect())

irab = sc.broadcast(fooAverageRating(item_user_pairs))
#item_user_pairs_avg_b.value
```

<div class="output stream stdout">

    CPU times: user 192 ms, sys: 37.8 ms, total: 230 ms
    Wall time: 1min 29s

</div>

</div>

<div class="cell markdown" id="9sW1z921d6vJ">

### 4- Calculate the top-N item recommendations for each user

    user_id -> [item1,item2,item3,...]

</div>

<div class="cell code" data-colab="{&quot;height&quot;:51,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="o89yjNQHGy1o" data-outputId="7f56143b-190a-4cb4-9ccc-03d5ea3ce54e">

``` python
%%time
def topNRecommendations(user_id,items_with_rating,item_sims,items_rating_average,n):
    '''
        user_id 
        items_with_rating : [(item_id_1, rating_1),
                   (item_id_2, rating_2)]
        item_sims : dict = {'item_id_i': [(item_id_j, (sim(item_id_i,item_id_j), co_raters_count)), (item_id_k, (sim(item_id_i,item_id_k), co_raters_count))],...} 
        items_rating_average : dict = {'item_id_i': avg ...}
    Calculate the top-N item recommendations for each user using the 
    weighted sums method
    '''

    # initialize dicts to store the score of each individual item,
    # since an item can exist in more than one item neighborhood
    totals = defaultdict(int)
    sim_sums = defaultdict(int)

    for (item,rating) in items_with_rating:

        # lookup the nearest neighbors for this item
        nearest_neighbors = item_sims.get(item,None)

        if nearest_neighbors:
            for (neighbor,(sim,count)) in nearest_neighbors:
                if neighbor != item: #ensure that it will not recommend items already rated #SG

                    # update totals and sim_sums with the rating data
                    totals[neighbor] += sim * (rating-items_rating_average[neighbor])
                    sim_sums[neighbor] += abs(sim)

    # create the normalized list of scored items 
    scored_items = []
    for item,total in totals.items():
        
        # strategy 1
        division = 0 if sim_sums[item]==0 else total/sim_sums[item]
        scored_items.append((division+items_rating_average[item],item)) 
        '''
        # strategy 2
        if (sim_sums[item]!=0 ):
            scored_items.append((total/sim_sums[item]+items_rating_average[item],item)) 
        '''
    # sort the scored items in ascending order
    scored_items.sort(reverse=True)

    # take out the item score
    # ranked_items = [x[1] for x in scored_items]

    return user_id,scored_items[:n]


'''
Calculate the top-N item recommendations for each user
    user_id -> [item1,item2,item3,...]
'''
user_item_based_recs = user_item_pairs.map(
    lambda p: topNRecommendations(p[0],p[1],isb.value,irab.value,500)).collect()

#user_item_recs.toDF(["user_id","recommendations"]).show(5)
```

<div class="output stream stdout">

    CPU times: user 19.4 s, sys: 5.66 s, total: 25 s
    Wall time: 2min 18s

</div>

</div>

<div class="cell code" id="uFP79Z-BGy1q">

``` python
def recommend(user_id,user_item_pairs,sim_matrix):
    '''
    Compute the top n recommended items'score for the user with user_id
    return a list
    '''
    prediction =  user_item_pairs.filter(lambda r : r[0]==user_id).map(
        lambda p: topNRecommendations(p[0],p[1],sim_matrix.value,500))
    predictions = prediction.collect()
    predictions = predictions[0][1] if len(predictions)>0 else None
    return predictions

#print(recommend ("13357",user_item_pairs,isb))
```

</div>

<div class="cell code" id="pEPfUlGcTkQf">

``` python
# read in the test data
df_test_list =df_test.rdd.collect() 
df_train_list = df_train.collect()
```

</div>

<div class="cell markdown" id="fr142507guCi">

### 5- Performance

</div>

<div class="cell code" data-colab="{&quot;height&quot;:153,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="G0ifbsOuGy1r" data-outputId="832bd40f-8ba2-4790-9376-efe645e9c81a">

``` python
%%time
'''
Read in test data and calculate specified metric 
'''
from sklearn.metrics import mean_squared_error,mean_absolute_error

def evaluate(df_test,df_train,user_item_recs,metric=None):
    test_ratings = defaultdict(list)
    train_ratings = defaultdict(list)
    
    for row in df_test:
        user = row.customer_id
        item = row.product_id
        rating = row.star_rating
        test_ratings[user] += [(item,rating)] 
        
    for row in df_train:
        user = row.customer_id
        item = row.product_id
        rating = row.star_rating
        train_ratings[user] += [(item,rating)]  
      
    def compute(foo_ratings) :
        # create train-test rating tuples
        y_pred = []
        y_true = []
        for (user,items_with_rating) in user_item_recs:
            for (rating,item) in items_with_rating:
                for (test_item,test_rating) in foo_ratings[user]:                
                    if str(test_item) == str(item):
                        y_true.append(float(test_rating))
                        y_pred.append(rating)
        #print(len(y_pred),'\n')
        if(metric=='rmse') :
            return mean_squared_error(y_true, y_pred)
        elif(metric=='mae'):
            return mean_absolute_error(y_true, y_pred)
        else : 
            return (mean_absolute_error(y_true, y_pred),mean_squared_error(y_true, y_pred))
        
    return compute(train_ratings) , compute(test_ratings)



result = (evaluate(df_test_list,df_train_list,user_item_based_recs))
print ("MAE: \n Train : {} \n Test  : {}\nRMSE: \n Train : {} \n Test  : {}".format(result[0][0],result[1][0],result[0][1],result[1][1] ))




#Mean Absolute Error:  0.644232031607937 # strategy 2 TopNrecomm
#Mean Absolute Error:  0.6334624623392405  # strategy 1 TopNrecomm - 1.0726394395209509
#Mean Absolute Error:  0.6288800241389374 in df_test
```

<div class="output stream stdout">

    MAE: 
     Train : 0.665225623907677 
     Test  : 0.6935194429936145
    RMSE: 
     Train : 1.0973471433958188 
     Test  : 1.3140512654094085
    CPU times: user 41 s, sys: 454 ms, total: 41.5 s
    Wall time: 41.5 s

</div>

</div>

<div class="cell markdown" id="1Z12II3kGy1t">

## User based approach

</div>

<div class="cell code" id="AT6gZQxWGy1t">

``` python
'''
Get all user-user pair combos:
    (user1_id,user2_id) -> [(rating1,rating2),
                            (rating1,rating2),
                            (rating1,rating2),
                            ...]
'''

def findUserPairs(item_id,users_with_rating):
    '''
    For each item, find all user-user pairs combos. (i.e. users with the same item) 
    '''
    userPairsList =[((user1[0],user2[0]),(user1[1],user2[1])) for user1,user2 in combinations(users_with_rating,2)]
    return userPairsList
pairwise_users = item_user_pairs.map(
    lambda p: findUserPairs(p[0],p[1])).flatMap(
    lambda p: (users_with_rating_pair for users_with_rating_pair in p)).groupByKey().filter(
    lambda p: len(p[1]) > 1) #filtering out user_pair with only 1 rating_pair (not enough data to compute sim for them #SG)


##pairwise_items.take(1).foreach(lambda row: print(row[0],row[1].collect))
#[print(row[0],iterate(row[1])) for row in pairwise_items.take(2)] 
#pairwise_items.toDF(["item1_item2","rating1_rating"]).show(7)
#pairwise_users.take(2)
```

</div>

<div class="cell code" id="hlKmbeKvGy1u">

``` python
#[print(row[0],iterate(row[1])) for row in pairwise_items.take(2)] 
```

</div>

<div class="cell code" id="9tOyFkiZGy1v">

``` python
'''
Calculate the similarity for each user pair and select the top-N nearest neighbors:
    (user1,user2) ->    (similarity,co_raters_count)
'''
top_n_user = 50

user_sims = pairwise_users.map(
    lambda p: calcSim(p[0],p[1])).map(
    lambda p: keyOnFirstFoo(p[0],p[1])).groupByKey().map(
    lambda p: nearestNeighbors(p[0],list(p[1]),top_n_user))



```

</div>

<div class="cell code" data-colab="{&quot;height&quot;:51,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="-heQYyEDGy1x" data-outputId="1667c060-5312-4230-a656-922803cad6d7">

``` python
%%time
''' 
Obtain the the item history for each user and store it as a broadcast variable
    user_id -> [(item_id_1, rating_1),
               [(item_id_2, rating_2),
                ...]
'''
def parseRowOnUser(row):
    '''
    Parse each row of the specified list.
    Converts each rating to a float
    '''
    return row.customer_id,(row.product_id,float(row.star_rating))

user_item_hist =df_train1.rdd.map(parseRowOnUser).groupByKey().collect()


ui_dict = {}
for (user,items_with_ratings) in user_item_hist: 
    ui_dict[user] = items_with_ratings

uib = sc.broadcast(ui_dict)
```

<div class="output stream stdout">

    CPU times: user 18.5 s, sys: 1.28 s, total: 19.8 s
    Wall time: 1min 42s

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:51,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="V6pJk0fUGy1y" data-outputId="4ed8b6d9-c148-491d-d3bc-2b7814717e33">

``` python
%%time
def fooAverageRating(foos_pairs):
    """
        Obtain foo_rating_average matrix containing average rating for each foo:
        dict : (foo_id) -> (avg_rating)
        where foo can be items or users and foos is either item_user or user_item
    """
    
    def averageRating(row):
        '''
        Compute the average rating of an item
        '''
        _, ratings = np.array([foo_rating for foo_rating in row[1]] ,dtype = object).T
        return row[0],np.mean(ratings)

    return dict(foos_pairs.map(averageRating).collect())

#u_i= df_train1.rdd.map(parseRowOnUser).groupByKey()
urab = sc.broadcast(fooAverageRating(df_train1.rdd.map(parseRowOnUser).groupByKey()))
#item_user_pairs_avg_b.value
```

<div class="output stream stdout">

    CPU times: user 0 ns, sys: 13 µs, total: 13 µs
    Wall time: 16 µs

</div>

</div>

<div class="cell markdown" id="13VbW-6Pqy0w">

### Recommandation

</div>

<div class="cell code" data-colab="{&quot;height&quot;:51,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="3uL0SpJmGy10" data-outputId="ffbac6d2-c38b-404f-cbca-f5e2a8a7e413" data-scrolled="false">

``` python
%%time
def topNRecommendationsUI(user_id,user_sims,users_with_rating,users_rating_average,n):
    '''
        user_id 
        users_with_rating : dict = {user_id_1 : [(item_id_1, rating_1),
                   (item_id_2, rating_2)],....}
        user_sims : list = [(user_id_j, (sim(user_id_i,user_id_j), co_raters_count)), (user_id_k, (sim(user_id_i,user_id_k), co_raters_count)),....]
        items_rating_average : dict = {'user_id_i': avg ...}
    Calculate the top-N item recommendations for each user using the 
    weighted sums method
    '''

    # initialize dicts to store the score of each individual item,
    # since an item can exist in more than one item neighborhood
    totals = defaultdict(int)
    sim_sums = defaultdict(int)

    for (neighbor,(sim,count)) in user_sims:

        # lookup the item predictions for this neighbor
        unscored_items = users_with_rating.get(neighbor,None)
        if unscored_items:
     
            for (item,rating) in unscored_items:
                if dict(users_with_rating[user_id]).get(item)==None: 

                    # update totals and sim_sums with the rating data
                    totals[item] += sim * (rating-users_rating_average[neighbor])
                    sim_sums[item] +=  abs(sim)

    # create the normalized list of scored items 
    scored_items = []
    for item,total in totals.items():
        
        # strategy 1
        division = 0 if sim_sums[item]==0 else total/sim_sums[item]
        scored_items.append((division+users_rating_average[user_id],item)) 
        '''
        # strategy 2
        if (sim_sums[item]!=0 ):
            scored_items.append((total/sim_sums[item]+items_rating_average[item],item)) 
        '''
    # sort the scored items in ascending order
    scored_items.sort(reverse=True)

    # take out the item score
    # ranked_items = [x[1] for x in scored_items]

    return user_id,scored_items[:n]


'''
Calculate the top-N item recommendations for each user
    user_id -> [item1,item2,item3,...]
'''
user_item_recs = user_sims.map(
    lambda p: topNRecommendationsUI(p[0],p[1],uib.value,urab.value,100)).collect()

#user_item_recs.toDF(["user_id","recommendations"]).show(5)
```

<div class="output stream stdout">

    CPU times: user 569 ms, sys: 100 ms, total: 670 ms
    Wall time: 47min 41s

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:119,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="TWxtyN75Gy12" data-outputId="3c5ad7af-bf89-4f35-a685-3da5dcae4c27" data-scrolled="true">

``` python
result = (evaluate(df_test_list,df_train_list,user_item_recs))
print ("MAE: \n Train : {} \n Test  : {}\nRMSE: \n Train : {} \n Test  : {}".format(result[0][0],result[1][0],result[0][1],result[1][1] ))

```

<div class="output stream stdout">

    MAE: 
     Train : 0.521125623907677 
     Test  : 0.535225623903678
    RMSE: 
     Train : 1.0013471433958179 
     Test  : 1.2973471433958188

</div>

</div>

<div class="cell markdown" id="WEKQZQOsGy13">

## Kmean

</div>

<div class="cell code" id="JLhmo0a0Gy14">

``` python
products = df_train1.select("product_id").distinct().collect()
```

</div>

<div class="cell code" id="g4-bueIBGy15" data-scrolled="true">

``` python
products = dict([(product.product_id,0.0) for product in products])
```

</div>

<div class="cell code" id="5tukBSQlGy16" data-scrolled="true">

``` python
def parseProducts (user_id,products_with_ratings,products):
    this_products = products.copy()
    for (product,rating) in products_with_ratings :
        this_products[product]=rating
    return user_id , list(this_products.values())

user_item_features = df_train1.rdd.map(parseRowOnUser).groupByKey().map(lambda p: parseProducts(p[0],p[1],products))
```

</div>

<div class="cell code" id="V43OdyOrb4FO">

``` python
def func (p):
  users , rating_pair = p
  return (users,(*rating_pair))
a=user_item_features.map(func).toDF()
#a.show(2)
```

</div>

<div class="cell code" id="_bD0nRvlGy17">

``` python
uif_df = user_item_features.toDF().withColumnRenamed('_1','user_id').withColumnRenamed('_2','features')
```

</div>

<div class="cell code" data-colab="{&quot;height&quot;:204,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="CnuNnz8MGy18" data-outputId="57c2dc45-c4f1-4e30-a56f-013d82c333e5" data-scrolled="true">

``` python
uif_df.show(5)
```

<div class="output stream stdout">

``` 
+--------+--------------------+
| user_id|            features|
+--------+--------------------+
|10171604|[0.0, 0.0, 0.0, 0...|
|10634752|[0.0, 0.0, 0.0, 0...|
|10840727|[0.0, 0.0, 0.0, 0...|
|11488922|[0.0, 0.0, 0.0, 0...|
|11654799|[0.0, 0.0, 0.0, 0...|
+--------+--------------------+
only showing top 5 rows

```

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:102,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="ID-07nOYGy1-" data-outputId="18a76fc4-313c-4d5a-8656-81001eb13120" data-scrolled="true">

``` python
uif_df.printSchema()
```

<div class="output stream stdout">

``` 
root
 |-- user_id: string (nullable = true)
 |-- features: array (nullable = true)
 |    |-- element: double (containsNull = true)

```

</div>

</div>

<div class="cell markdown" id="I4zdk_ZLGy1_">

features column containsNull attribute have to be false

</div>

<div class="cell code" id="Sjz1hCOFGy1_">

``` python
from pyspark.sql.types import ArrayType,DoubleType
from pyspark.sql.functions import udf
from pyspark.ml.clustering import KMeans
```

</div>

<div class="cell code" id="C1cnxmaaGy2A">

``` python
new_schema = ArrayType(DoubleType(), containsNull=False)
udf_foo = udf(lambda x:x, new_schema)
uif_df = uif_df.withColumn("features",udf_foo("features"))
```

</div>

<div class="cell code" id="bHerqLwtGy2B">

``` python
kmeans = KMeans(k=5, seed=1,maxIter=10) 
model = kmeans.fit(uif_df.select('features'))
```

</div>

<div class="cell code" id="uf2QmkZoGy2U">

``` python
transformed = model.transform(new_df)
transformed.show()    
```

</div>

<div class="cell code" id="YJuxoDRVf8n9">

``` python
```

</div>

<div class="cell markdown" id="ALPb0-Oss2JH">

\# Model Based Approches

</div>

<div class="cell markdown" id="6TMy4JxQs_G9">

## Matrix factorisation

</div>

<div class="cell code" id="p-UxpnOqVCdA">

``` python
def load_data(path, sample_size=300000):
  print("Loading data...")
  cols = ["customer_id", "product_id", "star_rating"]
  pdf = pd.read_csv(path, sep='\t', usecols=cols)
  print("Data loaded!", end="\n\n")
  return pdf.sample(sample_size)
```

</div>

<div class="cell code" id="58gQtha7JX85">

``` python
def get_sample(df, size=500, seed=42):
  np.random.seed(seed)
  random_indexes = np.random.randint(df.shape[0], size=size)
  return df.iloc[random_indexes, :].copy()
```

</div>

<div class="cell code" id="3_hCX_mCMCHB">

``` python
def convert_data_type(df):
  print("Converting data type...")
  encoder = LabelEncoder()
  encoder.fit(df.product_id.unique())
  df['product_id'] = encoder.transform(df.product_id.values)
  df.customer_id.astype(int)
  df.star_rating.astype(int)
  print("Data converted!", end="\n\n")
  return df, encoder
```

</div>

<div class="cell code" id="wZtlX9UiuLPa">

``` python
def panda2spark(pdf):
  print("Creating spark dataframe...")
  return spark.createDataFrame(pdf)
```

</div>

<div class="cell code" id="URnbRNlI1NcI">

``` python
def grid_search_als(estimator: ALS, evaluator: RegressionEvaluator, ranks=[12, 14, 16], maxIters=[19, 21], regParams=[.17, .19], parallelism=4):
  
  print("Grid search configuration...")
  param_grid = ParamGridBuilder()\
                .addGrid(estimator.rank, ranks)\
                .addGrid(estimator.maxIter, maxIters)\
                .addGrid(estimator.regParam, regParams)\
                .build()

  trainValidationSplit = TrainValidationSplit(
      estimator=estimator, 
      estimatorParamMaps=param_grid, 
      evaluator=evaluator, 
      parallelism = parallelism
      )
  
  return trainValidationSplit, param_grid
```

</div>

<div class="cell code" id="ZSa7R1CG1trm">

``` python
def get_evaluator(metricName='rmse'):
  print("Building evaluator...")
  evaluator = RegressionEvaluator(
    predictionCol='prediction', 
    labelCol='star_rating', 
    metricName=metricName
    )
  print("Evaluator build finished!", end="\n\n")
  return evaluator
```

</div>

<div class="cell code" id="FdyBetOydP5W">

``` python
def build_model(train, rank=20, maxIter=20, regParam=0.2, search_best_params=False, ranks=[21, 24, 26], maxIters=[20, 22], regParams=[.25, .3]):
    als = ALS(
    userCol='customer_id',
    itemCol='product_id', 
    ratingCol='star_rating', 
    coldStartStrategy='drop', 
    nonnegative=True,
    rank=rank,
    maxIter = maxIter,
    regParam = regParam
    )

    if search_best_params == True:
      evaluator = get_evaluator()
      (trainValidationSplit, param_grid) = grid_search_als(als, evaluator)

      print("Training model with best params...")
      models = trainValidationSplit.fit(train)
      print("Training finished successfully!", end="\n\n")

      model = models.bestModel 
    else:
      print("Model parammeters:")
      print(f"rank: {rank}")
      print(f"maxIter: {maxIter}")
      print(f"regParam: {regParam}\n")

      print("Training the model...")
      model = als.fit(train)
      print("Training finished!",  end="\n\n")

    return model
```

</div>

<div class="cell code" id="uiJWZz8VmdXz">

``` python
def evaluation(model, train, test, metricName='rmse'):
  print("Evaluating the model...")

  evaluator = get_evaluator(metricName)
  
  print("Computing loss..")
  train_predictions = model.transform(train)
  train_rmse = evaluator.evaluate(train_predictions)
  print(f"Training error: {train_rmse}")

  test_predictions = model.transform(test)
  test_rmse = evaluator.evaluate(test_predictions)
  print(f"Test error: {test_rmse}")

  model_evaluation = {
      "evaluator": evaluator,
      "rmse": {
          "train": train_rmse,
          "test": test_rmse
      },
      "predictions": {
          "train": train_predictions,
          "test": test_predictions
      }
  }

  print("Evaluation finished!", end="\n\n")
  return model_evaluation
```

</div>

<div class="cell code" id="wDXaYH_sT7mV">

``` python
def pipeline(path, search_best_params=False, ranks=[21, 24, 26], maxIters=[20, 22], regParams=[.25, .3]):
  print("Starting...")
  pdf = load_data(path, sample_size=1000000)
  (pdf, encoder) = convert_data_type(pdf)
  df = panda2spark(pdf)

  (train, test) = df.randomSplit([0.8, 0.2])

  model = build_model(train, search_best_params=search_best_params, regParam=0.5)
  model_evaluation = evaluation(model, train, test)

  model.save('matrix_factorization')
  print("Finished!", end="\n\n")

  return model, model_evaluation
```

</div>

<div class="cell code" data-colab="{&quot;height&quot;:476,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="4ADDfU2VYabp" data-outputId="260d380f-1e2a-42b1-c9eb-49a5ae970ddf">

``` python
path = 'amazon_reviews_us_Electronics_v1_00.tsv.gz'

(model, model_evaluation) = pipeline(path, search_best_params=False)
```

<div class="output stream stdout">

``` 
Starting...
Loading data...
Data loaded!

Converting data type...
Data converted!

Creating spark dataframe...
Model parammeters:
rank: 20
maxIter: 20
regParam: 0.5

Training the model...
Training finished!

Evaluating the model...
Building evaluator...
Evaluator build finished!

Computing loss..
Training error: 0.5374322041228355
Test error: 1.8636257049613567
Evaluation finished!

Finished!

```

</div>

</div>

<div class="cell code" id="od2_7FUN2mjH">

``` python
```

</div>

<div class="cell markdown" id="DSnbNZV6suyC">

## Deep Learning

</div>

<div class="cell markdown" id="DTLQDQ07qADS">

### 1.2 Fonctions utilitaires

</div>

<div class="cell code" id="vapyPsMxpwXj">

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import normalize

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

warnings.filterwarnings('ignore')
%matplotlib inline
```

</div>

<div class="cell code" id="fmZ5d0qptvZJ">

``` python
def load_data(path, sample_size='all'):
  
  print("Loading data...")
  cols = ["customer_id", "product_id", "star_rating"]
 
  pdf = pd.read_csv(path, sep='\t', usecols=cols)
  print("Data loaded!", end="\n\n")

  if sample_size == 'all':
    return pdf
  else:
    return pdf.sample(sample_size)

def convert_data_type(df):
  print("Converting data type...")
  encoder = LabelEncoder()
  encoder.fit(df.product_id.unique())
  df['product_id'] = encoder.transform(df.product_id.values)
  # df.customer_id.astype(int)
  # df.star_rating.astype(int)
  print("Data converted!")
  return df

def denormalize_rating(rating, min, max):
  return rating*(max - min) + min
```

</div>

<div class="cell markdown" id="O7K45cj3qHHl">

### 1.3 Chargement et prétraitement du dataset

</div>

<div class="cell code" data-colab="{&quot;height&quot;:68,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="prhEIuotpwXw" data-outputId="d481e348-b362-4d4d-9159-35f7c3b13db3">

``` python
path = 'amazon_reviews_us_Electronics_v1_00.tsv.gz'

dataset = load_data(path, sample_size='all')
```

<div class="output stream stdout">

``` 
Loading data...
Data loaded!

```

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:34,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="Zzq0O_mNlgfB" data-outputId="b400a734-c9d4-4fc7-f4d7-9e850319fd33">

``` python
dataset.shape
```

<div class="output execute_result" data-execution_count="63">

    (3091103, 3)

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:51,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="wbXqVsrevE6P" data-outputId="e6b56f4b-ed86-4dc9-8c37-c64f3e68dc76">

``` python
dataset = convert_data_type(dataset)
```

<div class="output stream stdout">

    Converting data type...
    Data converted!

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:34,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="Th_ERoDIniPq" data-outputId="2a47acca-3944-4c01-8070-4059c53ffc78">

``` python
num_duplicates = dataset.shape[0] - dataset.drop_duplicates().shape[0]
num_duplicates
```

<div class="output execute_result" data-execution_count="9">

    1417

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:34,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="VZegUbRxpwX8" data-outputId="a2a26d99-7f9b-411e-d030-9e7c744b8fb5">

``` python
dataset.shape
```

<div class="output execute_result" data-execution_count="10">

    (3091103, 3)

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:34,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="_gwsTIHNtjyx" data-outputId="e8c9c632-3a56-4615-8fe2-0a3821a31dc6">

``` python
# dataset.drop_duplicates(inplace=True)
# dataset.shape
```

<div class="output execute_result" data-execution_count="67">

    (3089686, 3)

</div>

</div>

<div class="cell code" id="_fi3eev0pwYE">

``` python
train, test = train_test_split(dataset, test_size=0.2, random_state=42)
```

</div>

<div class="cell code" data-colab="{&quot;height&quot;:34,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="dytXEkPlpwYJ" data-outputId="c70224b4-1103-482e-b8fa-12a7ac6c501c">

``` python
train.shape
```

<div class="output execute_result" data-execution_count="66">

    (2472882, 3)

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:34,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="Ln2h_D4YpwYP" data-outputId="ffa76857-73c9-4559-e312-c6375bcb9073">

``` python
test.shape
```

<div class="output execute_result" data-execution_count="67">

    (618221, 3)

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:255,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="U0i9ZVr4M0pu" data-outputId="589f7d8a-36a7-4a4b-81e9-6c811d10c738">

``` python
display(train.head())
print(train.customer_id.unique().size)
print(train.product_id.unique().size)
print(train.star_rating.unique().size)
```

<div class="output display_data">

``` 
         customer_id  product_id  star_rating
645939      17671807      132841            5
404773      48799690      163737            3
1066713     18878830       45657            1
236742      11481572      175744            5
215866        868194      105895            2
```

</div>

<div class="output stream stdout">

    1803114
    168601
    5

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:34,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="xIsZ2HHTB8Jt" data-outputId="2626a04e-006c-45bb-ab38-d5478ee26397">

``` python
dataset.product_id.max()
```

<div class="output execute_result" data-execution_count="69">

    185774

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:34,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="NuDcodcspwYW" data-outputId="41c61c40-73f6-4210-a4b5-71d62a8c1382">

``` python
n_customers = len(dataset.customer_id.unique())
n_customers
```

<div class="output execute_result" data-execution_count="70">

    2152825

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:34,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="VgZyGQlhpwYe" data-outputId="269d9597-2c93-48d9-d2b8-84f11da0c984">

``` python
n_products = dataset.product_id.nunique()
n_products
```

<div class="output execute_result" data-execution_count="71">

    185775

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:34,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="o0JiC_tSqV_w" data-outputId="f39f86bc-14c8-4eef-959e-83a1e97ca579">

``` python
train.customer_id.nunique()
```

<div class="output execute_result" data-execution_count="72">

    1803114

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:34,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="IKtPjTnaqcKG" data-outputId="aaaab65b-7eb8-48c5-d1b1-160627686f01">

``` python
train.product_id.nunique()
```

<div class="output execute_result" data-execution_count="73">

    168601

</div>

</div>

<div class="cell code" id="YalZrAXNNvB7">

``` python
n_latent_factors = 15
```

</div>

<div class="cell code" id="H8KrI9TIpwYu" data-scrolled="true">

``` python
#from keras.models import load_model

#if os.path.exists('regression_model.h5'):
    #model = load_model('regression_model.h5')
#else:

def fit_plot_save(model, path='regression_model.h5' , output=train.star_rating, batch_size=4096, epochs=10):
  history = model.fit([train.customer_id, train.product_id], train.star_rating, batch_size=batch_size, epochs=epochs, verbose=1)
  model.save(path)
  plt.plot(history.history['loss'])
  plt.xlabel("Epochs")
  plt.ylabel("Training Error")

# 1.4947
# 1.4978
# 1.5447
# 1.4787
# 1.4707
# 1.4867
```

</div>

<div class="cell markdown" id="8Qusg3v3sYzW">

### 2 - Construction du modèle

</div>

<div class="cell markdown" id="_xeYWvImr_q9">

\#\#\#\# 2.1 - Reproduction d'une Matrix Factorization à l'aide d'un
réseau de à une couche

</div>

<div class="cell code" id="HyWgwp3BpwYo">

``` python
################---Input Layer---################
# product input
product_input = Input(shape=[1], name="Product-Input")

# customer input 
customer_input = Input(shape=[1], name="Customer-Input")


################---Matrix-Factorization embeddings Layer---################
# customer Matrix-Factorization vector
customer_embedding = Embedding(n_customers, n_latent_factors, name="Customer-Embedding")(customer_input)
customer_vec = Flatten(name="Flatten-Customers")(customer_embedding)

# product Matrix-Factorization vector
product_embedding = Embedding(n_products + 1, n_latent_factors, name="Product-Embedding")(product_input)
product_vec = Flatten(name="Flatten-Products")(product_embedding)


################---Matrix-Factorization output Layer---################
mf_output_layer = Dot(name="Dot-Product", axes=1)([product_vec, customer_vec])

# Create model and compile
mf_model = Model([customer_input, product_input], mf_output_layer)
mf_model.compile('adam',  loss='mean_squared_error')
```

</div>

<div class="cell code" data-colab="{&quot;height&quot;:408,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="tMuAbEMopqKd" data-outputId="89d92a75-92cd-4302-a93f-e1ee2570ed14">

``` python
mf_model.summary()
```

<div class="output stream stdout">

    Model: "functional_13"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    Product-Input (InputLayer)      [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    Customer-Input (InputLayer)     [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    Product-Embedding (Embedding)   (None, 1, 15)        2786640     Product-Input[0][0]              
    __________________________________________________________________________________________________
    Customer-Embedding (Embedding)  (None, 1, 15)        32292375    Customer-Input[0][0]             
    __________________________________________________________________________________________________
    Flatten-Products (Flatten)      (None, 15)           0           Product-Embedding[0][0]          
    __________________________________________________________________________________________________
    Flatten-Customers (Flatten)     (None, 15)           0           Customer-Embedding[0][0]         
    __________________________________________________________________________________________________
    Dot-Product (Dot)               (None, 1)            0           Flatten-Products[0][0]           
                                                                     Flatten-Customers[0][0]          
    ==================================================================================================
    Total params: 35,079,015
    Trainable params: 35,079,015
    Non-trainable params: 0
    __________________________________________________________________________________________________

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:369,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="Kodb7HVH2c3i" data-outputId="70e8c804-f808-4574-b2ea-16090581b7e5">

``` python
plot_model(mf_model, to_file='mf_model.png')
```

<div class="output execute_result" data-execution_count="80">

![](4a4513b5384d5eda6b4600428949c27a905e3f70.png)

</div>

</div>

<div class="cell code" id="1fh62ciD0YU2">

``` python
fit_plot_save(mf_model, path='mf_model.h5') 
```

</div>

<div class="cell markdown" id="dvuygVq_nayb">

### Meilleur Loss: 18.7

### Temps de calcul: 27s

### Epochs 10

</div>

<div class="cell markdown" id="kqCrJqhlsRMI">

#### 2.2 - Ajout d'une couche densément connectée à la sortie du réseau de neurone (DNN)

</div>

<div class="cell code" id="a9bpK7SxwvKr">

``` python
################---Input Layer---################
# product input
product_input = Input(shape=[1], name="Product-Input")

# customer input 
customer_input = Input(shape=[1], name="Customer-Input")


################---Matrix-Factorization embeddings Layer---################
# product Matrix-Factorization vector
product_embedding = Embedding(n_products+1, n_latent_factors, name="Product-Embedding")(product_input)
product_vec = Flatten(name="Flatten-Products")(product_embedding)

# customer Matrix-Factorization vector
customer_embedding = Embedding(n_customers+1, n_latent_factors, name="Customer-Embedding")(customer_input)
customer_vec = Flatten(name="Flatten-Customers")(customer_embedding)

################---Matrix-Factorization output Layer---################
mf_output_layer = Dot(name="Dot-Product", axes=1)([product_vec, customer_vec])

################---Fully-connected Layers---################
fully_connected_layer = Dense(128, activation='relu')(mf_output_layer)
fully_connected_layer0 = Dense(128, activation='relu')(fully_connected_layer)
fully_connected_layer1 = Dense(128, activation='relu')(fully_connected_layer0)
fully_connected_layer2 = Dense(64, activation='relu')(fully_connected_layer1)


# ################---Output Layer---################
output_layer = Dense(1)(fully_connected_layer2)


# Create model and compile
neural_mf_model = Model([customer_input, product_input], output_layer)
neural_mf_model.compile('adam',  loss='mean_squared_error')
```

</div>

<div class="cell code" data-colab="{&quot;height&quot;:578,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="YpJrFXoxnn7l" data-outputId="4f8fa0b9-52c8-453b-e99b-23aacc314cb8">

``` python
neural_mf_model.summary()
```

<div class="output stream stdout">

    Model: "functional_15"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    Product-Input (InputLayer)      [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    Customer-Input (InputLayer)     [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    Product-Embedding (Embedding)   (None, 1, 15)        2786640     Product-Input[0][0]              
    __________________________________________________________________________________________________
    Customer-Embedding (Embedding)  (None, 1, 15)        32292390    Customer-Input[0][0]             
    __________________________________________________________________________________________________
    Flatten-Products (Flatten)      (None, 15)           0           Product-Embedding[0][0]          
    __________________________________________________________________________________________________
    Flatten-Customers (Flatten)     (None, 15)           0           Customer-Embedding[0][0]         
    __________________________________________________________________________________________________
    Dot-Product (Dot)               (None, 1)            0           Flatten-Products[0][0]           
                                                                     Flatten-Customers[0][0]          
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 128)          256         Dot-Product[0][0]                
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 128)          16512       dense[0][0]                      
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 128)          16512       dense_1[0][0]                    
    __________________________________________________________________________________________________
    dense_3 (Dense)                 (None, 64)           8256        dense_2[0][0]                    
    __________________________________________________________________________________________________
    dense_4 (Dense)                 (None, 1)            65          dense_3[0][0]                    
    ==================================================================================================
    Total params: 35,120,631
    Trainable params: 35,120,631
    Non-trainable params: 0
    __________________________________________________________________________________________________

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:856,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="RrhZioE_2AYn" data-outputId="09da58e5-e43d-420e-c723-1531c98ed52a">

``` python
plot_model(neural_mf_model, to_file='neural_mf_model.png')
```

<div class="output execute_result" data-execution_count="84">

![](b60376e8ac39dae756e84e4c538890b41022670f.png)

</div>

</div>

<div class="cell code" id="hMwnHNQn06ro">

``` python
# fit_plot_save(neural_mf_model, 'neural_mf_model.h5')
history = neural_mf_model.fit([train.customer_id, train.product_id], train.star_rating, batch_size=4096, epochs=10, verbose=1)
neural_mf_model.save('neural_mf_model.h5')
plt.plot(history.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Training Error")
```

</div>

<div class="cell markdown" id="S7t_ljsMnsRH">

### Meilleur Loss: 2.6

### Temps de calcul: 31s

### Epochs: 10

</div>

<div class="cell markdown" id="InsycR2DsrUm">

#### 2.3 - Model Final (MLP + MF + DNN)

#### Concaténation de la sortie du model précédent avec un Multi Layer Perceptron (MLP) puis ajout d'un DNN à la sortie de celle-ci

</div>

<div class="cell code" id="TzEfHqJZyjYp">

``` python
################---Input Layer---################
# product input
product_input = Input(shape=[1], name="Product-Input")

# customer input 
customer_input = Input(shape=[1], name="Customer-Input")

################---Matrix-Factorization embeddings Layer---################
# product Matrix-Factorization vector
product_embedding = Embedding(n_products+1, n_latent_factors, name="Product-Embedding")(product_input)
product_vec = Flatten(name="Flatten-Products")(product_embedding)

# customer Matrix-Factorization vector
customer_embedding = Embedding(n_customers+1, n_latent_factors, name="Customer-Embedding")(customer_input)
customer_vec = Flatten(name="Flatten-Customers")(customer_embedding)

################---Matrix-Factorization output Layer---################
mf_output_layer = Dot(name="Dot-Product", axes=1)([product_vec, customer_vec])


################---Multi-Layer-Perceptron embeddings Layer---################
# product Multi-Layer-Perceptron embedding
product_perceptron_embedding = Embedding(n_products+1, n_latent_factors, name="Product-Perceptron-Embedding")(product_input)
product_perceptron_vec = Flatten(name="Flatten-Perceptron-Products")(product_perceptron_embedding)

# customer Multi-Layer-Perceptron embedding
customer_perceptron_embedding = Embedding(n_customers+1, n_latent_factors, name="Customer-Perceptron-Embedding")(customer_input)
customer_perceptron_vec = Flatten(name="Flatten-Perceptron-Customers")(customer_perceptron_embedding)

################---Multi-Layer-Perceptron embeddings concactenation Layer---################
# mlp_conc = Concatenate()([product_perceptron_vec, customer_perceptron_vec, helpful_votes_perceptron_vec])
mlp_conc = Concatenate()([product_perceptron_vec, customer_perceptron_vec])

################---Multi-Layer-Perceptron fully-connected Layers---################
fully_connected_layer0 = Dense(128, activation='relu')(mlp_conc)
fully_connected_layer1 = Dense(128, activation='relu')(fully_connected_layer0)
fully_connected_layer2 = Dense(64, activation='relu')(fully_connected_layer1)

################---Multi-Layer-Perceptron output Layer---################
mlp_output_layer = Dense(1, activation='relu')(fully_connected_layer2)


# # ################---Concactenation Layer---################
conc = Concatenate()([mf_output_layer, fully_connected_layer2])

fully_connected_layer3 = Dense(128, activation='relu')(Dense(64, activation='relu')((conc)))

# ################---Output Layer---################
output_layer = Dense(1)(fully_connected_layer3)


# Create model and compile
neural_mf_mlp_model = Model([customer_input, product_input], output_layer)
neural_mf_mlp_model.compile('adam',  loss='mean_squared_error')
```

</div>

<div class="cell code" data-colab="{&quot;height&quot;:850,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="R0ubOOxFq5MC" data-outputId="cf3da255-a261-4c0b-b9c6-ebeb17eaf11b">

``` python
neural_mf_mlp_model.summary()
```

<div class="output stream stdout">

    Model: "functional_17"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    Product-Input (InputLayer)      [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    Customer-Input (InputLayer)     [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    Product-Perceptron-Embedding (E (None, 1, 15)        2786640     Product-Input[0][0]              
    __________________________________________________________________________________________________
    Customer-Perceptron-Embedding ( (None, 1, 15)        32292390    Customer-Input[0][0]             
    __________________________________________________________________________________________________
    Flatten-Perceptron-Products (Fl (None, 15)           0           Product-Perceptron-Embedding[0][0
    __________________________________________________________________________________________________
    Flatten-Perceptron-Customers (F (None, 15)           0           Customer-Perceptron-Embedding[0][
    __________________________________________________________________________________________________
    concatenate (Concatenate)       (None, 30)           0           Flatten-Perceptron-Products[0][0]
                                                                     Flatten-Perceptron-Customers[0][0
    __________________________________________________________________________________________________
    Product-Embedding (Embedding)   (None, 1, 15)        2786640     Product-Input[0][0]              
    __________________________________________________________________________________________________
    Customer-Embedding (Embedding)  (None, 1, 15)        32292390    Customer-Input[0][0]             
    __________________________________________________________________________________________________
    dense_5 (Dense)                 (None, 128)          3968        concatenate[0][0]                
    __________________________________________________________________________________________________
    Flatten-Products (Flatten)      (None, 15)           0           Product-Embedding[0][0]          
    __________________________________________________________________________________________________
    Flatten-Customers (Flatten)     (None, 15)           0           Customer-Embedding[0][0]         
    __________________________________________________________________________________________________
    dense_6 (Dense)                 (None, 128)          16512       dense_5[0][0]                    
    __________________________________________________________________________________________________
    Dot-Product (Dot)               (None, 1)            0           Flatten-Products[0][0]           
                                                                     Flatten-Customers[0][0]          
    __________________________________________________________________________________________________
    dense_7 (Dense)                 (None, 64)           8256        dense_6[0][0]                    
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 65)           0           Dot-Product[0][0]                
                                                                     dense_7[0][0]                    
    __________________________________________________________________________________________________
    dense_10 (Dense)                (None, 64)           4224        concatenate_1[0][0]              
    __________________________________________________________________________________________________
    dense_9 (Dense)                 (None, 128)          8320        dense_10[0][0]                   
    __________________________________________________________________________________________________
    dense_11 (Dense)                (None, 1)            129         dense_9[0][0]                    
    ==================================================================================================
    Total params: 70,199,469
    Trainable params: 70,199,469
    Non-trainable params: 0
    __________________________________________________________________________________________________

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:857,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="2mYsPLEm13vY" data-outputId="38622f60-ebbc-4489-9752-2418c294ff42">

``` python
plot_model(neural_mf_mlp_model, to_file='neural_mf_mlp_model.png', show_shapes=True)
```

<div class="output execute_result" data-execution_count="91">

![](24bd5eb4df5ab346cc541898e3248c9bb844d5af.png)

</div>

</div>

<div class="cell code" id="VXrPaqXi1Xg_">

``` python
fit_plot_save(neural_mf_mlp_model, input=[train.customer_id, train.product_id], path='neural_mf_mlp_model.h5')
```

</div>

<div class="cell markdown" id="FXOYmuTeQ9J5">

</div>

<div class="cell markdown" id="QtZRgQOpn5SK">

### Meilleur Loss: 1.4707

### Epochs: 100

### Batch Size: 10000

### Temps de calcul: 12min

</div>

<div class="cell markdown" id="XS5CvERDpe5n">

### 3 - Evaluation des performances du model sur des données test

</div>

<div class="cell code" id="npBXoT65pwY2">

``` python
mf_model.evaluate([test.customer_id, test.product_id], test.star_rating)
```

</div>

<div class="cell code" id="BqN2cEpB7Y74">

``` python
neural_mf_model.evaluate([test.customer_id, test.product_id], test.star_rating)
```

</div>

<div class="cell code" id="AgoMSsmJ7Z_d">

``` python
neural_mf_mlp_model.evaluate([test.customer_id, test.product_id], test.star_rating)
```

</div>

<div class="cell markdown" id="e24GHlnwp1hf">

### Meilleur Loss Test: 1.7

</div>

<div class="cell code" id="t958xXcupwY6">

``` python
predictions = model.predict([test.customer_id.head(10), test.product_id.head(10)])

[print(predictions[i], test.star_rating.iloc[i]) for i in range(0,10)]
```

</div>

<div class="cell code" id="3_urcVZFVLLd">

``` python
test_norm.head(3)
```

</div>

<div class="cell code" id="Bdm764jlTYqP">

``` python
train.head(3)
```

</div>

<div class="cell code" id="2u5CgA6gP9wv">

``` python
c = 49988212	
p = 36194
model.predict([np.array([c]), np.array([p])])
```

</div>

<div class="cell code" id="e0qd1f5xQI3-">

``` python
c = 22536317
p = 109375
model.predict([np.array([c]), np.array([p])])
```

</div>

<div class="cell markdown" id="Ln0OKXv6pNnI">

### 4 - Visualisation des facteurs latents dans un espace à 2 dimensions

</div>

<div class="cell code" id="iaDsnJHto0vI">

``` python
# Extract embeddings
product_em = model.get_layer('Product-Embedding')
product_em_weights = product_em.get_weights()[0]
```

</div>

<div class="cell code" id="4Vvlfmd_o0va" data-outputId="41596cf8-2cbd-4766-e9dd-69916224ab33">

``` python
product_em_weights[:5]
```

<div class="output execute_result" data-execution_count="19">

    array([[-0.00610029,  0.0156836 ,  0.03038192,  0.0425691 , -0.02896588],
           [-1.5306774 ,  0.5250458 ,  2.366743  ,  1.3422308 , -1.0649071 ],
           [-1.2195548 ,  0.41940942,  2.508028  ,  1.8110565 , -1.1244056 ],
           [-0.5183603 ,  1.0415146 ,  2.5360963 ,  0.71309656, -1.3203545 ],
           [-1.3798604 , -0.9113507 ,  1.9190255 ,  1.5787991 ,  0.7250671 ]],
          dtype=float32)

</div>

</div>

<div class="cell code" id="IZPIvv2Fo0vq" data-outputId="8d7a7b09-3002-45a9-ccfe-3cfe0d3ff0f4">

``` python
from sklearn.decomposition import PCA
import seaborn as sns

pca = PCA(n_components=2)
pca_result = pca.fit_transform(product_em_weights)
sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1])
```

<div class="output execute_result" data-execution_count="20">

    <matplotlib.axes._subplots.AxesSubplot at 0x211935af9e8>

</div>

<div class="output display_data">

![](57f860b05fe7f1e76ae51c85f1c2598f08c25f58.png)

</div>

</div>

<div class="cell code" id="WXY4VRgGo0x7" data-outputId="5c51f7ad-32fc-4c72-cd9c-558e96883abe">

``` python
product_em_weights = product_em_weights / np.linalg.norm(product_em_weights, axis = 1).reshape((-1, 1))
product_em_weights[0][:10]
np.sum(np.square(product_em_weights[0]))
```

<div class="output execute_result" data-execution_count="21">

    1.0

</div>

</div>

<div class="cell code" id="4KWQIs87o0yZ" data-outputId="e5eeac95-be01-4a9c-8c7d-b0b4f598fad7">

``` python
pca = PCA(n_components=2)
pca_result = pca.fit_transform(product_em_weights)
sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1])
```

<div class="output execute_result" data-execution_count="22">

    <matplotlib.axes._subplots.AxesSubplot at 0x21193665f60>

</div>

<div class="output display_data">

![](ddb086b65c69c06bd00713936c48ba92184b18c9.png)

</div>

</div>

<div class="cell code" id="TpMDrURgo0zc" data-outputId="91a6e971-cad0-467c-85d1-ae3ac48e764c">

``` python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tnse_results = tsne.fit_transform(product_em_weights)
```

<div class="output stream stdout">

    [t-SNE] Computing 121 nearest neighbors...
    [t-SNE] Indexed 10001 samples in 0.007s...
    [t-SNE] Computed neighbors for 10001 samples in 0.840s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 10001
    [t-SNE] Computed conditional probabilities for sample 2000 / 10001
    [t-SNE] Computed conditional probabilities for sample 3000 / 10001
    [t-SNE] Computed conditional probabilities for sample 4000 / 10001
    [t-SNE] Computed conditional probabilities for sample 5000 / 10001
    [t-SNE] Computed conditional probabilities for sample 6000 / 10001
    [t-SNE] Computed conditional probabilities for sample 7000 / 10001
    [t-SNE] Computed conditional probabilities for sample 8000 / 10001
    [t-SNE] Computed conditional probabilities for sample 9000 / 10001
    [t-SNE] Computed conditional probabilities for sample 10000 / 10001
    [t-SNE] Computed conditional probabilities for sample 10001 / 10001
    [t-SNE] Mean sigma: 0.018763
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 85.958824
    [t-SNE] KL divergence after 300 iterations: 2.987510

</div>

</div>

<div class="cell code" id="T-v35b8Do0zu" data-outputId="6922f0bd-34fc-4681-d5f3-9c3352d8167c">

``` python
sns.scatterplot(x=tnse_results[:,0], y=tnse_results[:,1])
```

<div class="output execute_result" data-execution_count="24">

    <matplotlib.axes._subplots.AxesSubplot at 0x21193aefac8>

</div>

<div class="output display_data">

![](e10d0df435bbf62b4ba8b0003d981c8c5b953e6a.png)

</div>

</div>
