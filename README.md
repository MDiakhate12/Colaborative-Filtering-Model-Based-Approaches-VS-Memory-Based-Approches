# Colaborative-Filtering-Model-Based-Approaches-VS-Memory-Based-Approches
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Projet_Transversal-Matrix Factorization.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "toc_visible": true,
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/MDiakhate12/Colaborative-Filtering-Model-Based-Approaches-VS-Memory-Based-Approches/blob/main/Projet_Transversal_Matrix_Factorization.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_Wp2p0AGr_Nr"
      },
      "source": [
        "!curl \"https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip\" -o \"awscliv2.zip\"\n",
        "!unzip awscliv2.zip\n",
        "!sudo ./aws/install\n",
        "\n",
        "!aws --version "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "M9Y6uzS1r_Nv",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        },
        "outputId": "6e9ddd17-b6bc-49d5-a606-8228fa33933d"
      },
      "source": [
        "#AKIAQBKLXKEALPR7EXOO\n",
        "#J5Rh5E1Y5galwVym3mYuHPySrp/wU39pIRMW1hOz\n",
        "!aws configure --output json --region us-west-2\n",
        "\n",
        "!aws s3 cp s3://amazon-reviews-pds/tsv/amazon_reviews_us_Electronics_v1_00.tsv.gz ./"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "AWS Access Key ID [None]: AKIAQBKLXKEALPR7EXOO\n",
            "AWS Secret Access Key [None]: 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\n",
            "Default region name [None]: \n",
            "Default output format [None]: \n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xst0KUykr_Ny",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 258
        },
        "outputId": "e022a58b-6c32-4cf1-b894-308d2b6e2aa3"
      },
      "source": [
        "!pip install pyspark"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Collecting pyspark\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/8e/b0/bf9020b56492281b9c9d8aae8f44ff51e1bc91b3ef5a884385cb4e389a40/pyspark-3.0.0.tar.gz (204.7MB)\n",
            "\u001b[K     |████████████████████████████████| 204.7MB 61kB/s \n",
            "\u001b[?25hCollecting py4j==0.10.9\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/9e/b6/6a4fb90cd235dc8e265a6a2067f2a2c99f0d91787f06aca4bcf7c23f3f80/py4j-0.10.9-py2.py3-none-any.whl (198kB)\n",
            "\u001b[K     |████████████████████████████████| 204kB 41.3MB/s \n",
            "\u001b[?25hBuilding wheels for collected packages: pyspark\n",
            "  Building wheel for pyspark (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for pyspark: filename=pyspark-3.0.0-py2.py3-none-any.whl size=205044182 sha256=533e7070666011b7ff5daed33a64efe9b07d8cbf3360d1439893d3b88cf4bcb7\n",
            "  Stored in directory: /root/.cache/pip/wheels/57/27/4d/ddacf7143f8d5b76c45c61ee2e43d9f8492fc5a8e78ebd7d37\n",
            "Successfully built pyspark\n",
            "Installing collected packages: py4j, pyspark\n",
            "Successfully installed py4j-0.10.9 pyspark-3.0.0\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Ea7ceBhpGy0o"
      },
      "source": [
        "#  Introduction"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "H1oJzVwVGy0p"
      },
      "source": [
        "Recommendation systems are in tons of things you interact with every day. Amazon, Spotify, and Facebook are some of the biggest players, and they're using all the data they can to suggest products that they think you'll love.\n",
        "\n",
        "<img src =\"./trends.png\" />\n",
        "\n",
        "Some companies have teams of people collection, cleaning, and building models around this data. However, with a few useful Python packages and some great data from Amazon's customer review dataset, Were're going to build a recommendation system with different collaborative filetring methods."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3VP6RZq3Gy0r"
      },
      "source": [
        "<hr>\n",
        "\n",
        "# Data Preprocessing"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lM3Ig067Gy0s"
      },
      "source": [
        "## Loading Data"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ZgxvRXu6Gy0t"
      },
      "source": [
        "import pandas as pd\n",
        "import sys\n",
        "from pyspark.sql import SparkSession\n",
        "from pyspark import SparkContext, SparkConf\n",
        "from pyspark.sql.types import StructType, StructField, IntegerType, StringType"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ujm1ZbjaGy0y",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        },
        "outputId": "2ca5dadd-0a84-4d34-e244-f3a5bd07353d"
      },
      "source": [
        "# create a spark session\n",
        "# using local[*] to run Spark locally with as many worker threads as logical cores has your machine.\n",
        "conf = SparkConf().setAppName(\"Projet Transversal Matrix Factorisation\").setMaster(\"local[*]\")\n",
        "sc = SparkContext.getOrCreate(conf)\n",
        "spark = SparkSession(sc)\n",
        "\n",
        "# spark and python version\n",
        "print(spark.version)\n",
        "print(sys.version)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "3.0.0\n",
            "3.6.9 (default, Jul 17 2020, 12:50:27) \n",
            "[GCC 8.4.0]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zzOiCBmBGy01"
      },
      "source": [
        "### [DATA COLUMNS](https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt)\n",
        "\n",
        "<pre>marketplace       - 2 letter country code of the marketplace where the review was written.\n",
        "customer_id       - Random identifier that can be used to aggregate reviews written by a single author.\n",
        "review_id         - The unique ID of the review.\n",
        "product_id        - The unique Product ID the review pertains to. In the multilingual dataset the reviews\n",
        "                    for the same product in different countries can be grouped by the same product_id.\n",
        "product_parent    - Random identifier that can be used to aggregate reviews for the same product.\n",
        "product_title     - Title of the product.\n",
        "product_category  - Broad product category that can be used to group reviews \n",
        "                    (also used to group the dataset into coherent parts).\n",
        "star_rating       - The 1-5 star rating of the review.\n",
        "helpful_votes     - Number of helpful votes.\n",
        "total_votes       - Number of total votes the review received.\n",
        "vine              - Review was written as part of the Vine program.\n",
        "verified_purchase - The review is on a verified purchase.\n",
        "review_headline   - The title of the review.\n",
        "review_body       - The review text.\n",
        "review_date       - The date the review was written.\n",
        "</pre>\n",
        "### DATA FORMAT\n",
        "Tab ('\\t') separated text file, without quote or escape characters.\n",
        "First line in each file is header; 1 line corresponds to 1 record."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "b_xiktSYGy02"
      },
      "source": [
        "path = 'amazon_reviews_us_Electronics_v1_00.tsv.gz'\n",
        "\n",
        "df_plain = spark.read.csv(path, sep='\\t', header=True)\n",
        "df = df_plain.select(\"customer_id\", \"product_id\", \"star_rating\",\"product_title\",\"product_parent\",\"review_date\",)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qrYT6lGSGy05",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 241
        },
        "outputId": "e764ac22-cba2-4e25-a6ad-48d4d51f4ee7"
      },
      "source": [
        "df_plain.show(5)\n",
        "df_plain.count()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "+-----------+-----------+--------------+----------+--------------+--------------------+----------------+-----------+-------------+-----------+----+-----------------+--------------------+--------------------+-----------+\n",
            "|marketplace|customer_id|     review_id|product_id|product_parent|       product_title|product_category|star_rating|helpful_votes|total_votes|vine|verified_purchase|     review_headline|         review_body|review_date|\n",
            "+-----------+-----------+--------------+----------+--------------+--------------------+----------------+-----------+-------------+-----------+----+-----------------+--------------------+--------------------+-----------+\n",
            "|         US|   41409413|R2MTG1GCZLR2DK|B00428R89M|     112201306|yoomall 5M Antenn...|     Electronics|          5|            0|          0|   N|                Y|          Five Stars|       As described.| 2015-08-31|\n",
            "|         US|   49668221|R2HBOEM8LE9928|B000068O48|     734576678|Hosa GPM-103 3.5m...|     Electronics|          5|            0|          0|   N|                Y|It works as adver...|It works as adver...| 2015-08-31|\n",
            "|         US|   12338275|R1P4RW1R9FDPEE|B000GGKOG8|     614448099|Channel Master Ti...|     Electronics|          5|            1|          1|   N|                Y|          Five Stars|         Works pissa| 2015-08-31|\n",
            "|         US|   38487968|R1EBPM82ENI67M|B000NU4OTA|      72265257|LIMTECH Wall char...|     Electronics|          1|            0|          0|   N|                Y|            One Star|Did not work at all.| 2015-08-31|\n",
            "|         US|   23732619|R372S58V6D11AT|B00JOQIO6S|     308169188|Skullcandy Air Ra...|     Electronics|          5|            1|          1|   N|                Y|Overall pleased w...|Works well. Bass ...| 2015-08-31|\n",
            "+-----------+-----------+--------------+----------+--------------+--------------------+----------------+-----------+-------------+-----------+----+-----------------+--------------------+--------------------+-----------+\n",
            "only showing top 5 rows\n",
            "\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "3093869"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1ND6LbEkGy07",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 340
        },
        "outputId": "04298e8a-edeb-480f-825b-acdf0c479d66"
      },
      "source": [
        "df.show(5)\n",
        "\n",
        "# display data types and check if they are correct, e.g. rating should be double\n",
        "df.printSchema()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "+-----------+----------+-----------+--------------------+--------------+-----------+\n",
            "|customer_id|product_id|star_rating|       product_title|product_parent|review_date|\n",
            "+-----------+----------+-----------+--------------------+--------------+-----------+\n",
            "|   41409413|B00428R89M|          5|yoomall 5M Antenn...|     112201306| 2015-08-31|\n",
            "|   49668221|B000068O48|          5|Hosa GPM-103 3.5m...|     734576678| 2015-08-31|\n",
            "|   12338275|B000GGKOG8|          5|Channel Master Ti...|     614448099| 2015-08-31|\n",
            "|   38487968|B000NU4OTA|          1|LIMTECH Wall char...|      72265257| 2015-08-31|\n",
            "|   23732619|B00JOQIO6S|          5|Skullcandy Air Ra...|     308169188| 2015-08-31|\n",
            "+-----------+----------+-----------+--------------------+--------------+-----------+\n",
            "only showing top 5 rows\n",
            "\n",
            "root\n",
            " |-- customer_id: string (nullable = true)\n",
            " |-- product_id: string (nullable = true)\n",
            " |-- star_rating: string (nullable = true)\n",
            " |-- product_title: string (nullable = true)\n",
            " |-- product_parent: string (nullable = true)\n",
            " |-- review_date: string (nullable = true)\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rMYUnFaCGy0-",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 187
        },
        "outputId": "2b92d557-e751-432a-8c91-72a3eb891db0"
      },
      "source": [
        "# show summary\n",
        "df.describe().show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "+-------+--------------------+--------------------+------------------+--------------------+--------------------+-----------+\n",
            "|summary|         customer_id|          product_id|       star_rating|       product_title|      product_parent|review_date|\n",
            "+-------+--------------------+--------------------+------------------+--------------------+--------------------+-----------+\n",
            "|  count|             3093869|             3093869|           3093861|             3093869|             3093869|    3093750|\n",
            "|   mean|2.8789345281890087E7|3.1505156710778136E9| 4.035506443243571|   2247.254098360656|5.1020043971820265E8|       null|\n",
            "| stddev|1.5430609004342766E7| 3.551824947508875E9|1.3874382233284288|  1721.9748059774513| 2.868316246944571E8|       null|\n",
            "|    min|            10000013|          0141186178|                 1|\" Burst \" Variabl...|           100007543| 1999-06-09|\n",
            "|    max|             9999987|          BT008V9J9U|                 5|☆ Power Adapter ◘...|           999998189| 2015-08-31|\n",
            "+-------+--------------------+--------------------+------------------+--------------------+--------------------+-----------+\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DCPt1unYGy1C"
      },
      "source": [
        "## Removing missing values\n",
        "\n",
        "As the summary shown there are some rows with empty rating and they've to be removed. In fact the count for `customer_id` and `product_id` is the same (`3093869`) but it's less than the `star_rating` column's count (`3093861`)"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "4nMQdsA_Gy1D",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 187
        },
        "outputId": "30d05096-a91a-40c8-b610-915888744372"
      },
      "source": [
        "df = df.na.drop(subset=[\"star_rating\"])\n",
        "\n",
        "df.describe().show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "+-------+--------------------+--------------------+------------------+--------------------+-------------------+-----------+\n",
            "|summary|         customer_id|          product_id|       star_rating|       product_title|     product_parent|review_date|\n",
            "+-------+--------------------+--------------------+------------------+--------------------+-------------------+-----------+\n",
            "|  count|             3093861|             3093861|           3093861|             3093861|            3093861|    3093750|\n",
            "|   mean|2.8789345585748035E7|3.1505156710778136E9| 4.035506443243571|   2247.254098360656|5.102008536329082E8|       null|\n",
            "| stddev|1.5430618553487921E7| 3.551824947508875E9|1.3874382233284288|  1721.9748059774513|2.868316559225427E8|       null|\n",
            "|    min|            10000013|          0141186178|                 1|\" Burst \" Variabl...|          100007543| 1999-06-09|\n",
            "|    max|             9999987|          BT008V9J9U|                 5|☆ Power Adapter ◘...|          999998189| 2015-08-31|\n",
            "+-------+--------------------+--------------------+------------------+--------------------+-------------------+-----------+\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tqvFFrfCGy1G"
      },
      "source": [
        "## Removing duplicates\n",
        "\n",
        "There could be some duplicated rows in here. To check, I'll see if there are any row duplicated in the data."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": true,
        "id": "LWKwVOq_Gy1G",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "077d2b24-ca3c-411f-abbe-7817ed0f81bf"
      },
      "source": [
        "num_duplicates = df.count() - df.dropDuplicates(['customer_id', 'product_id']).count()\n",
        "\n",
        "if num_duplicates > 0:\n",
        "    print('Data has {} duplicates'.format(num_duplicates))\n",
        "    df_clean = df.dropDuplicates()\n",
        "else : \n",
        "    print(\"Data hasn't duplicates\")"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Data has 1705 duplicates\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "54n5JboXGy1L"
      },
      "source": [
        "## Checking ratings distribution "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "yqHR4eNz_Wrm",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        },
        "outputId": "d94eb902-1577-4b30-f89f-a3c2bf240c9c"
      },
      "source": [
        "print(\"There is {} products\\n\".format(df_clean.select(\"product_id\").distinct().count()))\n",
        "print(\"There is {} customers\".format(df_clean.select(\"customer_id\").distinct().count()))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "There is 185848 products\n",
            "\n",
            "There is 2154352 customers\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": true,
        "id": "7OScgEDUGy1M"
      },
      "source": [
        "reviews_count = df_clean.groupBy('customer_id').count().withColumnRenamed(\"count\",\"reviews\")\n",
        "#reviews_count.show()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": true,
        "id": "qiKdNX4ZGy1O",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 459
        },
        "outputId": "fce4b56f-ca43-4a46-8447-11afc1d6d8d6"
      },
      "source": [
        "reviews_freq = reviews_count.groupBy('reviews').count().sort(\"reviews\").withColumnRenamed(\"count\",\"frequency\")\n",
        "reviews_freq.show()\n",
        "#reviews_freq.select(\"frequency\").groupBy().sum().show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "+-------+---------+\n",
            "|reviews|frequency|\n",
            "+-------+---------+\n",
            "|      1|  1688517|\n",
            "|      2|   281981|\n",
            "|      3|    91371|\n",
            "|      4|    38768|\n",
            "|      5|    19638|\n",
            "|      6|    11086|\n",
            "|      7|     6702|\n",
            "|      8|     4215|\n",
            "|      9|     2970|\n",
            "|     10|     2122|\n",
            "|     11|     1399|\n",
            "|     12|     1059|\n",
            "|     13|      817|\n",
            "|     14|      649|\n",
            "|     15|      509|\n",
            "|     16|      423|\n",
            "|     17|      308|\n",
            "|     18|      246|\n",
            "|     19|      201|\n",
            "|     20|      182|\n",
            "+-------+---------+\n",
            "only showing top 20 rows\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "oBKvAModGy1Q"
      },
      "source": [
        "The last table shows us that `1688102` users have been rating only once .In other word `50%` of users have rated only one item. Hence, **this shows how much the the interation matrix is sparse**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "x2xzE7tPGy1T"
      },
      "source": [
        "## Drop unused columns"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "W27W9Z29Gy1T"
      },
      "source": [
        "df = df_clean.drop('product_parent', 'review_date',  'product_title')\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "KotlALtyGy1V"
      },
      "source": [
        "## Train Test Split"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "v7vyPYJ0Gy1V"
      },
      "source": [
        "\n",
        "seed=42\n",
        "df_train, df_test = df.randomSplit([0.8, 0.2], seed)\n",
        "\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "JH3f8m3iGy1X"
      },
      "source": [
        "<hr>\n",
        "\n",
        "# Memory based Collabortive filtering "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uVE8as4nGy1X",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        },
        "outputId": "5a868f68-f617-43dc-c4c3-15755667cea5"
      },
      "source": [
        "df_train.show(5)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "+-----------+----------+-----------+\n",
            "|customer_id|product_id|star_rating|\n",
            "+-----------+----------+-----------+\n",
            "|   10005635|B0077V88V8|          3|\n",
            "|   10006647|B002S53LJ2|          2|\n",
            "|   10011241|B003SVHYAC|          5|\n",
            "|   10011614|B0083C8AWW|          2|\n",
            "|   10012362|B001JTQUYG|          2|\n",
            "+-----------+----------+-----------+\n",
            "only showing top 5 rows\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "smHhDsBuGy1Z"
      },
      "source": [
        "## Item based approach"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "sMpO0iM2Gy1Z"
      },
      "source": [
        "# Item-based Collaborative Filtering on pySpark \n",
        "\n",
        "from collections import defaultdict\n",
        "from itertools import combinations\n",
        "import numpy as np\n",
        "import random\n",
        "import csv\n",
        "from scipy.stats import pearsonr\n",
        "\n",
        "def parseRowOnUser(row):\n",
        "    '''\n",
        "    Parse each row of the specified list.\n",
        "    Converts each rating to a float\n",
        "    '''\n",
        "    return row.customer_id,(row.product_id,float(row.star_rating))\n",
        "\n",
        "def sampleInteractions(user_id,items_with_rating,n):\n",
        "    '''\n",
        "    For users with # interactions > n, replace their interaction history\n",
        "    with a sample of n items_with_rating\n",
        "    '''\n",
        "    if len(items_with_rating) > n:\n",
        "        return user_id, random.sample(items_with_rating,n)\n",
        "    else:\n",
        "        return user_id, items_with_rating\n",
        "\n",
        "def calcSim(item_pair,rating_pairs):\n",
        "    ''' \n",
        "    Inputs : \n",
        "        item_pair : (item1,item2)   \n",
        "        rating_pairs : [(item1_rating1,item2_rating1),(item1_rating2,item2_rating2)...]\n",
        "    Output : \n",
        "        (item1,item2) (pearson_corr,co_raters_count )\n",
        "    For each item-item pair, return the specified similarity measure,\n",
        "    along with co_raters_count\n",
        "    '''\n",
        "    rating_pairs = [rating_pair for rating_pair in rating_pairs]\n",
        "    '''\n",
        "    with open ('test','w') as f :\n",
        "        f.write(str(rating_pairs))\n",
        "        f.write('-')\n",
        "    '''\n",
        "    item1_ratings, item2_ratings = np.array(rating_pairs).T #decoupling item1_ratings & item2_ratings\n",
        "    \n",
        "    corr, _ = pearsonr(item1_ratings, item2_ratings)\n",
        "    #corr = 0 if np.isnan(corr) else corr\n",
        "    \n",
        "    return item_pair, (corr,len(item1_ratings))\n",
        "\n",
        "def keyOnFirstFoo(foo_pair,foo_sim_data):\n",
        "    '''\n",
        "    For each foo-foo pair, make the first item's id the key \n",
        "    foo can be either item or user \n",
        "    '''\n",
        "    (foo1_id,foo2_id) = foo_pair\n",
        "    return foo1_id,(foo2_id,foo_sim_data)\n",
        "\n",
        "def nearestNeighbors(item_id,items_and_sims,n,strategy=\"top-N\",threshold=0.5):\n",
        "    '''\n",
        "    Sort the predictions list by similarity and select the top-N neighbors \n",
        "    '''\n",
        "    '''\n",
        "    REMEMBER : \n",
        "    A cosine value of 0 means that the two vectors are at 90 degrees to each other (orthogonal) and have no match. \n",
        "    The closer the cosine value to 1, the smaller the angle and the greater the match between vectors.\n",
        "    \n",
        "    The Pearson correlation coefficient, r, can take a range of values from +1 to -1.\n",
        "    A value of 0 indicates that there is no association between the two variables. \n",
        "    A value greater than 0 indicates a positive association\n",
        "    '''\n",
        "    if strategy ==\"threshold\" :\n",
        "        items_and_sims = [number for number in test if number[1][0] > threshold]\n",
        "    else : \n",
        "        items_and_sims.sort(key=lambda x: x[1][0],reverse=True)\n",
        "    \n",
        "    return item_id, items_and_sims[:n]\n",
        "\n",
        "\n",
        "\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "aO4Pp2xOGy1e"
      },
      "source": [
        "#df_train1,_ = df_train.randomSplit([0.4,0.6],seed)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Lzg-1RQscf1s"
      },
      "source": [
        "### 1- Obtain the sparse user-item matrix:\n",
        "    user_id -> [(item_id_1, rating_1),\n",
        "               [(item_id_2, rating_2),\n",
        "                ...]"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": false,
        "id": "DfRN-sl1Gy1f"
      },
      "source": [
        "''' \n",
        "Obtain the sparse user-item matrix:\n",
        "    user_id -> [(item_id_1, rating_1),\n",
        "               [(item_id_2, rating_2),\n",
        "                ...]\n",
        "'''\n",
        "\n",
        "#user_item_pairs = df_train.rdd.filter(lambda r : r[0]==\"10000297\")\n",
        "\n",
        "user_item_pairs = df_train1.rdd.map(parseRowOnUser).groupByKey().map(\n",
        "    lambda p: sampleInteractions(p[0],list(p[1]),500)).cache()\n",
        "\n",
        "#user_item_pairs.take(5)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ueTBaqZhc8zg"
      },
      "source": [
        "### 2- Get all item-item pair combos:\n",
        "    (item1,item2) ->    [(item1_rating,item2_rating),\n",
        "                         (item1_rating,item2_rating),\n",
        "                         ...]"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "KEgeXMGUGy1i",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "16617752-308b-43a0-8983-1af89582ee95"
      },
      "source": [
        "%%time\n",
        "'''\n",
        "Get all item-item pair combos:\n",
        "    (item1,item2) ->    [(item1_rating,item2_rating),\n",
        "                         (item1_rating,item2_rating),\n",
        "                         ...]\n",
        "'''\n",
        "\n",
        "def findItemPairs(user_id,items_with_rating):\n",
        "    '''\n",
        "    For each user, find all item-item pairs combos. (i.e. items rated by the same user) \n",
        "    '''\n",
        "    itemPairsList =[((item1[0],item2[0]),(item1[1],item2[1])) for item1,item2 in combinations(items_with_rating,2)]\n",
        "    return itemPairsList\n",
        "\n",
        "pairwise_items = user_item_pairs.map(\n",
        "    lambda p: findItemPairs(p[0],p[1])).flatMap(\n",
        "    lambda p: (items_with_rating_pair for items_with_rating_pair in p)).groupByKey().filter(\n",
        "    lambda p: len(p[1]) > 1) #filtering out item_pair with only 1 rating_pair (not enough data to compute sim for them #SG)\n",
        "\n",
        "\n",
        "\n",
        "def iterate(iterable):\n",
        "    return [v1_iterable for v1_iterable in iterable ]\n",
        "\n",
        "##pairwise_items.take(1).foreach(lambda row: print(row[0],row[1].collect))\n",
        "#[print(row[0],iterate(row[1])) for row in pairwise_items.take(2)] \n",
        "#pairwise_items.toDF([\"item1_item2\",\"rating1_rating\"]).show(7)\n",
        "#pairwise_items.take(2)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "CPU times: user 11.9 ms, sys: 2.02 ms, total: 14 ms\n",
            "Wall time: 28.1 ms\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "rgN2CKF9dVJr"
      },
      "source": [
        "### 3- Calculate the similarity for each item pair and select the top-N nearest neighbors:\n",
        "    (item1,item2) ->    (similarity,co_raters_count)"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": true,
        "id": "HvTmUwg2Gy1j",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e8583d3d-cb5f-4375-c1a4-d00bda3e5361"
      },
      "source": [
        "%%time\n",
        "'''\n",
        "Calculate the similarity for each item pair and select the top-N nearest neighbors:\n",
        "    (item1,item2) ->    (similarity,co_raters_count)\n",
        "'''\n",
        "#Calculate the cosine similarity for each item pair \n",
        "# (item1,item2) ->    (similarity,co_raters_count)\n",
        "print(\"Computing the similarity for each item pair...\")\n",
        "top_n = 30\n",
        "item_sims = pairwise_items.map( #Computing sim for each pair of item\n",
        "        lambda p: calcSim(p[0],p[1])).filter(\n",
        "        lambda p: np.isnan(p[1][0])==False).map(    #(item1,item2) (pearson_corr,co_raters_count )\n",
        "        lambda p: keyOnFirstFoo(p[0],p[1])).groupByKey().map(  # (item1) [(item2,(pearson_corr,co_raters_count )),... (itemM,(pearson_corr,co_raters_count ))] m item\n",
        "        lambda p: nearestNeighbors(p[0],list(p[1]),top_n)).collect() # (item1) [(item2,(pearson_corr,co_raters_count )),... (itemN,(pearson_corr,co_raters_count ))] top n item\n",
        "\n",
        "# collecting it into a list \n",
        "#item_sims =item_sims.collect()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Computing the similarity for each item pair...\n",
            "CPU times: user 121 ms, sys: 22.9 ms, total: 144 ms\n",
            "Wall time: 49.8 s\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "MbULlb7pGy1l",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "6199ec30-9a59-4dba-e6d5-40cb19cc15b7"
      },
      "source": [
        "%%time\n",
        "'''\n",
        "Preprocess the item similarity matrix into a dictionary and store it as a broadcast variable:\n",
        "'''\n",
        "\n",
        "item_sim_dict = {}\n",
        "for (item,data) in item_sims: \n",
        "    item_sim_dict[item] = data\n",
        "\n",
        "isb = sc.broadcast(item_sim_dict)\n",
        "\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "CPU times: user 94.6 ms, sys: 16 ms, total: 111 ms\n",
            "Wall time: 127 ms\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GuC0wK20Gy1m",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "76c451a8-cc66-4cd0-aecd-dbaedc6bc7c5"
      },
      "source": [
        "%%time\n",
        "'''\n",
        "    Obtain the sparse item-user matrix:\n",
        "        item_id -> ((user_1,rating),(user2,rating))\n",
        "'''\n",
        "\n",
        "def parseRowOnItem(row):\n",
        "    return row.product_id,(row.customer_id,float(row.star_rating))\n",
        "\n",
        "item_user_pairs = df_train1.rdd.map(parseRowOnItem).groupByKey().map(\n",
        "    lambda p: sampleInteractions(p[0],list(p[1]),500)).cache()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "CPU times: user 16.7 ms, sys: 979 µs, total: 17.7 ms\n",
            "Wall time: 49.8 ms\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": true,
        "id": "gEUT7WKHGy1n",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "9497a6ea-aeb5-4ec4-aa4e-f9043b503550"
      },
      "source": [
        "%%time\n",
        "def fooAverageRating(foos_pairs):\n",
        "    \"\"\"\n",
        "        Obtain foo_rating_average matrix containing average rating for each foo:\n",
        "        dict : (foo_id) -> (avg_rating)\n",
        "        where foo can be items or users and foos is either item_user or user_item\n",
        "    \"\"\"\n",
        "    \n",
        "    def averageRating(row):\n",
        "        '''\n",
        "        Compute the average rating of an item\n",
        "        '''\n",
        "        _, ratings =np.array(row[1],dtype = object).T\n",
        "        return row[0],np.mean(ratings)\n",
        "\n",
        "    return dict(foos_pairs.map(averageRating).collect())\n",
        "\n",
        "irab = sc.broadcast(fooAverageRating(item_user_pairs))\n",
        "#item_user_pairs_avg_b.value"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "CPU times: user 192 ms, sys: 37.8 ms, total: 230 ms\n",
            "Wall time: 1min 29s\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9sW1z921d6vJ"
      },
      "source": [
        "### 4- Calculate the top-N item recommendations for each user\n",
        "    user_id -> [item1,item2,item3,...]"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "o89yjNQHGy1o",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "7f56143b-190a-4cb4-9ccc-03d5ea3ce54e"
      },
      "source": [
        "%%time\n",
        "def topNRecommendations(user_id,items_with_rating,item_sims,items_rating_average,n):\n",
        "    '''\n",
        "        user_id \n",
        "        items_with_rating : [(item_id_1, rating_1),\n",
        "                   (item_id_2, rating_2)]\n",
        "        item_sims : dict = {'item_id_i': [(item_id_j, (sim(item_id_i,item_id_j), co_raters_count)), (item_id_k, (sim(item_id_i,item_id_k), co_raters_count))],...} \n",
        "        items_rating_average : dict = {'item_id_i': avg ...}\n",
        "    Calculate the top-N item recommendations for each user using the \n",
        "    weighted sums method\n",
        "    '''\n",
        "\n",
        "    # initialize dicts to store the score of each individual item,\n",
        "    # since an item can exist in more than one item neighborhood\n",
        "    totals = defaultdict(int)\n",
        "    sim_sums = defaultdict(int)\n",
        "\n",
        "    for (item,rating) in items_with_rating:\n",
        "\n",
        "        # lookup the nearest neighbors for this item\n",
        "        nearest_neighbors = item_sims.get(item,None)\n",
        "\n",
        "        if nearest_neighbors:\n",
        "            for (neighbor,(sim,count)) in nearest_neighbors:\n",
        "                if neighbor != item: #ensure that it will not recommend items already rated #SG\n",
        "\n",
        "                    # update totals and sim_sums with the rating data\n",
        "                    totals[neighbor] += sim * (rating-items_rating_average[neighbor])\n",
        "                    sim_sums[neighbor] += abs(sim)\n",
        "\n",
        "    # create the normalized list of scored items \n",
        "    scored_items = []\n",
        "    for item,total in totals.items():\n",
        "        \n",
        "        # strategy 1\n",
        "        division = 0 if sim_sums[item]==0 else total/sim_sums[item]\n",
        "        scored_items.append((division+items_rating_average[item],item)) \n",
        "        '''\n",
        "        # strategy 2\n",
        "        if (sim_sums[item]!=0 ):\n",
        "            scored_items.append((total/sim_sums[item]+items_rating_average[item],item)) \n",
        "        '''\n",
        "    # sort the scored items in ascending order\n",
        "    scored_items.sort(reverse=True)\n",
        "\n",
        "    # take out the item score\n",
        "    # ranked_items = [x[1] for x in scored_items]\n",
        "\n",
        "    return user_id,scored_items[:n]\n",
        "\n",
        "\n",
        "'''\n",
        "Calculate the top-N item recommendations for each user\n",
        "    user_id -> [item1,item2,item3,...]\n",
        "'''\n",
        "user_item_based_recs = user_item_pairs.map(\n",
        "    lambda p: topNRecommendations(p[0],p[1],isb.value,irab.value,500)).collect()\n",
        "\n",
        "#user_item_recs.toDF([\"user_id\",\"recommendations\"]).show(5)\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "CPU times: user 19.4 s, sys: 5.66 s, total: 25 s\n",
            "Wall time: 2min 18s\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uFP79Z-BGy1q"
      },
      "source": [
        "def recommend(user_id,user_item_pairs,sim_matrix):\n",
        "    '''\n",
        "    Compute the top n recommended items'score for the user with user_id\n",
        "    return a list\n",
        "    '''\n",
        "    prediction =  user_item_pairs.filter(lambda r : r[0]==user_id).map(\n",
        "        lambda p: topNRecommendations(p[0],p[1],sim_matrix.value,500))\n",
        "    predictions = prediction.collect()\n",
        "    predictions = predictions[0][1] if len(predictions)>0 else None\n",
        "    return predictions\n",
        "\n",
        "#print(recommend (\"13357\",user_item_pairs,isb))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pEPfUlGcTkQf"
      },
      "source": [
        "# read in the test data\n",
        "df_test_list =df_test.rdd.collect() \n",
        "df_train_list = df_train.collect()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fr142507guCi"
      },
      "source": [
        "### 5- Performance"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "G0ifbsOuGy1r",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 153
        },
        "outputId": "832bd40f-8ba2-4790-9376-efe645e9c81a"
      },
      "source": [
        "%%time\n",
        "'''\n",
        "Read in test data and calculate specified metric \n",
        "'''\n",
        "from sklearn.metrics import mean_squared_error,mean_absolute_error\n",
        "\n",
        "def evaluate(df_test,df_train,user_item_recs,metric=None):\n",
        "    test_ratings = defaultdict(list)\n",
        "    train_ratings = defaultdict(list)\n",
        "    \n",
        "    for row in df_test:\n",
        "        user = row.customer_id\n",
        "        item = row.product_id\n",
        "        rating = row.star_rating\n",
        "        test_ratings[user] += [(item,rating)] \n",
        "        \n",
        "    for row in df_train:\n",
        "        user = row.customer_id\n",
        "        item = row.product_id\n",
        "        rating = row.star_rating\n",
        "        train_ratings[user] += [(item,rating)]  \n",
        "      \n",
        "    def compute(foo_ratings) :\n",
        "        # create train-test rating tuples\n",
        "        y_pred = []\n",
        "        y_true = []\n",
        "        for (user,items_with_rating) in user_item_recs:\n",
        "            for (rating,item) in items_with_rating:\n",
        "                for (test_item,test_rating) in foo_ratings[user]:                \n",
        "                    if str(test_item) == str(item):\n",
        "                        y_true.append(float(test_rating))\n",
        "                        y_pred.append(rating)\n",
        "        #print(len(y_pred),'\\n')\n",
        "        if(metric=='rmse') :\n",
        "            return mean_squared_error(y_true, y_pred)\n",
        "        elif(metric=='mae'):\n",
        "            return mean_absolute_error(y_true, y_pred)\n",
        "        else : \n",
        "            return (mean_absolute_error(y_true, y_pred),mean_squared_error(y_true, y_pred))\n",
        "        \n",
        "    return compute(train_ratings) , compute(test_ratings)\n",
        "\n",
        "\n",
        "\n",
        "result = (evaluate(df_test_list,df_train_list,user_item_based_recs))\n",
        "print (\"MAE: \\n Train : {} \\n Test  : {}\\nRMSE: \\n Train : {} \\n Test  : {}\".format(result[0][0],result[1][0],result[0][1],result[1][1] ))\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "#Mean Absolute Error:  0.644232031607937 # strategy 2 TopNrecomm\n",
        "#Mean Absolute Error:  0.6334624623392405  # strategy 1 TopNrecomm - 1.0726394395209509\n",
        "#Mean Absolute Error:  0.6288800241389374 in df_test\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "MAE: \n",
            " Train : 0.665225623907677 \n",
            " Test  : 0.6935194429936145\n",
            "RMSE: \n",
            " Train : 1.0973471433958188 \n",
            " Test  : 1.3140512654094085\n",
            "CPU times: user 41 s, sys: 454 ms, total: 41.5 s\n",
            "Wall time: 41.5 s\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "1Z12II3kGy1t"
      },
      "source": [
        "## User based approach"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "AT6gZQxWGy1t"
      },
      "source": [
        "'''\n",
        "Get all user-user pair combos:\n",
        "    (user1_id,user2_id) -> [(rating1,rating2),\n",
        "                            (rating1,rating2),\n",
        "                            (rating1,rating2),\n",
        "                            ...]\n",
        "'''\n",
        "\n",
        "def findUserPairs(item_id,users_with_rating):\n",
        "    '''\n",
        "    For each item, find all user-user pairs combos. (i.e. users with the same item) \n",
        "    '''\n",
        "    userPairsList =[((user1[0],user2[0]),(user1[1],user2[1])) for user1,user2 in combinations(users_with_rating,2)]\n",
        "    return userPairsList\n",
        "pairwise_users = item_user_pairs.map(\n",
        "    lambda p: findUserPairs(p[0],p[1])).flatMap(\n",
        "    lambda p: (users_with_rating_pair for users_with_rating_pair in p)).groupByKey().filter(\n",
        "    lambda p: len(p[1]) > 1) #filtering out user_pair with only 1 rating_pair (not enough data to compute sim for them #SG)\n",
        "\n",
        "\n",
        "##pairwise_items.take(1).foreach(lambda row: print(row[0],row[1].collect))\n",
        "#[print(row[0],iterate(row[1])) for row in pairwise_items.take(2)] \n",
        "#pairwise_items.toDF([\"item1_item2\",\"rating1_rating\"]).show(7)\n",
        "#pairwise_users.take(2)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hlKmbeKvGy1u"
      },
      "source": [
        "#[print(row[0],iterate(row[1])) for row in pairwise_items.take(2)] "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9tOyFkiZGy1v"
      },
      "source": [
        "'''\n",
        "Calculate the similarity for each user pair and select the top-N nearest neighbors:\n",
        "    (user1,user2) ->    (similarity,co_raters_count)\n",
        "'''\n",
        "top_n_user = 50\n",
        "\n",
        "user_sims = pairwise_users.map(\n",
        "    lambda p: calcSim(p[0],p[1])).map(\n",
        "    lambda p: keyOnFirstFoo(p[0],p[1])).groupByKey().map(\n",
        "    lambda p: nearestNeighbors(p[0],list(p[1]),top_n_user))\n",
        "\n",
        "\n",
        "\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-heQYyEDGy1x",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "1667c060-5312-4230-a656-922803cad6d7"
      },
      "source": [
        "%%time\n",
        "''' \n",
        "Obtain the the item history for each user and store it as a broadcast variable\n",
        "    user_id -> [(item_id_1, rating_1),\n",
        "               [(item_id_2, rating_2),\n",
        "                ...]\n",
        "'''\n",
        "def parseRowOnUser(row):\n",
        "    '''\n",
        "    Parse each row of the specified list.\n",
        "    Converts each rating to a float\n",
        "    '''\n",
        "    return row.customer_id,(row.product_id,float(row.star_rating))\n",
        "\n",
        "user_item_hist =df_train1.rdd.map(parseRowOnUser).groupByKey().collect()\n",
        "\n",
        "\n",
        "ui_dict = {}\n",
        "for (user,items_with_ratings) in user_item_hist: \n",
        "    ui_dict[user] = items_with_ratings\n",
        "\n",
        "uib = sc.broadcast(ui_dict)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "CPU times: user 18.5 s, sys: 1.28 s, total: 19.8 s\n",
            "Wall time: 1min 42s\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "V6pJk0fUGy1y",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "4ed8b6d9-c148-491d-d3bc-2b7814717e33"
      },
      "source": [
        "%%time\n",
        "def fooAverageRating(foos_pairs):\n",
        "    \"\"\"\n",
        "        Obtain foo_rating_average matrix containing average rating for each foo:\n",
        "        dict : (foo_id) -> (avg_rating)\n",
        "        where foo can be items or users and foos is either item_user or user_item\n",
        "    \"\"\"\n",
        "    \n",
        "    def averageRating(row):\n",
        "        '''\n",
        "        Compute the average rating of an item\n",
        "        '''\n",
        "        _, ratings = np.array([foo_rating for foo_rating in row[1]] ,dtype = object).T\n",
        "        return row[0],np.mean(ratings)\n",
        "\n",
        "    return dict(foos_pairs.map(averageRating).collect())\n",
        "\n",
        "#u_i= df_train1.rdd.map(parseRowOnUser).groupByKey()\n",
        "urab = sc.broadcast(fooAverageRating(df_train1.rdd.map(parseRowOnUser).groupByKey()))\n",
        "#item_user_pairs_avg_b.value"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "CPU times: user 0 ns, sys: 13 µs, total: 13 µs\n",
            "Wall time: 16 µs\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "13VbW-6Pqy0w"
      },
      "source": [
        "### Recommandation\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": false,
        "id": "3uL0SpJmGy10",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "ffbac6d2-c38b-404f-cbca-f5e2a8a7e413"
      },
      "source": [
        "%%time\n",
        "def topNRecommendationsUI(user_id,user_sims,users_with_rating,users_rating_average,n):\n",
        "    '''\n",
        "        user_id \n",
        "        users_with_rating : dict = {user_id_1 : [(item_id_1, rating_1),\n",
        "                   (item_id_2, rating_2)],....}\n",
        "        user_sims : list = [(user_id_j, (sim(user_id_i,user_id_j), co_raters_count)), (user_id_k, (sim(user_id_i,user_id_k), co_raters_count)),....]\n",
        "        items_rating_average : dict = {'user_id_i': avg ...}\n",
        "    Calculate the top-N item recommendations for each user using the \n",
        "    weighted sums method\n",
        "    '''\n",
        "\n",
        "    # initialize dicts to store the score of each individual item,\n",
        "    # since an item can exist in more than one item neighborhood\n",
        "    totals = defaultdict(int)\n",
        "    sim_sums = defaultdict(int)\n",
        "\n",
        "    for (neighbor,(sim,count)) in user_sims:\n",
        "\n",
        "        # lookup the item predictions for this neighbor\n",
        "        unscored_items = users_with_rating.get(neighbor,None)\n",
        "        if unscored_items:\n",
        "     \n",
        "            for (item,rating) in unscored_items:\n",
        "                if dict(users_with_rating[user_id]).get(item)==None: \n",
        "\n",
        "                    # update totals and sim_sums with the rating data\n",
        "                    totals[item] += sim * (rating-users_rating_average[neighbor])\n",
        "                    sim_sums[item] +=  abs(sim)\n",
        "\n",
        "    # create the normalized list of scored items \n",
        "    scored_items = []\n",
        "    for item,total in totals.items():\n",
        "        \n",
        "        # strategy 1\n",
        "        division = 0 if sim_sums[item]==0 else total/sim_sums[item]\n",
        "        scored_items.append((division+users_rating_average[user_id],item)) \n",
        "        '''\n",
        "        # strategy 2\n",
        "        if (sim_sums[item]!=0 ):\n",
        "            scored_items.append((total/sim_sums[item]+items_rating_average[item],item)) \n",
        "        '''\n",
        "    # sort the scored items in ascending order\n",
        "    scored_items.sort(reverse=True)\n",
        "\n",
        "    # take out the item score\n",
        "    # ranked_items = [x[1] for x in scored_items]\n",
        "\n",
        "    return user_id,scored_items[:n]\n",
        "\n",
        "\n",
        "'''\n",
        "Calculate the top-N item recommendations for each user\n",
        "    user_id -> [item1,item2,item3,...]\n",
        "'''\n",
        "user_item_recs = user_sims.map(\n",
        "    lambda p: topNRecommendationsUI(p[0],p[1],uib.value,urab.value,100)).collect()\n",
        "\n",
        "#user_item_recs.toDF([\"user_id\",\"recommendations\"]).show(5)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "CPU times: user 569 ms, sys: 100 ms, total: 670 ms\n",
            "Wall time: 47min 41s\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": true,
        "id": "TWxtyN75Gy12",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 119
        },
        "outputId": "3c5ad7af-bf89-4f35-a685-3da5dcae4c27"
      },
      "source": [
        "result = (evaluate(df_test_list,df_train_list,user_item_recs))\n",
        "print (\"MAE: \\n Train : {} \\n Test  : {}\\nRMSE: \\n Train : {} \\n Test  : {}\".format(result[0][0],result[1][0],result[0][1],result[1][1] ))\n",
        "\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "MAE: \n",
            " Train : 0.521125623907677 \n",
            " Test  : 0.535225623903678\n",
            "RMSE: \n",
            " Train : 1.0013471433958179 \n",
            " Test  : 1.2973471433958188\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "WEKQZQOsGy13"
      },
      "source": [
        "## Kmean"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JLhmo0a0Gy14"
      },
      "source": [
        "products = df_train1.select(\"product_id\").distinct().collect()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": true,
        "id": "g4-bueIBGy15"
      },
      "source": [
        "products = dict([(product.product_id,0.0) for product in products])"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": true,
        "id": "5tukBSQlGy16"
      },
      "source": [
        "def parseProducts (user_id,products_with_ratings,products):\n",
        "    this_products = products.copy()\n",
        "    for (product,rating) in products_with_ratings :\n",
        "        this_products[product]=rating\n",
        "    return user_id , list(this_products.values())\n",
        "\n",
        "user_item_features = df_train1.rdd.map(parseRowOnUser).groupByKey().map(lambda p: parseProducts(p[0],p[1],products))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "V43OdyOrb4FO"
      },
      "source": [
        "def func (p):\n",
        "  users , rating_pair = p\n",
        "  return (users,(*rating_pair))\n",
        "a=user_item_features.map(func).toDF()\n",
        "#a.show(2)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_bD0nRvlGy17"
      },
      "source": [
        "uif_df = user_item_features.toDF().withColumnRenamed('_1','user_id').withColumnRenamed('_2','features')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": true,
        "id": "CnuNnz8MGy18",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        },
        "outputId": "57c2dc45-c4f1-4e30-a56f-013d82c333e5"
      },
      "source": [
        "uif_df.show(5)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "+--------+--------------------+\n",
            "| user_id|            features|\n",
            "+--------+--------------------+\n",
            "|10171604|[0.0, 0.0, 0.0, 0...|\n",
            "|10634752|[0.0, 0.0, 0.0, 0...|\n",
            "|10840727|[0.0, 0.0, 0.0, 0...|\n",
            "|11488922|[0.0, 0.0, 0.0, 0...|\n",
            "|11654799|[0.0, 0.0, 0.0, 0...|\n",
            "+--------+--------------------+\n",
            "only showing top 5 rows\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": true,
        "id": "ID-07nOYGy1-",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 102
        },
        "outputId": "18a76fc4-313c-4d5a-8656-81001eb13120"
      },
      "source": [
        "uif_df.printSchema()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "root\n",
            " |-- user_id: string (nullable = true)\n",
            " |-- features: array (nullable = true)\n",
            " |    |-- element: double (containsNull = true)\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "I4zdk_ZLGy1_"
      },
      "source": [
        "features column containsNull attribute have to be false "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Sjz1hCOFGy1_"
      },
      "source": [
        "from pyspark.sql.types import ArrayType,DoubleType\n",
        "from pyspark.sql.functions import udf\n",
        "from pyspark.ml.clustering import KMeans"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "C1cnxmaaGy2A"
      },
      "source": [
        "new_schema = ArrayType(DoubleType(), containsNull=False)\n",
        "udf_foo = udf(lambda x:x, new_schema)\n",
        "uif_df = uif_df.withColumn(\"features\",udf_foo(\"features\"))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "bHerqLwtGy2B"
      },
      "source": [
        "kmeans = KMeans(k=5, seed=1,maxIter=10) \n",
        "model = kmeans.fit(uif_df.select('features'))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uf2QmkZoGy2U"
      },
      "source": [
        "transformed = model.transform(new_df)\n",
        "transformed.show()    "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YJuxoDRVf8n9"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ALPb0-Oss2JH"
      },
      "source": [
        "# Model Based Approches"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6TMy4JxQs_G9"
      },
      "source": [
        "## Matrix factorisation"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "p-UxpnOqVCdA"
      },
      "source": [
        "def load_data(path, sample_size=300000):\n",
        "  print(\"Loading data...\")\n",
        "  cols = [\"customer_id\", \"product_id\", \"star_rating\"]\n",
        "  pdf = pd.read_csv(path, sep='\\t', usecols=cols)\n",
        "  print(\"Data loaded!\", end=\"\\n\\n\")\n",
        "  return pdf.sample(sample_size)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "58gQtha7JX85"
      },
      "source": [
        "def get_sample(df, size=500, seed=42):\n",
        "  np.random.seed(seed)\n",
        "  random_indexes = np.random.randint(df.shape[0], size=size)\n",
        "  return df.iloc[random_indexes, :].copy()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3_hCX_mCMCHB"
      },
      "source": [
        "def convert_data_type(df):\n",
        "  print(\"Converting data type...\")\n",
        "  encoder = LabelEncoder()\n",
        "  encoder.fit(df.product_id.unique())\n",
        "  df['product_id'] = encoder.transform(df.product_id.values)\n",
        "  df.customer_id.astype(int)\n",
        "  df.star_rating.astype(int)\n",
        "  print(\"Data converted!\", end=\"\\n\\n\")\n",
        "  return df, encoder"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wZtlX9UiuLPa"
      },
      "source": [
        "def panda2spark(pdf):\n",
        "  print(\"Creating spark dataframe...\")\n",
        "  return spark.createDataFrame(pdf)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "URnbRNlI1NcI"
      },
      "source": [
        "def grid_search_als(estimator: ALS, evaluator: RegressionEvaluator, ranks=[12, 14, 16], maxIters=[19, 21], regParams=[.17, .19], parallelism=4):\n",
        "  \n",
        "  print(\"Grid search configuration...\")\n",
        "  param_grid = ParamGridBuilder()\\\n",
        "                .addGrid(estimator.rank, ranks)\\\n",
        "                .addGrid(estimator.maxIter, maxIters)\\\n",
        "                .addGrid(estimator.regParam, regParams)\\\n",
        "                .build()\n",
        "\n",
        "  trainValidationSplit = TrainValidationSplit(\n",
        "      estimator=estimator, \n",
        "      estimatorParamMaps=param_grid, \n",
        "      evaluator=evaluator, \n",
        "      parallelism = parallelism\n",
        "      )\n",
        "  \n",
        "  return trainValidationSplit, param_grid"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ZSa7R1CG1trm"
      },
      "source": [
        "def get_evaluator(metricName='rmse'):\n",
        "  print(\"Building evaluator...\")\n",
        "  evaluator = RegressionEvaluator(\n",
        "    predictionCol='prediction', \n",
        "    labelCol='star_rating', \n",
        "    metricName=metricName\n",
        "    )\n",
        "  print(\"Evaluator build finished!\", end=\"\\n\\n\")\n",
        "  return evaluator"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "FdyBetOydP5W"
      },
      "source": [
        "def build_model(train, rank=20, maxIter=20, regParam=0.2, search_best_params=False, ranks=[21, 24, 26], maxIters=[20, 22], regParams=[.25, .3]):\n",
        "    als = ALS(\n",
        "    userCol='customer_id',\n",
        "    itemCol='product_id', \n",
        "    ratingCol='star_rating', \n",
        "    coldStartStrategy='drop', \n",
        "    nonnegative=True,\n",
        "    rank=rank,\n",
        "    maxIter = maxIter,\n",
        "    regParam = regParam\n",
        "    )\n",
        "\n",
        "    if search_best_params == True:\n",
        "      evaluator = get_evaluator()\n",
        "      (trainValidationSplit, param_grid) = grid_search_als(als, evaluator)\n",
        "\n",
        "      print(\"Training model with best params...\")\n",
        "      models = trainValidationSplit.fit(train)\n",
        "      print(\"Training finished successfully!\", end=\"\\n\\n\")\n",
        "\n",
        "      model = models.bestModel \n",
        "    else:\n",
        "      print(\"Model parammeters:\")\n",
        "      print(f\"rank: {rank}\")\n",
        "      print(f\"maxIter: {maxIter}\")\n",
        "      print(f\"regParam: {regParam}\\n\")\n",
        "\n",
        "      print(\"Training the model...\")\n",
        "      model = als.fit(train)\n",
        "      print(\"Training finished!\",  end=\"\\n\\n\")\n",
        "\n",
        "    return model"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uiJWZz8VmdXz"
      },
      "source": [
        "def evaluation(model, train, test, metricName='rmse'):\n",
        "  print(\"Evaluating the model...\")\n",
        "\n",
        "  evaluator = get_evaluator(metricName)\n",
        "  \n",
        "  print(\"Computing loss..\")\n",
        "  train_predictions = model.transform(train)\n",
        "  train_rmse = evaluator.evaluate(train_predictions)\n",
        "  print(f\"Training error: {train_rmse}\")\n",
        "\n",
        "  test_predictions = model.transform(test)\n",
        "  test_rmse = evaluator.evaluate(test_predictions)\n",
        "  print(f\"Test error: {test_rmse}\")\n",
        "\n",
        "  model_evaluation = {\n",
        "      \"evaluator\": evaluator,\n",
        "      \"rmse\": {\n",
        "          \"train\": train_rmse,\n",
        "          \"test\": test_rmse\n",
        "      },\n",
        "      \"predictions\": {\n",
        "          \"train\": train_predictions,\n",
        "          \"test\": test_predictions\n",
        "      }\n",
        "  }\n",
        "\n",
        "  print(\"Evaluation finished!\", end=\"\\n\\n\")\n",
        "  return model_evaluation"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wDXaYH_sT7mV"
      },
      "source": [
        "def pipeline(path, search_best_params=False, ranks=[21, 24, 26], maxIters=[20, 22], regParams=[.25, .3]):\n",
        "  print(\"Starting...\")\n",
        "  pdf = load_data(path, sample_size=1000000)\n",
        "  (pdf, encoder) = convert_data_type(pdf)\n",
        "  df = panda2spark(pdf)\n",
        "\n",
        "  (train, test) = df.randomSplit([0.8, 0.2])\n",
        "\n",
        "  model = build_model(train, search_best_params=search_best_params, regParam=0.5)\n",
        "  model_evaluation = evaluation(model, train, test)\n",
        "\n",
        "  model.save('matrix_factorization')\n",
        "  print(\"Finished!\", end=\"\\n\\n\")\n",
        "\n",
        "  return model, model_evaluation"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "4ADDfU2VYabp",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 476
        },
        "outputId": "260d380f-1e2a-42b1-c9eb-49a5ae970ddf"
      },
      "source": [
        "path = 'amazon_reviews_us_Electronics_v1_00.tsv.gz'\n",
        "\n",
        "(model, model_evaluation) = pipeline(path, search_best_params=False)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Starting...\n",
            "Loading data...\n",
            "Data loaded!\n",
            "\n",
            "Converting data type...\n",
            "Data converted!\n",
            "\n",
            "Creating spark dataframe...\n",
            "Model parammeters:\n",
            "rank: 20\n",
            "maxIter: 20\n",
            "regParam: 0.5\n",
            "\n",
            "Training the model...\n",
            "Training finished!\n",
            "\n",
            "Evaluating the model...\n",
            "Building evaluator...\n",
            "Evaluator build finished!\n",
            "\n",
            "Computing loss..\n",
            "Training error: 0.5374322041228355\n",
            "Test error: 1.8636257049613567\n",
            "Evaluation finished!\n",
            "\n",
            "Finished!\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "od2_7FUN2mjH"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DSnbNZV6suyC"
      },
      "source": [
        "## Deep Learning\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DTLQDQ07qADS"
      },
      "source": [
        "### 1.2 Fonctions utilitaires"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vapyPsMxpwXj"
      },
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import os\n",
        "import warnings\n",
        "\n",
        "from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate\n",
        "from keras.models import Model, Sequential\n",
        "from tensorflow.keras.utils import plot_model\n",
        "from tensorflow.keras.utils import normalize\n",
        "\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.utils import shuffle\n",
        "\n",
        "warnings.filterwarnings('ignore')\n",
        "%matplotlib inline"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "fmZ5d0qptvZJ"
      },
      "source": [
        "def load_data(path, sample_size='all'):\n",
        "  \n",
        "  print(\"Loading data...\")\n",
        "  cols = [\"customer_id\", \"product_id\", \"star_rating\"]\n",
        " \n",
        "  pdf = pd.read_csv(path, sep='\\t', usecols=cols)\n",
        "  print(\"Data loaded!\", end=\"\\n\\n\")\n",
        "\n",
        "  if sample_size == 'all':\n",
        "    return pdf\n",
        "  else:\n",
        "    return pdf.sample(sample_size)\n",
        "\n",
        "def convert_data_type(df):\n",
        "  print(\"Converting data type...\")\n",
        "  encoder = LabelEncoder()\n",
        "  encoder.fit(df.product_id.unique())\n",
        "  df['product_id'] = encoder.transform(df.product_id.values)\n",
        "  # df.customer_id.astype(int)\n",
        "  # df.star_rating.astype(int)\n",
        "  print(\"Data converted!\")\n",
        "  return df\n",
        "\n",
        "def denormalize_rating(rating, min, max):\n",
        "  return rating*(max - min) + min"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "O7K45cj3qHHl"
      },
      "source": [
        "### 1.3 Chargement et prétraitement du dataset"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "prhEIuotpwXw",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        },
        "outputId": "d481e348-b362-4d4d-9159-35f7c3b13db3"
      },
      "source": [
        "path = 'amazon_reviews_us_Electronics_v1_00.tsv.gz'\n",
        "\n",
        "dataset = load_data(path, sample_size='all')"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Loading data...\n",
            "Data loaded!\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Zzq0O_mNlgfB",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "b400a734-c9d4-4fc7-f4d7-9e850319fd33"
      },
      "source": [
        "dataset.shape"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(3091103, 3)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 63
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wbXqVsrevE6P",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "e6b56f4b-ed86-4dc9-8c37-c64f3e68dc76"
      },
      "source": [
        "dataset = convert_data_type(dataset)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Converting data type...\n",
            "Data converted!\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Th_ERoDIniPq",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "2a47acca-3944-4c01-8070-4059c53ffc78"
      },
      "source": [
        "num_duplicates = dataset.shape[0] - dataset.drop_duplicates().shape[0]\n",
        "num_duplicates"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1417"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VZegUbRxpwX8",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "a2a26d99-7f9b-411e-d030-9e7c744b8fb5"
      },
      "source": [
        "dataset.shape"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(3091103, 3)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_gwsTIHNtjyx",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "e8c9c632-3a56-4615-8fe2-0a3821a31dc6"
      },
      "source": [
        "# dataset.drop_duplicates(inplace=True)\n",
        "# dataset.shape"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(3089686, 3)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 67
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_fi3eev0pwYE"
      },
      "source": [
        "train, test = train_test_split(dataset, test_size=0.2, random_state=42)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dytXEkPlpwYJ",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "c70224b4-1103-482e-b8fa-12a7ac6c501c"
      },
      "source": [
        "train.shape"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(2472882, 3)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 66
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Ln2h_D4YpwYP",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "ffa76857-73c9-4559-e312-c6375bcb9073"
      },
      "source": [
        "test.shape"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(618221, 3)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 67
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "U0i9ZVr4M0pu",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 255
        },
        "outputId": "589f7d8a-36a7-4a4b-81e9-6c811d10c738"
      },
      "source": [
        "display(train.head())\n",
        "print(train.customer_id.unique().size)\n",
        "print(train.product_id.unique().size)\n",
        "print(train.star_rating.unique().size)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>customer_id</th>\n",
              "      <th>product_id</th>\n",
              "      <th>star_rating</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>645939</th>\n",
              "      <td>17671807</td>\n",
              "      <td>132841</td>\n",
              "      <td>5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>404773</th>\n",
              "      <td>48799690</td>\n",
              "      <td>163737</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1066713</th>\n",
              "      <td>18878830</td>\n",
              "      <td>45657</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>236742</th>\n",
              "      <td>11481572</td>\n",
              "      <td>175744</td>\n",
              "      <td>5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>215866</th>\n",
              "      <td>868194</td>\n",
              "      <td>105895</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "         customer_id  product_id  star_rating\n",
              "645939      17671807      132841            5\n",
              "404773      48799690      163737            3\n",
              "1066713     18878830       45657            1\n",
              "236742      11481572      175744            5\n",
              "215866        868194      105895            2"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "stream",
          "text": [
            "1803114\n",
            "168601\n",
            "5\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xIsZ2HHTB8Jt",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "2626a04e-006c-45bb-ab38-d5478ee26397"
      },
      "source": [
        "dataset.product_id.max()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "185774"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 69
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "NuDcodcspwYW",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "41c61c40-73f6-4210-a4b5-71d62a8c1382"
      },
      "source": [
        "n_customers = len(dataset.customer_id.unique())\n",
        "n_customers"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "2152825"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 70
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VgZyGQlhpwYe",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "269d9597-2c93-48d9-d2b8-84f11da0c984"
      },
      "source": [
        "n_products = dataset.product_id.nunique()\n",
        "n_products"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "185775"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 71
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "o0JiC_tSqV_w",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "f39f86bc-14c8-4eef-959e-83a1e97ca579"
      },
      "source": [
        "train.customer_id.nunique()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1803114"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 72
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "IKtPjTnaqcKG",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "aaaab65b-7eb8-48c5-d1b1-160627686f01"
      },
      "source": [
        "train.product_id.nunique()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "168601"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 73
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YalZrAXNNvB7"
      },
      "source": [
        "n_latent_factors = 15"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": true,
        "id": "H8KrI9TIpwYu"
      },
      "source": [
        "#from keras.models import load_model\n",
        "\n",
        "#if os.path.exists('regression_model.h5'):\n",
        "    #model = load_model('regression_model.h5')\n",
        "#else:\n",
        "\n",
        "def fit_plot_save(model, path='regression_model.h5' , output=train.star_rating, batch_size=4096, epochs=10):\n",
        "  history = model.fit([train.customer_id, train.product_id], train.star_rating, batch_size=batch_size, epochs=epochs, verbose=1)\n",
        "  model.save(path)\n",
        "  plt.plot(history.history['loss'])\n",
        "  plt.xlabel(\"Epochs\")\n",
        "  plt.ylabel(\"Training Error\")\n",
        "\n",
        "# 1.4947\n",
        "# 1.4978\n",
        "# 1.5447\n",
        "# 1.4787\n",
        "# 1.4707\n",
        "# 1.4867"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "8Qusg3v3sYzW"
      },
      "source": [
        "### 2 - Construction du modèle  "
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_xeYWvImr_q9"
      },
      "source": [
        " #### 2.1 - Reproduction d'une Matrix Factorization à l'aide d'un réseau de à une couche"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "HyWgwp3BpwYo"
      },
      "source": [
        "################---Input Layer---################\n",
        "# product input\n",
        "product_input = Input(shape=[1], name=\"Product-Input\")\n",
        "\n",
        "# customer input \n",
        "customer_input = Input(shape=[1], name=\"Customer-Input\")\n",
        "\n",
        "\n",
        "################---Matrix-Factorization embeddings Layer---################\n",
        "# customer Matrix-Factorization vector\n",
        "customer_embedding = Embedding(n_customers, n_latent_factors, name=\"Customer-Embedding\")(customer_input)\n",
        "customer_vec = Flatten(name=\"Flatten-Customers\")(customer_embedding)\n",
        "\n",
        "# product Matrix-Factorization vector\n",
        "product_embedding = Embedding(n_products + 1, n_latent_factors, name=\"Product-Embedding\")(product_input)\n",
        "product_vec = Flatten(name=\"Flatten-Products\")(product_embedding)\n",
        "\n",
        "\n",
        "################---Matrix-Factorization output Layer---################\n",
        "mf_output_layer = Dot(name=\"Dot-Product\", axes=1)([product_vec, customer_vec])\n",
        "\n",
        "# Create model and compile\n",
        "mf_model = Model([customer_input, product_input], mf_output_layer)\n",
        "mf_model.compile('adam',  loss='mean_squared_error')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tMuAbEMopqKd",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 408
        },
        "outputId": "89d92a75-92cd-4302-a93f-e1ee2570ed14"
      },
      "source": [
        "mf_model.summary()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Model: \"functional_13\"\n",
            "__________________________________________________________________________________________________\n",
            "Layer (type)                    Output Shape         Param #     Connected to                     \n",
            "==================================================================================================\n",
            "Product-Input (InputLayer)      [(None, 1)]          0                                            \n",
            "__________________________________________________________________________________________________\n",
            "Customer-Input (InputLayer)     [(None, 1)]          0                                            \n",
            "__________________________________________________________________________________________________\n",
            "Product-Embedding (Embedding)   (None, 1, 15)        2786640     Product-Input[0][0]              \n",
            "__________________________________________________________________________________________________\n",
            "Customer-Embedding (Embedding)  (None, 1, 15)        32292375    Customer-Input[0][0]             \n",
            "__________________________________________________________________________________________________\n",
            "Flatten-Products (Flatten)      (None, 15)           0           Product-Embedding[0][0]          \n",
            "__________________________________________________________________________________________________\n",
            "Flatten-Customers (Flatten)     (None, 15)           0           Customer-Embedding[0][0]         \n",
            "__________________________________________________________________________________________________\n",
            "Dot-Product (Dot)               (None, 1)            0           Flatten-Products[0][0]           \n",
            "                                                                 Flatten-Customers[0][0]          \n",
            "==================================================================================================\n",
            "Total params: 35,079,015\n",
            "Trainable params: 35,079,015\n",
            "Non-trainable params: 0\n",
            "__________________________________________________________________________________________________\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Kodb7HVH2c3i",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 369
        },
        "outputId": "70e8c804-f808-4574-b2ea-16090581b7e5"
      },
      "source": [
        "plot_model(mf_model, to_file='mf_model.png')"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<IPython.core.display.Image object>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 80
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1fh62ciD0YU2"
      },
      "source": [
        "fit_plot_save(mf_model, path='mf_model.h5') "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dvuygVq_nayb"
      },
      "source": [
        "### Meilleur Loss: 18.7\n",
        "### Temps de calcul: 27s\n",
        "### Epochs 10 "
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kqCrJqhlsRMI"
      },
      "source": [
        "#### 2.2 - Ajout d'une couche densément connectée à la sortie du réseau de neurone (DNN)"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "a9bpK7SxwvKr"
      },
      "source": [
        "################---Input Layer---################\n",
        "# product input\n",
        "product_input = Input(shape=[1], name=\"Product-Input\")\n",
        "\n",
        "# customer input \n",
        "customer_input = Input(shape=[1], name=\"Customer-Input\")\n",
        "\n",
        "\n",
        "################---Matrix-Factorization embeddings Layer---################\n",
        "# product Matrix-Factorization vector\n",
        "product_embedding = Embedding(n_products+1, n_latent_factors, name=\"Product-Embedding\")(product_input)\n",
        "product_vec = Flatten(name=\"Flatten-Products\")(product_embedding)\n",
        "\n",
        "# customer Matrix-Factorization vector\n",
        "customer_embedding = Embedding(n_customers+1, n_latent_factors, name=\"Customer-Embedding\")(customer_input)\n",
        "customer_vec = Flatten(name=\"Flatten-Customers\")(customer_embedding)\n",
        "\n",
        "################---Matrix-Factorization output Layer---################\n",
        "mf_output_layer = Dot(name=\"Dot-Product\", axes=1)([product_vec, customer_vec])\n",
        "\n",
        "################---Fully-connected Layers---################\n",
        "fully_connected_layer = Dense(128, activation='relu')(mf_output_layer)\n",
        "fully_connected_layer0 = Dense(128, activation='relu')(fully_connected_layer)\n",
        "fully_connected_layer1 = Dense(128, activation='relu')(fully_connected_layer0)\n",
        "fully_connected_layer2 = Dense(64, activation='relu')(fully_connected_layer1)\n",
        "\n",
        "\n",
        "# ################---Output Layer---################\n",
        "output_layer = Dense(1)(fully_connected_layer2)\n",
        "\n",
        "\n",
        "# Create model and compile\n",
        "neural_mf_model = Model([customer_input, product_input], output_layer)\n",
        "neural_mf_model.compile('adam',  loss='mean_squared_error')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YpJrFXoxnn7l",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 578
        },
        "outputId": "4f8fa0b9-52c8-453b-e99b-23aacc314cb8"
      },
      "source": [
        "neural_mf_model.summary()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Model: \"functional_15\"\n",
            "__________________________________________________________________________________________________\n",
            "Layer (type)                    Output Shape         Param #     Connected to                     \n",
            "==================================================================================================\n",
            "Product-Input (InputLayer)      [(None, 1)]          0                                            \n",
            "__________________________________________________________________________________________________\n",
            "Customer-Input (InputLayer)     [(None, 1)]          0                                            \n",
            "__________________________________________________________________________________________________\n",
            "Product-Embedding (Embedding)   (None, 1, 15)        2786640     Product-Input[0][0]              \n",
            "__________________________________________________________________________________________________\n",
            "Customer-Embedding (Embedding)  (None, 1, 15)        32292390    Customer-Input[0][0]             \n",
            "__________________________________________________________________________________________________\n",
            "Flatten-Products (Flatten)      (None, 15)           0           Product-Embedding[0][0]          \n",
            "__________________________________________________________________________________________________\n",
            "Flatten-Customers (Flatten)     (None, 15)           0           Customer-Embedding[0][0]         \n",
            "__________________________________________________________________________________________________\n",
            "Dot-Product (Dot)               (None, 1)            0           Flatten-Products[0][0]           \n",
            "                                                                 Flatten-Customers[0][0]          \n",
            "__________________________________________________________________________________________________\n",
            "dense (Dense)                   (None, 128)          256         Dot-Product[0][0]                \n",
            "__________________________________________________________________________________________________\n",
            "dense_1 (Dense)                 (None, 128)          16512       dense[0][0]                      \n",
            "__________________________________________________________________________________________________\n",
            "dense_2 (Dense)                 (None, 128)          16512       dense_1[0][0]                    \n",
            "__________________________________________________________________________________________________\n",
            "dense_3 (Dense)                 (None, 64)           8256        dense_2[0][0]                    \n",
            "__________________________________________________________________________________________________\n",
            "dense_4 (Dense)                 (None, 1)            65          dense_3[0][0]                    \n",
            "==================================================================================================\n",
            "Total params: 35,120,631\n",
            "Trainable params: 35,120,631\n",
            "Non-trainable params: 0\n",
            "__________________________________________________________________________________________________\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "RrhZioE_2AYn",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 856
        },
        "outputId": "09da58e5-e43d-420e-c723-1531c98ed52a"
      },
      "source": [
        "plot_model(neural_mf_model, to_file='neural_mf_model.png')"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "image/png": "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/LLlibDyjktb+t/ixYstx19wcLB8++238vrrr4u3t7cAEH9/f1mzZo2sW7dO/P39BYD4+vrK2rVrRaR291tH6BHsd47R7+zh6UWHCl1UeyZOnCgNGjTQuoxa4ejLMmTIEPnpp580mTePv+rhenMsjt4jbufoy6Jlv7OH0OV0lxfpd7ce560PHGlZbj99n5qaCpPJhBYtWmhYEVH950g94k4caVnY76zVm99eJHIUM2bMwDPPPAMRwbhx4xAfH691SUREdYL9zhrPdDmhmTNn4l//+heys7PRokULJCUlaV1StTnisri7u+Puu+/GQw89hLi4OLRv317rkojqLUfsERVxxGVhv7OmiPz/r4fVqgBFQUJCAkaNGqVlGUROicdf9XC9ETmexMREREVFQcvYwzNdRERERCpg6CIiIiJSAUMXERERkQoYuoiIiIhUwNBFREREpAKGLiIiIiIVMHQRERERqYChi4iIiEgFDF1EREREKmDoIiIiIlIBQxcRERGRChi6iIiIiFTA0EVERESkAr3WBQDAokWLsH79eq3LICKyGfsWkWPJyMjQugTtQ1dERITWJVANfPLJJ+jevTuaNGmidSlUDREREWjWrJnWZTgc9q36KyUlBQDQvXt3jSuh2ta0aVPNj11FRETTCsihKYqChIQEjBo1SutSiIhq7FYvS0xM1LgSqo94TxcRERGRChi6iIiIiFTA0EVERESkAoYuIiIiIhUwdBERERGpgKGLiIiISAUMXUREREQqYOgiIiIiUgFDFxEREZEKGLqIiIiIVMDQRURERKQChi4iIiIiFTB0EREREamAoYuIiIhIBQxdRERERCpg6CIiIiJSAUMXERERkQoYuoiIiIhUwNBFREREpAKGLiIiIiIVMHQRERERqYChi4iIiEgFDF1EREREKmDoIiIiIlIBQxcRERGRChi6iIiIiFTA0EVERESkAoYuIiIiIhUwdBERERGpgKGLiIiISAUMXUREREQqYOgiIiIiUgFDFxEREZEKFBERrYsgx/D444/j4MGDVsPOnDmDhg0bwmw2W4YZDAZs3rwZgYGBapdIRGSzDz/8EIsXL0ZJSYll2KVLlwAADRs2tAzT6XSYMmUKxo4dq3aJVM/otS6AHEfbtm2xevXqMsPz8vKs/n333XczcBGR3evduzfGjRtX7nu//vqr1b979eqlRklUz/HyItlszJgxUBSl0nEMBgP/GiQih9C2bVuEhoZW2tcURUFoaCjuvvtuFSuj+oqhi2zWsmVLdOnSBS4uFe82xcXFiIqKUrEqIqLqe+KJJ6DT6Sp8X6/X48knn1SxIqrPGLqoSp544okKQ5eiKOjZsyeCg4PVLYqIqJoee+wxq3u6/oh/SFJtYuiiKomKikJpaWm577m4uOCJJ55QuSIioupr0qQJ+vTpU+4fky4uLujTpw+aNm2qQWVUHzF0UZUEBAQgLCyswtPx4eHhKldERFQzjz/+eLn3dSmKwj8kqVYxdFGVPf7442WGubi4oH///vD399egIiKi6ouMjKzwZnr+IUm1iaGLqiwyMrLcU/HlhTEiInvXoEEDDBgwAHr979+ipNPpMGDAAPj5+WlYGdU3DF1UZV5eXhg0aFCZBvXoo49qWBURUfXFxMRY3a8qIvxDkmodQxdVS0xMjOWJH71ej0ceeQTe3t4aV0VEVD2PPvooXF1dLf82GAx45JFHNKyI6iOGLqqWRx55BG5ubgCAkpISREdHa1wREVH1mc1mPPLIIzAYDNDr9Rg+fDg8PDy0LovqGYYuqhaTyYSRI0cCANzd3TF48GCNKyIiqpno6GgUFxejpKQEjz32mNblUD1U5rcXMzIy8P3332tRCzmYZs2aAQB69OiBTz75RONqyBE0a9YMvXv3rpNp7969G+fPn6+TaZNzKCkpgclkgoggLy8PiYmJWpdEDqzcfid/kJCQIAD44osvvmr9FRER8ceWU2siIiI0Xz6++OKLr1uv8vpdmTNdt4hIRW8RWcTFxeGVV16xepKRqDyRkZF1Po+IiAisX7++zudD9ddXX30FRVHQr18/rUshB1ZRv+P/lFQjDFxEVJ/cf//9WpdA9Rj/t6QaYeAiovqkvC9+Jqot3LuIiIiIVMDQRURERKQChi4iIiIiFTB0EREREamAoYuIiIhIBQxdRERERCpg6CIiIiJSAUMXERERkQoYuoiIiIhUwNBFREREpAKGLiIiIiIVMHQRERERqaDGoWvDhg0ICQmBoiiWl8FgQGBgIKKjo3HkyJHaqLNCEyZMgKenJxRFwcGDB+t0XpUpbz388RUcHFwr8+rRowd0Oh06d+5cK9O7na3rs6Lxtm7dCm9vb2zevLnWa6sKbo+b7GV72Ltjx45h0qRJ6NChAzw9PaHX6+Ht7Y02bdpg6NCh2L17t9Yl2h0eYzfZyzHG7XGTvWyPitQ4dIWHh+Onn35Cy5Yt4e3tDRFBVlYW3n33XXz33Xfo2bMnjh07Vhu1luuDDz7A+++/X2fTt1V560FEUFxcjPz8fPz6669wd3evlXnt27cP/fv3r5Vp/ZGt67Oi8USkLsqqMm6Pm+xle9izFStWIDQ0FKmpqVi4cCHOnz+P69ev48CBA5g9ezaysrKQlpamdZl2h8fYTfZyjHF73GQv26Mi+rqYqNlsxsMPP4ySkhKMGDECS5YswTvvvFMXs6pzBQUFePDBB/H9999X6/M6nQ5ubm5wc3NDmzZtarU2RVFqdXq1YejQocjOzta6jApxe9Dt9uzZg4kTJ+L+++/H559/Dr3+95YYEhKCkJAQ+Pj44MSJE7U635r2FXvGY8y+cHvYlzq9p6tnz54AgPT09LqcTZ1u+BUrVuDixYu1Mq1NmzbVynRuMRgMtTq9W2xdn2occCKC9evXY/ny5bU+bW6PqqvL7aGFOXPmoKSkBPPmzbMKXLcbOHAgnn/++Vqdb232FXvGY6zq2PPq9/ao09BVXFwMADAajQCAN954A+7u7vD09MTFixcxdepUBAYG4tixYxARLFy4EO3atYPRaISvry+GDx+Oo0ePWk1TRLBgwQK0bdsWRqMR3t7emD59utU4kydPhqurKwICAizDnnvuOZjNZiiKgsuXL1uNHx8fj+7du8NkMsFsNiM4OBizZ8/GlClTMHXqVJw6dQqKoqBVq1a1tm4WL14Ms9kMFxcXdOvWDf7+/jAYDDCbzejatSvCwsLQrFkzmEwm+Pj44MUXXywzjZMnT+Luu++G2WyGm5sbwsLC8N1331mNU1JSglmzZiEoKAhubm7o1KkTEhISLO/bsj5tHe+7775DUFAQFEWxnNlctmwZzGYz3N3d8fHHH2Pw4MHw8vJC06ZNsXbt2jK1vvbaa2jbti3c3Nxw1113oUWLFnjttdcwatQoy3jbtm2Dl5cX5s6dW/UVXwFuj+pvD0dVWFiI7du3w8/Pz/IH4p1Upbfs3LkTPXv2hLu7O7y8vBAaGoqcnJwK+4otPbCm+2ll+19l/bk28Bhjz6sv26NG5A8SEhKknMF31LJlS/H29rYaFh8fLwBk+vTplmEvv/yyAJDY2FhZsmSJjBw5Uo4cOSKzZs0SV1dXiY+Pl6ysLElNTZWuXbvKXXfdJRcuXLD6vKIo8tZbb8m1a9ckPz9fli5dKgDkwIEDlvGio6PF39/fqp4FCxYIALl06ZJl2KJFiwSAzJs3T65cuSJXr16V9957T6Kjo0VEJDw8XFq2bFmj9RAbGytpaWllxv3b3/4mACQ5OVmuX78uly9flkGDBgkA+fTTT+XSpUty/fp1mTx5sgCQgwcPWj774IMPSkhIiJw+fVqKiookPT1d7r33XjGZTHL8+HHLeNOmTROj0ShJSUly7do1mTlzpri4uMi+ffuqtD5tHe/8+fMCQJYsWWL1WQCyfft2yc7OlosXL0pYWJiYzWYpLCy0jDd37lzR6XTy8ccfS35+vvzwww/i7+8v/fr1s1pvW7ZsEU9PT3n11Ve5Pexge9gqIiJCIiIiqvXZupj+8ePHBYD06tWrSvOxpbfk5eWJl5eXzJ8/XwoKCuTChQsycuRIy/vl9RVbe2BN9lNb9r/y+nNleIyx5znb9rBFRf2oTkJXXl6eJCUlib+/vzRq1EgyMjIs491a+IKCAsuw/Px88fDwkNGjR1tNc+/evQLAsqPl5+eLu7u7DBgwwGq8tWvXVit0FRYWio+Pj/Tv399qvOLiYlm8eLGIVC90ASjzqmyHz83NtQz76KOPyox/az2sW7fOMuzBBx+Ue+65x2p6qampAkCmTZsmIiIFBQXi7u5utV7z8/PFaDTKs88+a/P6rMp6r2yHv32b3zpYTp48aRnWo0cP6dmzp9U8nnrqKXFxcZEbN26UWX+24Pawn+1hb6ErJSVFAMhDDz1UpfnY0lvS09MFgGzZsqXcafyxr9jaA0Wqv5/eaf8TKX/fuBMeY/ZzjIlwe9jL9qioH9Xq5cXs7GwoigJvb2/ExsZiyJAh2Lt3LwIDAyv93KFDh5CXl4fu3btbDe/RowdcXV2RnJwM4OapzPz8fDz44IO1Um9qaiqysrIwcOBAq+E6nQ6xsbHVnu7tT46ISJWm5erqCuD3S7PA79fNi4qKKv1saGgovL29kZqaCuDmY/D5+fno2LGjZRw3NzcEBATg6NGjNq/P2l7vwO/Lefsy/fbbb2WePCkpKYHBYIBOp6v2vLg97kzN7WEvPDw8AAD5+fm1Pu2QkBA0atQIMTExiIuLw5kzZyod39YeWBFb9tM77X81wWPsztjzbnLG7XG7Wg1dtzZ0cXExMjIysHLlSjRv3vyOn8vKygLwexO8nY+PD3JzcwEAGRkZAICGDRvWSr05OTmWedhqy5YtZb77JCYmptLPLF682Gqnq0sGg8GyE12/fh0A8Morr1jVe/bsWeTn59u8Pmt7vVdkyJAh+OGHH/Dxxx+joKAAKSkp2LRpE4YNG1ar/8lze9hGre2hleDgYJhMJhw/frzWp+3m5oYdO3agb9++mDt3LkJCQjB69GgUFBSUO76tPbAm7rT/VYQ9r+6w5znf9rCLb6S/FXrKayxZWVlo2rQpAMBkMgEAbty4USvzbdKkCQCUubG+MsOGDbP6C0JEsHr16lqpp6aKi4tx9epVBAUFAfh9B120aFGZmnfv3m3z+qzt9V6RuLg4PPDAAxg7diy8vLwwcuRIjBo1yi6+h606uD3sm9FoxMCBA3H58mXs2rWrwvGuXr2KCRMmVHn6HTp0wObNm5GZmYkZM2YgISEBb775Zrnj2toDa+JO+19F2PPqTn07xrg97swuQlfHjh3h4eGBlJQUq+HJyckoLCxEt27dLOO5uLhg586dd5ymXq+/46nQ4OBgNGjQAF988UX1i6+CX375BePGjauz6X/11VcoLS1F165dAcDy5ElF3+pr6/qsynqviUOHDuHUqVO4dOkSioqKcO7cOSxbtgy+vr51Mj9uj8qpvT20EBcXB6PRiBdeeKHCs1Dp6elWXydhS2/JzMzE4cOHAdz8j2fevHno2rWrZdgf2doDa+JO+19d4DFWOfY859sedhG6TCYTpk6dio0bN2L16tXIyclBWloannnmGTRu3BgTJ04EcLN5hYeHIykpCStWrEBOTg5SU1PL/f6MVq1a4erVq9i0aROKiopw6dIlnD171moco9GImTNn4ptvvsHkyZPx888/o7S0FLm5uZbm2KBBA2RmZuLMmTPIzc29Y7Mtj4igoKAAGzZsgJeXVzXWUPkKCwuRnZ2N4uJi7N+/H5MnT0bz5s0xduxYADfX67hx47B27VosW7YMOTk5KCkpQUZGBn755Reb12dV1ntNPP/88wgKCkJeXl6l43322Wc1enya28M2tm4PR9a5c2esWbMG6enpCAsLw9atW5GdnY2ioiKcPn0a77//PsaPH2/1fUS29JbMzEw8/fTTOHr0KAoLC3HgwAGcPXsWvXr1AlC2r+h0Opt6YE3caf+rTTzGbMOe55jbo0b+eGd9VZ9e3LVrl7Rp08byhETjxo0lMjKy3HHnz58vbm5uAkCaNWsm8fHxlvdKS0tlwYIF0rp1azEYDOLr6ysjRoyQY8eOWU0jNzdXJkyYIH5+fuLh4SF9+/aVWbNmCQBp2rSp/PjjjyIicuXKFenfv7+YTCZp0aKFTJo0SaZPny4ApFWrVnLu3DnLNN955x0JDQ0Vk8kkJssyQ1kAACAASURBVJNJunTpIkuXLhURkf3790vz5s3Fzc1N+vbta/Xo9u02btxY4VMjt79eeeUVERFZvHixuLu7CwAJDg6Wb7/9Vl5//XXx9vYWAOLv7y9r1qyRdevWib+/vwAQX19fWbt2rYiI/Otf/5L+/ftLo0aNRK/Xi5+fn4wZM0bOnj1rVdeNGzdkxowZEhQUJHq9Xho2bCjh4eFy6NChKq1PW8ZbsmSJBAQECABxd3eXRx55RJYuXWpZztatW8upU6dk+fLl4uXlJQCkefPmlseLd+zYIX5+flbry2AwSLt27WTDhg2WZdq6dat4enrKnDlzKtwvuT3U2x62srenF2937tw5mTZtmoSGhoqHh4fodDrx8fGRLl26yPjx42XXrl2WcW3pLWfOnJE+ffqIr6+v6HQ6adKkibz88stSXFwsIuX3FVt6YE3308r2v8r6c3l4jLHnOev2sEVF/UgRsb5VPzExEVFRUXb/+0VU/yxbtgwnTpzAokWLLMMKCwvx0ksvYdmyZbh27Rrc3Nw0rNC51Pb2iIyMBACsX7++1mtVY/pEtY09z77U5vaoqB/VyW8vElXVhQsXMHny5DLX/l1dXREUFISioiIUFRWxAamE24OobvEYsy9qbQ+7uKeLyM3NDQaDAStWrMCvv/6KoqIiZGZm4oMPPsCsWbMwevToWr0XgSrH7UFUt3iM2Re1tgdDF9kFb29vfPHFF0hPT0ebNm3g5uaG9u3b41//+hdef/11fPTRR1qX6FS4PYjqFo8x+6LW9uDlRbIbYWFh+O9//6t1GfT/cXsQ1S0eY/ZFje3BM11EREREKmDoIiIiIlIBQxcRERGRChi6iIiIiFTA0EVERESkAoYuIiIiIhUwdBERERGpgKGLiIiISAUMXUREREQqYOgiIiIiUgFDFxEREZEKGLqIiIiIVMDQRURERKQCfUVvJCYmqlkHEdVzGRkZaNq0aZ3Pg72LiLRWUb+rMHRFRUXVaUFE5HwiIiLqdPp79uxh7yIiu1Bev1NERDSohZzExYsXERgYiH//+9+IjIzUuhwiIovDhw+jQ4cO2L9/P7p06aJ1OeQEeE8X1alGjRrhgQcewJo1a7QuhYjISlpaGvR6Pdq1a6d1KeQkGLqozkVHR+Ozzz7DlStXtC6FiMgiLS0Nbdq0gclk0roUchIMXVTnwsPDYTAYkJSUpHUpREQWqamp6NSpk9ZlkBNh6KI6ZzabMWzYMF5iJCK7kpaWhtDQUK3LICfC0EWqiI6OxnfffYczZ85oXQoREXJzc3H27FmGLlIVQxepYtCgQfDz88O6deu0LoWICKmpqRARXl4kVTF0kSoMBgMiIyMRHx+vdSlEREhNTYWXlxeCgoK0LoWcCEMXqSY6OhqHDx/Gjz/+qHUpROTkbt3PpSiK1qWQE2HoItX06dMHLVq04A31RKS5tLQ0Xlok1TF0kWoURcGYMWOwZs0alJSUaF0OETkpEUF6ejpvoifVMXSRqmJiYpCZmYlvv/1W61KIyEmdO3cOWVlZPNNFqmPoIlW1a9cOXbp04SVGItJMWloaFEVBhw4dtC6FnAxDF6kuOjoa69evR0FBgdalEJETSk1NRVBQEHx8fLQuhZwMQxepbsyYMcjLy8Nnn32mdSlE5IR4Ez1phaGLVNekSRPcf//9vMRIRJrgby6SVhi6SBPR0dH49NNPkZWVpXUpROREbty4gRMnTvDJRdIEQxdpIiIiAi4uLtiwYYPWpRCREzly5AiKiooYukgTDF2kCS8vLwwdOpSXGIlIVampqTAajWjTpo3WpZATYugizURHR2Pnzp04f/681qUQkZNIS0tD+/btodfrtS6FnBBDF2lmyJAh8PX1xbp167QuhYicRGpqKi8tkmYYukgzrq6uGDlyJC8xEpFqbv3QNZEWGLpIU9HR0fjxxx+Rnp6udSlEVM9dvnwZv/zyC78ugjTD0EWauu+++xAcHIx///vfWpdCRPVcamoqADB0kWYYukhTiqIgKioKa9asQWlpqdblEFE9lpaWhrvuugsBAQFal0JOiqGLNPfEE0/g3Llz2LVrl9alEFE9xp//Ia0xdJHm2rdvj9DQUN5QT0R1ij//Q1pj6CK7EB0djcTERNy4cUPrUoioHiotLcXhw4f55CJpiqGL7MJjjz2G7OxsfP7551qXQkT10MmTJ3H9+nWe6SJNMXSRXWjWrBnCwsJ4iZGI6kRaWhpcXFzQrl07rUshJ8bQRXYjOjoan3zyCbKzs7UuhYjqmbS0NLRq1Qpms1nrUsiJMXSR3YiIiICI4D//+Y/WpRBRPcOb6MkeMHSR3fD19cXgwYN5iZGIah1/c5HsAUMX2ZXo6Gjs2LEDP//8s2WYiOD777/HsWPHNKyMiBxBcXExzp8/bzXs+vXrOH36NEMXaU6vdQFEtxs2bBi8vLyQmJiIQYMG4d///jc+/PBDZGRkIDExEW3bttW6RCKyY4qioHXr1nB1dUXHjh3RtWtXeHp6orS0FC1bttS6PHJyioiI1kUQ3ZKZmYmRI0fixIkTuHr1KlxdXVFYWAgASEhIwKhRozSukIjs3d133205M67X6yEiKCkpgaIoaNy4Mbp164bOnTujZ8+eGDZsmMbVkjPh5UXSXHZ2NlatWoXBgwcjKCgIP/zwA65duwYAlsAF3LzMSER0J/fccw9cXG7+91ZcXIySkhIAN3tIZmYmtmzZgtmzZyM5OVnLMskJ8fIiaSovLw/33HMPzp49C51OZ2mOf6QoCn8Qm4hs0r59exgMhkp/4aJBgwZ48cUXVayKiGe6SGMeHh6Ij4+vNHABDF1EZLt27dpZnSX/I0VRsGDBAnh6eqpYFRFDF9mBsLAw/OMf/7jjeLy8SES2aNeuXYX9QqfT4e6778aTTz6pclVEDF1kJ5599ln89a9/hV5f8RVvhi4iskXbtm2h0+nKfa+kpASLFy+u8H2iusTQRXZj6dKl6NmzJwwGQ7nv8/IiEdnC1dUVgYGBZYYbDAYMHToUAwYM0KAqIoYusiMGgwEbN25EgwYNypzxUhSFZ7qIyGb33HMPFEWxGlZaWoo33nhDo4qIGLrIzvj7+2Pr1q2Wx71vx9BFRLbq0KEDXF1dLf82GAx47rnn0L59ew2rImfH0EV2p2vXrvjggw/KDOflRSKyVbt27VBUVGT5t8lkwqxZszSsiIihi+zU448/jueff95ysysvLxJRVbRv397yh5pOp8Pf//53+Pn5aVwVOTuGLrJbixYtQp8+fWAwGCAiPNNFRDa7++67oSgKFEVB06ZN8dxzz2ldEhFDF9kvvV6P//znPwgICEBJSQnPdBGRzTw8PODv7w8RweLFi63u7yLSCn/w2kZ/fAqGiKouIiIC69ev17qMCiUmJiIqKkrrMoickr33h9rA316sgilTpqB3795al+GU9u7di6ysLPz5z3/WuhSqpkWLFmldgs0SEhK0LoFqQXx8PP70pz8hJCRE61LoDhypP9QEQ1cV9O7dG6NGjdK6DKc0atQoXLp0CQ0bNtS6FKomR/oLlsd5/dC3b180adJE6zLIBo7UH2qC93SRw2DgIqKqYOAie8PQRURERKQChi4iIiIiFTB0EREREamAoYuIiIhIBQxdRERERCpg6CIiIiJSAUMXERERkQoYuoiIiIhUwNBFREREpAKGLiIiIiIVMHQRERERqYChi4iIiEgFDF1EREREKmDoqgMbNmxASEgIFEWp8BUcHAwAePPNN9GoUSMoioJ3331X28KrqLzlNBgMCAwMRHR0NI4cOVKn858wYQI8PT2hKAoOHjxYp/OqLc6ybzgLZ9yex44dw6RJk9ChQwd4enpCr9fD29sbbdq0wdChQ7F7926tS3RozrhPOROGrjoQHh6On376CS1btoS3tzdEBCKC4uJi5Ofn49dff4W7uzsAYNq0afj+++81rrh6ylvOrKwsvPvuu/juu+/Qs2dPHDt2rM7m/8EHH+D999+vs+nXBWfZN5yFs23PFStWIDQ0FKmpqVi4cCHOnz+P69ev48CBA5g9ezaysrKQlpamdZkOzdn2KWfD0KUinU4HNzc3NGrUCG3atKnRtAoKCtCnT587DlOb2WzGww8/jLfffht5eXlYsmSJpvXUhJrr0xn2DWdSH7fnnj17MHHiRISFhWH79u0YOHAgfHx8YDQaERISgqioKMyaNQuFhYW1Ol/uuzfVx33KGem1LsBZbdq0qUafX7FiBS5evHjHYVrp2bMnACA9Pb1O56MoSp1NW6v1Wd/3DWdTX7bnnDlzUFJSgnnz5kGvL/+/joEDB2LgwIG1Ol/uu2XVl33KGfFMl5369ttv0b59e3h7e8NkMiE0NBSff/45AGDKlCmYOnUqTp06BUVR0KpVq3KHAUBJSQlmzZqFoKAguLm5oVOnTkhISAAALFu2DGazGe7u7vj4448xePBgeHl5oWnTpli7dm2N6i8uLgYAGI1GAMAbb7wBd3d3eHp64uLFi5g6dSoCAwNx7NgxiAgWLlyIdu3awWg0wtfXF8OHD8fRo0etpikiWLBgAdq2bQuj0Qhvb29Mnz7dapzJkyfD1dUVAQEBlmHPPfcczGYzFEXB5cuXrcaPj49H9+7dYTKZYDabERwcjNmzZ1e4Pnfu3ImePXvC3d0dXl5eCA0NRU5ODgBg27Zt8PLywty5c2u07u7E0fcNsuYI27OwsBDbt2+Hn5+f5Q+qO6nKsVjRcVXRstrSMxYvXgyz2QwXFxd069YN/v7+MBgMMJvN6Nq1K8LCwtCsWTOYTCb4+PjgxRdftKq/svVZWT9jj2CPqJSQTQBIQkJClT7TsmVL8fb2thq2fft2WbBggdWwEydOCAD55z//aRm2fv16iYuLk6tXr8qVK1ekV69e4ufnZ3k/PDxcWrZsaTWd8oZNmzZNjEajJCUlybVr12TmzJni4uIi+/btExGRl19+WQDI9u3bJTs7Wy5evChhYWFiNpulsLCw2ssZHx8vAGT69OmWYbfmFRsbK0uWLJGRI0fKkSNHZNasWeLq6irx8fGSlZUlqamp0rVrV7nrrrvkwoULVp9XFEXeeustuXbtmuTn58vSpUsFgBw4cMAyXnR0tPj7+1vVs2DBAgEgly5dsgxbtGiRAJB58+bJlStX5OrVq/Lee+9JdHR0ueszLy9PvLy8ZP78+VJQUCAXLlyQkSNHWqa5ZcsW8fT0lFdffbVa66w+7hu3i4iIkIiIiCp/Tk0JCQlSnbZY37fn8ePHBYD06tXL9pUith2LdzquyltWW3vG3/72NwEgycnJcv36dbl8+bIMGjRIAMinn34qly5dkuvXr8vkyZMFgBw8eLDK6/P2fpaSksIeUc0e4Qj9oTYwdNmouqELQJmXLQfNH7322msCQC5evCgith00BQUF4u7uLqNHj7YMy8/PF6PRKM8++6yI/H7QFBQUWMa5FWROnjxp83Leag55eXmSlJQk/v7+0qhRI8nIyLCMV9688vPzxcPDw6pGEZG9e/cKAEtzys/PF3d3dxkwYIDVeGvXrq1W6CosLBQfHx/p37+/1XjFxcWyePFiESm7PtPT0wWAbNmyxab1Uhln2Tdu5whNtSahqz5vz5SUFAEgDz300B3WhDVbjsU7HVd/XFZbe4bI76ErNzfXMuyjjz4SAJKWllbms+vWrROR6q9P9ojq9whH6A+1gZcX69jtT5+ICL766qtqTcdgMAC4ebrXVseOHUN+fj46duxoGebm5oaAgIAyl+5u5+rqCgAoKiqyeV7Z2dlQFAXe3t6IjY3FkCFDsHfvXgQGBlb6uUOHDiEvLw/du3e3Gt6jRw+4uroiOTkZAHDy5Enk5+fjwQcftLmmyqSmpiIrK6vM/Sc6nQ6xsbHlfiYkJASNGjVCTEwM4uLicObMmRrV4Cz7hrOoz9vTw8MDAJCfn29zTbaq6nFla8+oyK1lvnULBPD7Or+1Hqq7Ptkj2CPuhKFLZf369cO0adPuON6nn36Kfv36oWHDhjAajWXuN7DF9evXAQCvvPKK1Xe8nD17tkrNc8uWLWW+JyYmJsZqnFvNobi4GBkZGVi5ciWaN29+x2lnZWUB+L2p387Hxwe5ubkAgIyMDABAw4YNba67MrfusfDx8bH5M25ubtixYwf69u2LuXPnIiQkBKNHj0ZBQUGt1OSI+wZVzBG3Z0XHenBwMEwmE44fP17l2u6kqseVrT2jJqq7Ptkj6E4YuuzQuXPnMGLECAQEBCA5ORnZ2dmYP39+ladzK6AsWrTI6q8lEanSFxgOGzaszOdXr15d5XrKcyv0lNcos7Ky0LRpUwCAyWQCANy4caNW5tukSRMAKHNj/Z106NABmzdvRmZmJmbMmIGEhAS8+eabtVKTLext36CasbftWdGxbjQaMXDgQFy+fBm7du2q8PNXr17FhAkTqlx/VY4rW3tGTdRkfbJHUGUYuuxQWloaioqK8OyzzyIkJAQmk6laX41w68kce/629o4dO8LDwwMpKSlWw5OTk1FYWIhu3bpZxnNxccHOnTvvOE29Xn/HU9vBwcFo0KABvvjiC5trzczMxOHDhwHcbEjz5s1D165dLcPU4Ez7hjNwpO0ZFxcHo9GIF154ocIzN+np6VZfJ2HLsVjV48rWnlET1V2f7BF0JwxddigoKAgA8OWXX+K3337DiRMnytyn0KBBA2RmZuLMmTPIzc1FUVFRmWE6nQ7jxo3D2rVrsWzZMuTk5KCkpAQZGRn45ZdftFi0MkwmE6ZOnYqNGzdi9erVyMnJQVpaGp555hk0btwYEydOBHCzgYWHhyMpKQkrVqxATk4OUlNTsXz58jLTbNWqFa5evYpNmzahqKgIly5dwtmzZ63GMRqNmDlzJr755htMnjwZP//8M0pLS5Gbm2tpkH9cn2fPnsXTTz+No0ePorCwEAcOHMDZs2fRq1cvAMBnn31W54+DO9O+4QwcaXt27twZa9asQXp6OsLCwrB161ZkZ2ejqKgIp0+fxvvvv4/x48db7h8CbDsWMzMzKz2uyltWW3pGTZhMpmqtzzstC3sE8elFG6EKTy/u2rVL2rRpY3naJCAgQB588MFyx33rrbfE399fAIjZbJaRI0eKiMiMGTOkQYMG4uPjI5GRkfLOO+8IAGnZsqWcO3dO9u/fL82bNxc3Nzfp27evXLhwodxhN27ckBkzZkhQUJDo9Xpp2LChhIeHy6FDh2Tp0qXi7u4uAKR169Zy6tQpWb58uXh5eQkAad68uRw/ftzm5WzcuLFERkaWO+78+fPFzc1NAEizZs0kPj7e8l5paaksWLBAWrduLQaDQXx9fWXEiBFy7Ngxq2nk5ubKhAkTxM/PTzw8PKRv374ya9YsASBNmzaVH3/8UURErly5Iv379xeTySQtWrSQSZMmyfTp0wWAtGrVSs6dO2eZ5jvvvCOhoaFiMpnEZDJJly5dZOnSpSIiZdZncnKy9OnTR3x9fUWn00mTJk3k5ZdfluLiYhER2bp1q3h6esqcOXOcft8ojyM8nVTVpxedcXueO3dOpk2bJqGhoeLh4SE6nU58fHykS5cuMn78eNm1a5dlXFuOxTNnzlR6XJW3rLb0jMWLF1uWOTg4WL799lt5/fXXxdvbWwCIv7+/rFmzRtatW2fZLr6+vrJ27VoRkUrXZ0X97E7Lwh5RMUfoD7VBERGp21hXPyiKgoSEBIwaNUrrUogcUmRkJABg/fr1GldSscTERERFRYFtkUhdjtAfagMvLxIRERGpgKGLiIiISAUMXUREREQqYOgiIiIiUgFDFxEREZEKGLqIiIiIVMDQRURERKQChi4iIiIiFTB0EREREamAoYuIiIhIBQxdRERERCpg6CIiIiJSAUMXERERkQoYuoiIiIhUwNBFREREpAKGLiIiIiIVMHQRERERqUCvdQGOJCoqClFRUVqXQeSwIiIitC7BJoqiaF0CkdNxlP5QEwxdNkpISNC6BLJDmZmZePXVV2EymTB9+nQEBgZqXZJda9asmdYlVKpPnz481m1UUlKCpKQkbNq0CV27dsXUqVPh4sKLJ1R99t4faoMiIqJ1EUSO7JdffsHIkSNx6NAhfPTRRxgxYoTWJRHVqdOnT+Pxxx/H/v37MW/ePEyePJlnB4lswD9LiGqocePG+PrrrzFq1CiEh4fjpZdeQmlpqdZlEdWJVatWoVOnTsjJyUFycjJiY2MZuIhsxNBFVAuMRiM++OADvPvuu1i4cCGGDx+OnJwcrcsiqjVZWVkYM2YMxo4di7/85S9ISUlBaGio1mURORReXiSqZd9++y0iIiIQEBCATZs2oUWLFlqXRFQj27dvx5NPPgkXFxesWrUK/fr107okIofEM11EtSwsLAwpKSkwGAzo0aMHtm/frnVJRNVy48YNvPTSS/jzn/+MXr164eDBgwxcRDXA0EVUB5o1a4adO3eif//+GDRoEObPn691SURVcvjwYfTq1QvLli3DP//5TyQlJaFBgwZal0Xk0Bi6iOqI2WxGYmIi5syZg5kzZyImJgYFBQVal0VUKRHB8uXL0aNHDxiNRuzfvx9PPfWU1mUR1Qu8p4tIBVu3bkV0dDRCQkKwadMmp/g+GnI8Fy9exPjx47Ft2zZMnToVs2fPhsFg0LosonqDoYtIJcePH8fw4cORnZ2NDRs2oFevXlqXRGTx+eefY9y4cTAajYiPj0ffvn21Lomo3uHlRSKVtGnTBnv27EH37t3Rr18/rFy5UuuSiFBQUIDY2FgMHjwYffv2xYEDBxi4iOoIQxeRiry8vLBx40ZMmTIFEyZMwMSJE1FUVKR1WeSk0tLScO+99+Kjjz7C6tWrkZiYCB8fH63LIqq3GLqIVKbT6fD6669j7dq1WL16NR566CFcvHhR67LIiYgI3n77bXTv3h0NGzZEeno6HnvsMa3LIqr3eE8XkYYOHjyIESNGoKSkxPLDwUR16fz583jiiSewa9cuzJw5E7NmzeIPVROphEcakYY6d+6Mffv2oVWrVrj//vuxYcMGrUuieiwpKQmdO3fGhQsXsGfPHsTFxTFwEamIRxuRxu666y588cUX+Mtf/oLIyEj+YDbVutzcXEycOBGRkZEYOnQoUlJSeFaVSAO8vEhkR5YvX45JkyZhwIABWLNmDby9vbUuiRxccnIyYmJikJ2djRUrVuDhhx/WuiQip8UzXUR25KmnnsKOHTvwww8/oGfPnjh69KjWJZGDKi4uxvz58xEWFoaQkBAcPHiQgYtIYwxdRHbmT3/6E1JSUuDt7Y17770Xmzdv1rokcjBnzpxB//79ERcXhwULFmDbtm1o0qSJ1mUROT2GLiI7FBgYiG+++QYjRozAiBEj+IPZZLNVq1ahU6dOyMrKQnJyMmJjY6EoitZlEREYuojslslkwocffohly5bhlVdewZgxY5Cfn691WWSnsrOzER0djbFjx2LcuHFISUlBp06dtC6LiG7DG+mJHMDnn3+OMWPGoHnz5ti0aROaN2+udUlkR3bs2IEnn3wSxcXFWLlyJQYPHqx1SURUDp7pInIAAwcOxN69e1FUVITu3bvj66+/1roksgNFRUWIi4vDgAED0LNnT6SnpzNwEdkxhi4iB9GqVSvs3r0bYWFhGDBgAJYsWaJ1SaShI0eO4N5778WCBQuwcOFCbNiwAX5+flqXRUSVYOgiciCenp7YsGED5syZgylTpmDixIkoLCzUuixS2apVq9CjRw8YDAYcPHgQsbGxWpdERDbgPV1EDmrz5s2IiYlBp06dkJSUBH9/f61Lojp26dIljB8/Hlu3bsW0adMwe/ZsGAwGrcsiIhsxdBE5sNTUVAwfPhxFRUX4z3/+g+7du2tdEtWRL774AmPHjoWrqyvi4+MRFhamdUlEVEW8vEjkwDp16oR9+/ahbdu2CAsLw6pVq7QuiWrZb7/9htjYWAwaNAh9+/bFgQMHGLiIHBRDF5GD8/Pzw7Zt2xAbG4snn3wSsbGxKCkp0bosqgXp6em499578dFHH2HVqlVITEyEr6+v1mURUTUxdBHVA3q9Hq+//jpWr16N999/H8OGDUNWVpbWZVE1iQjefvttdO/eHWazGfv370dMTIzWZRFRDfGeLqJ6Zv/+/Rg+fDhMJhM2bdqE9u3ba10SVcGFCxcwbtw4fPnll3j55Zfxf//3f9DpdFqXRUS1gGe6iOqZrl27Ys+ePfDz80OvXr2wadMmrUsiG23cuBEdO3bEmTNnsGfPHsTFxTFwEdUjDF1E9VCTJk3w9ddfIzIyEiNHjsRLL70EntS2X7m5uZg4cSLCw8MxZMgQpKSkoFu3blqXRUS1TK91AURUN4xGI1asWIF7770Xzz//PE6fPo2VK1fCbDZrXRrdZu/evYiJiUFWVhY2bdqERx99VOuSiKiO8EwXUT331FNP4csvv8TXX3+NPn364PTp01qXRABKSkowf/589O3bF8HBwTh48CADF1E9x9BF5ATuu+8+pKSkQK/Xo0ePHtixY0el42dnZ6tUmXM6e/Ys+vfvj7i4OMyePRvbtm1DkyZNtC6LiOoYQxeRk2jWrBl27tyJfv36YeDAgZg/f365461YsQIjRozgPWDV9Pbbb6O0tLTC99evX4/OnTvj6tWr2LNnD2bMmAEXF7ZiImfAI53IiXh4eGD9+vWYM2cOZs6ciccffxy//fab5f3vv/8ezzzzDL766iusWbNGw0od06pVqzBlyhS89dZbZd7Lzs5GTEwMoqKiMGrUKOzduxf33HOPBlUSkVb4PV1ETurTTz9FdHQ0OnTogKSkJABA586dceXKFZSWlsLX1xcnT57kN6Db6KeffkJoaCjy8/Oh1+uxd+9edOnSBcDNMBsTE4P8/HysXLkSQ4YM0bhaItICz3QROamhQ4di165duHjxInr06IEHHngA165dQ0lJCUQEubm5+N///V+ty3QIxcXFFT/7zwAAIABJREFUGDNmDIqKiizDRo0ahZycHMTFxeG+++5D586dkZ6ezsBF5MR4povIyV29ehVdunTBzz//XOY3GxVFwa5du9C7d2+NqnMMt26Iv/1eLr1ejz59+mD//v1YtGgRJkyYoGGFRGQPGLqInNzChQsxbdq0cm+c1+v1aNWqFVJTU2EwGDSozv7t27cPvXv3LvdHxhVFwfLlyxm4iAgAQxeRU/vyyy8xcODASp+20+l0eOONN/DCCy+oWJljyMvLQ2hoKDIyMlBcXFzmfUVR4O3tjSNHjiAgIECDConInjB0ETmpU6dOoWvXrsjLy6s0dAGAyWTC8ePH0axZM5WqcwxPPPEE1q1bZ3Uv1x8ZDAbcd999+O9//wtFUVSsjojsDW+kJ3JSO3fuhNlsRmlp6R0vHZaUlGDSpEkqVeYYEhMTER8fX2ngAoCioiJs374dy5YtU6kyIrJXDF1ETuovf/kLMjMzkZKSgqeffhre3t5QFKXcAFZUVISPP/4Ymzdv1qBS+3P+/HlMmDCh0jNXOp0OLi4uUBQFoaGhyMnJUbFCIrJHvLxIRACAGzdu4IsvvsC6deuwYcMGFBcXQ0Qslx5dXFzg7++PEydOOPWPZpeWlqJ///7YvXt3mbNcBoMBRUVFcHNzQ//+/fHoo49i6NChCAwM1KhaIrInDF1EVMa1a9ewfv16fPjhh9izZw/0er0lYLz00kuYN2+exhVq5/XXX7d8f5lOp4OIQETQrl07PProoxgyZAh69+4NnU6ncaVEZG8YuohUtHv3bixcuFDrMqokPz8f586dw5kzZ5CXlwdFUTBgwAB4eXlpXZrqrl27hq+++gqlpaXQ6/Xw9/dHQEAAAgIC4ObmpnV5FVq/fr3WJRARGLqIVJWYmIioqChERERoXUq1ZGVl4dy5c8jPz0evXr20LkdVxcXFSElJgdlsRkBAAPz8/Oz+h6ozMjKwZ88e/ng5kZ3Qa10AkTNy9DMPt34qSK93nhYiIg73lQ+3Qj4R2Qfn6ZhEVGuc8X4lRwtcRGR/7PvcOBEREVE9wdBFREREpAKGLiIiIiIVMHQRERERqYChi4iIiEgFDF1EREREKmDoIiIiIlIBQxcRERGRChi6iIiIiFTA0EVERESkAoYuIiIiIhUwdBERERGpgKGLiIiISAUMXUR2bMOGDQgJCYGiKFYvV1dXNGrUCP369cOCBQtw7do1TWoxGAwIDAxEdHQ0jhw5UqfznzBhAjw9PaEoCg4ePFin86oKe9pGRGTfGLqI7Fh4eDh++n/t3X10lPWd///XlQm5mZBJKA0iTrhJRNnlpugqSwN+peth29SbtiSBFDCNHqyW7rZrWzbWWNZDpS5GG3tcOGyQpXvo2XRi7EHkCO0e2bKri23pciPQgMAJEgMkRSTkpglJ3r8//BFNIYHcfWYGn49z5g+uueb6vHPFpk9mrlwcO6bMzEylpKTIzNTZ2am6ujpVVFRowoQJKioq0uTJk7Vr1y7ns3zwwQdau3at3njjDc2YMUOHDh0asvVffPFFrVu3bsiO31+R9D0CENmILiDKeJ6n1NRUzZkzRxs2bFBFRYVOnz6tu+++W+fOnevz8VpaWpSVldWvWZKSknTvvffqJz/5iRobG/XCCy/06ziRYCDn4c9F0vcIQOQguoAol5ubq8LCQtXV1Wnt2rV9fv369etVV1c3oBlmzJghSdq/f/+AjnMlnucN2bEH4zz0JBK+RwDCj+gCrgGFhYWSpK1bt3ZtMzP9+Mc/1l/8xV8oPj5eI0aM0Je//GVVVVV17fMP//AP+u53v6ujR4/K8zzdeOON/Vq/vb1dkhQfHy9JeuaZZ+T3+5WcnKy6ujp997vf1Q033KBDhw5d1VwX5y8pKdHNN9+s+Ph4paSkaNmyZd32+da3vqW4uDiNHj26a9s3v/lNJSUlyfM8/fGPf+y2/8aNG3XbbbcpISFBSUlJGj9+vH74wx/2eB62bdumQCCglStX9uu8fFy4v0cAIoABcCYUCll//meXmZlpKSkpPT7f0NBgkiw9Pb1r2/Llyy0uLs42btxoH3zwge3bt89uvfVW+/SnP22nTp3q2i8nJ8cyMzMHNMvGjRtNki1btqxrW3FxsUmyb3/72/bCCy/YvHnz7A9/+MNVz1VcXGye59lzzz1nZ8+etebmZlu9erVJst27d3ftt2jRIrvuuuu6zVNSUmKSrL6+vmtbaWmpSbKnn37azpw5Y++//77967/+qy1atKjH87BlyxZLTk62FStW9Ou8fJzL79FF/f3vDcDQ4H+NgENDFV1mZp7nWWpqqpmZNTc32/Dhwy0/P7/bPr/97W9NUreIGEh0NTY2WmVlpV133XU2atQoq6mp6drvYnS1tLR0bbvauZqbm83v99vcuXO77VdeXt6v6Gpra7PU1FT73Oc+122/9vZ2e/755/t1Hv5cJH2PLiK6gMgS6/ytNQCDrqmpSWamQCAgSTpw4IAaGxt12223ddvv9ttvV1xcnH7zm98MaL1z587J8zz5fD6NHj1aX/ziF/VP//RPuuGGG3p93dXOdeTIETU3N+uuu+4a0JwX7du3Tx988IE+//nPd9vu8/n07W9/e1DWuBLX3yMAkYdruoBrwOHDhyVJkyZNkiR98MEHkqThw4dfsm9qaqrOnz/f47G2bNlyyT2nFi9e3G2fi7dGaG9vV01Njf7t3/5N48aNu+KcVztXTU2NJCktLe2Kx7waDQ0NXWuEy2B+jwBEJ97pAq4B27ZtkyRlZ2dL+iguLvd/3B988IGCwWCPx7rnnntkZkMw5dXPlZCQIElqbW0dlHXHjBkjSZdcWO/SYH6PAEQn3ukCotypU6dUWlqqYDCoBx98UJI0ZcoUDR8+/JKbcf7mN79RW1ub/uqv/ioco171XFOmTFFMTIx27NhxxWPGxsbqwoULve4zfvx4fepTn9KvfvWr/g8/ANH0PQIwdIguIEqYmRobG9XZ2SkzU319vUKhkGbNmiWfz6dNmzZ1XS+UkJCg7373u/rFL36hn/3sZ2poaNDbb7+tb3zjG7r++uv18MMPdx33U5/6lGpra1VdXa3z589fMWAG4mrnSktLU05OjiorK7V+/Xo1NDRo3759Kisru+SYN954o95//31t2rRJFy5cUH19vY4fP95tn/j4eD3++OP67//+b33rW9/Se++9p87OTp0/f14HDx7s8Txs3bq1T7eMuBa+RwCGUPiu4Qc+efr622SbN2+2adOmmd/vt7i4OIuJiTFJXb8FN2PGDFuxYoWdOXPmktd2dnZaSUmJTZw40YYNG2YjRoywr3zlK3bo0KFu+/3f//2fjRs3zhITE2327NndblXwcW+++abddNNNJskk2fXXX295eXmX3XfVqlWWmJjYdYuEjRs39nmu8+fP25IlS2zkyJE2fPhwmz17ti1fvtwkWTAYtL1795qZ2ZkzZ+xzn/ucJSQk2IQJE+zv//7vbdmyZSbJbrzxRnv33Xe7jvkv//IvNnXqVEtISLCEhAS75ZZbbPXq1T2eh9dee82Sk5Ptqaeeiorv0Z/jtxeByOKZDdHFGwAuUVFRoQULFgzZNVPAx/HfGxBZ+HgRAADAAaILAADAAaILAADAAaILAADAAaILAADAAaILAADAAaILAADAAaILAADAAaILAADAAaILAADAAaILAADAAaILAADAAaILAADAAaILAADAAaILAADAAaILAADAAaILAADAgdhwDwB8EuXl5YV7BHwC1NTUhHsEAB/DO12AQ+np6crNzQ33GFGntrZWmzdvDvcYUScYDPLfGxBBPDOzcA8BAL2pqKjQggULxI8rANGMd7oAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAcILoAAAAc8MzMwj0EAFz03nvv6d5779WFCxe6tjU1Nam+vl7jx4/vtu/06dO1ceNGxxMCQP/EhnsAAPi4G264QX/605/0hz/84ZLn9u/f3+3PCxYscDUWAAwYHy8CiDgFBQWKjb3y3wmJLgDRhI8XAUScd999V+PHj1dPP548z9Mtt9yi3//+944nA4D+450uABFn7Nixuv322xUTc/kfUT6fTwUFBY6nAoCBIboARKSCggJ5nnfZ5zo6OpSXl+d4IgAYGKILQESaP3/+Zbf7fD7deeedGjNmjOOJAGBgiC4AESktLU1z5syRz+e75Ln7778/DBMBwMAQXQAi1v3333/JxfQxMTGaN29emCYC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sVGdnZ0yM9XX1ysUCmnWrFny+XzatGlT1zVd1+o5AxBmji/cB9CDvv72opnZ+fPnbcmSJTZy5EgbPny4zZ4925YvX26SLBgM2t69e83MrLW11YqKimzs2LEWGxtraWlplpOTYwcOHLDVq1eb3+83STZx4kQ7evSolZWVWSAQMEk2btw4O3z4sFVXV1tWVpaNGDHCfD6fjRkzxoqLi629vf2Ka/TFzp07bdasWXb99debJJNko0ePtqysLNuxY0fXfq+99polJyfbU0891eOxNm/ebNOmTTO/329xcXEWExNjkrp+U3HGjBm2YsUKO3PmzCWvjaZzJn57EYgKnplZ2IoPQJe8vDxJ0ksvvRTmSRBtPM9TKBTS/Pnzwz0KgF7w8SIAAIADRBeAIVVVVdXtlgg9PfLz88M9KgAMqdhwDwDg2jZp0iRxFQMA8E4XAACAE0QXAACAA0QXAACAA0QXAACAA0QXAACAA0QXAACAA0QXAACAA0QXAACAA0QXAACAA0QXAACAA0QXAACAA0QXAACAA0QXAACAA0QXAACAA7HhHgDARyorK+V5XrjHAAAMAc/MLNxDAJB27typEydOhHuMiLRz5049//zzCoVC4R4lYmVlZSkYDIZ7DAC9ILoARLyKigotWLBA/LgCEM24pgsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMABogsAAMCB2HAPAAAf19LSopMnT3bbdvr0aUnSsWPHum33+XwaN26cs9kAYCA8M7NwDwEAF505c0ajR49We3v7Fff9whe+oK1btzqYCgAGjo8XAUSUkSNHau7cuYqJ6f3Hk+d5ys/PdzQVAAwc0QUg4ixevFhXehM+NjZWX/7ylx1NBAADR3QBiDhf+tKXFB8f3+PzsbGxuu+++5SSkuJwKgAYGKILQMRJSkrSl770JQ0bNuyyz3d0dGjRokWOpwKAgSG6AESkRYsW6cKFC5d9LjExUdnZ2Y4nAoCB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\n",
            "text/plain": [
              "<IPython.core.display.Image object>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 84
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hMwnHNQn06ro"
      },
      "source": [
        "# fit_plot_save(neural_mf_model, 'neural_mf_model.h5')\n",
        "history = neural_mf_model.fit([train.customer_id, train.product_id], train.star_rating, batch_size=4096, epochs=10, verbose=1)\n",
        "neural_mf_model.save('neural_mf_model.h5')\n",
        "plt.plot(history.history['loss'])\n",
        "plt.xlabel(\"Epochs\")\n",
        "plt.ylabel(\"Training Error\")"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "S7t_ljsMnsRH"
      },
      "source": [
        "### Meilleur Loss: 2.6\n",
        "### Temps de calcul: 31s\n",
        "### Epochs: 10"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "InsycR2DsrUm"
      },
      "source": [
        "#### 2.3 - Model Final (MLP + MF + DNN)\n",
        "#### Concaténation de la sortie du model précédent avec un Multi Layer Perceptron (MLP) puis ajout d'un DNN à la sortie de celle-ci"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "TzEfHqJZyjYp"
      },
      "source": [
        "################---Input Layer---################\n",
        "# product input\n",
        "product_input = Input(shape=[1], name=\"Product-Input\")\n",
        "\n",
        "# customer input \n",
        "customer_input = Input(shape=[1], name=\"Customer-Input\")\n",
        "\n",
        "################---Matrix-Factorization embeddings Layer---################\n",
        "# product Matrix-Factorization vector\n",
        "product_embedding = Embedding(n_products+1, n_latent_factors, name=\"Product-Embedding\")(product_input)\n",
        "product_vec = Flatten(name=\"Flatten-Products\")(product_embedding)\n",
        "\n",
        "# customer Matrix-Factorization vector\n",
        "customer_embedding = Embedding(n_customers+1, n_latent_factors, name=\"Customer-Embedding\")(customer_input)\n",
        "customer_vec = Flatten(name=\"Flatten-Customers\")(customer_embedding)\n",
        "\n",
        "################---Matrix-Factorization output Layer---################\n",
        "mf_output_layer = Dot(name=\"Dot-Product\", axes=1)([product_vec, customer_vec])\n",
        "\n",
        "\n",
        "################---Multi-Layer-Perceptron embeddings Layer---################\n",
        "# product Multi-Layer-Perceptron embedding\n",
        "product_perceptron_embedding = Embedding(n_products+1, n_latent_factors, name=\"Product-Perceptron-Embedding\")(product_input)\n",
        "product_perceptron_vec = Flatten(name=\"Flatten-Perceptron-Products\")(product_perceptron_embedding)\n",
        "\n",
        "# customer Multi-Layer-Perceptron embedding\n",
        "customer_perceptron_embedding = Embedding(n_customers+1, n_latent_factors, name=\"Customer-Perceptron-Embedding\")(customer_input)\n",
        "customer_perceptron_vec = Flatten(name=\"Flatten-Perceptron-Customers\")(customer_perceptron_embedding)\n",
        "\n",
        "################---Multi-Layer-Perceptron embeddings concactenation Layer---################\n",
        "# mlp_conc = Concatenate()([product_perceptron_vec, customer_perceptron_vec, helpful_votes_perceptron_vec])\n",
        "mlp_conc = Concatenate()([product_perceptron_vec, customer_perceptron_vec])\n",
        "\n",
        "################---Multi-Layer-Perceptron fully-connected Layers---################\n",
        "fully_connected_layer0 = Dense(128, activation='relu')(mlp_conc)\n",
        "fully_connected_layer1 = Dense(128, activation='relu')(fully_connected_layer0)\n",
        "fully_connected_layer2 = Dense(64, activation='relu')(fully_connected_layer1)\n",
        "\n",
        "################---Multi-Layer-Perceptron output Layer---################\n",
        "mlp_output_layer = Dense(1, activation='relu')(fully_connected_layer2)\n",
        "\n",
        "\n",
        "# # ################---Concactenation Layer---################\n",
        "conc = Concatenate()([mf_output_layer, fully_connected_layer2])\n",
        "\n",
        "fully_connected_layer3 = Dense(128, activation='relu')(Dense(64, activation='relu')((conc)))\n",
        "\n",
        "# ################---Output Layer---################\n",
        "output_layer = Dense(1)(fully_connected_layer3)\n",
        "\n",
        "\n",
        "# Create model and compile\n",
        "neural_mf_mlp_model = Model([customer_input, product_input], output_layer)\n",
        "neural_mf_mlp_model.compile('adam',  loss='mean_squared_error')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "R0ubOOxFq5MC",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 850
        },
        "outputId": "cf3da255-a261-4c0b-b9c6-ebeb17eaf11b"
      },
      "source": [
        "neural_mf_mlp_model.summary()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Model: \"functional_17\"\n",
            "__________________________________________________________________________________________________\n",
            "Layer (type)                    Output Shape         Param #     Connected to                     \n",
            "==================================================================================================\n",
            "Product-Input (InputLayer)      [(None, 1)]          0                                            \n",
            "__________________________________________________________________________________________________\n",
            "Customer-Input (InputLayer)     [(None, 1)]          0                                            \n",
            "__________________________________________________________________________________________________\n",
            "Product-Perceptron-Embedding (E (None, 1, 15)        2786640     Product-Input[0][0]              \n",
            "__________________________________________________________________________________________________\n",
            "Customer-Perceptron-Embedding ( (None, 1, 15)        32292390    Customer-Input[0][0]             \n",
            "__________________________________________________________________________________________________\n",
            "Flatten-Perceptron-Products (Fl (None, 15)           0           Product-Perceptron-Embedding[0][0\n",
            "__________________________________________________________________________________________________\n",
            "Flatten-Perceptron-Customers (F (None, 15)           0           Customer-Perceptron-Embedding[0][\n",
            "__________________________________________________________________________________________________\n",
            "concatenate (Concatenate)       (None, 30)           0           Flatten-Perceptron-Products[0][0]\n",
            "                                                                 Flatten-Perceptron-Customers[0][0\n",
            "__________________________________________________________________________________________________\n",
            "Product-Embedding (Embedding)   (None, 1, 15)        2786640     Product-Input[0][0]              \n",
            "__________________________________________________________________________________________________\n",
            "Customer-Embedding (Embedding)  (None, 1, 15)        32292390    Customer-Input[0][0]             \n",
            "__________________________________________________________________________________________________\n",
            "dense_5 (Dense)                 (None, 128)          3968        concatenate[0][0]                \n",
            "__________________________________________________________________________________________________\n",
            "Flatten-Products (Flatten)      (None, 15)           0           Product-Embedding[0][0]          \n",
            "__________________________________________________________________________________________________\n",
            "Flatten-Customers (Flatten)     (None, 15)           0           Customer-Embedding[0][0]         \n",
            "__________________________________________________________________________________________________\n",
            "dense_6 (Dense)                 (None, 128)          16512       dense_5[0][0]                    \n",
            "__________________________________________________________________________________________________\n",
            "Dot-Product (Dot)               (None, 1)            0           Flatten-Products[0][0]           \n",
            "                                                                 Flatten-Customers[0][0]          \n",
            "__________________________________________________________________________________________________\n",
            "dense_7 (Dense)                 (None, 64)           8256        dense_6[0][0]                    \n",
            "__________________________________________________________________________________________________\n",
            "concatenate_1 (Concatenate)     (None, 65)           0           Dot-Product[0][0]                \n",
            "                                                                 dense_7[0][0]                    \n",
            "__________________________________________________________________________________________________\n",
            "dense_10 (Dense)                (None, 64)           4224        concatenate_1[0][0]              \n",
            "__________________________________________________________________________________________________\n",
            "dense_9 (Dense)                 (None, 128)          8320        dense_10[0][0]                   \n",
            "__________________________________________________________________________________________________\n",
            "dense_11 (Dense)                (None, 1)            129         dense_9[0][0]                    \n",
            "==================================================================================================\n",
            "Total params: 70,199,469\n",
            "Trainable params: 70,199,469\n",
            "Non-trainable params: 0\n",
            "__________________________________________________________________________________________________\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2mYsPLEm13vY",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 857
        },
        "outputId": "38622f60-ebbc-4489-9752-2418c294ff42"
      },
      "source": [
        "plot_model(neural_mf_mlp_model, to_file='neural_mf_mlp_model.png', show_shapes=True)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "image/png": "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z5w5U/5zTyGEpaWlTqebPHny0qVL+cQTqL7z58+/8cYbCQkJZSp2pUVHR48fP95wax00FteuXXv66afv3btXYauFhYW5ufnChQvfe+89c3PzBs4GND0FBQXLly8vfeOjMqysrLy9vQcOHCgVEjp37sxbDwCAZo66AgAADaFDhw6JiYmV7GBhYeHm5vb5558PGjSowVIBjZRWq124cOGKFSvMzMzKf5/dwMrKavbs2fPnz2/IbDCKa9eueXt7V76PmZlZ165dv/766969ezdMKqBJ2rNnz8SJE9PT0yuZThUKxeDBg7///vuGDAYAAEwZdQUAQHO3cuXK48eP1+sQRUVFcXFx1dzZw8OjZ8+eNjY29RoJaLwyMjLOnDkjrYJeJScnp4CAgPqOVH3bt2+veycNMGvJ7saNG2fOnKnOnyoKhaJz586dOnUyMzNrgGBAU5KXl3fu3LmMjIzq7GxtbT106ND6jlQX06dP9/X1lTsFAADNBb98AwCau+PHj584caJeh7h//375jQqFovTNiC0tLTUaTatWrWxsbNLT0yn818WJEyfq+zU1BampqbGxsXKnaGgPHz68e/eui4uLq6urnZ2dtbV16VYzMzNzc/PSny9nZWWVv0u4LIz4ejXArCW7u3fvGv6tUCjMzMzKlA3Mzc3VanXLli1bt25dUlKSmZnZ4BmbqeYz88TGxqampsqdoh7pdLqMjAyNRuPq6qpWqw1vMYVCYW5uXn69hEePHmm12gaPWV2xsbEpKSlypwAAoBmxkDsAAADy69u3r1G+RPw4S5cuPXLkiPRvBwcHT0/P9u3bt23btk2bNm3/rUWLFvUXoLkJCgoSRvpiuCnbtm3b6NGjm/xpVqm4uDg9PT0tLe327du3bt26c+dOampqWlra9evX09PT79+///777/ft21fumP/7ehmrt/qetWTXoUMHpVLp5ubWqlWrtm3burq6uru7S/+VNqpUKrkzNlPNZ+ZRKBTTpk0LDg6WO0gD0el0t27dSk5Ovnr1anJyclJS0p9//nn16tW8vDzDPu+8886wYcNkDFkJFo4GAKCBUVcAAKDe+fv779u3r02bNl5eXtzgCDAuCwsLd3d3d3f3CluLioq4+qcxOn/+PJUDoCGZmZm1bt26devWAwYMKL39/v37UqXh6tWr/A4DAAAMqCsAAFDv/P395Y4ANFOW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TUmhUISGhu7atSs9PV3uLB1Ffn7+lClTFixY8N577x06dMjFxUXuRAAAAEB7xnOQAAAAgCam1Wr79+//3HPP/c///I/cWdq/b7/99q9//auxsfHWrVuHDRsmdxwAAACg/eN+BQAAAKCJderUKTQ0dNu2bZcvX5Y7S3tWXl6+ZMmS8ePHDxs2LC0tjaYCAAAA0DK4XwEAAABoelVVVb179x41atSmTZvkztI+nT9//tVXX71y5UpMTMzs2bPljgMAAAB0INyvAAAAADQ9AwOD995778svv7x27ZrcWdqb6i2aTUxMTp48SVMBAAAAaGH0FQAAAIBm4e/v/+STT/73f/+33EHalVu3bk2dOjUgIOCdd945ePCgq6ur3IkAAACADoe+AgAAANAsjIyMQkJCPv/889zcXLmztBPfffedh4fHmTNnfvzxxw8++MDIyEjuRAAAAEBHRF8BAAAAaC5z5syxsbGJioqSO0ibd//+/SVLlowbN27o0KGnTp0aPny43IkAAACAjou+AgAAANBclEplcHBwbGzsrVu35M7Shl24cGHIkCGxsbEbN27cuXOntbW13IkAAACADo2+AgAAANCMAgICVCrVxx9/LHeQtmrLli2DBw82NjY+efLk22+/LXccAAAAAPQVAAAAgOZkZmYWGBgYHR1dWFgod5Y25vbt21OnTn3jjTcWLlx46NAhNzc3uRMBAAAAEIK+AgAAANDcAgMDFQrFJ598IneQtuT7778fMGBAWlraDz/8wBbNAAAAQKtCXwEAAABoXhYWFgsXLoyKiiopKZE7Sxug0WjCw8PHjh3r5eWVlpY2YsQIuRMBAAAA+A/0FQAAAIBmFxwcXFFR8dlnn8kdpLW7ePHi888/HxkZuXHjxsTERLZoBgAAAFoh+goAAABAs+vSpcvcuXP//ve/q9VqubO0Xlu2bBk0aJChoeFvv/3GFs0AAABAq0VfAQAAAGgJ7777bnFx8ebNm+UO0hrdvn3b29v79ddff/PNNw8dOtSjRw+5EwEAAAB4KIUkSXJnAAAAADqEwMDApKSkjIwMY2NjubO0Ij/88MNrr73WqVOnr776auTIkXLHAQAAAPAI3K8AAAAAtJC//e1veXl5W7ZskTtIa6HbovnFF18cMmRIWloaTQUAAACgTeB+BQAAAKDlvP32299///2lS5cMDQ3lziKzS5cu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/frOnTv1Wdl2/frrr0KIdn+Zbcv169ednJzkTgEAAIDmRV8BAACg4YqLi4UQVlZWtcYLCwvFvz8Tr2ZlZVVSUtLgmkKIffv2ffTRR+np6cXFxXV+PC2EuHfvnhBi+fLly5cvrx50cHB45HmbtXiDPf3000ql8vfff69/GW/4g44cOeLr66vPyraug1xmG+Lj4yN3BAAAADQv9lcAAABouCeffFIIcfv27Vrjug+pa32oXVhYqM/XeB9WM+0fv7wAACAASURBVDs7e9q0aV27dj169GhRUdH69evrfLmtra0QYsOGDTXvUdV9rbsezVq8MUxMTMaNG3f79u3Dhw8/OFtQUPDWW28J3vC6dIRn0fAcpFaIpgIAAEBHQF8BAACg4Z5++mkbG5t//etftcb79etnbm5ec3/do0ePVlRUPPfccw2uefbsWY1GExAQ4OLiolQqFQpFnS9/6qmnlEplWlraY11IsxZvpPDwcBMTk8WLF6vV6lpT586dMzQ0FLzhAAAAANCC6CsAAAA0nImJydKlS3/++efAwMAbN25otdqSkpLz588rlcqQkJCvv/5669atxcXFZ8+enT9/voODw9y5cxtc09nZWQjx3XfflZeXX758uebOATY2Nrm5uVlZWSUlJQYGBq+//vqOHTtiY2OLi4urqqquX79+8+bN+k/arMUfKSUlxcLCIiIios5ZDw+Pbdu2nTt3bvjw4fv37y8qKtJoNFevXv3888/ffPNN3c4EvOEAAAAA0HLkvk0WAACg9RJCJCQkPHLZJ5984u7urlQqlUrlwIEDY2JiJEnSarUfffRRjx49jIyMrK2tp02bdunSJd36mJgYlUolhOjRo0dmZuamTZssLCyEEN26dfv999/rqRkaGmpjY2NlZTV9+vRPPvlECOHq6pqdnX3y5Mlu3bqZmpoOGzbsjz/+uH//fmhoqLOzs6Ghoa2t7SuvvJKenv7Ikzam+CPfol9//XXo0KHVGwN07drVy8vrp59+0s3u37+/c+fOa9asqadCdnb2u+++6+7ubm5ubmBgYGVlNXDgwDfffPPw4cO6BbzhNfn4+PAcJMiig/zsAQAAdHAKSZJaupUBAADQRigUioSEhBkzZsgdBHg806dPF0IkJibKHaR57dy509fXl79oWpUO8rMHAADQwfEcJAAAAAAAAAAAoC/6CgAAAGiUixcv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\n",
            "text/plain": [
              "<IPython.core.display.Image object>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 91
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VXrPaqXi1Xg_"
      },
      "source": [
        "fit_plot_save(neural_mf_mlp_model, input=[train.customer_id, train.product_id], path='neural_mf_mlp_model.h5')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "FXOYmuTeQ9J5"
      },
      "source": [
        ""
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "QtZRgQOpn5SK"
      },
      "source": [
        "### Meilleur Loss: 1.4707\n",
        "### Epochs: 100\n",
        "### Batch Size: 10000\n",
        "### Temps de calcul: 12min"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "XS5CvERDpe5n"
      },
      "source": [
        "### 3 - Evaluation des performances du model sur des données test "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "npBXoT65pwY2"
      },
      "source": [
        "mf_model.evaluate([test.customer_id, test.product_id], test.star_rating)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BqN2cEpB7Y74"
      },
      "source": [
        "neural_mf_model.evaluate([test.customer_id, test.product_id], test.star_rating)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "AgoMSsmJ7Z_d"
      },
      "source": [
        "neural_mf_mlp_model.evaluate([test.customer_id, test.product_id], test.star_rating)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "e24GHlnwp1hf"
      },
      "source": [
        "### Meilleur Loss Test: 1.7"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "t958xXcupwY6"
      },
      "source": [
        "predictions = model.predict([test.customer_id.head(10), test.product_id.head(10)])\n",
        "\n",
        "[print(predictions[i], test.star_rating.iloc[i]) for i in range(0,10)]"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3_urcVZFVLLd"
      },
      "source": [
        "test_norm.head(3)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Bdm764jlTYqP"
      },
      "source": [
        "train.head(3)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2u5CgA6gP9wv"
      },
      "source": [
        "c = 49988212\t\n",
        "p = 36194\n",
        "model.predict([np.array([c]), np.array([p])])"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "e0qd1f5xQI3-"
      },
      "source": [
        "c = 22536317\n",
        "p = 109375\n",
        "model.predict([np.array([c]), np.array([p])])"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Ln0OKXv6pNnI"
      },
      "source": [
        "### 4 - Visualisation des facteurs latents dans un espace à 2 dimensions"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "iaDsnJHto0vI"
      },
      "source": [
        "# Extract embeddings\n",
        "product_em = model.get_layer('Product-Embedding')\n",
        "product_em_weights = product_em.get_weights()[0]"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "4Vvlfmd_o0va",
        "outputId": "41596cf8-2cbd-4766-e9dd-69916224ab33"
      },
      "source": [
        "product_em_weights[:5]"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[-0.00610029,  0.0156836 ,  0.03038192,  0.0425691 , -0.02896588],\n",
              "       [-1.5306774 ,  0.5250458 ,  2.366743  ,  1.3422308 , -1.0649071 ],\n",
              "       [-1.2195548 ,  0.41940942,  2.508028  ,  1.8110565 , -1.1244056 ],\n",
              "       [-0.5183603 ,  1.0415146 ,  2.5360963 ,  0.71309656, -1.3203545 ],\n",
              "       [-1.3798604 , -0.9113507 ,  1.9190255 ,  1.5787991 ,  0.7250671 ]],\n",
              "      dtype=float32)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 19
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "IZPIvv2Fo0vq",
        "outputId": "8d7a7b09-3002-45a9-ccfe-3cfe0d3ff0f4"
      },
      "source": [
        "from sklearn.decomposition import PCA\n",
        "import seaborn as sns\n",
        "\n",
        "pca = PCA(n_components=2)\n",
        "pca_result = pca.fit_transform(product_em_weights)\n",
        "sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x211935af9e8>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 20
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "WXY4VRgGo0x7",
        "outputId": "5c51f7ad-32fc-4c72-cd9c-558e96883abe"
      },
      "source": [
        "product_em_weights = product_em_weights / np.linalg.norm(product_em_weights, axis = 1).reshape((-1, 1))\n",
        "product_em_weights[0][:10]\n",
        "np.sum(np.square(product_em_weights[0]))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1.0"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 21
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "4KWQIs87o0yZ",
        "outputId": "e5eeac95-be01-4a9c-8c7d-b0b4f598fad7"
      },
      "source": [
        "pca = PCA(n_components=2)\n",
        "pca_result = pca.fit_transform(product_em_weights)\n",
        "sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x21193665f60>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 22
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "TpMDrURgo0zc",
        "outputId": "91a6e971-cad0-467c-85d1-ae3ac48e764c"
      },
      "source": [
        "from sklearn.manifold import TSNE\n",
        "\n",
        "tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)\n",
        "tnse_results = tsne.fit_transform(product_em_weights)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[t-SNE] Computing 121 nearest neighbors...\n",
            "[t-SNE] Indexed 10001 samples in 0.007s...\n",
            "[t-SNE] Computed neighbors for 10001 samples in 0.840s...\n",
            "[t-SNE] Computed conditional probabilities for sample 1000 / 10001\n",
            "[t-SNE] Computed conditional probabilities for sample 2000 / 10001\n",
            "[t-SNE] Computed conditional probabilities for sample 3000 / 10001\n",
            "[t-SNE] Computed conditional probabilities for sample 4000 / 10001\n",
            "[t-SNE] Computed conditional probabilities for sample 5000 / 10001\n",
            "[t-SNE] Computed conditional probabilities for sample 6000 / 10001\n",
            "[t-SNE] Computed conditional probabilities for sample 7000 / 10001\n",
            "[t-SNE] Computed conditional probabilities for sample 8000 / 10001\n",
            "[t-SNE] Computed conditional probabilities for sample 9000 / 10001\n",
            "[t-SNE] Computed conditional probabilities for sample 10000 / 10001\n",
            "[t-SNE] Computed conditional probabilities for sample 10001 / 10001\n",
            "[t-SNE] Mean sigma: 0.018763\n",
            "[t-SNE] KL divergence after 250 iterations with early exaggeration: 85.958824\n",
            "[t-SNE] KL divergence after 300 iterations: 2.987510\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "T-v35b8Do0zu",
        "outputId": "6922f0bd-34fc-4681-d5f3-9c3352d8167c"
      },
      "source": [
        "sns.scatterplot(x=tnse_results[:,0], y=tnse_results[:,1])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x21193aefac8>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 24
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    }
  ]
}