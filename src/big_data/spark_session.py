"""
PySpark Session Configuration and Lifecycle Management
Configured for high-throughput distributed analytical ETL and feature engineering.

Author: Ankith Umadi, Haneesh E, Vikas R (PES University BDSRAE Team)
"""

import os
import sys
from pyspark.sql import SparkSession


def get_spark_session(app_name="BDSRAE-Healthcare-Subgroup-Pipeline", master="local[*]", driver_memory="4g", shuffle_partitions=8):
    """
    Initializes and returns an optimized SparkSession.
    """
    # Ensure Spark worker uses the exact Python interpreter on Windows
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
