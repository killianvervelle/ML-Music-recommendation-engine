package files

import org.apache.spark.sql.{DataFrame, SparkSession}

object Data {
    def readParquetFile(filePath: String): DataFrame = {
        val spark = SparkSession.builder()
          .appName("ParquetFileReader")
          .getOrCreate()

        val parquetData = spark.read.parquet(filePath)
        return parquetData
    }
}