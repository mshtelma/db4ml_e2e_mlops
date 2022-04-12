import os
import shutil
import unittest
from unittest.mock import MagicMock
import mlflow
from pyspark.sql import SparkSession
from telco_churn.entrypoint import SampleJob


class SampleJobUnitTest(unittest.TestCase):
    def setUp(self):
        try:
            shutil.rmtree("spark-warehouse", ignore_errors=True)
            shutil.rmtree("mlruns", ignore_errors=True)
            os.remove("mlruns.db")
        except Exception as e:
            print(e)
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        self.spark = SparkSession.builder.master("local[1]").getOrCreate()

        self.job = SampleJob(
            spark=self.spark, init_conf_path="tests/unit/mlproject.yaml"
        )

    def test_sample(self):
        # feel free to add new methods to this magic mock to mock some particular functionality
        self.job.dbutils = MagicMock()

        self.job.launch()

        output_count = self.spark.read.table("telco_churn_out_pandas").count()

        self.assertGreater(output_count, 0)


if __name__ == "__main__":
    unittest.main()
