import unittest
from pyspark.dbutils import DBUtils  # noqa
from telco_churn.entrypoint import SampleJob


class SampleJobIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.job = SampleJob()
        self.dbutils = DBUtils(self.job.spark)
        self.spark = self.job.spark

    def test_sample(self):
        self.job.launch()


if __name__ == "__main__":
    # please don't change the logic of test result checks here
    # it's intentionally done in this way to comply with jobs run result checks
    # for other tests, please simply replace the SampleJobIntegrationTest with your custom class name
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(SampleJobIntegrationTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(tests)
    if not result.wasSuccessful():
        raise RuntimeError(
            "One or multiple tests failed. Please check job logs for additional information."
        )
