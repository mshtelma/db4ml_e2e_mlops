from db4ml import execute
from telco_churn.common import Job


class SampleJob(Job):
    def launch(self):
        self.logger.info("Launching sample job")

        execute("telco_churn_train", self.spark, self.config_path)
        execute("telco_churn_scoring_pandas", self.spark, self.config_path)
        execute("telco_churn_scoring_spark", self.spark, self.config_path)

        self.logger.info("Sample job finished!")


if __name__ == "__main__":
    job = SampleJob()
    job.launch()
