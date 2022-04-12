from abc import ABC, abstractmethod
from argparse import ArgumentParser
from logging import Logger

import sys
from pyspark.sql import SparkSession


class Job(ABC):
    def __init__(self, spark=None, init_conf_path=None):
        self.spark = self._prepare_spark(spark)
        self.logger = self._prepare_logger()
        self.dbutils = self.get_dbutils()
        if init_conf_path:
            self.config_path = init_conf_path
            print("Using provided configuration file: ", self.config_path)
        else:
            self.config_path = self.get_conf_file()
            print("Using configuration file: ", self.config_path)

    @staticmethod
    def _prepare_spark(spark) -> SparkSession:
        if not spark:
            return SparkSession.builder.getOrCreate()
        else:
            return spark

    @staticmethod
    def _get_dbutils(spark: SparkSession):
        try:
            from pyspark.dbutils import DBUtils  # noqa

            if "dbutils" not in locals():
                utils = DBUtils(spark)
                return utils
            else:
                return locals().get("dbutils")
        except ImportError:
            return None

    def get_dbutils(self):
        utils = self._get_dbutils(self.spark)

        if not utils:
            self.logger.warn("No DBUtils defined in the runtime")
        else:
            self.logger.info("DBUtils class initialized")

        return utils

    @staticmethod
    def get_conf_file():
        p = ArgumentParser()
        p.add_argument("--conf-file", required=False, type=str)
        namespace = p.parse_known_args(sys.argv[1:])[0]
        return namespace.conf_file

    def _prepare_logger(self) -> Logger:
        log4j_logger = self.spark._jvm.org.apache.log4j  # noqa
        return log4j_logger.LogManager.getLogger(self.__class__.__name__)

    @abstractmethod
    def launch(self):
        """
        Main method of the job.

        :return:
        """
        pass
