import time
import os
import csv
import uuid
from typing import Union, Tuple
from datetime import date


class Logger:
    def __init__(self):
        self.pipelinepath: str = os.path.dirname(os.path.abspath(__file__))
        self.logdir = os.path.abspath(os.path.join(self.pipelinepath, '..', 'logs'))
        self.make_directory()


    def make_directory(self):
        if not os.path.exists(self.logdir):
            os.mkdir(self.logdir)
        return

    def update_train_log(self, data_shape: Tuple[int, int], eval_test: float, runtime: str, country: str, MODEL_VERSION: str, MODEL_VERSION_NOTE: str, test: bool = False):
        ## name the logfile using something that cycles with date (day, month, year)
        today = date.today()
        if test:
            logfile = os.path.join(self.logdir,"train-test.log")
        else:
            logfile = os.path.join(self.logdir,"train-{}-{}.log".format(today.year, today.month))

        ## write the data to a csv file
        header = ['unique_id','timestamp','x_shape','eval_test','country','model_version',
                  'model_version_note','runtime']
        write_header = False
        if not os.path.exists(logfile):
            write_header = True
        with open(logfile,'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            if write_header:
                writer.writerow(header)

            to_write = map(str,[uuid.uuid4(),time.time(),data_shape,eval_test,country,
                                MODEL_VERSION,MODEL_VERSION_NOTE,runtime])
            writer.writerow(to_write)
        return


    def update_predict_log(self, y_pred: float, y_proba: Union[str, float], query, runtime: str, country: str, MODEL_VERSION: str, test: bool = False):
        """
        update predict log file
        """

        ## name the logfile using something that cycles with date (day, month, year)
        today = date.today()
        if test:
            logfile = os.path.join(self.logdir, "predict-test.log")
        else:
            logfile = os.path.join(self.logdir, "predict-{}-{}.log".format(today.year, today.month))

        ## write the data to a csv file
        header = ['unique_id', 'timestamp', 'y_pred', 'y_proba', 'query', 'country', 'model_version', 'runtime']
        write_header = False
        if not os.path.exists(logfile):
            write_header = True
        with open(logfile, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            if write_header:
                writer.writerow(header)

            to_write = map(str, [uuid.uuid4(), time.time(), y_pred, y_proba, query, country,
                                 MODEL_VERSION, runtime])
            writer.writerow(to_write)
