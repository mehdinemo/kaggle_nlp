import codecs
from datetime import datetime
from time import sleep
from tqdm._utils import _term_move_up
import pyodbc
import numpy as np
import pandas as pd


# region HistoryManager
class HistoryManager:
    def __init__(self, connection_string: str, table_name: str, client_id: str):
        self.connection_string = connection_string
        self.table_name = table_name
        self.client_id = client_id

    def _select(self, query: str, connection_string: str) -> pd.DataFrame:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        response = cursor.execute(query)
        tbl = np.array(response.fetchall())
        try:
            df = pd.DataFrame(tbl, columns=[column[0] for column in response.description])
        except:
            df = None
        return df

    def _exec_query(self, query: str, connection_string: str):
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        cursor.execute(query)
        cursor.commit()

    def _delete_history(self, history_id: int, connection_string: str):
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        query = "DELETE FROM History WHERE Id = {0}".format(history_id)
        cursor.execute(query)
        cursor.commit()

    def get_last_activity(self, delete_init):
        date_format = '%Y-%m-%d %H:%M:%S.000'

        # check if init exists
        query = "SELECT Top(1) * FROM {0} WHERE Client = '{1}' AND message = 'Init' ORDER BY eventdate desc"
        query = query.format(self.table_name, self.client_id)
        res = self._select(query, self.connection_string)
        if res is not None:
            if delete_init:
                # delete init history
                self._delete_history(res['Id'][0], self.connection_string)
            return (res['LastActivity'][0]).strftime(date_format)

        # get latest activity
        query = "SELECT Top(1) * FROM {0} WHERE Client = '{1}' ORDER BY eventdate desc"
        query = query.format(self.table_name, self.client_id)
        res = self._select(query, self.connection_string)
        if res is not None:
            return (res['LastActivity'][0]).strftime(date_format)

        # return now datetime by default
        return (datetime.now()).strftime(date_format)

    def log_history(self, message: str,
                    last_activity_table_name: str,
                    event_date: datetime,
                    last_activity_date: datetime,
                    start_activity: datetime,
                    parameter: str = 0):
        query = """
                Insert into History (client,message,LastActivityTable,EventDate,LastActivity,StartActivity,Parameter) 
                values 
                ('{0}','{1}','{2}','{3}','{4}','{5}','{6}')
                """
        query = query.format(self.client_id, message, last_activity_table_name, event_date, last_activity_date,
                             start_activity, parameter)
        self._exec_query(query, self.connection_string)


# endregion


class LogManager:
    hm: HistoryManager

    def __init__(self, file_path,
                 history_manager_connection_string: str = None,
                 history_manager_table_name: str = None,
                 history_manager_client_id: str = None):
        self.fp = codecs.open(file_path, 'a', encoding="utf-8")
        self.log_text_template = "{0}\t{1}\t{2}\n"

        if (history_manager_connection_string is not None) & \
                (history_manager_table_name is not None) & \
                (history_manager_client_id is not None):
            self.hm = HistoryManager(history_manager_connection_string,
                                     history_manager_table_name,
                                     history_manager_client_id)
        else:
            self.hm = None

    def error(self, msg):
        log_text = self.log_text_template.format(datetime.now(), "Error", msg)
        self.fp.write(log_text)
        self.fp.flush()

        _term_move_up()
        print(log_text.strip() + "\r")

    def info(self, msg, log_to_file=False):
        log_text = self.log_text_template.format(datetime.now(), "Info", msg)
        if log_to_file:
            self.fp.write(log_text)
            self.fp.flush()

        _term_move_up()
        print(log_text.strip() + "\r")

    def get_latest_activity(self, delete_init=True):
        if self.hm is None:
            raise Exception("Not Enough Parameters Passed To Constructor In Order To Build History Manager!")
        else:
            return self.hm.get_last_activity(delete_init)

    def log_history(self, message: str,
                    last_activity_table_name: str,
                    event_date: datetime,
                    last_activity_date: datetime,
                    start_activity: datetime,
                    parameter: str = 0):
        if self.hm is None:
            raise Exception("Not Enough Parameters Passed To Constructor In Order To Build History Manager!")
        else:
            self.hm.log_history(message, last_activity_table_name, event_date, last_activity_date, start_activity,
                                parameter)
