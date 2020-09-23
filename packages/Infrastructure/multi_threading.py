import os
from threading import Thread
from multiprocessing import Process
from queue import Queue
import time
import threading
from concurrent.futures import ProcessPoolExecutor


class ParallelManager:
    def __init__(self, number_of_cores=os.cpu_count()):
        # اگر تعداد core های وارد شده از تعداد کل core های سیستم بیشتر باشد
        if int(number_of_cores) > os.cpu_count():
            raise Exception("Number of cores cannot be more than OS logical cores. Your maximum cores can be: {0}"
                            .format(os.cpu_count()))
        self.number_of_cores = int(number_of_cores)
        self.result = Queue(maxsize=0)

    def ParallelLoop(self, items, apply_function):

        # اگر تعداد item ها از تعداد core ها کمتر باشد
        cores_count = 0
        if len(items) <= self.number_of_cores:
            cores_count = len(items)
        else:
            cores_count = self.number_of_cores

        # به اندازه core های وارد شده cpu دسته درست کرده از فرآیندها و آنها را دسته دسته اجرا میکند
        items_per_list = int(len(items) / cores_count)
        sub_lists = []
        for i in range(cores_count):
            if i == (cores_count - 1):
                sub_lists.append((items[items_per_list * i:], apply_function))
                continue
            sub_lists.append((items[items_per_list * i:items_per_list * (i + 1)], apply_function))

        with ProcessPoolExecutor(max_workers=cores_count) as executor:
            output_generator = executor.map(self._loop_through_list, sub_lists)

        result = list(output_generator)
        final_result = []
        for r in result:
            final_result.extend(r)
        return final_result

    # هر thread این تابع را برای یک زیر لیست فراخوانی میکند
    @staticmethod
    def _loop_through_list(args):
        items = args[0]
        apply_function = args[1]
        result = []
        for item in items:
            output = apply_function(item)
            result.append(output)
        return result


if __name__ == "__main__":

    # تست
    pm = ParallelManager(4)


    def add(a):
        print("{0}-{1}".format(threading.current_thread().name, a))
        res = 0
        for i in range(a):
            res += a
        return res


    inputs1 = [i for i in range(100052) if i % 2 == 1]
    result = pm.ParallelLoop(inputs1, add)
    print(result)
