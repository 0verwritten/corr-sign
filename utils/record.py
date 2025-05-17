import os
import time

class Recorder:
    def __init__(self, work_dir, print_log=True, log_interval=10):
        self.start_time = time.time()
        self.print_log_flag = print_log
        self.log_interval = log_interval
        self.log_path = os.path.join(work_dir, 'log.txt')
        self.timer = {
            'dataloader': 0.001,
            'device': 0.001,
            'forward': 0.001,
            'backward': 0.001
        }

    def log(self, message, path=None, include_time=True):
        if path is None:
            path = self.log_path

        if include_time:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            message = f"[{timestamp}] {message}"

        print(message)

        if self.print_log_flag:
            try:
                with open(path, 'a', encoding='utf-8') as f:
                    f.write(message + '\n')
            except IOError as e:
                print(f"Failed to write to log file {path}: {e}")
