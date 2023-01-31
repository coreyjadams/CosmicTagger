import sys, os

from datetime import datetime

NOTSET   = 0
DEBUG    = 10
INFO     = 20
WARNING  = 30
ERROR    = 40
CRITICAL = 50


global_logger_dict = {}

def getLogger(logger_name):
    '''
    Factory function for loggers
    '''

    if logger_name in global_logger_dict:
        return global_logger_dict[logger_name]
    else:
        l = logger(name = logger_name)
        global_logger_dict[logger_name] = l
        return global_logger_dict[logger_name]

class logger():

    severity_map = {
        "NOTSET"  : NOTSET,    "notset"    : NOTSET,
        "DEBUG"   : DEBUG,     "debug"     : DEBUG,
        "INFO"    : INFO,      "info"      : INFO,
        "WARNING" : WARNING,   "warning"   : WARNING,
        "ERROR"   : ERROR,     "error"     : ERROR,
        "CRITICAL": CRITICAL,  "critical"  : CRITICAL,
    }

    severity_int_to_str = {
        CRITICAL  : "CRITICAL",
        DEBUG     : "DEBUG",
        INFO      : "INFO",
        WARNING   : "WARNING",
        ERROR     : "ERROR",

    }

    def __init__(self, name : str = "", level : int = NOTSET, file_path = None):

        self.name = name
        self.format_template = "{date} - {severity} - {msg}"
        self.setLevel(level)

        self.setFile(file_path)
        return

    def setFile(self, file_path):
        if file_path is not None:
            self.file = open(file_path, 'w')

    def __del__(self):
        if hasattr(self, "file"):
            self.file.close()

    def setLevel(self, level):
        target_level = self.validate_level(level)
        if isinstance(level, int):
            self.level = level
        return

    def getLevel(self):
        return self.level

    def validate_level(self, level):
        if isinstance(level, str):
            if level not in self.severity_map.keys():
                raise Exception(f"Unknown severity {level}")
            # Map it to int:
            level = self.severity_map[level]
            return level
        elif isinstance(level, int): return level
        else:
            raise Exception(f"Could not validate level {level} of type {type(level)}")

    def log(self, level, message):
        str_level = str(level)
        if isinstance(level, int):
            if level in self.severity_int_to_str.keys():
                str_level = self.severity_int_to_str[level]

        self.print(
            msg = self.format_template.format(
                date     = datetime.now(),
                severity = str_level,
                msg      = message,
            ),
            level = level,
        )

    def print(self, msg, level : int):
        if level >= self.level:
            print(msg, flush=True)
            if hasattr(self, "file"):
                self.file.write(msg)

    def info(self, message):
        self.print(
            msg = self.format_template.format(
                date     = datetime.now(),
                severity = "INFO",
                msg      = message,
            ),
            level = INFO,
        )

    def debug(self, message):
        self.print(
            msg = self.format_template.format(
                date     = datetime.now(),
                severity = "DEBUG",
                msg      = message,
            ),
            level = INFO,
        )

    def warning(self, message):
        self.print(
            msg = self.format_template.format(
                date     = datetime.now(),
                severity = "WARNING",
                msg      = message,
            ),
            level = WARNING,
        )

    def error(self, message):
        self.print(
            msg = self.format_template.format(
                date     = datetime.now(),
                severity = "ERROR",
                msg      = message,
            ),
            level = ERROR,
        )

    def critical(self, message):
        self.print(
            msg = self.format_template.format(
                date     = datetime.now(),
                severity = "CRITICAL",
                msg      = message,
            ),
            level = CRITICAL,
        )
