import sys, os
from collections import OrderedDict

''' 
The point of this class is to take a loose configuration specification
and return a properly formated larcv IO configuration.  So, it can spit out
a full configuration string, or even a temp file.

'''


class ProcessConfig(object):

    def __init__(self, proc_name, proc_type):
        self._name   = proc_name
        self._type   = proc_type
        self._params = OrderedDict()

    def set_param(self, key, value):
        self._params[key] = value

    def str(self, indent_level):
        # Return the string corresponding to this process's configuration
        output_str = ""
        output_str += "{indent}{name}: {{\n".format(indent=" "*indent_level, name=self._name)
        indent_level += 2
        for param in self._params:
            output_str += "{indent}{param}: {value}\n".format(
                indent = " "*indent_level,
                param  = param,
                value  = self._params[param]
            )
        indent_level -= 2
        output_str += "{indent}}}\n".format(indent=" "*indent_level)
        return output_str

class ProcessListConfig(object):

    def __init__(self):
        self._processes = []


    def str(self, indent_level):
        # For each process, get the string representation and concatenate them
        output_str = ""
        for proc in self._processes:
            output_str += proc.str(indent_level)
        return output_str

    def process_types(self):
        # Return the list of process types, properly formatted
        return "[{}]".format(",".join(["\"{}\"".format(proc._type) for proc in self._processes]))

    def process_names(self):
        # Return the list of process names, properly formated
        return "[{}]".format(",".join(["\"{}\"".format(proc._name) for proc in self._processes]))

    def add_process(self, process):
        self._processes.append(process)

class CoreConfig(object):

    def __init__(self):

        self._process_list = ProcessListConfig()

    def add_process(self, process):
        self._process_list.add_process(process)

class IOManagerConfig(object):


    def __init__(self, name):
        self._name = name
        self._params = OrderedDict()
        self._params["Verbosity"] =  None
        self._params["IOMode"] =  None
        self._params["OutFileName"] =  None
        self._params["InputFiles"] =  None
        self._params["InputDirs"] =  None
        self._params["StoreOnlyName"] =  None
        self._params["StoreOnlyType"] =  None
        self._params["ReadOnlyName"] =  None
        self._params["ReadOnlyType"] =  None

        self._defaults_set = False

    def set_defaults(self):

        if self._params['Verbosity'] is None:
            self._params['Verbosity'] = "0"
        if self._params["IOMode"] is None:
            self._params["IOMode"] = "2"
        # if self._params["OutFileName"] is None:
        #     self._params["OutFileName"] = 
        # if self._params["InputFiles"] is None:
        #     self._params["InputFiles"] = 
        if self._params["InputDirs"] is None:
            self._params["InputDirs"] = "[]"
        if self._params["StoreOnlyName"] is None:
            self._params["StoreOnlyName"] = "[]"
        if self._params["StoreOnlyType"] is None:
            self._params["StoreOnlyType"] = "[]"
        if self._params["ReadOnlyName"] is None:
            self._params["ReadOnlyName"] = "[]"
        if self._params["ReadOnlyType"] is None:
            self._params["ReadOnlyType"] = "[]"


        self._defaults_set = True

    def generate_config_str(self):

        if not self._defaults_set:
            self.set_defaults()

        indent_level = 0

        # Take the config and generate a parsable string:
        output_str = ""
        output_str += self._name + ": {\n"
        indent_level += 2

        for param in self._params:
            if param == 'InputFiles':
                output_str += "{indent}{param}: [\"{value}\"]\n".format(
                    indent = " "*indent_level,
                    param  = param,
                    value  = self._params[param])
            else:
                output_str += "{indent}{param}: {value} \n".format(
                    indent = " "*indent_level,
                    param  = param,
                    value  = self._params[param])


        # output_str += "{indent}}}\n".format(indent=" "*indent_level)
        indent_level -= 2
        output_str += "{indent}}}\n".format(indent=" "*indent_level)

        return output_str
        
    def set_param(self, param, value):
        self._params[param] = value

# class ProcessDriverConfig(CoreConfig):

#     def __init__(self):
#         CoreConfig.__init__(self)
#         self._io_config = IOManagerConfig()

#         self._params = OrderedDict()

#         self._params["ProcessList"] =  None
#         self._params["RandomAccess"] =  None
#         self._params["AnaFile"] =  None
#         self._params["StartEntry"] =  None
#         self._params["NumEntries"] =  None
#         self._params["ProcessType"] =  None
#         self._params["ProcessName"] =  None

#         self._params["IOManager"] =  self._io_config





class ThreadIOConfig(CoreConfig):


    def __init__(self, name):
        CoreConfig.__init__(self)
        self._name = name
        self._params = OrderedDict()

        self._params["Verbosity"] =  None
        self._params["EnableFilter"] =  None
        self._params["NumThreads"] =  None
        self._params["InputFiles"] =  None
        self._params["NumBatchStorage"] =  None
        self._params["RandomSeed"] =  None
        self._params["RandomAccess"] =  None


        self._defaults_set = False

    def set_param(self, param, value):
        self._params[param] = value

    def set_defaults(self):

        if self._params['Verbosity'] is None:
            self._params['Verbosity'] = "2"
        if self._params['EnableFilter'] is None:
            self._params['EnableFilter'] = "false"
        if self._params['NumThreads'] is None:
            self._params['NumThreads'] = "4"
        if self._params['NumBatchStorage'] is None:
            self._params['NumBatchStorage'] = "4"
        if self._params['RandomSeed'] is None:
            self._params['RandomSeed'] = "0"
        if self._params['RandomAccess'] is None:
            self._params['RandomAccess'] = "0"

        self._defaults_set = True

    def generate_config_str(self):

        if not self._defaults_set:
            self.set_defaults()

        indent_level = 0

        # Take the config and generate a parsable string:
        output_str = ""
        output_str += self._name + ": {\n"
        indent_level += 2

        for param in self._params:
            if param == 'InputFiles':
                output_str += "{indent}{param}: [\"{value}\"]\n".format(
                    indent = " "*indent_level,
                    param  = param,
                    value  = self._params[param])
            else:
                output_str += "{indent}{param}: {value} \n".format(
                    indent = " "*indent_level,
                    param  = param,
                    value  = self._params[param])

        # Add the list of process types and names:
        output_str += "{indent}ProcessType: {procs}\n".format(
            indent=" "*indent_level, procs=self._process_list.process_types())
        output_str += "{indent}ProcessName: {procs}\n".format(
            indent=" "*indent_level, procs=self._process_list.process_names())

        # Now, get the processes:
        output_str += "\n"
        output_str += "{indent}ProcessList: {{\n".format(indent=" "*indent_level)
        indent_level += 2
        output_str += self._process_list.str(indent_level=indent_level)
        indent_level -= 2

        output_str += "{indent}}}\n".format(indent=" "*indent_level)
        indent_level -= 2
        output_str += "{indent}}}\n".format(indent=" "*indent_level)


        return output_str







