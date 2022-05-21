import importlib
from .basic import BasicModule, BasicDataModule


def find_module_using_name(module_name):
    module_filename = "modules." + module_name
    modulelib = importlib.import_module(module_filename)
    module = None
    data_module = None
    target_module_name = module_name.replace('_', '') + 'Module'
    target_data_module_name = module_name.replace('_', '') + 'DataModule'
    for name, class_obj in modulelib.__dict__.items():
        if name.lower() == target_module_name.lower() and \
           issubclass(class_obj, BasicModule):
            module = class_obj
        if name.lower() == target_data_module_name.lower() and \
           issubclass(class_obj, BasicDataModule):
            data_module = class_obj

    if module is None or data_module is None:
        print(
            f"In {module_filename}.py, "
            f"there should be "
            f"a subclass of BasicModule with class name that matches {target_module_name} in lowercase and "
            f"a subclass of BasicDataModule with class name that matches {target_data_module_name} in lowercase"
        )
        exit(0)

    return module, data_module
