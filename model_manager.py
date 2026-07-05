from fastai.vision.all import *
import pathlib
from pathlib import Path
import os
import sys
import __main__

from legacy_fastcore import dispatch as legacy_dispatch
from legacy_fastcore import transform as legacy_transform


def _label_func(f):
    pass


def install_legacy_fastai_model_shim():
    sys.modules['fastcore.dispatch'] = legacy_dispatch
    sys.modules['fastcore.transform'] = legacy_transform
    if not hasattr(__main__, 'label_func'):
        __main__.label_func = _label_func

class ModelManager:
    def __init__(self, models_folder: str) -> None:
        self.models_folder = Path(__file__).resolve().with_name(models_folder)
        self.loaded_models = self.load_models()

    def load_models(self):
        models = []
        for file in self.models_folder.glob('*'):
            parts = str(file.name).split('_')
            posix_backup = pathlib.PosixPath
            try:
                if(os.name == 'nt'):
                    pathlib.PosixPath = pathlib.WindowsPath
                install_legacy_fastai_model_shim()
                model = load_learner(file)
            finally:
                pathlib.PosixPath = posix_backup
            priority = parts[0]
            target = parts[1]
            task = parts[2]
            version = parts[3]
            models.append({
                'priority': priority,
                'target': target,
                'task': task,
                'version': version,
                'model': model
            })
        return models
        
    def get_loaded_models_qt(self) -> int:
        return len(self.loaded_models)
    
    def get_models_info(self) -> str:
        info = []
        for model in self.loaded_models:
            model_info = f"\n{'*'*12}\nModel info:\ntarget: {model['target']}\ntask: {model['task']}\nversion: {model['version']}\npriority: {model['priority']}\n{'*'*12}"
            info.append(model_info)
        return ('').join(info)
