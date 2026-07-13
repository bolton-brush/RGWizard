import logging
import os
import pathlib
import sys
from pathlib import Path

from fastai.vision.all import load_learner

class ModelManager:
    def __init__(self, models_folder: str) -> None:
        self.models_folder = Path(__file__).resolve().with_name(models_folder)
        self.loaded_models = self.load_models()

    def load_models(self):
        models = []
        if not self.models_folder.exists():
            logging.warning(f"Models folder not found: {self.models_folder}")
            return models

        for file in self.models_folder.glob('*'):
            if not file.is_file():
                continue
            parts = file.stem.split('_')
            if len(parts) < 4:
                logging.warning(f"Skipping model with unexpected filename: {file.name}")
                continue
            posix_backup = pathlib.PosixPath
            try:
                if(os.name == 'nt'):
                    pathlib.PosixPath = pathlib.WindowsPath
                model = load_learner(file, cpu=True)
            finally:
                pathlib.PosixPath = posix_backup
            try:
                priority = int(parts[0])
            except ValueError:
                priority = sys.maxsize
                logging.warning(f"Model {file.name} has a non-numeric priority")
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
        return sorted(models, key=lambda m: (m['priority'], m['target'], m['task'], m['version']))

    def get_model(self, target=None, task=None):
        for model in self.loaded_models:
            if target is not None and model['target'] != target:
                continue
            if task is not None and model['task'] != task:
                continue
            return model
        return None
        
    def get_loaded_models_qt(self) -> int:
        return len(self.loaded_models)
    
    def get_models_info(self) -> str:
        info = []
        for model in self.loaded_models:
            model_info = f"\n{'*'*12}\nModel info:\ntarget: {model['target']}\ntask: {model['task']}\nversion: {model['version']}\npriority: {model['priority']}\n{'*'*12}"
            info.append(model_info)
        return ('').join(info)
