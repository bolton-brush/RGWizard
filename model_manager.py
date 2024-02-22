from fastai.vision.all import *
from pathlib import Path

class ModelManager:
    def __init__(self, models_folder: str) -> None:
        self.models_folder = Path(__file__).resolve().with_name(models_folder)
        self.loaded_models = self.load_models()

    def load_models(self):
        models = []
        for file in self.models_folder.glob('*'):
            parts = str(file.name).split('_')
            model = load_learner(file)
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