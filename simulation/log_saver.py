import os
import pickle
import shutil
import subprocess
from typing import Dict, List, Optional

import numpy as np
import yaml


class LogSaver:
    def __init__(self, run_name: str, directory: str, gcs_directory: Optional[str], save_every: int = 1) -> None:
        self.save_every = save_every
        self.path = os.path.join(directory, run_name)
        shutil.rmtree(self.path, ignore_errors=True)
        os.makedirs(self.path, exist_ok=True)
        if gcs_directory:
            self.gcs_directory = os.path.join(gcs_directory, run_name)
        else:
            self.gcs_directory = None

    def sync_with_gcs(self) -> None:
        if self.gcs_directory:
            subprocess.call(["gsutil", "-m", "cp", "-r", os.path.join(self.path, "*"), self.gcs_directory])
            # removing everything to make space
            shutil.rmtree(self.path, ignore_errors=True)
            os.makedirs(self.path, exist_ok=True)

    def save_dict(self, dictionary: Dict, file_name: str) -> None:
        file_path = os.path.join(self.path, file_name)
        with open(file_path, 'w') as f:
            yaml.dump(dictionary, f)

    def save_class(self, class_instance: type, file_name: str) -> None:
        file_path = os.path.join(self.path, file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(class_instance, f)

    def save_class_for_step(self, class_instance: type, file_name: str, step: int) -> None:
        if step % self.save_every == 0:
            file_name = file_name + f"_{step}"
            self.save_class(class_instance, file_name)

    def save_list(self, list_: List, file_name: str) -> None:
        file_path = os.path.join(self.path, file_name)
        array = np.array(list_)
        np.savetxt(file_path, array)
