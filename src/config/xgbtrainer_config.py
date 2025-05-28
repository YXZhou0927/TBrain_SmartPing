import json
from pathlib import Path


class XGBTrainerConfig:
    def __init__(self):
        self.root_path = Path(__file__).resolve().parent.parent.parent  # Assuming the config file is located at src/config/xgbtrainer_config.json
        self.config_path = Path(__file__).resolve().parent / "xgbtrainer_config.json"
        self._load_config()

    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.parameter_space = config.get("parameter_space", {})
        self.hyperparameter_tuning = config.get("hyperparameter_tuning", {})
        self.path_info = config.get("path_info", {})
        self._comment = config.get("_comment", "")

    def get_parameter_space(self) -> dict:
        return self.parameter_space

    def get_tuning_method(self) -> str:
        return self.hyperparameter_tuning.get("method", "optuna")

    def get_cv_folds(self) -> dict:
        return self.hyperparameter_tuning.get("cv_folds", {})

    def get_n_trials(self) -> int:
        return self.hyperparameter_tuning.get("n_trials", 10)

    def get_scoring(self) -> str:
        return self.hyperparameter_tuning.get("scoring", "auc")

    def get_random_state(self) -> int:
        return self.hyperparameter_tuning.get("random_state", 42)

    def get_model_dir(self) -> Path:
        return self.root_path / self.path_info.get("model_dir", "models")

    def get_output_dir(self) -> Path:
        return self.root_path / self.path_info.get("output_dir", "output")


if __name__ == "__main__":
    # Example usage
    config = XGBTrainerConfig()
    print("Loaded parameter space:", config.get_parameter_space())
    print("Tuning method:", config.get_tuning_method())
    print("CV folds:", config.get_cv_folds())
    print("Model directory:", config.get_model_dir())
