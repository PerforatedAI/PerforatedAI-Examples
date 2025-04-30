import torch
class TrainingConfig:
    def __init__(
        self,
        batch_size=32,
        width_mult=0.5,
        learning_rate=0.001,
        model_name="MobileNetV3Small",
        use_pai=False,
        num_epochs=150
    ):
        self.batch_size = batch_size
        self.width_mult = width_mult
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.use_pai = use_pai
        self.num_epochs = num_epochs if not use_pai else 300  # PAI mode uses 300 epochs by default
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pai_config = {
            "switch_mode": "doingHistory",
            "n_epochs_to_switch": 10,
            "cap_at_n": True,
            "p_epochs_to_switch": 10,
            "input_dimensions": [-1, 0, -1, -1],
            "history_lookback": 1,
            "max_dendrites": 5,
            "modules_to_convert": ['Conv2dNormActivation']
        }