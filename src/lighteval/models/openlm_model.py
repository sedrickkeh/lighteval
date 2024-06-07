from contextlib import nullcontext

import yaml
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from lighteval.logging.hierarchical_logger import hlog
from lighteval.models.base_model import BaseModel
from lighteval.models.model_config import OpenLMModelConfig, EnvConfig
from lighteval.models.utils import _get_dtype


class OpenLMModel(BaseModel):
    def _create_auto_model(
        self,
        config: OpenLMModelConfig,
        env_config: EnvConfig,
    ) -> AutoModelForCausalLM:
        self._config = config

        try:
            from open_lm.model import create_params  # noqa: F811
            from open_lm.utils.transformers.hf_config import OpenLMConfig  # noqa: F811
            from open_lm.utils.transformers.hf_model import OpenLMforCausalLM  # noqa: F811
            from open_lm.main import load_model # noqa: F811
            from open_lm.params import add_model_args, add_training_args  # noqa: F811
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'open_lm' LM type, but package `open_lm` is not installed." \
                "please install open_lm from `https://github.com/TRI-ML/open_lm`",
            )
        print("config: ", config)
        print("env_config: ", env_config)

        openlm_config_dict = self._create_config_dict(config.openlm_model_type, config.openlm_model_config)
        self.config = OpenLMConfig(create_params(openlm_config_dict))
        self.model = OpenLMforCausalLM(self.config).model
        print("model: ", self.model)

        openlm_config_dict.resume = config.openlm_model_path
        openlm_config_dict.distributed = False
        openlm_config_dict.load_not_strict = True
        load_model(openlm_config_dict, self.model)
        return self.model


    def _create_config_dict(self, pretrained: str, config_file: str) -> None:
        try:
            from open_lm.model import create_params  # noqa: F811
            from open_lm.utils.transformers.hf_config import OpenLMConfig  # noqa: F811
            from open_lm.utils.transformers.hf_model import OpenLMforCausalLM  # noqa: F811
            from open_lm.main import load_model # noqa: F811
            from open_lm.params import add_model_args, add_training_args  # noqa: F811
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'open_lm' LM type, but package `open_lm` is not installed." \
                "please install open_lm from `https://github.com/TRI-ML/open_lm`",
            )

        parser = argparse.ArgumentParser()
        add_training_args(parser)
        add_model_args(parser)

        config = parser.parse_args([])
        config.model = pretrained

        if config_file is not None:
            with open(config_file, "r") as f:
                config_to_override = yaml.safe_load(f)
            for k, v in config_to_override.items():
                if v == "None":
                    v = None

                # we changed args
                if k == "batch_size":
                    k = "per_gpu_batch_size"
                if k == "val_batch_size":
                    k = "per_gpu_val_batch_size"
                setattr(config, k, v)
        return config


    def _model_call(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)[0]
