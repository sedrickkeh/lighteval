# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from argparse import Namespace
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import yaml
from transformers import AutoConfig, BitsAndBytesConfig, GPTQConfig, PretrainedConfig

from lighteval.logging.hierarchical_logger import hlog
from lighteval.models.utils import _get_model_sha
from lighteval.utils import (
    NO_AUTOGPTQ_ERROR_MSG,
    NO_BNB_ERROR_MSG,
    NO_PEFT_ERROR_MSG,
    is_accelerate_available,
    is_autogptq_available,
    is_bnb_available,
    is_peft_available,
)


if is_accelerate_available():
    from accelerate import Accelerator


@dataclass
class EnvConfig:
    """
    Configuration class for environment settings.

    Attributes:
        cache_dir (str): directory for caching data.
        token (str): authentication token used for accessing the HuggingFace Hub.
    """

    cache_dir: str = None
    token: str = None


@dataclass
class BaseModelConfig:
    """
    Base configuration class for models.

    Attributes:
        pretrained (str):
            HuggingFace Hub model ID name or the path to a pre-trained
            model to load. This is effectively the `pretrained_model_name_or_path`
            argument of `from_pretrained` in the HuggingFace `transformers` API.
        accelerator (Accelerator): accelerator to use for model training.
        tokenizer (Optional[str]): HuggingFace Hub tokenizer ID that will be
            used for tokenization.
        multichoice_continuations_start_space (Optional[bool]): Whether to add a
            space at the start of each continuation in multichoice generation.
            For example, context: "What is the capital of France?" and choices: "Paris", "London".
            Will be tokenized as: "What is the capital of France? Paris" and "What is the capital of France? London".
        subfolder (Optional[str]): The subfolder within the model repository.
        revision (str): The revision of the model.
        batch_size (int): The batch size for model training.
        max_gen_toks (Optional[int]): The maximum number of tokens to generate.
        max_length (Optional[int]): The maximum length of the generated output.
        add_special_tokens (bool, optional, defaults to True): Whether to add special tokens to the input sequences.
           If `None`, the default value will be set to `True` for seq2seq models (e.g. T5) and
            `False` for causal models.
        model_parallel (bool, optional, defaults to False):
            True/False: force to uses or not the `accelerate` library to load a large
            model across multiple devices.
            Default: None which correspond to comparing the number of process with
                the number of GPUs. If it's smaller => model-parallelism, else not.
        dtype (Union[str, torch.dtype], optional, defaults to None):):
            Converts the model weights to `dtype`, if specified. Strings get
            converted to `torch.dtype` objects (e.g. `float16` -> `torch.float16`).
            Use `dtype="auto"` to derive the type from the model's weights.
        device (Union[int, str]): device to use for model training.
        quantization_config (Optional[BitsAndBytesConfig]): quantization
            configuration for the model. Needed for 4-bit and 8-bit precision.
        trust_remote_code (bool): Whether to trust remote code during model
            loading.

    Methods:
        __post_init__(): Performs post-initialization checks on the configuration.
        _init_configs(model_name, env_config): Initializes the model configuration.
        init_configs(env_config): Initializes the model configuration using the environment configuration.
        get_model_sha(): Retrieves the SHA of the model.

    """

    pretrained: str
    accelerator: "Accelerator" = None
    tokenizer: Optional[str] = None
    multichoice_continuations_start_space: Optional[bool] = None
    subfolder: Optional[str] = None
    revision: str = "main"
    batch_size: int = -1
    max_gen_toks: Optional[int] = 256
    max_length: Optional[int] = None
    add_special_tokens: bool = True
    model_parallel: Optional[bool] = None
    dtype: Optional[Union[str, torch.dtype]] = None
    device: Union[int, str] = "cuda"
    quantization_config: Optional[BitsAndBytesConfig] = None
    trust_remote_code: bool = False
    use_chat_template: bool = False

    def __post_init__(self):
        if self.quantization_config is not None and not is_bnb_available():
            raise ImportError(NO_BNB_ERROR_MSG)

        if not isinstance(self.pretrained, str):
            raise ValueError("Pretrained model name must be passed as string.")
        if not isinstance(self.device, str):
            raise ValueError("Current device must be passed as string.")

    def _init_configs(self, model_name: str, env_config: EnvConfig) -> PretrainedConfig:
        revision = self.revision
        if self.subfolder:
            revision = f"{self.revision}/{self.subfolder}"
        auto_config = AutoConfig.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=self.trust_remote_code,
            cache_dir=env_config.cache_dir,
            token=env_config.token,
        )
        if getattr(auto_config, "quantization_config", False) and self.quantization_config is None:
            if not is_autogptq_available():
                raise ImportError(NO_AUTOGPTQ_ERROR_MSG)
            hlog(
                "`quantization_config` is None but was found in the model's config, using the one found in config.json"
            )
            self.quantization_config = GPTQConfig(**auto_config.quantization_config, disable_exllama=True)

        return auto_config

    def init_configs(self, env_config: EnvConfig) -> PretrainedConfig:
        return self._init_configs(self.pretrained, env_config=env_config)

    def get_model_sha(self):
        return _get_model_sha(repo_id=self.pretrained, revision=self.revision)


@dataclass
class DeltaModelConfig(BaseModelConfig):
    # Delta models look at the pretrained (= the delta weights) for the tokenizer and model config
    base_model: str = None

    def __post_init__(self):
        self.revision = "main"

        if not self.base_model:  # must have a default value bc of dataclass inheritance, but can't actually be None
            raise ValueError("The base_model argument must not be null for a delta model config")

        return super().__post_init__()

    def get_model_sha(self):
        return _get_model_sha(repo_id=self.pretrained, revision="main")


@dataclass
class AdapterModelConfig(BaseModelConfig):
    # Adapter models have the specificity that they look at the base model (= the parent) for the tokenizer and config
    base_model: str = None

    def __post_init__(self):
        if not is_peft_available():
            raise ImportError(NO_PEFT_ERROR_MSG)

        if not self.base_model:  # must have a default value bc of dataclass inheritance, but can't actually be None
            raise ValueError("The base_model argument must not be null for an adapter model config")

        return super().__post_init__()

    def init_configs(self, env_config: EnvConfig):
        return self._init_configs(self.base_model, env_config)


@dataclass
class TGIModelConfig:
    inference_server_address: str
    inference_server_auth: str


@dataclass
class InferenceModelConfig:
    model: str
    add_special_tokens: bool = True


@dataclass
class InferenceEndpointModelConfig:
    name: str
    repository: str
    accelerator: str
    vendor: str
    region: str
    instance_size: str
    instance_type: str
    model_dtype: str
    framework: str = "pytorch"
    endpoint_type: str = "protected"
    should_reuse_existing: bool = False
    add_special_tokens: bool = True
    revision: str = "main"
    namespace: str = None  # The namespace under which to launch the endopint. Defaults to the current user's namespace

    def get_dtype_args(self) -> Dict[str, str]:
        model_dtype = self.model_dtype.lower()
        if model_dtype in ["awq", "eetq", "gptq"]:
            return {"QUANTIZE": model_dtype}
        if model_dtype == "8bit":
            return {"QUANTIZE": "bitsandbytes"}
        if model_dtype == "4bit":
            return {"QUANTIZE": "bitsandbytes-nf4"}
        if model_dtype in ["bfloat16", "float16"]:
            return {"DTYPE": model_dtype}
        return {}

    @staticmethod
    def nullable_keys() -> list[str]:
        """
        Returns the list of optional keys in an endpoint model configuration. By default, the code requires that all the
        keys be specified in the configuration in order to launch the endpoint. This function returns the list of keys
        that are not required and can remain None.
        """
        return ["namespace"]


@dataclass
class OpenLMModelConfig(BaseModelConfig):

    openlm_model_type: str = None
    openlm_model_path: str = None
    openlm_model_config: str = None


def create_model_config(args: Namespace, accelerator: Union["Accelerator", None]) -> BaseModelConfig:  # noqa: C901
    """
    Create a model configuration based on the provided arguments.

    Args:
        args (Namespace): command-line arguments.
        accelerator (Union[Accelerator, None]): accelerator to use for model training.

    Returns:
        BaseModelConfig: model configuration.

    Raises:
        ValueError: If both an inference server address and model arguments are provided.
     ValueError: If multichoice continuations both should start with a space and should not start with a space.
        ValueError: If a base model is not specified when using delta weights or adapter weights.
        ValueError: If a base model is specified when not using delta weights or adapter weights.
    """
    # Default
    if args.model_args:
        args_dict = {k.split("=")[0]: k.split("=")[1] for k in args.model_args.split(",")}
        args_dict["accelerator"] = accelerator
        args_dict["use_chat_template"] = args.use_chat_template

        return BaseModelConfig(**args_dict)

    # OpenLM
    else:
        args_dict = {}
        args_dict["pretrained"] = args.tokenizer
        args_dict["accelerator"] = accelerator
        args_dict["use_chat_template"] = args.use_chat_template
        args_dict["tokenizer"] = args.tokenizer
        args_dict["openlm_model_type"] = args.openlm_model_type
        args_dict["openlm_model_path"] = args.openlm_model_path
        args_dict["openlm_model_config"] = args.openlm_model_config

        return OpenLMModelConfig(**args_dict)