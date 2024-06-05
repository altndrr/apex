from typing import Literal

import torch
import torch.nn.functional as F
from PIL import Image

from src import utils
from src.data._sampler import RequestSampler
from src.models._api import register_model
from src.utils import pad_and_concat
from src.utils.classes import Prompt, Request, Response
from src.utils.types import PILImage

log = utils.get_logger(__name__, rank_zero_only=True)

__all__ = ["HuggingFaceModel"]


class HuggingFaceModel(torch.nn.Module):
    """HuggingFace model wrapper.

    Args:
    ----
        name_or_path (str): The name or path of the model to load.
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Extra arguments to pass to the model.

    Attributes:
    ----------
        model (torch.nn.Module): The model.
        model_name (str): The name of the model.
        model_type (str): The type of the model.
        config (dict): The model configuration.
        text_config (dict): The text configuration.
        processor (dict): The processor.
        image_processor (ImageProcessor): The image processor.
        text_processor (Tokenizer): The text processor.
        prompt (Prompt | None): The prompt to use.

    Dimension keys:
    ----------
        B: batch size
        T: text sequence length
        V: text vocabulary size
        C: image channels
        H: image height
        W: image width

    """

    def __init__(
        self,
        name_or_path: str,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        assert utils.TRANSFORMERS_AVAILABLE, "transformers package not installed"
        from transformers import AutoModelForVision2Seq, AutoProcessor, IdeficsForVisionText2Text
        from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES

        model_kwargs, processor_kwargs = {}, {}
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = getattr(torch, kwargs.get("dtype", "float16"))

        if load_in_8bit or load_in_4bit:
            if utils.BITSANDBYTES_AVAILABLE and utils.ACCELERATE_AVAILABLE:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit,
                    bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
                )
            elif not utils.ACCELERATE_AVAILABLE:
                log.warning("accelerate package not available. Skipping quantization...")
            elif not utils.BITSANDBYTES_AVAILABLE:
                log.warning("bitsandbytes package not available. Skipping quantization...")

        model = None
        if "idefics" in name_or_path:
            model = IdeficsForVisionText2Text.from_pretrained(name_or_path, **model_kwargs)
        else:
            model = AutoModelForVision2Seq.from_pretrained(name_or_path, **model_kwargs)

        if model is None:
            supported_types = MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
            supported_types.update(dict(idefics="IdeficsForVisionText2Text"))
            raise ValueError(
                f"Unrecognized model type: {name_or_path}. " f"Supported types: {supported_types}"
            )

        processor = AutoProcessor.from_pretrained(name_or_path, **processor_kwargs)
        image_processor = getattr(processor, "image_processor", None)
        text_processor = getattr(processor, "tokenizer", processor)
        text_config = getattr(model.config, "text_config", model.config)

        self._prompt = None
        self.model = model
        self.model_name = name_or_path
        self.model_type = model.config.model_type
        self.config = model.config
        self.text_config = text_config
        self.processor = processor
        self.image_processor = image_processor
        self.text_processor = text_processor

        if self.model_type not in ("blip-2", "idefics", "instructblip", "llava", "llava_next"):
            log.warning(
                "Unrecognized language model type %s. Model may not work as expected.",
                self.text_config.model_type,
            )

        self.generate_until_processor_kwargs = {"return_tensors": "pt"}
        self.decode_processor_kwargs = {"skip_special_tokens": True}
        self.rank_answers_processor_kwargs = {"return_tensors": "pt"}
        self.kwargs = kwargs

    @property
    def device(self) -> torch.device:
        """Get the model device."""
        return self.model.device

    @property
    def name(self) -> str:
        """Get the model name."""
        return self.config.model_type

    @property
    def prompt(self) -> Prompt | None:
        """Get the prompt."""
        if self._prompt is None:
            return None

        return self._prompt

    @prompt.setter
    def prompt(self, prompt: str) -> None:
        """Set the prompt.

        Args:
        ----
            prompt (str): The prompt to set.

        """
        self._prompt = Prompt(prompt)

    def generate_until(self, inputs: list[Request], **kwargs) -> list[Response]:  # noqa: D417
        """Answer to a list of requests by generating the response.

        Args:
        ----
            inputs (list[Request]): The requests to generate.
            **kwargs: Extra arguments to control the generation process.

        Dimension keys:
        ----------
            B: batch size
            T: text sequence length
            V: text vocabulary size
            C: image channels
            H: image height
            W: image width

        """
        batch_size = kwargs.pop("batch_size", len(inputs))
        kwargs["max_new_tokens"] = kwargs.pop("max_new_tokens", 50)

        if self.model_type == "idefics":
            bad_words = ["<image>", "<fake_token_around_image>"]
            bad_words_ids = self.processor.tokenizer(bad_words, add_special_tokens=False).input_ids
            kwargs["bad_words_ids"] = bad_words_ids

        outputs = []
        request_sampler = RequestSampler(inputs, init_group_by=None, batch_size=batch_size)
        for requests in request_sampler:
            input_BT = [torch.tensor(context_T) for context_T in requests.context_BT]
            input_len_B = [input_T.shape[0] for input_T in input_BT]
            padding_T = max(input_len_B)

            call_kwargs = {}
            call_kwargs["input_ids"] = pad_and_concat(input_BT, max_len=padding_T)
            call_kwargs["pixel_values"] = requests.image_BCHW

            # add other kwargs, additionally handling their padding
            for key in requests.kwargs:
                attention_keys = ("attention_mask", "image_attention_mask")
                qformer_keys = ("qformer_input_ids", "qformer_attention_mask")
                if key in attention_keys + qformer_keys:
                    call_kwargs[key] = requests.kwargs[key][:, :padding_T]
                else:
                    call_kwargs[key] = requests.kwargs[key]

            # add the history dimension for the idefics model
            if self.model_type == "idefics":
                call_kwargs["pixel_values"].unsqueeze_(1)

            call_kwargs = {k: v.to(self.device) for k, v in call_kwargs.items()}
            output_BT = self.model.generate(**call_kwargs, **kwargs)

            for output_T, input_len_1 in zip(output_BT, input_len_B, strict=True):
                if self.model_type in ("blip-2", "instructblip"):
                    pass
                elif self.model_type in ("idefics", "llava", "llava_next"):
                    output_T = output_T[min(len(output_T), input_len_1) :]
                else:
                    raise ValueError(f"Unrecognized model type: {self.model_type}")
                output = self.processor.decode(output_T, **self.decode_processor_kwargs)
                outputs.append(Response(output, True, None, None))

        return request_sampler.restore_order(outputs)

    def generate_until_processor(
        self, texts_context: list[str] | str, images_pil: list[PILImage] | PILImage
    ) -> list[list[Request] | Request]:
        """Preprocess a batch of data for the generate until method.

        Args:
        ----
            texts_context (list[str] | str): The textual context containing the question.
            images_pil (list[PILImage] | PILImage): The visual context.

        """
        if self.prompt is None:
            raise ValueError("Prompt must be set before encoding requests")

        # if single sample is passed, convert to list
        if isinstance(texts_context, str):
            if isinstance(images_pil, list):
                raise ValueError("Single context with images")
            texts_context = [texts_context]
            images_pil = [images_pil]

        requests = []
        processor_kwargs = self.generate_until_processor_kwargs
        for text_context, image_pil in zip(texts_context, images_pil, strict=True):
            context = self.prompt.substitute(question=text_context)
            image_pil = Image.open(image_pil) if isinstance(image_pil, str) else image_pil

            n_spaces = len(context) - len(context.rstrip())
            if n_spaces > 0:
                context = context[:-n_spaces]

            whole = context
            if self.model_type == "idefics":
                # encode text and images in sequence (i.e., as a single interleaved list)
                whole_parts = whole.split("<end_of_utterance>")
                whole = [whole_parts[0], image_pil, "<end_of_utterance>", whole_parts[1]]
                whole_inputs = self.processor(whole, **processor_kwargs)

                # make the context in the request identical to the decoded context
                context = [
                    "<s> " + whole[0],
                    "<fake_token_around_image><image><fake_token_around_image>",
                    whole[2] + " ",
                    whole[3],
                ]
                context = "".join(context)
            elif self.model_type in ("blip-2", "instructblip", "llava", "llava_next"):
                # encode text and images in parallel
                whole_inputs = self.processor(text=whole, images=image_pil, **processor_kwargs)

                # make the context in the request identical to the decoded context
                if self.model_type in ("blip-2"):
                    pass
                elif self.model_type in ("instructblip"):
                    context = "</s> " + context
                elif self.model_type in ("llava", "llava_next"):
                    context = context.replace("<image>", "<image> ")
                    context = "<s> " + context
            else:
                raise ValueError(f"Unrecognized model type: {self.model_type}")

            context_T = whole_inputs.pop("input_ids").squeeze(0).tolist()
            image_CHW = whole_inputs.pop("pixel_values").squeeze(0)
            kwargs = whole_inputs

            kwargs = {k: v.squeeze(0).tolist() for k, v in kwargs.items()}

            requests.append(Request(context, None, image_pil, context_T, None, image_CHW, kwargs))

        return requests

    def rank_answers(  # noqa: D417
        self, inputs: list[list[Request]], **kwargs
    ) -> list[list[Response]]:
        """Answer to a list of grouped requests by ranking the possible answers.

        Args:
        ----
            inputs (list[list[Request]]): The requests to rank. The inputs to the function are
                expected to be the outputs of the `rank_answers_processor` method.
            kwargs: Extra arguments to control the ranking process.

        Dimension keys:
        ----------
            B: batch size
            T: text sequence length
            V: text vocabulary size
            C: image channels
            H: image height
            W: image width

        """
        batch_size = kwargs.pop("batch_size", len(inputs))

        outputs = []
        request_sampler = RequestSampler(inputs, init_group_by="context", batch_size=batch_size)
        for requests in request_sampler:
            start = -(self.text_config.max_position_embeddings + 1)

            # select up to the last token before the end of the context
            input_BT = [torch.tensor(whole_T[start:][:-1]) for whole_T in requests.whole_BT]
            input_len_B = [input_T.shape[0] for input_T in input_BT]
            padding_T = max(input_len_B)

            call_kwargs = {}
            call_kwargs["input_ids"] = pad_and_concat(input_BT, max_len=padding_T)
            call_kwargs["pixel_values"] = requests.image_BCHW

            # add other kwargs, additionally handling their padding
            for key in requests.kwargs:
                attention_keys = ("attention_mask", "image_attention_mask")
                qformer_keys = ("qformer_input_ids", "qformer_attention_mask")
                if key in attention_keys + qformer_keys:
                    call_kwargs[key] = requests.kwargs[key][:, :padding_T]
                else:
                    call_kwargs[key] = requests.kwargs[key]

            # add the history dimension for the idefics model
            if self.model_type == "idefics":
                call_kwargs["pixel_values"].unsqueeze_(1)

            call_kwargs = {k: v.to(self.device) for k, v in call_kwargs.items()}
            logits_BTV = F.log_softmax(self.model(**call_kwargs).logits, dim=-1)

            zipped_data = zip(requests, logits_BTV, input_len_B, strict=True)
            for request, logits_TV, input_len_1 in zipped_data:
                context_len_T = input_len_1 + (logits_TV.shape[0] - padding_T)

                logits_TV = self._select_completion_token(
                    logits_TV, completion_len=len(request.choice_T), input_len=context_len_T
                )
                logits_1TV = logits_TV.unsqueeze(0)

                # check if per-token argmax is exactly equal to choice
                greedy_1T = logits_1TV.argmax(dim=-1)

                # check for one-token choice cache hits
                # noop in case group != "context" or no cache hit and returns the original args.
                # otherwise, expands the logits batch dimension and yields each batch along with
                # matching choice tokens and prompt strings
                for cache in request_sampler.retrieve_cache(request, logits_1TV):
                    request, choice_T, logits_1TV = cache
                    choice_1T = torch.tensor(
                        choice_T, dtype=torch.long, device=self.device
                    ).unsqueeze(0)
                    max_equal = (greedy_1T == choice_1T).all()

                    # obtain log-prob at the corresponding choice token indices
                    logits_1T = torch.gather(logits_1TV, 2, choice_1T.unsqueeze(-1)).squeeze(-1)

                    # answer: (choice, is_equal_to_greedy, log_prob_sum, log_prob_mean)
                    answer = Response(request.choice, max_equal, logits_1T.sum(), logits_1T.mean())
                    outputs.append(answer)

        return request_sampler.restore_order_and_groups(outputs)

    def rank_answers_processor(
        self,
        texts_context: list[str] | str,
        texts_choices: list[list[str]] | list[str],
        images_pil: list[PILImage] | PILImage,
        group_by: Literal["context"] | None = "context",
    ) -> list[list[Request] | Request]:
        """Preprocess a batch of data for the rank answers method.

        Args:
        ----
            texts_context (list[str] | str): The textual context containing the question.
            texts_choices (list[list[str]] | list[str]): The possible textual answers.
            images_pil (list[PILImage] | PILImage): The visual context.
            group_by ("context" | None): The key to group the requests by. With
                `group_by="context"`, the requests will be grouped by the image and text input
                context, having lists of requests differing only on the choice. Defaults to None.

        """
        if self.prompt is None:
            raise ValueError("Prompt must be set before encoding requests")

        # if single sample is passed, convert to list
        if isinstance(texts_context, str):
            if isinstance(texts_choices[0], list):
                raise ValueError("Single context with list of list of choices")
            elif isinstance(images_pil, list):
                raise ValueError("Single context with multiple images")
            texts_context = [texts_context]
            texts_choices = [texts_choices]
            images_pil = [images_pil]

        requests_groups = []
        processor_kwargs = self.rank_answers_processor_kwargs
        zipped_data = zip(texts_context, texts_choices, images_pil, strict=True)
        for text_context, text_choices, image_pil in zipped_data:
            requests_group = []

            for choice in text_choices:
                context = self.prompt.substitute(question=text_context)
                choice = choice.strip()  # remove leading/trailing whitespace
                choice = choice[0].upper() + choice[1:]  # uppercase initial
                choice = " " + choice  # ensure leading space
                image_pil = Image.open(image_pil) if isinstance(image_pil, str) else image_pil

                n_spaces = len(context) - len(context.rstrip())
                if n_spaces > 0:
                    choice = context[-n_spaces:] + choice
                    context = context[:-n_spaces]

                whole = context + choice
                if self.model_type == "idefics":
                    # encode text and images in sequence (i.e., as a single interleaved list)
                    whole_parts = whole.split("<end_of_utterance>")
                    context_parts = context.split("<end_of_utterance>")
                    whole = [whole_parts[0], image_pil, "<end_of_utterance>", whole_parts[1]]
                    context = [context_parts[0], image_pil, "<end_of_utterance>", context_parts[1]]
                    whole_inputs = self.processor(whole, **processor_kwargs)
                    context_inputs = self.processor(context, **processor_kwargs)

                    # make the context in the request identical to the decoded context
                    context = [
                        "<s> " + context[0],
                        "<fake_token_around_image><image><fake_token_around_image>",
                        context[2] + " ",
                        context[3],
                    ]
                    context = "".join(context)
                elif self.model_type in ("blip-2", "instructblip", "llava", "llava_next"):
                    # encode text and images in parallel
                    whole_inputs = self.processor(text=whole, images=image_pil, **processor_kwargs)
                    context_inputs = self.processor(text=context, **processor_kwargs)

                    # make the context in the request identical to the decoded context
                    if self.model_type in ("blip-2"):
                        pass
                    elif self.model_type in ("instructblip"):
                        context = "</s> " + context
                    elif self.model_type in ("llava", "llava_next"):
                        context = context.replace("<image>", "<image> ")
                        context = "<s> " + context
                else:
                    raise ValueError(f"Unrecognized model type: {self.model_type}")

                whole_T = whole_inputs.pop("input_ids").squeeze(0).tolist()
                context_T = context_inputs.input_ids.squeeze(0).tolist()
                image_CHW = whole_inputs.pop("pixel_values").squeeze(0)
                kwargs = whole_inputs

                context_T_len = len(context_T)
                choice_T = whole_T[context_T_len:]
                kwargs = {k: v.squeeze(0).tolist() for k, v in kwargs.items()}

                requests_group.append(
                    Request(context, choice, image_pil, context_T, choice_T, image_CHW, kwargs)
                )

            requests_groups.append(requests_group)

        if group_by == "context":
            requests = requests_groups
        elif group_by is None:
            requests = sum(requests_groups, [])
        else:
            raise ValueError(f"Invalid group_by key: {group_by}")

        return requests

    def _select_completion_token(
        self, logits: torch.Tensor, completion_len: int | None = None, input_len: int | None = None
    ) -> torch.Tensor:
        """Select the token from the logits that will be used as the completion.

        Args:
        ----
            logits (torch.Tensor): The logits from the model.
            completion_len (int, optional): The length of the completion. Defaults to None.
            input_len (int, optional): The length of the input. Defaults to None.

        """
        if not completion_len or not input_len:
            raise ValueError("completion_len and input_len must be provided for causal models")

        # discard right padding and input/context tokens
        start, end = input_len - completion_len, input_len
        logits = logits[start:end]

        return logits


@register_model("blip2-opt-2.7b")
def blip2_opt_small(
    load_in_8bit: bool = False, load_in_4bit: bool = False, **kwargs
) -> HuggingFaceModel:
    """Load the BLIP-2 model with OPT 2.7B.

    Args:
    ----
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Extra arguments to pass to the model.

    """
    name = "Salesforce/blip2-opt-2.7b"
    model = HuggingFaceModel(name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, **kwargs)
    model.prompt = "Question: ${question}\nAnswer: ${answer}"
    model.generate_until_processor_kwargs["add_special_tokens"] = False
    model.rank_answers_processor_kwargs["add_special_tokens"] = False
    return model


@register_model("blip2-opt-6.7b")
def blip2_opt_large(
    load_in_8bit: bool = False, load_in_4bit: bool = False, **kwargs
) -> HuggingFaceModel:
    """Load the BLIP-2 model with OPT 6.7B.

    Args:
    ----
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Extra arguments to pass to the model.

    """
    name = "Salesforce/blip2-opt-6.7b"
    model = HuggingFaceModel(name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, **kwargs)
    model.prompt = "Question: ${question}\nAnswer: ${answer}"
    model.generate_until_processor_kwargs["add_special_tokens"] = False
    model.rank_answers_processor_kwargs["add_special_tokens"] = False
    return model


@register_model("blip2-flan-t5-xl")
def blip2_flan_t5_xl(
    load_in_8bit: bool = False, load_in_4bit: bool = False, **kwargs
) -> HuggingFaceModel:
    """Load the BLIP-2 model with Flan T5-xl.

    Args:
    ----
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Extra arguments to pass to the model.

    """
    name = "Salesforce/blip2-flan-t5-xl"
    model = HuggingFaceModel(name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, **kwargs)
    model.prompt = "Question: ${question}\nAnswer: ${answer}"
    model.generate_until_processor_kwargs["add_special_tokens"] = False
    model.rank_answers_processor_kwargs["add_special_tokens"] = False
    return model


@register_model("blip2-flan-t5-xxl")
def blip2_flan_t5_xxl(
    load_in_8bit: bool = False, load_in_4bit: bool = False, **kwargs
) -> HuggingFaceModel:
    """Load the BLIP-2 model with Flan T5-xxl.

    Args:
    ----
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Extra arguments to pass to the model.

    """
    name = "Salesforce/blip2-flan-t5-xxl"
    model = HuggingFaceModel(name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, **kwargs)
    model.prompt = "Question: ${question}\nAnswer: ${answer}"
    model.generate_until_processor_kwargs["add_special_tokens"] = False
    model.rank_answers_processor_kwargs["add_special_tokens"] = False
    return model


@register_model("idefics-9b-instruct")
def idefics_9b_instruct(
    load_in_8bit: bool = False, load_in_4bit: bool = False, **kwargs
) -> HuggingFaceModel:
    """Load the IDEFICS 9B model instruction-tuned.

    Args:
    ----
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Extra arguments to pass to the model.

    """
    name = "HuggingFaceM4/idefics-9b-instruct"
    model = HuggingFaceModel(name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, **kwargs)
    model.prompt = "User: ${question}<end_of_utterance>\nAssistant: ${answer}"
    model.generate_until_processor_kwargs["add_end_of_utterance_token"] = False
    model.rank_answers_processor_kwargs["add_end_of_utterance_token"] = False
    return model


@register_model("idefics-80b-instruct")
def idefics_80b_instruct(
    load_in_8bit: bool = False, load_in_4bit: bool = False, **kwargs
) -> HuggingFaceModel:
    """Load the IDEFICS 80B model instruction-tuned.

    Args:
    ----
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Extra arguments to pass to the model.

    """
    name = "HuggingFaceM4/idefics-80b-instruct"
    model = HuggingFaceModel(name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, **kwargs)
    model.prompt = "User: ${question}<end_of_utterance>\nAssistant: ${answer}"
    model.generate_until_processor_kwargs["add_end_of_utterance_token"] = False
    model.rank_answers_processor_kwargs["add_end_of_utterance_token"] = False
    return model


@register_model("instructblip-vicuna-7b")
def instructblip_vicuna_7b(
    load_in_8bit: bool = False, load_in_4bit: bool = False, **kwargs
) -> HuggingFaceModel:
    """Load the InstructBLIP model with Vicuna 7B.

    Args:
    ----
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Extra arguments to pass to the model.

    """
    name = "Salesforce/instructblip-vicuna-7b"
    model = HuggingFaceModel(name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, **kwargs)
    model.prompt = "${question} ${answer}"
    return model


@register_model("instructblip-vicuna-13b")
def instructblip_vicuna_13b(
    load_in_8bit: bool = False, load_in_4bit: bool = False, **kwargs
) -> HuggingFaceModel:
    """Load the InstructBLIP model with Vicuna 13B.

    Args:
    ----
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Extra arguments to pass to the model.

    """
    name = "Salesforce/instructblip-vicuna-13b"
    model = HuggingFaceModel(name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, **kwargs)
    model.prompt = "${question} ${answer}"
    return model


@register_model("instructblip-flan-t5-xl")
def instructblip_flan_t5_xl(
    load_in_8bit: bool = False, load_in_4bit: bool = False, **kwargs
) -> HuggingFaceModel:
    """Load the InstructBLIP model with Flan T5-xl.

    Args:
    ----
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Extra arguments to pass to the model.

    """
    name = "Salesforce/instructblip-flan-t5-xl"
    model = HuggingFaceModel(name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, **kwargs)
    model.prompt = "${question} ${answer}"
    return model


@register_model("instructblip-flan-t5-xxl")
def instructblip_flan_t5_xxl(
    load_in_8bit: bool = False, load_in_4bit: bool = False, **kwargs
) -> HuggingFaceModel:
    """Load the InstructBLIP model with Flan T5-xxl.

    Args:
    ----
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Extra arguments to pass to the model.

    """
    name = "Salesforce/instructblip-flan-t5-xxl"
    model = HuggingFaceModel(name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, **kwargs)
    model.prompt = "${question} ${answer}"
    return model


@register_model("llava-1.5-7b")
def llava_7b(load_in_8bit: bool = False, load_in_4bit: bool = False, **kwargs) -> HuggingFaceModel:
    """Load the LLaVA 7B model version 1.5.

    Args:
    ----
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Extra arguments to pass to the model.

    """
    name = "llava-hf/llava-1.5-7b-hf"
    model = HuggingFaceModel(name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, **kwargs)
    model.prompt = "USER: <image>\n${question}\nASSISTANT: ${answer}"
    return model


@register_model("llava-1.5-13b")
def llava_13b(
    load_in_8bit: bool = False, load_in_4bit: bool = False, **kwargs
) -> HuggingFaceModel:
    """Load the LLaVA 13B model version 1.5.

    Args:
    ----
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Extra arguments to pass to the model.

    """
    name = "llava-hf/llava-1.5-13b-hf"
    model = HuggingFaceModel(name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, **kwargs)
    model.prompt = "USER: <image>\n${question}\nASSISTANT: ${answer}"
    return model


@register_model("llava-1.6-vicuna-7b")
def llava_next_vicuna_7b(
    load_in_8bit: bool = False, load_in_4bit: bool = False, **kwargs
) -> HuggingFaceModel:
    """Load the LLaVA 7B model version 1.6 with Vicuna.

    Args:
    ----
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Extra arguments to pass to the model.

    """
    name = "llava-hf/llava-v1.6-vicuna-7b-hf"
    model = HuggingFaceModel(name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, **kwargs)
    model.prompt = (
        "A chat between a curious human and an artificial intelligence assistant."
        " The assistant gives helpful, detailed, and polite answers to the human's questions."
        " USER: <image>\n${question}\nASSISTANT: ${answer}"
    )
    return model


@register_model("llava-1.6-mistral-7b")
def llava_next_mistral_7b(
    load_in_8bit: bool = False, load_in_4bit: bool = False, **kwargs
) -> HuggingFaceModel:
    """Load the LLaVA 7B model version 1.6 with Mistral.

    Args:
    ----
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Extra arguments to pass to the model.

    """
    name = "llava-hf/llava-v1.6-mistral-7b-hf"
    model = HuggingFaceModel(name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, **kwargs)
    model.prompt = "[INST] <image>\n${question}? [/INST]\n${answer}"
    return model


@register_model("llava-1.6-vicuna-13b")
def llava_next_vicuna_13b(
    load_in_8bit: bool = False, load_in_4bit: bool = False, **kwargs
) -> HuggingFaceModel:
    """Load the LLaVA 13B model version 1.6 with Vicuna.

    Args:
    ----
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Extra arguments to pass to the model.

    """
    name = "llava-hf/llava-v1.6-vicuna-13b-hf"
    model = HuggingFaceModel(name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, **kwargs)
    model.prompt = (
        "A chat between a curious human and an artificial intelligence assistant."
        " The assistant gives helpful, detailed, and polite answers to the human's questions."
        " USER: <image>\n${question}\nASSISTANT: ${answer}"
    )
    return model


@register_model("llava-1.6-yi-34b")
def llava_next_34b(
    load_in_8bit: bool = False, load_in_4bit: bool = False, **kwargs
) -> HuggingFaceModel:
    """Load the LLaVA 34B model version 1.6 with Yi.

    Args:
    ----
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Extra arguments to pass to the model.

    """
    name = "llava-hf/llava-v1.6-34b-hf"
    model = HuggingFaceModel(name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, **kwargs)
    model.prompt = (
        "<|im_start|>system\nAnswer the questions.<|im_end|>"
        "<|im_start|>user\n<image>\n${question}<|im_end|>"
        "<|im_start|>assistant\n${answer}<|im_end|>"
    )
    return model
