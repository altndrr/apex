import pytest
import requests
import torch
from PIL import Image

from src.models import HuggingFaceModel, get_model, list_models
from src.utils.classes import Request, Response
from src.utils.types import PILImage


@pytest.fixture
def generate_until_sample() -> tuple[str, PILImage]:
    """Return a sample of context and choices for answer generation."""
    image_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    image_pil = Image.open(requests.get(image_url, stream=True, timeout=10).raw).convert("RGB")
    text_context = "What's in the image?"

    return text_context, image_pil


@pytest.fixture
def rank_answer_batch() -> tuple[list[str], list[list[str]], list[PILImage]]:
    """Return a batch of samples for answer ranking."""
    image_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    image_pil = Image.open(requests.get(image_url, stream=True, timeout=10).raw).convert("RGB")

    texts_context = ["How many dogs are in the image?", "How many women or men are in the image?"]
    texts_choices = [["1", "2", "3"], ["1", "2", "3"]]
    images_pil = [image_pil, image_pil]

    return texts_context, texts_choices, images_pil


@pytest.fixture
def blip2() -> HuggingFaceModel:
    """Return the BLIP-2 model."""
    model: HuggingFaceModel = get_model("blip2-opt-2.7b", load_in_8bit=True)

    return model


@pytest.fixture
def idefics() -> HuggingFaceModel:
    """Return the IDEFICS model."""
    model: HuggingFaceModel = get_model("idefics-9b-instruct", load_in_8bit=True)

    return model


@pytest.fixture
def instructblip() -> HuggingFaceModel:
    """Return the InstructBLIP model."""
    model: HuggingFaceModel = get_model("instructblip-vicuna-7b", load_in_8bit=True)

    return model


@pytest.fixture
def llava() -> HuggingFaceModel:
    """Return the LLaVA 1.5 model."""
    model: HuggingFaceModel = get_model("llava-1.5-7b", load_in_8bit=True)

    return model


@pytest.fixture
def llava_next() -> HuggingFaceModel:
    """Return the LLaVA 1.6 model."""
    model: HuggingFaceModel = get_model("llava-1.6-vicuna-7b", load_in_8bit=True)

    return model


@pytest.mark.slow
def test_blip2_model_names() -> None:
    """Test the names of the BLIP-2 models."""
    model_names = [name for name in list_models() if name.startswith("blip2")]
    for model_name in model_names:
        model = get_model(model_name)
        assert isinstance(model, HuggingFaceModel)


def test_blip2_generate_until(blip2: HuggingFaceModel, generate_until_sample: tuple) -> None:
    """Test the BLIP-2 model for generation.

    Args:
    ----
        blip2 (HuggingFaceModel): The BLIP-2 model.
        generate_until_sample (tuple): A sample of context and image for generation.

    """
    # test the processor
    inputs = blip2.generate_until_processor(*generate_until_sample)

    assert isinstance(inputs, list) and len(inputs) == 1
    assert isinstance(inputs[0], Request)

    for request in inputs:
        assert isinstance(request.context_T, list) and isinstance(request.context_T[0], int)
        assert request.choice_T is None
        assert isinstance(request.whole_T, list) and isinstance(request.whole_T[0], int)
        assert len(request.whole_T) == len(request.context_T)
        assert "attention_mask" in request.kwargs
        assert len(request.kwargs["attention_mask"]) == len(request.context_T)

        # decode and check if the tokens are correct
        decoded_context = blip2.processor.decode(request.context_T)
        decoded_whole = blip2.processor.decode(request.whole_T)
        assert decoded_context == request.context
        assert decoded_whole == request.context

    # test the method
    with torch.no_grad():
        outputs = blip2.generate_until(inputs)

    assert isinstance(outputs, list) and len(outputs) == 1
    assert isinstance(outputs[0], Response)

    # decode the answers and check the first sentence
    answer = [output.text[: output.text.rfind(".") + 1] for output in outputs]
    assert answer[0].startswith(
        "A woman and her dog on the beach at sunset, looking at the ocean and the sky, with the"
        " sun setting behind them."
    )


def test_blip2_rank_answers(blip2: HuggingFaceModel, rank_answer_batch: tuple) -> None:
    """Test the BLIP-2 model for answer ranking.

    Args:
    ----
        blip2 (HuggingFaceModel): The BLIP-2 model.
        rank_answer_batch (tuple): A sample of context, choices, and image for answer ranking.

    """
    # test the processor
    texts_choices = rank_answer_batch[1]
    inputs = blip2.rank_answers_processor(*rank_answer_batch)

    assert isinstance(inputs, list) and len(inputs) == 2
    assert isinstance(inputs[0], list) and len(inputs[0]) == len(texts_choices[0])
    assert isinstance(inputs[0][0], Request)

    for request in inputs[0]:
        assert isinstance(request.context_T, list) and isinstance(request.context_T[0], int)
        assert isinstance(request.choice_T, list) and isinstance(request.choice_T[0], int)
        assert isinstance(request.whole_T, list) and isinstance(request.whole_T[0], int)
        assert len(request.whole_T) == len(request.context_T) + len(request.choice_T)
        assert "attention_mask" in request.kwargs
        assert len(request.kwargs["attention_mask"]) == len(request.whole_T)

        # decode and check if the tokens are correct
        decoded_context = blip2.processor.decode(request.context_T)
        choice_str = blip2.processor.decode(request.choice_T)
        decoded_whole = blip2.processor.decode(request.whole_T)
        assert decoded_context == request.context
        assert choice_str in request.choice
        assert decoded_whole == request.context + request.choice

    # test the method
    with torch.no_grad():
        outputs = blip2.rank_answers(inputs)

    assert isinstance(outputs, list) and len(outputs) == 2
    assert isinstance(outputs[0], list) and len(outputs[0]) == len(texts_choices[0])
    assert isinstance(outputs[0][0], Response)

    # get the answer that maximizes the log_prob_sum and convert it to its index
    answers = [max(output, key=lambda x: x.log_prob_sum).text for output in outputs]
    assert answers == ["1", "1"]


@pytest.mark.slow
def test_idefics_model_names() -> None:
    """Test the names of the IDEFICS models."""
    model_names = [name for name in list_models() if name.startswith("idefics")]
    for model_name in model_names:
        model = get_model(model_name)
        assert isinstance(model, HuggingFaceModel)


def test_idefics_generate_until(idefics: HuggingFaceModel, generate_until_sample: tuple) -> None:
    """Test the IDEFICS model for generation.

    Args:
    ----
        idefics (HuggingFaceModel): The IDEFICS model.
        generate_until_sample (tuple): A sample of context and image for generation.

    """
    # test the processor
    inputs = idefics.generate_until_processor(*generate_until_sample)

    assert isinstance(inputs, list) and len(inputs) == 1
    assert isinstance(inputs[0], Request)

    for request in inputs:
        assert isinstance(request.context_T, list) and isinstance(request.context_T[0], int)
        assert request.choice_T is None
        assert isinstance(request.whole_T, list) and isinstance(request.whole_T[0], int)
        assert len(request.whole_T) == len(request.context_T)
        assert "attention_mask" in request.kwargs
        assert len(request.kwargs["attention_mask"]) == len(request.context_T)

        # decode and check if the tokens are correct
        decoded_context = idefics.processor.decode(request.context_T)
        decoded_whole = idefics.processor.decode(request.whole_T)
        assert decoded_context == request.context
        assert decoded_whole == request.context

    # test the method
    with torch.no_grad():
        outputs = idefics.generate_until(inputs)

    assert isinstance(outputs, list) and len(outputs) == 1
    assert isinstance(outputs[0], Response)

    # decode the answers and check the first sentence
    answer = [output.text[: output.text.rfind(".") + 1] for output in outputs]
    assert answer[0].startswith(
        "The image features a woman sitting on a sandy beach with her dog."
    )


def test_idefics_rank_answers(idefics: HuggingFaceModel, rank_answer_batch: tuple) -> None:
    """Test the IDEFICS model for answer ranking.

    Args:
    ----
        idefics (HuggingFaceModel): The IDEFICS model.
        rank_answer_batch (tuple): A sample of context, choices, and image for answer ranking.

    """
    # test the processor
    texts_choices = rank_answer_batch[1]
    inputs = idefics.rank_answers_processor(*rank_answer_batch)

    assert isinstance(inputs, list) and len(inputs) == 2
    assert isinstance(inputs[0], list) and len(inputs[0]) == len(texts_choices[0])
    assert isinstance(inputs[0][0], Request)

    for request in inputs[0]:
        assert isinstance(request.context_T, list) and isinstance(request.context_T[0], int)
        assert isinstance(request.choice_T, list) and isinstance(request.choice_T[0], int)
        assert isinstance(request.whole_T, list) and isinstance(request.whole_T[0], int)
        assert len(request.whole_T) == len(request.context_T) + len(request.choice_T)
        assert "attention_mask" in request.kwargs
        assert len(request.kwargs["attention_mask"]) == len(request.whole_T)

        # decode and check if the tokens are correct
        decoded_context = idefics.processor.decode(request.context_T)
        choice_str = idefics.processor.decode(request.choice_T)
        decoded_whole = idefics.processor.decode(request.whole_T)
        assert decoded_context == request.context
        assert choice_str in request.choice
        assert decoded_whole == request.context + request.choice

    # test the method
    with torch.no_grad():
        outputs = idefics.rank_answers(inputs)

    assert isinstance(outputs, list) and len(outputs) == 2
    assert isinstance(outputs[0], list) and len(outputs[0]) == len(texts_choices[0])
    assert isinstance(outputs[0][0], Response)

    # get the answer that maximizes the log_prob_sum and convert it to its index
    answers = [max(output, key=lambda x: x.log_prob_sum).text for output in outputs]
    assert answers == ["1", "1"]


@pytest.mark.slow
def test_instructblip_model_names() -> None:
    """Test the names of the Instruct-BLIP models."""
    model_names = [name for name in list_models() if name.startswith("instructblip")]
    for model_name in model_names:
        model = get_model(model_name)
        assert isinstance(model, HuggingFaceModel)


def test_instructblip_generate_until(
    instructblip: HuggingFaceModel, generate_until_sample: tuple
) -> None:
    """Test the InstructBLIP model for generation.

    Args:
    ----
        instructblip (HuggingFaceModel): The InstructBLIP model.
        generate_until_sample (tuple): A sample of context and image for generation.

    """
    # test the processor
    inputs = instructblip.generate_until_processor(*generate_until_sample)

    assert isinstance(inputs, list) and len(inputs) == 1
    assert isinstance(inputs[0], Request)

    for request in inputs:
        assert isinstance(request.context_T, list) and isinstance(request.context_T[0], int)
        assert request.choice_T is None
        assert isinstance(request.whole_T, list) and isinstance(request.whole_T[0], int)
        assert len(request.whole_T) == len(request.context_T)
        assert "attention_mask" in request.kwargs
        assert len(request.kwargs["attention_mask"]) == len(request.context_T)

        # decode and check if the tokens are correct
        decoded_context = instructblip.processor.decode(request.context_T)
        decoded_whole = instructblip.processor.decode(request.whole_T)
        assert decoded_context == request.context
        assert decoded_whole == request.context

    # test the method
    with torch.no_grad():
        outputs = instructblip.generate_until(inputs)

    assert isinstance(outputs, list) and len(outputs) == 1
    assert isinstance(outputs[0], Response)

    # decode the answers and check the first sentence
    answer = [output.text[: output.text.rfind(".") + 1] for output in outputs]
    assert answer[0].startswith(
        "The image features a woman sitting on the beach with a dog, who is pawing at her hand."
    )


def test_instructblip_rank_answers(
    instructblip: HuggingFaceModel, rank_answer_batch: tuple
) -> None:
    """Test the InstructBLIP model for answer ranking.

    Args:
    ----
        instructblip (HuggingFaceModel): The InstructBLIP model.
        rank_answer_batch (tuple): A sample of context, choices, and image for answer ranking.

    """
    # test the processor
    texts_choices = rank_answer_batch[1]
    inputs = instructblip.rank_answers_processor(*rank_answer_batch)

    assert isinstance(inputs, list) and len(inputs) == 2
    assert isinstance(inputs[0], list) and len(inputs[0]) == len(texts_choices[0])
    assert isinstance(inputs[0][0], Request)

    for request in inputs[0]:
        assert isinstance(request.context_T, list) and isinstance(request.context_T[0], int)
        assert isinstance(request.choice_T, list) and isinstance(request.choice_T[0], int)
        assert isinstance(request.whole_T, list) and isinstance(request.whole_T[0], int)
        assert len(request.whole_T) == len(request.context_T) + len(request.choice_T)
        assert "attention_mask" in request.kwargs
        assert len(request.kwargs["attention_mask"]) == len(request.whole_T)

        # decode and check if the tokens are correct
        decoded_context = instructblip.processor.decode(request.context_T)
        choice_str = instructblip.processor.decode(request.choice_T)
        decoded_whole = instructblip.processor.decode(request.whole_T)
        assert decoded_context == request.context
        assert choice_str in request.choice
        assert decoded_whole == request.context + request.choice

    # test the method
    with torch.no_grad():
        outputs = instructblip.rank_answers(inputs)

    assert isinstance(outputs, list) and len(outputs) == 2
    assert isinstance(outputs[0], list) and len(outputs[0]) == len(texts_choices[0])
    assert isinstance(outputs[0][0], Response)

    # get the answer that maximizes the log_prob_sum and convert it to its index
    answers = [max(output, key=lambda x: x.log_prob_sum).text for output in outputs]
    assert answers == ["1", "1"]


@pytest.mark.slow
def test_llava_model_names() -> None:
    """Test the names of the LLaVA 1.5 models."""
    blip_model_names = [name for name in list_models() if name.startswith("llava-1.5")]
    for model_name in blip_model_names:
        model = get_model(model_name)
        assert isinstance(model, HuggingFaceModel)


def test_llava_generate_until(llava: HuggingFaceModel, generate_until_sample: tuple) -> None:
    """Test the LLaVA 1.5 model for generation.

    Args:
    ----
        llava (HuggingFaceModel): The LLaVA 1.5 model.
        generate_until_sample (tuple): A sample of context and image for generation.

    """
    # test the processor
    inputs = llava.generate_until_processor(*generate_until_sample)

    assert isinstance(inputs, list) and len(inputs) == 1
    assert isinstance(inputs[0], Request)

    for request in inputs:
        assert isinstance(request.context_T, list) and isinstance(request.context_T[0], int)
        assert request.choice_T is None
        assert isinstance(request.whole_T, list) and isinstance(request.whole_T[0], int)
        assert len(request.whole_T) == len(request.context_T)
        assert "attention_mask" in request.kwargs
        assert len(request.kwargs["attention_mask"]) == len(request.context_T)

        # decode and check if the tokens are correct
        decoded_context = llava.processor.decode(request.context_T)
        decoded_whole = llava.processor.decode(request.whole_T)
        assert decoded_context == request.context
        assert decoded_whole == request.context

    # test the method
    with torch.no_grad():
        outputs = llava.generate_until(inputs)

    assert isinstance(outputs, list) and len(outputs) == 1
    assert isinstance(outputs[0], Response)

    # decode the answers and check the first sentence
    answer = [output.text[: output.text.rfind(".") + 1] for output in outputs]
    assert answer[0].startswith(
        "The image features a woman sitting on the beach, with a dog sitting next to her."
    )


def test_llava_rank_answers(llava: HuggingFaceModel, rank_answer_batch: tuple) -> None:
    """Test the LLaVA 1.5 model for answer ranking.

    Args:
    ----
        llava (HuggingFaceModel): The LLaVA 1.5 model.
        rank_answer_batch (tuple): A sample of context, choices, and image for answer ranking.

    """
    # test the processor
    texts_choices = rank_answer_batch[1]
    inputs = llava.rank_answers_processor(*rank_answer_batch)

    assert isinstance(inputs, list) and len(inputs) == 2
    assert isinstance(inputs[0], list) and len(inputs[0]) == len(texts_choices[0])
    assert isinstance(inputs[0][0], Request)

    for request in inputs[0]:
        assert isinstance(request.context_T, list) and isinstance(request.context_T[0], int)
        assert isinstance(request.choice_T, list) and isinstance(request.choice_T[0], int)
        assert isinstance(request.whole_T, list) and isinstance(request.whole_T[0], int)
        assert len(request.whole_T) == len(request.context_T) + len(request.choice_T)
        assert "attention_mask" in request.kwargs
        assert len(request.kwargs["attention_mask"]) == len(request.whole_T)

        # decode and check if the tokens are correct
        decoded_context = llava.processor.decode(request.context_T)
        choice_str = llava.processor.decode(request.choice_T)
        decoded_whole = llava.processor.decode(request.whole_T)
        assert decoded_context == request.context
        assert choice_str in request.choice
        assert decoded_whole == request.context + request.choice

    # test the method
    with torch.no_grad():
        outputs = llava.rank_answers(inputs)

    assert isinstance(outputs, list) and len(outputs) == 2
    assert isinstance(outputs[0], list) and len(outputs[0]) == len(texts_choices[0])
    assert isinstance(outputs[0][0], Response)

    # get the answer that maximizes the log_prob_sum and convert it to its index
    answers = [max(output, key=lambda x: x.log_prob_sum).text for output in outputs]
    assert answers == ["1", "1"]


@pytest.mark.slow
def test_llava_next_model_names() -> None:
    """Test the names of the LLaVA 1.6 models."""
    model_names = [name for name in list_models() if name.startswith("llava-1.6")]
    for model_name in model_names:
        model = get_model(model_name)
        assert isinstance(model, HuggingFaceModel)


def test_llava_next_generate_until(
    llava_next: HuggingFaceModel, generate_until_sample: tuple
) -> None:
    """Test the LLaVA 1.6 model for generation.

    Args:
    ----
        llava_next (HuggingFaceModel): The LLaVA 1.6 model.
        generate_until_sample (tuple): A sample of context and image for generation.

    """
    # test the processor
    inputs = llava_next.generate_until_processor(*generate_until_sample)

    assert isinstance(inputs, list) and len(inputs) == 1
    assert isinstance(inputs[0], Request)

    for request in inputs:
        assert isinstance(request.context_T, list) and isinstance(request.context_T[0], int)
        assert request.choice_T is None
        assert isinstance(request.whole_T, list) and isinstance(request.whole_T[0], int)
        assert len(request.whole_T) == len(request.context_T)
        assert "attention_mask" in request.kwargs
        assert len(request.kwargs["attention_mask"]) == len(request.context_T)

        # decode and check if the tokens are correct
        decoded_context = llava_next.processor.decode(request.context_T)
        decoded_whole = llava_next.processor.decode(request.whole_T)
        assert decoded_context == request.context
        assert decoded_whole == request.context

    # test the method
    with torch.no_grad():
        outputs = llava_next.generate_until(inputs)

    assert isinstance(outputs, list) and len(outputs) == 1
    assert isinstance(outputs[0], Response)

    # decode the answers and check the first sentence
    answer = [output.text[: output.text.rfind(".") + 1] for output in outputs]
    assert answer[0].startswith(
        "The image shows a person sitting on a sandy beach with a golden retriever dog."
    )


def test_llava_next_rank_answers(llava_next: HuggingFaceModel, rank_answer_batch: tuple) -> None:
    """Test the LLaVA 1.6 model for answer ranking.

    Args:
    ----
        llava_next (HuggingFaceModel): The LLaVA 1.6 model.
        rank_answer_batch (tuple): A sample of context, choices, and image for answer ranking.

    """
    # test the processor
    texts_choices = rank_answer_batch[1]
    inputs = llava_next.rank_answers_processor(*rank_answer_batch)

    assert isinstance(inputs, list) and len(inputs) == 2
    assert isinstance(inputs[0], list) and len(inputs[0]) == len(texts_choices[0])
    assert isinstance(inputs[0][0], Request)

    for request in inputs[0]:
        assert isinstance(request.context_T, list) and isinstance(request.context_T[0], int)
        assert isinstance(request.choice_T, list) and isinstance(request.choice_T[0], int)
        assert isinstance(request.whole_T, list) and isinstance(request.whole_T[0], int)
        assert len(request.whole_T) == len(request.context_T) + len(request.choice_T)

        # decode and check if the tokens are correct
        decoded_context = llava_next.processor.decode(request.context_T)
        choice_str = llava_next.processor.decode(request.choice_T)
        decoded_whole = llava_next.processor.decode(request.whole_T)
        assert decoded_context == request.context
        assert choice_str in request.choice
        assert decoded_whole == request.context + request.choice

    # test the method
    with torch.no_grad():
        outputs = llava_next.rank_answers(inputs)

    assert isinstance(outputs, list) and len(outputs) == 2
    assert isinstance(outputs[0], list) and len(outputs[0]) == len(texts_choices[0])
    assert isinstance(outputs[0][0], Response)

    # get the answer that maximizes the log_prob_sum and convert it to its index
    answers = [max(output, key=lambda x: x.log_prob_sum).text for output in outputs]
    assert answers == ["1", "1"]
