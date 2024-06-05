import gc
import os
from collections import deque
from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import instructor
import openai
import torch
from omegaconf import DictConfig
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field, ValidationInfo, computed_field, model_validator
from torchmetrics.classification import Accuracy

from src import utils
from src.data import ExperimentDataset, get_dataloader
from src.data.loggers import close_loggers, get_logger, log_hyperparameters
from src.models import HuggingFaceModel, get_model
from src.tools import get_tools_docstrings
from src.tools._api import AVAIL_CATEGORIES

log = utils.get_logger(__name__, rank_zero_only=True)


def chat_completion_to_list(chat_completion: ChatCompletion) -> list[dict]:
    """Convert a chat completion to a list of messages.

    Args:
    ----
        chat_completion (ChatCompletion): Chat completion message.

    """
    messages = []

    chat_completion_message = chat_completion.choices[0].message.model_dump(exclude_none=True)
    messages.append(chat_completion_message)

    if "tool_calls" in chat_completion_message:
        for tool_call in chat_completion_message["tool_calls"]:
            message = dict(
                role="tool",
                name=tool_call["function"]["name"],
                tool_call_id=tool_call["id"],
                content=tool_call["function"]["arguments"],
            )
            messages.append(message)

    return messages


def evaluate_model(name: str, question: str, data: ExperimentDataset, cfg: DictConfig) -> dict:
    """Evaluate a model on an experimental dataset.

    Args:
    ----
        name (str): Name of the model.
        question (str): Question to ask the model.
        data (ExperimentDataset): Experimental dataset.
        cfg (DictConfig): Configuration composed by Hydra.

    """
    model: HuggingFaceModel = get_model(name=name, load_in_8bit=True)

    logger = None
    if cfg.get("logger"):
        log.info("Logging hyperparameters!")
        logger = get_logger(**cfg.logger)
        instances = dict(cfg=cfg, data=data, model=model)
        log_hyperparameters(logger, instances)

    model.eval()
    torch.set_grad_enabled(False)

    mode = cfg.get("mode", "rank")
    dataloader = get_dataloader(data, cfg.data)
    num_classes = len(data.answer_names)
    class_names = data.answer_names

    abstention_count = 0
    can_abstain = cfg.get("can_abstain_from_answer", False)
    if can_abstain:
        num_classes += 1
        class_names = class_names + ["unknown"]

    accuracy = Accuracy(task="multiclass", average="none", num_classes=num_classes)

    num_batches = len(dataloader)
    iterable = utils.get_iterable(dataloader, desc="Testing", total=num_batches)
    for _, batch in enumerate(iterable):
        images = batch["images_pil"]
        target_idx = torch.tensor(batch["labels_answer_idx"])

        if mode == "rank":
            contexts, choices = [question] * len(images), [class_names] * len(images)
            inputs = model.rank_answers_processor(contexts, choices, images)
            outputs = model.rank_answers(inputs)

            # get the answer that maximizes the log_prob_sum and convert it to its index
            preds_name = [max(output, key=lambda x: x.log_prob_mean).text for output in outputs]
            preds_name = [name.lower() for name in preds_name]  # convert to lowercase
            preds_idx = torch.tensor([class_names.index(pred) for pred in preds_name])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # update metrics
        accuracy.update(preds_idx, target_idx)
        if can_abstain:
            abstention_count += sum([pred == num_classes - 1 for pred in preds_idx]).item()

        # compute the metrics
        metrics = {}
        accuracy_values = accuracy.compute()[:-1] if can_abstain else accuracy.compute()
        metrics["average_accuracy"] = round(accuracy_values.mean().item(), 3)
        for i, acc in enumerate(accuracy_values):
            metrics[f"accuracy_answer_{i+1}"] = round(acc.item(), 3)

        # log the current metrics
        metrics_to_iterable = {}
        keep_in_iterable = ("accuracy",)
        for key, value in metrics.items():
            if any(keep in key for keep in keep_in_iterable):
                metrics_to_iterable[key] = value
        utils.format_iterable(iterable, metrics_to_iterable)
        if logger is not None:
            logger.log(metrics)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if can_abstain:
        metrics["abstention_rate"] = round(abstention_count / len(data), 3)

    torch.set_grad_enabled(True)
    close_loggers()

    return metrics


def init_master(master_cfg: DictConfig, seed: int | None) -> tuple[Callable, deque, dict]:
    """Load the master client, initialize the chat history, and the context.

    Args:
    ----
        master_cfg (DictConfig): Configuration composed by Hydra.
        seed (int | None): Random seed.

    """
    if master_cfg.get("parent") == "ollama":
        client = instructor.from_openai(
            openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
            mode=instructor.Mode.JSON,
        )
    elif master_cfg.get("parent") == "azure":
        client = instructor.from_openai(
            openai.AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version="2024-02-01",
            )
        )
    elif master_cfg.get("parent") == "openai":
        client = instructor.from_openai(openai.OpenAI())
    else:
        raise NotImplementedError(f"Model {master_cfg.parent} not implemented.")

    resources = master_cfg.system_resources.format(tools=get_tools_docstrings()).strip()
    system_prompt = master_cfg.system_prompt.format(resources=resources)

    history = deque()
    history.append(dict(role="system", content=system_prompt.strip()))

    master = partial(
        client.create_with_completion,
        messages=history,
        model=master_cfg.name,
        max_retries=master_cfg.max_retries,
        seed=seed,
    )

    return master, history, dict(resources=resources)


class Function(BaseModel):
    """Class describing a function call."""

    module_path: str = Field(
        description="module path of the function",
        examples=["src.tools.select", "src.tools.transform"],
    )
    name: str = Field(description="name of the function")
    kwargs: dict[str, Any] | None = Field(
        default_factory=dict,
        description="dictionary of key-value arguments to pass to the function",
        examples=[{"angle": 90}, {"width": 256, "height": 256}],
    )

    @computed_field
    @property
    def full_name(self) -> str:
        """Return the full name of the function."""
        return f"{self.module_path}.{self.name}"

    @model_validator(mode="after")
    def validate_properties(self, info: ValidationInfo) -> "Function":
        """Validate the properties of the function.

        Args:
        ----
            info (ValidationInfo): Information about the validation process.

        """
        resources = None if info.context is None else info.context.get("resources")

        self.kwargs = {} if self.kwargs is None else self.kwargs

        if self.name.endswith("()"):
            self.name = self.name[:-2]

        if self.module_path.endswith(self.name):
            self.module_path, _ = self.module_path.rsplit(".", 1)

        full_name = f"{self.module_path}.{self.name}"
        if resources and full_name not in resources:
            healed = False

            curr_category = self.module_path.rsplit(".", 1)[-1]
            for category in AVAIL_CATEGORIES:
                module = self.module_path.replace(curr_category, category)
                if f"{module}.{self.name}" in resources:
                    self.module_path = module
                    healed = True
                    break

            if not healed:
                raise ValueError(f"Resource {full_name} not found in the provided resources.")

        return self

    def __str__(self) -> str:
        """Return the string representation of the function."""
        return f"{self.full_name}({', '.join([f'{k}={v}' for k, v in self.kwargs.items()])})"


class Answer(BaseModel):
    """Class describing an answer to a question.

    Design an answer based on the following rules:
    - The answer must be short and capitalized. It must contain only the answer.
    - Answers must be specific and assertive.
    - Define specific class names for the images, no group names (e.g., animals).
    - Select relevant and specific images by retrieval or generation.
    - If no specific class names are needed, select a random class name.
    - Transform the samples to introduce the visual properties relevant to the question.
    - Function options must be specific and valid.
    - For alternative answers, provide valid and verifiable options that don't overlap.

    """

    id: int = Field(description="unique id of the answer")
    chain_of_thought: str = Field(description="think step by step about the answer")

    text: str = Field(description="short, capitalized, assertive")
    image_select_function: Function = Field(description="function to select the image")
    image_transform_function: Function = Field(description="function to transform the image")

    @model_validator(mode="after")
    def validate_properties(self, info: ValidationInfo) -> "Answer":
        """Validate the properties of the answer.

        Args:
        ----
            info (ValidationInfo): Information about the validation process.

        """
        self.text = self.text.capitalize()

        if "select" not in self.image_select_function.module_path:
            raise ValueError(
                f"{self.image_select_function.full_name} is an invalid select function"
            )

        if "transform" not in self.image_transform_function.module_path:
            raise ValueError(
                f"{self.image_transform_function.full_name} is an invalid transform function"
            )

        return self


class Experiment(BaseModel):
    """Class describing an experiment.

    Design an experiment based on the following rules:
    - The experiments must relate to the report and the query.
    - The experiment must be a visual question answering (VQA) task.
    - The question must be about visual properties.
    - The answers must be valid and verifiable.
    - At least two answers must be provided.
    - Experiments must have increasing difficulty levels.
    - Each experiment should be unique and build on the previous one.

    """

    id: int = Field(description="unique id of the experiment")

    chain_of_thought: str = Field(description="think step by step about about the experiment")

    difficulty: Literal["easy", "medium", "hard"] = Field(
        description="difficulty of the experiment"
    )

    vqa_question: str = Field(
        description="about visual properties, capitalized",
        examples=["Is the image [...]?", "What [...] is the image?"],
    )
    vqa_answers: list[Answer] = Field(description="numbered list, min 2 items")

    recap: str = Field(description="recap of the experiment")
    interpretation: str = Field(description="interpretation of the experiment")

    results: dict[str, Any] = Field(default={})
    findings: str = Field(default="")

    @computed_field
    @property
    def dataset_type(self) -> str:
        """Determine the type of dataset."""
        if all("Retrieval" in answer.image_select_function.name for answer in self.vqa_answers):
            return "natural"
        return "synthetic"

    @computed_field
    @property
    def image_type(self) -> str:
        """Determine the type of image."""
        return "single"

    @model_validator(mode="after")
    def validate_properties(self, info: ValidationInfo) -> "Experiment":
        """Validate the properties of the experiment.

        Args:
        ----
            info (ValidationInfo): Information about the validation process.

        """
        report = None if info.context is None else info.context.get("report")

        self.id = len(report.get("past_experiments", [])) + 1 if report else self.id
        self.vqa_question = self.vqa_question.capitalize()
        self.results = {}
        self.findings = ""

        if len(self.vqa_answers) < 2:
            raise ValueError("At least two answers must be provided")

        return self

    def summary(self) -> dict:
        """Summarize the experiment."""
        return dict(
            question=self.vqa_question,
            answers=[answer.text for answer in self.vqa_answers],
            select_tools=[str(answer.image_select_function) for answer in self.vqa_answers],
            transform_tools=[str(answer.image_transform_function) for answer in self.vqa_answers],
            results=self.results,
            findings=self.findings,
        )


class ExperimentDiscussion(BaseModel):
    """Class describing the discussion on the experiment.

    Design an experiment discussion based on the following rules:
    - The discussion must relate to the experiment.
    - The discussion must be a step-by-step analysis.
    - Recap and interpretations should reference the question and answers.

    """

    experiment_id: int = Field(description="unique id of the experiment")

    chain_of_thought: str = Field(description="think step by step about about the discussion")

    recap: str = Field(description="recap of the experiment")
    interpretation: str = Field(description="interpretation of the results")


class Report(BaseModel):
    """Class describing the evaluation report.

    Design an evaluation report based on the following rules:
    - The report must relate to a query.
    - The query must be an interrogative form.
    - Select the models to evaluate based on the query.
    - Select all models if no reference to models is made.

    """

    id: int = Field(description="unique id of the report")

    chain_of_thought: str = Field(description="think step by step about about the report")

    query: str = Field(description="interrogative form")
    open_questions: list[str] = Field(default_factory=list, description="questions to be answered")

    image_generation_model: Literal["stable-diffusion-xl-turbo"] = Field(
        default="stable-diffusion-xl-turbo", description="model for data generation"
    )
    image_retrieval_dataset: Literal["imagenet-x"] = Field(
        default="imagenet-x", description="source for data retrieval"
    )
    image_retrieval_dataset_groups_names: list[str] = Field(default=[])
    image_retrieval_dataset_image_types: list[str] = Field(default=[])

    models_to_evaluate: list[str] = Field(
        default_factory=list, description="relevant to the query"
    )

    past_experiments: list = Field(default=[])

    results: dict[str, Any] = Field(default={})

    conclusions: str = Field(default="")

    @model_validator(mode="after")
    def validate_resources(self, info: ValidationInfo) -> "Report":
        """Validate the user query message."""
        resources = None if info.context is None else info.context.get("resources")

        self.query = self.query.strip()
        if self.image_retrieval_dataset in ["imagenet", "imagenet-x"]:
            groups = (
                "device,dog,commodity,bird,structure,covering,wheeled vehicle,food,equipment,"
                "insect,vehicle,furniture,primate,vessel,snake,natural object,other"
            )
            self.image_retrieval_dataset_groups_names = groups.split(",")
            self.image_retrieval_dataset_image_types = ["photo"]
        self.past_experiments = []
        self.results = {}
        self.conclusions = ""

        if resources:
            for resource in self.models_to_evaluate:
                if resource not in resources:
                    raise ValueError(f"Resource {resource} not found in the provided resources.")

        return self


class ReportDiscussion(BaseModel):
    """Class describing the discussion of the report.

    Design a report discussion based on the following rules:
    - The discussion must relate to the report.
    - The discussion must be a step-by-step analysis.
    - Summarize the experiments so far with a recap and interpretation.
    - If necessary or sufficient experiments are missing, list them as open questions.

    """

    report_id: int = Field(description="unique id of the report")

    chain_of_thought: str = Field(description="think step by step about the report")

    recap: str = Field(description="summary of the report")
    interpretation: str = Field(description="interpretation of the experiments")

    has_sufficient_findings: bool = Field(
        default=False,
        description="whether sufficient findings have been obtained to draw conclusions",
    )
    has_necessary_findings: bool = Field(
        default=False,
        description="whether necessary findings have been obtained to draw conclusions",
    )

    open_questions: list[str] = Field(
        default_factory=list, description="questions that need to be answered"
    )

    @computed_field
    @property
    def stop_evaluation(self) -> bool:
        """Return whether the evaluation should stop."""
        return self.has_sufficient_findings and self.has_necessary_findings


class ReportConclusion(BaseModel):
    """Class describing the conclusions of the report.

    Design a report conclusion based on the following rules:
    - The conclusion must relate to the report and respond to the user query.
    - The conclusion must be a step-by-step analysis.
    -- All the fields are composed to generate the conclusion.

    """

    report_id: int = Field(description="unique id of the report")

    chain_of_thought: str = Field(description="think step by step about the report")

    recap: str = Field(description="summary of the report")
    interpretation: str = Field(description="interpretation of the experiments")

    introduction: str = Field(description="introduction to the conclusions")
    evaluation_process: str = Field(description="evaluation process")
    evaluation_recap: str = Field(description="recap of the evaluation")
    evaluation_interpretation: str = Field(description="interpretation of the evaluation")

    @computed_field
    @property
    def conclusions(self) -> str:
        """Return the conclusions of the report."""
        return " ".join(
            [
                self.introduction,
                self.evaluation_process,
                self.evaluation_recap,
                self.evaluation_interpretation,
            ]
        ).strip()
