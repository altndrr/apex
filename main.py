import random
import shutil
from collections import defaultdict
from contextlib import suppress
from pathlib import Path

import hydra
import numpy as np
import rootutils
import torch
from instructor.retry import InstructorRetryException
from omegaconf import DictConfig, open_dict
from pydantic import ValidationError
from rich import print

from src import utils
from src.data import ExperimentDataset, ImageDataset, get_dataset
from src.tools import reset_singletons
from utils import (
    Experiment,
    ExperimentDiscussion,
    Report,
    ReportConclusion,
    ReportDiscussion,
    chat_completion_to_list,
    evaluate_model,
    init_master,
)

log = utils.get_logger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="configs", config_name="main.yaml")
def main(cfg: DictConfig) -> None:
    """Evaluate the user query and generate a report via experiments.

    Args:
    ----
        cfg (DictConfig): Configuration composed by Hydra.

    """
    seed = None
    if cfg.get("seed"):
        seed = int(cfg.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    utils.extras(cfg)

    log.info("Loading the master model <%s>", cfg.master.name)
    master, history, context = init_master(cfg.master, seed)

    # initiate the report from the user query
    content = f"Initiate a report from the user query: {cfg.query.strip()}"
    history.append(dict(role="user", content=content.strip()))
    report, chat_completion = master(response_model=Report, validation_context=context)
    history.extend(chat_completion_to_list(chat_completion))
    log.info('User query: "%s"', report.query)
    log.info("Evaluating models: <%s>", ">, <".join(report.models_to_evaluate))
    print(report)

    # load the source dataset
    source_name, source_split = "imagenet-x", "val"
    source: ImageDataset = get_dataset(source_name, split=source_split)

    num_experiments = 1
    while num_experiments <= cfg.get("max_experiments", 9):
        dataset = None
        experiment, experiment_discussion, report_discussion = None, None, None
        experiment_dir = Path(cfg.paths.output_dir) / "experiments" / str(num_experiments)
        context["report"] = report.model_dump()

        # empty the history
        while len(history) > 1:
            history.pop()

        # design a new experiment
        try:
            content = f"Design a new experiment to test the report:\nReport: {report.model_dump()}"
            history.append(dict(role="user", content=content.strip()))
            experiment, chat_completion = master(
                response_model=Experiment, validation_context=context
            )
            history.extend(chat_completion_to_list(chat_completion))
            print(experiment)

            log.info('Question: "%s"', experiment.vqa_question)
            log.info("Answers: <%s>", ">, <".join([ans.text for ans in experiment.vqa_answers]))
            log.info("%s %s", experiment.recap, experiment.interpretation)
        except (ValidationError, InstructorRetryException) as e:
            log.warning("Error designing the experiment: %s", e)
            log.warning("Restarting the round.")
            continue

        # generate data
        try:
            dataset: ExperimentDataset = ExperimentDataset.from_dict(
                experiment_dir, source, cfg.samples_per_class, **experiment.model_dump()
            )
        except (ValueError, KeyError) as e:
            log.warning("Error generating the experiment dataset: %s", e)
            log.warning("Restarting the round.")
            if experiment_dir.exists():
                shutil.rmtree(experiment_dir)
            continue
        finally:
            reset_singletons()

        # evaluate models
        for model_name in report.models_to_evaluate:
            with open_dict(cfg):
                cfg.data.name = source_name
                cfg.data.split = source_split
                cfg.model.name = model_name

            log.info("Evaluating model <%s>", model_name)
            model_results = evaluate_model(model_name, experiment.vqa_question, dataset, cfg)
            experiment.results[model_name] = model_results

        # discuss the results
        content = f"Discuss the results of the experiment: {experiment.model_dump()}"
        history.append(dict(role="user", content=content.strip()))
        experiment_discussion, chat_completion = master(response_model=ExperimentDiscussion)
        log.info("%s %s", experiment_discussion.recap, experiment_discussion.interpretation)
        history.extend(chat_completion_to_list(chat_completion))

        # update the report
        experiment.findings = experiment_discussion.interpretation
        report.past_experiments.append(experiment.summary())

        # discuss the report
        content = "Discuss the current state of the report."
        content += f"\nReport: {report.model_dump()}"
        history.append(dict(role="user", content=content.strip()))
        report_discussion, chat_completion = master(
            response_model=ReportDiscussion, validation_context=context
        )
        log.info("%s %s", report_discussion.recap, report_discussion.interpretation)
        print(report_discussion)

        if report_discussion.stop_evaluation and num_experiments >= cfg.get("min_experiments", 3):
            history.extend(chat_completion_to_list(chat_completion))
            break

        # update the report
        report.open_questions = report_discussion.open_questions
        num_experiments += 1

        print(report.model_dump())

    # empty the next experiments list
    report.open_questions = []

    # empty the history
    while len(history) > 1:
        history.pop()

    content = "Draw conclusions from the report."
    content += f"\nReport: {report.model_dump()}"
    history.append(dict(role="user", content=content.strip()))
    report_conclusion, chat_completion = master(
        response_model=ReportConclusion, validation_context=context
    )

    # evaluate the average experiment accuracy for each model
    average_accuracy = defaultdict(float)
    for experiment in report.past_experiments:
        for model, results in experiment["results"].items():
            average_accuracy[model] += results["average_accuracy"] / num_experiments
    average_accuracy = {k: round(v, 3) for k, v in average_accuracy.items()}

    # update the report
    report.results = average_accuracy
    report.conclusions = report_conclusion.conclusions

    log.info("%s", report_conclusion.conclusions)
    log.info("Results: %s", report.results)

    print(report.model_dump())


if __name__ == "__main__":
    rootutils.setup_root(search_from=__file__, indicator="pyproject.toml", pythonpath=True)
    with suppress(KeyboardInterrupt):
        main()
