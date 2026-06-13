# AISI SWE Coding Challenge

This repository contains research code for evaluating language models, specifically focusing on question-answering tasks.

This coding assessment is about **re-factoring** and **adding new features** to an existing codebase. This research code is a first version of a framework that will someday be deployed in production to perform the same tasks.

You should allow time for:

- Understanding the existing code, where it falls short, and where its current approach is appropriate.
- Adding well-chosen tests to ensure that you do not introduce regressions to critical parts of the code.
- Enhancing the framework by adding the new features listed under _Feature requests_.

You are free to use additional external Python libraries to complete any of the tasks. If you do so, please add them to `pyproject.toml`.

In the follow up interview you will be asked to screen-share and run your code and discuss any design considerations you made, as well as next steps for deploying and further improving your code.

You may choose to use an AI coding assistant to accomplish the task. However, you should be prepared to explain in-depth every line of code you've submitted. During the screen-share we might ask you to demonstrate how you use the assistant. We are looking for people who, if they use such tools to accelerate their productivity, do so thoughtfully and consciously (and not by blindly accepting their suggestions).

## Design Considerations

Please keep these considerations in mind while making design trade-offs in your implementation. You will not have time to prioritise all of these things while also completing the tasks, so **be prepared to talk about the choices you made, and where you would take the code next, in a follow up interview**.

- Keeping the framework easily adaptable to integrate and switch between different language models.
- Ensuring that all relevant parameters, model versions, and data versions are logged and can be easily referenced.
- Developing tests for your code to validate its functionality and to check for potential edge cases.
- Accounting for the fact that language models expose a variety of different features to their end users, and differ significantly from one another in the interfaces they provide. (You might like to consider: function-calling, multi-modal models, models which support fine-tuning, and self-hosted models)
- Allowing for different approaches to prompt-engineering.

Please spend no more than 3 hours working on the project. Extra credit will NOT be awarded for going beyond the suggested time - for example, with work that significantly exceeds the scope of the below tasks, complicates the solution, or introduces features and functionalities that are not requested in the descriptions of tasks below. **Our evaluation will focus on the quality, efficiency, and correctness of your implementation of the required features within the allotted time, as well as your decision-making in choosing which features to implement. We value concise, clean, and well-organized code over additional complexity.**

## Setup and install dependencies

This project uses [uv](https://docs.astral.sh/uv/) for package management

```sh
# Install uv (see docs for more options)
pip install uv

# Install python and dependencies
uv sync
```

For info, the main party libraries used in this project are:

- _Transformers_ by Hugging Face ([docs](https://huggingface.co/docs/transformers/index))
- _openai_ ([docs](https://github.com/openai/openai-python))
- _anthropic_ ([docs](https://github.com/anthropics/anthropic-sdk-python))

## Run the code

Sample data for testing the framework is included in the data/ directory.

Run the code on the test data by running:

```sh
# Run eval dataset against a mocked model with a mocked grader
uv run python main.py \
  --model mock-lm \
  --grader_model mock-lm \
  --eval_dataset eval_dataset.csv
```

This should create a relatively boring output.csv file in the root directory of the project.

## Feature requests

Please add as many of the following features to the code base as you can. Most people will complete one or two new features in the time. Each task can be implemented stand-alone, so if you get stuck feel free to move onto the next one and come back to it:

1. The models currently retry immediately if they fail. Please implement exponential backoff between retries.
2. The models currently make every API call sequentially, please rewrite the code to support sending requests for every prompt in the dataset concurrently or asynchronously. You can start by implementing this for a single model (eg OpenaiLM), then if you have time, adapt it to work for all appropriate models.
3. Some evaluation jobs may be long-running and need to be paused and resumed on the fly.
