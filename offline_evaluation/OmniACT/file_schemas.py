from pydantic import BaseModel, conlist
from typing import List

class PlanJsonlModel(BaseModel):
    """
    Model to validate `plan_jsonl` file entries.

    Fields:
    - id: Unique identifier of the task.
    - task: Relative path to the task file.
    - image: Relative path to the image file.
    - box: Relative path to the box (bounding box) file.
    - gpt_output: The GPT-generated output script.
    - example_ids: List of example task IDs used for in-context learning.
    """
    id: str
    task: str
    image: str
    box: str
    gpt_output: str
    example_ids: conlist(str)

class SeqJsonlModel(BaseModel):
    """
    Model to validate `seq_jsonl` file entries.

    Fields:
    - id: Unique identifier of the task.
    - task: Relative path to the task file.
    - image: Relative path to the image file.
    - box: Relative path to the box (bounding box) file.
    - gpt_output: The GPT-generated output script.
    - example_ids: List of example task IDs used for in-context learning.
    - ideal_score: The ideal score for the task.
    - seq_score: The sequence match score for the task.
    """
    id: str
    task: str
    image: str
    box: str
    gpt_output: str
    example_ids: conlist(str)
    ideal_score: float
    seq_score: float

class QueryJsonlModel(BaseModel):
    """
    Model to validate `query_jsonl` file entries.

    Fields:
    - id: Unique identifier of the task.
    - image: Relative path to the image file.
    - description: Element description that GPT should operate on.
    - scale: Scale factor to adjust the element coordinates for UGround.
    """
    id: str
    image: str
    description: str
    scale: float

class AnsJsonlModel(BaseModel):
    """
    Model to validate `ans_jsonl` file entries.

    Fields:
    - id: Unique identifier of the task.
    - image: Relative path to the image file.
    - description: Element description from the query.
    - scale: Scale factor to adjust the element coordinates for UGround.
    - output: The output coordinates predicted by the grounding model.
    """
    id: str
    image: str
    description: str
    scale: float
    output: str