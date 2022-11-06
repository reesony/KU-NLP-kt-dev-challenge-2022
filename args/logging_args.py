from typing import Optional
from dataclasses import dataclass, field


@dataclass
class LoggingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dotenv_path: Optional[str] = field(
        default='./wandb.env',
        metadata={"help":'input your dotenv path'},
    )

    project_name: Optional[str] = field(
        default="ai-competition",
        metadata={"help": "project name"},
    )