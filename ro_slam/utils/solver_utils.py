from typing import Optional
from os.path import isfile
import attr
import logging

logger = logging.getLogger(__name__)


@attr.s(frozen=True)
class GtsamSolverParams:
    verbose: bool = attr.ib()
    save_results: bool = attr.ib()
    init_technique: str = attr.ib()
    custom_init_file: Optional[str] = attr.ib(default=None)
    init_translation_perturbation: Optional[float] = attr.ib(default=None)
    init_rotation_perturbation: Optional[float] = attr.ib(default=None)

    @init_technique.validator
    def _check_init_technique(self, attribute, value):
        init_options = ["gt", "compose", "random", "custom"]
        if value not in init_options:
            raise ValueError(
                f"init_technique must be one of {init_options}, not {value}"
            )

    @custom_init_file.validator
    def _check_custom_init_file(self, attribute, value):
        if value is not None:
            if not isfile(value):
                raise ValueError(f"custom_init_file {value} does not exist")
