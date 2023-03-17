from py_factor_graph.utils.solver_utils import save_to_tum
import pickle
import sys
import os
from typing import List, Tuple
import logging, coloredlogs

logger = logging.getLogger(__name__)
field_styles = {
    "filename": {"color": "green"},
    "filename": {"color": "green"},
    "levelname": {"bold": True, "color": "black"},
    "name": {"color": "blue"},
}
coloredlogs.install(
    level="INFO",
    fmt="[%(filename)s:%(lineno)d] %(name)s %(levelname)s - %(message)s",
    field_styles=field_styles,
)


def recursively_find_result_files(dir) -> List[Tuple[str, str]]:
    """Recursively finds all .pickle files in the directory and its subdirectories

    Args:
        dir (str): the directory to search in

    Returns:
        List[Tuple[str, str]]: a list of tuples of the form (root, file_name)

    """

    pickle_files = []
    for root, _, files in os.walk(dir):
        for file in files:
            assert isinstance(file, str)
            if file.endswith("gtsam_result.pickle") or file.endswith(
                "score_result.pickle"
            ):
                pickle_files.append((root, file))

    return pickle_files


if __name__ == "__main__":
    """This file is meant to be run on a directory in which there are gtsam
    result files either in the root or in subdirectories.

    "Usage: python3 solution_to_tum.py <path_to_results>"
    """
    import argparse
    from os.path import join

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "data_dir",
        type=str,
        help="Path to the directory all of the results are held in",
    )
    args = arg_parser.parse_args()

    result_files = recursively_find_result_files(args.data_dir)

    for gtsam_result_file in result_files:
        result_filepath = join(gtsam_result_file[0], gtsam_result_file[1])
        with open(result_filepath, "rb") as f:
            gtsam_result = pickle.load(f)

            save_to_tum(gtsam_result, result_filepath, strip_extension=True)
            logger.info("Saved TUM files to {}".format(result_filepath))
