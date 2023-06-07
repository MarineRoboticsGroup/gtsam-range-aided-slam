if __name__ == "__main__":
    import argparse
    from os.path import join
    from py_factor_graph.parsing.parse_efg_file import parse_efg_file
    from py_factor_graph.parsing.parse_pickle_file import parse_pickle_file

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

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "data_dir", type=str, help="Path to the directory the PyFactorGraph is held in"
    )
    arg_parser.add_argument("pyfg_filename", type=str, help="name of the PyFactorGraph")
    arg_parser.add_argument(
        "results_dir", type=str, help="Path to the directory the results are saved to"
    )

    args = arg_parser.parse_args()

    fg_filepath = join(args.data_dir, args.pyfg_filename)
    if fg_filepath.endswith(".pickle") or fg_filepath.endswith(".pkl"):
        fg = parse_pickle_file(fg_filepath)
    elif fg_filepath.endswith(".fg"):
        fg = parse_efg_file(fg_filepath)
    else:
        raise ValueError(f"Unknown file type: {fg_filepath}")
    logger.info(f"Loaded data: {fg_filepath}")
    logger.info(f"# Poses: {fg.num_poses}  # Landmarks: {len(fg.landmark_variables)}")

    fg.write_pose_gt_to_tum(args.results_dir)
