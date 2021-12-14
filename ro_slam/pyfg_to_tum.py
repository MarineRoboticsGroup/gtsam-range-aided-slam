if __name__ == "__main__":
    import argparse
    from os.path import join
    from py_factor_graph.parse_factor_graph import (
        parse_efg_file,
        parse_pickle_file,
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
    if fg_filepath.endswith(".pickle"):
        fg = parse_pickle_file(fg_filepath)
    elif fg_filepath.endswith(".fg"):
        fg = parse_efg_file(fg_filepath)
    else:
        raise ValueError(f"Unknown file type: {fg_filepath}")
    print(f"Loaded data: {fg_filepath}")
    print(f"# Poses: {fg.num_poses}  # Landmarks: {len(fg.landmark_variables)}")

    fg.write_pose_gt_to_tum(args.results_dir)
