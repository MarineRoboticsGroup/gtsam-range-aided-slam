from ro_slam.utils.solver_utils import save_to_tum
import pickle
import sys

if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python3 solution_to_tum.py <path_to_solution> <path_to_tum_file>"
    path_to_solution = sys.argv[1]
    path_to_tum = sys.argv[2]

    with open(path_to_solution, "rb") as f:
        solution = pickle.load(f)
        save_to_tum(solution, path_to_tum, strip_extension=True)
        print("Saved TUM files to {}".format(path_to_tum))