import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import shutil

def execute_notebook(notebook_path):
    with open(notebook_path) as ff:
        nb_in = nbformat.read(ff, nbformat.NO_CONVERT)
    
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    nb_out = ep.preprocess(nb_in)

    return nb_out


if __name__ == "__main__":
    filename = 'compare_mean.ipynb'
    execute_experiments = [
        "0213_ict_sh_nblock_3",
        "0220_iCT_512_alignment_loss_from_300k",
        "0220_ict_sh_nblock_2"
    ]
    for experiment in execute_experiments:
        config_file = f"experiments/{experiment}/config.yaml"
        shutil.copyfile(config_file, "configs/config_consistency.yaml")
        result = execute_notebook(filename)
        breakpoint()