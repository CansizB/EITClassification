# Assuming that the data has already been installed and converted to a CSV file as described in "EITRecon.m" file.
# That CSV file should be placed in the same folder as these code files.
# 
# Additionally, make sure the following dependencies are installed:
#
# - TensorFlow version should be 2.15:
#   pip install --upgrade tensorflow==2.15
#
# - The classification-models-3D library (version 1.0.10) should be imported correctly 
#   from the repository: https://github.com/ZFTurbo/classification_models_3D
#   pip install classification-models-3D==1.0.10


from workflow import FitWorkflow

if __name__ == "__main__":
  FitWorkflow(CSVname="EIT Image", mfile_path="Metadata.txt", EITname="FilePairs.txt", model_name="densenet201", nb_classes=5)
  # CSVname: the name of the converted CSV data.
  # mfile_path: path to the metadata file from the installed data.
  # EITname: path to the EIT file pairs from the installed data.
  # model_name: model definition, depends on the usage (e.g., 'densenet201').
  # nb_classes: number of classes, which depends on the experiment.

