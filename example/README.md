## Workflow of the project

### 1. Data Generation Using Finite Difference Method

The data is generated using the finite difference method, whereas the finite 
difference source code cannot be open sourced. This steps is done by the running 
the Jupiter notebook `generate_data_constant.ipynb` in terminal as follows:

```bash
bash 01_run_generate_data.sh
```

Though the source code cannot be open sourced, the data is provided in the
`data` folder. Please download the data from the shared google drive.

### 2. Train the model

The training code is provided in the `train.py` file. The training can be run
by the following command:

```bash
bash 02_run_train.sh
```

The trained model is saved in the specified folder in the `02_run_train.sh` file.
Typically, the model is saved in the `model` folder of this project.

Note
- The hyperparameters of the model can be changed in the `train.py` file.
- The training process saves the data used for training and validation in the
  `data` folder. The test data is also saved for later testing purpose.

### 3. Test the model
After the model is trained, the model can be tested by running the following
command:

```bash
bash 03_run_test.sh
```

The test result can be visualized inside the Jupiter notebook `test.ipynb`.