run scripts
=============

Aside from the *jupyter* notebooks showcased as [examples](../examples) we provide these scripts to be able to run each step directly from the command line by just providing an appropriate configuration file

 - `01_train.py`
   Run the *training* process. The configuration file determines, among other things, what data should be used for training (*train_data*), what kind of model should be used (*model*) and for how long it should be trained (*train*) and with what kind of augmentations (*train.augment*).
 - `02_predict.py`
   Run the *prediction* process. The configuration file determines which previously trained model should be used for prediction and what data should be used (*validate_data*, *test_data*).
 - `03_extract_edges.py`
   Run the *extract edges* process (For each detected cell candidate look in the spatial neighborhood of the previous time frame to find potential parent cell candidates). The configuration file determines how far a cell could have moved at most to count as a potential link.
 - `04_solve.py`
   Run the *solve* process (tracking). The configuration file determines what ILP weights should be used to solve the ILP, and if a grid search should be executed to find optimal weights.
 - `05_evaluate.py`
   Run the *evaluation* process. The configuration file determines how close detected cell and ground truth annotation have to be to be counted as a correct match and what ROI (region of interest) of the provided data we want to evaluate on.
 - `06_run_best_config.py`
   Run *solve* and *evaluate* on the test data using the best parameters/weights as determined on the validation data.

For further convenience we provide a final script that, depending on the given flags, calls the appropriate script with the correct arguments: `linajea` (should be called from within an experiment directory)

```
mkdir $setup_dir
cd $setup_dir
linajea --config config.toml --train
```

For information on the available flags use `python linajea --help`:
```
usage: linajea [-h] [--config CONFIG] [--checkpoint CHECKPOINT]
               [--train] [--predict] [--extract_edges] [--solve] [--evaluate] [--best]
               [--validation] [--validate_on_train]
               [--param_id PARAM_ID] [--val_param_id VAL_PARAM_ID] [--param_ids PARAM_IDS [PARAM_IDS ...]]
               [--local] [--slurm] [--gridengine] [--interactive]
               [--array_job] [--eval_array_job] [--wait_job_id WAIT_JOB_ID] [--no_block_after_eval]
```
