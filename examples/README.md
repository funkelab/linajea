Examples
==========

We provide two examples to enable you to familiarize yourself with the tracking pipeline.
If anything remained unclear after going through these examples (or you encountered any bugs), just open a github issue and we will get back to you!


Basic
------

In this example we showcase the basic setup.
We train on a single sample volume, validate on another one and finally evaluate the performance on a third one.

The configuration used to run the code is split into multiple smaller files:
 - config\_model.toml: Definition of the U-Net model
 - config\_train.toml: Definition of the training parameters and the training data
 - config\_val.toml: Definition of the postprocessing parameters used for validation and the validation data
 - config\_test.toml: Definition of the postprocessing parameters used for testing and the test data
 - config\_parameters.toml: Definition of a list of tracking parameters, the best one on the validation data ("grid search") will be used to evaluate on the test data

When running the steps of the pipeline the correct configuration file has to be chosen manually (and some values cannot be determined automatically, such as the name of the database to be used).

For more information open `run_basic.ipynb` in jupyter and get started!

Advanced
---------

In this example we use some additional, advanced functionality.
Each data set (train, validation, test) can contain multiple sample volumes.

The code automatically uses all samples that are provided in the configuration file.
Depending on the executed step the appropriate data is selected (e.g. validation or test).

For this example the configuration is contained in a single monolithic file (though this is not mandatory).
The tracking parameters for the grid search on the validation data are computed on-the-fly.
The validation is run automatically on all provided samples and all desired model checkpoints.

For more information open `run_advanced.ipynb` in jupyter and get started!
