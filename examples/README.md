Examples
==========

We provide two examples to enable you to familiarize yourself with the tracking pipeline.
If anything remained unclear after going through these examples (or you encountered any bugs), just open a github issue and we will get back to you!


Configuration
---------------

We use [toml](https://toml.io) configuration files.
`toml` has implementations available for many languages and provides a simple, easy-to-read configuration file format.
The format is mostly self-explanatory, if something is unclear please refer to the website.
The content is mapped more or less directly to a python dictionary.
There are key/value pairs, nested structures, etc.
Two things to note:
 - `[[key]]` starts a new element of a list (typically a list of dictionaries, see [an example](example_basic/config_parameters.toml)). The first use of `[[key]]` additionally marks the start of a new list. The element ends when `[[key]]` is used again to define the next element or a different structure is defined (which also equals the end of the list)
 - The configuration for a *linajea* experiment can be modular or monolithic.
   - In the monolithic case all values are defined in a single file (see [example](example_advanced/config.toml)).
   - In the modular case different sections of the configuration can be defined in separate files. Only the sections necessary for the step of the pipeline you want to run are required. However, as the code expects a single file, create a small aggregate *toml* file that only has to include the sections you want:
 Instead of
        ```
        [model]
        input = [...]
        upsampling = [...]
        [data]
        file = [...]
        roi = [...]
        ...
        ```
        you can use separate files:

        *model.toml*
        ```
        input = [...]
        upsampling = [...]
        ```
        *data.toml*
        ```
        file = [...]
        roi = [...]
        ```
        and *config.toml*
        ```
        model = 'model.toml'
        data = 'data.toml'
        ```
        This is **not** a native `toml` functionality.
        The contents of the individual files are only imported when using our *config* data classes (see the separate config files in [the basic example](example_basic))

For detailed information on the composition of the content of the configuration files checkout the different examples and see the [documentation/docstrings](../linajea/config).


Example 1: Basic
------------------

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

Example 2: Advanced
---------------------

In this example we use some additional, advanced functionality.
Each data set (train, validation, test) can contain multiple sample volumes.

The code automatically uses all samples that are provided in the configuration file.
Depending on the executed step the appropriate data is selected (e.g. validation or test).

For this example the configuration is contained in a single monolithic file (though this is not mandatory).
The tracking parameters for the grid search on the validation data are computed on-the-fly.
The validation is run automatically on all provided samples and all desired model checkpoints.

For more information open `run_advanced.ipynb` in jupyter and get started!
