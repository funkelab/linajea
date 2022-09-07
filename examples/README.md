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


Data
-----

### Image Data

The image data has to be in the [zarr](https://zarr.readthedocs.io/en/stable/) format.
It supports N-dimensional arrays and concurrent read and write access.
Multiple arrays can be stored in a single container.

It is already installed as part of the `linajea` dependencies, but if you want to use it separately:
```
pip install zarr
pip install numcodecs
```

You can use both the default storage format and the zip storage format.
`zarr` supports storing additional attributes as key-value pairs per array (internally stored as JSON).

Each array in a container has a name.
For use in `linajea` when defining a data source in the configuration file you have to supply both, the file and the name of the array:
```
filename = "sample.zarr"
array = "raw"
```

Through the use of the `voxel_size` parameter both isotropic and anisotropic data is supported.
```
voxel_size = [1, 5, 1, 1]
```


To create a container you can either use the `zarr` package directly:
```
root = zarr.open('data/group.zarr', mode='w')
raw = root.zeros(raw', shape=(10000, 10000), chunks=(1000, 1000), dtype='i4')
raw[:] = arr
raw.attrs['foo'] = 'bar'
```
or (compatible with hdf/`h5py`):
```
root.create_dataset('raw, data=arr, shape=(10000, 10000), chunks=(1000, 1000), dtype='i4')
```
Please refer to their [documentation](https://zarr.readthedocs.io/en/stable/) for more information.

Alternatively you can use [daisy](https://github.com/funkelab/daisy) functionality.
To access an existing file:
```
raw = daisy.open_ds('data/group.zarr', 'raw', 'r')
attrs = raw.data.attrs
```

or to create a new (empty) one:
```
raw = daisy.prepare_ds(
  'data/group.zarr, 'raw', roi, voxel_size, dtype)
```

### Tracks Data

The tracks data is stored in text/csv files, e.g.:
```
t       z       y       x       cell_id parent_id  track_id  radius  name
0       95.0    215.0   255.0   0       -1         0         13.5    AB
0       95.0    306.0   365.0   1       -1         1         16.5    P1
1       105.0   228.0   225.0   2       0          0         14.5    ABa
1       110.0   211.0   272.0   3       0          0         15.5    ABp
1       90.0    300.0   369.0   5       1          1         13.5    P2
1       95.0    304.0   345.0   4       1          1         13.5    EMS
2       105.0   226.0   220.0   6       2          0         15.5    ABa
2       100.0   212.0   274.0   7       3          0         15.5    ABp
2       100.0   309.0   329.0   8       4          1         13.5    EMS
2       95.0    313.0   377.0   9       5          1         13.5    P2
```

`t` (time point, frame) starts at `0`.\
`z`, `y` and `x` are in world units (important for anisotropic data, dividing the coordinate given in the tracks file by the voxel size gives you the array index; all of this is handled automatically, if `voxel_size` is set correctly).\
`cell_id` is an ID identifying each cell snapshot (cell per frame).\
`parent_id` points to the `cell_id` of the same cell in the previous frame (or its mother cell after a division).\
`track_id` identifies each track, each cell originating from the same cell has the same `track_id`, even after divisions (if you are tracking a developing embryo, starting at a single cell, all cells have the same `track_id`).\
Other columns are optional. If you know the (approximate) radius for each cell, this information can be used during training to fit the size of the cell mask used for the cell indicator map and the movement vectors better.


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
