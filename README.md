# label_training

Train a small CNN on a dataset built with ImSpec Labeler, and run the trained model on new files.

These scripts are separate from the web app. They read the files that the app's **Data → Training dataset →
Build dataset** writes into `labeled/training_data/`. How those files are made is described in
[docs/training_data.md](../../docs/training_data.md).

- [What you need](#what-you-need)
- [1. Build the dataset in the app](#1-build-the-dataset-in-the-app)
- [2. Train: main.py](#2-train-mainpy)
- [3. Predict on new files: infer.py](#3-predict-on-new-files-inferpy)
- [Step by step: the notebook](#step-by-step-the-notebook)
- [The network](#the-network)
- [model_config.txt](#model_configtxt)
- [The scripts](#the-scripts)
- [Limits](#limits)

## What you need

| For | Needed |
| --- | --- |
| Training (`main.py`) | Python with `numpy` and `torch`. Any Python that has them works. |
| Inference (`infer.py`) | The same, plus the **SciFiReaders fork** the app uses, and the app's `py/` folder. In practice: run it with the repository's `.venv`. |

The original SciFiReaders from PyPI reads Nanonis files differently and does not work for inference. If you run
`infer.py` with a Python that has it, the script stops and says so.

A GPU is used automatically when PyTorch sees one (`--device auto`). On a machine without one, everything runs on
the CPU.

## 1. Build the dataset in the app

In the app, under **Data → Training dataset**:

1. Set `"normalize": "zscore"` in the settings box (it is the default).
2. Tick **NaN to zero** and **Add NaN mask**.
3. If your images have different sizes, set `"image": {"size": [height, width]}`.
4. Press **Build dataset**.

The training code checks the first two points and stops with a message if they were not used.

A dataset is the group of files with one prefix, `<mode>_<kind>`, for example `binary_image`:

| File | Used for |
| --- | --- |
| `binary_image_info.json` | mode, kind of data, shapes, what the class numbers mean, how samples were resized |
| `binary_image_normalize.json` | how the arrays were normalized |
| `binary_image_input.npy` | the samples |
| `binary_image_output.npy` | the class numbers, or the scores |
| `binary_image_aux.npy` | scan size or bias range, mean and std of every sample (not needed with `--no-aux`) |
| `binary_image_meta.csv` | which channel, direction and flatten each sample was labeled with (optional, see below) |

## 2. Train: main.py

```bash
python main.py PATH/TO/labeled/training_data
```

What it does:

1. **Finds the dataset** in the folder. If a required file is missing, or a JSON file is damaged, it stops and
   names the file. If the folder holds several datasets, it lists them; pick one with `--set binary_image`.
2. **Works out the task** from `_info.json` and `_normalize.json`:

   | In the dataset | Becomes |
   | --- | --- |
   | kind `image` or `rgb` | a 2D network |
   | kind `spectrum` | a 1D network |
   | mode `binary` or `multiclass` | classification (cross-entropy loss) |
   | mode `score` | regression on the z-scored score (mean squared error) |
   | the shape of a sample | the number of input channels |
   | the aux columns | the size of the aux branch |

3. **Splits** the samples 80 : 20 with seed 42. For classification the split is stratified: every class keeps
   about 20 % of its samples for validation.
4. **Trains** for up to 100 epochs with the Adam optimizer. After every epoch it checks the validation loss. When
   the loss has not dropped by more than `--min-delta` for `--patience` epochs in a row, it stops. The weights of
   the best epoch are the ones that are kept.
5. **Saves** into `<folder>/model/`:

   | File | Content |
   | --- | --- |
   | `model.pt` | the best weights, together with the configuration |
   | `model_config.txt` | the configuration, readable (see [below](#model_configtxt)) |
   | `history.csv` | the losses (and accuracies) of every epoch |

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--set NAME` | | which dataset, when the folder holds several |
| `--epochs` | `100` | maximum number of epochs |
| `--patience` | `10` | epochs without improvement before stopping |
| `--min-delta` | `0.0001` | how much the validation loss must drop to count as an improvement |
| `--batch-size` | `16` | |
| `--lr` | `0.001` | learning rate |
| `--weight-decay` | `0.0001` | |
| `--val-fraction` | `0.2` | share of the samples kept for validation |
| `--seed` | `42` | for the split and the weights |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:0` ... |
| `--out DIR` | `<folder>/model` | where to save |
| `--no-aux` | | do not use the aux values |
| `--no-class-weights` | | do not weight the classes by how rare they are |
| `--config FILE` | | reuse the network settings of an earlier `model_config.txt` |
| `--weights FILE` | | start from the weights of an earlier `model.pt` |

Retraining after you have labeled more files and built the dataset again:

```bash
python main.py PATH/TO/training_data --config PATH/TO/model/model_config.txt --weights PATH/TO/model/model.pt --out PATH/TO/model_v2
```

`--config` only works if the new data has the same kind, number of channels, aux columns and classes.

## 3. Predict on new files: infer.py

Run it with the Python that has the SciFiReaders fork:

```bash
.venv\Scripts\python to_ignore\label_training\infer.py MODEL_DIR FILE_OR_FOLDER --csv predictions.csv
```

`MODEL_DIR` is the folder written by `main.py`. The inputs are raw files (`.sxm`, `.dat`, `.3ds`, jpg, png,
tif ...). A folder means every supported file directly inside it. You can give several files and folders.

Each file goes through the same steps as a training sample, all taken from `model_config.txt`:

1. Read the file and take the channel, scan direction, flatten (and slice) that the samples were labeled with.
2. Measure the sample: its own mean and std, and its scan size or bias range. These are the aux values.
3. Resize or resample as the dataset was built.
4. Normalize the input with the sample's own mean and std, set NaN to zero, add the NaN mask.
5. Normalize every aux value with the dataset's mean and std, after a log where the dataset used one.
6. Run the model. A score is converted back to your original scale.

Reading and transforming use the app's own code (`py/reader.py` and `py/dataset.py`), so a new file is prepared
exactly like a labeled one.

Output: one line per file, and a table with `--csv`.

| Task | Columns |
| --- | --- |
| classification | `file, channel, prediction, class_id, probability, p_<label>..., notes` |
| score | `file, channel, score, notes` |

`notes` says when a file was skipped (for example, it does not have the channel), or when one of its aux values
is far from anything in the training data. In that second case, treat the prediction with care.

Options:

| Option | Meaning |
| --- | --- |
| `--channel NAME` | use this channel instead of the one the samples were labeled with |
| `--direction forward\|backward` | scan direction |
| `--flatten none\|offset\|line\|plane` | flatten |
| `--slice N` | slice of a `.3ds` grid or a tif stack |
| `--csv FILE` | also write the predictions to a csv file |
| `--device` | `auto`, `cpu`, `cuda` ... |
| `--app-py DIR` | where the app's `py/` folder is, if these scripts were moved (default: `../../py`) |

Where the channel comes from: the built dataset records the channel, direction and flatten only per sample, in
`_meta.csv`. `main.py` reads them and stores the most common combination in `model_config.txt`. If the samples
were not all labeled the same way, both scripts say so, and inference uses the most common one. If there was no
`_meta.csv` at training time, give `--channel` yourself.

## Step by step: the notebook

`train_inspect.ipynb` does what `main.py` and `infer.py` do, one step per cell, so you can look at the data and
change the settings in between. All settings are in one cell at the top.

| Section | What you see |
| --- | --- |
| 1. Get the data and inspect it | a summary of the dataset; samples per class or the spread of the scores; a few samples as the model sees them, with their NaN masks; the value range of every channel; the aux values per class |
| 2. Train / validation split | the size of both parts, the validation files, the classes in each part |
| 3. Create the model | the network settings (a dict you can edit) and the layers |
| 4. Train | the loss of every epoch; loss and accuracy curves with the best epoch marked; the confusion matrix or predicted against true scores; the validation samples one by one, wrong ones first |
| 5. Save the model | `model.pt`, `model_config.txt`, `history.csv` |
| 6. Reload the model | the saved model loaded again, and a check that it gives the same outputs |
| 7. Inference on new files | predictions for the raw files in `INFER_FILES`, with the prepared input of each drawn |

Open it from this folder, so the scripts next to it can be imported. Sections 1 to 6 need `numpy`, `torch` and
`matplotlib`. Section 7 reads raw files, so pick the kernel of the repository's `.venv` for it.

## The network

One class, `LabelCNN` in `model.py`, covers both kinds of data.

```
input  (n, C, H, W) for images    or    (n, C, P) for spectra
  -> [ convolution -> batch norm -> ReLU -> max pooling ]  x 3     channels 16, 32, 64
  -> global average pooling                                        so any input size works
  -> joined with the aux branch (linear 16 -> ReLU)                if the dataset has aux values
  -> linear 64 -> ReLU -> dropout 0.3 -> linear                    class scores, or one number
```

Input channels, as the app builds them with the NaN mask:

| Kind | Channels |
| --- | --- |
| image | data, mask |
| spectrum without backward sweeps | bias, forward, mask of forward |
| spectrum with backward sweeps | bias, forward, backward, mask of forward, mask of backward |
| colour photo | red, green, blue, mask |

- Images use 3 x 3 kernels, spectra use kernels of 5 points.
- A very small input gets fewer blocks, so that pooling does not shrink it to nothing.
- The bias row of spectra is the one channel the app leaves in volts. It is brought to mean 0 and std 1 with the
  training samples' statistics, which are stored in `model_config.txt` under `preprocessing`.
- For classification, classes are weighted by how rare they are in the training split.

## model_config.txt

It is JSON text. The first entry, `description`, is a few plain sentences. The sections:

| Section | Content |
| --- | --- |
| `data` | the folder and dataset, number of samples, shapes, channel names, aux columns, how samples were resized |
| `source` | the channel, direction, flatten and slice the samples were labeled with |
| `normalize` | the dataset's whole `normalize.json` |
| `preprocessing` | the bias scaling for spectra |
| `task` | classification or regression; the classes with their index, class id and label; the score mean and std |
| `model` | everything that defines the network |
| `training` | epochs run, best epoch, patience, batch size, learning rate, loss, class weights, seed, the indices of the training and validation samples, device |
| `results` | the best validation loss; loss, accuracy and confusion matrix (or score errors) on both splits |

With `source` and `normalize` inside, the model folder alone is enough for `infer.py`.

Loading a model in your own code:

```python
import training_fns as tf
model, config = tf.load_model("PATH/TO/model/model.pt")   # the network is rebuilt from the config
```

## The scripts

| Script | Job |
| --- | --- |
| `main.py` | the one you run for training |
| `infer.py` | the one you run on new files |
| `model.py` | the network, and its default settings for a kind of data |
| `training_fns.py` | device choice, data loaders, the training loop with early stopping, evaluation, saving and loading |
| `data_io.py` | finding, checking and loading a dataset; the train / validation split |
| `plots.py` | the figures of the notebook |
| `train_inspect.ipynb` | training and inference step by step |

## Limits

- **All data is loaded into memory.** That is fine for thousands of 256 x 256 images. Much larger sets would need
  a loader that reads from disk.
- **One labeling setup per model.** A model is tied to the channel, flatten and sizes it was trained with.
- **Validation is a single 80 : 20 split.** With few samples the validation numbers are noisy.
- **Small datasets overfit.** The network is deliberately small, but a few dozen samples are still few.
- **GPU runs are untested here.** The code moves the model and the data to the chosen device; it was only run on
  a CPU so far.
- **Retrain models made before `infer.py` existed.** They lack the `source` and `normalize` sections.
