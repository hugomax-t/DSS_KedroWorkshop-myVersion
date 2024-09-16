# ⚠️⚠️⚠️ THIS BRANCH CONTAINS THE SOLUTION. TRY TO COMPLETE THE WORKSHOP ON YOUR OWN BEFORE LOOKING AT THE SOLUTION! ⚠️⚠️⚠️

# Data Science Summit 2024 - Kedro Workshop

By : Pierre Adeikalam (AXA Direct France / Direct Assurance)

Supported by : Juan Pablo Usuga, Benjamin Wallyn

## What is Kedro?

> Kedro is an open source library that provides a structured and easy-to-follow framework for developing ML and Data Engineering pipelines. It was built to help Data Scientists, Analysts start writing production ready code for reproducible, maintainable and modular data pipelines.
>
>
> Kedro standardizes the project structure by providing a repository template, simplifies the ingestion and writing of data through a data catalog, and streamlines the process of creating data pipelines, as you will see in this workshop.

## Project Setup

### Install required dependencies

> Create a virtual environment. For example using `pyenv` and `venv`:
>    
> > `pyenv install 3.10`
>
>
> > `pyenv local 3.10` 
>
> > `python -m venv envs\kedro_env`
>
> Activate the environment using the following command:
>
> > `envs\kedro_env\Scripts\activate`
>
> Install the project requirements:
> > `pip install -r requirements.txt`
>
> Verify that the project requirements were installed by lauching the following command:
>
> > `kedro info` 

### Create a Kedro Project

> To create a Kedro Project, use the following command:
> > `kedro new`
>
> * Input a project name. This name will be used to create the repository containing the project.
>
> * Include the Data Folder and Kedro-Viz tools by inputing the option `5,7` when prompted.
>
> * Do not include an example pipeline when prompted.
>
> * Open the repository that you just created and paste the provided notebooks in the notebooks folder.
>
> * Launch a jupyter notebook server and explore the contents of the provided notebooks.

## Getting Started

### Kedro pipelines

> The `src` directory will contain all the source code that should be delivered at the end of our Data Science project. Usually, when we try to create packages as Data Scientists, it quickly becomes a problematic mess and thus adds a lot of friction to potential collaboration. To avoid this problem, we will let Kedro handle the package structure for us.
>
>
> Source code in Kedro is structured as pipelines. Kedro will create a new folder for each pipeline we will need. We will want to create multiple pipelines to keep each step of our project seperated, which will make collaboration easier. Let's create our first pipeline.
>
>
> Run the following command in your terminal to create a Kedro pipeline **(Make sure you are inside your project root directory and not in the `src` folder or somewhere else)**:
> > `kedro pipeline create my_first_pipeline`
>
> Kedro will automatically create a folder named `my_first_pipeline` in the src folder. This folder contains two important files:
> * The `nodes.py` file: where you will write the functions that define the nodes (a.k.a. the steps) of your pipeline.
> * The `pipeline.py` file: where you will be defining the inputs and outputs of the nodes and the order you want them to be executed.
>
> In the `nodes.py` file, paste the following code:
>
> ```py
> import numpy as np
> import pandas as pd
> 
> def my_first_node(input_df: pd.DataFrame) -> pd.DataFrame:
>     """
>     Return the first 5 rows of the input DataFrame.
> 
>     input_df : pd.DataFrame
>     """
>
>     return input_df.head()
> ```
>
> In the `pipeline.py` file, you will see that a function called `create_pipeline` is already defined. This function is what defines the pipeline so that it can be packaged and run later on. Replace the contents of this file with the following code:
>
> ```py
> from kedro.pipeline import Pipeline, node, pipeline
> 
> from .nodes import my_first_node
>
> def create_pipeline(**kwargs) -> Pipeline:
>     return pipeline(
>         [
>             node(
>                 my_first_node,
>                 inputs="train_df",
>                 outputs="train_df_head",
>                 name="sample_train_df",
>             )
>         ]
>     )
> ```
>
> what is the meaning of this code you ask?
>
> Kedro pipelines are defined as list of nodes where each node needs to be given a function to execute, an input, an output and a name. 
>
> We set the input as `"train_df"` and the output as `"train_df_head"`, but they do not exist yet. For the pipeline to know what it is supposed to work with, we need to at least insert the `"train_df"` entry into the Data Catalog.

### The Data Catalog

> The data catalog is a YAML file that was automatically created at the `conf/base/catalog.yml` filepath.
>
>
> This file is extremely important as it will keep track of all the files that you will read and write in your project. This means that all the data we will need and all the artifacts we will create will have to be referenced in this catalog. This file provides a lot of transparency to a Data Science project and makes it easier to collaborate and modify our data pipelines later on.
>
> 
> Kedro can access a multitude of data sources. To keep it easy for this workshop, we will be working with CSV files. For now, copy the `train.csv` file in the `data/01_raw/` directory.
>
>
> Now, open the `conf/base/catalog.yml` file and paste the following entries:
>
>
> ```yml
> train_df:
>   type: pandas.CSVDataset
>   filepath: data/01_raw/train.csv
> 
> train_df_head:
>   type: pandas.CSVDataset
>   filepath: data/01_raw/train_head.csv
>   save_args:
>     index: False
> ```
>
>
> What we did here is define a new source of data (`"train_df"`) and a new artefact (`"train_df_head"`) in the Kedro namespace. This means that the pipeline will now know what we refer to when we add `"train_df"` and `"train_df_head"` as inputs and outputs to the nodes in the pipeline.
>
>
> For each entry, we defined a dataset `type`. This tells Kedro in which format the data should be read or written as, so that we don't have to worry about it anymore. Here, we used the `pandas.CSVDataset` type so that we end up with a `pandas` DataFrame as input to our node whenever the data will be loaded by Kedro.
>
>
> For node outputs, we do not need to write them systematically in the catalog. We only add them if we want to save them. Here, we added the entry  "`train_df_head`" so that Kedro saves it as a CSV file. We also added the parameter `"save_args"` to provide Kedro additional arguments to pass to the `.to_csv()` method whenever Kedro will write the file.
>
>
> You can find the list of connectors you can use with Kedro here : https://docs.kedro.org/projects/kedro-datasets/en/kedro-datasets-2.0.0.post1/api/kedro_datasets.html

### Kedro Viz

> Kedro Viz is a powerful tool to visualize your pipelines. This is very helpful for collaboration as it lays out the contents of your projects in a neat way, and the visualisation of your code is generated automatically.
>
>
> In order to visualize your pipeline, run the following command in your terminal:
>
> > `kedro viz run`

### Running your pipeline

> In order to run your entire Kedro project, run the following command in your terminal: 
>
> > `kedro run`

## Exercise

> You have been provided with 3 notebooks that contain the code for a Data Science project.
> Your objective is to refactor the code contained in these notebooks into a nice Kedro project so that the same artifacts are generated.
>
> To achieve this, here are some tips:
> * Create 3 Kedro pipelines, each corresponding to a notebook (Data Preparation, Model Training, Model Evaluation).
> * Add the required entries for inputs and artifacts to the Data Catalog. Use the following Kedro documentation to use the proper connectors for saving pickle files and matplotlib plots : https://docs.kedro.org/projects/kedro-datasets/en/kedro-datasets-2.0.0.post1/api/kedro_datasets.html
> *  Regularly visualise and run your pipeline using the `kedro run` and `kedro viz run` commands.

## Bonus Exercise

> Parametrize your model training pipeline so that you can define the hyperparameter ranges for tuning in the corresponding YAML file. You will find help with setting up pipeline parameters with the following documentation : https://docs.kedro.org/en/0.19.5/configuration/parameters.html

### Bonus Bonus exercise

> Use Kedro's pipeline modularity to create multiple versions of your model training pipeline, each training a different type of model.
