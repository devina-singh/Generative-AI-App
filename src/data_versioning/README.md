## Create a Data Store folder in GCS Bucket

- Go to `https://console.cloud.google.com/storage/browser`
- Go to the bucket `ac215-rsvp-radar`
- Create a folder `dvc_store` inside the bucket

## Run DVC Container

We will be using [DVC](https://dvc.org/) as our data versioning tool. DVC (Data Version Control) is an Open-source, Git-based data science tool. It applies version control to machine learning development, make your repo the backbone of your project.

### Run `docker-shell.sh`

Based on your OS, run the startup script to make building & running the container easy

- Make sure you are inside the `data-versioning` folder and open a terminal at this location
- Run `sh docker-shell.sh`

### Download Labeled Data

In this step we will download all the labeled data from the GCS bucket and create `dataset_v1` version of our dataset.

- Go to the shell where ran the docker container for `data-versioning`
- Run `python cli.py -d`

If you check inside the `data-versioning` folder you should see the a `email_dataset` folder with labeled images in them.
The dataset from the data labeling step will be downloaded to a local folder called `email_dataset`

### Ensure we do not push data files to git

Make sure to have your gitignore to ignore the dataset folders. We do not want the dataset files going into our git repo.

```
/email_dataset_prep
/email_dataset
```

### Version Data using DVC

In this step we will start tracking the dataset using DVC.

- Create a data registry using DVC `dvc init`
- Add Remote Registry to GCS Bucket (For Data) `dvc remote add -d email_dataset gs://ac215-rsvp-radar/dvc_store`
- Add the dataset to registry `dvc add email_dataset`
- Push Data to Remote Registry `dvc push`

You can go to your GCS Bucket folder `dvs_store` to view the tracking files

### Update Git to track DVC

- First run git status `git status`
- Add changes `git add .`
- Commit changes `git commit -m 'dataset updates...'`
- Add a dataset tag `git tag -a 'dataset_v1' -m 'tag dataset'`
- Push changes `git push --atomic origin main dataset_v1`

### Download newly Labeled Data

In this step we will download the labeled data from the GCS bucket and create `dataset_v2` version of our dataset.

- Go to the shell where ran the docker container for `data-versioning`
- Run `python cli.py -d`
- Add the dataset to registry `dvc add email_dataset`
- Push Data to Remote Registry `dvc push`

#### Update Git to track DVC changes

- First run git status `git status`
- Add changes `git add .`
- Commit changes `git commit -m 'dataset updates...'`
- Add a dataset tag `git tag -a 'dataset_v2' -m 'tag dataset'`
- Push changes `git push --atomic origin main dataset_v2`
