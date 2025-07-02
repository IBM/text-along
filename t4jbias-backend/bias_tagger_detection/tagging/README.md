## Set UP
At the project's root directory, create and install the required modules in a venv:
```
python -m venv venv

source ./venv/bin/activate

pip install -r requirements.txt
```

You can pass your artifactory user name and api key directly to the python script using `--arte_user` and `--arte_pwd` arguments. OR you can create a `.env` file in the `src` directory and add your username and api key for artifactory in the following variables. If you don't already have one, check out [this](https://taas.cloud.ibm.com/guides/create-apikey-in-artifactory.md) page. Please make sure you have been granted access to the team's artifact `res-media-bias-4374-team-ml-generic-local` [here](https://eu.artifactory.swg-devops.com/artifactory/res-media-bias-4374-team-ml-generic-local/)

`ARTE_USER=<?>`

`ARTE_PWD=<?>`

### Using Docker
If you've got docker installed already, you can build and run the app within a container using:

```
sh run_app_docker.sh -a <your-password>
```
## Run Flask App

Run the flask app to access the api endpoint for detection.
First Download models using the `download-models.sh` script. Then run
```
python app.py
```
