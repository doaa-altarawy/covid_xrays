Flask app for COVID Screening using xray images 
=================================

This is a Flask app for screening patients' xray in order to recommend further COIVD19 testing.

How to run the app for deployment with docker
==========================

Run docker-compose directly, or optionally, change any desired environment variables by creating 
a `.env` file which will be read automatically by docker-compose.

** It is very important to make sure that the shared folder has the right user NOT root owner. **

```bash
git clone git@github.com:doaa-altarawy/covid_xrays.git
cd covid_xrays
# create the host volume folder with non-root access
mkdir docker_data
chown -R $USER:$USER docker_data
# clean unused docker images and containers (optional)
./dockerclean.sh
# build and run containers
docker-compose up -d
```

The page will be available at
`http://localhost/`


How to run and use for development:
===================================

### 1- Install Python requirements:

Run in shell, create python env, and install requirements:

```bash
conda create -n covid_xrays
conda activate covid_xrays
pip install -r requirements.txt
```


### 2- Install JavaScript requirements:

Next, install Node (front-end), and install JS requirements, 
which will be fetched from package.json automatically. In Ubuntu:

```bash
sudo apt-get install nodejs
cd covid_xrays/app/static
npm install
```

### 3- Database setup

1. Install mongodb based on your operating system from 
https://docs.mongodb.com/guides/server/install/

2. Create a `covid_xrays/.env` file, and add your DB URI to the config file:
```.env
MONGO_URI='mongodb://usr_username:user_password@localhost:27017/covid_apis_db'
```

Replace `user_username` and `user_password` with your own values from your installation. 
You **don't** have to create a database after your install mongodb because the application will do
 it later.


Note: In the future when you need to, add PUBLICALLY shared environment attributes to `.flaskenv` file, with key values that will be exported to the environment (dev, prod, etc).
Use `.env` file for private variables that won't be shared or pushed to Github. Note that `.env` overrides `.flaskenv`, and both override `config.py`.



### 4- Run the local server

To run the website locally, use: 

```bash
flask run
```



