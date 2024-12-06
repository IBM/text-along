# T4J-UI
This is the User interface for the project that will allow you to analyse for subjective and non obvious biases in text.

## System Requirements
Minimum of:
- Node v16.13.0
- NPM 8.1.2

(Optional)
Recommended to use `nvm` to manage your node installations if needed. 
You can `nvm list` to check your available node versions and `nvm use <version-number>` to switch between node versions.

## Installation
1. Clone this repository to a new directory and follow the `build setup` instructions below
2. For the service to work the t4jbias-backend-api must be running 
3. set the server base url to the api host E.g. by running this command in the terminal 
```
export SERVER_BASE_URL=http://127.0.0.1:6006/
```
```
export HOST="localhost"
```
<!-- #4. create a `.env` file and set the redirect url for the film ui, e.g. `VUE_APP_FILM_UI="https://ui-api-media-bias-uk-dev.bx.cloud9.ibm.com/#/scripts"` -->

## Build Setup

```bash
# Install Dependencies
$ rm -rf node_modules && npm install

# Build for development environment and launch server. (Serve with hot reload at localhost:3000)
$ npm run dev

# Build for production and launch server 
$ npm run build
$ npm run start

# Generate static project
$ npm run generate
```

## Movie scripts are to be implemented soon
