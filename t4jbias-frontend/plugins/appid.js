// https://cloud.ibm.com/docs/appid?topic=appid-web-apps

const express = require('express');
const session = require('express-session');
const passport = require('passport');
const appID = require('ibmcloud-appid');
const WebAppStrategy = require('ibmcloud-appid').WebAppStrategy;
const { createProxyMiddleware } = require("http-proxy-middleware");
const userProfileManager = appID.UserProfileManager;
// const cfEnv = require('cfenv');
const bodyParser = require('body-parser');

const CALLBACK_URL = '/t4j/demo';
const UI_BASE_URL = '/';

const app = express();
const port = process.env.PORT || 3000;
const host = "0.0.0.0";
// const isLocal = cfEnv.getAppEnv().isLocal

const config = getLocalConfig();

// Set up Redis if on cloud or use Express MemoryStore if local
let sessionStore;
const redis = require('redis');
let RedisStore = require('connect-redis')(session);

if (process.env.REDIS_HOST != undefined){
  let redisClient = redis.createClient({
    host: process.env.REDIS_HOST
  });
  redisClient.on('connect', function() {
    console.log('Successfully connected to Redis');
  });  
  redisClient.on('error', function(error) {
    console.error(error);
  });  
  sessionStore = new RedisStore({ client: redisClient });
}
else{
  sessionStore = new session.MemoryStore();
}

// Set up your express app to use express-session middleware.
app.use(session({
  store: sessionStore, 
  secret: '123456',
  resave: true,
  saveUninitialized: false,
  disableTTLRefresh:true,
  proxy: true
}));
app.use(passport.initialize());
app.use(passport.session());

// Configure passport with serialization and deserialization. This configuration step is required for authenticated session persistence across HTTP requests.
passport.serializeUser(function(user, cb) {
  cb(null, user);
});
 
passport.deserializeUser(function(user, cb) {
  cb(null, user);
});

// By using the information obtained in the previous steps, initialize the SDK.
// passport.use(new WebAppStrategy(config));

// Login
app.post('/login', express.urlencoded({extended: false}), function(req, res, next){
  passport.authenticate(WebAppStrategy.STRATEGY_NAME, function(err, user) {
    if (err){
      return res.redirect('/?error=' + err);
    }
    if (! user) {
      return res.redirect('/?error=Invalid Credentials');
    }
    req.login(user, loginErr => {
      if (loginErr) {
        return res.redirect('/?error=' + loginErr);
      }
      return res.redirect('t4j/demo');
    });      
    //successRedirect: 'fhe/workspace',
    //failureFlash : false // allow flash messages
  }) (req, res, next);
});

app.post('/news', express.urlencoded({extended: false}), function(req, res, next){
  return res.redirect('t4j/demo'); 
  (req, res, next);
});

// app.use(
//   createProxyMiddleware("/", {
//     target: "http://127.0.0.1:6006/",
//     // target: `http://127.0.0.1:${port}${CALLBACK_URL}`,
//   })
// );
// app.listen(port, host);

// Protect pages
app.get('/t4j/*', function(req, res, next) {
  console.log('Trying to get T4J page');
  console.log(req);
  if (req.user){
    next();
  }
  else {
    res.redirect('/');
  }
});

// Check whether user is logged in or not
app.get('/check_status', function(req, res) {
  let accessToken = null;
  let identityToken = null;
  // Check tokens exist 
  if (req && req.session[WebAppStrategy.AUTH_CONTEXT] && req.session[WebAppStrategy.AUTH_CONTEXT].accessToken) {
    accessToken = req.session[WebAppStrategy.AUTH_CONTEXT].accessToken;
    identityToken = req.session[WebAppStrategy.AUTH_CONTEXT].identityToken;
    userProfileManager.init(getLocalConfig());

    // Get profile
    let profile = userProfileManager.getUserInfo(accessToken, identityToken);
    profile.then(function (profile) {
      // Send profile to UI
      res.json(profile);
    });
    profile.catch(() => {
      // Send error to UI
      res.json({'error': '401: Unauthorised'});
    });
  }
  // If not, handle gracefully. UI picks up on response
  else {
    res.json({'404': 'User not found'});
  }
});

// Logout endpoint. Clears authentication information from session
app.get('/logout', function(req, res) {
  req.session.destroy(function(err) {
    if (err) {
      console.log(err);
    }
    
    res.json({'200': 'Success'});
    // res.redirect(UI_BASE_URL);
  });
});

// Set up auth path so app knows difference between API calls and AppID
module.exports = { path:'/', handler:app };

// If running locally, get the config from the App ID file
function getLocalConfig() {
  let config = {};
  // try {
  //   const localConfig = require('../config.json');
  //   const requiredParams = ['clientId', 'secret', 'tenantId', 'oAuthServerUrl', 'profilesUrl'];
  //   requiredParams.forEach(function (requiredParam) {
  //     if (!localConfig[requiredParam]) {
  //       console.error('When running locally, make sure to create a file *config.json* in the root directory. See config.template.json for an example of a configuration file.');
  //       console.error(`Required parameter is missing: ${requiredParam}`);
  //       process.exit(1);
  //     }
  //     config[requiredParam] = localConfig[requiredParam];
  //   });
  // }
  // catch (e) {
  //   if (process.env.APPID_SERVICE_BINDING) { // if running on Kubernetes this env variable would be defined
  //     config = JSON.parse(process.env.APPID_SERVICE_BINDING);
  //     config.redirectUri = process.env.redirectUri;
  //   } else {
  //     return {};
  //   }
  // }
  config['redirectUri'] = `http://127.0.0.1:${port}${CALLBACK_URL}`;
  return config; 
}