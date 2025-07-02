import { handler } from "./plugins/appid";

export default {
  // Global page headers (https://go.nuxtjs.dev/config-head)
  head: {
    title: 'T4J Demo',
    meta: [
      { charset: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      { hid: 'description', name: 'description', content: '' }
    ]
  },

  // Plugins to run before rendering page (https://go.nuxtjs.dev/config-plugins)
  plugins: [
    { src: '@/plugins/carbon.js', mode: 'client' },
    // { src: '@/plugins/appid.js', mode: 'server' }
  ],

  // Auto import components (https://go.nuxtjs.dev/config-components)
  components: true,

  // Modules for dev and build (recommended) (https://go.nuxtjs.dev/config-modules)
  buildModules: [
    '@nuxtjs/dotenv'
  ],

  // Modules (https://go.nuxtjs.dev/config-modules)
  modules: [
    '@nuxtjs/axios',
    '@nuxtjs/proxy'
  ],

  axios: {
    proxy: true,
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      'Access-Control-Allow-Credentials': 'true',
      "Connection": "keep-alive"
    }
  },

    //pathrewrite will cover the word before it and replace
  proxy: {
    '/api/': { target: process.env.SERVER_BASE_URL, pathRewrite: {'^/api/': ''}, changeOrigin: true },
    '/predict/': { target: process.env.SERVER_BASE_URL, changeOrigin: true }
  },

  // Build Configuration (https://go.nuxtjs.dev/config-build)

  build: {
    transpile: ['vue-highlightable-input'],
    extend(config, ctx) {
      if (ctx.isDev) {
        config.devtool = ctx.isClient ? 'source-map' : 'inline-source-map';
      }
    }
  },

  serverMiddleware: [
    // Any time the user navigates to any page in the application, run it through appid to check if the links match
    // for appid and cloudant
    { path: '/', handler: '~/plugins/appid.js'},
  ],

  publicRuntimeConfig: {
    clusterURL: process.env.CLUSTER_URL,
    apiVersion: process.env.API_VERSION,
    VUE_APP_FILM_UI: process.env.VUE_APP_FILM_UI
  },
};
