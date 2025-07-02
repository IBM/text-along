<template>
  <div class="header-nav">
    <client-only>
<cv-header aria-label="Carbon header">
  <cv-header-menu-button aria-label="Header menu" aria-controls="side-nav" />
  <cv-skip-to-content href="#main-content">
    Skip to content
  </cv-skip-to-content>
  <cv-header-name @click="demo" prefix="IBM">
    Tech For Justice
  </cv-header-name>
  <cv-header-nav aria-label="Carbon nav">
  <cv-header-menu-item @click="demo">
    Home Page
  </cv-header-menu-item>
  <cv-header-menu-item href="javascript:void(0)">
    About the project
  </cv-header-menu-item>
  <cv-header-menu aria-label="Link 3" title="Analysis tools">
    <cv-header-menu-item @click="movie">
      Movie script
    </cv-header-menu-item>
    <cv-header-menu-item @click="news">
      News article
    </cv-header-menu-item>
    <!-- <cv-header-menu-item @click="line">
      Line of text
    </cv-header-menu-item> -->
  </cv-header-menu>
</cv-header-nav>
    <template v-slot:left-panels v-if="areLeftPanels">
      
  <cv-side-nav id="side-nav" fixed>
    <cv-side-nav-items>
      <cv-header-side-nav-items>
        <cv-header-menu-item @click="demo">
        Home Page
      </cv-header-menu-item>
      <cv-header-menu-item href="javascript:void(0)">
        About the project
      </cv-header-menu-item>
      <cv-header-menu aria-label="Link 3" title="Analysis tools" :hover-toggle="false">
        <cv-header-menu-item @click="movie">
          Movie script
        </cv-header-menu-item>
        <cv-header-menu-item @click="news">
          News article
        </cv-header-menu-item>
        <!-- <cv-header-menu-item @click="line">
          Line of text
        </cv-header-menu-item> -->
      </cv-header-menu>
    </cv-header-side-nav-items>
    </cv-side-nav-items>
  </cv-side-nav>
    </template>
    <template v-slot:right-panels v-if="areRightPanels">
    
  </template>
</cv-header>
    </client-only>
  </div>
</template>

<script>
import Vue from 'vue';
import VueResource from 'vue-resource';

Vue.use(VueResource);

export default {
    props: {
    showNav: {
      type: Boolean,
      default: true,
      required:false
    }
  },
  data () {
    return {
      loggedIn: null,
      checkedSignIn: false,
      loading: false,
      name: "",
      areRightPanels: false,
      areLeftPanels: true
    };
  },
  mounted () {
    // this.checkUserStatus();
  },
  computed: {
   
  },
  methods: {
        demo(){
            this.$router.push('demo') //go to movie
        },
        movie(){
            //this.$router.push('movie') //go to movie
            window.location.href = process.env.VUE_APP_FILM_UI || "http://localhost:8081";
        },
        news(){
            this.$router.push('news') //go to news
        },
        line(){
            this.$router.push('line') //go to line
        },
        // checkUserStatus () {
        //   Vue.http.get('/check_status')
        //     .then((response) => {
        //       if (response.data.given_name) {
        //         console.log(response)
        //         this.name = response.data.given_name
        //         // Update store to say user is signed in
        //         this.$store.commit('checkLoggedIn', true);
        //         // Show header options
        //         this.loggedIn = this.$store.state.isLoggedIn;
        //       } 
        //       else {
        //         // Error handle on a 401 response (it doesn't come through as an error)
        //         if (response.body.error == '401: Unauthorised') {
        //           window.location.href='/public';
        //       }
        //         // Handle 404 response from API
        //         console.warn('No profile found');
        //         //window.location = ("https://eu-gb.appid.cloud.ibm.com/oauth/v4/e8d9d261-5f43-4446-9435-83a168f6d7cd")
        //      }
        //     })
        //     .catch(error => {
        //       // Error handle
        //       console.error('Error getting profile ->', error);
        //       this.checkedSignIn = true;
        //       if (error.status != 500) {
        //         window.location.href='/public';
        //       }
        //     });
        // }, 
      }

};
</script>