<template>
  <div class="header-nav">
    <client-only>
    <cv-header aria-label="Carbon header">
        <cv-header-name @click="login" prefix="IBM">
        Tech For Justice
      </cv-header-name>
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
  computed: {
   
  },
  methods: {
        login(){
            this.$router.push('login') //go to movie
        }
  }

};
</script>