
<template>
    <div class="workspace">
        <!-- <LoginHeader /> -->
        <client-only>
          <div class="bx--grid">

                <div class="bx--col">
                    <div class="header__title">
                        <h1>Welcome to the T4J Bias Text Analysis Tool</h1>
                        <h3>With this tool, you will be able to enter and analyse media content for biases or harsh language. </h3>
                    
                        <p> <br>Please use your credentials to log in and access the T4J Bias Text Analysis Tool</p>
                    </div>
                </div>
                <div class="bx--col-lg-6">
                    <cv-button
                        id='login'
                        kind='primary'
                        size='field'
                        :disabled="false"
                        type='submit'
                    >
                            Submit
                    </cv-button>
                    <!-- <div> -->
                            <!-- <form 
                                action="/login" 
                                method="post"
                            >
                            <cv-text-input 
                            id="username"
                            value=''
                            label="Username/Email"
                            placeholder="Username/Email"
                            name="username">
                            </cv-text-input> -->
                            
                            <!-- <cv-text-input
                            id="password"
                            value=''
                            label="Password"
                            placeholder="Password"
                            :type="'password'"
                            :password-visible="false"
                            name="password">
                            </cv-text-input> -->
<!-- 
                            <div class="u-margin-top-small">
                                <p 
                                v-if="invalidCredentials" 
                                class="invalid-credentials"
                                >
                                {{ invalidMessage }}
                                </p>
                            </div> -->
                            <!-- <div class="bx--row u-margin-small"> -->
                                <!-- <cv-button
                                    id='login'
                                    kind='primary'
                                    size='field'
                                    :disabled="false"
                                    type='submit'
                                >
                                        Submit
                                    </cv-button> -->
                            <!-- </div>
                        </form> -->
                    <!-- </div> -->
                </div>
          
          </div>
        </client-only>
    </div>
</template>


<script>
import Vue from 'vue';
import VueResource from 'vue-resource';
import LoginHeader from '../components/LoginHeader.vue';

Vue.use(VueResource);


export default {
//   components: { LoginHeader },
   
    data(){
        return {
            // username: '',
            // password: ''
            invalidCredentials: false,
            invalidMessage: ''
        }
    },
    mounted () {
    this.checkRedirectUrl();
    },
    methods: {
        async actionSubmit(){
        //     //this.username is the username
        //     //this.password is the password
        //     let user = document.getElementById("username").value
        //     let pass = document.getElementById("password").value
        //     this.usename = user
        //     this.password = pass


        //     //https://prepiam.ice.ibmcloud.com/authsvc/mtfim/sps/authsvc?PolicyId=urn:ibm:security:authentication:asf:basicldapuser&Target=https%3A%2F%2Fprepiam.ice.ibmcloud.com%2Foidc%2Fendpoint%2Fdefault%2Fauthorize%3FqsId%3D369ccfa2-e6a9-434e-bb95-1c3df6cde327%26client_id%3DMTNmMjhiMmItMDg2OC00
            // this.$router.push('/t4j/demo') //go to demo
            this.$router.push('t4j/demo') //go to demo
   
        },
        checkRedirectUrl() {
            // Display error message if credentials are wrong
            if (this.$route.fullPath == '/?error=Invalid%20Credentials') {
                this.invalidCredentials = true;
                this.invalidMessage = 'Please check the credentials and try again';
                console.warn('Invalid credentials');
            }
            // Catch other errors
            else if (this.$route.fullPath == '/?error=') {
                this.invalidCredentials = true;
                this.invalidMessage = 'There has been a problem logging in. Please check the logs for more information';
            }
            // The URL is to be expected i.e. '/' so no need to display errors
            else {
                this.invalidCredentials = false;
                this.invalidMessage = false;
                this.actionSubmit()
            }
        }
    }
}
</script>
