<template>
    <div class="workspace">
        <Header />
        <client-only>
            <div class="bx--grid">
                <div class="bx--row u-margin-top-small">
                    <h1> Tech for Justice Bias Text Analysis Tool</h1>
                </div>
                <div class="bx--row u-margin-top-small">
                    <h2>{{this.welcome}}</h2>
                </div>

                <!-- <div class="bx--row u-margin-top-small">
                    <cv-tile
                        kind="expandable"
                        :light="light"
                        :expanded="expanded"
                        :tile-collapsed-label="tileCollapsed"
                        :tile-expanded-label="tileExpanded">
                        <h1>Single Line</h1><p>Expand to enter a line of text into the software.</p>
                        
                        <template slot="below">
                            <div class="u-margin-top-small">
                                <h3> You can enter lines like this: "When one crosses the street, they ought mind their little nose.", "The truth is, they are very angry people."</h3>
                                <ol class="u-margin-top-small"> <p style="text-decoration: underline;">Instructions </p>
                                    <li>Copy text for testing</li>
                                    <li>Click button below</li>
                                    <li>Paste line on next page</li>
                                    <li>Review suggested edits.</li>
                                    <li> - </li>
                                </ol>
                           </div>
                        </template>
                       
                    </cv-tile>
                    <cv-button class="u-margin-top-small"
                    @click="Line()">
                        Test Line Here
                    </cv-button>
                </div> -->

                <div class="bx--row u-margin-top-small">
                    <cv-tile
                        kind="expandable"
                        :light="light"
                        :expanded="expanded"
                        :tile-collapsed-label="tileCollapsed"
                        :tile-expanded-label="tileExpanded">
                        <h1>News Article</h1><p>Expand to enter a news article into the software.</p>
                        
                        <template slot="below">
                            <div class="u-margin-top-small">
                                <h3> You can enter an article as text like this: "Local students raise awareness about formula shortage by raising funds after school. We have been ... "</h3>
                                <ol class="u-margin-top-small"> <p style="text-decoration: underline;">Instructions </p>
                                    <li>Copy text for testing</li>
                                    <li>Click button below</li>
                                    <li>Paste entire article on next page</li>
                                    <li>Review suggested edits.</li>
                                    <li> - </li>
                                </ol>
                           </div>
                        </template>
                       
                    </cv-tile>
                    <cv-button class="u-margin-top-small"
                    @click="News()">
                        Test News Article Here
                     </cv-button>
                </div>
                <div class="bx--row u-margin-top-small">
                    <cv-tile
                        kind="expandable"
                        :light="light"
                        :expanded="expanded"
                        :tile-collapsed-label="tileCollapsed"
                        :tile-expanded-label="tileExpanded">
                        <h1>Movie Script</h1><p>Expand to enter a movie script into the software.</p>
                        
                        <template slot="below">
                            <div class="u-margin-top-small">
                                <h3> You can enter a script as text like this: "Jane: "How will I ever win this race against one of them." "Peter: "You certainly won't win with a bigoted design like that."... "</h3> 
                                <ol class="u-margin-top-small"> <p style="text-decoration: underline;">Instructions </p>
                                    <li>Copy text for testing</li>
                                    <li>Click button below</li>
                                    <li>Paste entire movie script on next page</li>
                                    <li>Review suggested edits.</li>
                                    <li> - </li>
                                </ol>
                           </div>
                        </template>
                       
                    </cv-tile>
                    <cv-button class="u-margin-top-small"
                    @click="Movie()">
                        Test Movie Scripts Here
                    </cv-button>
                </div>
          </div>
        </client-only>
    </div>
</template>






<script>
import Vue from 'vue';
import VueResource from 'vue-resource';
import Header from '../../components/Header'

Vue.use(VueResource);


export default {
    props: {"use_slotBelow": true,
  "light": false,
  "expanded": false,
  "tileCollapsed": "",
  "tileExpanded": ""},
    components:{
        Header
    },
    data(){
        return{
            welcome: "Please select the type of text you would like to have analysed."
        }
    },
    methods: {
        Movie(){
            // this.$router.push("movie")
            var filmUI = process.env.VUE_APP_FILM_UI || "http://localhost:8081";
            console.log("The film ui url is ====== ",filmUI);
            window.location.href = filmUI;
        },
        News(){
            this.$router.push('news')
        },
        Line(){
            this.$router.push('line')
        }
    
    }
}
</script>

