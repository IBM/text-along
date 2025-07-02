<template>
    <div class="workspace">
        <Header />
        <client-only>
          <div class="bx--grid">
              <div class="bx--row">
                <div class="bx--col">
                    <div class="header__title">
                        <h1>Tech for Justice Bias Text Analysis Tool</h1>
                        <h2>Movie Script</h2>
                    </div>
                </div>
              </div> 
              <div class="bx--row">      
                <cv-text-area
                    id="text"
                    :light="light"
                    :label="label"
                    :value="value"
                    :disabled="disabled"
                    :placeholder="placeholder"
                    class="bx--grid u-margin-top-medium">
                </cv-text-area>
            </div>
                <div class="bx--row">
                    <div class="bx--col-lg-1">
                        <cv-progress
                            v-if="progress_show"
                            :initial-step="initialStep"
                            :steps="steps"
                            :vertical="vertical_progress" @step-clicked="actionStepClicked">
                            </cv-progress>
                        </div>
                    </div>
                <div class="bx--row">
                    <div class="bx--col-lg-1 u-margin-top-small"/>
                    <div class="bx--col-lg-2 bx--col-md-1 u-margin-small">
                        <cv-button 
                        @click="CheckBias()">
                            Analyse for Bias
                        </cv-button>
                    </div>
                    <div class="bx--col-lg-2 bx--col-md-1 u-margin-small">
                        <cv-button
                        @click="openModal()">
                            upload file
                        </cv-button>
                    </div>
                </div>
                <div class="bx--row">
                    <div class="bx--col-lg">
                        <highlightable-input
                            id="output"
                            :value="message"
                            highlight-style="background-color:yellow" 
                            :highlight-enabled="highlightEnabled" 
                            :highlight="data_hig" 
                            class="bx--text-area"
                        />
                    </div>
                    <div class="bx--col-lg-2">
                        <cv-button
                            v-if="bias_shown"
                            @click='ShowComplex()'>
                            Show word Bias
                        </cv-button>
                    </div>
                </div>
            </div>
            <cv-modal
                :close-aria-label="closeAriaLabel"
                :size="size"
                :visible="visible"
                :auto-hide-off="autoHideOff"
                @close-modal="closeModal"
                @modal-hide-request="actionHideRequest"
                @after-modal-hidden="actionAfterHidden">
                    <template v-if="use_label" slot="label">Upload File</template>
                    <template v-if="use_title" slot="title">For bias analysis</template>
                    <template slot="content">
                        <div class= "bx--form-item">
                            <cv-file-uploader
                                id="files"
                                ref="files" 
                                kind="drag-target"
                                label="Model File(s)"
                                helper-text="Please upload your text file. Acceptable type is .txt "
                                drop-target-label="Drag and drop files here or click to upload"
                                accept=".txt"
                                :clear-on-reselect="clearOnReselect"
                                :initial-state-uploading="initialStateUploading"
                                :multiple="false"
                                :removable="removable"
                                >
                            </cv-file-uploader>
                        </div>
                        <cv-button
                        @click="SaveText()">
                            upload
                        </cv-button>
                    </template>
            </cv-modal>
            <cv-modal
                :close-aria-label="closeAriaLabel"
                :size="size"
                :visible="visible_complex"
                :auto-hide-off="autoHideOff"
                @close-modal="closeModal"
                @modal-hide-request="actionHideRequest"
                @after-modal-hidden="actionAfterHidden">
                    <template v-if="use_title" slot="title">Bias scores</template>
                    <template slot="content">
                        <cv-data-table
                        :columns="columns" :data="data"   ref="table">
                        </cv-data-table>
                        
                    </template>
            </cv-modal>
          
        </client-only>
    </div>
</template>






<script>
import Vue from 'vue';
import VueResource from 'vue-resource';
import HighlightableInput from 'vue-highlightable-input';
import Header from '../../components/Header'
Vue.use(VueResource);
export default {
    components : {
        HighlightableInput,
        Header
    },
    data(){
        return{
            modal: false,
            modalTitle: "Upload file.",
            light: false,
            label: "Enter the sample text below",
            disabled: false,
            placeholder: "Enter test text here!",
            inputType: String,
            value: '',
            files: [],
            //modal
            closeAriaLabel: '',
            size: null,
            visible: false,
            autoHideOff: true,
            hideModal: false,
            use_title: true,
            use_content: true,
            use_label: true,
            description: '',
            title: '',
            content: '',
            visible_complex: false,
            //file uploader
            kind: "",
            label: "Choose files to upload",
            helperText: "Select the files you want to upload",
            dropTargetLabel: "",
            accept: ".txt",
            clearOnReselect: false,
            initialStateUploading: false,
            multiple: false,
            removable: false,
            removeAriaLabel: "Custom remove aria label",
            text: '',
            //progress
            initialStep: 0,
            steps: [
                "Sending text to server",
                "Recieved bias results",
                "Highlighting text"
            ],
            vertical_progress: false,
            progress_show: false,
            // highlight text
            data_hig: [
                {text:'test_string_impossible_to_obtain'}
            ],
            message: '',
            highlightEnabled: true,
            test:'hello there',
            bias_output: '',
            closeAriaLabel: "close",
            bias_shown: false,
            //data table
            columns: [
                "Word",
                "score"
            ],
            data: [],
            
        }
        
    },
    computed:{
        msg:{
            get(){
                return this.message;
            },
            set(Val){
                this.message = Val
            }
            
        },
        highlight: {
            //add highlights to the highlight dictionary
            get(){
                return this.data_hig
            },
            set(newValue){
                this.data_hig.push(newValue)
            }
        },
        // bias: {
        //     get(){
        //         return this.data
        //     },
        //     set(newValue){
        //         console.log(newValue[0])
        //         console.log(newValue[1])
        //         this.data[0].push(newValue[0])
        //         this.data[1].push(newValue[1])
        //         console.log(this.data)
                
        //     }
        //}
    },
    methods: {
        actionStepClicked(){
            
        },
        async CheckBias(){
            this.progress_show = true
            const textarea = document.getElementById('text');
            const text_out = document.getElementById('output')
            let CheckText = textarea.value
            let x = 0
                let test = await this.$axios.$post('http://localhost:8080/api/v0/annotate/text',{ 'docs':{ 'lines': { 'text': CheckText}}, 'text_id': '<text_type>'}, {headers: {'accept': 'application/json', 'Content-Type': 'application/json'}})
                this.initialStep = 2
                console.log(test)
                let check_words = CheckText.split(' ')
                console.log(check_words)
                for (const[key, value] of Object.entries(test)){
                    console.log(key, value)
                    let list = [key, value]
                    this.data.push(list)
                    for (var i = 0; i < check_words.length; i++){
                        if (check_words[i] == key ){
                            console.log(typeof(key))
                            console.log(value)
                            if (value > 0){
                                if (value > 0.5){
                                    this.highlight = {text:key, style:"background-color:#00FF00"}
                                    console.log("3")  
                                }
                                else if(0.5 > value >= 0.3){
                                    this.highlight = {text:key, style:"background-color:#92FF92"}
                                    console.log("2")  
                                }else{
                                    this.highlight = {text:key, style:"background-color:#C3FFC3"}
                                    console.log("1")
                                }
                            }
                            else{
                                if(-0.5 > value){
                                    this.highlight = {text:key, style:"background-color:#FF0000"}
                                    console.log("0")  
                                }else if(-0.5 >= value > -0.2){
                                    this.highlight = {text:key, style:"background-color:#FF5E5E"}
                                    console.log("-1")  
                                }else{
                                    this.highlight = {text:key, style:"background-color:#FFA1A1"}
                                    console.log("-2")  
                            }
                        }
                        
                        //   happy sad ok fine angry
                    }
                }
            }
            this.initialStep = 3
            textarea.value = CheckText
            this.msg = CheckText
            console.log(this.msg)
            console.log(text_out.value)
            this.bias_shown = true
            
            
            
        },
        showText(){
            const textarea = document.getElementById('text');
            
            console.log(this.msg)
            console.log(this.message)
            this.msg = textarea.value
        },
        openModal(){
            console.log("test function")
            this.visible = true;
            this.Title = "File Upload"
            console.log(this.modal)
        },
        closeModal(reloadPage) {
            this.visible = false
            this.visible_complex = false
        },
        actionHideRequest(){
            this.visible = false
            this.visible_complex = false
        },
        actionAfterHidden(){
            
        },
        SaveText(){
            this.unpackTextFromFile()
            this.visible = false
        },
        unpackTextFromFile(){
            this.files = this.$refs.files.internalFiles;
            if (this.files.length > 0) {
            
                    
                    console.log(this.files[0].file)
                    const reader = new FileReader();
                    reader.onload = function(){
                        console.log(reader.result)
                        if (reader.result != ""){
                            var text = reader.result
                            console.log(text)
                            const textarea = document.getElementById('text');
                            console.log(text)
                            textarea.value = text
                            console.log(textarea.value)
                            
                        }  
                    }
                    reader.readAsText(this.files[0].file)
                    
            }
            
        },
        ShowComplex(){
            console.log("test function")
            this.visible_complex = true;
            this.Title = "Bias's found"
            console.log(this.modal)
        },
        actionHidden(){
            this.modal = false
            this.modelComplex = false
            console.log("test")
        }
    }
}
</script>