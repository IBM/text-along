<template>
    <div class="workspace">
        <Header />
        <client-only>
                <div class='title'>
                    <h1>Tech for Justice Bias Text Analysis Tool</h1>
                    <br>
                    <h2>News Article</h2>
                </div>
                <div class="body">
                    <html>
                        <body>
                            <div class="top-body">
                                <div class="radio-buttons">
                                    <cv-radio-group :label="radioButtonsHeading"> 
                                        <cv-radio-button name="group-1" label="Headline" value="value-1" :checked="checked1"
                                            :hide-label="hideLabel" :label-left="labelLeft" />
                                        <cv-radio-button name="group-1" label="Complete Article" value="value-2"
                                            :hide-label="hideLabel" :label-left="labelLeft" />
                                    </cv-radio-group>
                                </div>
                                <div class="rhs-button">
                                    <cv-button @click="openModal()" :kind="uploadButtonKind" :icon="uploadIcon">
                                        Upload File
                                    </cv-button>
                                </div>
                            </div>
                            <br>
                            <div class="analyse-area">
                                <div class="text-area">
                                    <cv-text-area  id="text" ref="textarea" :light="light" :helper-text="HelperTextArea" :value="value"
                                        :disabled="disabled" :placeholder="placeholder" rows="6">
                                    </cv-text-area>
                                </div>
                                
                                <br>
                                <div class="buttons">
                                    <div class="url-button">
                                        <cv-button @click="url()" :icon="checkURLIcon" :kind="checkURLKind">
                                            Check URL
                                        </cv-button>
                                    </div>
                                    <div class="bias-button">
                                        <cv-button @click="CheckBias()" :icon="analyseButton">
                                            Analyse for Bias
                                        </cv-button>
                                        <div v-if="response_data">
                                            
                                        </div>
                                    </div>
                                    
                                    
                                </div>
                            </div>
                            <br>
                            <div class="output-header">
                                <div>
                                    <p>Output</p>
                                </div>
                                <div class="details-button">
                                <cv-button  :kind="showDetailsButtonKind" @click='ShowComplex()' :disabled="!bias_shown" :icon="detailsButton">
                                    Show Details
                                </cv-button>
                            </div>
                            </div>
                            <br>
                            <div class="output-area">
                                <div class="output-box" id="output">
                                    <div>
                                        <highlightable-input class="output-text" ref="highlightInput" v-for="(d, $index) in data_hig" :key="$index" :message="d.message"
                                            :highlight="d.text" :highlightmsg="d.highlightmsg" :probability="d.probability"/>
                                    </div>
                                    <div class="loading" v-if="!analysis_done">
                                        <!-- <cv-inline-loading :state="loadingState" :loadingText="loadingText" loaded-text="" ending-text=""></cv-inline-loading> -->
                                        <cv-loading :active="!analysis_done" :small="smallLoading"></cv-loading>
                                        <p>Running the analysis...</p>
                                    </div>
                                            
                                </div>
                            </div> 
                            
                            <p>Red is for probability greater than 0.5 and yellow is for probability lower than 0.5</p>
                            <br>
                            <p> The presented potential stereotypes of the tagged words is a result of the work produced by
                                Teyun Kwon and Anandha Gopalan. </p>
                            <p> <a target="_blank" rel="noopener noreferrer" href="https://arxiv.org/abs/2112.00819"> CO-STAR: Conceptualisation of Stereotypes for
                            Analysis and Reasoning</a>. 2021. </p>
                            
                            
                            
                    </body>
                </html>
                </div>

            <!-- Upload Files Modal -->
            <cv-modal :close-aria-label="closeAriaLabel" :size="size" :visible="visible" :auto-hide-off="autoHideOff"
                @close-modal="closeModal" @modal-hide-request="actionHideRequest" @after-modal-hidden="actionAfterHidden">
                <template v-if="use_label" slot="label">Upload File</template>
                <template v-if="use_title" slot="title">For Bias Analysis</template>
                <template slot="content">
                    <div class="file-uploader">
                        <cv-file-uploader id="files" ref="files" kind="drag-target" label="Model File(s)"
                            helper-text="Please upload your text file. Valid file type: .txt "
                            drop-target-label="Drag and drop files here or click to upload" accept=".txt"
                            :clear-on-reselect="clearOnReselect" :initial-state-uploading="initialStateUploading"
                            :multiple="false" :removable="removable">
                        </cv-file-uploader>
                    </div>
                </template>
                <template slot="primary-button" @click="SaveText()">Upload</template>
            </cv-modal>
            <!-- Show Details Modal -->
            <cv-modal :close-aria-label="closeAriaLabel" size="large" :visible="visible_complex"
                :auto-hide-off="autoHideOff" @close-modal="closeModal" @modal-hide-request="actionHideRequest"
                @after-modal-hidden="actionAfterHidden">
                <template v-if="use_title" slot="title">Tagged Word Information</template>
                <template slot="content">
                    <cv-data-table :columns="columns"  ref="table">
                        <template slot="data">
                            <cv-data-table-row
                                v-for="(row, rowIndex) in data"
                                :key="`${rowIndex}`"
                                :value="`${rowIndex}`"
                            >
                            <cv-data-table-cell v-for="(cell, cellIndex) in row" :key="`${cellIndex}`" :value="`${cellIndex}`" >
                                <div v-if="cellIndex === 2 && cell?.length > 0">
                                    <cv-tag v-for="label in cell" :label="label" :style="{marginLeft: 0}" />
                                </div> 
                                <span v-if="cellIndex === 1">{{cell.toFixed(4)}}</span>
                                <span v-if="cellIndex !== 2 && cellIndex !== 1">{{cell}}</span>
                            </cv-data-table-cell>
                            </cv-data-table-row>
                        </template>
                    </cv-data-table>
                </template>
            </cv-modal>

        </client-only>
    </div>
</template>






<script>
import Vue from 'vue';
import VueResource from 'vue-resource';
import Header from '../../components/Header'
import LoadingDialog from '../../components/loading-dialog.vue'
import HighLight from '../t4j/highlightResult.vue'
import UploadButton from '../../node_modules/@carbon/icons-vue/lib/upload/32.js'
import CheckURLButton from '../../node_modules/@carbon/icons-vue/lib/text-link/32.js'
import AnalyseButton from '../../node_modules/@carbon/icons-vue/lib/search--locate/32.js'
import DetailsButton from '../../node_modules/@carbon/icons-vue/lib/information/32.js'


Vue.use(VueResource);
Vue.prototype.$refs = 'highlight'
export default {
    components: {
        Header,
        'highlightable-input': HighLight,
        'loading-dialog': LoadingDialog
    },
    data() {
        return {
            modal: false,
            modalTitle: "Upload file.",
            light: false,
            label: "Enter the sample text below",
            radioButtonsHeading: 'Text Type',
            uploadButtonKind: 'tertiary',
            checkURLKind: 'secondary',
            showDetailsButtonKind: 'ghost',
            uploadIcon: UploadButton,
            checkURLIcon: CheckURLButton,
            analyseButton: AnalyseButton,
            detailsButton: DetailsButton,
            smallLoading: true,
            disabled: false,
            placeholder: "Zuckerberg claims facebook can revolutionise the world.",
            ph: "Analysis results will display here",
            inputType: String,
            value: '',
            files: [],
            closeAriaLabel: '',
            size: null,
            visible: false,
            autoHideOff: true,
            hideModal: false,
            use_title: true,
            use_content: true,
            use_label: true,
            description: '',
            response_data: false,
            analysis_done: true,
            // loadingActive: false,
            title: '',
            content: '',
            visible_complex: false,
            kind: "",
            HelperTextArea: "Use a period (.) to separate multiple headlines. You can also supply a url with the `check url` button option to get content from a url.",
            helperText: "Select the files you want to upload",
            dropTargetLabel: "",
            accept: ".txt",
            clearOnReselect: false,
            initialStateUploading: false,
            multiple: false,
            removable: false,
            removeAriaLabel: "Custom remove aria label",
            text: '',
            initialStep: 0,
            steps: [
                "Sending text to server",
                "Recieved bias results",
                "Highlighting text"
            ],
            vertical_progress: false,
            progress_show: false,
            data_hig: [],
            message: '',
            highlightEnabled: true,
            test: 'hello there',
            bias_output: '',
            closeAriaLabel: "close",
            bias_shown: false,
            show: true,
            valueout: '',
            columns: [
                "Tagged Word",
                "Bias Probability Score",
                "Associated Types",
                "Potential Stereotype (PS) Related To",
                "PS Closeness to sentence",
                "Concept of Stereotype (CS)",
                "CS Closeness to sentence",
                "Extra Information"
            ],
            data: [],
            vertical: true,
            checked1: true,
            hideLabel: false,
            labelLeft: false,
            high_probability: false,
            low_probability: false,
            highlightmsg: ""

        }

    },
    computed: {
        msg: {
            get() {
                return this.message;
            },
            set(Val) {
                this.message = Val
            }

        },
        highlight: {
            //add highlights to the highlight dictionary
            get() {
                return this.data_hig
            },
            set(newValue) {
                this.data_hig.push(newValue)
            }

        },
    },
    methods: {
        actionStepClicked() {

        },
 
        async getPredictions(textarray) {
            this.analysis_done = false
            let output = []
            const arrayValid = []

            // filter the textarray so that all sentences with less than 3 words will not be sent to the back-end.
            for (let i = 0; i < textarray.length; i ++){
                if (textarray[i].split(" ").length > 3){
                    arrayValid.push(textarray[i])
                }
            }

            // get tagger and costar obj for all sentences in one go 
            let fetchData = {
            method: 'POST',
            body: JSON.stringify({'text': arrayValid}),
            headers: new Headers({'accept': 'application/json',
                'Content-Type': 'application/json'
            })
            }

            var response = await Promise.all(
                [fetch('/predict/costar',fetchData)
                .then(function(response){return response.json()})
                .then((value) => {return value}), 
                fetch('/predict/tagger', fetchData)
                .then(function(response2){return response2.json()})
                .then((value2) => {return value2})
            ]).catch((error)=>{
                    this.analysis_done = true; 
                    console.log(error)
                    alert("Error in retrieving the analysis")
                    throw error
                })

            var costar_obj = JSON.parse(response[0])
            var tagger_obj = JSON.parse(response[1])

            if (costar_obj == null || typeof(costar_obj) != 'object'){
                alert("Empty or bad response.");
                this.analysis_done = true;
            }
            if (tagger_obj == null || typeof(tagger_obj) != 'object'){
                alert("Empty or bad response.");
                this.analysis_done = true;
            }
            //loop through individual results for highlights
            let indexOfResponse = 0
            for (const key in textarray){
                let i = parseInt(key)
                let len = textarray[i].split(" ");
                if (len.length <= 3) {
                    let list = ['No Bias words found here.'];
                    this.data.push(list);
                    output.push(`Skipped Sentence: "${textarray[i]}" is too short and requires more context`);
                    let stringout = output.join("\n");
                    this.highlight = { message: stringout, text: `Skipped Sentence: "${textarray[i]}" is too short and requires more context`, style: "background-color: transparent", highlightmsg: "none", probability: "low" };
                    this.msg = stringout;
                    let textOutput = document.getElementById('output');
                    textOutput.value = this.msg
                    continue;
                } else {
                    let word = tagger_obj[indexOfResponse]['words'][0]
                    let prob = tagger_obj[indexOfResponse]['probability'][0]
                    let epbias_type = tagger_obj[indexOfResponse]['epbias_type']
                    let epbias_def = tagger_obj[indexOfResponse]['epbias_def']
                    let epbias_link = tagger_obj[indexOfResponse]['epbias_link']
                    let epbias_alternative = tagger_obj[indexOfResponse]['alternative']
                    if (prob >= 0.5) {
                        console.log(typeof(costar_obj))
                        let stereotype = costar_obj[indexOfResponse]['stereotypes']

                        let s_distance = costar_obj[indexOfResponse]['stereotype_distances']
                        let costar_concept = costar_obj[indexOfResponse]['concepts']
                        let c_distance = costar_obj[indexOfResponse]['concept_distances']
                        let list = [word, prob, epbias_type, "none", "none", "none", "none", 'No extra information']
                        if (epbias_type == "regular") {
                            list = [word, prob,epbias_type, stereotype, s_distance, costar_concept, c_distance, 'This word has been associated as a ' + epbias_type + ' term. ' + epbias_def + '. For more information, please use the following link/s provided: ' + epbias_link]
                            this.highlight = { message: textarray[i], text: word, style: "background-color: transparent", highlightmsg: `A regular word`, probability: "low" } //no highlight required
                        }
                        else {
                            list = [word, prob, epbias_type, stereotype, s_distance, costar_concept,c_distance, 'This word has been associated as a/an ' + epbias_type + ' term. Which means that it ' + epbias_def + '. For more information, please use the following link/s provided: ' + epbias_link]

                            this.highlight = { message: textarray[i], text: word, style: "background-color:#FF0000", highlightmsg:  `A less subjective alternative might be: "${epbias_alternative}."`, probability: "high" } //highlight red
                            this.high_probability = true
                        }
                        this.data.push(list)
                        output.push(textarray[i] + ".")   
                    }
                    else if (prob > 0.1 && prob < 0.5) { //if the probability of the tagged word is not high enough, display only the explanable epistemological bias type to the user.
                        let list = []
                        if (epbias_type == "regular") {
                            list = [word, prob, epbias_type, "None", "None", "None", "None", 'This word might be associated as a/an ' + epbias_type + ' term. Although ' + epbias_def + '. For more information, please use the following link/s provided: ' + epbias_link]
                            this.highlight = { message: textarray[i], text: word, style: "background-color: transparent", highlightmsg: "None", probability: "medium" } //no highlight

                        } else {
                            list = [word, prob, epbias_type, "None", "None", "None", "None", 'This word might be associated as a/an ' + epbias_type + ' term. Which means ' + epbias_def + '. For more information, please use the following link/s provided: ' + epbias_link]

                            this.highlight = { message: textarray[i], text: word, style: "background-color:#FFFF00", highlightmsg: "Yellow is for probability lower than 0.5" + `\nAn alternative might be: "${epbias_alternative}"`, probability: "medium" } //highlight yellow
                            this.low_probability = true
                        }
                        this.data.push(list)
                        //output.push(`\n For sentence ${i + 1} the word tagged is ${word} with a bias probability score of ${prob}. This word was highlighted yellow because the probability of it been biased falls below a threshold of 0.5.`)
                        output.push(textarray[i] + ".")
                       
                    }
                    else { //if the probability of the tagged word is too low, there is no bias detected, tell the user.
                        let list = ['No Bias words found here.']
                        this.data.push(list)
                        output.push(`\n No bias detected in the following: ${textarray[i]} `)
                        this.highlight = { message: textarray[i], text: 'No bias detected in the following: ' + textarray[i], style: "background-color: transparent", highlightmsg: "none", probability: "low"} //no highlights
                    }

                    console.log("output", output)
                    let stringout = output.join("\n");
                    this.msg = stringout;
                    let textOutput = document.getElementById('output');
                    textOutput.value = this.msg
                    this.bias_shown = true
                    this.analysis_done = true
                    console.log("analysis done", this.analysis_done)
                    indexOfResponse ++    
                }
            }
        },
        async CheckBias() {
            Object.assign(this.$data, this.$options.data())
            this.progress_show = true
            this.response_data = true
            const textarea = document.getElementById('text');
            const textType = document.querySelector('input[name="group-1"]:checked').value
            let CheckText = textarea.value
            this.show = false
            if (CheckText.split(" ").length < 4) {
                this.placeholder = "Sentence is too short."
            } else {
                this.placeholder = CheckText;
                if (textType == 'value-1') {
                    var check_fullstop = CheckText.search(/\./);
                    if (check_fullstop == -1) {
                        CheckText = CheckText.concat(".");
                    }
                    const textarray = CheckText.split(".");
                    const indexToRemove = textarray.indexOf("");
                    textarray.splice(indexToRemove, 1);
                    this.getPredictions(textarray);
                }
                else if (textType == 'value-2') {
                    var check_fullstop = CheckText.search(/\./)
                    if (check_fullstop == -1) {
                        CheckText = CheckText.concat(".")
                    }
                    const textarray = []
                    const oldTextarray = CheckText.split("\n")
                    for (let i = 0; i < oldTextarray.length; i++) {
                        textarray.push(...oldTextarray[i].split("."))
                        let indexToRemove = textarray.indexOf("")
                        textarray.splice(indexToRemove, 1)
                    }
                    console.log(textarray)
                    this.getPredictions(textarray);
                }
                else {
                    let test = await this.$axios.$post('http://127.0.0.1:8080/api/v0/annotate/text', { 'docs': { 'lines': { 'text': CheckText } }, 'text_id': '<text_type>' }, { headers: { 'accept': 'application/json', 'Content-Type': 'application/json' } })
                }
                console.log("analysis_done",this.analysis_done);
                
            }

        },
        openModal() {
            this.visible = true;
            this.Title = "File Upload"
            console.log(this.modal)
        },
        closeModal(reloadPage) {
            this.visible = false
            this.visible_complex = false
        },
        actionHideRequest() {
            this.visible = false
            this.visible_complex = false
        },
        actionAfterHidden() {

        },
        SaveText() {
            this.unpackTextFromFile()
            this.visible = false
        },
        unpackTextFromFile() {
            this.files = this.$refs.files.internalFiles;
            if (this.files.length > 0) {
                console.log(this.files[0].file)
                const reader = new FileReader();
                reader.onload = function () {
                    console.log(reader.result)
                    if (reader.result != "") {
                        var text = reader.result
                        const textarea = document.getElementById('text');
                        textarea.value = text;
                        //TO DO: should automatically call this.CheckBias() afterwards
                    }
                }
                reader.readAsText(this.files[0].file);
            }
        },
        // loadingShow(){
        //     if (this.analysis_done){
        //         this.loadingActive = false
        //     } else{
        //         this.loadingActive = true
        //     }

        // },
        
        ShowComplex() {
            this.visible_complex = true;
            this.Title = "Bias's found"
            console.log(this.Title)
        },
        actionHidden() {
            this.modal = false
            this.modelComplex = false
        },
        async url() {
            Object.assign(this.$data, this.$options.data())
            this.progress_show = true
            this.response_data = true
            const textarea = document.getElementById('text');
            const textType = document.querySelector('input[name="group-1"]:checked').value
            let CheckText = textarea.value
            this.show = false
            this.placeholder = CheckText;

            // let test_url_headline = await this.$axios.$post('/predict/get_headline_from_url', { 'url': CheckText }, { headers: { 'accept': 'application/json', 'Content-Type': 'application/json' } })
            await this.$axios.post('/predict/get_headline_from_url', { 'url': CheckText }, { headers: { 'accept': 'application/json', 'Content-Type': 'application/json' } })
                .then((response) => {
                    console.log('response: ', response)
                    const data = response.data;
                    let url_text = JSON.parse(data)['text'];
                    console.log('This is the headline from given url:', url_text[0])
                    if (textType == 'value-1') {
                        textarea.value = url_text[0];
                    } else {
                        let stringout = url_text.toString();
                        stringout = stringout.replaceAll(',', '');
                        textarea.value = stringout;
                    }
                    this.CheckBias();
                    if (!response.statusText.toLowerCase() == "ok") {
                        const error = (data && data.message) || response.status;
                        console.log("Error status code", response.status)
                        return Promise.reject(error);
                        // add reload the window
                    }
                })
                .catch(error => {
                    console.error(error);
                    textarea.value = error;
                });
        }

    }
}
</script>
<style>

.title{
    padding: 2rem;
    color: black;
}

.body{
    padding: 2rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.top-body{
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    padding: 0px;
    gap: 8px;
    align-items: center;
}

.analyse-area{
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.buttons{
    display: flex;
    flex-direction: row;
    justify-content: flex-end;
    padding-bottom: 3rem;
    /* padding-top: 20rem; */
    /* padding: 1rem; */
}

.url-button{
    padding-right: 1rem;
}

.pre-format {
    white-space: pre-wrap;
}
.text-area {
   resize: none;
   height: 10rem;
   display: flex;
   justify-content: flex-start;
   padding-bottom: 1rem;
   /* padding: 1rem; */
   /* min-height: 10rem; */
   /* max-height : 500px; */
   /* max-width : 500px; */
}
.output-header{
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}
.output-area{
    box-sizing: border-box;
    border: 1px solid #C6C6C6;
    border-style: solid;
    min-height: 8.5rem;
    height: fit-content;
    display: flex;

}

.output-box{
    display: flex;
    align-items: flex-start;

}
.output-text{

    justify-content: flex-start;

}
.loading{
    display: flex;

    align-items: center;
    justify-content: flex-end;
    
}
/* Tooltip container */
.tooltip {
    position: relative;
    display: inline-block;
    /* If you want dots under the hoverable text */
}

/* Tooltip text */
.tooltip .tooltiptext {
    visibility: hidden;
    width: 300px;
    background-color: black;
    font-size: 0.875rem;
    font-weight: 400;
    color: #fff;
    text-align: center;
    padding: 5px 0;
    border-radius: 6px;

    /* to position at top, margin-left set to .5 of width value */
    bottom: 100%;
    left: 30%;
    margin-left: -150px;

    /* Position the tooltip text - see examples below! */
    position: absolute;
    z-index: 1;
}

/* Show the tooltip text when you mouse over the tooltip container */
.tooltip:hover .tooltiptext {
    visibility: visible;
}

#output{
    width: max-content;
}
</style>
