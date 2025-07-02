const { Readability } = require('@mozilla/readability');
var express = require('express');
var router = express.Router();
var { JSDOM } = require('jsdom');


/* GET home page. */
router.post('/', function(req, res) {
  try{
    const receivedDocument = new JSDOM(req.body.text)
    const object = new Readability(receivedDocument.window.document).parse()
    const byline = object.byline
    const filteredDocument = object.content
    const excerpt = object.excerpt
    data = {
      doc: filteredDocument, 
      byline: byline, 
      excerpt: excerpt
    }
    res.send(data)
  }
  catch(error){
    res.status(400).send()
  }
  
});

module.exports = router;
