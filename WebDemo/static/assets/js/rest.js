
document.getElementById('translate').onclick = function() {

    var doc_input = document.getElementById('doc_input').value;
   
    // Whether chinese ?
    if(ch_test(doc_input)){
        document.getElementById('source_label').innerHTML ="Please use English";
        return
    }
    // For scrolling in CNN
    var tokens = doc_input.split(" "); 
    if(tokens.length<=5){
        document.getElementById('source_label').innerHTML ="Please input data with more than 5 tokens";
        return
    };

    // main function 
    document.getElementById('summary_output').innerHTML = "";
    $(".loader").show();
    document.getElementById('source_label').innerHTML ="Loading ...";
    fetch('/run_decode',{
        method: 'POST',
        headers: { 'Content-type': 'application/json'},
        body: JSON.stringify({
            source: update_text(tokens)
            }),
        })
        .then(response => response.json())
        .then(function(data){
            var hyp_words = doc_input.replace("\n", " ").split(" ");
            var ref_set = new Set(data['final'].replace("\n", " ").split(" ")); 
            var ans = "";
            var sum = data['final'].replace("\n","<br>");
            var ind = data['sentNum'];
            $(".loader").hide();
            document.getElementById('summary_output').innerHTML = sum;
            document.getElementById('source_label').innerHTML = "Inpur your document";
            for(let ref_words of ref_set){
                for(var a=0;a<hyp_words.length;a++){
                    if(hyp_words[a].toLowerCase() == ref_words.toLowerCase()){
                        hyp_words[a] = "<span style=\"background-color: yellow\">"+ ref_words +"</span>";
                    }
                }
              }
            for(var a=0;a<hyp_words.length;a++){
                ans+=hyp_words[a] + " ";
            }

            document.getElementById('summary_output').innerHTML += "<hr><b>OriginalText</b><br><br><div>" + ans + "</div>";
            // document.getElementById('summary_output').innerHTML += "<hr><b>OriginalText</b><br><br><div>" + doc_input + "</div>";
        })
        .catch(function(err){
            console.log(err);
    });
}

//doughut chart
var ctx = document.getElementById( "doughutChart" );
ctx.height = 150;
var myChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
        datasets: [ {
            data: [ 1143808, 3994412
                , 2368257, 2369282
            ],
            backgroundColor: [
                                "#ab8ce4",
                                "#00c292",
                                "#03a9f3",
                                "#fb9678"
                            ],
            hoverBackgroundColor: [
                                "#ab8ce4",
                                "#00c292",
                                "#03a9f3",
                                "#fb9678"
                            ]

                        } ],
            labels: [
                        "Document Encoder",
                        "Sentence Encoder",
                        "Extractor",
                        "Abstractor"
                    ]
    },
    options: {
        responsive: true
    }
} );

$(document).ready(function() { 
   
    $(".card" ).hover(
    function() {
       $(this).addClass('shadow-lg').css('cursor', 'pointer');
       $(this).animate({ 'zoom': 1.1}, 300);
     }, 
     function() {
       $(this).removeClass('shadow-lg');
       $(this).animate({ 'zoom': 1 }, 300);
     }, 
    );
    $(".data").click(function(){
        window.open('https://github.com/ChenRocks/cnn-dailymail', '_blank');
    })
});


function ch_test(doc_inputs) {
    var s = doc_inputs;
    for(var i = 0; i < s.length; i++) {
        if(s.charCodeAt(i) >= 0x4E00 && s.charCodeAt(i) <= 0x9FA5) {
            return true;
        }
    }
    return false
};

function update_text(tokens){
    var sep  = false;
    var new_doc_input = "";

    for(var i=0; i<tokens.length;i++){
        if(tokens[i].includes("\n")){
            if(!sep) new_doc_input += "\n";
            else new_doc_input += tokens[i].replace("\n","") + ' ';
            sep = true;
        }
        else {
            new_doc_input += tokens[i] + ' ';
            sep = false;
        }
    }
    return new_doc_input.substring(0, new_doc_input.length - 1);
};
