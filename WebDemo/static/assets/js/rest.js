async function post_data(url, data, wrapper)
{
    const response = await fetch(url, {
        method: 'POST',
        headers:
        {
            'Content-type': 'application/json'
        },
        body: JSON.stringify(
            {
                source: wrapper(data)
            }),
    });
    return await response.json();
};

document.getElementById('translate').onclick = function() {

    let doc_input = document.getElementById('doc_input').value;
    // if types in chinese or not
    if(ch_test(doc_input))
    {
        document.getElementById('source_label').innerHTML ="Please use English.";
        alert("Please type words in English.");
        return;
    };
    let tokens = doc_input.split(" "); 
    // type enough length text
    if(length_test(tokens))
    {
        document.getElementById('source_label').innerHTML ="Please fill in the textarea with more than 5 tokens.";
        alert("Please fill in the textarea with more than 5 tokens.");
        return;
    };
    document.getElementById('summary_output').innerHTML = "";
    document.getElementById('source_label').innerHTML = "Loading ...";
    $('.loader').show();

    post_data(url='/run_decode', data=tokens, wrapper=update_text)
        .then((data) => 
        {
            let hyp_words = doc_input.replace("\n", " ").split(" ");
            let ref_set = new Set(data['final'].replace("\n", " ").split(" ")); 
            let ans = "";
            let sum = data['final'].replace("\n","<br>");
            let ind = data['sentNum'];
            $('.loader').hide();
            document.getElementById('summary_output').innerHTML = sum;
            document.getElementById('source_label').innerHTML = "Inpur your document";
            for(let ref_words of ref_set){
                for(let i = 0; i < hyp_words.length; i++) 
                {
                    if(hyp_words[i].toLowerCase() == ref_words.toLowerCase())
                    {
                        hyp_words[i] = "<span style=\"background-color: yellow\">"+ ref_words +"</span>";
                    };
                };
              };
            for(let i = 0; i < hyp_words.length; i++)
            {
                ans += hyp_words[i] + " ";
            };
            document.getElementById('summary_output').innerHTML += "<hr><b>OriginalText</b><br><br><div>" + ans + "</div>";
        })
        .catch(function(err)
        {
            console.error(err);
            alert(JSON.stringify(err['message']));
        });
};

document.getElementById('history').onclick = function(e){
    const tr = e.target.closest('tr');
    let answer = window.confirm('Are you sure to delete this record?');
    if(answer){
        tr.remove();
    };
};

window.addEventListener('DOMContentLoaded', function(){
    $('#history').dataTable();
});

document.addEventListener('click', function(e){
    if (e.target.matches('.dataset')){
        window.open('https://github.com/ChenRocks/cnn-dailymail', '_blank');
    };
});

function resizeIframe(obj) {
    obj.style.height = obj.contentWindow.document.documentElement.scrollHeight + 'px';
};
function ch_test(doc_inputs) {
    let s = doc_inputs;
    for(let i = 0; i < s.length; i++) {
        if(s.charCodeAt(i) >= 0x4E00 && s.charCodeAt(i) <= 0x9FA5) {
            return true;
        };
    };
    return false;
};
function length_test(doc_inputs){
    if(doc_inputs.length <= 5) return true;
    else return false;
};
function update_text(tokens){
    let sep  = false;
    let new_doc_input = "";

    for(let i = 0; i < tokens.length; i++){
        if(tokens[i].includes("\n")){
            if(!sep) new_doc_input += "\n";
            else new_doc_input += tokens[i].replace("\n","") + ' ';
            sep = true;
        }
        else {
            new_doc_input += tokens[i] + ' ';
            sep = false;
        }
    };
    return new_doc_input.substring(0, new_doc_input.length - 1);
};




//doughut chart
let ctx = document.getElementById( "doughutChart" );
ctx.height = 150;
let myChart = new Chart(ctx, {
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
});