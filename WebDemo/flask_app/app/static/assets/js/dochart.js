
// doughut chart
const ctx = document.getElementById('doughutChart')
ctx.height = 150
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