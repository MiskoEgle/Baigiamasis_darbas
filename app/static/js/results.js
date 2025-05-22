document.addEventListener('DOMContentLoaded', function() {
    if (!window.chartResults) return;
    window.chartResults.forEach(function(result, idx) {
        // Tikslumo grafikas
        const accCtx = document.getElementById('accuracyChart' + (idx + 1));
        if (accCtx) {
            new Chart(accCtx, {
                type: 'line',
                data: {
                    labels: ['Mokymo', 'Validacijos'],
                    datasets: [{
                        label: 'Tikslumas',
                        data: [
                            result.rezultatai.tikslumas_mokymo !== undefined ? result.rezultatai.tikslumas_mokymo : null,
                            result.rezultatai.tikslumas_validacijos !== undefined ? result.rezultatai.tikslumas_validacijos : null
                        ],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Modelio Tikslumas'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1
                        }
                    }
                }
            });
        }
        // Nuostolių grafikas
        const lossCtx = document.getElementById('lossChart' + (idx + 1));
        if (lossCtx) {
            new Chart(lossCtx, {
                type: 'line',
                data: {
                    labels: ['Mokymo', 'Validacijos'],
                    datasets: [{
                        label: 'Nuostoliai',
                        data: [
                            result.rezultatai.nuostoliai_mokymo !== undefined ? result.rezultatai.nuostoliai_mokymo : null,
                            result.rezultatai.nuostoliai_validacijos !== undefined ? result.rezultatai.nuostoliai_validacijos : null
                        ],
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Modelio Nuostoliai'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }
    });
}); 