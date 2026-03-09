document.addEventListener("DOMContentLoaded", function(){

// Complexity Chart
const ctx1 = document.getElementById('complexityChart');

window.complexityChart = new Chart(ctx1, {
  type: 'bar',
  data: {
    labels: ["Baseline", "Specialized"],
    datasets: [{
      label: "Predictors",
      data: [17.5, null],
      backgroundColor: ["#c9ced6", "#3b7ddd"]
    }]
  },
  options: {
    plugins:{
        legend:{
            display: false
        }
    },
    scales: {
      y: { beginAtZero: true }
    }
  }
});


// MSE Chart
const ctx2 = document.getElementById('mseChart');

window.mseChart = new Chart(ctx2, {
  type: 'bar',
  data: {
    labels: ["Baseline", "Specialized"],
    datasets: [{
      label: "Test MSE",
      data: [0.895, null],
      backgroundColor: ["#c9ced6", "#3b7ddd"]
    }]
  },
  options: {
    plugins:{
        legend:{
            display: false
        }
    },
    scales: {
      y: {
        min: 0.85,
        max: 0.90
      }
    }
  }
});

});

function showComplexity(){
  complexityChart.data.datasets[0].data[1] = 3.8;
  complexityChart.update();
}

function showMSE(){
  mseChart.data.datasets[0].data[1] = 0.877;
  mseChart.update();
}