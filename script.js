// Define variables/nodes from your dataframe
const nodesArray = [
  { id: 1, label: 'Crude Oil', group: 'market' },
  { id: 2, label: 'Gold', group: 'market' },
  { id: 3, label: 'Energy', group: 'market' },
  { id: 4, label: 'Financial Sector', group: 'market' },
  { id: 5, label: 'Tech Sector', group: 'market' },
  { id: 6, label: 'S&P 500', group: 'market' },
  { id: 7, label: '10Y Treasury Yield', group: 'market' },
  { id: 8, label: 'Unemployment Rates', group: 'macro' },
  { id: 9, label: 'Inflation Rates', group: 'macro' },
  { id: 10, label: 'Interest Rates', group: 'macro' },
  { id: 11, label: 'Yield Curve', group: 'macro' },
  { id: 12, label: 'Volatility Index', group: 'macro' },
  { id: 13, label: 'Credit Risk', group: 'macro' }
];

// visual of baseline edges 
const baselineEdges = [
  { from: 1, to: 3 }, { from: 1, to: 6 }, { from: 2, to: 6 }, { from: 2, to: 4 },
  { from: 3, to: 4 }, { from: 4, to: 6 }, { from: 5, to: 6 }, { from: 5, to: 3 },
  { from: 7, to: 10 }, { from: 8, to: 6 }, { from: 9, to: 10 }, { from: 8, to: 9 },
  { from: 10, to: 7 }, { from: 10, to: 4 }, { from: 11, to: 4 }, { from: 9, to: 11 },
  { from: 12, to: 6 }, { from: 13, to: 4 }, { from: 12, to: 13 }, { from: 1, to: 12 },
  { from: 3, to: 7 }, { from: 8, to: 13 }, { from: 11, to: 2 }, { from: 10, to: 12 }
];

// specialized edges 
const specializedEdges = [
  { from: 1, to: 3, arrows: 'to', color: { color: '#16a34a' } },  // Oil -> Energy
  { from: 10, to: 7, arrows: 'to', color: { color: '#16a34a' } }, // FedFunds -> TNX
  { from: 8, to: 10, arrows: 'to', color: { color: '#16a34a' } }, // Unemployment -> FedFunds
  { from: 12, to: 6, arrows: 'to', color: { color: '#16a34a' } }, // Volatility -> S&P 500
  { from: 9, to: 10, arrows: 'to', color: { color: '#16a34a' } }, // Inflation -> FedFunds
  { from: 7, to: 4, arrows: 'to', color: { color: '#16a34a' } },  // TNX -> Financials
];

let network = null;

// func to draw graph based on selection
function drawGraph(type) {
  // update button active states
  document.getElementById('btn-baseline').classList.remove('active');
  document.getElementById('btn-specialized').classList.remove('active');
  document.getElementById(`btn-${type}`).classList.add('active');

  const container = document.getElementById('network-container');
  
  const data = {
    nodes: new vis.DataSet(nodesArray),
    edges: new vis.DataSet(type === 'baseline' ? baselineEdges : specializedEdges)
  };

  const options = {
    nodes: {
      shape: 'dot',
      size: 16,
      font: { size: 14, face: 'Inter', color: '#334155' },
      borderWidth: 2,
    },
    edges: {
      width: 2,
      color: { color: '#cbd5e1', inherit: false },
      smooth: { type: 'continuous' }
    },
    groups: {
      market: { color: { background: '#bfdbfe', border: '#3b82f6' } },
      macro: { color: { background: '#fef08a', border: '#eab308' } }
    },
    physics: {
      barnesHut: {
        gravitationalConstant: -2000,
        centralGravity: 0.3,
        springLength: 150
      }
    }
  };

  if (network !== null) {
    network.destroy();
  }
  network = new vis.Network(container, data, options);
}

// initialize baseline graph when first load
window.onload = () => {
  drawGraph('baseline');
};