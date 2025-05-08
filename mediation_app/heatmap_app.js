// Configuration
const config = {
    margin: { top: 80, right: 50, bottom: 50, left: 150 },
    cellSize: 40,
    colors: ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"],
    transitionDuration: 500
};

// État de l'application
const state = {
    data: [],
    selectedTerm: "informatique",
    selectedPeriods: ["1960s", "1970s", "1980s", "1990s"],
    similarityThreshold: 0.6,
    vizType: "heatmap"
};

// Chargement des données
document.addEventListener('DOMContentLoaded', () => {
    // Charger les données
    loadData();
    
    // Initialiser les contrôles
    initControls();
});

// Chargement des données
function loadData() {
    // Chemin vers le fichier CSV (pour compatibilité)
    const csvPath = 'data/results/exports/collections/heatmap_similars/bd607f58-f47b-4c3e-9ef0-1ca126b15d2f/source_files/similar_terms_term_tracking_results.csv';
    
    // Chemin vers le fichier Parquet (nouvelle version)
    const parquetPath = csvPath.replace('.csv', '.parquet');
    
    // Vérifier si le fichier Parquet existe
    fetch(parquetPath, { method: 'HEAD' })
        .then(response => {
            if (response.ok) {
                // Le fichier Parquet existe, on l'utilise
                console.log('Chargement des données depuis Parquet...');
                return loadParquetData(parquetPath);
            } else {
                // Sinon, on utilise le CSV comme avant
                console.log('Fichier Parquet non trouvé, utilisation du CSV...');
                return d3.csv(csvPath);
            }
        })
        .then(data => {
            // Stocker les données
            state.data = data;
            
            // Créer la visualisation
            createVisualization();
        })
        .catch(error => {
            console.error('Erreur lors du chargement des données:', error);
            document.getElementById('heatmap-container').innerHTML = 
                '<div class="error-message">Erreur lors du chargement des données. Veuillez réessayer.</div>';
        });
}

// Fonction pour charger les données depuis un fichier Parquet
async function loadParquetData(parquetPath) {
    try {
        // Vérifier si Apache Arrow est disponible
        if (typeof arrow === 'undefined') {
            // Charger Apache Arrow si nécessaire
            await loadScript('https://cdn.jsdelivr.net/npm/apache-arrow@latest/Arrow.es2015.min.js');
        }
        
        // Charger le fichier Parquet
        const response = await fetch(parquetPath);
        const arrayBuffer = await response.arrayBuffer();
        
        // Utiliser Apache Arrow pour lire le fichier Parquet
        const table = await arrow.Table.from(new Uint8Array(arrayBuffer));
        
        // Convertir la table Arrow en format compatible avec d3
        return table.toArray().map(row => {
            const obj = {};
            table.schema.fields.forEach((field, i) => {
                obj[field.name] = row[i];
            });
            return obj;
        });
    } catch (error) {
        console.error('Erreur lors du chargement du fichier Parquet:', error);
        // En cas d'erreur, on revient au CSV
        return d3.csv(parquetPath.replace('.parquet', '.csv'));
    }
}

// Fonction pour charger un script externe
function loadScript(src) {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = src;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

// Initialisation des contrôles
function initControls() {
    // Sélection du terme de référence
    document.querySelectorAll('input[name="reference-term"]').forEach(input => {
        input.addEventListener('change', e => {
            state.selectedTerm = e.target.value;
            createVisualization();
        });
    });
    
    // Sélection des périodes
    document.querySelectorAll('input[name="period"]').forEach(input => {
        input.addEventListener('change', e => {
            if (e.target.checked) {
                if (!state.selectedPeriods.includes(e.target.value)) {
                    state.selectedPeriods.push(e.target.value);
                }
            } else {
                state.selectedPeriods = state.selectedPeriods.filter(period => period !== e.target.value);
            }
            createVisualization();
        });
    });
    
    // Sélection du type de visualisation
    document.getElementById('viz-type').addEventListener('change', e => {
        state.vizType = e.target.value;
        createVisualization();
    });
    
    // Sélection du seuil de similarité
    const similaritySlider = document.getElementById('similarity-slider');
    const similarityValue = document.getElementById('similarity-value');
    
    similaritySlider.addEventListener('input', e => {
        state.similarityThreshold = parseFloat(e.target.value);
        similarityValue.textContent = state.similarityThreshold.toFixed(2);
        createVisualization();
    });
}

// Création de la visualisation
function createVisualization() {
    // Filtrer les données selon les sélections
    const filteredData = state.data.filter(d => {
        return d.term === state.selectedTerm && 
               state.selectedPeriods.includes(d.period) &&
               parseFloat(d.similarity) >= state.similarityThreshold;
    });
    
    // Choisir le type de visualisation
    if (state.vizType === "heatmap") {
        createHeatmap(filteredData);
    } else {
        createNetworkGraph(filteredData);
    }
}

// Création de la heatmap
function createHeatmap(data) {
    // Vider le conteneur
    const container = document.getElementById('heatmap-container');
    container.innerHTML = '';
    
    // Si aucune donnée, afficher un message
    if (data.length === 0) {
        container.innerHTML = '<div class="no-data-message">Aucune donnée ne correspond aux critères sélectionnés.</div>';
        return;
    }
    
    // Dimensions
    const width = container.clientWidth;
    const height = Math.max(500, data.length * config.cellSize / 2);
    
    // Créer le SVG
    const svg = d3.select('#heatmap-container')
        .append('svg')
        .attr('width', width)
        .attr('height', height);
    
    // Groupe principal avec marge
    const g = svg.append('g')
        .attr('transform', `translate(${config.margin.left},${config.margin.top})`);
    
    // Extraire les périodes et les termes uniques
    const periods = Array.from(new Set(data.map(d => d.period))).sort();
    const terms = Array.from(new Set(data.map(d => d.similar_word)));
    
    // Trier les termes par similarité moyenne
    terms.sort((a, b) => {
        const avgSimilarityA = d3.mean(data.filter(d => d.similar_word === a).map(d => +d.similarity));
        const avgSimilarityB = d3.mean(data.filter(d => d.similar_word === b).map(d => +d.similarity));
        return avgSimilarityB - avgSimilarityA;
    });
    
    // Échelles
    const x = d3.scaleBand()
        .domain(periods)
        .range([0, Math.min(width - config.margin.left - config.margin.right, periods.length * config.cellSize)])
        .padding(0.1);
    
    const y = d3.scaleBand()
        .domain(terms)
        .range([0, Math.min(height - config.margin.top - config.margin.bottom, terms.length * config.cellSize)])
        .padding(0.1);
    
    const color = d3.scaleSequential()
        .domain([state.similarityThreshold, 1])
        .interpolator(d3.interpolateInferno);
    
    // Créer les cellules de la heatmap
    g.selectAll('.heatmap-cell')
        .data(data)
        .enter()
        .append('rect')
        .attr('class', 'heatmap-cell')
        .attr('x', d => x(d.period))
        .attr('y', d => y(d.similar_word))
        .attr('width', x.bandwidth())
        .attr('height', y.bandwidth())
        .attr('fill', d => color(+d.similarity))
        .attr('data-period', d => d.period)
        .attr('data-term', d => d.similar_word)
        .attr('data-similarity', d => d.similarity)
        .on('mouseover', function(event, d) {
            // Mettre en évidence la cellule
            d3.select(this)
                .attr('stroke', 'white')
                .attr('stroke-width', 2);
            
            // Afficher le tooltip
            const tooltip = d3.select('#tooltip');
            tooltip.style('display', 'block')
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 28) + 'px')
                .html(`
                    <strong>${d.similar_word}</strong><br>
                    Période: ${d.period}<br>
                    Similarité avec "${state.selectedTerm}": ${(+d.similarity).toFixed(3)}<br>
                    Rang: ${d.rank}
                `);
        })
        .on('mouseout', function() {
            // Restaurer la cellule
            d3.select(this)
                .attr('stroke', '#1e1e1e')
                .attr('stroke-width', 1);
            
            // Masquer le tooltip
            d3.select('#tooltip').style('display', 'none');
        });
    
    // Ajouter les axes
    const xAxis = g.append('g')
        .attr('class', 'axis x-axis')
        .attr('transform', `translate(0,${y.range()[1] + 10})`)
        .call(d3.axisBottom(x));
    
    const yAxis = g.append('g')
        .attr('class', 'axis y-axis')
        .attr('transform', `translate(-10,0)`)
        .call(d3.axisLeft(y));
    
    // Ajouter un titre
    svg.append('text')
        .attr('class', 'chart-title')
        .attr('x', width / 2)
        .attr('y', 30)
        .attr('text-anchor', 'middle')
        .attr('fill', '#ccc')
        .style('font-size', '16px')
        .text(`Termes similaires à "${state.selectedTerm}" par période`);
    
    // Mettre à jour la légende
    updateLegend(state.similarityThreshold, 1);
}

// Création du graphe en réseau
function createNetworkGraph(data) {
    // Vider le conteneur
    const container = document.getElementById('heatmap-container');
    container.innerHTML = '';
    
    // Si aucune donnée, afficher un message
    if (data.length === 0) {
        container.innerHTML = '<div class="no-data-message">Aucune donnée ne correspond aux critères sélectionnés.</div>';
        return;
    }
    
    // Dimensions
    const width = container.clientWidth;
    const height = 500;
    
    // Créer le SVG
    const svg = d3.select('#heatmap-container')
        .append('svg')
        .attr('width', width)
        .attr('height', height);
    
    // Créer les nœuds et les liens
    const nodes = [];
    const links = [];
    
    // Ajouter le terme de référence
    nodes.push({
        id: state.selectedTerm,
        group: "reference",
        size: 20
    });
    
    // Ajouter les termes similaires
    data.forEach(d => {
        // Vérifier si le nœud existe déjà
        const existingNode = nodes.find(node => node.id === d.similar_word);
        
        if (!existingNode) {
            nodes.push({
                id: d.similar_word,
                group: d.period,
                size: 10
            });
        }
        
        // Ajouter le lien
        links.push({
            source: state.selectedTerm,
            target: d.similar_word,
            value: +d.similarity,
            period: d.period
        });
    });
    
    // Créer la simulation
    const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(d => 200 * (1 - d.value)))
        .force('charge', d3.forceManyBody().strength(-200))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(d => d.size * 1.5));
    
    // Créer les liens
    const link = svg.append('g')
        .selectAll('line')
        .data(links)
        .enter()
        .append('line')
        .attr('stroke-width', d => d.value * 3)
        .attr('stroke', d => {
            // Couleur selon la période
            switch(d.period) {
                case '1960s': return '#4e79a7';
                case '1970s': return '#f28e2c';
                case '1980s': return '#e15759';
                case '1990s': return '#76b7b2';
                default: return '#ccc';
            }
        })
        .attr('opacity', 0.6);
    
    // Créer les nœuds
    const node = svg.append('g')
        .selectAll('circle')
        .data(nodes)
        .enter()
        .append('circle')
        .attr('r', d => d.size)
        .attr('fill', d => {
            if (d.group === "reference") {
                return '#fff';
            }
            
            // Couleur selon la période
            switch(d.group) {
                case '1960s': return '#4e79a7';
                case '1970s': return '#f28e2c';
                case '1980s': return '#e15759';
                case '1990s': return '#76b7b2';
                default: return '#ccc';
            }
        })
        .attr('stroke', d => d.group === "reference" ? '#333' : 'none')
        .attr('stroke-width', 2)
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));
    
    // Ajouter les étiquettes
    const label = svg.append('g')
        .selectAll('text')
        .data(nodes)
        .enter()
        .append('text')
        .text(d => d.id)
        .attr('font-size', d => d.group === "reference" ? 14 : 10)
        .attr('dx', d => d.group === "reference" ? 0 : 12)
        .attr('dy', 4)
        .attr('text-anchor', d => d.group === "reference" ? 'middle' : 'start')
        .attr('fill', '#ccc');
    
    // Ajouter un titre
    svg.append('text')
        .attr('class', 'chart-title')
        .attr('x', width / 2)
        .attr('y', 30)
        .attr('text-anchor', 'middle')
        .attr('fill', '#ccc')
        .style('font-size', '16px')
        .text(`Réseau des termes similaires à "${state.selectedTerm}"`);
    
    // Ajouter une légende
    const legend = svg.append('g')
        .attr('class', 'legend')
        .attr('transform', `translate(20, 60)`);
    
    const periods = ['1960s', '1970s', '1980s', '1990s'];
    const colors = ['#4e79a7', '#f28e2c', '#e15759', '#76b7b2'];
    
    periods.forEach((period, i) => {
        legend.append('circle')
            .attr('cx', 10)
            .attr('cy', i * 25)
            .attr('r', 6)
            .attr('fill', colors[i]);
        
        legend.append('text')
            .attr('x', 25)
            .attr('y', i * 25 + 4)
            .text(period)
            .attr('fill', '#ccc')
            .attr('font-size', 12);
    });
    
    // Tooltip
    node.on('mouseover', function(event, d) {
        // Mettre en évidence le nœud
        d3.select(this)
            .attr('stroke', 'white')
            .attr('stroke-width', 2);
        
        // Mettre en évidence les liens connectés
        link.filter(l => l.source.id === d.id || l.target.id === d.id)
            .attr('stroke', 'white')
            .attr('stroke-width', 3)
            .attr('opacity', 1);
        
        // Afficher le tooltip
        if (d.group !== "reference") {
            const linkedData = links.find(l => l.target.id === d.id);
            
            const tooltip = d3.select('#tooltip');
            tooltip.style('display', 'block')
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 28) + 'px')
                .html(`
                    <strong>${d.id}</strong><br>
                    Période: ${d.group}<br>
                    Similarité avec "${state.selectedTerm}": ${linkedData ? linkedData.value.toFixed(3) : 'N/A'}
                `);
        }
    })
    .on('mouseout', function(event, d) {
        // Restaurer le nœud
        d3.select(this)
            .attr('stroke', d.group === "reference" ? '#333' : 'none')
            .attr('stroke-width', d.group === "reference" ? 2 : 0);
        
        // Restaurer les liens
        link.attr('stroke', l => {
            switch(l.period) {
                case '1960s': return '#4e79a7';
                case '1970s': return '#f28e2c';
                case '1980s': return '#e15759';
                case '1990s': return '#76b7b2';
                default: return '#ccc';
            }
        })
        .attr('stroke-width', d => d.value * 3)
        .attr('opacity', 0.6);
        
        // Masquer le tooltip
        d3.select('#tooltip').style('display', 'none');
    });
    
    // Fonctions pour le glisser-déposer
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
    
    // Mettre à jour la position des éléments à chaque tick
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        
        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
        
        label
            .attr('x', d => d.x)
            .attr('y', d => d.y);
    });
}

// Mise à jour de la légende
function updateLegend(min, max) {
    const colorScale = d3.select('.color-scale');
    const gradient = d3.scaleSequential()
        .domain([0, 1])
        .interpolator(d3.interpolateInferno);
    
    // Mettre à jour les étiquettes
    const scaleLabels = document.querySelector('.scale-labels');
    scaleLabels.innerHTML = `
        <span>Similarité: ${min.toFixed(2)}</span>
        <span>${max.toFixed(2)}</span>
    `;
}
