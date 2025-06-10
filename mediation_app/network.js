// topic_network.js (version finale avec affichage des articles)

document.addEventListener('DOMContentLoaded', () => {
    initTopicNetwork();
});

// --- CONFIGURATION ---
const config = {
    clusteringResultsPath: 'data/source/collections/topic_modeling/bde3ace6-289e-409e-a312-9e301efabc42/source_files/doc_clusters_k25.json',
    topicNamesPath: 'data/source/collections/topic_modeling/bde3ace6-289e-409e-a312-9e301efabc42/source_files/topic_names_llm_gensim_lda_20250524-163956_93a8c0ec.json',
    // **AJOUT : Chemins vers les données des articles**
    docTopicsPath: 'data/source/collections/topic_modeling/bde3ace6-289e-409e-a312-9e301efabc42/source_files/doc_topic_matrix_gensim_lda_20250524-163956_93a8c0ec.json',
    articlesPath: 'data/source/articles_v1_filtered.json',
    
    linkThreshold: 0.01,
    chargeStrength: -50,
    centerStrength: 0.1,
};

// --- APPLICATION STATE ---
let state = {
    clusteringResults: null,
    topicNames: null,
    // **AJOUT : États pour les données des articles**
    docTopics: null,
    articles: null,
    nodes: [],
    links: [],
    simulation: null,
    svg: null,
    tooltip: null,
    zoom: null,
};

// --- INITIALIZATION ---
async function initTopicNetwork() {
    state.tooltip = d3.select('#tooltip');
    try {
        await loadData();
        processDataForNetwork();
        createNetworkGraph();
        setupZoomSlider();
        initModal(); // **AJOUT : Initialisation de la modale**
    } catch (error) {
        console.error('Initialization failed:', error);
        document.getElementById('network-container').innerHTML =
            `<p style="color:red; text-align:center; padding: 2rem;">Error: Could not load or process network data. ${error.message}</p>`;
    }
}

// --- DATA LOADING ---
async function loadData() {
    // **MODIFICATION : Recharger TOUTES les données nécessaires**
    const [clusteringRes, topicNamesRes, docTopicsRes, articlesRes] = await Promise.all([
        fetch(config.clusteringResultsPath),
        fetch(config.topicNamesPath),
        fetch(config.docTopicsPath),
        fetch(config.articlesPath),
    ]);
    if (!clusteringRes.ok) throw new Error(`Failed to load clustering results from ${config.clusteringResultsPath}`);
    if (!topicNamesRes.ok) throw new Error(`Failed to load topic names from ${config.topicNamesPath}`);
    if (!docTopicsRes.ok) throw new Error(`Failed to load doc topics from ${config.docTopicsPath}`);
    if (!articlesRes.ok) throw new Error(`Failed to load articles from ${config.articlesPath}`);

    state.clusteringResults = await clusteringRes.json();
    state.topicNames = (await topicNamesRes.json()).topic_names;
    state.docTopics = (await docTopicsRes.json()).doc_topics;
    state.articles = await articlesRes.json();
}

// --- LOGIQUE CENTRALE (inchangée) ---
function processDataForNetwork() {
    const centers = state.clusteringResults.cluster_centers;
    if (!centers || centers.length === 0) throw new Error("`cluster_centers` not found.");
    const numTopics = centers[0].length;
    const similarityMatrix = Array(numTopics).fill(0).map(() => Array(numTopics).fill(0));
    for (const clusterVector of centers) {
        for (let i = 0; i < numTopics; i++) {
            for (let j = i; j < numTopics; j++) {
                const similarityInCluster = clusterVector[i] * clusterVector[j];
                similarityMatrix[i][j] += similarityInCluster;
                if (i !== j) similarityMatrix[j][i] += similarityInCluster;
            }
        }
    }
    state.nodes = [];
    for (let i = 0; i < numTopics; i++) {
        state.nodes.push({
            id: i.toString(),
            name: state.topicNames[i.toString()] || `Topic ${i}`,
            importance: similarityMatrix[i][i],
        });
    }
    state.links = [];
    for (let i = 0; i < numTopics; i++) {
        for (let j = i + 1; j < numTopics; j++) {
            const similarity = similarityMatrix[i][j];
            if (similarity > config.linkThreshold) {
                state.links.push({
                    source: i.toString(),
                    target: j.toString(),
                    value: similarity,
                });
            }
        }
    }
}

// --- D3 NETWORK VISUALIZATION ---
function createNetworkGraph() {
    const container = document.getElementById('network-container');
    container.innerHTML = '';
    const width = container.clientWidth;
    const height = container.clientHeight;
    const svg = d3.select(container).append('svg')
        .attr('width', width)
        .attr('height', height);
    const g = svg.append('g');
    state.svg = svg;
    state.zoom = d3.zoom().scaleExtent([0.5, 4]).on("zoom", (event) => {
        g.attr("transform", event.transform);
        d3.select('#zoom-slider').property('value', event.transform.k);
    });
    svg.call(state.zoom);

    const maxImportance = d3.max(state.nodes, d => d.importance);
    const radiusScale = d3.scaleSqrt().domain([0, maxImportance]).range([4, 20]);
    const maxLinkValue = d3.max(state.links, d => d.value);
    const linkScale = d3.scaleLinear().domain([0, maxLinkValue]).range([0, 1]);

    state.simulation = d3.forceSimulation(state.nodes)
        .force('link', d3.forceLink(state.links).id(d => d.id)
            .distance(d => 150 / (linkScale(d.value) + 0.1))
            .strength(d => linkScale(d.value))
        )
        .force('charge', d3.forceManyBody().strength(config.chargeStrength))
        .force('center', d3.forceCenter(width / 2, height / 2).strength(config.centerStrength))
        .force('collide', d3.forceCollide().radius(d => radiusScale(d.importance) + 2));

    const link = g.append('g').selectAll('line')
        .data(state.links).enter().append('line')
        .attr('class', 'link')
        .style('stroke-opacity', d => 0.1 + linkScale(d.value) * 0.6)
        .attr('stroke-width', d => 1 + linkScale(d.value) * 3);

    const node = g.append('g').selectAll('circle')
        .data(state.nodes).enter().append('circle')
        .attr('class', 'node')
        .attr('r', d => radiusScale(d.importance))
        .attr('fill', d => d3.schemeSet2[d.id % 8])
        .call(drag(state.simulation));

    const labels = g.append('g').selectAll('text')
        .data(state.nodes).enter().append('text')
        .attr('class', 'node-label')
        .text(d => d.name)
        .style('fill', 'white').style('font-family', 'Courier New, monospace')
        .style('font-size', '10px').style('pointer-events', 'none')
        .attr('text-anchor', 'middle');

    // --- Interactivity ---
    node.on('mouseover', (event, d) => highlightNeighbors(g, d, true))
        .on('mouseout', (event, d) => highlightNeighbors(g, d, false))
        // **AJOUT : Réactivation du clic pour afficher les articles**
        .on('click', (event, d) => displayArticlesForTopic(d));
    
    state.simulation.on('tick', () => {
        link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        node.attr('cx', d => d.x).attr('cy', d => d.y);
        labels.attr('x', d => d.x).attr('y', d => d.y + radiusScale(d.importance) + 10);
    });
}

function setupZoomSlider() {
    d3.select('#zoom-slider').on('input', function() {
        const newScale = +this.value;
        state.zoom.scaleTo(state.svg.transition().duration(100), newScale);
    });
}

function highlightNeighbors(g, d, isHighlighted) {
    const lowOpacity = 0.1;
    const nodeSelection = g.selectAll('.node');
    const linkSelection = g.selectAll('.link');
    const labelSelection = g.selectAll('.node-label');
    
    if (!isHighlighted) {
        nodeSelection.style('opacity', 1);
        labelSelection.style('opacity', 1);
        linkSelection.style('stroke-opacity', l => 0.1 + d3.scaleLinear().domain([0, d3.max(state.links, li => li.value)]).range([0, 1])(l.value) * 0.6);
        return;
    }
    
    const linkedNodes = new Set([d.id]);
    state.links.forEach(l => {
        if (l.source.id === d.id) linkedNodes.add(l.target.id);
        if (l.target.id === d.id) linkedNodes.add(l.source.id);
    });
    
    nodeSelection.style('opacity', n => linkedNodes.has(n.id) ? 1 : lowOpacity);
    labelSelection.style('opacity', n => linkedNodes.has(n.id) ? 1 : lowOpacity);
    linkSelection.style('stroke-opacity', l => (l.source.id === d.id || l.target.id === d.id) ? 0.8 : lowOpacity);
}

function drag(simulation) {
    function dragstarted(event, d) { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }
    function dragged(event, d) { d.fx = event.x; d.fy = event.y; }
    function dragended(event, d) { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }
    return d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended);
}


// **AJOUT : Toutes les fonctions pour afficher les articles**
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
function displayArticlesForTopic(topicNode) {
    // Trouve tous les articles où ce topic est dominant
    const articlesForTopic = Object.entries(state.docTopics)
        .filter(([id, doc]) => doc.dominant_topic.toString() === topicNode.id)
        .map(([id, doc]) => id);
    
    const articleObjects = articlesForTopic
        .map(id => state.articles.find(a => a.id === id))
        .filter(Boolean); // Filtre les articles non trouvés

    updateArticleList(articleObjects, `Articles pour: ${topicNode.name}`);
}

function updateArticleList(articles, title) {
    const listEl = document.getElementById('articles-list');
    const titleEl = document.getElementById('articles-title');
    const countEl = document.getElementById('articles-count');

    titleEl.textContent = title;
    countEl.textContent = `${articles.length} articles`;

    if (articles.length === 0) {
        listEl.innerHTML = '<p class="no-articles">Aucun article trouvé pour cette sélection.</p>';
        return;
    }

    listEl.innerHTML = articles
        .sort((a, b) => (b.date || '').localeCompare(a.date || '')) // Tri par date décroissante
        .map(article => `
            <div class="article-item" data-article-id="${article.id}">
                <div class="article-title">${article.title || 'Titre inconnu'}</div>
                <div class="article-meta">
                    <span>${article.date || 'Date inconnue'}</span> | 
                    <span>${article.newspaper || 'Journal inconnu'}</span>
                </div>
            </div>
        `).join('');

    // Ajoute les gestionnaires de clic pour ouvrir la modale
    listEl.querySelectorAll('.article-item').forEach(item => {
        item.addEventListener('click', () => {
            const articleId = item.dataset.articleId;
            const article = state.articles.find(a => a.id === articleId);
            if (article) showArticleModal(article);
        });
    });
}

function initModal() {
    const modal = document.getElementById('article-modal');
    const closeBtn = modal.querySelector('.close');
    closeBtn.onclick = () => modal.style.display = "none";
    window.onclick = (event) => {
        if (event.target == modal) {
            modal.style.display = "none";
        }
    };
}

function showArticleModal(article) {
    const modal = document.getElementById('article-modal');
    document.getElementById('modal-title').textContent = article.title || 'Titre inconnu';
    document.getElementById('article-content').innerHTML = (article.content || 'Contenu non disponible').replace(/\n/g, '<br>');
    document.getElementById('article-date').textContent = article.date || 'Date inconnue';
    document.getElementById('article-newspaper').textContent = article.newspaper || 'Journal inconnu';
    modal.style.display = 'block';
}