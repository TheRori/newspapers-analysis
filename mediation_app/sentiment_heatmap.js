// Configuration
const config = {
    margin: { top: 50, right: 50, bottom: 50, left: 50 },
    mapWidth: 800,
    mapHeight: 500,
    // Using RdBu for compound sentiment (red for negative, blue for positive) - more contrast than RdYlBu
    colors: d3.interpolateRdBu,
    // Alternative color scales that can be used for different sentiment types
    colorScales: {
        compound: d3.interpolateRdBu,
        positive: d3.interpolateBlues,
        negative: d3.interpolateReds,
        neutral: d3.interpolateGreys
    },
    // Facteur d'amplification pour les valeurs de sentiment
    sentimentAmplificationFactor: 2.0,
    transitionDuration: 500,
    // Error handling
    errorDelay: 2000,
    // Fallback projection settings for Switzerland
    switzerlandCenter: [8.3, 46.8],
    switzerlandScale: 12000,
    // Names of cantons for display
    cantonNames: {
        'ZH': 'Zürich', 'BE': 'Bern/Berne', 'LU': 'Luzern', 'UR': 'Uri', 'SZ': 'Schwyz',
        'OW': 'Obwalden', 'NW': 'Nidwalden', 'GL': 'Glarus', 'ZG': 'Zug', 'FR': 'Fribourg',
        'SO': 'Solothurn', 'BS': 'Basel-Stadt', 'BL': 'Basel-Landschaft', 'SH': 'Schaffhausen',
        'AR': 'Appenzell Ausserrhoden', 'AI': 'Appenzell Innerrhoden', 'SG': 'St. Gallen',
        'GR': 'Graubünden/Grigioni', 'AG': 'Aargau', 'TG': 'Thurgau', 'TI': 'Ticino',
        'VD': 'Vaud', 'VS': 'Valais/Wallis', 'NE': 'Neuchâtel', 'GE': 'Genève', 'JU': 'Jura'
    }
};

// État de l'application
const state = {
    data: [],
    cantonData: {},
    selectedSentimentType: "compound",
    selectedYearRange: "all",
    cantonGeoJson: null,
    useGridFallback: false // Utiliser la visualisation en grille comme secours
};

// Chargement des données
document.addEventListener('DOMContentLoaded', () => {
    // Charger les données
    loadData()
        .then(() => {
            // Initialiser les contrôles
            initControls();
            
            // Créer la visualisation une fois les données chargées
            createVisualization();
        })
        .catch(error => {
            console.error("Erreur lors du chargement des données:", error);
            // Afficher un message d'erreur à l'utilisateur
            document.getElementById('map-container').innerHTML = 
                '<div class="error-message">Erreur lors du chargement des données. Veuillez réessayer.</div>';
        });
});

// Chargement des données
function loadData() {
    return new Promise((resolve, reject) => {
        // Chemin vers le fichier JSON des articles avec sentiments
        const jsonPath = 'data/source/collections/heatmap_sentiments/896c9873-6e56-47ce-8d6e-0fb766a77213/source_files/articles_with_sentiment_89606e08.json';
        console.log("Chemin vers le fichier JSON:", jsonPath);
        
        // Chemins possibles vers le fichier GeoJSON des cantons suisses
        const geoJsonPaths = [
            'data/source/collections/heatmap_sentiments/38215e35-dfec-43a3-b541-6a78e2b0d77d/source_files/cantons.geojson',
            'cantons.geojson', // Essayer dans le répertoire racine
            'data/cantons.geojson',
            'data/cantons-suisses.json',
            'cantons-suisses.json',
            'data/source/collections/heatmap_sentiments/38215e35-dfec-43a3-b541-6a78e2b0d77d/source_files/cantons-suisses.json',
            'data/source/cantons.geojson',
            'data/source/cantons-suisses.json'
        ];
        
        // Afficher un message de chargement
        const loadingMessage = document.createElement('div');
        loadingMessage.className = 'loading-message';
        loadingMessage.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Chargement des données...';
        document.getElementById('map-container').appendChild(loadingMessage);
        
        // Fonction pour essayer de charger le fichier GeoJSON à partir de différents chemins
        function tryLoadGeoJson(pathIndex) {
            if (pathIndex >= geoJsonPaths.length) {
                // Tous les chemins ont été essayés sans succès
                console.error("Impossible de charger les données GeoJSON à partir des chemins disponibles");
                
                // Supprimer le message de chargement
                if (loadingMessage.parentNode) {
                    loadingMessage.parentNode.removeChild(loadingMessage);
                }
                
                // Activer la visualisation en grille comme secours
                console.log("Activation de la visualisation en grille comme secours");
                state.useGridFallback = true;
                
                // Charger les données de sentiment et résoudre la promesse
                loadSentimentData().then(resolve).catch(reject);
                return;
            }
        
        const currentPath = geoJsonPaths[pathIndex];
        console.log(`Tentative de chargement du fichier GeoJSON depuis: ${currentPath}`);
        
        // Essayer de charger le fichier GeoJSON
        d3.json(currentPath)
            .then(geoJson => {
                console.log("Données GeoJSON chargées avec succès:", currentPath);
                state.cantonGeoJson = geoJson;
                
                // Charger les données de sentiment
                return loadSentimentData();
            })
            .then(resolve)
            .catch(error => {
                console.warn(`Erreur lors du chargement depuis ${currentPath}:`, error);
                // Essayer le chemin suivant
                tryLoadGeoJson(pathIndex + 1);
            });
    }
    
    // Fonction pour charger les données de sentiment
    function loadSentimentData() {
        return new Promise((resolve, reject) => {
            console.log("Chargement des données de sentiment depuis:", jsonPath);
            
            d3.json(jsonPath)
                .then(articles => {
                    console.log("Données d'articles chargées avec succès, nombre d'articles:", articles.length);
                    
                    // Supprimer le message de chargement
                    if (loadingMessage.parentNode) {
                        loadingMessage.parentNode.removeChild(loadingMessage);
                    }
                    
                    // Traiter les données d'articles
                    if (articles && articles.length > 0) {
                        // Calculer les moyennes de sentiment par canton
                        calculateCantonSentiments(articles);
                        resolve(articles);
                    } else {
                        // Générer des données de test si aucun article n'est disponible
                        console.warn("Aucun article trouvé, utilisation de données de test");
                        generateTestData();
                        resolve([]);
                    }
                })
                .catch(error => {
                    console.error("Erreur lors du chargement des données d'articles:", error);
                    
                    // Supprimer le message de chargement
                    if (loadingMessage.parentNode) {
                        loadingMessage.parentNode.removeChild(loadingMessage);
                    }
                    
                    // Générer des données de test en cas d'erreur
                    console.warn("Utilisation de données de test suite à une erreur");
                    generateTestData();
                    reject(error);
                });
        });
    }
    
    // Commencer à essayer les chemins
    tryLoadGeoJson(0);
    });
}

// Générer des données de test pour la démo
function generateTestData() {
    // Liste des cantons suisses (correspond aux IDs dans le TopoJSON)
    const cantons = Object.keys(config.cantonNames);
    
    // Générer des données aléatoires pour chaque canton
    const articles = [];
    
    // Générer 1000 articles aléatoires
    for (let i = 0; i < 1000; i++) {
        const canton = cantons[Math.floor(Math.random() * cantons.length)];
        const year = 1990 + Math.floor(Math.random() * 20); // Années entre 1990 et 2010
        
        // Générer des scores de sentiment aléatoires
        const compound = Math.random() * 2 - 1; // Entre -1 et 1
        const positive = Math.random(); // Entre 0 et 1
        const negative = Math.random(); // Entre 0 et 1
        const neutral = Math.random(); // Entre 0 et 1
        
        articles.push({
            id: `article-${i}`,
            canton: canton,
            year: year,
            title: `Test Article ${i}`,
            content: `This is random content for article ${i}.`,
            sentiment: {
                compound: compound,
                positive: positive,
                negative: negative,
                neutral: neutral
            }
        });
    }
    
    // Calculer les moyennes de sentiment par canton
    calculateCantonSentiments(articles);
    
    // Activer la visualisation en grille comme secours
    state.useGridFallback = true;
}

// Calcul des moyennes de sentiment par canton
function calculateCantonSentiments(articles) {
    // Initialiser les données par canton
    const cantonData = {};
    
    // Grouper les articles par canton
    const articlesByCanton = {};
    
    console.log("Traitement de", articles.length, "articles pour le calcul des sentiments");
    
    // Parcourir les articles
    articles.forEach(article => {
        // Vérifier si l'article a un canton et des données de sentiment
        if (!article.canton || !article.sentiment) {
            return; // Ignorer cet article
        }
        
        const canton = article.canton;
        
        if (!articlesByCanton[canton]) {
            articlesByCanton[canton] = [];
        }
        
        articlesByCanton[canton].push(article);
    });
    
    console.log("Articles groupés par canton:", Object.keys(articlesByCanton).map(canton => `${canton}: ${articlesByCanton[canton].length} articles`));
    
    // Calculer les moyennes pour chaque canton
    Object.keys(articlesByCanton).forEach(canton => {
        const cantonArticles = articlesByCanton[canton];
        const count = cantonArticles.length;
        
        // Initialiser les sommes
        let sumCompound = 0;
        let sumPositive = 0;
        let sumNegative = 0;
        let sumNeutral = 0;
        
        // Calculer les sommes
        cantonArticles.forEach(article => {
            if (article.sentiment) {
                sumCompound += article.sentiment.compound || 0;
                sumPositive += article.sentiment.positive || 0;
                sumNegative += article.sentiment.negative || 0;
                sumNeutral += article.sentiment.neutral || 0;
            }
        });
        
        // Calculer les moyennes
        const avgCompound = sumCompound / count;
        const avgPositive = sumPositive / count;
        const avgNegative = sumNegative / count;
        const avgNeutral = sumNeutral / count;
        
        // Stocker les résultats
        cantonData[canton] = {
            avgCompound,
            avgPositive,
            avgNegative,
            avgNeutral,
            count,
            articles: cantonArticles, // Garder tous les articles pour le modal
            yearRanges: {}
        };
        
        // Calculer les moyennes par plage d'années
        const yearRanges = {
            "1960-1965": { articles: [], count: 0, sumCompound: 0, sumPositive: 0, sumNegative: 0, sumNeutral: 0 },
            "1965-1970": { articles: [], count: 0, sumCompound: 0, sumPositive: 0, sumNegative: 0, sumNeutral: 0 },
            "1970-1975": { articles: [], count: 0, sumCompound: 0, sumPositive: 0, sumNegative: 0, sumNeutral: 0 },
            "1975-1980": { articles: [], count: 0, sumCompound: 0, sumPositive: 0, sumNegative: 0, sumNeutral: 0 },
            "1980-1985": { articles: [], count: 0, sumCompound: 0, sumPositive: 0, sumNegative: 0, sumNeutral: 0 },
            "1985-1990": { articles: [], count: 0, sumCompound: 0, sumPositive: 0, sumNegative: 0, sumNeutral: 0 },
            "1990-1995": { articles: [], count: 0, sumCompound: 0, sumPositive: 0, sumNegative: 0, sumNeutral: 0 },
            "1995-2000": { articles: [], count: 0, sumCompound: 0, sumPositive: 0, sumNegative: 0, sumNeutral: 0 }
        };
        
        // Grouper les articles par plage d'années
        cantonArticles.forEach(article => {
            let year;
            if (article.year) {
                year = article.year;
            } else if (article.date && typeof article.date === 'string') {
                year = parseInt(article.date.split('-')[0], 10);
            } else {
                return;
            }
            
            let range = "";
            if (year >= 1960 && year < 1965) range = "1960-1965";
            else if (year >= 1965 && year < 1970) range = "1965-1970";
            else if (year >= 1970 && year < 1975) range = "1970-1975";
            else if (year >= 1975 && year < 1980) range = "1975-1980";
            else if (year >= 1980 && year < 1985) range = "1980-1985";
            else if (year >= 1985 && year < 1990) range = "1985-1990";
            else if (year >= 1990 && year < 1995) range = "1990-1995";
            else if (year >= 1995 && year <= 2000) range = "1995-2000";
            
            if (range && yearRanges[range] && article.sentiment) {
                yearRanges[range].count++;
                yearRanges[range].sumCompound += article.sentiment.compound || 0;
                yearRanges[range].sumPositive += article.sentiment.positive || 0;
                yearRanges[range].sumNegative += article.sentiment.negative || 0;
                yearRanges[range].sumNeutral += article.sentiment.neutral || 0;
            }
        });
        
        // Calculer les moyennes pour chaque plage d'années
        Object.keys(yearRanges).forEach(range => {
            const rangeData = yearRanges[range];
            
            if (rangeData.count > 0) {
                rangeData.avgCompound = rangeData.sumCompound / rangeData.count;
                rangeData.avgPositive = rangeData.sumPositive / rangeData.count;
                rangeData.avgNegative = rangeData.sumNegative / rangeData.count;
                rangeData.avgNeutral = rangeData.sumNeutral / rangeData.count;
            } else {
                rangeData.avgCompound = 0;
                rangeData.avgPositive = 0;
                rangeData.avgNegative = 0;
                rangeData.avgNeutral = 0;
            }
            
            delete rangeData.sumCompound;
            delete rangeData.sumPositive;
            delete rangeData.sumNegative;
            delete rangeData.sumNeutral;
            
            // Correction : toujours utiliser cantonData[canton].yearRanges
            if (cantonData[canton] && cantonData[canton].yearRanges) {
                cantonData[canton].yearRanges[range] = rangeData;
            }
        });
    });
    
    console.log("Données de sentiment calculées pour", Object.keys(cantonData).length, "cantons");
    
    state.cantonData = cantonData;
}

// Initialisation des contrôles
function initControls() {
    try {
        document.querySelectorAll('input[name="sentiment-type"]').forEach(radio => {
            radio.addEventListener('change', function() {
                state.selectedSentimentType = this.value;
                config.colors = config.colorScales[state.selectedSentimentType] || config.colorScales.compound;
                updateVisualization();
            });
        });
        
        document.getElementById('year-range').addEventListener('change', function() {
            state.selectedYearRange = this.value;
            updateVisualization();
        });
    } catch (error) {
        console.error("Erreur lors de l'initialisation des contrôles:", error);
        document.querySelector('.controls').innerHTML += 
            '<div class="error-message">Erreur lors de l\'initialisation des contrôles. Veuillez recharger la page.</div>';
    }
}

// Création de la visualisation
function createVisualization() {
    const containerWidth = document.getElementById('map-container').clientWidth;
    const containerHeight = document.getElementById('map-container').clientHeight;
    const width = Math.min(containerWidth, config.mapWidth);
    const height = Math.min(containerHeight, config.mapHeight);
    
    const svg = d3.select('#map-container')
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', [0, 0, width, height])
        .attr('style', 'max-width: 100%; height: auto;');
    
    const mapGroup = svg.append('g')
        .attr('transform', `translate(${config.margin.left}, ${config.margin.top})`);
    
    const cantonGroup = mapGroup.append('g')
        .attr('class', 'cantons-group');
        
    const tooltip = d3.select('#tooltip');
    
    if (state.useGridFallback) {
        console.log("Utilisation de la visualisation en grille");
        createGridVisualization(cantonGroup, tooltip, width, height);
        updateVisualization();
        return;
    }
    
    let features = [];
    if (state.cantonGeoJson) {
        try {
            if (state.cantonGeoJson.type === 'Topology' && state.cantonGeoJson.objects && state.cantonGeoJson.objects.cantons) {
                features = topojson.feature(state.cantonGeoJson, state.cantonGeoJson.objects.cantons).features;
            } else if (state.cantonGeoJson.type === 'FeatureCollection' && state.cantonGeoJson.features) {
                features = state.cantonGeoJson.features;
            }
        } catch (error) {
            console.error("Erreur lors de l'extraction des features:", error);
        }
    }
    
    if (!features || features.length === 0) {
        console.error("Aucune feature GeoJSON/TopoJSON valide trouvée. Passage en mode grille.");
        state.useGridFallback = true;
        createGridVisualization(cantonGroup, tooltip, width, height);
        updateVisualization();
        return;
    }
    
    const projection = d3.geoMercator().fitSize(
        [width - config.margin.left - config.margin.right, height - config.margin.top - config.margin.bottom],
        { type: "FeatureCollection", features: features }
    );
    const path = d3.geoPath().projection(projection);

    cantonGroup.selectAll('.canton')
        .data(features)
        .enter()
        .append('path')
        .attr('class', 'canton')
        .attr('d', path)
        .attr('id', feature => `canton-${feature.id || feature.properties.id}`)
        .on('mouseover', function(event, feature) {
            d3.select(this).raise().attr('stroke-width', 2).attr('stroke', '#fff');
            const cantonId = feature.id || feature.properties.id;
            const cantonName = (config.cantonNames && config.cantonNames[cantonId]) || cantonId;
            const sentimentData = getSentimentDataForDisplay(cantonId);
            let tooltipContent = `<strong>Canton: ${cantonName}</strong><br>`;
            if (!isNaN(sentimentData.value)) {
                tooltipContent += `Sentiment ${state.selectedSentimentType}: ${sentimentData.value.toFixed(3)}<br>`;
                tooltipContent += `Nombre d'articles: ${sentimentData.count}`;
            } else {
                tooltipContent += "Aucune donnée disponible";
            }
            tooltip.html(tooltipContent)
                .style('left', (event.pageX + 15) + 'px')
                .style('top', (event.pageY - 28) + 'px')
                .style('opacity', 1);
        })
        .on('mouseout', function() {
            d3.select(this).attr('stroke-width', 0.5).attr('stroke', '#333');
            tooltip.style('opacity', 0);
        })
        .on('click', (event, feature) => {
            const cantonId = feature.id || (feature.properties ? feature.properties.id : null);
            if (cantonId) {
                showCantonModal(cantonId);
            }
        });
    
    updateVisualization();
}

// Création d'une visualisation en grille pour les cantons
function createGridVisualization(cantonGroup, tooltip, width, height) {
    const cantons = Object.keys(config.cantonNames);
    const rectSize = 60;
    const padding = 10;
    const cols = 6;
    
    const cantonRects = cantonGroup.selectAll('.canton-group')
        .data(cantons)
        .enter()
        .append('g')
        .attr('class', 'canton-group');
    
    cantonRects.append('rect')
        .attr('class', 'canton')
        .attr('id', d => `canton-${d}`)
        .attr('x', (d, i) => (i % cols) * (rectSize + padding) + 50)
        .attr('y', (d, i) => Math.floor(i / cols) * (rectSize + padding + 20) + 50)
        .attr('width', rectSize)
        .attr('height', rectSize)
        .attr('rx', 5).attr('ry', 5)
        .attr('stroke', '#333').attr('stroke-width', 0.5);
    
    cantonRects.append('text')
        .attr('x', (d, i) => (i % cols) * (rectSize + padding) + 50 + rectSize / 2)
        .attr('y', (d, i) => Math.floor(i / cols) * (rectSize + padding + 20) + 50 + rectSize / 2)
        .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
        .style('font-size', '14px').style('font-weight', 'bold').style('fill', '#fff')
        .text(d => d);
    
    cantonRects.on('mouseover', function(event, d) {
            d3.select(this).select('rect').raise().attr('stroke-width', 2).attr('stroke', '#fff');
            const cantonName = config.cantonNames[d] || d;
            const sentimentData = getSentimentDataForDisplay(d);
            let tooltipContent = `<strong>Canton: ${cantonName}</strong><br>`;
            if (!isNaN(sentimentData.value)) {
                tooltipContent += `Sentiment ${state.selectedSentimentType}: ${sentimentData.value.toFixed(3)}<br>`;
                tooltipContent += `Nombre d'articles: ${sentimentData.count}`;
            } else {
                tooltipContent += "Aucune donnée disponible";
            }
            tooltip.html(tooltipContent)
                .style('left', (event.pageX + 15) + 'px')
                .style('top', (event.pageY - 28) + 'px')
                .style('opacity', 1);
        })
        .on('mouseout', function() {
            d3.select(this).select('rect').attr('stroke-width', 0.5).attr('stroke', '#333');
            tooltip.style('opacity', 0);
        })
        .on('click', (event, d) => {
            showCantonModal(d);
        });
}

// ** NOUVELLE FONCTION POUR AFFICHER LE MODAL **
function showCantonModal(cantonId) {
    const cantonData = state.cantonData[cantonId];
    if (!cantonData) {
        console.warn("Pas de données pour le canton:", cantonId);
        return;
    }

    const cantonName = config.cantonNames[cantonId] || cantonId;
    const yearRange = state.selectedYearRange;
    const sentimentType = state.selectedSentimentType;

    // 1. Obtenir les articles pour la période sélectionnée
    let articlesForPeriod = [];
    if (yearRange === 'all') {
        articlesForPeriod = cantonData.articles || [];
    } else {
        articlesForPeriod = (cantonData.articles || []).filter(article => {
            let year = article.year || (article.date ? parseInt(article.date.split('-')[0], 10) : undefined);
            if (!year) return false;
            const [start, end] = yearRange.split('-').map(Number);
            return year >= start && year <= end;
        });
    }

    // 2. Obtenir les statistiques
    const nbArticles = articlesForPeriod.length;
    const sentimentData = getSentimentDataForDisplay(cantonId);
    const avgScore = sentimentData.value;
    
    // 3. Trier pour les articles les plus négatifs et positifs
    const mostNegative = [...articlesForPeriod]
        .filter(a => a.sentiment && typeof a.sentiment.negative !== 'undefined')
        .sort((a, b) => b.sentiment.negative - a.sentiment.negative)
        .slice(0, 5);

    const mostPositive = [...articlesForPeriod]
        .filter(a => a.sentiment && typeof a.sentiment.positive !== 'undefined')
        .sort((a, b) => b.sentiment.positive - a.sentiment.positive)
        .slice(0, 5);

    // 4. Générer le HTML du modal
    let html = `<div class='modal-section'><strong>${cantonName}</strong> — <span>Période: ${yearRange === 'all' ? 'Toutes les années' : yearRange}</span></div>`;
    html += `<div class='modal-section'>Nombre d'articles: <strong>${nbArticles}</strong></div>`;
    html += `<div class='modal-section'>Score moyen (${sentimentType}): <strong>${!isNaN(avgScore) ? avgScore.toFixed(3) : 'N/A'}</strong></div>`;
    html += `<div class='modal-section modal-articles'>`;
    
    html += `<div class='modal-article-list'><h4>Top 5 articles négatifs</h4>`;
    if (mostNegative.length === 0) {
        html += `<div>Aucun article disponible</div>`;
    } else {
        html += `<ul class="article-list">`;
        mostNegative.forEach(article => {
            let score = article.sentiment.negative;
            let title = article.title || article.titre || 'Sans titre';
            let url = article.url || '';
            let citation = (article.content || article.text || '').slice(0, 60) + '...';
            
            html += `<li class="article-item negative">
                <div class="article-score">${score.toFixed(3)}</div>
                <div class="article-content">
                    ${url ? `<a href="${url}" target="_blank" class="article-title">${title}</a>` : `<span class="article-title">${title}</span>`}
                    <div class="article-citation">${citation}</div>
                </div>
            </li>`;
        });
        html += `</ul>`;
    }
    html += `</div>`;
    
    html += `<div class='modal-article-list'><h4>Top 5 articles positifs</h4>`;
    if (mostPositive.length === 0) {
        html += `<div>Aucun article disponible</div>`;
    } else {
        html += `<ul class="article-list">`;
        mostPositive.forEach(article => {
            let score = article.sentiment.positive;
            let title = article.title || article.titre || 'Sans titre';
            let url = article.url || '';
            let citation = (article.content || article.text || '').slice(0, 60) + '...';
            
            html += `<li class="article-item positive">
                <div class="article-score">${score.toFixed(3)}</div>
                <div class="article-content">
                    ${url ? `<a href="${url}" target="_blank" class="article-title">${title}</a>` : `<span class="article-title">${title}</span>`}
                    <div class="article-citation">${citation}</div>
                </div>
            </li>`;
        });
        html += `</ul>`;
    }
    html += `</div></div>`;

    // 5. Afficher le modal
    const modal = document.getElementById('sentiment-modal');
    const modalBody = document.getElementById('sentiment-modal-body');
    const modalTitle = document.getElementById('sentiment-modal-title');
    modalBody.innerHTML = html;
    modalTitle.textContent = `Détails du canton : ${cantonName}`;
    modal.style.display = 'block';

    document.getElementById('sentiment-modal-close').onclick = function() {
        modal.style.display = 'none';
    };
    window.onclick = function(event) {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    };
}


// Mise à jour de la visualisation
function updateVisualization() {
    try {
        const sentimentValues = Object.keys(state.cantonData).map(canton => {
            return getSentimentDataForDisplay(canton).value;
        }).filter(value => !isNaN(value));
        
        if (sentimentValues.length === 0) {
            console.warn("Aucune donnée de sentiment valide trouvée pour la visualisation");
            return;
        }
        
        let min = Math.min(...sentimentValues);
        let max = Math.max(...sentimentValues);
        
        if (state.selectedSentimentType === 'compound') {
            min = Math.min(min, -0.5) * config.sentimentAmplificationFactor;
            max = Math.max(max, 0.5) * config.sentimentAmplificationFactor;
            min = Math.max(min, -1);
            max = Math.min(max, 1);
        } else {
            min = 0;
            max = Math.max(max, 1) * config.sentimentAmplificationFactor;
            max = Math.min(max, 1);
        }
        
        const colorInterpolator = config.colorScales[state.selectedSentimentType] || config.colors;
        const colorScale = d3.scaleSequential().domain([min, max]).interpolator(colorInterpolator);
        
        d3.selectAll('.canton')
            .transition()
            .duration(config.transitionDuration)
            .attr('fill', function(d) {
                const cantonId = state.useGridFallback ? d : (d.id || d.properties.id);
                const sentimentData = getSentimentDataForDisplay(cantonId);
                if (state.cantonData[cantonId] && !isNaN(sentimentData.value)) {
                    return colorScale(amplifySentimentValue(sentimentData.value, state.selectedSentimentType));
                }
                return '#ccc'; // Couleur par défaut
            });
        
        updateLegend(min, max, colorInterpolator);
    } catch (error) {
        console.error("Erreur lors de la mise à jour de la visualisation:", error);
    }
}

// Amplifier une valeur de sentiment pour un meilleur contraste visuel
function amplifySentimentValue(value, sentimentType) {
    const amplificationFactor = 3.0;
    let amplifiedValue = value * amplificationFactor;
    if (sentimentType === 'compound') {
        return Math.max(-1, Math.min(1, amplifiedValue));
    } else {
        return Math.max(0, Math.min(1, amplifiedValue));
    }
}

// Récupérer les données de sentiment pour l'affichage
function getSentimentDataForDisplay(cantonId) {
    const cantonData = state.cantonData[cantonId];
    if (!cantonData) return { value: NaN, count: 0 };
    
    if (state.selectedYearRange === 'all') {
        const key = `avg${state.selectedSentimentType.charAt(0).toUpperCase() + state.selectedSentimentType.slice(1)}`;
        return { value: cantonData[key] || 0, count: cantonData.count };
    } else {
        const yearRangeData = cantonData.yearRanges[state.selectedYearRange];
        if (!yearRangeData || yearRangeData.count === 0) return { value: NaN, count: 0 };
        const key = `avg${state.selectedSentimentType.charAt(0).toUpperCase() + state.selectedSentimentType.slice(1)}`;
        return { value: yearRangeData[key] || 0, count: yearRangeData.count };
    }
}

// Mise à jour de la légende
function updateLegend(min, max, colorInterpolator) {
    if (isNaN(min) || isNaN(max)) return;
    
    const interpolator = colorInterpolator || config.colors;
    const legendMin = document.querySelector('.legend-min');
    const legendMid = document.querySelector('.legend-mid');
    const legendMax = document.querySelector('.legend-max');
    
    if (state.selectedSentimentType === 'compound') {
        legendMin.textContent = 'Négatif';
        legendMid.textContent = 'Neutre';
        legendMax.textContent = 'Positif';
    } else if (state.selectedSentimentType === 'positive') {
        legendMin.textContent = 'Peu positif';
        legendMid.textContent = 'Positif';
        legendMax.textContent = 'Très positif';
    } else if (state.selectedSentimentType === 'negative') {
        legendMin.textContent = 'Peu négatif';
        legendMid.textContent = 'Négatif';
        legendMax.textContent = 'Très négatif';
    } else if (state.selectedSentimentType === 'neutral') {
        legendMin.textContent = 'Peu neutre';
        legendMid.textContent = 'Neutre';
        legendMax.textContent = 'Très neutre';
    }
    
    const colorLegend = document.getElementById('color-legend');
    if (!colorLegend) return;
    
    const gradientColors = [];
    const steps = 10;
    for (let i = 0; i < steps; i++) {
        const value = min + (max - min) * (i / (steps - 1));
        const normalizedValue = (max - min === 0) ? 0.5 : (value - min) / (max - min);
        gradientColors.push(interpolator(normalizedValue));
    }
    colorLegend.style.background = `linear-gradient(to right, ${gradientColors.join(', ')})`;
}