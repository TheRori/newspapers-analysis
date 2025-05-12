// Configuration
const config = {
    margin: { top: 50, right: 50, bottom: 50, left: 50 },
    mapWidth: 800,
    mapHeight: 500,
    // Using RdYlBu for compound sentiment (red for negative, blue for positive)
    colors: d3.interpolateRdYlBu,
    // Alternative color scales that can be used for different sentiment types
    colorScales: {
        compound: d3.interpolateRdYlBu,
        positive: d3.interpolateBlues,
        negative: d3.interpolateReds,
        neutral: d3.interpolateGreys
    },
    transitionDuration: 500,
    // Error handling
    errorDelay: 2000,
    // Fallback projection settings for Switzerland
    switzerlandCenter: [8.3, 46.8],
    switzerlandScale: 12000
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
        const jsonPath = 'data/source/collections/heatmap_sentiments/c1cf64cd-dc23-4081-979a-46c4f0ed659c/source_files/articles_with_sentiment_d0f5e607.json';
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
    const cantons = [
        'ZH', 'BE', 'LU', 'UR', 'SZ', 'OW', 'NW', 'GL', 'ZG', 'FR',
        'SO', 'BS', 'BL', 'SH', 'AR', 'AI', 'SG', 'GR', 'AG', 'TG',
        'TI', 'VD', 'VS', 'NE', 'GE', 'JU'
    ];
    
    // Noms des cantons pour l'affichage
    const cantonNames = {
        'ZH': 'Zürich',
        'BE': 'Bern/Berne',
        'LU': 'Luzern',
        'UR': 'Uri',
        'SZ': 'Schwyz',
        'OW': 'Obwalden',
        'NW': 'Nidwalden',
        'GL': 'Glarus',
        'ZG': 'Zug',
        'FR': 'Fribourg',
        'SO': 'Solothurn',
        'BS': 'Basel-Stadt',
        'BL': 'Basel-Landschaft',
        'SH': 'Schaffhausen',
        'AR': 'Appenzell Ausserrhoden',
        'AI': 'Appenzell Innerrhoden',
        'SG': 'St. Gallen',
        'GR': 'Graubünden/Grigioni',
        'AG': 'Aargau',
        'TG': 'Thurgau',
        'TI': 'Ticino',
        'VD': 'Vaud',
        'VS': 'Valais/Wallis',
        'NE': 'Neuchâtel',
        'GE': 'Genève',
        'JU': 'Jura'
    };
    
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
            articles: cantonArticles,
            yearRanges: {}
        };
        
        // Calculer les moyennes par plage d'années
        const yearRanges = {
            "1990-1995": { articles: [], count: 0, sumCompound: 0, sumPositive: 0, sumNegative: 0, sumNeutral: 0 },
            "1995-2000": { articles: [], count: 0, sumCompound: 0, sumPositive: 0, sumNegative: 0, sumNeutral: 0 },
            "2000-2005": { articles: [], count: 0, sumCompound: 0, sumPositive: 0, sumNegative: 0, sumNeutral: 0 },
            "2005-2010": { articles: [], count: 0, sumCompound: 0, sumPositive: 0, sumNegative: 0, sumNeutral: 0 },
            "2010-2015": { articles: [], count: 0, sumCompound: 0, sumPositive: 0, sumNegative: 0, sumNeutral: 0 },
            "2015-2020": { articles: [], count: 0, sumCompound: 0, sumPositive: 0, sumNegative: 0, sumNeutral: 0 },
            "2020-2025": { articles: [], count: 0, sumCompound: 0, sumPositive: 0, sumNegative: 0, sumNeutral: 0 }
        };
        
        // Grouper les articles par plage d'années
        cantonArticles.forEach(article => {
            // Extraire l'année de la date (format: "YYYY-MM-DD")
            let year;
            if (article.year) {
                year = article.year; // Si l'année est déjà extraite
            } else if (article.date && typeof article.date === 'string') {
                // Extraire l'année de la date au format "YYYY-MM-DD"
                year = parseInt(article.date.split('-')[0], 10);
            } else {
                return; // Ignorer cet article si pas de date valide
            }
            
            let range = "";
            
            if (year >= 1990 && year < 1995) {
                range = "1990-1995";
            } else if (year >= 1995 && year < 2000) {
                range = "1995-2000";
            } else if (year >= 2000 && year < 2005) {
                range = "2000-2005";
            } else if (year >= 2005 && year <= 2010) {
                range = "2005-2010";
            } else if (year >= 2010 && year < 2015) {
                range = "2010-2015";
            } else if (year >= 2015 && year < 2020) {
                range = "2015-2020";
            } else if (year >= 2020 && year <= 2025) {
                range = "2020-2025";
            }
            
            if (range && yearRanges[range] && article.sentiment) {
                yearRanges[range].articles.push(article);
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
                // Valeurs par défaut si aucun article
                rangeData.avgCompound = 0;
                rangeData.avgPositive = 0;
                rangeData.avgNegative = 0;
                rangeData.avgNeutral = 0;
            }
            
            // Supprimer les données temporaires
            delete rangeData.sumCompound;
            delete rangeData.sumPositive;
            delete rangeData.sumNegative;
            delete rangeData.sumNeutral;
            delete rangeData.articles;
            
            // Stocker les résultats
            cantonData[canton].yearRanges[range] = rangeData;
        });
    });
    
    console.log("Données de sentiment calculées pour", Object.keys(cantonData).length, "cantons");
    
    // Mettre à jour l'état
    state.cantonData = cantonData;
}

// Initialisation des contrôles
function initControls() {
    try {
        // Gérer le changement de type de sentiment
        document.querySelectorAll('input[name="sentiment-type"]').forEach(radio => {
            radio.addEventListener('change', function() {
                state.selectedSentimentType = this.value;
                
                // Mettre à jour les couleurs en fonction du type de sentiment sélectionné
                config.colors = config.colorScales[state.selectedSentimentType] || config.colorScales.compound;
                
                // Mettre à jour la visualisation
                updateVisualization();
            });
        });
        
        // Gérer le changement de plage d'années
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
    // Dimensions du conteneur
    const containerWidth = document.getElementById('map-container').clientWidth;
    const containerHeight = document.getElementById('map-container').clientHeight;
    
    // Dimensions de la carte
    const width = Math.min(containerWidth, config.mapWidth);
    const height = Math.min(containerHeight, config.mapHeight);
    
    // Créer le SVG
    const svg = d3.select('#map-container')
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', [0, 0, width, height])
        .attr('style', 'max-width: 100%; height: auto;');
    
    // Créer un groupe pour la carte
    const mapGroup = svg.append('g')
        .attr('transform', `translate(${config.margin.left}, ${config.margin.top})`);
    
    // Créer un groupe pour les cantons
    const cantonGroup = mapGroup.append('g')
        .attr('class', 'cantons-group');
        
    // Créer un titre pour la visualisation
    svg.append('text')
        .attr('class', 'map-title')
        .attr('x', width / 2)
        .attr('y', 25)
        .attr('text-anchor', 'middle')
        .style('font-size', '18px')
        .style('font-weight', 'bold')
        .text('Analyse de sentiment par canton');
    
    // Créer un tooltip
    const tooltip = d3.select('#tooltip');
    
    // Dessiner les cantons
    let features = [];
    
    // Vérifier si on doit utiliser la visualisation en grille
    if (state.useGridFallback) {
        console.log("Utilisation de la visualisation en grille");
        createGridVisualization(cantonGroup, tooltip, width, height);
        
        // Mettre à jour les couleurs
        updateVisualization();
        return;
    }
    
    // Vérifier le format des données GeoJSON/TopoJSON et extraire les features
    if (state.cantonGeoJson) {
        console.log("Format des données GeoJSON:", state.cantonGeoJson);
        
        try {
            // Format TopoJSON (Topology)
            if (state.cantonGeoJson.type === 'Topology' && state.cantonGeoJson.objects && state.cantonGeoJson.objects.cantons) {
                console.log("Détecté comme TopoJSON");
                
                // Extraire les features du TopoJSON
                const featureCollection = topojson.feature(state.cantonGeoJson, state.cantonGeoJson.objects.cantons);
                console.log("Features extraites du TopoJSON:", featureCollection);
                
                if (featureCollection && featureCollection.features) {
                    features = featureCollection.features;
                    console.log("Nombre de features extraites:", features.length);
                }
            } 
            // Format GeoJSON standard
            else if (state.cantonGeoJson.type === 'FeatureCollection' && state.cantonGeoJson.features) {
                console.log("Détecté comme GeoJSON FeatureCollection");
                features = state.cantonGeoJson.features;
            }
            // Format GeoJSON avec un seul feature
            else if (state.cantonGeoJson.type === 'Feature') {
                console.log("Détecté comme GeoJSON Feature");
                features = [state.cantonGeoJson];
            }
        } catch (error) {
            console.error("Erreur lors de l'extraction des features:", error);
            state.useGridFallback = true;
            createGridVisualization(cantonGroup, tooltip, width, height);
            
            // Mettre à jour les couleurs
            updateVisualization();
            return;
        }
    }
    
    // Si aucune feature valide n'a été trouvée, utiliser la visualisation en grille
    if (!features || features.length === 0) {
        console.error("Format de données GeoJSON incorrect ou aucun canton trouvé:", state.cantonGeoJson);
        state.useGridFallback = true;
        createGridVisualization(cantonGroup, tooltip, width, height);
        
        // Mettre à jour les couleurs
        updateVisualization();
        return;
    }
    
    // Projection géographique pour la Suisse
    let projection;
    
    try {
        // Adapter automatiquement la projection aux données
        // Créer un objet FeatureCollection pour la projection
        const featureCollection = {type: "FeatureCollection", features: features};
        console.log("FeatureCollection créée pour la projection avec", features.length, "features");
        
        // Vérifier si les features ont des coordonnées valides
        const hasValidGeometry = features.some(f => 
            f.geometry && 
            ((f.geometry.type === 'Polygon' && f.geometry.coordinates && f.geometry.coordinates.length > 0) ||
            (f.geometry.type === 'MultiPolygon' && f.geometry.coordinates && f.geometry.coordinates.length > 0))
        );
        
        if (hasValidGeometry) {
            projection = d3.geoMercator()
                .fitSize(
                    [width - config.margin.left - config.margin.right, 
                    height - config.margin.top - config.margin.bottom], 
                    featureCollection
                );
        } else {
            // Fallback si les géométries ne sont pas valides
            console.warn("Géométries non valides, utilisation d'une projection fixe");
            projection = d3.geoMercator()
                .center(config.switzerlandCenter)
                .scale(config.switzerlandScale)
                .translate([width / 2, height / 2]);
        }
    } catch (error) {
        console.error("Erreur lors de la création de la projection:", error);
        // Fallback sur une projection fixe pour la Suisse
        projection = d3.geoMercator()
            .center(config.switzerlandCenter)
            .scale(config.switzerlandScale)
            .translate([width / 2, height / 2]);
    }
    
    // Générateur de chemin
    const path = d3.geoPath().projection(projection);
    
    // Vérifier si on doit utiliser la visualisation en grille
    if (state.useGridFallback) {
        console.log("Utilisation de la visualisation en grille");
        createGridVisualization(cantonGroup, tooltip, width, height);
    } else {
        // Vérifier le format des données GeoJSON/TopoJSON et extraire les features
        if (state.cantonGeoJson) {
            console.log("Format des données GeoJSON:", state.cantonGeoJson);
            
            try {
                // Format TopoJSON (Topology)
                if (state.cantonGeoJson.type === 'Topology' && state.cantonGeoJson.objects && state.cantonGeoJson.objects.cantons) {
                    console.log("Détecté comme TopoJSON");
                    
                    // Extraire les features du TopoJSON
                    const featureCollection = topojson.feature(state.cantonGeoJson, state.cantonGeoJson.objects.cantons);
                    console.log("Features extraites du TopoJSON:", featureCollection);
                    
                    if (featureCollection && featureCollection.features) {
                        features = featureCollection.features;
                        console.log("Nombre de features extraites:", features.length);
                    }
                } 
                // Format GeoJSON standard
                else if (state.cantonGeoJson.type === 'FeatureCollection' && state.cantonGeoJson.features) {
                    console.log("Détecté comme GeoJSON FeatureCollection");
                    features = state.cantonGeoJson.features;
                }
                // Format GeoJSON avec un seul feature
                else if (state.cantonGeoJson.type === 'Feature') {
                    console.log("Détecté comme GeoJSON Feature");
                    features = [state.cantonGeoJson];
                }
            } catch (error) {
                console.error("Erreur lors de l'extraction des features:", error);
                state.useGridFallback = true;
                createGridVisualization(cantonGroup, tooltip, width, height);
                return;
            }
        }
        
        // Vérifier si on a des features valides
        if (features && features.length > 0) {
            cantonGroup.selectAll('.canton')
                .data(features)
                .enter()
                .append('path')
                .attr('class', 'canton')
                .attr('d', path)
                .attr('id', feature => {
                // Dans le TopoJSON, l'ID est directement dans la propriété 'id'
                const cantonId = feature.id || (feature.properties ? feature.properties.id : null);
                return `canton-${cantonId}`;
            })
                .on('mouseover', function(event, feature) {
                    // Mettre en évidence le canton
                    d3.select(this)
                        .attr('stroke-width', 2)
                        .attr('stroke', '#fff');
                    
                    // Afficher le tooltip
                    const cantonId = feature.id || (feature.properties ? feature.properties.id : null);
                    const cantonName = feature.properties ? feature.properties.name : cantonId;
                    const cantonData = cantonId ? state.cantonData[cantonId] : null;
                    
                    let tooltipContent = `<strong>Canton: ${cantonName}</strong><br>`;
                    
                    if (cantonData) {
                        const sentimentData = getSentimentDataForDisplay(cantonId);
                        const sentimentValue = sentimentData.value.toFixed(3);
                        const articleCount = sentimentData.count;
                        
                        tooltipContent += `Sentiment ${state.selectedSentimentType}: ${sentimentValue}<br>`;
                        tooltipContent += `Nombre d'articles: ${articleCount}`;
                    } else {
                        tooltipContent += "Aucune donnée disponible";
                    }
                    
                    tooltip.html(tooltipContent)
                        .style('left', (event.pageX + 10) + 'px')
                        .style('top', (event.pageY - 28) + 'px')
                        .style('opacity', 1);
                })
                .on('mouseout', function() {
                    // Réinitialiser le style du canton
                    d3.select(this)
                        .attr('stroke-width', 0.5)
                        .attr('stroke', '#333');
                    
                    // Masquer le tooltip
                    tooltip.style('opacity', 0);
                })
                .on('click', function(event, feature) {
                    // Afficher les détails du canton
                    const cantonId = feature.id || (feature.properties ? feature.properties.id : null);
                    const cantonName = feature.properties ? feature.properties.name : cantonId;
                    const cantonData = cantonId ? state.cantonData[cantonId] : null;
                    
                    if (cantonData) {
                        alert(`Canton: ${cantonName}\nNombre d'articles: ${cantonData.count}`);
                        // Ici, on pourrait ouvrir un modal avec plus de détails
                    }
                });
    } else {
        // Afficher un message d'erreur si les données GeoJSON ne sont pas correctes
        console.error("Format de données GeoJSON incorrect ou aucun canton trouvé:", state.cantonGeoJson);
        state.useGridFallback = true;
        createGridVisualization(cantonGroup, tooltip, width, height);
    }
    
    // Mettre à jour les couleurs
    updateVisualization();
}

// Création d'une visualisation en grille pour les cantons
function createGridVisualization(cantonGroup, tooltip, width, height) {
    // Liste des cantons suisses
    const cantons = [
        'ZH', 'BE', 'LU', 'UR', 'SZ', 'OW', 'NW', 'GL', 'ZG', 'FR',
        'SO', 'BS', 'BL', 'SH', 'AR', 'AI', 'SG', 'GR', 'AG', 'TG',
        'TI', 'VD', 'VS', 'NE', 'GE', 'JU'
    ];
    
    // Noms des cantons pour l'affichage
    const cantonNames = {
        'ZH': 'Zürich',
        'BE': 'Bern/Berne',
        'LU': 'Luzern',
        'UR': 'Uri',
        'SZ': 'Schwyz',
        'OW': 'Obwalden',
        'NW': 'Nidwalden',
        'GL': 'Glarus',
        'ZG': 'Zug',
        'FR': 'Fribourg',
        'SO': 'Solothurn',
        'BS': 'Basel-Stadt',
        'BL': 'Basel-Landschaft',
        'SH': 'Schaffhausen',
        'AR': 'Appenzell Ausserrhoden',
        'AI': 'Appenzell Innerrhoden',
        'SG': 'St. Gallen',
        'GR': 'Graubünden/Grigioni',
        'AG': 'Aargau',
        'TG': 'Thurgau',
        'TI': 'Ticino',
        'VD': 'Vaud',
        'VS': 'Valais/Wallis',
        'NE': 'Neuchâtel',
        'GE': 'Genève',
        'JU': 'Jura'
    };
    
    // Créer une grille de rectangles pour représenter les cantons
    const rectSize = 60;
    const padding = 10;
    const cols = 6;
    const rows = Math.ceil(cantons.length / cols);
    
    // Ajouter un texte explicatif
    cantonGroup.append('text')
        .attr('x', width / 2 - 150)
        .attr('y', 30)
        .attr('class', 'grid-info')
        .style('font-size', '14px')
        .style('fill', '#fff')
        .text('Visualisation simplifiée des cantons suisses');
    
    // Créer les rectangles pour chaque canton
    const cantonRects = cantonGroup.selectAll('.canton')
        .data(cantons)
        .enter()
        .append('g')
        .attr('class', 'canton-group');
    
    // Ajouter les rectangles
    cantonRects.append('rect')
        .attr('class', 'canton')
        .attr('id', d => `canton-${d}`)
        .attr('x', (d, i) => (i % cols) * (rectSize + padding) + 50)
        .attr('y', (d, i) => Math.floor(i / cols) * (rectSize + padding + 20) + 50)
        .attr('width', rectSize)
        .attr('height', rectSize)
        .attr('rx', 5)
        .attr('ry', 5)
        .attr('stroke', '#333')
        .attr('stroke-width', 0.5);
    
    // Ajouter les labels des cantons
    cantonRects.append('text')
        .attr('x', (d, i) => (i % cols) * (rectSize + padding) + 50 + rectSize / 2)
        .attr('y', (d, i) => Math.floor(i / cols) * (rectSize + padding + 20) + 50 + rectSize / 2)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .style('font-size', '14px')
        .style('font-weight', 'bold')
        .style('fill', '#fff')
        .text(d => d);
    
    // Ajouter les interactions
    cantonRects.on('mouseover', function(event, d) {
        // Mettre en évidence le canton
        d3.select(this).select('rect')
            .attr('stroke-width', 2)
            .attr('stroke', '#fff');
        
        // Afficher le tooltip
        const cantonData = state.cantonData[d];
        const cantonName = cantonNames[d] || d;
        
        let tooltipContent = `<strong>Canton: ${cantonName}</strong><br>`;
        
        if (cantonData) {
            const sentimentData = getSentimentDataForDisplay(d);
            const sentimentValue = sentimentData.value.toFixed(3);
            const articleCount = sentimentData.count;
            
            tooltipContent += `Sentiment ${state.selectedSentimentType}: ${sentimentValue}<br>`;
            tooltipContent += `Nombre d'articles: ${articleCount}`;
        } else {
            tooltipContent += "Aucune donnée disponible";
        }
        
        tooltip.html(tooltipContent)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 28) + 'px')
            .style('opacity', 1);
    })
    .on('mouseout', function() {
        // Réinitialiser le style du canton
        d3.select(this).select('rect')
            .attr('stroke-width', 0.5)
            .attr('stroke', '#333');
        
        // Masquer le tooltip
        tooltip.style('opacity', 0);
    })
    .on('click', function(event, d) {
        // Afficher les détails du canton
        const cantonData = state.cantonData[d];
        const cantonName = cantonNames[d] || d;
        
        if (cantonData) {
            alert(`Canton: ${cantonName}\nNombre d'articles: ${cantonData.count}`);
        }
    });
}
    
    // Mettre à jour les couleurs
    updateVisualization();
}

// Mise à jour de la visualisation
function updateVisualization() {
    try {
        // Récupérer toutes les valeurs de sentiment pour la plage sélectionnée
        const sentimentValues = Object.keys(state.cantonData).map(canton => {
            return getSentimentDataForDisplay(canton).value;
        }).filter(value => !isNaN(value));
        
        // Vérifier si nous avons des données valides
        if (sentimentValues.length === 0) {
            console.warn("Aucune donnée de sentiment valide trouvée pour la visualisation");
            // Afficher un message à l'utilisateur
            const errorMessage = document.createElement('div');
            errorMessage.className = 'error-message';
            errorMessage.textContent = 'Aucune donnée de sentiment disponible pour cette sélection.';
            document.getElementById('map-container').appendChild(errorMessage);
            
            // Supprimer le message après un délai
            setTimeout(() => {
                if (errorMessage.parentNode) {
                    errorMessage.parentNode.removeChild(errorMessage);
                }
            }, config.errorDelay);
            
            return;
        }
        
        // Calculer le min et max pour l'échelle de couleur
        let min = Math.min(...sentimentValues);
        let max = Math.max(...sentimentValues);
        
        // Ajuster pour les sentiments compound qui vont de -1 à 1
        if (state.selectedSentimentType === 'compound') {
            min = Math.min(min, -0.5);
            max = Math.max(max, 0.5);
        } else {
            min = 0;
            max = Math.max(max, 1);
        }
        
        // Sélectionner l'interpolateur de couleur approprié en fonction du type de sentiment
        const colorInterpolator = config.colorScales[state.selectedSentimentType] || config.colors;
        
        // Créer l'échelle de couleur
        const colorScale = d3.scaleSequential()
            .domain([min, max])
            .interpolator(colorInterpolator);
        
        // Mettre à jour les couleurs des cantons
        if (state.useGridFallback) {
            // Pour la visualisation en grille
            d3.selectAll('.canton')
                .transition()
                .duration(config.transitionDuration)
                .attr('fill', function(d) {
                    const cantonId = d; // Dans la grille, d est directement l'ID du canton
                    const cantonData = state.cantonData[cantonId];
                    
                    if (cantonData) {
                        const sentimentData = getSentimentDataForDisplay(cantonId);
                        if (!isNaN(sentimentData.value)) {
                            return colorScale(sentimentData.value);
                        }
                    }
                    return '#ccc'; // Couleur par défaut pour les cantons sans données
                });
        } else {
            // Pour la visualisation GeoJSON
            d3.selectAll('.canton')
                .transition()
                .duration(config.transitionDuration)
                .attr('fill', function(feature) {
                    const cantonId = feature.id || feature.properties.id;
                    const cantonData = state.cantonData[cantonId];
                    
                    if (cantonData) {
                        const sentimentData = getSentimentDataForDisplay(cantonId);
                        if (!isNaN(sentimentData.value)) {
                            return colorScale(sentimentData.value);
                        }
                    }
                    return '#ccc'; // Couleur par défaut pour les cantons sans données
                });
        }
        
        // Mettre à jour la légende
        updateLegend(min, max, colorInterpolator);
    } catch (error) {
        console.error("Erreur lors de la mise à jour de la visualisation:", error);
        // Afficher un message d'erreur à l'utilisateur
        document.getElementById('map-container').innerHTML += 
            '<div class="error-message">Erreur lors de la mise à jour de la carte. Veuillez réessayer.</div>';
    }
}

// Récupérer les données de sentiment pour l'affichage
function getSentimentDataForDisplay(cantonId) {
    const cantonData = state.cantonData[cantonId];
    
    if (!cantonData) {
        return { value: NaN, count: 0 };
    }
    
    // Si toutes les années sont sélectionnées
    if (state.selectedYearRange === 'all') {
        switch (state.selectedSentimentType) {
            case 'compound':
                return { value: cantonData.avgCompound || 0, count: cantonData.count };
            case 'positive':
                return { value: cantonData.avgPositive || 0, count: cantonData.count };
            case 'negative':
                return { value: cantonData.avgNegative || 0, count: cantonData.count };
            case 'neutral':
                return { value: cantonData.avgNeutral || 0, count: cantonData.count };
            default:
                return { value: cantonData.avgCompound || 0, count: cantonData.count };
        }
    } else {
        // Si une plage d'années spécifique est sélectionnée
        const yearRangeData = cantonData.yearRanges[state.selectedYearRange];
        
        if (!yearRangeData || yearRangeData.count === 0) {
            return { value: NaN, count: 0 };
        }
        
        switch (state.selectedSentimentType) {
            case 'compound':
                return { value: yearRangeData.avgCompound || 0, count: yearRangeData.count };
            case 'positive':
                return { value: yearRangeData.avgPositive || 0, count: yearRangeData.count };
            case 'negative':
                return { value: yearRangeData.avgNegative || 0, count: yearRangeData.count };
            case 'neutral':
                return { value: yearRangeData.avgNeutral || 0, count: yearRangeData.count };
            default:
                return { value: yearRangeData.avgCompound || 0, count: yearRangeData.count };
        }
    }
}

// Mise à jour de la légende
function updateLegend(min, max, colorInterpolator) {
    // Vérifier si les paramètres sont valides
    if (isNaN(min) || isNaN(max)) {
        console.warn("Valeurs min/max invalides pour la légende");
        return;
    }
    
    // Utiliser l'interpolateur par défaut si non spécifié
    const interpolator = colorInterpolator || config.colors;
    
    // Adapter les étiquettes de la légende en fonction du type de sentiment
    const legendMin = document.querySelector('.legend-min');
    const legendMid = document.querySelector('.legend-mid');
    const legendMax = document.querySelector('.legend-max');
    
    if (!legendMin || !legendMid || !legendMax) {
        console.warn("Éléments de légende introuvables dans le DOM");
        return;
    }
    
    // Définir les étiquettes en fonction du type de sentiment
    if (state.selectedSentimentType === 'compound') {
        legendMin.textContent = 'Négatif';
        legendMid.textContent = 'Neutre';
        legendMax.textContent = 'Positif';
    } else if (state.selectedSentimentType === 'positive') {
        legendMin.textContent = 'Peu positif';
        legendMid.textContent = 'Moyennement positif';
        legendMax.textContent = 'Très positif';
    } else if (state.selectedSentimentType === 'negative') {
        legendMin.textContent = 'Peu négatif';
        legendMid.textContent = 'Moyennement négatif';
        legendMax.textContent = 'Très négatif';
    } else if (state.selectedSentimentType === 'neutral') {
        legendMin.textContent = 'Peu neutre';
        legendMid.textContent = 'Moyennement neutre';
        legendMax.textContent = 'Très neutre';
    }
    
    // Mettre à jour le gradient de couleur de la légende
    const colorLegend = document.getElementById('color-legend');
    
    if (!colorLegend) {
        console.warn("L'élément de légende de couleur est introuvable");
        return;
    }
    
    // Créer un gradient pour la légende
    const gradientColors = [];
    const steps = 10;
    
    try {
        for (let i = 0; i < steps; i++) {
            const value = min + (max - min) * (i / (steps - 1));
            const normalizedValue = (value - min) / (max - min);
            const color = interpolator(normalizedValue);
            gradientColors.push(`${color} ${i * 100 / (steps - 1)}%`);
        }
        
        colorLegend.style.background = `linear-gradient(to right, ${gradientColors.join(', ')})`;
    } catch (error) {
        console.error("Erreur lors de la création du gradient de couleur:", error);
        // Utiliser un gradient par défaut en cas d'erreur
        colorLegend.style.background = 'linear-gradient(to right, #313695, #4575b4, #74add1, #abd9e9, #e0f3f8, #fee090, #fdae61, #f46d43, #d73027, #a50026)';
    }
}
