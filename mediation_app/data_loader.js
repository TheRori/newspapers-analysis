// Module de chargement de données global pour l'application de médiation
// Ce module centralise le chargement des données pour toutes les visualisations

// Configuration globale
const dataConfig = {
    dataPath: 'data/source/collections/term_tracking_chrono/0cc5204f-5f9b-464e-89aa-880a31b514d1/source_files/term_tracking_results.csv',
    articlesPath: 'data/source/articles_v1_filtered.json',
    timelineEventsPath: 'config/timeline_events.json',
    timelineDataPath: 'data/source/collections/term_tracking_chrono/ab7dd30b-8f09-46d0-ae5e-08aabddd1a90/source_files/term_tracking_results.csv',
    maxTermsToShow: 8,
    maxArticlesToShow: 20,
    colors: d3.schemeSet2,
    cantonColors: {
        'FR': '#c8102e', // Rouge
        'GE': '#e3000f', // Rouge
        'JU': '#cf142b', // Rouge
        'NE': '#008000', // Vert
        'VD': '#008000', // Vert
        'VS': '#ff0000', // Rouge
        'BE': '#ff0000', // Rouge
        'TI': '#ff0000', // Rouge
        'ZH': '#0066cc', // Bleu
        'other': '#666666' // Gris pour les autres
    }
};

// État global des données
const dataState = {
    isLoaded: false,
    isLoading: false,
    error: null,
    data: null,
    rawData: null,
    articles: null,
    terms: [],
    years: [],
    cantons: [],
    newspapers: [],
    timelineEvents: [],
    callbacks: [] // Callbacks à exécuter une fois les données chargées
};

// Fonction pour charger toutes les données nécessaires
async function loadAllData() {
    // Éviter les chargements multiples
    if (dataState.isLoading) {
        console.log('Chargement des données déjà en cours...');
        return new Promise((resolve) => {
            dataState.callbacks.push(resolve);
        });
    }
    
    if (dataState.isLoaded) {
        console.log('Données déjà chargées');
        return Promise.resolve(dataState);
    }
    
    console.log('Début du chargement des données...');
    dataState.isLoading = true;
    dataState.error = null;
    
    try {
        console.log('Chargement des données depuis:', dataConfig.dataPath);
        
        // Charger les données CSV
        const response = await fetch(dataConfig.dataPath);
        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }
        
        const csvText = await response.text();
        const data = d3.csvParse(csvText);
        
        // Stocker les données brutes
        dataState.rawData = data;
        
        console.log('Données CSV chargées:', data.length, 'lignes');
        
        // Charger les articles d'abord pour pouvoir les associer aux données de timeline
        let articles = [];
        try {
            console.log('Chargement des articles depuis:', dataConfig.articlesPath);
            const articlesResponse = await fetch(dataConfig.articlesPath);
            if (articlesResponse.ok) {
                articles = await articlesResponse.json();
                dataState.articles = articles;
                console.log('Articles chargés:', articles.length);
            } else {
                console.warn('Impossible de charger les articles:', articlesResponse.statusText);
            }
        } catch (err) {
            console.warn('Erreur lors du chargement des articles:', err);
        }
        
        // Charger les données spécifiques pour la timeline (nouveaux mots-clés des événements historiques)
        let timelineData = [];
        try {
            console.log('Chargement des données de timeline depuis:', dataConfig.timelineDataPath);
            const timelineResponse = await fetch(dataConfig.timelineDataPath);
            if (timelineResponse.ok) {
                const timelineCsvText = await timelineResponse.text();
                console.log('Contenu CSV brut de timeline:', timelineCsvText.substring(0, 200) + '...');
                
                // Analyser le CSV
                timelineData = d3.csvParse(timelineCsvText);
                console.log('Données de timeline chargées:', timelineData.length, 'lignes');
                console.log('Structure des données de timeline:', Object.keys(timelineData[0]));
                
                // Vérifier si les données sont valides
                if (timelineData.length > 0 && timelineData[0].key) {
                    // Convertir les valeurs en nombres
                    timelineData = timelineData.map(d => {
                        const newRow = { key: d.key };
                        Object.keys(d).forEach(key => {
                            if (key !== 'key') {
                                newRow[key] = parseFloat(d[key]) || 0;
                            }
                        });
                        return newRow;
                    });
                    
                    // Transformer les données par article en données par année
                    console.log('Transformation des données par article en données par année...');
                    const yearlyData = {};
                    
                    // Préparer une structure pour stocker les articles disponibles par terme et par année
                    dataState.availableArticles = {};
                    
                    // Compteur pour suivre le nombre d'articles associés
                    let matchedArticlesCount = 0;
                    
                    timelineData.forEach(article => {
                        // Extraire l'année de la clé d'article (format: article_YYYY-MM-DD_...)
                        const match = article.key.match(/article_(\d{4})-/);
                        if (match) {
                            const year = match[1];
                            
                            // Initialiser l'entrée pour cette année si elle n'existe pas
                            if (!yearlyData[year]) {
                                yearlyData[year] = { key: year };
                                // Initialiser tous les termes à 0
                                Object.keys(article).forEach(key => {
                                    if (key !== 'key') {
                                        yearlyData[year][key] = 0;
                                    }
                                });
                            }
                            
                            // Ajouter les valeurs de cet article aux totaux de l'année
                            Object.keys(article).forEach(key => {
                                if (key !== 'key' && article[key]) {
                                    const value = parseFloat(article[key]) || 0;
                                    yearlyData[year][key] += value;
                                    
                                    // Si l'article contient ce terme (valeur > 0), l'ajouter aux articles disponibles
                                    if (value > 0) {
                                        // Initialiser la structure si nécessaire
                                        if (!dataState.availableArticles[year]) {
                                            dataState.availableArticles[year] = {};
                                        }
                                        if (!dataState.availableArticles[year][key]) {
                                            dataState.availableArticles[year][key] = [];
                                        }
                                        
                                        // Fonction pour normaliser les termes (pour la comparaison)
                                        function normalizeTerm(term) {
                                            if (!term) return '';
                                            return term.toLowerCase()
                                                .normalize("NFD")
                                                .replace(/[\u0300-\u036f]/g, "") // Supprimer les accents
                                                .replace(/[^a-z0-9]/g, ""); // Garder uniquement les lettres et chiffres
                                        }
                                        
                                        // Chercher l'article correspondant dans le fichier JSON
                                        // Convertir les IDs en string pour la comparaison
                                        const articleId = article.key;
                                        
                                        // Débogage: afficher les 5 premiers articles pour comprendre leur structure
                                        if (timelineData.indexOf(article) < 5) {
                                            console.log('Article de timeline à associer:', articleId);
                                            console.log('Exemple d\'ID d\'article JSON:', articles.length > 0 ? articles[0].id : 'aucun article');
                                            console.log('Termes disponibles dans cet article:');
                                            Object.keys(article).forEach(key => {
                                                if (key !== 'key' && parseFloat(article[key]) > 0) {
                                                    console.log(`  - ${key}: ${article[key]}`);
                                                }
                                            });
                                        }
                                        
                                        // Essayer de trouver l'article avec différentes méthodes de correspondance
                                        let fullArticle = articles.find(a => String(a.id) === String(articleId));
                                        
                                        // Si pas trouvé, essayer de trouver l'article par une correspondance partielle
                                        if (!fullArticle && articleId) {
                                            // Extraire les composants de l'ID pour une correspondance plus flexible
                                            const idMatch = articleId.match(/article_(\d{4})-(\d{2})-(\d{2})_([^_]+)/);
                                            if (idMatch) {
                                                const year = idMatch[1];
                                                const month = idMatch[2];
                                                const day = idMatch[3];
                                                const journal = idMatch[4].replace(/_/g, ' ');
                                                
                                                // Rechercher par date et journal
                                                fullArticle = articles.find(a => {
                                                    if (!a.id) return false;
                                                    const aId = String(a.id);
                                                    // Vérifier si l'ID contient la date et le journal
                                                    return aId.includes(`${year}-${month}-${day}`) && 
                                                           (aId.includes(journal) || 
                                                            (a.newspaper && a.newspaper.toLowerCase().includes(journal.toLowerCase())));
                                                });
                                                
                                                if (fullArticle && timelineData.indexOf(article) < 5) {
                                                    console.log('Article trouvé par date et journal:', articleId, '->', fullArticle.id);
                                                }
                                            }
                                            
                                            // Si toujours pas trouvé, essayer une correspondance plus large
                                            if (!fullArticle) {
                                                // Extraire l'identifiant de base (sans le préfixe 'article_')
                                                const baseId = articleId.replace(/^article_/, '');
                                                fullArticle = articles.find(a => {
                                                    if (!a.id) return false;
                                                    const aId = String(a.id);
                                                    return aId.includes(baseId) || baseId.includes(aId);
                                                });
                                                
                                                if (fullArticle && timelineData.indexOf(article) < 5) {
                                                    console.log('Article trouvé par correspondance partielle:', articleId, '->', fullArticle.id);
                                                }
                                            }
                                        }
                                        
                                        // Si toujours pas trouvé, créer un article fictif avec les informations disponibles
                                        if (!fullArticle) {
                                            // Extraire les informations de l'ID
                                            const dateParts = articleId.match(/article_(\d{4})-(\d{2})-(\d{2})_([^_]+)/);
                                            let extractedDate = '';
                                            let extractedJournal = '';
                                            
                                            if (dateParts) {
                                                extractedDate = `${dateParts[1]}-${dateParts[2]}-${dateParts[3]}`;
                                                extractedJournal = dateParts[4].replace(/_/g, ' ');
                                            }
                                            
                                            fullArticle = {
                                                id: articleId,
                                                title: `Article du ${extractedDate} - ${extractedJournal}`,
                                                date: extractedDate,
                                                newspaper: extractedJournal,
                                                content: `Contenu non disponible pour l'article ${articleId}. Cet article contient les termes suivants: ${Object.keys(article).filter(key => key !== 'key' && parseFloat(article[key]) > 0).join(', ')}.`
                                            };
                                        }
                                        
                                        if (fullArticle) {
                                            matchedArticlesCount++;
                                        } else if (timelineData.indexOf(article) < 10) {
                                            console.log('Article non trouvé dans le fichier JSON:', articleId);
                                        }
                                        
                                        // Créer un objet article avec les informations nécessaires
                                        const articleObj = {
                                            id: articleId,
                                            content: fullArticle ? fullArticle.content || fullArticle.original_content || '' : '',
                                            title: fullArticle ? fullArticle.title || '' : '',
                                            date: fullArticle ? fullArticle.date || year : year,
                                            newspaper: fullArticle ? fullArticle.newspaper || '' : '',
                                            year: year,
                                            value: value
                                        };
                                        
                                        dataState.availableArticles[year][key].push(articleObj);
                                    }
                                }
                            });
                        }
                    });
                    
                    console.log(`Articles associés aux données de timeline: ${matchedArticlesCount}`);
                    
                    // Convertir l'objet en tableau
                    const yearlyDataArray = Object.values(yearlyData);
                    console.log('Données transformées par année:', yearlyDataArray.length, 'années');
                    console.log('Articles disponibles par terme et par année:', dataState.availableArticles);
                    
                    // Stocker les données de timeline séparément
                    dataState.timelineData = yearlyDataArray;
                    console.log('Données de timeline traitées:', dataState.timelineData);
                } else {
                    console.warn('Format de données de timeline invalide');
                }
            } else {
                console.warn('Impossible de charger les données de timeline:', timelineResponse.statusText);
            }
        } catch (err) {
            console.warn('Erreur lors du chargement des données de timeline:', err);
        }
        
        // Les articles ont déjà été chargés au début de la fonction
        
        // Charger les événements de la timeline
        try {
            console.log('Chargement des événements de la timeline depuis:', dataConfig.timelineEventsPath);
            const timelineResponse = await fetch(dataConfig.timelineEventsPath);
            if (timelineResponse.ok) {
                const timelineEventsData = await timelineResponse.json();
                dataState.timelineEvents = timelineEventsData.events || [];
                console.log('Événements de la timeline chargés:', dataState.timelineEvents.length);
            } else {
                console.warn('Impossible de charger les événements de la timeline:', timelineResponse.statusText);
            }
        } catch (err) {
            console.warn('Erreur lors du chargement des événements de la timeline:', err);
        }
        
        // Extraire les termes (colonnes sauf 'key')
        dataState.terms = Object.keys(data[0]).filter(key => key !== 'key');
        console.log('Termes extraits du CSV:', dataState.terms);
        
        // Extraire les années, journaux et cantons à partir des clés d'articles
        const articleInfo = data.map(row => {
            const parts = row.key.split('_');
            const dateStr = parts[1];
            
            // Trouver l'article correspondant dans le fichier JSON si disponible
            const articleDetails = articles.find(a => a.id === row.key) || {};
            
            // Fonction pour nettoyer les noms de journaux
            function cleanJournalName(name) {
                if (!name) return 'inconnu';
                
                // Supprimer les numéros d'édition (comme "18." à la fin)
                let cleanedName = name.replace(/\s+\d+\.?\s*$/, '');
                
                // Supprimer les points à la fin
                cleanedName = cleanedName.replace(/\.\s*$/, '');
                
                // Supprimer les espaces en trop
                cleanedName = cleanedName.trim();
                
                return cleanedName || 'inconnu';
            }
            
            // Utiliser le nom complet du journal depuis le fichier JSON si disponible
            const rawJournalName = articleDetails.newspaper || parts[3] || 'inconnu';
            
            // Nettoyer le nom du journal
            const journal = cleanJournalName(rawJournalName);
            const canton = articleDetails.canton || 'unknown';
            
            return {
                id: row.key,
                year: dateStr.substring(0, 4),
                date: dateStr,
                journal: journal,
                canton: canton,
                values: dataState.terms.reduce((acc, term) => {
                    acc[term] = row[term] ? parseFloat(row[term]) : 0;
                    return acc;
                }, {}),
                details: articleDetails
            };
        });
        
        // Obtenir la liste des années uniques et les trier
        dataState.years = [...new Set(articleInfo.map(item => item.year))].sort();
        
        // Obtenir la liste des cantons et journaux uniques
        dataState.cantons = [...new Set(articleInfo.map(item => item.canton).filter(c => c !== 'unknown'))];
        dataState.newspapers = [...new Set(articleInfo.map(item => item.journal).filter(j => j !== 'inconnu'))];
        
        // Agréger les données par année pour la timeline
        const aggregatedData = [];
        dataState.years.forEach(year => {
            const yearArticles = articleInfo.filter(item => item.year === year);
            const yearData = { key: year };
            
            // Calculer la moyenne des valeurs pour chaque terme
            dataState.terms.forEach(term => {
                const values = yearArticles.map(article => article.values[term]).filter(v => !isNaN(v));
                if (values.length > 0) {
                    const sum = values.reduce((a, b) => a + b, 0);
                    yearData[term] = sum / values.length;
                } else {
                    yearData[term] = 0;
                }
            });
            
            aggregatedData.push(yearData);
        });
        
        // Trier les données agrégées par année
        aggregatedData.sort((a, b) => parseInt(a.key) - parseInt(b.key));
        
        // Mettre à jour les données pour la timeline
        dataState.data = aggregatedData;
        console.log('Données agrégées par année:', aggregatedData);
        
        console.log('Années disponibles:', dataState.years);
        console.log('Cantons disponibles:', dataState.cantons);
        console.log('Journaux disponibles:', dataState.newspapers);
        
        // Marquer les données comme chargées
        dataState.isLoaded = true;
        dataState.isLoading = false;
        dataState.error = null;
        
        // Exécuter tous les callbacks en attente
        dataState.callbacks.forEach(callback => callback(dataState));
        dataState.callbacks = [];
        
        return dataState;
    } catch (error) {
        console.error('Erreur lors du chargement des données:', error);
        dataState.isLoading = false;
        dataState.error = error;
        
        // Exécuter tous les callbacks en attente avec l'erreur
        dataState.callbacks.forEach(callback => callback(null, error));
        dataState.callbacks = [];
        
        throw error;
    }
}

// Fonction pour charger les articles depuis un fichier Parquet
async function loadArticlesFromParquet(parquetPath) {
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
        
        // Convertir la table Arrow en format compatible avec notre application
        return table.toArray().map(row => {
            const obj = {};
            table.schema.fields.forEach((field, i) => {
                obj[field.name] = row[i];
            });
            return obj;
        });
    } catch (error) {
        console.error('Erreur lors du chargement du fichier Parquet:', error);
        // En cas d'erreur, on revient au JSON
        const response = await fetch(parquetPath.replace('.parquet', '.json'));
        return await response.json();
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

// Fonction pour obtenir les données (avec chargement si nécessaire)
function getData() {
    if (dataState.isLoaded) {
        return Promise.resolve(dataState);
    } else {
        return loadAllData();
    }
}

// Exposer les fonctions et l'état pour l'accès externe
window.dataLoader = {
    config: dataConfig,
    state: dataState,
    loadAllData,
    getData,
    availableArticles: dataState.availableArticles
};

// Charger les données automatiquement au chargement de la page
document.addEventListener('DOMContentLoaded', () => {
    console.log('Chargement automatique des données...');
    loadAllData().catch(error => {
        console.error('Erreur lors du chargement automatique des données:', error);
    });
});
