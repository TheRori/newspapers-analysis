// Entity Cards JavaScript (version finale avec sélection de source de données)

// --- CONFIGURATION ---
const entityConfig = {
    // MODIFIÉ: Utilisation de templates pour les chemins
    pathTemplate: {
        summary: 'data/source/collections/entities/065dd3cd-0462-419e-b2ef-d8fde883eb71/source_files/entity_summary_{source_id}.json',
        articles: 'data/source/collections/entities/065dd3cd-0462-419e-b2ef-d8fde883eb71/source_files/articles_with_entities_{source_id}.json'
    },
    maxArticlesToShow: 50,
    entityColors: { 'ORG': '#00ff9d', 'LOC': '#00a0ff', 'PERSON': '#ff00a0' },
    defaultEntityType: 'ORG'
};

// --- APPLICATION STATE ---
let entityState = {
    summary: null, articles: null,
    selectedEntityType: entityConfig.defaultEntityType,
    selectedEntity: null, swiper: null, entityArticlesCache: {}, isLoading: false
};

// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

async function initApp() {
    // Initialiser les composants statiques qui ne dépendent pas des données
    initEntityTypeButtons();
    initSwiper();
    initModal();
    initDataSourceSelector(); // NOUVEAU: Initialise le menu déroulant

    // Charger les données pour la première fois avec la valeur par défaut du menu
    const initialSourceId = document.getElementById('date-range-select').value;
    await reloadData(initialSourceId);
}

// NOUVEAU: Fonction principale pour charger/recharger les données
async function reloadData(sourceId) {
    console.log(`Chargement des données pour la source : ${sourceId}`);
    document.querySelector('.container').style.opacity = '0.5'; // Effet de chargement
    entityState.isLoading = true;

    try {
        const [summary, articles] = await loadInitialData(sourceId);
        entityState.summary = summary;
        entityState.articles = articles;
        entityState.entityArticlesCache = {}; // Vider le cache lors du changement de source
        
        console.log(`Données initiales chargées: ${entityState.articles.length} articles disponibles.`);
        
        // Recharger la vue avec les nouvelles données
        loadEntityType(entityState.selectedEntityType);
        
    } catch (error) {
        console.error('Erreur critique lors du chargement des données:', error);
        document.querySelector('.container').innerHTML = 
            `<div class="error-message">Impossible de charger les données pour la source ${sourceId}: ${error.message}</div>`;
    } finally {
        entityState.isLoading = false;
        document.querySelector('.container').style.opacity = '1';
    }
}

// --- DATA LOADING (MODIFIÉ) ---
async function loadInitialData(sourceId) {
    // Construit les chemins dynamiquement
    const summaryPath = entityConfig.pathTemplate.summary.replace('{source_id}', sourceId);
    const articlesPath = entityConfig.pathTemplate.articles.replace('{source_id}', sourceId);

    const [summaryRes, articlesRes] = await Promise.all([
        fetch(summaryPath),
        fetch(articlesPath)
    ]);

    if (!summaryRes.ok) throw new Error(`Échec du chargement du résumé (${summaryPath}): ${summaryRes.statusText}`);
    if (!articlesRes.ok) throw new Error(`Échec du chargement des articles (${articlesPath}): ${articlesRes.statusText}`);

    return [await summaryRes.json(), await articlesRes.json()];
}

// --- GESTIONNAIRES D'ÉVÉNEMENTS (NOUVEAU) ---
function initDataSourceSelector() {
    const selector = document.getElementById('date-range-select');
    selector.addEventListener('change', (event) => {
        if (entityState.isLoading) return;
        reloadData(event.target.value);
    });
}

// --- LE RESTE DU CODE RESTE GLOBALEMENT INCHANGÉ ---

// --- GRAPHIQUE DE TENDANCE ---
function generateTrendGraph(entityName, entityType) {
    // ... (Cette fonction reste exactement la même)
    if (!entityState.articles) return `<text x="50" y="50" text-anchor="middle" font-size="10" fill="${entityConfig.entityColors[entityType]}">Chargement...</text>`;
    const entityArticles = entityState.articles.filter(article => article.entities && article.entities.some(e => e.text === entityName && e.label === entityType));
    if (entityArticles.length === 0) return `<text x="50" y="50" text-anchor="middle" font-size="10" fill="${entityConfig.entityColors[entityType]}">Pas de données</text>`;
    const yearCounts = {};
    entityArticles.forEach(article => {
        if (article.date) {
            const year = parseInt(article.date.substring(0, 4));
            if (year) yearCounts[year] = (yearCounts[year] || 0) + 1;
        }
    });
    const dataPoints = Object.entries(yearCounts).map(([year, value]) => ({ year: parseInt(year), value }));
    if (dataPoints.length < 2) return `<text x="50" y="50" text-anchor="middle" font-size="10" fill="${entityConfig.entityColors[entityType]}">Données insuffisantes</text>`;
    dataPoints.sort((a, b) => a.year - b.year);
    const minYear = dataPoints[0].year;
    const maxYear = dataPoints[dataPoints.length - 1].year;
    const yearRange = maxYear - minYear || 1;
    const maxValue = Math.max(...dataPoints.map(d => d.value));
    const normalizedPoints = dataPoints.map(d => ({ x: 10 + ((d.year - minYear) / yearRange) * 80, y: 90 - ((d.value / maxValue) * 70) }));
    const pathData = "M " + normalizedPoints.map(p => `${p.x},${p.y}`).join(" L ");
    return `<line x1="10" y1="90" x2="90" y2="90" stroke="${entityConfig.entityColors[entityType]}" stroke-width="1" opacity="0.5" /><text x="10" y="98" text-anchor="middle" font-size="8" fill="${entityConfig.entityColors[entityType]}" opacity="0.7">${minYear}</text><text x="90" y="98" text-anchor="middle" font-size="8" fill="${entityConfig.entityColors[entityType]}" opacity="0.7">${maxYear}</text><path d="${pathData}" fill="none" stroke="${entityConfig.entityColors[entityType]}" stroke-width="2" />${normalizedPoints.map(p => `<circle cx="${p.x}" cy="${p.y}" r="1.5" fill="${entityConfig.entityColors[entityType]}" />`).join('')}`;
}

// --- UI & LOGIC ---
function loadEntityType(entityType) {
    entityState.selectedEntityType = entityType;
    entityState.selectedEntity = null;
    resetEntityDetails();
    if (!entityState.summary) { // Sécurité si les données ne sont pas prêtes
        console.warn("Les données de résumé ne sont pas encore disponibles.");
        return;
    }
    const topEntities = entityState.summary.entity_frequencies[entityType];
    if (!topEntities || Object.keys(topEntities).length === 0) {
        document.getElementById('entity-cards').innerHTML = `<div class="swiper-slide"><p>Aucune entité de type ${entityType} trouvée.</p></div>`;
        if (entityState.swiper) entityState.swiper.update();
        return;
    }
    const entityCardsHTML = createEntityCards(topEntities, entityType);
    document.getElementById('entity-cards').innerHTML = entityCardsHTML;
    if (entityState.swiper) {
        entityState.swiper.update();
        entityState.swiper.slideTo(0);
    }
    document.querySelectorAll('.entity-card').forEach(card => {
        card.addEventListener('click', () => {
            if (entityState.isLoading) return;
            const entityName = card.getAttribute('data-entity');
            const entityCount = parseInt(card.getAttribute('data-count'));
            selectEntity(entityName, entityCount, entityType);
        });
    });
}

async function selectEntity(entityName, entityCount, entityType) {
    if (entityState.isLoading || entityState.selectedEntity === entityName) return;
    entityState.isLoading = true;
    entityState.selectedEntity = entityName;
    document.querySelectorAll('.entity-card').forEach(card => {
        card.classList.toggle('active', card.getAttribute('data-entity') === entityName);
    });
    document.getElementById('selected-entity-name').textContent = entityName;
    document.getElementById('entity-count').textContent = entityCount.toLocaleString();
    document.getElementById('entity-articles-list').innerHTML = '<p class="loading">Filtrage des articles...</p>';
    await new Promise(resolve => setTimeout(resolve, 50));
    let articlesForEntity = [];
    if (entityState.entityArticlesCache[entityName]) {
        articlesForEntity = entityState.entityArticlesCache[entityName];
    } else {
        if (entityState.articles) {
            articlesForEntity = entityState.articles.filter(article => article.entities && article.entities.some(e => e.text === entityName && e.label === entityType));
            entityState.entityArticlesCache[entityName] = articlesForEntity;
        }
    }
    document.getElementById('article-count').textContent = articlesForEntity.length.toLocaleString();
    updateYearRange(articlesForEntity);
    displayArticlesForEntity(articlesForEntity, entityName);
    entityState.isLoading = false;
}

function createEntityCards(entities, entityType) {
    // ... (Cette fonction reste exactement la même)
    return Object.entries(entities).map(([name, count]) => ({ name, count })).sort((a, b) => b.count - a.count).slice(0, 20).map((entity, index) => `<div class="swiper-slide"><div class="entity-card" data-entity="${entity.name}" data-count="${entity.count}"><div class="entity-rank">${index + 1}</div><h3 class="entity-name">${entity.name}</h3><div class="entity-count">${entity.count.toLocaleString()}</div><div class="entity-visual"><svg viewBox="0 0 100 100">${generateTrendGraph(entity.name, entityType)}</svg></div></div></div>`).join('');
}

function displayArticlesForEntity(articles, entityName) {
    // ... (Cette fonction reste exactement la même)
    const container = document.getElementById('entity-articles-list');
    if (articles.length === 0) {
        container.innerHTML = '<p class="no-articles">Aucun article trouvé pour cette entité.</p>';
        return;
    }
    container.innerHTML = articles.sort((a, b) => (b.date || '').localeCompare(a.date || '')).slice(0, entityConfig.maxArticlesToShow).map(article => `<div class="article-item" data-article-id="${article.id}"><div class="article-title">${article.title || 'Sans titre'}</div><div class="article-meta"><span class="article-date">${formatDate(article.date)}</span><span class="article-newspaper">${article.newspaper || 'Journal inconnu'}</span></div></div>`).join('');
    container.querySelectorAll('.article-item').forEach(item => {
        item.addEventListener('click', () => {
            const articleId = item.getAttribute('data-article-id');
            const article = entityState.articles.find(a => a.id === articleId);
            if (article) showArticleModal(article, entityName);
        });
    });
}

// --- FONCTIONS UTILITAIRES ---
function initEntityTypeButtons() {
    // ... (Cette fonction reste exactement la même)
    document.querySelectorAll('.entity-type-button').forEach(button => {
        button.addEventListener('click', () => {
            if (entityState.isLoading) return;
            document.querySelector('.entity-type-button.active').classList.remove('active');
            button.classList.add('active');
            loadEntityType(button.dataset.type);
        });
    });
}

function initSwiper() { /* ... (inchangée) ... */ 
    entityState.swiper = new Swiper('.entity-swiper', { slidesPerView: 1, spaceBetween: 20, centeredSlides: true, grabCursor: true, keyboard: { enabled: true }, pagination: { el: '.swiper-pagination', clickable: true }, navigation: { nextEl: '.swiper-button-next', prevEl: '.swiper-button-prev' }, breakpoints: { 640: { slidesPerView: 2 }, 1024: { slidesPerView: 3 }, 1200: { slidesPerView: 4 } } });
}
function initModal() { /* ... (inchangée) ... */ 
    const modal = document.getElementById('article-modal');
    const closeBtn = modal.querySelector('.close');
    closeBtn.onclick = () => { modal.style.display = 'none'; };
    window.onclick = (event) => { if (event.target == modal) modal.style.display = 'none'; };
}
function showArticleModal(article, entityName) { /* ... (inchangée) ... */ 
    const modal = document.getElementById('article-modal');
    document.getElementById('modal-title').textContent = article.title || 'Sans titre';
    document.getElementById('article-date').textContent = formatDate(article.date);
    document.getElementById('article-newspaper').textContent = article.newspaper || 'Journal inconnu';
    let content = article.content || 'Contenu non disponible';
    const regex = new RegExp(`\\b(${entityName})\\b`, 'gi');
    content = content.replace(regex, '<span class="highlighted-entity">$1</span>');
    document.getElementById('article-content').innerHTML = content.replace(/\n/g, '<br>');
    modal.style.display = 'block';
}
function updateYearRange(articles) { /* ... (inchangée) ... */ 
    const yearRangeEl = document.getElementById('year-range');
    if (articles.length === 0) { yearRangeEl.textContent = '-'; return; }
    const years = articles.map(a => a.date ? parseInt(a.date.substring(0, 4)) : null).filter(Boolean);
    if (years.length === 0) { yearRangeEl.textContent = '-'; return; }
    const minYear = Math.min(...years);
    const maxYear = Math.max(...years);
    yearRangeEl.textContent = minYear === maxYear ? minYear : `${minYear}-${maxYear}`;
}
function resetEntityDetails() { /* ... (inchangée) ... */ 
    document.getElementById('selected-entity-name').textContent = 'Sélectionnez une entité';
    document.getElementById('entity-count').textContent = '0';
    document.getElementById('article-count').textContent = '0';
    document.getElementById('year-range').textContent = '-';
    document.getElementById('entity-articles-list').innerHTML = '<p class="no-selection">Sélectionnez une entité dans le carrousel pour voir les articles associés</p>';
}
function formatDate(dateString) { /* ... (inchangée) ... */
    if (!dateString) return 'Date inconnue';
    try { const [year, month, day] = dateString.split('-'); return `${day}/${month}/${year}`; } catch (e) { return dateString; }
}