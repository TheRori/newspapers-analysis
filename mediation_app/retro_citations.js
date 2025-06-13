// Script pour la page de citations rétro
document.addEventListener('DOMContentLoaded', async () => {
    // Initialiser les variables
    let allArticles = [];
    let filteredArticles = [];
    let swiper;
    let activeTerms = ['informatique', 'ordinateur', 'internet'];
    
    // Fonction pour extraire un extrait pertinent contenant le terme recherché
    function extractRelevantExcerpt(content, term, maxLength = 300) {
        if (!content) return "Contenu non disponible";
        
        // Normaliser le terme pour la recherche (sans accents, minuscules)
        const normalizedTerm = term.toLowerCase()
            .normalize("NFD")
            .replace(/[\u0300-\u036f]/g, "");
        
        // Normaliser le contenu pour la recherche
        const normalizedContent = content.toLowerCase()
            .normalize("NFD")
            .replace(/[\u0300-\u036f]/g, "");
        
        // Trouver l'index du terme dans le contenu normalisé
        const termIndex = normalizedContent.indexOf(normalizedTerm);
        
        if (termIndex === -1) {
            // Si le terme n'est pas trouvé, retourner le début du contenu
            return content.substring(0, maxLength) + "...";
        }
        
        // Calculer les positions de début et de fin de l'extrait
        const startPos = Math.max(0, termIndex - 100);
        const endPos = Math.min(content.length, termIndex + normalizedTerm.length + 200);
        
        // Extraire l'extrait
        let excerpt = content.substring(startPos, endPos);
        
        // Ajouter des points de suspension si nécessaire
        if (startPos > 0) excerpt = "..." + excerpt;
        if (endPos < content.length) excerpt = excerpt + "...";
        
        return excerpt;
    }
    
    // Fonction pour charger les articles depuis le fichier JSON
    async function loadArticles() {
        try {
            // Afficher l'overlay de chargement
            showLoaderOverlay();
            console.log('Chargement des articles...');
            const response = await fetch('data/source/articles_v1_filtered.json');
            if (!response.ok) {
                throw new Error(`Erreur HTTP: ${response.status}`);
            }
            const articles = await response.json();
            console.log(`${articles.length} articles chargés`);

            // Filtrer les articles qui contiennent au moins un des termes recherchés
            allArticles = articles.filter(article => {
                if (!article.topics) return false;
                return activeTerms.some(term => article.topics.includes(term));
            });
            console.log(`${allArticles.length} articles pertinents trouvés`);
            allArticles.sort((a, b) => {
                if (!a.date) return 1;
                if (!b.date) return -1;
                return new Date(a.date) - new Date(b.date);
            });

            // Grouper les articles par terme
            const articlesByTerm = {};
            activeTerms.forEach(term => {
                articlesByTerm[term] = allArticles.filter(article =>
                    article.topics && article.topics.includes(term)
                );
                articlesByTerm[term].sort((a, b) => {
                    if (!a.date) return 1;
                    if (!b.date) return -1;
                    return new Date(a.date) - new Date(b.date);
                });
                console.log(`${term}: ${articlesByTerm[term].length} articles`);
            });
            const firstArticlesByTerm = {};
            activeTerms.forEach(term => {
                if (articlesByTerm[term] && articlesByTerm[term].length > 0) {
                    firstArticlesByTerm[term] = articlesByTerm[term].slice(0, 5);
                } else {
                    firstArticlesByTerm[term] = [];
                }
            });
            // Créer une liste d'articles à afficher dans le carrousel
            filteredArticles = [];
            activeTerms.forEach(term => {
                firstArticlesByTerm[term].forEach(article => {
                    filteredArticles.push({ ...article, primaryTerm: term });
                });
            });
            shuffleArray(filteredArticles);

            // Chargement progressif par lots
            await loadArticlesInBatches(filteredArticles, 20);

        } catch (error) {
            console.error('Erreur lors du chargement des articles:', error);
        } finally {
            // Retirer l'overlay de chargement
            hideLoaderOverlay();
        }
    }

    // Affichage progressif des articles dans le carrousel
    async function loadArticlesInBatches(articles, batchSize = 20) {
        const total = articles.length;
        let loaded = 0;
        filteredArticles = [];
        while (loaded < total) {
            const batch = articles.slice(loaded, loaded + batchSize);
            filteredArticles.push(...batch);
            displayArticles();
            loaded += batchSize;
            await new Promise(resolve => setTimeout(resolve, 0)); // Laisse le navigateur respirer
        }
    }

    // Overlay spinner (inspiré entity_cards.js)
    function showLoaderOverlay() {
        let overlay = document.getElementById('loader-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'loader-overlay';
            overlay.innerHTML = `
                <div class="loading-spinner">
                    <div class="spinner-circle"></div>
                    <div class="spinner-circle"></div>
                    <div class="spinner-circle"></div>
                    <div class="spinner-circle"></div>
                </div>
                <p>Chargement des citations...</p>
            `;
            const container = document.querySelector('.container');
            if (container) {
                container.appendChild(overlay);
            }
        } else {
            overlay.style.display = 'flex';
        }
    }
    function hideLoaderOverlay() {
        const overlay = document.getElementById('loader-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }
    
    // Fonction pour mélanger un tableau (algorithme de Fisher-Yates)
    function shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
    }
    
    // Fonction pour afficher les articles dans le carrousel
    function displayArticles() {
        const container = document.getElementById('citations-container');
        container.innerHTML = '';
        
        // Filtrer les articles en fonction des termes actifs
        const articlesToDisplay = filteredArticles.filter(article => 
            activeTerms.includes(article.primaryTerm)
        );
        
        if (articlesToDisplay.length === 0) {
            container.innerHTML = `
                <div class="swiper-slide">
                    <div class="citation-card">
                        <div class="term-title">Aucun article trouvé</div>
                        <div class="citation-text">Aucun article ne correspond aux critères sélectionnés.</div>
                    </div>
                </div>
            `;
        } else {
            articlesToDisplay.forEach(article => {
                const term = article.primaryTerm;
                const year = article.date ? article.date.substring(0, 4) : 'Date inconnue';
                const newspaper = article.newspaper || 'Source inconnue';
                
                // Extraire un extrait pertinent contenant le terme
                const excerpt = extractRelevantExcerpt(
                    article.content || article.original_content, 
                    term
                );
                
                // Créer une carte pour l'article
                const slide = document.createElement('div');
                slide.className = 'swiper-slide';
                
                // Déterminer l'icône en fonction du terme
                let iconSvg = '';
                if (term === 'ordinateur') {
                    iconSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="80" height="80" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="20" y="20" width="60" height="45" rx="2" />
                        <line x1="20" y1="55" x2="80" y2="55" />
                        <rect x="35" y="65" width="30" height="5" />
                        <rect x="30" y="70" width="40" height="10" />
                    </svg>`;
                } else if (term === 'internet') {
                    iconSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="80" height="80" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="50" cy="50" r="40" />
                        <ellipse cx="50" cy="50" rx="40" ry="20" />
                        <line x1="10" y1="50" x2="90" y2="50" />
                        <line x1="50" y1="10" x2="50" y2="90" />
                    </svg>`;
                } else if (term === 'informatique') {
                    iconSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="80" height="80" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="20" y="20" width="60" height="40" rx="2" />
                        <rect x="35" y="60" width="30" height="20" rx="2" />
                        <line x1="50" y1="30" x2="50" y2="50" />
                        <line x1="40" y1="40" x2="60" y2="40" />
                        <line x1="35" y1="70" x2="65" y2="70" />
                    </svg>`;
                }
                
                slide.innerHTML = `
                    <div class="citation-card ${term}">
                        <div class="term-icon" style="color: var(--retro-${term === 'informatique' ? 'cyan' : term === 'ordinateur' ? 'green' : 'magenta'})">
                            ${iconSvg}
                        </div>
                        <div class="term-title ${term}">${term.charAt(0).toUpperCase() + term.slice(1)}</div>
                        <div class="term-year ${term}">${year}</div>
                        <div class="citation-text">${excerpt}</div>
                        <div class="citation-source">${newspaper}, ${article.date || 'Date inconnue'}</div>
                        <div class="article-links">
                            ${article.url ? `<a href="${article.url}" target="_blank" class="view-article-link">Voir l'article original</a>` : ''}
                            <button class="view-full-article" data-id="${article.id}">Voir l'article complet</button>
                        </div>
                    </div>
                `;
                
                container.appendChild(slide);
            });
        }
        
        // Initialiser ou mettre à jour le swiper
        if (swiper) {
            swiper.destroy();
        }
        
        swiper = new Swiper('.retro-swiper', {
            slidesPerView: 1,
            spaceBetween: 30,
            loop: articlesToDisplay.length > 1,
            autoplay: {
                delay: 8000,
                disableOnInteraction: false,
            },
            pagination: {
                el: '.swiper-pagination',
                clickable: true,
            },
            navigation: {
                nextEl: '.swiper-button-next',
                prevEl: '.swiper-button-prev',
            },
            // Ajouter des marges sur les côtés pour les boutons de navigation
            slidesOffsetBefore: 20,
            slidesOffsetAfter: 20,
            // Améliorer l'effet de transition
            effect: 'fade',
            fadeEffect: {
                crossFade: true
            },
        });
    }
    
    // Gérer les clics sur les filtres de termes
    document.querySelectorAll('.term-filter').forEach(button => {
        button.addEventListener('click', () => {
            const term = button.getAttribute('data-term');
            
            if (button.classList.contains('active')) {
                // Si c'est le seul terme actif, ne pas le désactiver
                if (activeTerms.length === 1 && activeTerms.includes(term)) {
                    return;
                }
                
                // Désactiver le terme
                button.classList.remove('active');
                activeTerms = activeTerms.filter(t => t !== term);
            } else {
                // Activer le terme
                button.classList.add('active');
                if (!activeTerms.includes(term)) {
                    activeTerms.push(term);
                }
            }
            
            // Mettre à jour l'affichage
            displayArticles();
        });
    });
    
    // Gérer le clic sur le bouton de citation aléatoire
    document.getElementById('random-citation').addEventListener('click', () => {
        if (swiper && filteredArticles.length > 0) {
            const randomIndex = Math.floor(Math.random() * filteredArticles.length);
            swiper.slideTo(randomIndex);
        }
    });
    
    // Fonction pour afficher l'article complet dans le modal
    function showFullArticle(articleId) {
        console.log('Affichage de l\'article complet:', articleId);
        
        // Trouver l'article correspondant
        const article = allArticles.find(a => a.id === articleId || a.base_id === articleId);
        
        if (!article) {
            console.error('Article non trouvé:', articleId);
            return;
        }
        
        // Mettre à jour le contenu du modal
        document.getElementById('modal-title').textContent = article.title || 'Article sans titre';
        document.getElementById('full-article-content').textContent = article.content || article.original_content || 'Contenu non disponible';
        document.getElementById('article-date').textContent = article.date || 'Date inconnue';
        document.getElementById('article-newspaper').textContent = article.newspaper || 'Source inconnue';
        
        // Mettre à jour le lien vers la source originale
        const sourceLink = document.getElementById('article-source-link');
        if (article.url) {
            sourceLink.href = article.url;
            sourceLink.style.display = 'block';
        } else {
            sourceLink.style.display = 'none';
        }
        
        // Afficher le modal
        const modal = document.getElementById('article-modal');
        modal.style.display = 'block';
        
        // Mettre en surbrillance le terme recherché dans le contenu
        highlightTermsInModal();
    }
    
    // Fonction pour mettre en surbrillance les termes recherchés dans le modal
    function highlightTermsInModal() {
        const content = document.getElementById('full-article-content');
        const text = content.textContent;
        
        // Remplacer le contenu avec le texte mis en surbrillance
        let highlightedText = text;
        
        // Mettre en surbrillance chaque terme actif
        activeTerms.forEach(term => {
            // Créer une expression régulière pour trouver le terme (insensible à la casse)
            const regex = new RegExp(term, 'gi');
            
            // Remplacer toutes les occurrences du terme par une version mise en surbrillance
            highlightedText = highlightedText.replace(regex, match => 
                `<span class="highlight-${term}">${match}</span>`
            );
        });
        
        // Mettre à jour le contenu avec le texte mis en surbrillance
        content.innerHTML = highlightedText;
    }
    
    // Gérer la fermeture du modal
    document.querySelector('.close').addEventListener('click', () => {
        document.getElementById('article-modal').style.display = 'none';
    });
    
    // Fermer le modal si l'utilisateur clique en dehors du contenu
    window.addEventListener('click', (event) => {
        const modal = document.getElementById('article-modal');
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
    
    // Ajouter un gestionnaire d'événements pour les boutons "Voir l'article complet"
    document.addEventListener('click', (event) => {
        if (event.target.classList.contains('view-full-article')) {
            const articleId = event.target.getAttribute('data-id');
            showFullArticle(articleId);
        }
    });
    
    // Charger les articles au démarrage
    await loadArticles();
});
