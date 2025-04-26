document.addEventListener("DOMContentLoaded", function() {
    document.body.addEventListener("click", function(e) {
        if (e.target && e.target.matches("a[data-article-id]")) {
            const articleId = e.target.getAttribute("data-article-id");
            const event = new CustomEvent("article-selected", {
                detail: { articleId: articleId }
            });
            document.dispatchEvent(event);
        }
    });

    document.addEventListener('article-selected', function(e) {
        const articleId = e.detail.articleId;
        const store = document.getElementById('selected-article-id-store');
        if (store && store._dashprivate_) {
            store._dashprivate_.setProps({
                'data': articleId
            });
        }
    });
});
