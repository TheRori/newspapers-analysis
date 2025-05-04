# Script PowerShell pour lancer l'application de manière sécurisée
# Avec gestion d'erreurs et journalisation

# Activer l'environnement virtuel
Write-Host "Activation de l'environnement virtuel..." -ForegroundColor Cyan
try {
    & .\.venv\Scripts\activate.ps1
    Write-Host "Environnement virtuel activé avec succès." -ForegroundColor Green
} catch {
    Write-Host "Erreur lors de l'activation de l'environnement virtuel:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

# Vérifier que les modules nécessaires sont installés
Write-Host "Vérification des modules nécessaires..." -ForegroundColor Cyan
$requiredModules = @("dash", "plotly", "pandas", "numpy")
$missingModules = @()

foreach ($module in $requiredModules) {
    try {
        $moduleCheck = python -c "import $module; print('Module $module disponible, version:', $module.__version__)"
        Write-Host $moduleCheck -ForegroundColor Green
    } catch {
        Write-Host "Module $module manquant!" -ForegroundColor Red
        $missingModules += $module
    }
}

if ($missingModules.Count -gt 0) {
    Write-Host "Modules manquants détectés. Installation en cours..." -ForegroundColor Yellow
    foreach ($module in $missingModules) {
        Write-Host "Installation de $module..." -ForegroundColor Yellow
        pip install $module
    }
}

# Créer un fichier de log
$logFile = "app_log_$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss').txt"
Write-Host "Les logs seront écrits dans $logFile" -ForegroundColor Cyan

# Lancer l'application avec redirection des erreurs
Write-Host "Lancement de l'application..." -ForegroundColor Cyan
try {
    # Utiliser Start-Process pour éviter les problèmes d'affichage dans PowerShell
    $env:PYTHONUNBUFFERED = "1"  # Désactiver la mise en tampon pour voir les logs en temps réel
    Write-Host "Application démarrée. Accédez à http://127.0.0.1:8050/ dans votre navigateur." -ForegroundColor Green
    Write-Host "Appuyez sur Ctrl+C pour arrêter l'application." -ForegroundColor Yellow
    python src\webapp\run_app.py | Out-File -FilePath $logFile -Append
} catch {
    Write-Host "Erreur lors du lancement de l'application:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "Consultez le fichier $logFile pour plus de détails." -ForegroundColor Yellow
}
