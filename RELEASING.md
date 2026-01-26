# Release Process

Ce document explique comment publier une nouvelle version de basalt-sdk sur PyPI.

## Processus Automatique (Recommandé)

Le projet utilise GitHub Actions pour automatiser la publication sur PyPI lors de la création d'un tag.

### Étapes pour publier une nouvelle version:

1. **Mettez à jour la version**

   ```bash
   # Pour une version patch (1.1.1 -> 1.1.2)
   hatch version patch

   # Pour une version minor (1.1.1 -> 1.2.0)
   hatch version minor

   # Pour une version major (1.1.1 -> 2.0.0)
   hatch version major

   # Ou définissez une version spécifique
   hatch version 1.2.0
   ```

2. **Committez le changement de version**

   ```bash
   git add basalt/_version.py
   git commit -m "chore: bump version to $(hatch version)"
   git push origin master
   ```

3. **Créez et poussez un tag**

   ```bash
   # Le tag doit commencer par 'v' et correspondre à la version
   VERSION=$(hatch version)
   git tag -a "v${VERSION}" -m "Release version ${VERSION}"
   git push origin "v${VERSION}"
   ```

4. **Automatique!** 🎉

   GitHub Actions va automatiquement:
   - ✅ Exécuter les tests sur Python 3.10, 3.11, 3.12, 3.13, 3.14
   - ✅ Vérifier que la version du package correspond au tag
   - ✅ Builder le package (wheel + source dist)
   - ✅ Publier sur PyPI
   - ✅ Créer une release GitHub avec notes

5. **Vérifiez la publication**

   - PyPI: https://pypi.org/project/basalt-sdk/
   - GitHub Releases: https://github.com/basalt-ai/basalt-python/releases

## Processus Manuel (Backup)

Si vous devez publier manuellement:

```bash
# 1. Assurez-vous que les tests passent
hatch run test

# 2. Buildez le package
hatch build

# 3. Publiez sur PyPI
export HATCH_INDEX_USER="__token__"
export HATCH_INDEX_AUTH="pypi-..."  # Votre token PyPI
hatch publish
```

## Prérequis

### Pour le processus automatique:
- Le secret `PYPI_API_TOKEN` doit être configuré dans GitHub Actions
  (Settings → Secrets and variables → Actions → New repository secret)

### Pour le processus manuel:
- Token PyPI avec permissions pour le projet `basalt-sdk`
- Hatch installé: `uv tool install hatch`

## Workflow GitHub Actions

Le workflow `.github/workflows/publish-to-pypi.yml` se déclenche automatiquement sur:
- Création de tags commençant par `v` (ex: `v1.1.1`, `v2.0.0`)
- Exécution manuelle via l'onglet Actions

### Étapes du workflow:
1. **Test** - Exécute la suite de tests sur toutes les versions de Python
2. **Publish** - Si les tests passent:
   - Vérifie que la version du package correspond au tag
   - Build le package
   - Publie sur PyPI
   - Crée une release GitHub

## Checklist avant release

- [ ] Tous les tests passent localement (`hatch run test`)
- [ ] La version a été mise à jour dans `basalt/_version.py`
- [ ] CHANGELOG.md a été mis à jour (si vous en avez un)
- [ ] Les changements sont committés et poussés sur master
- [ ] Le tag correspond exactement à la version du package

## Rollback

Si vous devez annuler une release:

1. **Sur PyPI**: Vous ne pouvez pas supprimer une version, mais vous pouvez publier une nouvelle version corrective
2. **Sur GitHub**: Supprimez la release et le tag si nécessaire

```bash
# Supprimer un tag localement et à distance
git tag -d v1.2.3
git push origin :refs/tags/v1.2.3
```

## Troubleshooting

### "Version déjà publiée sur PyPI"
- Vous ne pouvez pas republier la même version
- Incrémentez la version et recréez un tag

### "Version mismatch"
- Le tag doit correspondre exactement à la version dans `basalt/_version.py`
- Format: tag `v1.2.3` → version `1.2.3` (sans le 'v')

### "Tests échouent en CI mais pas localement"
- Vérifiez que tous les fichiers sont committés
- Vérifiez les différences d'environnement (versions de dépendances)
