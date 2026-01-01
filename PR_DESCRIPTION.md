# Modernize PyPI packaging infrastructure

## Summary

This PR modernizes the PyPI packaging infrastructure to align with current Python packaging best practices (2025 standards). The setup is now configured to **publish automatically when you create a GitHub release**.

### Key Changes

- ✅ Add `pyproject.toml` with modern build configuration (PEP 517/518/621)
- ✅ Switch from versioneer to `setuptools-scm` for automatic version management
- ✅ Update GitHub Actions workflow to trigger on **GitHub releases** (not all pushes)
- ✅ Upgrade all GitHub Actions to latest versions (checkout@v4, setup-python@v5)
- ✅ Enable Trusted Publishing (OIDC) for secure authentication
- ✅ Use `python -m build` instead of deprecated `setup.py sdist bdist_wheel`
- ✅ Bump minimum Python version from 3.7 → 3.8 (Python 3.7 EOL: June 2023)
- ✅ Remove unused versioneer files (versioneer.py, setup.cfg)

### Breaking Changes

- **Minimum Python version is now 3.8+**
- Version management moved from versioneer to setuptools-scm
- Publishing now requires creating a GitHub release (not just pushing tags)

---

## 📋 Setup Instructions

### Step 1: Enable Trusted Publishing on PyPI

Trusted Publishing (OIDC) is more secure than API tokens and doesn't require managing secrets.

#### For Production PyPI:

1. Go to https://pypi.org/manage/project/pytorch-accelerated/settings/publishing/
2. Click **"Add a new publisher"**
3. Fill in the following details:
   - **PyPI Project Name:** `pytorch-accelerated`
   - **Owner:** `Chris-hughes10`
   - **Repository name:** `pytorch-accelerated`
   - **Workflow name:** `release.yaml`
   - **Environment name:** (leave blank)
4. Click **"Add"**

#### For Test PyPI (Optional):

1. Go to https://test.pypi.org/manage/project/pytorch-accelerated/settings/publishing/
2. Click **"Add a new publisher"**
3. Fill in the same details as above:
   - **PyPI Project Name:** `pytorch-accelerated`
   - **Owner:** `Chris-hughes10`
   - **Repository name:** `pytorch-accelerated`
   - **Workflow name:** `release.yaml`
   - **Environment name:** (leave blank)
4. Click **"Add"**

> **Note:** If your project doesn't exist on Test PyPI yet, you may need to do an initial manual upload first, or the workflow will skip it (it's configured with `continue-on-error: true`).

---

### Step 2: Publishing a New Release

Once Trusted Publishing is configured, releasing is simple:

1. **Create a GitHub release:**
   - Go to https://github.com/Chris-hughes10/pytorch-accelerated/releases/new
   - Create a new tag (e.g., `v0.5.0` or `0.5.0`)
   - Add release notes
   - Click **"Publish release"**

2. **The workflow automatically:**
   - Builds the distribution packages
   - Publishes to Test PyPI (optional, for verification)
   - Publishes to production PyPI

3. **Monitor the workflow:**
   - Check the Actions tab: https://github.com/Chris-hughes10/pytorch-accelerated/actions
   - Verify successful publication

---

## 🔄 Alternative: Continue Using API Tokens (Not Recommended)

If you prefer to keep using API tokens temporarily instead of Trusted Publishing:

1. Keep your existing `PYPI_API_TOKEN` and `TEST_PYPI_API_TOKEN` secrets in GitHub
2. The workflow will need a small modification to use tokens instead of OIDC

However, **Trusted Publishing is the recommended approach** as it's more secure and doesn't require managing secrets.

---

## 🧪 Testing

Before merging, you can test the build locally:

```bash
pip install build
python -m build
```

This will create distribution files in `dist/` directory.

---

## 📚 References

- [PEP 621 - Storing project metadata in pyproject.toml](https://peps.python.org/pep-0621/)
- [PyPI Trusted Publishing Guide](https://docs.pypi.org/trusted-publishers/)
- [setuptools-scm Documentation](https://setuptools-scm.readthedocs.io/)
