# Streamlit dashboard deployment

This folder ('/dashboard/') contains the streamlit dashboard 

Streamlit is configured through configuration files in a hidden root folder `.streamlit` (gitignore) :
- `config.toml` : contains theming, debugging etc
- `secrets.toml` : contains secret keys, flask api url etc.

## To deploy locally
- Run command `streamlit run dashboard/main.py`

## test en local avec docker

docker build -t ihm-dashboard .
docker run -p 8501:8501 ihm-dashboard:latest
## github actions  fait le ci/cd en google cloud
 1-creer un cluster kubernetes autopilot-cluster-1  
 2- creer Artifact registery  home-credit-repo
 3- creer cloud storage Buckets  : data-model-home-credit 
 4- faire un push et github action build.xml creer le livrable docker ,push dans le registry puis instance avec ressources.yaml dans kubernetes .
 5-ajouter le token google .json et le id de projet  dans secert action dans github.

