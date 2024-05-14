# Scrape HTML

``` sh
Rscript -e "install.packages(c('tidyverse', 'rvest', 'remotes'))"
Rscript scrape.R
```

# Convert HTML to Rmarkdown

``` sh
pandoc --from html-native_divs-native_spans scrape.html --to gfm -o notebook.md --no-highlight
```

# Convert Rmarkdown to Jupyter Notebook

``` sh
Rscript -e "install.packages('remotes')"
Rscript -e "remotes::install_github('mkearney/rmd2jupyter')"
Rscript -e "rmd2jupyter::rmd2jupyter('notebook.Rmd')"
```

# Convert Jupyter Notebook to Python script

``` sh
pip install jupytext
jupytext --to py notebook.ipynb   
```

# Remove comments

``` sh
sed -i -e '/# -/,/# +/d' notebook.py
```

# Style

``` sh
docker run -e VALIDATE_PYTHON=true -e VALIDATE_MARKDOWN=false -e RUN_LOCAL=true -v "$(pwd)":/tmp/lint github/super-linter
```

# SQL Enviroment

``` sh
set -euo pipefail
mkdir -p "$HOME"/.local/docker/postgresql
docker run --rm --name pg-docker -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=local -d -p 5432:5432 -e PGDATA=/var/lib/postgresql/data/pgdata -v "$HOME"/.local/docker/postgresql/data:/var/lib/postgresql/data postgres
docker exec -it pg-docker /bin/bash
docker cp books.csv pg-docker:/tmp/books.csv
psql -d local postgres

```

