# Tutorial

## Trénink modelu
1. vytvoření složky v `code/src/<nova-slozka>`, ve které budou soubory `config.json`, `errors_config.json` a `f_score_dev.json`
2. přesuň se do nově vytvořeného adresáře a spusť trénink modelu pomocí:
   ```bash
   python3 ../pipeline/run.py --config config.json
   ```
   Do této složky se budou nahrávat checkpointy, backupy a vyhodnocení
3. Jednoduchá evaluace modelu se spouští pomocí:
   ```bash
   python3 ../pipeline/run.py --config config.json --eval
   ```