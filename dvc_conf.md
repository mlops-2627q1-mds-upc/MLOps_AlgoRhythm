# Requirements
* Create a DagsHub account.
* Have GitHub repository cloned.
* Have uv installed.

Notes:
* This was done using Linux/WSL.
* Once DagsHub account is created, must notify group to be added to the DagsHub repository.

# Instructions
1. Once DagsHub account is created, must notify group to be added to the DagsHub repository.

2. Inside cloned repository folder, execute:
```
uv sync
```

3. Activate environment
```
source .venv/bin/activate
```

4. In DagsHub interface, inside repository, go to 'Data' and copy setup credentials in terminal.

Example of what must be copied:
```
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <user_name>
dvc remote modify origin --local password <password>
```