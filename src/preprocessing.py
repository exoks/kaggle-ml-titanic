#  ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⣦⣴⣶⣾⣿⣶⣶⣶⣶⣦⣤⣄⠀⠀⠀⠀⠀⠀⠀
#  ⠀⠀⠀⠀⠀⠀⠀⢠⡶⠻⠛⠟⠋⠉⠀⠈⠤⠴⠶⠶⢾⣿⣿⣿⣷⣦⠄⠀⠀⠀        𓐓  preprocessing.py 𓐔           
#  ⠀⠀⠀⠀⠀⢀⠔⠋⠀⠀⠤⠒⠒⢲⠀⠀⠀⢀⣠⣤⣤⣬⣽⣿⣿⣿⣷⣄⠀⠀
#  ⠀⠀⠀⣀⣎⢤⣶⣾⠅⠀⠀⢀⡤⠏⠀⠀⠀⠠⣄⣈⡙⠻⢿⣿⣿⣿⣿⣿⣦⠀       Dev: oezzaou <OussamaEzzaou@gmail.com>
#  ⢀⠔⠉⠀⠊⠿⠿⣿⠂⠠⠢⣤⠤⣤⣼⣿⣶⣶⣤⣝⣻⣷⣦⣍⡻⣿⣿⣿⣿⡀
#  ⢾⣾⣆⣤⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠉⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇
#  ⠀⠈⢋⢹⠋⠉⠙⢦⠀⠀⠀⠀⠀⠀⢀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇       Created: 2025/07/24 13:45:35 by oezzaou
#  ⠀⠀⠀⠑⠀⠀⠀⠈⡇⠀⠀⠀⠀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇       Updated: 2025/07/30 07:21:29 by oezzaou
#  ⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⢀⣾⣿⣿⠿⠟⠛⠋⠛⢿⣿⣿⠻⣿⣿⣿⣿⡿⠀
#  ⠀⠀⠀⠀⠀⠀⠀⢀⠇⠀⢠⣿⣟⣭⣤⣶⣦⣄⡀⠀⠀⠈⠻⠀⠘⣿⣿⣿⠇⠀
#  ⠀⠀⠀⠀⠀⠱⠤⠊⠀⢀⣿⡿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠘⣿⠏⠀⠀                             𓆩♕𓆪
#  ⠀⠀⠀⠀⠀⡄⠀⠀⠀⠘⢧⡀⠀⠀⠸⣿⣿⣿⠟⠀⠀⠀⠀⠀⠀⠐⠋⠀⠀⠀                     𓄂 oussama ezzaou𓆃
#  ⠀⠀⠀⠀⠀⠘⠄⣀⡀⠸⠓⠀⠀⠀⠠⠟⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀

# ===[ Imports: ]==============================================================
import numpy as np
import pandas as pd
from src.utils.logger import getLogger
from src.utils.nan_counter import nan_counter


# ===[ rename_labels: ]========================================================
def rename_labels(data: pd.DataFrame) -> pd.DataFrame:
    logger = getLogger(__name__)
    logger.info("Renaming columns ...")
    columns = {
        "PassengerId": "Id",
        "Pclass": "Class",
        "SibSp": "Sibling/Spouse",
        "Parch": "Parent/Child",
        "Fare": "Price",
        "Embarked": "Port"
    }
    logger.debug(f"Renaming {columns}")
    return data.rename(columns=columns)


# ===[ remove_duplicates: ]====================================================
def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    '''
    It removes samples (or cases) that are duplicated from `data` if exist
    '''
    logger = getLogger(__name__)
    logger.info("Droping duplicates from raw data ...")
    no_dup_data = data.drop_duplicates()
    if no_dup_data.shape[0] < data.shape[0]:
        logger.debug(f"{data.shape[0] - no_dup_data.shape[0]}: rows dropped")
    else:
        logger.debug("No duplicated rows found")
    return data


# ===[ handle_missing: ]=======================================================
def handle_missing(data: pd.DataFrame) -> pd.DataFrame:
    # Define how to handle missings (Drop missings or Imputation)
    logger = getLogger(__name__)
    logger.info("Handling missings in dataset ...")
    logger.debug("Checking NaN Counts by column/input feature")
    # Count the missing values for each column
    nan_dict = nan_counter(data)
    logger.debug(nan_dict)
    # 1|> [Age]: Imputation by mean
    data['Age'] = data['Age'].fillna(data['Age'].mean()).astype(np.int64)
    logger.debug("Age: fill NaN values by (Imputation By the mean)")
    # 2|> [Price]: Imputation using backward-filling
    data['Price'] = data['Price'].bfill()
    logger.debug("Price: Back filling the price")
    # 3|> Droping columns that does not effect the survive of passengers
    columns = ['Cabin', 'Name', 'Id', 'Port']
    logger.debug(f"Cabin: droping columns: {columns}")
    return data.drop(columns=columns)


# ===[ fix_inconsistent_formats: ]=============================================
def fix_inconsistent_formats(data) -> pd.DataFrame:
    logger = getLogger(__name__)
    logger.info("Fixing inconsistent formats in dataset ...")
    # Fix 'Ticket' column fromats / Caputering the degit part
    logger.debug("Keeping only numerical part of Ticket ...")
    data['Ticket'] = data['Ticket'].str.extract(r'(\d+)$').astype(np.int64)
    # Replacing 1/0 to yes/no
    logger.debug("Replacing 1/0 by yes/no in 'Survived' column ...")
    data['Survived'] = data['Survived'].astype(str).replace({
        "1": "Yes",
        "0": "No",
    })
    return data


# ===[ clean_data: ]===========================================================
def clean_data(data: pd.DataFrame):
    getLogger(__name__).info("-> Cleaning Dataset ...")
    # 1. Rename labels (label managment)
    renamed_data = rename_labels(data)
    # 2. Remove Duplicates
    no_dup_data = remove_duplicates(renamed_data)
    # 3. Handle Missings
    no_missing_data = handle_missing(no_dup_data)
    # 4. Fix Inconsistent formats
    cleaned_data = fix_inconsistent_formats(no_missing_data)
    return cleaned_data

# ===[ feature_scaling: ]======================================================

# ===[ feature_engineering: ]==================================================
