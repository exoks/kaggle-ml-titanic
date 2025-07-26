#  ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⣦⣴⣶⣾⣿⣶⣶⣶⣶⣦⣤⣄⠀⠀⠀⠀⠀⠀⠀
#  ⠀⠀⠀⠀⠀⠀⠀⢠⡶⠻⠛⠟⠋⠉⠀⠈⠤⠴⠶⠶⢾⣿⣿⣿⣷⣦⠄⠀⠀⠀        𓐓  preprocessing.py 𓐔           
#  ⠀⠀⠀⠀⠀⢀⠔⠋⠀⠀⠤⠒⠒⢲⠀⠀⠀⢀⣠⣤⣤⣬⣽⣿⣿⣿⣷⣄⠀⠀
#  ⠀⠀⠀⣀⣎⢤⣶⣾⠅⠀⠀⢀⡤⠏⠀⠀⠀⠠⣄⣈⡙⠻⢿⣿⣿⣿⣿⣿⣦⠀       Dev: oezzaou <OussamaEzzaou@gmail.com>
#  ⢀⠔⠉⠀⠊⠿⠿⣿⠂⠠⠢⣤⠤⣤⣼⣿⣶⣶⣤⣝⣻⣷⣦⣍⡻⣿⣿⣿⣿⡀
#  ⢾⣾⣆⣤⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠉⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇
#  ⠀⠈⢋⢹⠋⠉⠙⢦⠀⠀⠀⠀⠀⠀⢀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇       Created: 2025/07/24 13:45:35 by oezzaou
#  ⠀⠀⠀⠑⠀⠀⠀⠈⡇⠀⠀⠀⠀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇       Updated: 2025/07/26 21:47:07 by oezzaou
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
        "SibSp": "Sibling/spouse",
        "Parch": "Parent/Child",
        "Fare": "Price",
        "Embarked": "Port"
    }
    logger.debug(f"Renaming {columns}")
    renamed_data = data.rename(columns=columns)
    logger.debug("Setting Index to 'Id'")
    return renamed_data.set_index('Id')


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
    # 2|> [Price]: Imputation using backward-fill
    data['Price'] = data[data['Class'] == 3].Price.bfill()
    logger.debug("Price: Back filling the price")
    # 3|> [Cabin]: 327 is nan from 418 cases, It is good to drop the column
    logger.debug("Cabin: droping 'Cabin' column")
    return data.drop(columns='Cabin')


# ===[ fix_inconsistent_formats: ]=============================================
def fix_inconsistent_formats(data) -> pd.DataFrame:
    logger = getLogger(__name__)
    logger.info("Fixing inconsistent formats in dataset ...")
    return data


# ===[ clean_data: ]===========================================================
def clean_data(data: pd.DataFrame):
    # 1. Rename labels (label managment)
    renamed_data = rename_labels(data)
    # 2. Remove Duplicates
    no_dup_data = remove_duplicates(renamed_data)
    # 3. Handle Missings
    no_missing_data = handle_missing(no_dup_data)
    # 4. Fix Inconsistent formats
    cleaned_data = fix_inconsistent_formats(no_missing_data)
    return cleaned_data
