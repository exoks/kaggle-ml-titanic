#  ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⣦⣴⣶⣾⣿⣶⣶⣶⣶⣦⣤⣄⠀⠀⠀⠀⠀⠀⠀
#  ⠀⠀⠀⠀⠀⠀⠀⢠⡶⠻⠛⠟⠋⠉⠀⠈⠤⠴⠶⠶⢾⣿⣿⣿⣷⣦⠄⠀⠀⠀               𓐓  logger.py 𓐔           
#  ⠀⠀⠀⠀⠀⢀⠔⠋⠀⠀⠤⠒⠒⢲⠀⠀⠀⢀⣠⣤⣤⣬⣽⣿⣿⣿⣷⣄⠀⠀
#  ⠀⠀⠀⣀⣎⢤⣶⣾⠅⠀⠀⢀⡤⠏⠀⠀⠀⠠⣄⣈⡙⠻⢿⣿⣿⣿⣿⣿⣦⠀       Dev: oezzaou <OussamaEzzaou@gmail.com>
#  ⢀⠔⠉⠀⠊⠿⠿⣿⠂⠠⠢⣤⠤⣤⣼⣿⣶⣶⣤⣝⣻⣷⣦⣍⡻⣿⣿⣿⣿⡀
#  ⢾⣾⣆⣤⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠉⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇
#  ⠀⠈⢋⢹⠋⠉⠙⢦⠀⠀⠀⠀⠀⠀⢀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇       Created: 2025/07/24 22:12:05 by oezzaou
#  ⠀⠀⠀⠑⠀⠀⠀⠈⡇⠀⠀⠀⠀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇       Updated: 2025/07/26 13:46:56 by oezzaou
#  ⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⢀⣾⣿⣿⠿⠟⠛⠋⠛⢿⣿⣿⠻⣿⣿⣿⣿⡿⠀
#  ⠀⠀⠀⠀⠀⠀⠀⢀⠇⠀⢠⣿⣟⣭⣤⣶⣦⣄⡀⠀⠀⠈⠻⠀⠘⣿⣿⣿⠇⠀
#  ⠀⠀⠀⠀⠀⠱⠤⠊⠀⢀⣿⡿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠘⣿⠏⠀⠀                             𓆩♕𓆪
#  ⠀⠀⠀⠀⠀⡄⠀⠀⠀⠘⢧⡀⠀⠀⠸⣿⣿⣿⠟⠀⠀⠀⠀⠀⠀⠐⠋⠀⠀⠀                     𓄂 oussama ezzaou𓆃
#  ⠀⠀⠀⠀⠀⠘⠄⣀⡀⠸⠓⠀⠀⠀⠠⠟⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀

# ===[ Imports: ]==============================================================
import logging
import coloredlogs


# ===[ getLogger: ]============================================================
def getLogger(name: str) -> logging.Logger:
    '''
        It Creats a custom logger based with `name` and return it.
        logger contains 2 handlers:
            1. streamHandler: to show logs in the stdout
                > logs are colored using the external ColoredFormatter module
                  from ColoredLogs package
            2. fileHandler:
                > logs are written in the fil logs/__name__.log,
                  (logs are not colored)
    '''
    # Creating a Logger Instance
    logger = logging.getLogger(name)
    # Set logging level to "DEBUG"
    logger.setLevel("DEBUG")
    # Creating A file Handler for logger
    file_handler = logging.FileHandler(f"logs/{name}.log", mode='w')
    # Creating a console handler
    console_handler = logging.StreamHandler()
    # Creating a ColoredFormatter for handler (read =>  INFO: section below)
    colored_formatter = coloredlogs.ColoredFormatter(
        fmt="[%(levelname)-5s][%(asctime)s] %(message)s")
    # Creating a standard formatter for file_handler
    formatter = logging.Formatter(
        fmt="[%(levelname)-5s][%(asctime)s] %(message)s")
    # setting a fromatter to both handlers (console_handler & file_handler)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(colored_formatter)
    # Adding Both handler to 'logger'
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


# NOTE:[ 'Coloredlogs' Module Usage in My logger ]-----------------------------
# - As you can guys see i used ColoredFromatter module instead of the standard
#   logging.Fromatter module with different standard 'logging' handlers to
#   get colored logs.
# INFO:[ Coloredlogs Module ]--------------------------------------------------
# - `Coloredlogs`: is third-party python package that makes your log messages
#   automatically colorized in the terminal based on their severity level
#   (DEBUG, INFO, WARNING, etc)
# - It is buils on the top of the standard 'logging' module, so It's 100%
#   compatibel with existing 'logging' steups
