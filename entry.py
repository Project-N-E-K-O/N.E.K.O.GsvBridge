import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(current_dir)

from api_neko.server import main


main()
