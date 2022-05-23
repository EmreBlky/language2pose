import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data import KITMocap

KITMocap('/scr-ssd/ying1123/dataset/kit-mocap', preProcess_flag=True)
