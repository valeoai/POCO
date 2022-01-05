from .shapenet import ShapeNet

from .synthetic_room import ShapeNetSyntheticRooms as SyntheticRooms

from .scenenet import SceneNet
from .scenenet import SceneNet as SceneNet20
from .scenenet import SceneNet as SceneNet100
from .scenenet import SceneNet as SceneNet500
from .scenenet import SceneNet as SceneNet1000

from .abc import ABCTrain as ABC
from .abc_test import ABCTest, ABCTestNoiseFree, ABCTestExtraNoise
from .real_world import RealWorld
from .famous_test import FamousTest, FamousTestNoiseFree, FamousTestExtraNoisy, FamousTestSparse, FamousTestDense
from .thingi10k_test import Thingi10kTest, Thingi10kTestNoiseFree, Thingi10kTestExtraNoisy, Thingi10kTestSparse, Thingi10kTestDense
