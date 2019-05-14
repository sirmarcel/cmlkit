from .krr import KRR, ExtensiveKRR
from .qmml import classes as qmml_classes

classes = {KRR.kind: KRR, ExtensiveKRR.kind: ExtensiveKRR, **qmml_classes}
