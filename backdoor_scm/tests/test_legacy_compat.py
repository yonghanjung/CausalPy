import sys
import types
import unittest


if "pyvis" not in sys.modules:
    pyvis = types.ModuleType("pyvis")
    pyvis_network = types.ModuleType("pyvis.network")
    pyvis_network.Network = type("Network", (), {})
    pyvis.network = pyvis_network
    sys.modules["pyvis"] = pyvis
    sys.modules["pyvis.network"] = pyvis_network

if "pyperclip" not in sys.modules:
    pyperclip = types.ModuleType("pyperclip")
    pyperclip.copy = lambda value: None
    sys.modules["pyperclip"] = pyperclip

if "plotly" not in sys.modules:
    plotly = types.ModuleType("plotly")
    plotly_graph_objects = types.ModuleType("plotly.graph_objects")
    plotly.graph_objects = plotly_graph_objects
    sys.modules["plotly"] = plotly
    sys.modules["plotly.graph_objects"] = plotly_graph_objects

import SCM
import random_generator


class LegacyCompatibilityTests(unittest.TestCase):
    def test_legacy_public_imports_remain_available(self):
        self.assertTrue(callable(random_generator.random_graph_generator))
        self.assertTrue(callable(random_generator.random_SCM_generator))
        self.assertTrue(hasattr(SCM, "StructuralCausalModel"))


if __name__ == "__main__":
    unittest.main()
