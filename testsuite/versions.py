import unittest

import pkg_resources


class PackageVersionTest(unittest.TestCase):
    def test_versions(self):
        required_versions = {
            "numpy": "1.22.3",
            "pint": "0.19.2",
            "psutil": "5.9.0",
            "pyinstaller": "5.0.1",
            "pyqt5": "5.15.6",
            "pyqtgraph": "0.12.4",
            "pyzmq": "22.3.0",
            "scipy": "1.8.0",
            "sip": "6.6.1",
            "six": "1.16.0",
            "wheel": "0.37.1",
        }

        installed_versions = {}
        for p in required_versions.keys():
            try:
                installed_versions[p] = pkg_resources.get_distribution(p).version
            except pkg_resources.DistributionNotFound:
                installed_versions[p] = "Not installed!"

        # for p, v in required_versions.items():
        #     if not installed_versions[p] >= v:
        #         raise Exception(f"Version of {p} is {installed_versions[p]}, but should be at least {v}.")

        msg = "    ".join([f"{p}: {v}" for p, v in installed_versions.items()])
        raise Exception(msg)


if __name__ == "__main__":
    unittest.main()
