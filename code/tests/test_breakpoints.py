import unittest
import os
import fileinput
import pathlib


class TestBreakpointStrings(unittest.TestCase):
    def test_no_breakpoints(self):
        root = os.path.dirname(__file__)
        path = pathlib.Path(root)

        found_breakpoints = []
        for root, _, files in os.walk(path.parent.as_posix()):
            for fn in files:
                if fn.endswith(".py") and fn != os.path.basename(__file__):
                    fp = os.path.join(root, fn)
                    with fileinput.input(files=(fp,)) as f:
                        for line in f:
                            if 'breakpoint()' in line:
                                found_breakpoints.append(f.filename())
                                break

        self.assertFalse(
            found_breakpoints,
            f"Found 'breakpoint' in the following files: {found_breakpoints}"
        )


if __name__ == '__main__':
    unittest.main()
