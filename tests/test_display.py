from __future__ import annotations

import unittest

from youarebeingwatched.display import _should_continue


class DisplayKeyboardTest(unittest.TestCase):
    def test_q_exits_display_loop(self) -> None:
        self.assertFalse(_should_continue(ord("q")))
        self.assertFalse(_should_continue(ord("Q")))

    def test_escape_exits_display_loop(self) -> None:
        self.assertFalse(_should_continue(27))

    def test_other_keys_continue_display_loop(self) -> None:
        self.assertTrue(_should_continue(-1 & 0xFF))
        self.assertTrue(_should_continue(ord("x")))


if __name__ == "__main__":
    unittest.main()
