import unittest


class MyTestCase(unittest.TestCase):
    def test_members(self):
        with open("rl_exercises/members.txt") as fh:
            lines = fh.readlines()

        self.assertTrue(lines[0].startswith("member 1: "))
        self.assertTrue(lines[1].startswith("member 2: "))
        self.assertTrue(lines[2].startswith("member 3: "))

    def test_names_replaced(self):
        with open("rl_exercises/members.txt") as fh:
            lines = fh.readlines()

        self.assertFalse("name1" in lines[0])
        self.assertFalse("name2" in lines[1])
        self.assertFalse("name3" in lines[2])


if __name__ == "__main__":
    unittest.main()
