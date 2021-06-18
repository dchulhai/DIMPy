import os
import textwrap
import unittest

import dimpy

class TestMethods(unittest.TestCase):

    def _run_input(self, string):

        with open('__temp__.dimpy', 'w') as input_file:
            print(string, file=input_file)

        try:
            calc = dimpy.run('__temp__.dimpy', run_calc=False)
        finally:
            if os.path.isfile('__temp__.dimpy'):
                os.remove('__temp__.dimpy')

        return calc

    def test_DDAs(self):

        dimpy_input = textwrap.dedent(
        """\
        NANOPARTICLE
         Atoms
          Ag 0 0 0 
         End 
        ENDNANOPARTICLE
        VERBOSE 0
        METHOD
         Interaction DDA 
        ENDMETHOD
        """)

        calc = self._run_input(dimpy_input)

        self.assertIsInstance(calc, dimpy.DDAs)

    def test_DDAr(self):

        dimpy_input = textwrap.dedent(
        """\
        NANOPARTICLE
         Atoms
          Ag 0 0 0 
         End 
        ENDNANOPARTICLE
        VERBOSE 0
        METHOD
         Interaction DDA
         Kdir x
        ENDMETHOD
        """)

        calc = self._run_input(dimpy_input)

        self.assertIsInstance(calc, dimpy.DDAr)

    def test_DDAsPBC(self):

        dimpy_input = textwrap.dedent(
        """\
        NANOPARTICLE
         Atoms
          Ag 0 0 0 
         End 
         PBC
          0 0 1.89
         END
        ENDNANOPARTICLE
        VERBOSE 0
        METHOD
         Interaction DDA
        ENDMETHOD
        """)

        calc = self._run_input(dimpy_input)

        self.assertIsInstance(calc, dimpy.DDAsPBC)

    def test_DDArPBC(self):

        dimpy_input = textwrap.dedent(
        """\
        NANOPARTICLE
         Atoms
          Ag 0 0 0 
         End 
         PBC
          0 0 1.89
         END
        ENDNANOPARTICLE
        VERBOSE 0
        METHOD
         Interaction DDA
         Kdir x
        ENDMETHOD
        """)

        calc = self._run_input(dimpy_input)

        self.assertIsInstance(calc, dimpy.DDArPBC)

#    def test_DIMs(self):
#
#        dimpy_input = textwrap.dedent(
#        """\
#        NANOPARTICLE
#         Atoms
#          Ag 0 0 0 
#         End 
#        ENDNANOPARTICLE
#        VERBOSE 0
#        METHOD
#         Interaction DIM
#        ENDMETHOD
#        """)
#
#        calc = self._run_input(dimpy_input)
#
#        self.assertIsInstance(calc, dimpy.DIMs)
#
#    def test_DIMr(self):
#
#        dimpy_input = textwrap.dedent(
#        """\
#        NANOPARTICLE
#         Atoms
#          Ag 0 0 0 
#         End 
#        ENDNANOPARTICLE
#        VERBOSE 0
#        METHOD
#         Interaction DIM
#         Kdir x
#        ENDMETHOD
#        """)
#
#        calc = self._run_input(dimpy_input)
#
#        self.assertIsInstance(calc, dimpy.DIMr)
#
#    def test_DIMsPBC(self):
#
#        dimpy_input = textwrap.dedent(
#        """\
#        NANOPARTICLE
#         Atoms
#          Ag 0 0 0 
#         End
#         PBC
#          0 0 1.89
#         End
#        ENDNANOPARTICLE
#        VERBOSE 0
#        METHOD
#         Interaction DIM
#        ENDMETHOD
#        """)
#
#        calc = self._run_input(dimpy_input)
#
#        self.assertIsInstance(calc, dimpy.DIMsPBC)
#
#    def test_DIMrPBC(self):
#
#        dimpy_input = textwrap.dedent(
#        """\
#        NANOPARTICLE
#         Atoms
#          Ag 0 0 0 
#         End
#         PBC
#          0 0 1.89
#         End
#        ENDNANOPARTICLE
#        VERBOSE 0
#        METHOD
#         Interaction DIM
#         kdir x
#        ENDMETHOD
#        """)
#
#        calc = self._run_input(dimpy_input)
#
#        self.assertIsInstance(calc, dimpy.DIMrPBC)

if __name__ == '__main__':
    unittest.main()

