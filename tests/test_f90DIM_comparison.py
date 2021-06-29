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
            calc.run()
        finally:
            if os.path.isfile('__temp__.dimpy'):
                os.remove('__temp__.dimpy')

        return calc

    def test_DDA_nofreq(self):
        dimpy_input = textwrap.dedent(
        """\
        NANOPARTICLE
         Atoms
          Ag  0.000000  0.000000  0.000000
          Ag  2.460139  0.000000 -1.520449
          Ag  2.460139  0.000000  1.520449
          Ag -2.460139  0.000000 -1.520449
          Ag -2.460139  0.000000  1.520449
          Ag -1.520449  2.460139  0.000000
          Ag  1.520449  2.460139  0.000000
          Ag -1.520449 -2.460139  0.000000
          Ag  1.520449 -2.460139  0.000000
          Ag  0.000000 -1.520449  2.460139
          Ag  0.000000  1.520449  2.460139
          Ag  0.000000 -1.520449 -2.460139
          Ag  0.000000  1.520449 -2.460139
         End
         AtomParam Ag exp Ag_jc
        ENDNANOPARTICLE
        VERBOSE 0
        METHOD
         Interaction DDA
         Solver direct
        ENDMETHOD
        """)

        calc = self._run_input(dimpy_input)

        self.assertAlmostEqual(calc.isotropic_polarizabilities[0],
                               590.73, places=2)

    def test_DDA_freq(self):
        dimpy_input = textwrap.dedent(
        """\
        NANOPARTICLE
         Atoms
          Ag  0.000000  0.000000  0.000000
          Ag  2.460139  0.000000 -1.520449
          Ag  2.460139  0.000000  1.520449
          Ag -2.460139  0.000000 -1.520449
          Ag -2.460139  0.000000  1.520449
          Ag -1.520449  2.460139  0.000000
          Ag  1.520449  2.460139  0.000000
          Ag -1.520449 -2.460139  0.000000
          Ag  1.520449 -2.460139  0.000000
          Ag  0.000000 -1.520449  2.460139
          Ag  0.000000  1.520449  2.460139
          Ag  0.000000 -1.520449 -2.460139
          Ag  0.000000  1.520449 -2.460139
         End
         AtomParam Ag exp Ag_jc
        ENDNANOPARTICLE
        VERBOSE 0
        METHOD
         Interaction DDA
         Solver direct
         Frequency nm 350
        ENDMETHOD
        """)

        calc = self._run_input(dimpy_input)

        self.assertAlmostEqual(calc.isotropic_polarizabilities[0].real,
                               -2840.20, places=1)
        self.assertAlmostEqual(calc.isotropic_polarizabilities[0].imag,
                               4069.21, places=1)

    def test_DDA_PBC_nofreq(self):
        dimpy_input = textwrap.dedent(
        """\
        NANOPARTICLE
         Atoms
          Ag 0 0 0
         End
         PBC
          0 0 1
         End
         AtomParam Ag exp Ag_jc
        ENDNANOPARTICLE
        VERBOSE 0
        METHOD
         Interaction DDA
         Solver direct
        ENDMETHOD
        """)

        calc = self._run_input(dimpy_input)

        self.assertAlmostEqual(calc.isotropic_polarizabilities[0],
                               1.32, places=2)

    def test_DDA_PBC_freq(self):
        dimpy_input = textwrap.dedent(
        """\
        NANOPARTICLE
         Atoms
          Ag 0 0 0
         End
         PBC
          0 0 1
         End
         AtomParam Ag exp Ag_jc
        ENDNANOPARTICLE
        VERBOSE 0
        METHOD
         Interaction DDA
         Solver direct
         Frequency nm 319.6
        ENDMETHOD
        """)

        calc = self._run_input(dimpy_input)

        self.assertAlmostEqual(calc.isotropic_polarizabilities[0].real,
                               1.46, places=2)
        self.assertAlmostEqual(calc.isotropic_polarizabilities[0].imag,
                               0.30, places=2)

    def test_DIM_nofreq(self):
        dimpy_input = textwrap.dedent(
        """\
        NANOPARTICLE
         Atoms
          Ag  0.000000  0.000000  0.000000
          Ag  2.460139  0.000000 -1.520449
          Ag  2.460139  0.000000  1.520449
          Ag -2.460139  0.000000 -1.520449
          Ag -2.460139  0.000000  1.520449
          Ag -1.520449  2.460139  0.000000
          Ag  1.520449  2.460139  0.000000
          Ag -1.520449 -2.460139  0.000000
          Ag  1.520449 -2.460139  0.000000
          Ag  0.000000 -1.520449  2.460139
          Ag  0.000000  1.520449  2.460139
          Ag  0.000000 -1.520449 -2.460139
          Ag  0.000000  1.520449 -2.460139
         End
         AtomParam Ag exp Ag_jc
        ENDNANOPARTICLE
        VERBOSE 0
        METHOD
         Interaction DIM
         Solver direct
        ENDMETHOD
        """)

        calc = self._run_input(dimpy_input)

        self.assertAlmostEqual(calc.isotropic_polarizabilities[0],
                               623.01, places=2)

    def test_DIM_freq(self):
        dimpy_input = textwrap.dedent(
        """\
        NANOPARTICLE
         Atoms
          Ag  0.000000  0.000000  0.000000
          Ag  2.460139  0.000000 -1.520449
          Ag  2.460139  0.000000  1.520449
          Ag -2.460139  0.000000 -1.520449
          Ag -2.460139  0.000000  1.520449
          Ag -1.520449  2.460139  0.000000
          Ag  1.520449  2.460139  0.000000
          Ag -1.520449 -2.460139  0.000000
          Ag  1.520449 -2.460139  0.000000
          Ag  0.000000 -1.520449  2.460139
          Ag  0.000000  1.520449  2.460139
          Ag  0.000000 -1.520449 -2.460139
          Ag  0.000000  1.520449 -2.460139
         End
         AtomParam Ag exp Ag_jc
        ENDNANOPARTICLE
        VERBOSE 0
        METHOD
         Interaction DIM
         Solver direct
         Frequency nm 350
        ENDMETHOD
        """)

        calc = self._run_input(dimpy_input)

        self.assertAlmostEqual(calc.isotropic_polarizabilities[0].real,
                               2446.42, places=1)
        self.assertAlmostEqual(calc.isotropic_polarizabilities[0].imag,
                               1601.71, places=1)

#    def test_DIM_PBC_nofreq(self):
#        dimpy_input = textwrap.dedent(
#        """\
#        NANOPARTICLE
#         Atoms
#          Ag 0 0 0
#         End
#         PBC
#          0 0 1.89
#         End
#         AtomParam Ag exp Ag_jc
#        ENDNANOPARTICLE
#        VERBOSE 0
#        METHOD
#         Interaction DIM
#         Solver direct
#        ENDMETHOD
#        """)
#
#        calc = self._run_input(dimpy_input)
#
#        self.assertAlmostEqual(calc.isotropic_polarizabilities[0],
#                               19754.95, places=0)
#
#    def test_DIM_PBC_freq(self):
#        dimpy_input = textwrap.dedent(
#        """\
#        NANOPARTICLE
#         Atoms
#          Ag 0 0 0
#         End
#         PBC
#          0 0 1.89
#         End
#         AtomParam Ag exp Ag_jc
#        ENDNANOPARTICLE
#        VERBOSE 0
#        METHOD
#         Interaction DIM
#         Solver direct
#         Frequency nm 319.6
#        ENDMETHOD
#        """)
#
#        calc = self._run_input(dimpy_input)
#
#        self.assertAlmostEqual(calc.isotropic_polarizabilities[0].real,
#                               3.815, places=2)
#        self.assertAlmostEqual(calc.isotropic_polarizabilities[0].imag,
#                               11.92, places=2)

if __name__ == '__main__':
    unittest.main(warnings="ignore")

