import os
import textwrap
import unittest

import dimpy

class TestMethods(unittest.TestCase):
    pass

def test_solvers(solver):

    def test(self):

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

        dimpy_input = dimpy_input.replace('Solver direct',
                                          'Solver '+solver)

        with open('__temp__.dimpy', 'w') as input_file:
            print(dimpy_input, file=input_file)

        try:
            calc = dimpy.run('__temp__.dimpy', run_calc=False)
            calc.run()
        finally:
            if os.path.isfile('__temp__.dimpy'):
                os.remove('__temp__.dimpy')

        self.assertAlmostEqual(calc.isotropic_polarizabilities[0].real,
                           -2840.20, places=0)
        self.assertAlmostEqual(calc.isotropic_polarizabilities[0].imag,
                           4069.21, places=0)

    return test

for solver in dimpy.ReadInput.solvers:
    test = test_solvers(solver)
    test.__name__ = 'test_'+solver.upper()
    setattr(TestMethods, test.__name__, test)

if __name__ == '__main__':
    unittest.main(warnings="ignore")

