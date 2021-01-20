from sys import exit
from os import uname
from io import StringIO
from ._version import __version__

__all__ = ['input_formatter']


def input_formatter(namespace, outname, starttime):
    """Takes a general input file that has been read in by python and
    formats it so that it is easy to read in FORTRAN.
    """
    # Make a string that looks like a file
    formatted = StringIO()

    # On the first line, give the name of the temp file to write to
    print(outname, file=formatted)

    # Some important booleans
    print(p2f(namespace.cpim),           file=formatted)
    print(p2f(namespace.dda),            file=formatted)
    print(p2f(namespace.noninteracting), file=formatted)
    print(p2f(namespace.nopol),          file=formatted)
    print(p2f(namespace.nochar),         file=formatted)
    print(p2f(namespace.nocross),        file=formatted)
    print(p2f(namespace.coorddepend),    file=formatted)
    # media refractive number  Xing Chen
    print(p2f(namespace.nsolv),          file=formatted)

    # Some important numbers
    print(p2f(namespace.algorithm),     file=formatted)
    print(p2f(namespace.tol),           file=formatted)
    print(p2f(namespace.totalcharge),   file=formatted)
    print(p2f(namespace.natoms),        file=formatted)
    print(p2f(namespace.nfreq),         file=formatted)
    print(p2f(len(namespace.elements)), file=formatted)
    print(p2f(namespace.volume),        file=formatted)
    print(p2f(namespace.atomspercell),  file=formatted)
    print(p2f(namespace.grid),          file=formatted)
    print(p2f(namespace.multiplier[0]), file=formatted)
    print(p2f(namespace.multiplier[1]), file=formatted)
    print(p2f(namespace.multiplier[2]), file=formatted)
    print(p2f(namespace.niter),         file=formatted)
    print(p2f(namespace.precon),        file=formatted)
    print(p2f(namespace.scscale),       file=formatted)
    print(p2f(namespace.dirs[0]),       file=formatted)
    print(p2f(namespace.dirs[1]),       file=formatted)
    print(p2f(namespace.dirs[2]),       file=formatted)


    # On the next three lines, write out some runtime conditions
    print(p2f(starttime.strftime('%X')),            file=formatted)
    print(p2f(starttime.strftime('%A, %b %d, %Y')), file=formatted)
    print(p2f(uname()[1]),                          file=formatted)
    print(p2f(__version__),                         file=formatted)

    # Write the title
    try:
        print(namespace.title, file=formatted)
    except AttributeError:
        print('', file=formatted)

    # Now print what to print
    print(p2f('pol'    in namespace.print_rules), file=formatted)
    print(p2f('atmdip' in namespace.print_rules), file=formatted)
    print(p2f('energy' in namespace.print_rules), file=formatted)
    print(p2f('eff'    in namespace.print_rules), file=formatted)
    print(p2f('coords' in namespace.print_rules), file=formatted)

# find out the metal elements, which is coordination dependent in the nanoparticle
    imetal = []
    elemdepend=[]
    n=0
    for elem in namespace.elements:
        pars = getattr(namespace, elem.lower())
        if pars.coorddepend == True:
            elemdepend.append(elem)
    for elem in namespace.elements:
        n = n + 1
        if elem in elemdepend:
            imetal.append(n)
#    print(p2f(len(elemdepend)),file=formatted)

# find out the coordination dependent elements 
    depend = 0
    independ = 0
    lcoorddepend = []
    for elem in namespace.elements:
        pars = getattr(namespace, elem.lower())
        lcoorddepend.append(pars.coorddepend)
        if pars.coorddepend==True:
            for atomname in namespace.atoms:
                if atomname == elem:
                    depend=depend+1
        else:
            independ=independ+1
# find out the coordination dependent atoms
    atomcoorddepend = []
    for atomname in namespace.atoms:
        n = -1
        for elem in namespace.elements:
            n = n + 1
            if atomname == elem:
                atomcoorddepend.append(lcoorddepend[n])
#    for i in range(namespace.natoms):
#        print(p2f(atomcoorddepend[i]),file=formatted)
#        print("atomcoorddepend {0}".format(atomcoorddepend[i]))

#    for i in range(len(elemdepend)):
#        print(p2f(imetal[i]),file=formatted)
#    print("imetal {0}".format(imetal))

#---------------------------    

    # We will now give each frequency to calculate
    # First, we list the number of frequencies to collect
    # Then, each frequency is printed on its own line in hartrees
    # No frequencies for static
    for freq in namespace.freqs:
        print(p2f(freq), file=formatted)

    # Next, print off the coordinates.
    for atom, coord in zip(namespace.atoms, namespace.coords):
        print(' '.join(['{0:<2}'.format(atom)]+[p2f(x) for x in coord]),
              file=formatted)

    # Last, print the type information
    #

#count the number of elements and the number of atoms of each element in the system
    elemtype = []
    for atomname in namespace.atoms:
        itype = 0
        for elem in namespace.elements:
            itype = itype + 1
            if elem==atomname:
                elemtype.append(itype)

    nelemtype=[]
    ielemtype=[]
    for i in range(len(namespace.elements)):
        count = 0
        itemp=[]
        for j in range(namespace.natoms):
            if elemtype[j] == i + 1:
                count = count + 1
                itemp.append(j+1)
        nelemtype.append(count)
        ielemtype.append(itemp)
#    for i in range(len(namespace.elements)):
#        print(p2f(nelemtype[i]),file=formatted)
#        for j in range(nelemtype[i]):
#            print(p2f(ielemtype[i][j]),file=formatted)


    numatom = []
    import numpy as np
    index=np.zeros((namespace.natoms))
    for elem in namespace.elements:
        count = 0
        iatom = -1
        for  atomname in namespace.atoms:
            iatom = iatom + 1
            if atomname == elem:
                count = count + 1
                index[iatom]=count
        numatom.append(count)
    #print("elemtype: {0}".format(elemtype))
    #print("numatom: {0}".format(numatom))
    #print("namespace.natoms {0}".format(namespace.natoms))

    tot_drude=[]
    for elem in namespace.elements:
        pars = getattr(namespace, elem.lower())
        tot_drude.append(pars.drude)

    n=-1
    tot_rad=[]
    tot_static_pol=[]
    #print("-----------")
    for atomname in namespace.atoms:
        n = n + 1
        for elem in namespace.elements:
            pars = getattr(namespace, elem.lower())
            if atomname==elem:
                if pars.coorddepend==True:
                    tot_rad.append(pars.cd_rad[int(index[n])-1])
                    tot_static_pol.append(pars.cd_static_pol[int(index[n])-1])
                    tot_drude.append(pars.cd_drude[int(index[n])-1])
                else:
                    tot_rad.append(pars.rad)
                    tot_static_pol.append(pars.static_pol)
                    tot_drude.append(pars.drude)
        #print("atom {0}, rad {1}".format(atomname, tot_rad[n]))
        #print("atom {0}, pol {1}".format(atomname, tot_static_pol[n]))
    #n=-1
    #for elem in namespace.elements:
        #n = n + 1
        #print(elem)
        #print("atom {0}, drude {1}".format(elem, tot_drude[n]))
    #for atomname in namespace.atoms:
        #n = n + 1
        #print("atom {0}, drude {1}".format(atomname, tot_drude[n]))
    #print("-----------")

    for i in range(namespace.natoms):
        print(p2f(tot_rad[i]),p2f(tot_static_pol[i]),file=formatted)
    for i in range(namespace.natoms+len(namespace.elements)):
        print(p2f(tot_drude[i]),file=formatted)

    for elem in namespace.elements:
        # If so, print the atom type
        pars = getattr(namespace, elem.lower())

        print(p2f(elem), file=formatted)

        # Polariazability parameter type
        # We've hijacked this to enable static calcualtions of certain atoms within
        # a FD calculation.
        if pars.static:
            print(p2f(0), file=formatted)
        else:
            print(p2f(1), file=formatted)

        # Print the cpim parameters
        print(p2f([pars.om1, pars.om2, pars.gm1, pars.gm2, pars.size]),
              file=formatted)
        # Print the bound dielectric and fermi velocity.
        print(p2f(pars.bound), p2f(pars.fermi), file=formatted)
        # Print the drude term
        # Print off each lorentzian
        for l in pars.lrtz:
            print(p2f(l), file=formatted)
        # Print the experimental dielectric for each frequency
        for om in pars.exp:
            print(p2f(om), file=formatted)

    return formatted

def p2f(val):
    """Convert from python to FORTRAN."""
    if isinstance(val, bool):
        return 'T' if val else 'F'
    elif isinstance(val, complex):
        return ' '.join([p2f(val.real), p2f(val.imag)])
    elif isinstance(val, float):
        return '{0:>21.14E}'.format(val)
    elif isinstance(val, list) or isinstance(val, tuple):
        return ' '.join([p2f(x) for x in val])
    else:
        return str(val)
