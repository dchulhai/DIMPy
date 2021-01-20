#! /usr/bin/python
from sys import exit
try:
    from .dimclass import DIM
except ImportError:
    exit("Cannot import the DIM tools... they may not have been compiled.")
from pylab import *
from scipy.interpolate import InterpolatedUnivariateSpline
from .constants import HART2NM, HART2EV


def plot_dim(args):
    '''Print the absorbance cross-section.'''

    # Set up plotting environment
    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 20})

    # Read data
    data = DIM(args.file)
    data.collect()
    if data.calctype == 'STATIC':
        exit("Cannot plot static calculations.")

    # Define the points to plot
    if args.mode == 'cross':
        y = data.cross_section(unit=args.absunit)
        unitstr = r'$\sigma^\mathrm{{abs}}(\omega)$ ({0}$^2$/nanoparticle)'
        if args.absunit == 'angstroms':
            unitstr = unitstr.format(r'\AA')
        else:
            unitstr = unitstr.format(args.absunit)
    elif args.mode == 'molar':
        y = data.molar_absorptivitty()
        unitstr = r'$\epsilon(\omega)$ M$^{-1}$ cm$^{-1}$'
    elif args.mode == 'absorbance':
        y = data.absorbance(args.path, args.conc)
        unitstr = r'abs (\omega) (unitless)'
    elif args.mode == 'transmittance':
        y = data.transmittance(args.path, args.conc)
        unitstr = r'Transittance (normalized)'
    elif args.mode == 'pol':
        if args.real:
            y = data.isotropic.real
            unitstr = r'$\alpha^R(\omega) (a.u.)$'
        elif args.imag:
            y = data.isotropic.imag
            unitstr = r'$\alpha^I(\omega) (a.u.)$'
        else:
            y = data.isotropic.real
            unitstr = r'$\alpha^R(\omega) (a.u.)$'
            y2 = data.isotropic.imag
            unitstr2 = r'$\alpha^I(\omega) (a.u.)$'

    # Get the domain info and make sure it is in ascending order
    x = data.e_frequencies
    if args.unit == 'nm':
        x = HART2NM(x)
    elif args.unit == 'ev':
        x = HART2EV(x)
    indx = argsort(x)
    x = x[indx]
    y = y[indx]
    domain = linspace(x[0], x[-1], 500)

    # Smooth the points
    fitcoeff = InterpolatedUnivariateSpline(x, y)
    values = fitcoeff(domain)

    # Either plot only the smooth line or both that and the data points
    if args.points:
        smooth, points = plot(domain, values, args.co+args.ls,
                              x, y, args.co+args.pt,
                              lw=args.lw, ms=args.ms, label=unitstr)
    else:
        smooth, = plot(domain, values, args.co+args.ls,
                       lw=args.lw, label=unitstr)

    # Titles, labels, and limits
    if args.title:
        title(args.title)
    if args.unit == 'nm':
        xlabel(r'$\lambda$ (nm)')
    elif args.unit == 'ev':
        xlabel(r'Frequency (eV)')
    elif args.unit == 'au':
        xlabel(r'Frequency (Hartrees)')
    ylabel(unitstr)
    xlim(args.xlimits[0], args.xlimits[1])
    ylim(args.ylimits[0], args.ylimits[1])
    ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

    # If plotting real and imaginary, add the second axis now
    if args.mode == 'pol' and not (args.real or args.imag):
        fitcoeff = InterpolatedUnivariateSpline(x, y2)
        values2 = fitcoeff(domain)
        twinx()
        if args.points:
            smooth2, points2 = plot(domain, values2,
                                    args.color2+args.linestyle2,
                                    x, y2, args.color2+args.pointstyle2,
                                    lw=args.lw, ms=args.ms, label=unitstr2)
        else:
            smooth2, = plot(domain, values2, args.color2+args.linestyle2,
                            lw=args.lw, label=unitstr2)
        ylabel(unitstr2)
        ylim(args.ylimits2[0], args.ylimits2[1])
        legend([smooth, smooth2],
               [smooth.get_label(), smooth2.get_label()],
               fancybox=True, shadow=True, prop={'size': 16},
               loc='best')

    # Now Plot
    show()

if __name__ == '__main__':
    main()
